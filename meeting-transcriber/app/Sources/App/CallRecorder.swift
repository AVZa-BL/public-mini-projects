import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia
import CoreGraphics

enum RecorderError: LocalizedError, Equatable {
    case microphoneDenied, screenDenied, noDisplay, noMicrophone, cancelled
    /// AVAudioEngine raised an Objective-C exception (its "required condition is false: …" checks), caught and turned
    /// into this error with the engine's own words, instead of ending the app.
    case engine(String)

    var errorDescription: String? {
        switch self {
        case .engine(let reason):
            return "The microphone could not be set up: the audio engine raised “\(reason)”. Try again; if it happens again, try with the computer-sound box unticked, or with the microphone menu on the Mac's current input."
        case .microphoneDenied:
            return "Microphone access was denied. Allow Meeting Transcriber under System Settings > Privacy & Security > Microphone."
        case .screenDenied:
            return "The computer's sound is not being recorded: macOS reports no permission for Screen & System Audio Recording. If you allowed Meeting Transcriber already, quit and reopen the app; if the app was rebuilt or updated, remove its old entry under System Settings > Privacy & Security > Screen & System Audio Recording (−) and allow it again, or run in Terminal: tccutil reset ScreenCapture com.github.avza-bl.meeting-transcriber (an ad-hoc signed build is a new identity at every update; see README: Permissions and updates). No picture is recorded. Your microphone is being recorded alone."
        case .noDisplay:
            return "No display was found to capture the system sound from."
        case .noMicrophone:
            return "No microphone was found."
        case .cancelled:
            return "The recording was stopped before it began."
        }
    }
}

/// One track being written: every channel of what arrives is averaged into one (at the buffer's own rate), then converted
/// to 16 kHz and appended. Until 1.10.1 the rate was converted first, into a format with the buffer's channel count, and
/// AVAudioFormat(commonFormat:sampleRate:channels:interleaved:) gives nil for more than two channels: every buffer of the
/// voice processor's 3-channel microphone input (macOS 27.2, a Bluetooth speaker and a Bluetooth headset) was dropped
/// unseen, while the silence fill kept the track's length, so the microphone track came out as zeros.
private final class TrackSink {
    let writer: WavWriter
    private var converter: AVAudioConverter?
    private let stats = NSLock()
    private var bufferCount = 0
    private var droppedCount = 0
    private var loudest: Float = 0
    private var channelPeaks: [Float] = []

    init(writer: WavWriter) { self.writer = writer }

    /// How many buffers arrived, and the loudest sample seen (0…1): a track that got none, or only silence, was not
    /// really recorded (nothing was playing, or the capture delivered nothing).
    var delivered: (buffers: Int, peak: Double) {
        stats.lock(); defer { stats.unlock() }
        return (bufferCount, Double(loudest))
    }

    /// Buffers that arrived but could not be written (a sample format without float data, a converter that failed).
    var dropped: Int {
        stats.lock(); defer { stats.unlock() }
        return droppedCount
    }

    /// The loudest sample of each input channel so far ("0.210/0.000/0.000"), empty for a one-channel input: which of
    /// several channels carries the sound.
    var channelPeakText: String {
        stats.lock(); defer { stats.unlock() }
        guard channelPeaks.count > 1 else { return "" }
        return channelPeaks.map { String(format: "%.3f", $0) }.joined(separator: "/")
    }

    private func drop() {
        stats.lock(); droppedCount += 1; stats.unlock()
    }

    func consume(_ buffer: AVAudioPCMBuffer, startedAt: Date) {
        let duration = Double(buffer.frameLength) / buffer.format.sampleRate
        // Where this buffer should end if nothing had been missed since the recording began. When the wall clock says it is
        // much later (the track began late, a device changed, buffers were dropped), fill the gap with silence, so both
        // tracks stay on one timeline and the lines of the two sides merge in the right order.
        let late = Date().timeIntervalSince(startedAt) - (writer.seconds + duration)
        if late > 0.3 { writer.appendSilence(seconds: late) }
        guard let mono = mixedToMono(buffer),
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(WavWriter.sampleRate),
                                         channels: 1, interleaved: false) else { drop(); return }
        if converter == nil || converter?.inputFormat != mono.format {
            converter = AVAudioConverter(from: mono.format, to: target)        // one channel in, one out: only the rate changes
        }
        guard let converter = converter else { drop(); return }
        let capacity = AVAudioFrameCount(Double(mono.frameLength) * target.sampleRate / mono.format.sampleRate) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { drop(); return }
        var consumed = false
        var error: NSError?
        let status = converter.convert(to: out, error: &error) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return mono
        }
        guard status != .error else { drop(); return }
        guard out.frameLength > 0, let data = out.floatChannelData else { return }
        let count = Int(out.frameLength)
        var samples = [Int16](repeating: 0, count: count)
        var peak: Float = 0
        for i in 0..<count {
            let sample = max(-1, min(1, data[0][i]))
            peak = max(peak, abs(sample))
            samples[i] = Int16(sample * 32767)
        }
        stats.lock(); bufferCount += 1; loudest = max(loudest, peak); stats.unlock()
        writer.append(samples)
    }

    /// The buffer with every channel averaged into one, float32 at the buffer's own rate, interleaved or not; nil for a
    /// sample format without float data (the microphone and both system captures deliver float32).
    private func mixedToMono(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let format = buffer.format
        let channels = Int(format.channelCount)
        guard channels > 0, format.commonFormat == .pcmFormatFloat32, let data = buffer.floatChannelData,
              let monoFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate, channels: 1, interleaved: false),
              let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: max(1, buffer.frameLength)),
              let out = mono.floatChannelData?[0] else { return nil }
        let frames = Int(buffer.frameLength)
        mono.frameLength = buffer.frameLength
        var peaks = [Float](repeating: 0, count: channels)
        let interleaved = format.isInterleaved && channels > 1             // interleaved: one block, channels side by side
        for i in 0..<frames {
            var sum: Float = 0
            for c in 0..<channels {
                let value = interleaved ? data[0][i * channels + c] : data[c][i]
                peaks[c] = max(peaks[c], abs(value))
                sum += value
            }
            out[i] = sum / Float(channels)
        }
        stats.lock()
        if channelPeaks.count != channels { channelPeaks = [Float](repeating: 0, count: channels) }
        for c in 0..<channels { channelPeaks[c] = max(channelPeaks[c], peaks[c]) }
        stats.unlock()
        return mono
    }
}

/// Records a call as two tracks written side by side: the microphone (what you say) and the sound this Mac plays
/// (what the other side says), the second through ScreenCaptureKit (macOS 13 or newer; it needs the Screen & System
/// Audio Recording permission, but no picture is taken). Each track is a mono 16 kHz WAV file for `mt.py call`.
final class CallRecorder: NSObject, SCStreamOutput, SCStreamDelegate {
    struct Tracks {
        let mic: URL
        let system: URL
    }

    /// Called on the main queue when the capture ends by itself (permission withdrawn, device gone …).
    var onFailure: ((String) -> Void)?
    /// Called on the main queue for something worth telling that does not end the recording (the microphone changed).
    var onNotice: ((String) -> Void)?
    /// Called on the main queue about once a second with the microphone's loudness (0…1).
    var onLevel: ((Double) -> Void)?
    /// Whether the microphone's echo cancellation (Apple's voice processing) is on: what the Mac plays is taken out
    /// of the microphone, so a call without headphones does not put the other side on your track.
    private(set) var echoCancelled = false

    private let queue = DispatchQueue(label: "meeting-transcriber.recorder")
    private let engine = AVAudioEngine()
    private let lifecycle = NSLock()                    // guards the fields below that start() and stop() both touch
    private var isStopped = false
    private var stream: SCStream?
    private var micSink: TrackSink?
    private var systemSink: TrackSink?
    private var syncTimer: DispatchSourceTimer?
    private var configObserver: NSObjectProtocol?
    private var startedAt = Date()
    private var filter: SCContentFilter?
    private var configuration: SCStreamConfiguration?
    private var healthTimer: DispatchSourceTimer?
    private var stalls = StallDetector()
    private var micDead = false
    private var micSilenceTold = false                  // the early note for a microphone that delivers only zeros, once
    private var micFirstBufferLogged = false            // on the recorder's queue: capture.log has the first buffer since the (re)start
    private var deviceMicrophone: DeviceMicrophone?     // a chosen microphone through its own HAL unit (guarded by `lifecycle`)
    private var engineInUse = false                     // on the recorder's queue: the audio engine records the microphone
    private var systemDead = false
    private var systemRestarts = 0
    private var audioTap: AnyObject?                    // SystemAudioTap on macOS 14.2+ (the type is not available below)
    private var tapInUse = false
    private var earlySilenceTold = false
    private var levelSum: Float = 0
    private var levelCount = 0
    private var levelAt = Date()
    private var clock = PauseClock()                    // guarded by `lifecycle`
    private var microphoneChoice: (uid: String, name: String)?   // the input device chosen in the window (nil: the Mac's current one)
    private var outputChoice: (uid: String, name: String)?       // the output device whose sound is the other side (nil: the current one)
    // capture.log in the recording's folder: how the two tracks were set up and what arrived, every 10 s, so a silent
    // track can be explained from the file instead of guessed at (guarded by `lifecycle`).
    private var captureLog: [String] = []
    private var captureLogURL: URL?
    private var healthTicks = 0

    /// Writes capture.log one line at a time, in order, off the recorder's queue.
    private let logWriter = DispatchQueue(label: "meeting-transcriber.capture-log")

    /// One line for capture.log. The file is written again after every line (it is small), not only at Stop: in 1.10.0 a
    /// Stop that never finished left no capture.log at all.
    private func log(_ text: String) {
        let stamp = String(format: "%7.1f s", Date().timeIntervalSince(startedAt))
        lifecycle.lock(); captureLog.append("\(stamp)  \(text)"); lifecycle.unlock()
        logWriter.async { [weak self] in self?.writeCaptureLog() }
    }

    private func writeCaptureLog() {
        lifecycle.lock(); let lines = captureLog; let url = captureLogURL; lifecycle.unlock()
        guard let url = url, !lines.isEmpty else { return }
        try? (lines.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
    }

    var isPaused: Bool {
        lifecycle.lock(); defer { lifecycle.unlock() }
        return clock.isPaused
    }

    /// Nothing is written until `resume()`: the buffers that arrive meanwhile are dropped, so both tracks' timelines leave
    /// the pause out (the recorded seconds, and the limit counted on them, stand still).
    func pause() {
        lifecycle.lock(); clock.pause(); lifecycle.unlock()
    }

    /// Goes on recording; the pause is told as a timed note. The timeline's origin moves by the pause, so the silence
    /// fill (which compares the wall clock with what was written) does not pad it.
    func resume() {
        let recorded = recordedSeconds
        lifecycle.lock()
        guard clock.isPaused else { lifecycle.unlock(); return }
        let length = clock.resume()
        startedAt = startedAt.addingTimeInterval(length)
        lifecycle.unlock()
        queue.async { [weak self] in
            self?.stalls.saw("microphone")
            self?.stalls.saw("the computer's sound")
        }
        let note = PauseClock.note(recordedSeconds: recorded, pausedFor: length)
        DispatchQueue.main.async { [weak self] in self?.onNotice?(note) }
    }

    private func sinks() -> (mic: TrackSink?, system: TrackSink?) {
        lifecycle.lock(); defer { lifecycle.unlock() }
        return (micSink, systemSink)
    }

    /// Seconds of audio written so far on the longer track: the recording's own clock (a sleeping Mac does not advance
    /// it, and a track that died does not stop it while the other goes on).
    var recordedSeconds: Double {
        let (mic, system) = sinks()
        return max(mic?.writer.seconds ?? 0, system?.writer.seconds ?? 0)
    }

    private func checkRunning() throws {
        lifecycle.lock(); defer { lifecycle.unlock() }
        if isStopped { throw RecorderError.cancelled }
    }

    /// Starts both tracks in `folder` (mic.wav, system.wav). The result arrives on the main queue. `stop()` may be called
    /// at any moment, also while a permission prompt is open: whatever this opens afterwards is closed again.
    /// `microphone` is the input device to record and `output` the output device whose sound is the other side, each
    /// by UID with its name; nil means the Mac's current one. A chosen device that is not connected is told as a notice
    /// and the current one is used instead.
    func start(in folder: URL, microphone: (uid: String, name: String)? = nil, output: (uid: String, name: String)? = nil,
               completion: @escaping (Result<Tracks, Error>) -> Void) {
        let tracks = Tracks(mic: folder.appendingPathComponent("mic.wav"), system: folder.appendingPathComponent("system.wav"))
        microphoneChoice = microphone
        outputChoice = output
        captureLogURL = folder.appendingPathComponent("capture.log")
        startedAt = Date()
        let requirement = CodeSignature.designatedRequirement() ?? "unreadable"
        log("Meeting Transcriber \(Runner.appVersion), macOS \(ProcessInfo.processInfo.operatingSystemVersionString)")
        log("code signature: \(SigningIdentity.kind(ofRequirement: requirement) == .stable ? "stable identity" : "ad-hoc (a new identity at every build: a permission granted to an older build does not count)"); designated requirement: \(requirement)")
        log("microphone: \(microphone.map { "\($0.name) (\($0.uid))" } ?? "the Mac's current input"); computer's sound from: \(output.map { "\($0.name) (\($0.uid)) only" } ?? "every output (every app, every output device; the capture clocked on the Mac's current output)"); capture path: \(CallRecorder.useAudioTap ? "Core Audio tap" : "screen capture")")
        Task { [self] in
            do {
                // Your side: without a microphone there is no call to record.
                try await CallRecorder.requestMicrophone()
                try checkRunning()
                if let chosen = microphone, AudioDevices.id(forUID: chosen.uid) == nil {
                    microphoneChoice = nil
                    DispatchQueue.main.async { [weak self] in
                        self?.onNotice?(DeviceChoice(uid: chosen.uid, label: chosen.name).notConnectedNote(what: "microphone", instead: "the Mac's current sound input"))
                    }
                }
                if let chosen = output, AudioDevices.id(forUID: chosen.uid) == nil {
                    outputChoice = nil
                    DispatchQueue.main.async { [weak self] in
                        self?.onNotice?(DeviceChoice(uid: chosen.uid, label: chosen.name).notConnectedNote(what: "sound output", instead: "the Mac's current output"))
                    }
                }
                let micWriter = try WavWriter(url: tracks.mic)
                let systemWriter = try WavWriter(url: tracks.system)       // a valid (maybe empty) file either way
                for writer in [micWriter, systemWriter] {
                    writer.onFailure = { [weak self] error in
                        DispatchQueue.main.async {
                            self?.onFailure?("The recording could not be saved: \(error.localizedDescription)")
                        }
                    }
                }
                let mic = TrackSink(writer: micWriter)
                let system = TrackSink(writer: systemWriter)
                lifecycle.lock(); micSink = mic; systemSink = system; lifecycle.unlock()

                // Your microphone, on the recorder's queue (one thread at a time touches the engine) and inside an
                // Objective-C catch (what the engine raises becomes an error, not the end of the app).
                try await startMicrophone()

                // The other side: the sound this Mac plays. When it cannot be captured (permission, no display, a
                // ScreenCaptureKit error) the call goes on with the microphone alone and says so; it is not an error.
                log("microphone running: echo cancellation \(echoCancelled ? "on" : "off"), input format \(microphoneFormatText)\(duckingText)")
                // One Bluetooth device as microphone and output (a headset, a speakerphone): its microphone puts it in its
                // call mode, and the capture of the computer's sound runs on that same device, a second client of its
                // microphone. Said before the capture starts, in the log and the window.
                if let micUID = microphoneChoice?.uid ?? AudioDevices.defaultUID(.input),
                   let outputUID = outputChoice?.uid ?? AudioDevices.defaultUID(.output),
                   AudioDevices.sameBluetoothDevice(micUID, outputUID) {
                    let name = AudioDevices.id(forUID: outputUID).flatMap { AudioDevices.name(of: $0) } ?? outputUID
                    let note = TrackSilence.sharedBluetoothNote(device: name)
                    log("microphone and output are one Bluetooth device: \(name) (\(micUID), \(outputUID))")
                    DispatchQueue.main.async { [weak self] in self?.onNotice?(note) }
                }
                do {
                    try await startSystemCapture()
                } catch RecorderError.cancelled {
                    throw RecorderError.cancelled
                } catch {
                    lifecycle.lock(); systemDead = true; lifecycle.unlock()
                    log("computer's sound: could not be started: \(error.localizedDescription)")
                    let reason = (error as? RecorderError) == .screenDenied ? error.localizedDescription
                        : "The computer's sound could not be recorded (\(error.localizedDescription)); your microphone is being recorded alone."
                    DispatchQueue.main.async { [weak self] in self?.onNotice?(reason) }
                    checkBothDead()        // the microphone may have died while this was awaited: then nothing records, and the call ends
                }
                try checkRunning()
                startSyncTimer()
                startHealthTimer()
                DispatchQueue.main.async { completion(.success(tracks)) }
            } catch {
                shutDown()
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    /// Whether the computer's sound goes through the Core Audio tap (macOS 14.2+; only the System Audio Recording Only
    /// permission) rather than ScreenCaptureKit (Screen Recording). `MT_SYSTEM_CAPTURE=sck` or the default `systemCapture`
    /// = "sck" keep the old path, for a comparison.
    static var useAudioTap: Bool {
        if ProcessInfo.processInfo.environment["MT_SYSTEM_CAPTURE"] == "sck" { return false }
        if UserDefaults.standard.string(forKey: "systemCapture") == "sck" { return false }
        if #available(macOS 14.2, *) { return true }
        return false
    }

    /// The other side: the Core Audio tap where it exists, else (or when it cannot be made) ScreenCaptureKit. Throws when
    /// neither works; `stop()` meanwhile ends in `.cancelled`.
    private func startSystemCapture() async throws {
        if #available(macOS 14.2, *), CallRecorder.useAudioTap {
            do {
                try startAudioTap()
                return
            } catch {
                let reason = error.localizedDescription
                DispatchQueue.main.async { [weak self] in
                    self?.onNotice?("The computer's sound could not be captured through the audio tap (\(reason)); trying the screen capture instead.")
                }
            }
        }
        try await startScreenCapture()
    }

    @available(macOS 14.2, *)
    private func startAudioTap() throws {
        let tap = SystemAudioTap(owner: queue, output: outputChoice)
        tap.onBuffer = { [weak self] buffer in                              // on the tap's own delivery queue
            guard let self = self else { return }
            if !self.isPaused { self.sinks().system?.consume(buffer, startedAt: self.startedAt) }
            self.queue.async { self.stalls.saw("the computer's sound") }
        }
        tap.onNotice = { [weak self] text in
            self?.log("computer's sound: \(text)")
            DispatchQueue.main.async { self?.onNotice?(text) }
        }
        tap.onFailure = { [weak self] text in
            guard let self = self else { return }
            self.lifecycle.lock(); self.systemDead = true; self.lifecycle.unlock()
            self.log("computer's sound: FAILED: \(text)")
            DispatchQueue.main.async { self.onNotice?(text) }
            self.checkBothDead()
        }
        tap.onLog = { [weak self] text in self?.log("computer's sound: \(text)") }
        tap.onDiagnostic = { [weak self] text in
            self?.log("computer's sound: \(text)")
            DispatchQueue.main.async { self?.onNotice?("Computer's sound: Core Audio \(text)") }
        }
        try tap.start()
        lifecycle.lock(); audioTap = tap; tapInUse = true; lifecycle.unlock()
    }

    /// The tap stops on its own control queue, where its listeners and rebuilds run too, so a change of output arriving at
    /// Stop cannot rebuild a capture that has just been torn down. The recording waits for it at most `tapStopWait`: a
    /// Core Audio call that does not return (seen in 1.10.0 after a Bluetooth speaker's rate was changed by hand during a
    /// call) must not hold the files, capture.log and the transcription.
    private func stopAudioTap() {
        lifecycle.lock(); let tap = audioTap; audioTap = nil; lifecycle.unlock()
        guard #available(macOS 14.2, *), let running = tap as? SystemAudioTap else { return }
        if !running.stop(timeout: CallRecorder.tapStopWait) {
            log("computer's sound: the capture did not stop within \(Int(CallRecorder.tapStopWait)) s (a Core Audio call did not return); the recording is saved without waiting for it")
        }
    }

    /// For capture.log: the loudest sample of each channel of a track with several, and the buffers it could not write.
    private static func sinkDetails(_ sink: TrackSink?) -> String {
        guard let sink = sink else { return "" }
        var parts: [String] = []
        let channels = sink.channelPeakText
        if !channels.isEmpty { parts.append("channels \(channels)") }
        let dropped = sink.dropped
        if dropped > 0 { parts.append("\(dropped) buffers dropped") }
        return parts.isEmpty ? "" : " [" + parts.joined(separator: ", ") + "]"
    }

    /// How long Stop waits for the computer's-sound capture, and for the recorder's queue, before it goes on without them.
    private static let tapStopWait: TimeInterval = 3

    /// Waits until the work already on the recorder's queue has run, at most `seconds`; false when it did not in time.
    private func waitForQueue(seconds: TimeInterval) -> Bool {
        let done = DispatchSemaphore(value: 0)
        queue.async { done.signal() }
        return done.wait(timeout: .now() + seconds) == .success
    }

    /// The ScreenCaptureKit side: permission, a display to attach the audio capture to, the stream. Throws when any of it
    /// is refused; `stop()` meanwhile ends in `.cancelled`.
    private func startScreenCapture() async throws {
        if let chosen = outputChoice {                                 // the screen capture takes the sound of every output
            DispatchQueue.main.async { [weak self] in
                self?.onNotice?("The screen capture takes the sound of every output, so the chosen output “\(chosen.name)” does not apply to it.")
            }
        }
        if !CGPreflightScreenCaptureAccess() {
            _ = CGRequestScreenCaptureAccess()                         // macOS shows its prompt; the grant counts after a relaunch
            throw RecorderError.screenDenied
        }
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        try checkRunning()
        guard let display = content.displays.first else { throw RecorderError.noDisplay }
        let configuration = SCStreamConfiguration()
        configuration.capturesAudio = true
        configuration.excludesCurrentProcessAudio = true      // not this app's own sounds
        configuration.sampleRate = 48000
        configuration.channelCount = 2
        configuration.width = 2                                // no picture is wanted; keep the video part tiny
        configuration.height = 2
        configuration.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let contentFilter = SCContentFilter(display: display, excludingWindows: [])
        lifecycle.lock(); self.filter = contentFilter; self.configuration = configuration; lifecycle.unlock()
        let captured = SCStream(filter: contentFilter, configuration: configuration, delegate: self)
        try captured.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await captured.startCapture()
        lifecycle.lock(); stream = captured; lifecycle.unlock()
        if (try? checkRunning()) == nil {                      // stop() came while the capture was starting
            try? await captured.stopCapture()
            throw RecorderError.cancelled
        }
    }

    /// Stops both tracks and closes the files; `completion` runs on the main queue.
    func stop(completion: @escaping () -> Void) {
        let recorded = recordedSeconds
        lifecycle.lock()
        isStopped = true
        let captured = stream
        stream = nil
        let neverStarted = systemDead && systemRestarts == 0 && captured == nil && !tapInUse
        // Stopped while paused: the pause is told, but the clock STAYS paused, so the buffers that still arrive until the
        // capture and the microphone are shut down are dropped like the rest of the pause (not written, not padded).
        let pauseNote: String? = clock.isPaused ? PauseClock.note(recordedSeconds: recorded, pausedFor: clock.currentPause()) : nil
        lifecycle.unlock()
        if let note = pauseNote { DispatchQueue.main.async { [weak self] in self?.onNotice?(note) } }
        Task { [self] in
            lifecycle.lock(); let tap = audioTap; lifecycle.unlock()     // kept past stopAudioTap(): its counters are the report
            if let captured = captured { try? await captured.stopCapture() }
            stopAudioTap()
            if !waitForQueue(seconds: CallRecorder.tapStopWait) {       // let the work already queued run (restarts, notes)
                log("the recorder's queue did not answer within \(Int(CallRecorder.tapStopWait)) s; the recording is saved without waiting for it")
            }
            let (micTrack, systemTrack) = sinks()                        // read before shutDown() lets the sinks go
            let system = systemTrack?.delivered
            let mic = micTrack?.delivered
            let seconds = recordedSeconds
            var tapReport = ""
            if #available(macOS 14.2, *), let running = tap as? SystemAudioTap { tapReport = running.report }
            log("stop after \(Int(seconds)) s: microphone \(mic?.buffers ?? 0) buffers peak \(String(format: "%.3f", mic?.peak ?? 0))\(CallRecorder.sinkDetails(micTrack)); computer's sound \(system?.buffers ?? 0) buffers peak \(String(format: "%.3f", system?.peak ?? 0))\(CallRecorder.sinkDetails(systemTrack))"
                + (tapReport.isEmpty ? "" : "; tap: \(tapReport)"))
            shutDown()
            logWriter.sync { writeCaptureLog() }                        // after the lines still queued for the file
            if let mic = mic, let note = TrackSilence.microphoneNote(buffers: mic.buffers, peak: mic.peak, seconds: seconds) {
                DispatchQueue.main.async { [weak self] in self?.onNotice?(note) }
            }
            if let system = system, !neverStarted,
               let note = TrackSilence.systemNote(buffers: system.buffers, peak: system.peak, seconds: seconds) {
                let details = " Details: \(system.buffers) buffers reached the track, loudest sample \(String(format: "%.3f", system.peak))"
                    + (tapReport.isEmpty ? "" : "; tap: \(tapReport)") + "; the whole account is in capture.log in the recording's folder."
                DispatchQueue.main.async { [weak self] in self?.onNotice?(note + details) }
            }
            DispatchQueue.main.async(execute: completion)
        }
    }

    // MARK: - Private

    private static func requestMicrophone() async throws {
        let allowed: Bool = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { continuation.resume(returning: $0) }
        }
        guard allowed else { throw RecorderError.microphoneDenied }
    }

    /// Runs `body` inside an Objective-C catch: an exception the engine raises (Swift cannot catch those, and uncaught
    /// one ends the app with SIGABRT, as in 1.9.6 and 1.9.7) comes back as `RecorderError.engine` with the engine's own
    /// reason; a Swift error thrown by `body` passes through unchanged.
    static func catchingObjC<T>(_ body: () throws -> T) throws -> T {
        var outcome: Result<T, Error>?
        let raised = withoutActuallyEscaping(body) { body in
            MTCatchObjCException { outcome = Result { try body() } }
        }
        if let raised = raised {
            throw RecorderError.engine("\(raised.name.rawValue): \(raised.reason ?? "no reason given")")
        }
        guard let outcome = outcome else { throw RecorderError.engine("the engine call did not return") }
        return try outcome.get()
    }

    /// The microphone bring-up, serialised on the recorder's queue with the restart after a configuration change, so
    /// that two threads never install a tap or start the engine at the same time.
    private func startMicrophone() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            queue.async { [self] in
                do {
                    try bringUpMicrophone()
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// On the recorder's queue: echo cancellation where it applies, the chosen device, the tap, the engine; a voice
    /// processor that refuses to start gives way to the plain microphone. Every engine call runs inside the
    /// Objective-C catch. The configuration-change observer is registered only at the end: a change notified during
    /// the bring-up must not start a second, concurrent one.
    /// `resettingTimeline`: the recording's timeline starts now (the first bring-up only; a fallback in the middle of a call
    /// keeps it, so the gap is filled with silence and both tracks stay on one timeline).
    private func bringUpMicrophone(resettingTimeline: Bool = true) throws {
        // A chosen microphone is recorded through its own HAL unit (DeviceMicrophone): the audio engine's input and output
        // are one unit bound to the Mac's default devices, and an engine pointed at another input gave no buffer at all
        // (1.10.2, macOS 27.2). Echo cancellation needs the engine, so it is off then, and the note says so.
        if let chosen = microphoneChoice {
            if let device = AudioDevices.id(forUID: chosen.uid) {
                do {
                    try startDeviceMicrophone(device, resettingTimeline: resettingTimeline)
                    DispatchQueue.main.async { [weak self] in
                        self?.onNotice?("Echo cancellation is off while a microphone is chosen in the settings (macOS's voice processing follows the Mac's current input only): use headphones, or what the Mac plays may show up on your track; whatever does is trimmed from your lines.")
                    }
                    return
                } catch {
                    log("microphone: “\(chosen.name)” could not be opened through its own input unit (\(error.localizedDescription)); the Mac's current input is used")
                    DispatchQueue.main.async { [weak self] in
                        self?.onNotice?("The microphone “\(chosen.name)” could not be opened (\(error.localizedDescription)); the Mac's current input is recorded instead.")
                    }
                }
            }
            microphoneChoice = nil
        }
        // The Mac's current input, through the audio engine. Echo cancellation: Apple's voice processing takes what the
        // Mac plays out of the microphone, so a call on the speakers does not put the other side on your own track. Not
        // every setup takes it.
        engineInUse = true
        try CallRecorder.catchingObjC {
            echoCancelled = (try? engine.inputNode.setVoiceProcessingEnabled(true)) != nil
            applyLeastDucking()
        }
        // The engine applies a new input binding (the voice processor, a chosen device) on its own queue; prepare()
        // waits for it, so the tap is installed against the settled hardware format.
        try CallRecorder.catchingObjC { engine.prepare() }
        guard try CallRecorder.catchingObjC({ try installMicrophoneTap() }) else { throw RecorderError.noMicrophone }
        if resettingTimeline { startedAt = Date() }
        do {
            try CallRecorder.catchingObjC { try engine.start() }
        } catch {
            guard echoCancelled else { throw error }
            echoCancelled = false                                  // the voice processor refused: plain microphone
            try CallRecorder.catchingObjC {
                engine.inputNode.removeTap(onBus: 0)
                try? engine.inputNode.setVoiceProcessingEnabled(false)
                engine.prepare()                                   // the plain unit is a new one: let it settle first
            }
            guard try CallRecorder.catchingObjC({ try installMicrophoneTap() }) else { throw RecorderError.noMicrophone }
            try CallRecorder.catchingObjC { try engine.start() }
            DispatchQueue.main.async { [weak self] in
                self?.onNotice?("Echo cancellation could not be enabled for the microphone: use headphones, or the other side may show up on your track.")
            }
        }
        let observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            self?.restartMicrophone(.configurationChange)
        }
        lifecycle.lock(); configObserver = observer; lifecycle.unlock()
    }

    /// The voice processor lowers ("ducks") what other apps play while it runs: Apple, "its volume will be significantly
    /// reduced. This is intentional" (developer forums, thread 829753). The smallest ducking is asked for, at the start and
    /// again at every restart of the engine: a configuration change makes the input unit anew (seen 0.7 s into a call), and
    /// nothing documents that the setting survives it; in a 1.10.1 call the sound the Mac played was inaudible from the start
    /// on, after such a restart. Inside the Objective-C catch, with echo cancellation on, macOS 14 or newer.
    private func applyLeastDucking() {
        guard echoCancelled, #available(macOS 14.0, *) else { return }
        engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration =
            AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
    }

    /// For capture.log: how much the voice processor says it lowers other apps' sound ("min" is what is asked for).
    private var duckingText: String {
        guard echoCancelled, #available(macOS 14.0, *) else { return "" }
        let ducking = engine.inputNode.voiceProcessingOtherAudioDuckingConfiguration
        let names = [0: "default", 10: "min", 20: "mid", 30: "max"]
        let raw = ducking.duckingLevel.rawValue
        return "; other apps' sound lowered: level \(names[raw] ?? String(raw)), advanced ducking \(ducking.enableAdvancedDucking.boolValue ? "on" : "off")"
    }

    /// Closes everything that is open. Safe to call twice and from either side of a start/stop race.
    private func shutDown() {
        lifecycle.lock()
        let mic = micSink, system = systemSink, timer = syncTimer, observer = configObserver, health = healthTimer
        micSink = nil
        systemSink = nil
        syncTimer = nil
        healthTimer = nil
        configObserver = nil
        lifecycle.unlock()
        timer?.cancel()
        health?.cancel()
        if let observer = observer { NotificationCenter.default.removeObserver(observer) }
        stopAudioTap()
        // The engine is touched on the recorder's queue only, so this runs after a restart that may still be going on there,
        // never beside it; Stop waits for it at most tapStopWait (a Bluetooth device's audio engine call that does not return
        // must not hold the files and the transcription). Never called on that queue.
        let engineDown = DispatchSemaphore(value: 0)
        queue.async { [self] in
            // A chosen microphone's own unit, here too: after a restart that may be going on, and within the time limit.
            lifecycle.lock(); let device = deviceMicrophone; deviceMicrophone = nil; lifecycle.unlock()
            device?.stop()
            if engineInUse {                                            // a chosen microphone never touched it
                try? CallRecorder.catchingObjC {
                    engine.inputNode.removeTap(onBus: 0)
                    engine.stop()
                    try? engine.inputNode.setVoiceProcessingEnabled(false)  // the voice processor is shared: free it for the next recording
                }
            }
            engineDown.signal()
        }
        if engineDown.wait(timeout: .now() + CallRecorder.tapStopWait) == .timedOut {
            log("the microphone did not stop within \(Int(CallRecorder.tapStopWait)) s (an audio engine call did not return); the recording is saved without waiting for it")
        }
        mic?.writer.close()                                             // a late microphone buffer finds no sink: nothing is written
        system?.writer.close()
    }

    /// Points the engine's input at the chosen microphone (the HAL unit's current device); without a choice, or when
    /// the device cannot be selected, the engine keeps the Mac's current input and the window is told. Called again
    /// whenever the input unit is made anew (voice processing switched off, a configuration change).
    private func applyMicrophoneChoice() {
        guard let chosen = microphoneChoice else { return }
        guard var device = AudioDevices.id(forUID: chosen.uid) else {
            microphoneChoice = nil
            DispatchQueue.main.async { [weak self] in
                self?.onNotice?("The microphone “\(chosen.name)” went away; the Mac's current sound input is used.")
            }
            return
        }
        guard let unit = engine.inputNode.audioUnit else { return }
        let status = AudioUnitSetProperty(unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                          &device, UInt32(MemoryLayout<AudioObjectID>.size))
        if status != noErr {
            microphoneChoice = nil
            DispatchQueue.main.async { [weak self] in
                self?.onNotice?("The microphone “\(chosen.name)” could not be selected (Core Audio error \(status)); the Mac's current sound input is used.")
            }
        }
    }

    /// Installs the microphone tap; false when there is no usable input. The tap takes the node's own format (nil): a
    /// format handed in must equal the hardware format at that very moment, and the engine raises an exception, not an
    /// error, when it does not (an input binding still changing, as in the crash of 1.9.6). The hardware side is checked
    /// first: an input with no channels or no sample rate (no device, no permission) makes the engine raise as well.
    private func installMicrophoneTap() throws -> Bool {
        let input = engine.inputNode
        input.removeTap(onBus: 0)                                      // never a second tap on the bus (the engine raises for one)
        let hardware = input.inputFormat(forBus: 0)
        let output = input.outputFormat(forBus: 0)
        guard hardware.sampleRate > 0, hardware.channelCount > 0, output.sampleRate > 0, output.channelCount > 0 else { return false }
        input.installTap(onBus: 0, bufferSize: 4096, format: nil) { [weak self] buffer, _ in
            self?.microphoneBuffer(buffer)
        }
        return true
    }

    /// A microphone buffer, from the engine's tap or from a chosen device's own unit: written to the track, and noted on
    /// the recorder's queue.
    private func microphoneBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let sink = sinks().mic else { return }
        if isPaused {                                                   // dropped, but the track is alive
            queue.async { self.stalls.saw("microphone") }
            return
        }
        sink.consume(buffer, startedAt: startedAt)
        queue.async { self.noteMicrophone(buffer) }
    }

    private var usesDeviceMicrophone: Bool {
        lifecycle.lock(); defer { lifecycle.unlock() }
        return deviceMicrophone != nil
    }

    /// For capture.log: the microphone's format, and through what it is recorded.
    private var microphoneFormatText: String {
        lifecycle.lock(); let device = deviceMicrophone; lifecycle.unlock()
        if let device = device { return "\(device.format.sampleRate) Hz \(device.format.channelCount) ch, through its own input unit" }
        let format = engine.inputNode.inputFormat(forBus: 0)
        return "\(format.sampleRate) Hz \(format.channelCount) ch"
    }

    /// For capture.log's 10 s line: the state of what records the microphone.
    private var microphoneStateText: String {
        if usesDeviceMicrophone { return "own input unit" }
        return "audio engine " + (engine.isRunning ? "running" : "NOT running")
    }

    /// On the recorder's queue: the chosen microphone through its own HAL unit (DeviceMicrophone), echo cancellation off.
    private func startDeviceMicrophone(_ device: AudioObjectID, resettingTimeline: Bool = true) throws {
        let mic = try DeviceMicrophone(device: device) { [weak self] buffer in self?.microphoneBuffer(buffer) }
        if resettingTimeline { startedAt = Date() }
        try mic.start()
        echoCancelled = false
        lifecycle.lock(); deviceMicrophone = mic; lifecycle.unlock()
    }

    /// On the recorder's queue: the chosen microphone's unit made anew (it stalled). When the device is gone or will not
    /// open, the Mac's current input is recorded instead, through the audio engine with echo cancellation.
    private func restartDeviceMicrophone() {
        let deliveredSinceOpen = micFirstBufferLogged                   // false: not one buffer since this unit was opened
        lifecycle.lock(); let old = deviceMicrophone; deviceMicrophone = nil; lifecycle.unlock()
        old?.stop()
        guard (try? checkRunning()) != nil else { return }
        // A device that delivered before (and stalled) is opened again; one that gave nothing since it was opened is not
        // opened again and again: the Mac's current input takes over.
        if deliveredSinceOpen, let chosen = microphoneChoice, let device = AudioDevices.id(forUID: chosen.uid) {
            do {
                let mic = try DeviceMicrophone(device: device) { [weak self] buffer in self?.microphoneBuffer(buffer) }
                try mic.start()
                lifecycle.lock()
                let stopped = isStopped
                if !stopped { deviceMicrophone = mic; micDead = false }
                lifecycle.unlock()
                if stopped {                                            // Stop came meanwhile: its teardown has run
                    mic.stop()
                    return
                }
                stalls.saw("microphone")
                micFirstBufferLogged = false
                log("microphone running again: “\(chosen.name)”, \(mic.format.sampleRate) Hz \(mic.format.channelCount) ch, through its own input unit")
                DispatchQueue.main.async { self.onNotice?("The microphone was started again after 10 s without sound.") }
                return
            } catch {
                log("microphone: “\(chosen.name)” could not be opened again (\(error.localizedDescription))")
            }
        }
        if let chosen = microphoneChoice {
            log("microphone: “\(chosen.name)” \(deliveredSinceOpen ? "is gone or would not open again" : "gave nothing since it was opened"); the Mac's current input takes over")
        }
        microphoneChoice = nil                                          // gone, or it would not open: the Mac's current input
        do {
            try bringUpMicrophone(resettingTimeline: false)
            lifecycle.lock(); micDead = false; lifecycle.unlock()
            stalls.saw("microphone")
            micFirstBufferLogged = false
            log("microphone running again: the Mac's current input, echo cancellation \(echoCancelled ? "on" : "off"), input format \(microphoneFormatText)\(duckingText)")
            DispatchQueue.main.async { self.onNotice?("The chosen microphone stopped delivering; the Mac's current input is recorded instead.") }
        } catch {
            lifecycle.lock(); micDead = true; lifecycle.unlock()
            log("microphone: FAILED to start again: \(error.localizedDescription)")
            DispatchQueue.main.async {
                self.onNotice?("The microphone stopped (\(error.localizedDescription)); your side is no longer recorded, the computer's sound still is.")
            }
            checkBothDead()
        }
    }

    /// On the recorder's queue: the microphone delivered; its loudness is reported about once a second.
    private func noteMicrophone(_ buffer: AVAudioPCMBuffer) {
        stalls.saw("microphone")
        if !micFirstBufferLogged {                       // when the input began to deliver (or again, after a restart)
            micFirstBufferLogged = true
            log("microphone: first buffer (\(Int(buffer.format.sampleRate)) Hz, \(buffer.format.channelCount) ch, \(buffer.frameLength) frames)")
        }
        if let data = buffer.floatChannelData, buffer.frameLength > 0 {
            let n = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<n { sum += data[0][i] * data[0][i] }
            levelSum += sum
            levelCount += n
        }
        if Date().timeIntervalSince(levelAt) >= 1, levelCount > 0 {
            let rms = Double((levelSum / Float(levelCount)).squareRoot())
            levelSum = 0; levelCount = 0; levelAt = Date()
            DispatchQueue.main.async { [weak self] in self?.onLevel?(min(1, rms * 4)) }   // a speaking voice reaches about 0.25 RMS
        }
    }

    /// Every 5 s: a track that delivered nothing for 10 s is reported, and started again where that can help.
    private func startHealthTimer() {
        lifecycle.lock(); let systemRuns = !systemDead; lifecycle.unlock()
        queue.async { [self] in                          // the stall detector lives on the recorder's queue, like stalls.saw
            stalls.register("microphone")
            if systemRuns { stalls.register("the computer's sound") }
        }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 5, repeating: 5)
        timer.setEventHandler { [weak self] in
            guard let self = self, (try? self.checkRunning()) != nil, !self.isPaused else { return }
            let (mic, system) = self.sinks()
            self.healthTicks += 1
            if self.healthTicks % 2 == 0, let s = system?.delivered, let m = mic?.delivered {     // every 10 s
                var tapReport = ""
                if #available(macOS 14.2, *), let running = self.audioTap as? SystemAudioTap { tapReport = running.report }
                self.log("microphone \(m.buffers) buffers peak \(String(format: "%.3f", m.peak))\(CallRecorder.sinkDetails(mic)) (\(self.microphoneStateText)); computer's sound \(s.buffers) buffers peak \(String(format: "%.3f", s.peak))\(CallRecorder.sinkDetails(system))"
                         + (tapReport.isEmpty ? "" : "; tap: \(tapReport)"))
            }
            // A microphone that delivers buffers of zeros: the stall detector cannot see it (buffers come), the user can
            // (the level bar stays flat). Said once, at 20 s; with echo cancellation on the microphone is started again
            // without it, the one remedy the recording has (seen in 1.9.11 on a Bluetooth headset that was the Mac's
            // input and output: the transcript found no speech on the microphone; not yet measured which unit delivered zeros).
            if !self.micSilenceTold, !self.micDead, let m = mic?.delivered,
               let note = TrackSilence.microphoneEarlyNote(seconds: self.recordedSeconds, buffers: m.buffers, peak: m.peak, echoCancelled: self.echoCancelled) {
                self.micSilenceTold = true
                self.log("microphone: \(note)")
                DispatchQueue.main.async { self.onNotice?(note) }
                if self.echoCancelled { self.restartMicrophone(.silentWithEchoCancellation) }
            }
            if !self.earlySilenceTold, !self.systemDead, let system = system?.delivered, let mic = mic?.delivered,
               let note = TrackSilence.earlyNote(seconds: self.recordedSeconds, systemBuffers: system.buffers, systemPeak: system.peak,
                                                 microphonePeak: mic.peak, audioTap: self.tapInUse) {
                self.earlySilenceTold = true
                DispatchQueue.main.async { self.onNotice?(note) }
                // A tap that runs but delivers only zeros while sound is audible is a reported state of Core Audio's taps
                // (after a device or rate change); the one remedy reported is to make tap and aggregate anew. Once.
                if #available(macOS 14.2, *), let tap = self.audioTap as? SystemAudioTap {
                    tap.rebuild(reason: "it had delivered only silence while your microphone had sound")
                }
            }
            for track in self.stalls.stalled(since: self.startedAt) {
                if track == "microphone" && !self.micDead {
                    if TrackSilence.restartsWithoutEchoCancellation(deliveredSinceStart: self.micFirstBufferLogged, echoCancelled: self.echoCancelled) {
                        DispatchQueue.main.async { self.onNotice?(TrackSilence.nothingWithEchoCancellationNote) }
                        self.restartMicrophone(.nothingWithEchoCancellation)
                    } else {
                        DispatchQueue.main.async { self.onNotice?("No sound has come from the microphone for 10 s; starting it again.") }
                        self.restartMicrophone(.stall)
                    }
                } else if track != "microphone" && !self.systemDead {
                    DispatchQueue.main.async { self.onNotice?("No sound has come from the computer for 10 s (quiet, or the capture stalled).") }
                }
            }
        }
        timer.resume()
        lifecycle.lock(); healthTimer = timer; lifecycle.unlock()
    }

    /// Both sides gone: nothing is being recorded any more.
    private func checkBothDead() {
        if micDead && systemDead {
            DispatchQueue.main.async { [weak self] in
                self?.onFailure?("Neither the microphone nor the computer's sound is being recorded any more.")
            }
        }
    }

    /// The system-audio capture died: one new stream on the same filter; if that fails too, the microphone goes on alone.
    private func restartSystemCapture(after error: Error) {
        Task { [self] in
            guard (try? checkRunning()) != nil else { return }
            lifecycle.lock()
            let filter = self.filter, configuration = self.configuration
            systemRestarts += 1
            let attempts = systemRestarts
            lifecycle.unlock()
            if attempts <= 1, let filter = filter, let configuration = configuration {
                do {
                    let captured = SCStream(filter: filter, configuration: configuration, delegate: self)
                    try captured.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
                    try await captured.startCapture()
                    lifecycle.lock()
                    let stopped = isStopped
                    if !stopped { stream = captured }
                    lifecycle.unlock()
                    if stopped {                                        // Stop or Cancel came while this one was starting
                        try? await captured.stopCapture()
                        return
                    }
                    DispatchQueue.main.async { self.onNotice?("The recording of the computer's sound stopped (\(error.localizedDescription)) and was started again.") }
                    return
                } catch {
                    // fall through: the microphone goes on alone
                }
            }
            lifecycle.lock(); systemDead = true; lifecycle.unlock()
            DispatchQueue.main.async {
                self.onNotice?("The recording of the computer's sound stopped (\(error.localizedDescription)) and could not be started again; your microphone goes on being recorded.")
            }
            checkBothDead()
        }
    }

    /// Why the microphone is started again; each is logged in capture.log and told in its own words.
    private enum MicrophoneRestart: String {
        case configurationChange = "the audio engine reported a configuration change (the input device, its format or its profile changed)"
        case stall = "no buffer came for 10 s"
        case silentWithEchoCancellation = "only zeros came for 20 s with echo cancellation on: it is started again without it"
        case nothingWithEchoCancellation = "not one buffer came in 10 s with echo cancellation on: it is started again without it"

        /// The restarts that give up the voice processor.
        var dropsEchoCancellation: Bool { return self == .silentWithEchoCancellation || self == .nothingWithEchoCancellation }
    }

    /// The input device changed (headphones plugged in or out, a headset connected …): AVAudioEngine stopped itself, so
    /// start it again on the new device. The gap is filled with silence by the track's timeline. Also the remedy for a
    /// microphone that stalled or that delivered only zeros; for the latter the voice processor is given up.
    private func restartMicrophone(_ reason: MicrophoneRestart) {
        queue.async { [weak self] in
            guard let self = self, (try? self.checkRunning()) != nil else { return }
            self.log("microphone: starting again: \(reason.rawValue)")
            if self.usesDeviceMicrophone {
                self.restartDeviceMicrophone()
                return
            }
            do {
                try CallRecorder.catchingObjC {
                    self.engine.inputNode.removeTap(onBus: 0)
                    if reason.dropsEchoCancellation && self.echoCancelled {
                        self.engine.stop()                             // voice processing is switched only on a stopped engine
                        try? self.engine.inputNode.setVoiceProcessingEnabled(false)
                        self.echoCancelled = false
                    }
                    self.applyMicrophoneChoice()                       // the chosen device stays, if it is still there
                    self.applyLeastDucking()                           // again: the restarted unit may be back at Apple's default
                    self.engine.prepare()                              // the new binding first, then the tap
                }
                guard (try? self.checkRunning()) != nil else { return }        // Stop came meanwhile: its teardown follows on this queue
                guard try CallRecorder.catchingObjC({ try self.installMicrophoneTap() }) else { throw RecorderError.noMicrophone }
                guard (try? self.checkRunning()) != nil else { return }
                try CallRecorder.catchingObjC { try self.engine.start() }
                self.lifecycle.lock(); self.micDead = false; self.lifecycle.unlock()
                self.stalls.saw("microphone")                          // the 10 s count starts afresh after a restart
                self.micFirstBufferLogged = false                      // the log says when the restarted input delivers
                let format = self.engine.inputNode.inputFormat(forBus: 0)
                self.log("microphone running again: echo cancellation \(self.echoCancelled ? "on" : "off"), input format \(format.sampleRate) Hz \(format.channelCount) ch\(self.duckingText)")
                let note: String
                switch reason {
                case .configurationChange: note = "The microphone changed; the recording went on with the new one."
                case .stall: note = "The microphone was started again after 10 s without sound."
                case .silentWithEchoCancellation, .nothingWithEchoCancellation:
                    note = "The microphone was started again without echo cancellation (what the Mac plays may now show up on your track; whatever does is trimmed from your lines)."
                }
                DispatchQueue.main.async { self.onNotice?(note) }
            } catch {
                self.lifecycle.lock(); self.micDead = true; self.lifecycle.unlock()
                self.log("microphone: FAILED to start again: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.onNotice?("The microphone stopped (\(error.localizedDescription)); your side is no longer recorded, the computer's sound still is.")
                }
                self.checkBothDead()
            }
        }
    }

    /// Every 30 s the sizes in the file headers are written, so a recording that is cut short can still be read.
    private func startSyncTimer() {
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 30, repeating: 30)
        timer.setEventHandler { [weak self] in
            let (mic, system) = self?.sinks() ?? (nil, nil)
            mic?.writer.sync()
            system?.writer.sync()
        }
        timer.resume()
        lifecycle.lock(); syncTimer = timer; lifecycle.unlock()
    }

    private static func pcmBuffer(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let basic = CMAudioFormatDescriptionGetStreamBasicDescription(description) else { return nil }
        var streamDescription = basic.pointee
        guard let format = AVAudioFormat(streamDescription: &streamDescription) else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        buffer.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        return status == noErr ? buffer : nil
    }

    // MARK: - SCStreamOutput, SCStreamDelegate

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferIsValid(sampleBuffer), let buffer = CallRecorder.pcmBuffer(from: sampleBuffer) else { return }
        if !isPaused { sinks().system?.consume(buffer, startedAt: startedAt) }    // paused: dropped, but the track is alive
        stalls.saw("the computer's sound")
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        lifecycle.lock()
        let current = self.stream === stream
        if current { self.stream = nil }
        lifecycle.unlock()
        guard current else { return }
        restartSystemCapture(after: error)
    }
}
