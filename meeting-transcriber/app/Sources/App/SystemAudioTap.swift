import Foundation
import AVFoundation
import CoreAudio

/// The sound this Mac plays, captured with a Core Audio process tap (macOS 14.2 or newer): a tap on every process but
/// this one, and on every output device unless one is chosen in the window (Apple: a tap's `deviceUID` "will have a
/// value if this tap only taps a specific hardware device"; without it the tap is not limited to one), read through a
/// private aggregate device whose clock is the chosen output device, else the Mac's current output. The tap's stereo
/// mix is averaged to one channel by the track sink, so both channels are on the track. Measured once, on one Mac (macOS 27.2, Apple
/// silicon, a Bluetooth headset) with `app/diagnose_tap.swift`: an aggregate clocked on the idle built-in speakers while
/// the sound went to the headset (the construction of 1.9.9 and 1.9.10) accepted AudioDeviceStart with noErr, reported
/// itself not running and never called back; clocked on the current output, the headset in its 16 kHz hands-free
/// profile, it ran. Whether an aggregate clocked on a chosen output that is not the current one runs was not measured;
/// the look 1.5 s after the start says what happened. The buffers arrive at the aggregate's sample rate (the clock's),
/// not at the rate the tap's format property states, so the format handed on takes the aggregate's rate. A change of
/// the clock's nominal sample rate, or of the current output, rebuilds the capture. It needs only the *System Audio Recording Only* permission
/// (`NSAudioCaptureUsageDescription`), not Screen Recording. There is no public way to ask whether that permission was
/// given: without it the tap delivers silence, every call returns noErr, and macOS shows its prompt the first time a
/// signed build starts the capture. The recorder notices the silence.
///
/// Threads: every Core Audio call that builds, rebuilds or tears down the capture, and every listener, runs on this
/// object's own control queue; the IOProc delivers on a second queue of its own. Neither is the recorder's queue, and
/// `stop(timeout:)` waits for the control queue only so long: in 1.10.0 a Core Audio call that did not return (a rebuild
/// after a Bluetooth speaker's rate was changed by hand in Audio MIDI Setup during a call) held the recorder's queue,
/// so the computer's-sound track stopped, Stop never finished, capture.log was never written and nothing was transcribed.
@available(macOS 14.2, *)
final class SystemAudioTap {
    enum TapError: LocalizedError {
        case status(String, OSStatus)
        case noOutputDevice
        case format

        var errorDescription: String? {
            switch self {
            case .status(let step, let code): return "\(step) failed (Core Audio error \(code))"
            case .noOutputDevice: return "no sound output device was found to attach the capture to"
            case .format: return "the tap's audio format could not be read"
            }
        }
    }

    /// Every buffer of the computer's sound, as a PCM buffer in the tap's own format, on this object's delivery queue
    /// (one buffer at a time, never after `stop(timeout:)` was called). Set before `start()`.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?
    /// Something worth telling (the output device changed and the capture moved with it), on the owner's queue.
    var onNotice: ((String) -> Void)?
    /// The capture ended by itself (the output device changed and the tap could not follow), on the owner's queue.
    var onFailure: ((String) -> Void)?
    /// What the tap is made of and what its callbacks bring, on the owner's queue: for the recording's capture.log, so a
    /// silent track can be told apart (no callbacks, callbacks with an unexpected layout, or callbacks full of zeros).
    var onDiagnostic: ((String) -> Void)?
    /// A line for the recording's capture.log only (not the window), on the owner's queue: a rebuild as it begins.
    var onLog: ((String) -> Void)?

    /// What one aggregate device's IOProc delivers through. Ended before that device is torn down, so a callback that
    /// still comes (the HAL may call once more while AudioDeviceStop waits) is dropped instead of written.
    private final class Delivery {
        let format: AVAudioFormat
        private let lock = NSLock()
        private var active = true
        private var count = 0

        init(format: AVAudioFormat) { self.format = format }

        /// Counts the callback and says whether it is still wanted.
        func take() -> Bool {
            lock.lock(); defer { lock.unlock() }
            if active { count += 1 }
            return active
        }

        func end() {
            lock.lock(); active = false; lock.unlock()
        }

        /// The callbacks of this aggregate so far (for the look 1.5 s after its start).
        var callbacks: Int {
            lock.lock(); defer { lock.unlock() }
            return count
        }
    }

    // The counters and the device state for the report: written on the delivery and the control queue, read from any
    // thread (the recorder's 10 s line, the report after the stop), so guarded by `stats`. Reading them never waits for
    // Core Audio.
    private let stats = NSLock()
    private var callbacks = 0                     // IOProc callbacks seen
    private var delivered = 0                     // buffers handed on
    private var emptyCallbacks = 0                // callbacks with an empty buffer list
    private var layoutMismatches = 0              // callbacks whose buffers did not add up to the tap's channels
    private var firstLayout = ""                  // the buffer list of the first callback: "3 buffers: 1ch×2048B, …"
    private var deviceState = "no capture device" // as last seen on the control queue
    private var stopped = false                   // stop(timeout:) was called: nothing more is handed on or built

    /// The counters so far, one line, with whether the capture device reports itself running (the aggregate of 1.9.9
    /// and 1.9.10 did not, with 0 callbacks, when the sound went to another output than its clock). The running state
    /// is the one last read on the control queue; a fresh look is asked for there, for the next report.
    var report: String {
        control.async { [weak self] in self?.refreshDeviceState() }
        stats.lock(); defer { stats.unlock() }
        return "\(callbacks) callbacks, \(delivered) delivered, \(emptyCallbacks) empty, \(layoutMismatches) with an unexpected layout; \(deviceState)"
            + (firstLayout.isEmpty ? "" : "; first callback: \(firstLayout)")
    }

    private var isStopped: Bool {
        stats.lock(); defer { stats.unlock() }
        return stopped
    }

    private func setDeviceState(_ text: String) {
        stats.lock(); deviceState = text; stats.unlock()
    }

    /// On the control queue: whether the HAL reports the aggregate device as running (kAudioDevicePropertyDeviceIsRunning);
    /// false when unknown.
    private var aggregateIsRunning: Bool {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsRunning,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        return AudioObjectGetPropertyData(aggregateID, &address, 0, nil, &size, &value) == noErr && value != 0
    }

    /// On the control queue: the state the report quotes, while there is an aggregate device and the capture was not stopped.
    private func refreshDeviceState() {
        guard aggregateID != kAudioObjectUnknown, !isStopped else { return }
        setDeviceState(aggregateIsRunning ? "capture device running" : "capture device NOT running")
    }

    private let owner: DispatchQueue                                   // where onNotice, onFailure and onDiagnostic are called
    private let control = DispatchQueue(label: "meeting-transcriber.tap.control")
    private let io = DispatchQueue(label: "meeting-transcriber.tap.delivery")

    // The capture itself: touched on the control queue only (start() and stop(timeout:) put their work there).
    /// The output device whose sound is taken (its UID and the name to show), or nil for whichever is the Mac's current
    /// output: the tap then follows a change of the output (headphones plugged in), while a chosen device is kept until
    /// it is disconnected, after which the capture goes on from the current output and says so.
    private var chosenOutput: (uid: String, name: String)?
    private var description: CATapDescription?
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var delivery: Delivery?
    private var format: AVAudioFormat?           // the tap's channels and sample format at the aggregate's rate
    private var tapRate = 0.0                     // the rate the tap's format property states (for the log)
    private var clockUID: String?                 // the sub-device the aggregate is clocked on
    private var clockIsDefaultOutput = false      // no chosen output: the clock follows the Mac's current output
    private var outputListener: AudioObjectPropertyListenerBlock?
    private var aliveListener: (device: AudioObjectID, block: AudioObjectPropertyListenerBlock)?
    private var rateListener: (device: AudioObjectID, block: AudioObjectPropertyListenerBlock)?
    private var running = false
    private var watchGeneration = 0               // bumped by unwatchOutputDevice(): a notification of an older registration still queued does nothing

    /// `owner`: the queue the notices, failures and diagnostics are called on (the recorder's).
    init(owner: DispatchQueue, output: (uid: String, name: String)? = nil) {
        self.owner = owner
        self.chosenOutput = output
    }

    private func notice(_ text: String) {
        let handler = onNotice
        owner.async { handler?(text) }
    }

    private func failure(_ text: String) {
        let handler = onFailure
        owner.async { handler?(text) }
    }

    private func diagnostic(_ text: String) {
        let handler = onDiagnostic
        owner.async { handler?(text) }
    }

    private func logLine(_ text: String) {
        let handler = onLog
        owner.async { handler?(text) }
    }

    /// Creates the tap and the aggregate device on the output device and starts delivering buffers. Called once, at the
    /// start of a call (never on the control queue); waits for the control queue.
    func start() throws {
        try control.sync { try startOnControl() }
    }

    private func startOnControl() throws {
        try createTap()
        do {
            try buildAggregate()
        } catch {
            tearDownTap()
            throw error
        }
        running = true
        watchOutputDevice()
        diagnostic(setupDescription)
    }

    /// The tap and the aggregate as built: format, clock device, exclusions.
    private var setupDescription: String {
        let formatText: String
        if let f = format {
            formatText = "\(Int(tapRate)) Hz stated, \(f.channelCount) ch, \(f.isInterleaved ? "interleaved" : "non-interleaved"), \(f.commonFormat == .pcmFormatFloat32 ? "float32" : "format \(f.commonFormat.rawValue)")"
        } else {
            formatText = "format unknown"
        }
        let clock = clockUID.flatMap { AudioDevices.id(forUID: $0) }.flatMap { AudioDevices.name(of: $0) } ?? "none"
        let limited = chosenOutput.map { " limited to output \($0.name)" } ?? ""
        let rate = format.map { "\(Int($0.sampleRate)) Hz" } ?? "rate unknown"
        return "tap \(tapID) (\(formatText))\(limited); aggregate \(aggregateID) clocked on \(clock) (\(clockUID ?? "-")), delivering at \(rate)"
            + "; excluded processes: \(description?.processes.count ?? 0)"
    }

    /// Ends the capture: from this call on no buffer is handed on and nothing is built again. Waits up to `timeout`
    /// seconds for the tap and the aggregate to be torn down and for the buffers already delivered to be handed on; false
    /// when that did not happen in time (a Core Audio call that did not return). The teardown then finishes, or not, on
    /// the control queue; the recording does not wait for it. Never called on the control or the delivery queue.
    @discardableResult
    func stop(timeout: TimeInterval) -> Bool {
        stats.lock(); stopped = true; stats.unlock()
        let done = DispatchSemaphore(value: 0)
        control.async { [self] in
            stopOnControl()
            io.async { done.signal() }                                 // after the buffers already queued there
        }
        return done.wait(timeout: .now() + timeout) == .success
    }

    private func stopOnControl() {
        running = false
        delivery?.end()                                                // a callback during the teardown is dropped
        stats.lock(); let before = deviceState; stats.unlock()        // kept when there was no device to look at
        let step: (String) -> Void = { [weak self] call in self?.setDeviceState("stopping: in \(call)") }
        let state: String?
        if aggregateID != kAudioObjectUnknown {
            step("AudioObjectGetPropertyData (is the capture device running)")
            state = aggregateIsRunning ? "capture device was running when stopped" : "capture device was NOT running when stopped"
        } else {
            state = nil
        }
        step("AudioObjectRemovePropertyListenerBlock")
        unwatchOutputDevice()
        tearDownAggregate(step: step)
        step("AudioHardwareDestroyProcessTap")
        tearDownTap()
        setDeviceState(state ?? before)
    }

    /// The whole capture made anew, tap and aggregate device both (recreating only one of them is reported not to help
    /// when a tap has gone silent while sound is audible). `reason` goes into the note. Returns at once; the work is done
    /// on the control queue.
    func rebuild(reason: String) {
        control.async { [weak self] in self?.rebuildOnControl(reason: reason) }
    }

    /// Tears the capture down and builds it again on `chosenOutput`, else the Mac's current output; `done` is the notice
    /// for a success, `failed` the failure text for an error. Nothing is built again once the capture was stopped, also
    /// when the teardown itself took long (the stop waiting behind it then tears down what is left).
    private func rebuildCapture(why: String, done: String, failed: (Error) -> String) {
        logLine("rebuilding the capture (\(why))")
        let step: (String) -> Void = { [weak self] call in self?.setDeviceState("rebuilding (\(why)): in \(call)") }
        step("AudioObjectRemovePropertyListenerBlock")
        unwatchOutputDevice()
        tearDownAggregate(step: step)
        step("AudioHardwareDestroyProcessTap")
        tearDownTap()
        guard !isStopped else {                                        // Stop came while the old capture was torn down
            setDeviceState("no capture device (stopped during a rebuild)")
            return
        }
        do {
            step("AudioHardwareCreateProcessTap")
            try createTap()
            step("AudioHardwareCreateAggregateDevice / AudioDeviceStart")
            try buildAggregate()                                       // "capture device started" once it is
            watchOutputDevice()
            notice(done)
        } catch {
            running = false
            setDeviceState("capture device NOT running (it could not be built again)")
            failure(failed(error))
        }
    }

    private func rebuildOnControl(reason: String) {
        guard running, !isStopped else { return }
        rebuildCapture(why: reason, done: "The capture of the computer's sound was built again (\(reason)).",
                       failed: { "The capture of the computer's sound could not be built again (\(reason); \($0.localizedDescription)); your microphone goes on being recorded." })
    }

    /// The tap on every process but this one; with a chosen output device, only what is played to that device
    /// (`deviceUID`), so the sound of another output (a second screen's speakers) stays off the track.
    private func createTap() throws {
        let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: SystemAudioTap.ownProcessObjects())
        desc.name = "Meeting Transcriber"
        desc.isPrivate = true
        desc.muteBehavior = CATapMuteBehavior.unmuted
        if let chosen = chosenOutput { desc.deviceUID = chosen.uid }
        var tap = AudioObjectID(kAudioObjectUnknown)
        let created = AudioHardwareCreateProcessTap(desc, &tap)
        guard created == noErr else { throw TapError.status("creating the audio tap", created) }
        description = desc
        tapID = tap
    }

    // MARK: - The aggregate device around the tap

    private func buildAggregate() throws {
        guard let desc = description else { return }
        // The clock: the output the sound goes to (see the class comment): the chosen output, else the Mac's current one.
        guard let outputUID = chosenOutput?.uid ?? SystemAudioTap.defaultOutputDeviceUID() else { throw TapError.noOutputDevice }
        clockUID = outputUID
        clockIsDefaultOutput = chosenOutput == nil
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        guard AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &asbd) == noErr,
              let tapFormat = AVAudioFormat(streamDescription: &asbd) else { throw TapError.format }
        tapRate = tapFormat.sampleRate
        format = tapFormat
        // The output device must be the main sub-device (the clock); a tap as the main sub-device with no sub-device gives
        // zero samples. The tap's UID is the description's uuid, not the tap object's.
        let properties: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Meeting Transcriber system sound",
            kAudioAggregateDeviceUIDKey: "com.github.avza-bl.meeting-transcriber.tap." + desc.uuid.uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: desc.uuid.uuidString,
                                               kAudioSubTapDriftCompensationKey: true]],
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
        ]
        var aggregate = AudioObjectID(kAudioObjectUnknown)
        let made = AudioHardwareCreateAggregateDevice(properties as CFDictionary, &aggregate)
        guard made == noErr else { throw TapError.status("creating the capture device", made) }
        aggregateID = aggregate
        // The buffers come at the aggregate's rate (its clock's: 16 kHz for a Bluetooth headset in the hands-free profile,
        // measured), whatever rate the tap's format property states; the format handed on says the rate the data has.
        if let rate = SystemAudioTap.nominalSampleRate(of: aggregate), rate > 0, rate != tapFormat.sampleRate,
           let delivered = AVAudioFormat(commonFormat: tapFormat.commonFormat == .otherFormat ? .pcmFormatFloat32 : tapFormat.commonFormat,
                                         sampleRate: rate, channels: tapFormat.channelCount, interleaved: tapFormat.isInterleaved) {
            format = delivered
        }
        guard let deliveredFormat = format else { throw TapError.format }
        let delivery = Delivery(format: deliveredFormat)
        var proc: AudioDeviceIOProcID?
        let registered = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregate, io) { [weak self] _, input, _, _, _ in
            self?.deliver(input, through: delivery)
        }
        guard registered == noErr, let ioProc = proc else {
            delivery.end()
            tearDownAggregate()
            throw TapError.status("attaching to the capture device", registered)
        }
        procID = ioProc
        self.delivery = delivery
        let started = AudioDeviceStart(aggregate, ioProc)
        guard started == noErr else {
            tearDownAggregate()
            throw TapError.status("starting the capture", started)
        }
        setDeviceState("capture device started")
        // noErr from AudioDeviceStart does not mean the device runs (see the class comment): a look a moment later tells,
        // and says so in the log and the notes instead of leaving a silent track to explain itself after 10 s.
        let clockName = AudioDevices.id(forUID: outputUID).flatMap { AudioDevices.name(of: $0) } ?? outputUID
        let chosen = chosenOutput != nil
        control.asyncAfter(deadline: .now() + 1.5) { [weak self] in
            guard let self = self, self.running, !self.isStopped, self.aggregateID == aggregate else { return }
            let seen = delivery.callbacks
            if self.aggregateIsRunning {
                self.setDeviceState("capture device running")
                self.diagnostic("capture device \(aggregate) running on \(clockName), \(seen) callbacks in its first 1.5 s")
            } else if chosen {
                // Measured to work: the Mac's current output. Rather than a silent track and advice for next time, the
                // capture is built again on it now (the chosen device stays the clock only while it runs).
                self.setDeviceState("capture device NOT running")
                self.notice("The capture of the computer's sound did not start on “\(clockName)”, the output chosen under “Take the computer's sound from” (\(seen) callbacks in 1.5 s); the Mac's current output is used instead.")
                self.chosenOutput = nil
                self.rebuildOnControl(reason: "the chosen output did not run")
            } else {
                // A device that starts late is not told as a failure: in a 1.10.1 call the capture ran from about 3 s on,
                // after the microphone's engine had restarted. A second look 6 s after the start decides.
                self.setDeviceState("capture device NOT running")
                self.logLine("capture device \(aggregate) not running on \(clockName) after 1.5 s (\(seen) callbacks); looked at again at 6 s")
                self.control.asyncAfter(deadline: .now() + 4.5) { [weak self] in
                    guard let self = self, self.running, !self.isStopped, self.aggregateID == aggregate else { return }
                    let later = delivery.callbacks
                    if self.aggregateIsRunning || later > seen {
                        self.setDeviceState("capture device running")
                        self.diagnostic("capture device \(aggregate) running on \(clockName) after 6 s, \(later) callbacks")
                    } else {
                        self.notice("The capture of the computer's sound did not start on the Mac's current output “\(clockName)” (\(later) callbacks in 6 s); the computer's-sound track may stay silent. capture.log in the recording's folder has the details.")
                    }
                }
            }
        }
    }

    /// The aggregate's nominal sample rate, the rate its IOProc buffers have.
    private static func nominalSampleRate(of device: AudioObjectID) -> Double? {
        var address = rateAddress
        var rate: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &rate) == noErr else { return nil }
        return rate
    }

    /// `step` is told each Core Audio call before it is made (for the report, while a rebuild or a stop runs).
    private func tearDownAggregate(step: (String) -> Void = { _ in }) {
        delivery?.end()                                                // from now on a late callback is dropped
        delivery = nil
        if aggregateID != kAudioObjectUnknown {
            if let proc = procID {
                step("AudioDeviceStop")
                _ = AudioDeviceStop(aggregateID, proc)
                step("AudioDeviceDestroyIOProcID")
                _ = AudioDeviceDestroyIOProcID(aggregateID, proc)
            }
            step("AudioHardwareDestroyAggregateDevice")
            _ = AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        procID = nil
        aggregateID = AudioObjectID(kAudioObjectUnknown)
    }

    private func tearDownTap() {
        if tapID != kAudioObjectUnknown { _ = AudioHardwareDestroyProcessTap(tapID) }
        tapID = AudioObjectID(kAudioObjectUnknown)
        description = nil
    }

    /// On the delivery queue: the aggregate's input holds the tap's streams and, when the output device is a duplex one
    /// (a USB headset, an audio interface, a Bluetooth device in its hands-free profile), that device's own input
    /// streams too: those must not land on the other side's track. The tap's streams come after the sub-device's, so they
    /// are the last buffers whose channels add up to the tap's channel count: one interleaved buffer, or one buffer per channel.
    private func deliver(_ input: UnsafePointer<AudioBufferList>, through delivery: Delivery) {
        guard !isStopped, delivery.take() else { return }
        let tapFormat = delivery.format
        let all = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        stats.lock()
        callbacks += 1
        if firstLayout.isEmpty {
            firstLayout = "\(all.count) buffer\(all.count == 1 ? "" : "s"): "
                + all.map { "\($0.mNumberChannels)ch×\($0.mDataByteSize)B" }.joined(separator: ", ")
        }
        stats.unlock()
        let wanted = Int(tapFormat.channelCount)
        guard wanted > 0, !all.isEmpty else {
            stats.lock(); emptyCallbacks += 1; stats.unlock()
            return
        }
        var chosen: [AudioBuffer] = []
        var channels = 0
        for index in stride(from: all.count - 1, through: 0, by: -1) {
            let buffer = all[index]
            guard buffer.mDataByteSize > 0 else { continue }
            chosen.insert(buffer, at: 0)
            channels += Int(buffer.mNumberChannels)
            if channels >= wanted { break }
        }
        guard channels == wanted, let first = chosen.first else {           // not the tap's layout: nothing is written
            stats.lock(); layoutMismatches += 1; stats.unlock()
            return
        }
        stats.lock(); delivered += 1; stats.unlock()
        let interleaved = chosen.count == 1 && first.mNumberChannels > 1
        guard let layout = AVAudioFormat(commonFormat: tapFormat.commonFormat == .otherFormat ? .pcmFormatFloat32 : tapFormat.commonFormat,
                                         sampleRate: tapFormat.sampleRate, channels: AVAudioChannelCount(wanted), interleaved: interleaved) else { return }
        let bytesPerFrame = Int(layout.streamDescription.pointee.mBytesPerFrame)
        guard bytesPerFrame > 0 else { return }
        let own = AudioBufferList.allocate(maximumBuffers: chosen.count)   // the tap's buffers only (the sound itself is not copied)
        defer { own.unsafeMutablePointer.deallocate() }
        for (i, buffer) in chosen.enumerated() { own[i] = buffer }
        guard let pcm = AVAudioPCMBuffer(pcmFormat: layout, bufferListNoCopy: own.unsafePointer, deallocator: nil) else { return }
        pcm.frameLength = AVAudioFrameCount(Int(first.mDataByteSize) / bytesPerFrame)
        if pcm.frameLength > 0 { onBuffer?(pcm) }
    }

    // MARK: - The output device (headphones plugged in or out, the chosen device disconnected)

    /// On the control queue; the listeners are called there too.
    private func watchOutputDevice() {
        let generation = watchGeneration
        // The clock device's sample rate: a change (a Bluetooth profile switch, another app renegotiating the rate) is
        // the kind of event after which a tap keeps running but delivers zeros, so the capture is built again.
        if let uid = clockUID, let clock = AudioDevices.id(forUID: uid) {
            var rate = SystemAudioTap.rateAddress
            let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
                guard let self = self, self.watchGeneration == generation else { return }
                self.rebuildOnControl(reason: "the sound output's sample rate changed")
            }
            if AudioObjectAddPropertyListenerBlock(clock, &rate, control, block) == noErr { rateListener = (clock, block) }
        }
        if let chosen = chosenOutput {
            guard let device = AudioDevices.id(forUID: chosen.uid) else { return }
            var address = SystemAudioTap.aliveAddress
            let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
                guard let self = self, self.watchGeneration == generation else { return }
                self.chosenOutputChanged(device)
            }
            if AudioObjectAddPropertyListenerBlock(device, &address, control, block) == noErr { aliveListener = (device, block) }
            return
        }
        guard clockIsDefaultOutput else { return }       // a chosen output is kept until it goes away (above)
        var address = SystemAudioTap.defaultOutputAddress
        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self = self, self.watchGeneration == generation else { return }
            self.outputDeviceChanged()
        }
        if AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, block) == noErr {
            outputListener = block
        }
    }

    private func unwatchOutputDevice() {
        watchGeneration += 1
        if let block = outputListener {
            var address = SystemAudioTap.defaultOutputAddress
            _ = AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, block)
            outputListener = nil
        }
        if let listener = aliveListener {
            var address = SystemAudioTap.aliveAddress
            _ = AudioObjectRemovePropertyListenerBlock(listener.device, &address, control, listener.block)
            aliveListener = nil
        }
        if let listener = rateListener {
            var address = SystemAudioTap.rateAddress
            _ = AudioObjectRemovePropertyListenerBlock(listener.device, &address, control, listener.block)
            rateListener = nil
        }
    }

    /// On the control queue: the aggregate device is clocked by the output device it was made with, so the capture is built
    /// again on the new one, tap and aggregate both, and the listeners move with it (the rate listener sat on the old clock).
    private func outputDeviceChanged() {
        guard running, !isStopped else { return }
        rebuildCapture(why: "the sound output changed", done: "The sound output changed; the computer's sound goes on from the new one.",
                       failed: { "The sound output changed and the computer's sound could not be captured from the new one (\($0.localizedDescription)); your microphone goes on being recorded." })
    }

    /// On the control queue: the chosen output device went away (unplugged, switched off). The tap was tied to it, so a
    /// new tap on whatever is the Mac's current output takes over, and from then on follows the output like an unchosen one.
    private func chosenOutputChanged(_ device: AudioObjectID) {
        guard running, !isStopped, let chosen = chosenOutput else { return }
        var address = SystemAudioTap.aliveAddress
        var alive: UInt32 = 1
        var size = UInt32(MemoryLayout<UInt32>.size)
        if AudioObjectGetPropertyData(device, &address, 0, nil, &size, &alive) == noErr, alive != 0 { return }
        chosenOutput = nil
        rebuildCapture(why: "the chosen output was disconnected", done: "The sound output “\(chosen.name)” was disconnected; the computer's sound goes on from the Mac's current output.",
                       failed: { "The sound output “\(chosen.name)” was disconnected and the computer's sound could not be captured from the current output (\($0.localizedDescription)); your microphone goes on being recorded." })
    }

    // MARK: - Core Audio lookups

    private static let defaultOutputAddress = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                                                         mScope: kAudioObjectPropertyScopeGlobal,
                                                                         mElement: kAudioObjectPropertyElementMain)
    private static let aliveAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDeviceIsAlive,
                                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                                 mElement: kAudioObjectPropertyElementMain)
    private static let rateAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyNominalSampleRate,
                                                                mScope: kAudioObjectPropertyScopeGlobal,
                                                                mElement: kAudioObjectPropertyElementMain)

    static func defaultOutputDeviceUID() -> String? {
        return AudioDevices.defaultUID(.output)
    }

    /// The audio process object of this app (its own sounds must not land on the other side's track), from the system
    /// object's PID-to-process translation. An empty list means "exclude nothing": the tap still works, and a
    /// notification sound of this app would be captured.
    static func ownProcessObjects() -> [AudioObjectID] {
        var pid = ProcessInfo.processInfo.processIdentifier
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var object = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = withUnsafePointer(to: &pid) { pidPointer in
            AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                       UInt32(MemoryLayout<pid_t>.size), pidPointer, &size, &object)
        }
        guard status == noErr, object != kAudioObjectUnknown else { return [] }
        return [object]
    }
}
