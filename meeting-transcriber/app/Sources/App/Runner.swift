import Foundation
import AppKit
import Combine

/// Runs mt.py and the one-time engine setup, and turns their output into RunState. Two things can run at once: a
/// recording (its own slot: a record-only mt.py process, or the in-app call capture with its live follower) and one
/// processing job from the queue (a transcription, a translation, the setup). A stopped recording queues its transcription.
final class Runner: ObservableObject {
    static let shared = Runner()

    enum EngineStatus: Equatable {
        case needsSetup
        case settingUp
        case ready
        case failed(String)
    }

    private struct Step {
        let executable: String
        let arguments: [String]
        let environment: [String: String]
        let label: String
        var optional = false          // a failure is reported and skipped (the Apple-GPU engine on a Mac that cannot take it)
    }

    /// A job waiting in the queue, with what it runs and the state it starts from (a call's live lines and notes).
    private struct Work {
        let job: QueuedJob
        let steps: [Step]
        let seed: RunState?
        let targets: [String]          // the languages the job translates into: the only ones its pane offers
    }

    /// The running processing job (a transcription, a translation, the setup).
    @Published private(set) var state: RunState? = nil
    /// The recording in progress, or the one that just ended.
    @Published private(set) var recordingState: RunState? = nil
    /// The job whose result the alert shows.
    @Published private(set) var lastFinished: RunState? = nil
    @Published private(set) var queue = JobQueue()
    @Published private(set) var engineStatus: EngineStatus = .needsSetup
    @Published private(set) var ffmpegMissing = false
    @Published private(set) var recordings: [Recordings.Entry] = []     // what the data folder holds
    @Published private(set) var hasAPIKey = false
    @Published private(set) var hasHFToken = false        // the Hugging Face token the optional pyannote backend needs once
    @Published private(set) var lastFolder: String? = nil // the data folder of the last finished job (its speakers can be named)
    @Published private(set) var keySource = ""
    @Published private(set) var recordingSince: Date? = nil
    @Published private(set) var recordingPaused = false
    /// Whether this build keeps its permissions across updates (its code signature names a certificate) or not (ad-hoc).
    let signing: SigningIdentity.Kind = SigningIdentity.kind(ofRequirement: CodeSignature.designatedRequirement())
    /// The window's settings as they are now: a recording's transcription takes from them what only matters afterwards.
    var currentSettings: (() -> JobSettings)?
    private var pauseClock = PauseClock()
    @Published private(set) var startupError: String? = nil
    @Published var showOutcomeAlert = false

    let engine = Engine(home: FileManager.default.homeDirectoryForCurrentUser)

    private let lock = NSLock()
    private var current: Process? = nil           // the processing job's process
    private var recordProcess: Process? = nil     // the record-only mt.py process
    private var works: [UUID: Work] = [:]
    private var activity: NSObjectProtocol?
    private var sleepObservers: [NSObjectProtocol] = []
    private var sleptAt: Date?
    private var stopRequested = false             // the processing job
    private var recordingCancelled = false        // the recording
    private var stopSent = false
    private var recordingSettings: JobSettings?   // the settings a recording started with (its transcription uses them)
    /// The data folder of the recording in progress (a call's from the start; mt.py names its own with @@FOLDER, which
    /// comes right after its signal handlers are in place: a Pause or Extend sent before that line would be lost).
    @Published private(set) var recordingFolder: String?
    // A call recorded here (microphone + the sound this Mac plays), before mt.py transcribes it.
    private var recorder: CallRecorder?
    private var callTracks: CallRecorder.Tracks?
    private var callScript: URL?
    private var callTimer: Timer?
    private var callFolder: URL?
    private var follower: Process?                // mt.py follow: transcribes the call's files while they are written
    private var callLimit: TimeInterval = 0       // recorded seconds after which the call recording stops by itself
    private var callNoticed = false               // the 10-minute notice was given for the current limit
    private static let noticeBefore: TimeInterval = 600
    private var callNote: String?          // something to tell about the call (a capture that stopped), shown with its result
    @Published private var recordingHeld = false      // the "Recording" automatic-termination hold, taken and released exactly once

    /// The recording slot is taken from Start until the recording's transcription is queued (or its end reported): the
    /// follower may still be finishing after the capture stopped, and a second recording must not touch its fields meanwhile.
    var isRecording: Bool { return recordingHeld }
    var isProcessing: Bool { return queue.current != nil }
    var isRunning: Bool { return isRecording || !queue.isIdle }

    // MARK: - Bundle

    static var appVersion: String {
        return (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0.0.0"
    }

    /// Contents/Resources/engine: mt.py, VERSION and requirements.txt.
    static var engineFolder: URL? {
        return Bundle.main.url(forResource: "engine", withExtension: nil)
    }

    static func bundleIsComplete() -> Bool {
        guard let folder = engineFolder else { return false }
        return ["mt.py", "VERSION", "requirements.txt", "requirements-mlx.txt", "requirements-pyannote.txt"].allSatisfy {
            FileManager.default.fileExists(atPath: folder.appendingPathComponent($0).path)
        }
    }

    // MARK: - Status

    func refresh() {
        switch engineStatus {
        case .settingUp, .failed:
            break                      // keep the message until the user tries again
        default:
            let marker = engine.marker
            let ready = engine.isReady(appVersion: Runner.appVersion, readMarker: { try? String(contentsOf: marker, encoding: .utf8) })
            engineStatus = ready ? .ready : .needsSetup
        }
        // ffmpeg: the engine fetches it at setup when the system has none, so only a failed fetch (reported by `prepare`) matters.
        recordings = Recordings.scan(data: engine.data)
        hasHFToken = KeyStore.load(KeyStore.hfAccount) != nil
        if let saved = KeyStore.load() {
            hasAPIKey = true
            keySource = saved.place == .keychain ? "the Keychain" : "a private file"
        } else if !(ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"] ?? "").isEmpty {
            hasAPIKey = true
            keySource = "the environment"
        } else {
            hasAPIKey = false
            keySource = ""
        }
    }

    /// Saves (or, for an empty key, removes) the key. The outcome says where it went or why it could not be saved.
    @discardableResult
    func saveAPIKey(_ key: String) -> KeyStore.Outcome {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        var outcome = KeyStore.Outcome.saved(.keychain, note: nil)
        if trimmed.isEmpty { KeyStore.delete() } else { outcome = KeyStore.save(trimmed) }
        refresh()
        return outcome
    }

    private var apiKey: String? {
        return KeyStore.load()?.key ?? ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]
    }

    /// Saves (or, for an empty token, removes) the Hugging Face token the pyannote backend needs for its one-time download.
    @discardableResult
    func saveHFToken(_ token: String) -> KeyStore.Outcome {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        var outcome = KeyStore.Outcome.saved(.keychain, note: nil)
        if trimmed.isEmpty { KeyStore.delete(KeyStore.hfAccount) } else { outcome = KeyStore.save(trimmed, KeyStore.hfAccount) }
        refresh()
        return outcome
    }

    /// Names the speakers of a stored meeting (Speaker 1 → Anna): a queued job that rewrites its files.
    func renameSpeakers(folder: String, names: [String: String]) {
        guard engineStatus == .ready, let engineFolder = Runner.engineFolder else { return }
        let arguments = JobSettings.renameArguments(scriptPath: engineFolder.appendingPathComponent("mt.py").path, folder: folder, names: names)
        guard arguments.count > 3 else { return }                  // nothing to rename
        let step = Step(executable: engine.python.path, arguments: arguments,
                        environment: JobSettings().environment(home: engine.data.path, apiKey: nil), label: "")
        enqueue([step], kind: .translate, label: "Naming the speakers of " + URL(fileURLWithPath: folder).lastPathComponent, folder: folder)
    }

    /// The environment of a job. A processing job started beside a recording, and a recording started beside a processing
    /// job, leave half the cores to the other (the live transcription and the final pass share the CPU; `yield` says so).
    private func environment(_ settings: JobSettings, apiKey key: String?, yield: Bool) -> [String: String] {
        var env = settings.environment(home: engine.data.path, apiKey: key, hfToken: KeyStore.load(KeyStore.hfAccount)?.key)
        let cores = ProcessInfo.processInfo.activeProcessorCount
        if yield && cores > 1 { env["MT_THREADS"] = String(max(1, cores / 2)) }
        if let out = env["MT_OUT"] {                 // the default output folder is made on first use (a chosen one must exist)
            try? FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
        }
        return env
    }

    /// The data folders something is being done with right now: the recording's and those of the queued jobs.
    var busyFolders: Set<String> {
        var set = queue.folders
        if isRecording, let f = recordingFolder { set.insert(f) }
        return set
    }

    // MARK: - Jobs

    /// The window's Start button: a recording starts in its own slot; a transcription or translation joins the queue.
    func start(_ settings: JobSettings) {
        guard engineStatus == .ready, let folder = Runner.engineFolder else {
            startupError = "The speech engine is not set up yet."
            return
        }
        startupError = nil
        let script = folder.appendingPathComponent("mt.py")
        switch settings.mode {
        case .record:
            guard !isRecording else { return }
            if settings.captureSystem { startCall(settings, script: script) } else { startRecordingProcess(settings, script: script) }
        case .files, .translate:
            let step = Step(executable: engine.python.path, arguments: settings.arguments(scriptPath: script.path),
                            environment: environment(settings, apiKey: apiKey, yield: isRecording), label: "")
            let what = settings.mode == .files ? "Transcribing \(settings.files.count) file\(settings.files.count == 1 ? "" : "s")"
                                                : "Translating \(settings.files.count) file\(settings.files.count == 1 ? "" : "s")"
            enqueue([step], kind: settings.mode == .files ? .transcribe : .translate, label: what, targets: settings.targets)
        }
    }

    /// Transcribes a recording left in the data folder (continuing its checkpoint, if any).
    func transcribeRecording(_ entry: Recordings.Entry, settings: JobSettings) {
        guard engineStatus == .ready, let folder = Runner.engineFolder else { return }
        let step = Step(executable: engine.python.path,
                        arguments: settings.transcribeRecordingArguments(scriptPath: folder.appendingPathComponent("mt.py").path, entry: entry),
                        environment: environment(settings, apiKey: apiKey, yield: isRecording), label: "")
        enqueue([step], kind: .transcribe, label: "Transcribing " + entry.label, folder: entry.folder, targets: settings.targets)
    }

    /// Translates a stored meeting into the languages chosen in the window.
    func translateStored(_ entry: Recordings.Entry, settings: JobSettings) {
        guard engineStatus == .ready, let folder = Runner.engineFolder, !settings.targets.isEmpty else { return }
        let step = Step(executable: engine.python.path,
                        arguments: settings.translateStoredArguments(scriptPath: folder.appendingPathComponent("mt.py").path, entry: entry),
                        environment: environment(settings, apiKey: apiKey, yield: isRecording), label: "")
        enqueue([step], kind: .translate, label: "Translating " + entry.label, folder: entry.folder, targets: settings.targets)
    }

    /// Takes a waiting job out of the queue.
    func removeQueued(_ id: UUID) {
        if queue.remove(id: id) { works[id] = nil }
    }

    func moveQueued(_ id: UUID, up: Bool) {
        queue.move(id: id, up: up)
    }

    // MARK: - The recording slot (record-only mt.py process)

    private func startRecordingProcess(_ settings: JobSettings, script: URL) {
        recordingSettings = settings
        recordingCancelled = false
        stopSent = false
        recordingState = RunState(kind: .record)
        recordingState?.targets = settings.translatesLive ? settings.targets : []
        recordingSince = Date()
        pauseClock = PauseClock()
        recordingPaused = false
        holdRecording()
        updateActivity()
        recordingFolder = nil                             // mt.py names its folder (@@FOLDER) as soon as it exists
        if !settings.microphone.isDefault && settings.microphoneName.isEmpty {      // chosen, but not connected now
            addNote(settings.microphone.notConnectedNote(what: "microphone", instead: "the Mac's current sound input"))
        }
        // The key goes along for the live translation with Claude (the environment adds it only when that is asked for).
        let step = Step(executable: engine.python.path, arguments: settings.recordingArguments(scriptPath: script.path),
                        environment: environment(settings, apiKey: settings.translatesLive ? apiKey : nil, yield: isProcessing), label: "")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let status = self.run(step, slot: .recording)
            DispatchQueue.main.async { self.finishRecordingProcess(status: status) }
        }
    }

    private func finishRecordingProcess(status: Int32) {
        defer { releaseRecording(); updateActivity() }   // the slot is free only once the transcription is queued
        recordingSince = nil
        recordingPaused = false
        guard var s = recordingState else { updateActivity(); return }
        s.finish(exitStatus: status, stopRequested: recordingCancelled)
        recordingState = s
        recordings = Recordings.scan(data: engine.data)
        let folder = s.outputs.first
        let settings = recordingSettings?.atStop(now: currentSettings?())   // languages, who spoke, translation … as chosen now
        // The transcription is a queued job on the folder mt.py named: it continues what was transcribed live.
        if status == 0, !recordingCancelled, let folder = folder, let settings = settings, !settings.transcribeLater,
           let engineFolder = Runner.engineFolder, let entry = recordings.first(where: { $0.folder == folder }) {
            var arguments = settings.transcribeRecordingArguments(scriptPath: engineFolder.appendingPathComponent("mt.py").path, entry: entry)
            for note in s.notes { arguments += ["--note", note] }        // what happened goes at the top of the transcript
            let step = Step(executable: engine.python.path, arguments: arguments,
                            environment: environment(settings, apiKey: apiKey, yield: false), label: "")
            var seed = RunState(kind: .transcribe)
            seed.lines = s.lines                      // the live lines stay on screen; the final pass refines them in place
            seed.notes = s.notes
            if queue.isIdle { recordingState = nil }  // the recording panel gives way to the job (pump() does it when the job waited)
            enqueue([step], kind: .transcribe, label: "Transcribing " + entry.label, seed: seed, folder: entry.folder,
                    targets: settings.targets)
        } else {
            if status == 0, !recordingCancelled, folder != nil, settings?.transcribeLater == false {
                s.apply(.note("The recording could not be found in the data folder, so it was not transcribed."))
                recordingState = s
            }
            lastFinished = s
            NSSound(named: NSSound.Name(status == 0 ? "Glass" : "Basso"))?.play()
            showOutcomeAlert = true
        }
    }

    /// Records stop with Ctrl-C semantics: mt.py saves the recording; its transcription is then queued.
    func stopRecording() {
        if let rec = recorder {
            // Still starting (a permission prompt is open): there is nothing to transcribe yet, so stopping is cancelling.
            if let tracks = callTracks { finishCall(rec, tracks) } else { cancelCall(rec) }
            return
        }
        lock.lock(); let p = recordProcess; lock.unlock()
        guard let p = p, !stopSent else { return }
        stopSent = true
        kill(p.processIdentifier, SIGINT)
    }

    /// Cancels the recording: the audio stays in the data folder, nothing is transcribed.
    func cancelRecording() {
        if let rec = recorder {
            cancelCall(rec)
            return
        }
        lock.lock(); let p = recordProcess; lock.unlock()
        guard let p = p else { return }
        recordingCancelled = true
        kill(p.processIdentifier, SIGTERM)
        terminateLater(p)
    }

    /// Cancels the processing job that runs now; the waiting ones stay queued.
    func cancelProcessing() {
        lock.lock(); let p = current; lock.unlock()
        guard let p = p else { return }
        stopRequested = true
        kill(p.processIdentifier, SIGTERM)
        terminateLater(p)
    }

    /// Everything: the recording, the running job and the queue (Quit).
    func cancel() {
        for id in queue.pending.map({ $0.id }) { works[id] = nil }
        queue.removeAll()
        cancelRecording()
        cancelProcessing()
    }

    private func terminateLater(_ p: Process) {
        let pid = p.processIdentifier
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            if p.isRunning { ProcessTree.terminate(rootPID: pid) }
        }
    }

    // MARK: - Awake while anything runs

    /// While a job runs the Mac must not go to sleep by itself (a closed lid still sleeps it; that is reported on wake).
    private func updateActivity() {
        if isRunning { holdActivity() } else { releaseActivity() }
    }

    private func holdActivity() {
        guard activity == nil else { return }
        activity = ProcessInfo.processInfo.beginActivity(options: [.idleSystemSleepDisabled, .userInitiated],
                                                         reason: "Recording or transcribing a meeting")
        let center = NSWorkspace.shared.notificationCenter
        sleepObservers = [
            center.addObserver(forName: NSWorkspace.willSleepNotification, object: nil, queue: .main) { [weak self] _ in
                self?.sleptAt = Date()
            },
            center.addObserver(forName: NSWorkspace.didWakeNotification, object: nil, queue: .main) { [weak self] _ in
                guard let self = self, let since = self.sleptAt else { return }
                self.sleptAt = nil
                let f = DateFormatter(); f.dateFormat = "HH:mm:ss"
                self.addNote("The Mac slept from \(f.string(from: since)) to \(f.string(from: Date())); nothing was recorded or processed meanwhile.")
            },
        ]
    }

    private func releaseActivity() {
        if let activity = activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
        for observer in sleepObservers { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
        sleepObservers = []
        sleptAt = nil
    }

    private func holdRecording() {
        guard !recordingHeld else { return }
        recordingHeld = true
        ProcessInfo.processInfo.disableAutomaticTermination("Recording")
    }

    private func releaseRecording() {
        guard recordingHeld else { return }
        recordingHeld = false
        ProcessInfo.processInfo.enableAutomaticTermination("Recording")
    }

    /// A note goes to the recording when one runs, else to the processing job.
    private func addNote(_ text: String) {
        if isRecording, var s = recordingState { s.apply(.note(text)); recordingState = s }
        else if var s = state { s.apply(.note(text)); state = s }
    }

    // MARK: - Calls: the microphone and the sound this Mac plays

    private func startCall(_ settings: JobSettings, script: URL) {
        recordingSettings = settings
        recordingCancelled = false
        stopSent = false
        callNote = nil
        recordingState = RunState(kind: .record)
        recordingState?.targets = settings.translatesLive ? settings.targets : []
        recordingSince = Date()
        pauseClock = PauseClock()
        recordingPaused = false
        holdRecording()
        // Named and numbered as mt.py names its own data folders: another job of this name begun in the same second (a
        // file named like the call, transcribed from the queue) keeps its folder, and the call takes -2 (DataFolder).
        let folder: URL
        do {
            folder = try DataFolder.make(in: engine.data, name: settings.recordName)
        } catch {
            callNotStarted("The call's folder could not be made in \(engine.data.path): \(error.localizedDescription)")
            return
        }
        let rec = CallRecorder()
        recorder = rec
        callScript = script
        callFolder = folder
        recordingFolder = folder.path
        callLimit = Double(settings.recordHours) * 3600
        callNoticed = false
        updateActivity()
        rec.onFailure = { [weak self, weak rec] message in
            guard let self = self, let rec = rec, self.recorder === rec else { return }     // not for a recorder that is over
            self.callStoppedByItself(rec, message)
        }
        rec.onNotice = { [weak self, weak rec] message in
            guard let self = self, let rec = rec, self.recorder === rec else { return }
            self.addNote(message)                      // kept for the result and the transcript, not only the log
        }
        rec.onLevel = { [weak self, weak rec] level in
            guard let self = self, let rec = rec, self.recorder === rec else { return }
            if var s = self.recordingState { s.apply(.level(level)); self.recordingState = s }
        }
        let microphone = settings.microphone.isDefault ? nil : (uid: settings.microphone.uid, name: settings.microphone.label)
        let output = settings.output.isDefault ? nil : (uid: settings.output.uid, name: settings.output.label)
        rec.start(in: folder, microphone: microphone, output: output) { [weak self] result in
            guard let self = self, self.recorder === rec else { return }                  // cancelled meanwhile: ignore
            switch result {
            case .success(let tracks):
                self.callTracks = tracks
                self.recordingSince = Date()          // the panel's timer counts from here, not from the permission prompts
                self.pauseClock = PauseClock()
                self.callTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self, weak rec] _ in
                    guard let self = self, let rec = rec else { return }
                    self.checkCallLimit(rec)
                }
                if settings.live && !settings.transcribeLater { self.startFollower(settings, script: script, tracks: tracks, folder: folder) }
            case .failure(let error):
                self.callFailed(rec, error.localizedDescription)
            }
        }
    }

    private static func limitText(_ seconds: TimeInterval) -> String {
        let hours = seconds / 3600
        return hours == hours.rounded() ? "\(Int(hours)) h" : String(format: "%.1f h", hours)
    }

    /// Once a second while a call is recorded: the 10-minute notice, then the stop at the limit (recorded time, not wall time).
    private func checkCallLimit(_ rec: CallRecorder) {
        guard recorder === rec, callTracks != nil, !stopSent else { return }
        let recorded = rec.recordedSeconds
        if !callNoticed && recorded >= callLimit - Runner.noticeBefore {
            callNoticed = true
            addNote("The recording stops in 10 minutes (\(Runner.limitText(callLimit)) limit)")
        }
        if recorded >= callLimit {
            addNote("Recording stopped at the \(Runner.limitText(callLimit)) limit")
            stopRecording()
        }
    }

    /// Whether the recording can take a Pause or an Extend now: a call recording once its tracks are open, a plain
    /// recording once mt.py has named its folder (before that its handlers are not in place and the signal would be lost,
    /// while the window would show the state it never reached).
    var recordingReady: Bool {
        guard isRecording, !stopSent, recordingFolder != nil else { return false }
        return recorder == nil || callTracks != nil
    }

    /// "Pause": nothing is recorded until "Resume"; the recording and its transcript leave the pause out, and a timed note
    /// says where it was and how long. A call recording pauses in the app; a plain recording tells mt.py (SIGUSR2).
    func pauseRecording() {
        guard recordingReady, !recordingPaused else { return }
        if let rec = recorder {
            rec.pause()
        } else {
            lock.lock(); let p = recordProcess; lock.unlock()
            guard let p = p else { return }
            kill(p.processIdentifier, SIGUSR2)
        }
        pauseClock.pause()
        recordingPaused = true
    }

    func resumeRecording() {
        guard recordingPaused else { return }
        if let rec = recorder {
            rec.resume()
        } else {
            lock.lock(); let p = recordProcess; lock.unlock()
            if let p = p { kill(p.processIdentifier, SIGUSR2) }
        }
        pauseClock.resume()
        recordingPaused = false
    }

    /// Seconds recorded so far by the window's clock (wall time without the pauses), for the timer next to the level bar.
    func recordedTime(at now: Date) -> TimeInterval? {
        return recordingSince.map { pauseClock.recorded(since: $0, at: now) }
    }

    /// "Extend by 1 h": a call recording moves its own limit; a plain recording tells mt.py (SIGUSR1).
    func extendRecording() {
        guard recordingReady else { return }
        if recorder != nil {
            callLimit += 3600
            callNoticed = false
            addNote("Recording extended: it now stops after \(Runner.limitText(callLimit))")
            return
        }
        lock.lock(); let p = recordProcess; lock.unlock()
        guard let p = p else { return }
        kill(p.processIdentifier, SIGUSR1)
    }

    /// Starts `mt.py follow` on the two files: every closed stretch of speech is transcribed while the call goes on.
    private func startFollower(_ settings: JobSettings, script: URL, tracks: CallRecorder.Tracks, folder: URL) {
        let step = Step(executable: engine.python.path,
                        arguments: settings.followArguments(scriptPath: script.path, mic: tracks.mic.path, system: tracks.system.path, into: folder.path),
                        environment: environment(settings, apiKey: settings.translatesLive ? apiKey : nil, yield: isProcessing), label: "")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            _ = self?.run(step, slot: .follower)
        }
    }

    /// Tells the follower to finish its current chunk and waits for it (at most `timeout` s), then `completion` on main.
    private func stopFollower(timeout: TimeInterval = 120, completion: @escaping () -> Void) {
        lock.lock(); let f = follower; lock.unlock()
        guard let f = f else { completion(); return }
        kill(f.processIdentifier, SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            while f.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.2) }
            if f.isRunning { ProcessTree.terminate(rootPID: f.processIdentifier) }
            self?.lock.lock(); self?.follower = nil; self?.lock.unlock()
            DispatchQueue.main.async(execute: completion)
        }
    }

    /// The recording is over: close the two files, then queue `mt.py call` on them.
    private func finishCall(_ rec: CallRecorder, _ tracks: CallRecorder.Tracks) {
        guard !stopSent, let started = recordingSettings, let script = callScript else { return }
        let settings = started.atStop(now: currentSettings?())      // languages, who spoke, translation … as chosen now
        stopSent = true
        callTimer?.invalidate()
        callTimer = nil
        recordingSince = nil
        recordingPaused = false
        if var s = recordingState { s.apply(.stage("Saving the recording")); recordingState = s }
        rec.stop { [weak self] in
            guard let self = self else { return }
            self.recorder = nil
            self.callTracks = nil
            if var s = self.recordingState, self.follower != nil { s.apply(.stage("Finishing what was transcribed live")); self.recordingState = s }
            self.stopFollower { [weak self] in
                guard let self = self else { return }
                self.recordings = Recordings.scan(data: self.engine.data)
                var s = self.recordingState ?? RunState(kind: .record)
                if let folder = self.callFolder { s.apply(.output(path: folder.path)) }
                if let note = self.callNote, !s.notes.contains(note) { s.apply(.note(note)) }
                if self.recordingCancelled || settings.transcribeLater {    // Cancel while the files were closing, or record only
                    s.finish(exitStatus: 0, stopRequested: self.recordingCancelled)
                    self.recordingState = s
                    self.lastFinished = s
                    NSSound(named: NSSound.Name("Glass"))?.play()
                    self.showOutcomeAlert = true
                    self.releaseRecording()
                    self.updateActivity()
                    return
                }
                s.finish(exitStatus: 0, stopRequested: false)
                self.recordingState = s
                var arguments = settings.callArguments(scriptPath: script.path, mic: tracks.mic.path, system: tracks.system.path,
                                                       into: self.callFolder?.path)
                for note in s.notes { arguments += ["--note", note] }        // what happened goes at the top of the transcript
                let step = Step(executable: self.engine.python.path, arguments: arguments,
                                environment: self.environment(settings, apiKey: self.apiKey, yield: false), label: "")
                var seed = RunState(kind: .transcribe)
                seed.lines = s.lines                      // the live lines stay on screen; the final pass refines them in place
                seed.notes = s.notes
                let name = settings.recordName.trimmingCharacters(in: .whitespaces)
                if self.queue.isIdle { self.recordingState = nil }   // the recording panel gives way to the job (pump() does it when the job waited)
                self.enqueue([step], kind: .transcribe, label: "Transcribing the call" + (name.isEmpty ? "" : " " + name),
                             seed: seed, folder: self.callFolder?.path, targets: settings.targets)
                self.releaseRecording()
                self.updateActivity()
            }
        }
    }

    /// Cancel: the audio files stay in the data folder, nothing is transcribed.
    private func cancelCall(_ rec: CallRecorder) {
        recordingCancelled = true
        guard !stopSent else { return }                   // a stop is already closing the files; it ends as cancelled
        stopSent = true
        callTimer?.invalidate()
        callTimer = nil
        recordingPaused = false
        rec.stop { [weak self] in
            guard let self = self else { return }
            self.recorder = nil
            self.callTracks = nil
            self.recordingSince = nil
            // The follower finishes its chunk and ends (its checkpoint stays); the slot is held until it is gone, or a
            // recording started right after Cancel would get this one's last lines and run beside it.
            self.stopFollower { [weak self] in
                guard let self = self else { return }
                self.recordings = Recordings.scan(data: self.engine.data)
                var s = self.recordingState ?? RunState(kind: .record)
                if let folder = self.callFolder { s.apply(.output(path: folder.path)) }
                s.finish(exitStatus: 0, stopRequested: true)
                self.recordingState = s
                self.lastFinished = s
                self.showOutcomeAlert = true
                self.releaseRecording()
                self.updateActivity()
            }
        }
    }

    /// The call could not begin, before any recorder existed (its folder could not be made): told as a failed recording,
    /// as callFailed tells one, with nothing to stop.
    private func callNotStarted(_ message: String) {
        recordingSince = nil
        recordingPaused = false
        var s = recordingState ?? RunState(kind: .record)
        s.apply(.text(message))
        s.finish(exitStatus: 1, stopRequested: false)
        recordingState = s
        lastFinished = s
        NSSound(named: NSSound.Name("Basso"))?.play()
        showOutcomeAlert = true
        releaseRecording()
        updateActivity()
    }

    /// The recording could not start. A start that ends because Cancel or Stop came meanwhile is not a failure: that stop
    /// is already closing the files and will report.
    private func callFailed(_ rec: CallRecorder, _ message: String) {
        guard recorder === rec, !stopSent else { return }
        callTimer?.invalidate()
        callTimer = nil
        stopSent = true
        rec.stop { [weak self] in
            guard let self = self else { return }
            self.recorder = nil
            self.callTracks = nil
            self.recordingSince = nil
            self.recordingPaused = false
            var s = self.recordingState ?? RunState(kind: .record)
            s.apply(.text(message))
            s.finish(exitStatus: 1, stopRequested: false)
            self.recordingState = s
            self.lastFinished = s
            NSSound(named: NSSound.Name("Basso"))?.play()
            self.showOutcomeAlert = true
            self.releaseRecording()
            self.updateActivity()
        }
    }

    /// Nothing can be recorded any more (both sides gone, or the files cannot be written): keep what there is and transcribe it.
    private func callStoppedByItself(_ rec: CallRecorder, _ message: String) {
        guard recorder === rec, !stopSent else { return }
        callNote = message
        addNote(message)
        if let tracks = callTracks {
            finishCall(rec, tracks)
        } else {
            callFailed(rec, message)
        }
    }

    // MARK: - Engine setup

    func setUpEngine() {
        guard !isRunning else { return }
        guard let folder = Runner.engineFolder else {
            engineStatus = .failed("The engine files are missing from the app bundle.")
            return
        }
        let bundled = Runner.bundledRuntime()
        let python = bundled == nil ? Runner.findPython() : nil
        guard bundled != nil || python != nil else {
            engineStatus = .failed("This build carries no Python runtime for this Mac and none (3.9 or newer) was found. Install the Command Line Tools (xcode-select --install) or Homebrew Python (brew install python).")
            return
        }
        try? FileManager.default.createDirectory(at: engine.support, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: engine.venv)
        try? FileManager.default.removeItem(at: engine.runtime)
        try? FileManager.default.removeItem(at: engine.marker)
        engineStatus = .settingUp
        let env = ["MT_EVENTS": "1", "MT_HOME": engine.data.path, "PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"]
        let mlx = Runner.machine == "arm64"       // the Apple-GPU engine is tried on Apple silicon; a failure is not fatal
        let pyannote = UserDefaults.standard.bool(forKey: "pyannote")   // the window's "Use pyannote" choice
        var labels = ["Creating a private Python environment…", "Updating pip…",
                      "Installing faster-whisper and the other packages (a few minutes)…"]
        if mlx { labels.append("Installing the Apple-GPU engine (optional)…") }
        if pyannote { labels.append("Installing pyannote (optional, about 2 GB)…") }
        labels.append("")
        if bundled != nil { labels.insert("Copying the app's Python runtime…", at: 0) }
        let commands = engine.setupSteps(basePython: python, engineFolder: folder.path, bundledRuntime: bundled, mlx: mlx, pyannote: pyannote)
        var steps: [Step] = []
        for (index, command) in commands.enumerated() {
            steps.append(Step(executable: command[0], arguments: Array(command.dropFirst()), environment: env,
                              label: index < labels.count ? labels[index] : "", optional: Engine.isOptionalStep(command)))
        }
        enqueue(steps, kind: .setup, label: "Setting up the speech engine")
    }

    /// The architecture this process runs as: "arm64" or "x86_64".
    static var machine: String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
    }

    /// The Python runtime bundled for this Mac, if the build carries one.
    static func bundledRuntime() -> URL? {
        return Engine.bundledRuntime(resources: Bundle.main.resourceURL, arch: machine)
    }

    /// The first Python of Engine.pythonCandidates that is 3.9 or newer.
    static func findPython() -> String? {
        for path in Engine.pythonCandidates where FileManager.default.isExecutableFile(atPath: path) {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: path)
            p.arguments = ["--version"]
            let pipe = Pipe()
            p.standardOutput = pipe
            p.standardError = pipe
            do { try p.run() } catch { continue }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            if p.terminationStatus == 0, let version = Engine.pythonVersion(String(decoding: data, as: UTF8.self)),
               Engine.isSupported(version) { return path }
        }
        return nil
    }

    // MARK: - The queue of processing jobs

    private func enqueue(_ steps: [Step], kind: RunState.Kind, label: String, seed: RunState? = nil, folder: String? = nil,
                         targets: [String] = []) {
        let job = QueuedJob(id: UUID(), label: label, kind: kind, folder: folder)
        works[job.id] = Work(job: job, steps: steps, seed: seed, targets: targets)
        queue.add(job)
        updateActivity()
        pump()
    }

    /// Starts the next waiting job when none runs.
    private func pump() {
        guard let job = queue.startNext(), let work = works[job.id] else { return }
        stopRequested = false
        // The alert of a job that has just finished stays up (it is dismissed by its OK); clearing it here made it vanish
        // whenever another job was waiting. A recording's panel, kept while its job waited, gives way to the job now.
        if work.seed != nil && !isRecording { recordingState = nil }    // not a newer recording's panel
        state = work.seed ?? RunState(kind: job.kind)
        state?.targets = work.targets              // fixed when the job was queued; the form may have changed since
        ProcessInfo.processInfo.disableAutomaticTermination("Transcribing")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var status: Int32 = 0
            for step in work.steps {
                guard let self = self else { return }
                status = self.run(step, slot: .processing)
                if status != 0 {
                    if step.optional && !self.stopRequested {       // a cancelled optional step is a cancel, not a skip
                        DispatchQueue.main.async { self.apply([.text("An optional step did not succeed; going on without it.")]) }
                        status = 0
                        continue
                    }
                    break
                }
                if self.stopRequested { status = 1; break }         // Cancel between steps: nothing more is started
            }
            DispatchQueue.main.async { self?.finish(status: status, job: job) }
        }
    }

    private enum Slot { case processing, recording, follower }

    /// Runs one process to its end, feeding its output lines to the right state; returns the exit status.
    private func run(_ step: Step, slot: Slot) -> Int32 {
        let toRecording = slot != .processing
        if !step.label.isEmpty {
            DispatchQueue.main.async { self.apply([.stage(step.label)], toRecording: toRecording) }
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: step.executable)
        p.arguments = step.arguments
        var env = ProcessInfo.processInfo.environment
        // The window's offline model is passed as --translator (NLLB, mt.py's default, as nothing): an MT_TRANSLATOR
        // inherited from a shell or launchctl would otherwise override a choice of NLLB.
        env.removeValue(forKey: "MT_TRANSLATOR")
        for (k, v) in step.environment { env[k] = v }
        // Apps started from the Finder get a short PATH; ffmpeg lives in Homebrew's folders.
        env["PATH"] = Engine.extraPath
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch {
            DispatchQueue.main.async { self.apply([.text("Could not start \(step.executable): \(error.localizedDescription)")], toRecording: toRecording) }
            return 127
        }
        lock.lock()
        switch slot {
        case .processing: current = p
        case .recording: recordProcess = p
        case .follower: follower = p
        }
        lock.unlock()

        let handle = pipe.fileHandleForReading
        var buffer = Data()
        var pending: [MTEvent] = []
        func flush() {
            guard !pending.isEmpty else { return }
            let batch = pending
            pending.removeAll()
            DispatchQueue.main.async { self.apply(batch, toRecording: toRecording) }
        }
        // Lines end with a newline or (progress redraws) a carriage return.
        while true {
            let chunk = handle.availableData
            if chunk.isEmpty { break }
            buffer.append(chunk)
            while let end = buffer.firstIndex(where: { $0 == 10 || $0 == 13 }) {
                let line = String(decoding: buffer[buffer.startIndex..<end], as: UTF8.self)
                buffer.removeSubrange(buffer.startIndex...end)
                if !line.isEmpty { pending.append(EventParser.parse(line)) }
            }
            flush()                      // events are rare (one per chunk or file), so no throttling is needed
        }
        if !buffer.isEmpty { pending.append(EventParser.parse(String(decoding: buffer, as: UTF8.self))) }
        flush()
        p.waitUntilExit()
        lock.lock()
        switch slot {
        case .processing: if current === p { current = nil }
        case .recording: if recordProcess === p { recordProcess = nil }
        case .follower: if follower === p { follower = nil }
        }
        lock.unlock()
        return p.terminationStatus
    }

    private func apply(_ events: [MTEvent], toRecording: Bool = false) {
        if toRecording {
            guard var s = recordingState else { return }
            for e in events { s.apply(e) }
            recordingState = s
            if let f = s.folder, recordingFolder == nil {
                recordingFolder = f
                recordingSince = Date()                              // mt.py opens its stream right after naming the folder
                pauseClock = PauseClock()
                recordings = Recordings.scan(data: engine.data)      // the new folder is listed nowhere else
            }
        } else {
            guard var s = state else { return }
            for e in events { s.apply(e) }
            state = s
        }
    }

    private func finish(status: Int32, job: QueuedJob) {
        ProcessInfo.processInfo.enableAutomaticTermination("Transcribing")
        state?.finish(exitStatus: status, stopRequested: stopRequested)
        works[job.id] = nil
        queue.finishCurrent()
        recordings = Recordings.scan(data: engine.data)
        if job.kind == .setup {
            if status == 0 {
                try? Runner.appVersion.write(to: engine.marker, atomically: true, encoding: .utf8)
                engineStatus = .ready
            } else {
                let tail = state?.log.suffix(3).joined(separator: "\n") ?? ""
                engineStatus = .failed(stopRequested ? "Setup was cancelled." : "Setup failed (exit \(status)).\n\(tail)")
            }
            if let checks = state?.checks["ffmpeg"] { ffmpegMissing = checks != "ok" }
            refresh()
            updateActivity()
            pump()
            return
        }
        lastFinished = state
        lastFolder = state?.folder ?? job.folder
        let sound: String
        switch state?.outcome {
        case .completed?: sound = "Glass"
        case .completedWithErrors?: sound = "Purr"
        default: sound = "Basso"
        }
        NSSound(named: NSSound.Name(sound))?.play()
        NSApp.requestUserAttention(.informationalRequest)
        showOutcomeAlert = true
        updateActivity()
        pump()
    }
}
