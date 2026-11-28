import SwiftUI
import AppKit
import CoreAudio
import UniformTypeIdentifiers

/// Choices remembered between launches. An ObservableObject instead of @AppStorage/@State: the Command Line Tools
/// (no Xcode) cannot expand SwiftUI's macro-backed property wrappers.
final class Prefs: ObservableObject {
    @Published var mode: JobSettings.Mode { didSet { UserDefaults.standard.set(mode.rawValue, forKey: "mode") } }
    @Published var outputDir: String { didSet { UserDefaults.standard.set(outputDir, forKey: "outputDir") } }
    /// The languages to translate into, and those spoken in the audio, as chosen in their drop-downs.
    @Published var checked: [String] { didSet { UserDefaults.standard.set(checked, forKey: "checked") } }
    @Published var spoken: [String] { didSet { UserDefaults.standard.set(spoken, forKey: "spoken") } }
    @Published var granularity: Granularity { didSet { UserDefaults.standard.set(granularity.rawValue, forKey: "granularity") } }
    @Published var captureSystem: Bool { didSet { UserDefaults.standard.set(captureSystem, forKey: "captureSystem") } }
    @Published var meName: String { didSet { UserDefaults.standard.set(meName, forKey: "meName") } }
    @Published var themName: String { didSet { UserDefaults.standard.set(themName, forKey: "themName") } }
    @Published var offline: Bool { didSet { UserDefaults.standard.set(offline, forKey: "offline") } }
    @Published var translator: OfflineTranslator { didSet { UserDefaults.standard.set(translator.rawValue, forKey: "translator") } }
    @Published var recordHours: Int { didSet { UserDefaults.standard.set(recordHours, forKey: "recordHours") } }
    @Published var transcribeLater: Bool { didSet { UserDefaults.standard.set(transcribeLater, forKey: "transcribeLater") } }
    @Published var live: Bool { didSet { UserDefaults.standard.set(live, forKey: "live") } }
    @Published var plainLanguage: String { didSet { UserDefaults.standard.set(plainLanguage, forKey: "plainLanguage") } }
    @Published var speakers: Bool { didSet { UserDefaults.standard.set(speakers, forKey: "speakers") } }
    @Published var speakerCount: String { didSet { UserDefaults.standard.set(speakerCount, forKey: "speakerCount") } }
    @Published var pyannote: Bool { didSet { UserDefaults.standard.set(pyannote, forKey: "pyannote") } }
    /// The microphone to record and, for a call, the output device whose sound is the other side: by UID, with the name
    /// the device had when it was chosen (so it can still be named when it is not connected). "" = the Mac's current one.
    @Published var microphone: String { didSet { UserDefaults.standard.set(microphone, forKey: "microphone"); remember(microphone, forKey: "microphoneLabel", in: inputs) } }
    @Published var output: String { didSet { UserDefaults.standard.set(output, forKey: "output"); remember(output, forKey: "outputLabel", in: outputs); followOutput() } }
    @Published private(set) var microphoneLabel: String
    @Published private(set) var outputLabel: String
    /// The devices connected now, kept current by Core Audio's notification (a headset plugged in appears by itself).
    @Published private(set) var inputs: [AudioDevices.Device] = []
    @Published private(set) var outputs: [AudioDevices.Device] = []
    @Published var track = ""
    @Published var recordName = ""
    @Published var files: [String] = []
    /// The output the computer's-sound box was unticked for by itself, because the sound plays through a speaker
    /// ("" = the box is as the user left it). Kept between launches, so the box is ticked again when headphones come back.
    @Published private(set) var untickedFor: String { didSet { UserDefaults.standard.set(untickedFor, forKey: "captureSystemUntickedFor") } }
    /// What the user did with the box for each output (OutputFollowing): a speaker learned from an untick, two tracks kept by a tick.
    private var following: OutputFollowing {
        didSet {
            UserDefaults.standard.set(following.speakers.sorted(), forKey: "speakerOutputs")
            UserDefaults.standard.set(following.twoTracks.sorted(), forKey: "twoTrackOutputs")
        }
    }
    /// Whether a recording runs (the box is then left alone, and looked at again when it ends). Asked of the one Runner
    /// from the start: a window opened again while a recording runs makes a new Prefs, whose first look comes in init.
    var isRecording: () -> Bool = { Runner.shared.isRecording }
    private var deviceWatch: AudioObjectPropertyListenerBlock?
    private var defaultOutputWatch: AudioObjectPropertyListenerBlock?
    private var dataSourceWatch: (device: AudioObjectID, block: AudioObjectPropertyListenerBlock)?

    init() {
        let d = UserDefaults.standard
        mode = JobSettings.Mode(rawValue: d.string(forKey: "mode") ?? "") ?? .files
        outputDir = d.string(forKey: "outputDir") ?? ""
        // Languages typed into the "Others" fields before 1.10.6 join the choice once, and those fields are gone
        // ("language" was an even older single field for the spoken one).
        // Local values: a @Published property cannot be read back before every stored property is set.
        let targets = Languages.merged(d.stringArray(forKey: "checked") ?? ["English"], typed: d.string(forKey: "custom") ?? "",
                                       names: Languages.targetAll)
        let spokenNow = Languages.merged(d.stringArray(forKey: "spoken") ?? [],
                                         typed: d.string(forKey: "spokenCustom") ?? d.string(forKey: "language") ?? "",
                                         names: Languages.spokenAll)
        checked = targets
        spoken = spokenNow
        for old in ["custom", "spokenCustom", "language"] where d.object(forKey: old) != nil {
            d.removeObject(forKey: old)
            d.set(targets, forKey: "checked")
            d.set(spokenNow, forKey: "spoken")
        }
        granularity = Granularity(rawValue: d.string(forKey: "granularity") ?? "") ?? .auto
        captureSystem = d.bool(forKey: "captureSystem")
        meName = d.string(forKey: "meName") ?? "You"
        themName = d.string(forKey: "themName") ?? "Them"
        offline = d.object(forKey: "offline") == nil ? true : d.bool(forKey: "offline")     // offline unless chosen otherwise
        // Hy-MT3 on Apple silicon for a user who never picked a model; a stored choice is kept. The key is written only by
        // the picker (didSet does not run for this assignment in init), so this default is not stored as a choice.
        translator = OfflineTranslator.initial(stored: d.string(forKey: "translator"), appleSilicon: Runner.machine == "arm64")
        let hours = d.integer(forKey: "recordHours")
        recordHours = (1...8).contains(hours) ? hours : 4
        transcribeLater = d.bool(forKey: "transcribeLater")
        live = d.object(forKey: "live") == nil ? true : d.bool(forKey: "live")
        plainLanguage = d.string(forKey: "plainLanguage") ?? ""
        speakers = d.bool(forKey: "speakers")
        speakerCount = d.string(forKey: "speakerCount") ?? ""
        pyannote = d.bool(forKey: "pyannote")
        microphone = d.string(forKey: "microphone") ?? ""
        microphoneLabel = d.string(forKey: "microphoneLabel") ?? ""
        output = d.string(forKey: "output") ?? ""
        outputLabel = d.string(forKey: "outputLabel") ?? ""
        untickedFor = d.string(forKey: "captureSystemUntickedFor") ?? ""
        following = OutputFollowing(speakers: Set(d.stringArray(forKey: "speakerOutputs") ?? []),
                                    twoTracks: Set(d.stringArray(forKey: "twoTrackOutputs") ?? []))
        refreshDevices()
        deviceWatch = AudioDevices.watch(queue: .main) { [weak self] in
            self?.refreshDevices()
            self?.followOutput()                           // a chosen output plugged in or out
        }
        defaultOutputWatch = AudioDevices.watchDefaultOutput(queue: .main) { [weak self] in self?.followOutput() }
        followOutput()
    }

    deinit {
        if let watch = deviceWatch { AudioDevices.stopWatching(watch, queue: .main) }
        if let watch = defaultOutputWatch { AudioDevices.stopWatchingDefaultOutput(watch, queue: .main) }
        if let watch = dataSourceWatch { AudioDevices.stopWatchingDataSource(of: watch.device, watch.block, queue: .main) }
    }

    /// The box ticked or unticked by the user: kept so, and remembered for the output the sound goes to now. An untick
    /// marks that output as a speaker (macOS does not say that a Bluetooth device is one), a tick keeps two tracks with it.
    func userSetCaptureSystem(_ on: Bool) {
        captureSystem = on
        untickedFor = ""
        if let now = followedOutput() { following.userChose(ticked: on, uid: followingKey(now.uid, now.device)) }
    }

    /// The output's UID, with its selected data source when it has one (OutputFollowing.key).
    private func followingKey(_ uid: String, _ device: AudioObjectID) -> String {
        return OutputFollowing.key(uid: uid, dataSource: AudioDevices.dataSource(of: device))
    }

    /// The output whose sound is the other side: the one chosen under "Take the computer's sound from" when it is
    /// connected, else the Mac's current output.
    private func followedOutput() -> (uid: String, device: AudioObjectID)? {
        if !output.isEmpty, let device = AudioDevices.id(forUID: output) { return (output, device) }
        guard let uid = AudioDevices.defaultUID(.output), let device = AudioDevices.id(forUID: uid) else { return nil }
        return (uid, device)
    }

    /// Unticks the computer's-sound box when the sound goes to a speaker (the microphone hears both sides), and ticks it
    /// again when the sound goes elsewhere, if it was unticked by itself (OutputFollowing). Never while a recording runs:
    /// its two tracks or one were fixed at Start.
    func followOutput() {
        guard let now = followedOutput() else { return }
        watchDataSource(of: now.device)
        guard !isRecording() else { return }
        let speaker = following.isSpeaker(uid: followingKey(now.uid, now.device), kind: AudioDevices.outputKind(of: now.device))
        let name = AudioDevices.name(of: now.device) ?? now.uid
        switch OutputFollowing.box(ticked: captureSystem, untickedByApp: !untickedFor.isEmpty, speaker: speaker) {
        case .some(false):
            captureSystem = false
            untickedFor = name
        case .some(true):
            captureSystem = true
            untickedFor = ""
        case .none:
            if speaker && !captureSystem && !untickedFor.isEmpty && untickedFor != name { untickedFor = name }   // from one speaker to another
        }
    }

    /// Follows the data source of the followed output (an Intel Mac's jack: speakers or headphones on one device).
    private func watchDataSource(of device: AudioObjectID) {
        if dataSourceWatch?.device == device { return }
        if let old = dataSourceWatch { AudioDevices.stopWatchingDataSource(of: old.device, old.block, queue: .main) }
        dataSourceWatch = AudioDevices.watchDataSource(of: device, queue: .main) { [weak self] in self?.followOutput() }
            .map { (device: device, block: $0) }
    }

    /// Reads the device lists again (a device was plugged in or out, or the user asked).
    func refreshDevices() {
        inputs = AudioDevices.list(.input)
        outputs = AudioDevices.list(.output)
    }

    private func remember(_ uid: String, forKey key: String, in devices: [AudioDevices.Device]) {
        guard let name = devices.first(where: { $0.uid == uid })?.name else { return }
        UserDefaults.standard.set(name, forKey: key)
        if key == "microphoneLabel" { microphoneLabel = name } else { outputLabel = name }
    }

    var settings: JobSettings {
        var s = JobSettings()
        s.mode = mode
        s.files = files
        s.outputDir = outputDir
        s.targets = Languages.targets(checked: checked, custom: "")
        s.offline = offline
        s.spoken = Languages.targets(checked: spoken, custom: "")
        s.granularity = granularity
        s.captureSystem = captureSystem
        s.microphone = DeviceChoice(uid: microphone, label: microphoneLabel)
        s.microphoneName = s.microphone.recorderName(among: inputs.map { (uid: $0.uid, name: $0.name) })
        s.output = DeviceChoice(uid: output, label: outputLabel)
        s.meName = meName
        s.themName = themName
        s.track = track
        s.recordName = recordName
        s.recordHours = recordHours
        s.transcribeLater = transcribeLater
        s.live = live
        s.translator = Runner.machine == "arm64" ? translator : .nllb      // Hy-MT3 and TranslateGemma run on the Apple GPU only
        s.plainLanguage = plainLanguage
        s.speakers = speakers
        s.speakerCount = speakerCount
        s.pyannote = pyannote
        return s
    }
}

/// Window-only flags.
final class ViewFlags: ObservableObject {
    @Published var showKeySheet = false
    @Published var keyText = ""
    @Published var keyError = ""
    @Published var keyInfo = ""
    @Published var showOptions = false
    @Published var showLog = false
    @Published var notice = ""
    @Published var scan = ScanTracker()      // folder scans running in the background
    @Published var transcriptFilter = ""
    @Published var followTranscript = true
    @Published var shownTranslations: [String]? = nil   // the translations shown beside the transcript (nil = all on offer)
    @Published var showTranslationMenu = false
    @Published var translationQuery = ""
    @Published var showTokenSheet = false
    @Published var tokenText = ""
    @Published var tokenError = ""
    @Published var showRename = false
    @Published var renameNames: [String: String] = [:]
    @Published var renameFolder = ""
    @Published var confirmCancelRecording = false
    @Published var confirmCancelJob = false
    @Published var showStored = false
    @Published var showTargetLanguages = false   // the "translate to" drop-down is open
    @Published var targetQuery = ""
    @Published var showSpokenLanguages = false   // the "languages spoken" drop-down is open
    @Published var spokenQuery = ""
}

struct ContentView: View {
    @EnvironmentObject private var runner: Runner
    @StateObject private var prefs = Prefs()
    @StateObject private var flags = ViewFlags()

    private var settings: JobSettings { return prefs.settings }
    private var recording: Bool { return runner.isRecording }
    private var engineReady: Bool { return runner.engineStatus == .ready }

    /// Translating with Claude is asked for, but there is no key: mt.py would refuse the whole job before transcribing.
    private var keyMissing: Bool { return settings.needsAPIKey && !runner.hasAPIKey }

    private var problems: [String] {
        var list = settings.validationProblems()
        if flags.scan.busy { list.append("Wait until the folders have been read.") }
        if keyMissing { list.append("Add your Claude API key to translate (Set Key…).") }
        return list
    }

    var body: some View {
        ScrollViewReader { window in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if !engineReady { engineSection }
                    if runner.ffmpegMissing { ffmpegNotice }
                    inputSection
                    if prefs.mode != .translate { spokenSection; speakersSection }
                    outputSection
                    translateSection
                    optionsSection
                    actionBar
                    if let state = runner.recordingState { recordingSection(state).id("recording") }
                    if let state = runner.state { processingSection(state).id("processing") }
                    if !runner.recordings.isEmpty { recordingsSection }
                }
                .padding(16)
            }
            // The panels sit under the settings, below the window's fold at its default size: bring the recording's into
            // view when it appears and when its first line arrives, and a job's when it starts.
            .onChange(of: runner.recordingState != nil) { shown in
                if shown { withAnimation { window.scrollTo("recording", anchor: .top) } }
            }
            .onChange(of: runner.recordingState?.lines.isEmpty == false) { hasLines in
                if hasLines { withAnimation { window.scrollTo("recording", anchor: .top) } }
            }
            .onChange(of: runner.queue.current?.id) { current in
                if current != nil && runner.recordingState == nil { withAnimation { window.scrollTo("processing", anchor: .top) } }
            }
        }
        .frame(minWidth: 720, minHeight: 640)
        .onAppear {
            runner.refresh()
            runner.currentSettings = { [weak prefs] in prefs?.settings ?? JobSettings() }
        }
        .onChange(of: recording) { now in
            if !now { prefs.followOutput() }               // the output may have changed during the recording
        }
        .onDrop(of: [UTType.fileURL], isTargeted: nil) { providers in handleDrop(providers) }
        .sheet(isPresented: $flags.showKeySheet) { keySheet }
        .sheet(isPresented: $flags.showTokenSheet) { tokenSheet }
        .sheet(isPresented: $flags.showRename) { renameSheet }
        .alert(alertTitle, isPresented: $runner.showOutcomeAlert) {
            Button("OK") {}
            if let first = runner.lastFinished?.outputs.first {
                Button("Show Files") { reveal(first) }
            }
        } message: { Text(alertMessage) }
    }

    // MARK: - Engine

    private var engineSection: some View {
        GroupBox("Speech engine") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Transcription runs on this Mac with the Whisper model. The one-time setup copies the app's own Python runtime, installs the packages and downloads the speech model, the speaker-separation models and ffmpeg (about 2 GB in all, a few minutes; internet is needed only for this). Nothing else has to be installed.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                engineStatusRow
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder private var engineStatusRow: some View {
        switch runner.engineStatus {
        case .settingUp:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(runner.state?.stage ?? "Setting up…")
                Spacer()
                Button("Cancel") { runner.cancelProcessing() }
            }
        case .failed(let message):
            Text(message).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            Button("Try Again") { runner.setUpEngine() }.disabled(runner.isRunning).help(setupHelp)
        default:
            Button("Set Up Speech Engine") { runner.setUpEngine() }.keyboardShortcut(.defaultAction).disabled(runner.isRunning).help(setupHelp)
        }
        if runner.isRunning {
            Text("The setup rebuilds the engine every recording and job runs on, so it waits until the recording and the queued jobs are done.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var setupHelp: String {
        return "Not while a recording or a job runs: the setup rebuilds the engine they run on."
    }

    private var ffmpegNotice: some View {
        GroupBox {
            HStack {
                Text("ffmpeg could not be fetched during the setup (no internet?). It reads audio and video files: run the setup again when online, or install it with Homebrew:")
                Text("brew install ffmpeg").font(.system(.body, design: .monospaced)).textSelection(.enabled)
                Spacer()
                Button("Check Again") { runner.refresh() }
            }
            .foregroundStyle(.orange)
            .padding(6)
        }
    }

    // MARK: - Input

    private var inputSection: some View {
        GroupBox("Input") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("", selection: $prefs.mode) {
                    ForEach(JobSettings.Mode.allCases) { m in Text(m.label).tag(m) }
                }
                .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 480)
                if prefs.mode == .record { recordRows } else { fileRows }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var fileRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            if prefs.files.isEmpty {
                Text(prefs.mode == .translate
                     ? "Drop transcript text files or folders of them here, or add them. Plain text works too."
                     : "Drop audio or video files or whole folders here, or add them: mp3, m4a, wav, flac, ogg, opus, mp4, mov, mkv … (a video's audio track is used; a folder adds the audio and video files inside it, at any depth).")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(prefs.files, id: \.self) { path in
                            HStack {
                                Image(systemName: "doc")
                                Text(URL(fileURLWithPath: path).lastPathComponent).lineLimit(1).truncationMode(.middle)
                                Spacer()
                                Button { prefs.files.removeAll { $0 == path } } label: { Image(systemName: "xmark.circle") }
                                    .buttonStyle(.borderless)
                            }
                        }
                    }
                }
                .frame(maxHeight: 150)
            }
            HStack {
                Button("Add Files or Folders…") { addFiles() }
                Button("Clear") { prefs.files = []; flags.notice = ""; flags.scan.invalidate() }
                    .disabled(prefs.files.isEmpty && !flags.scan.busy)
            }
            if prefs.mode == .translate {
                HStack {
                    Text("Language of plain text files")
                    TextField("e.g. Russian (needed for offline translation of text without [tags])", text: $prefs.plainLanguage).frame(maxWidth: 420)
                }
            }
            if flags.scan.busy {
                HStack(spacing: 8) { ProgressView().controlSize(.small); Text("Reading folders…") }
            }
            if !flags.notice.isEmpty {
                Text(flags.notice).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// What the recording itself is started with is locked while it runs; "Transcribe afterwards" is read when it stops, so it stays free.
    private var recordRows: some View {
        VStack(alignment: .leading, spacing: 8) {
            Group {
                HStack {
                    Text("Name")
                    TextField("optional, e.g. weekly", text: $prefs.recordName).frame(maxWidth: 240)
                    Text("Stop after")
                    Picker("", selection: $prefs.recordHours) {
                        ForEach(Array(1...8), id: \.self) { h in Text("\(h) h").tag(h) }
                    }
                    .labelsHidden().frame(width: 80)
                }
                Text("A notice comes 10 minutes before the recording stops by itself; \"Extend by 1 h\" keeps it going; \"Pause\" leaves a gap out of the recording.")
                    .foregroundStyle(.secondary)
                HStack {
                    Text("Microphone")
                    devicePicker(selection: $prefs.microphone, devices: prefs.inputs, savedLabel: prefs.microphoneLabel,
                                 current: "The Mac's current input (System Settings > Sound > Input)")
                    Button("Refresh") { prefs.refreshDevices() }.help("Read the connected devices again (a plugged-in device appears by itself; this is for when it does not).")
                }
                Toggle("Transcribe while recording (the lines appear as the meeting goes on; at the end only the tail is left to do)", isOn: $prefs.live)
                    .toggleStyle(.checkbox)
            }
            .disabled(recording)
            Toggle("Transcribe afterwards (only record now; the recording stays in the data folder and can be transcribed later)", isOn: $prefs.transcribeLater)
                .toggleStyle(.checkbox)
            Toggle("Also record the sound this computer plays (the other side of a call)",
                   isOn: Binding(get: { prefs.captureSystem }, set: { prefs.userSetCaptureSystem($0) })).toggleStyle(.checkbox)
                .disabled(recording)
            if !prefs.captureSystem && !prefs.untickedFor.isEmpty {
                Text(OutputFollowing.untickedNote(output: prefs.untickedFor))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Also while the box was unticked by itself: a speaker chosen here unticks it, and choosing another output
            // (headphones, "Every output") must stay possible without a tick, which would be remembered for that speaker.
            if prefs.captureSystem || !prefs.untickedFor.isEmpty {
                HStack {
                    Text("Take the computer's sound from")
                    devicePicker(selection: $prefs.output, devices: prefs.outputs, savedLabel: prefs.outputLabel,
                                 current: "Every output (all the sound this Mac plays, whatever it goes to)")
                }
                .disabled(recording)
            }
            if prefs.captureSystem {
                HStack {
                    Text("Your microphone is")
                    TextField("You", text: $prefs.meName).frame(width: 110)
                    Text("the other side is")
                    TextField("Them", text: $prefs.themName).frame(width: 110)
                }
                .disabled(recording)
                Text("Two tracks are recorded side by side, so every line of the transcript says who spoke. With \"Every output\" the computer's-sound track takes everything this Mac plays, from every app to every output device (both channels, mixed to one), and the capture follows a change of the current output; a chosen device limits the track to what is played to that device, until it is disconnected. The microphone track takes every channel of the input, mixed to one. On macOS 14.2 or newer the computer's sound is taken from a Core Audio tap: the first time, macOS asks for the System Audio Recording Only permission (no screen, no picture); allow Meeting Transcriber under System Settings > Privacy & Security > Screen & System Audio Recording. On older systems the screen capture is used, which needs the full Screen Recording permission and takes the sound of every output, whatever is chosen above. A build without the project's signing identity has to be allowed again after every update (README: Permissions and updates). When the computer's sound cannot be captured, or stays silent, the call goes on with the microphone alone and a warning says so. The microphone's echo cancellation keeps what the Mac plays off your track where macOS allows it, and only with the Mac's current input (with a chosen microphone it is off, and a note says so); headphones are cleaner still, and whatever still comes through is trimmed from your lines. While the sound plays through a speaker the microphone hears both sides by itself, so this box is unticked by itself for the Mac's own speakers, HDMI, DisplayPort and AirPlay; for a Bluetooth or USB speaker, untick it once while the sound plays through it and the app remembers that speaker.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Text("Records the chosen microphone (or the Mac's current sound input, which follows System Settings > Sound > Input); macOS asks for permission the first time. A chosen microphone that is not connected when the recording starts is reported, and the current input is recorded instead. Tick the box above to record the other side of a call too. While the sound plays through a speaker the box is unticked by itself, since the microphone then hears both sides: the Mac's own speakers, a display or TV over HDMI or DisplayPort, AirPlay, and any output you untick the box with (macOS does not say that a Bluetooth or USB device is a speaker, so the app learns it from you); it is ticked again when the sound goes to an output that is not a speaker (headphones, or a device you have not unticked it with).")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text("Recording people requires their consent in many places.")
                .foregroundStyle(.secondary)
            if let warning = SigningIdentity.warning(for: runner.signing) {
                Text(warning).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// A menu of the connected devices of one kind, with the Mac's current device first. A saved device that is not
    /// connected now stays selectable (named, marked), so the choice survives unplugging the headset.
    private func devicePicker(selection: Binding<String>, devices: [AudioDevices.Device], savedLabel: String, current: String) -> some View {
        let saved = selection.wrappedValue
        let missing = !saved.isEmpty && !devices.contains(where: { $0.uid == saved })
        return Picker("", selection: selection) {
            Text(current).tag("")
            ForEach(devices) { device in Text(device.name).tag(device.uid) }
            if missing { Text((savedLabel.isEmpty ? saved : savedLabel) + " (not connected)").tag(saved) }
        }
        .labelsHidden().frame(maxWidth: 420)
    }

    // MARK: - Output, translation, options

    private var outputSection: some View {
        GroupBox("Output folder") {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    PathField(path: prefs.outputDir, placeholder: "Default: " + JobSettings.defaultOutputDir)
                    Button("Choose…") {
                        if let p = pickFolder(start: prefs.outputDir.isEmpty ? JobSettings.defaultOutputDir : prefs.outputDir,
                                              message: "Choose the folder for the meetings' files") {
                            prefs.outputDir = p
                        }
                    }
                    Button("Default") { prefs.outputDir = "" }.disabled(prefs.outputDir.isEmpty)
                }
                Text("Every meeting gets a subfolder here named after the hour it began (26-11-22 14: yy-mm-dd hh) with its transcript, its translations and the recording itself (a copy that takes no extra space on an APFS disk). Two meetings of one hour share the subfolder, and the second one's files are numbered (recording-2) so nothing is overwritten.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var translateSection: some View {
        GroupBox(prefs.mode == .translate ? "Translate to" : "Also translate to (a separate file per language)") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Languages")
                    LanguageMenu(names: Languages.targets(offline: prefs.offline), selection: $prefs.checked,
                                 none: prefs.mode == .translate ? "Choose languages" : "No translation",
                                 isPresented: $flags.showTargetLanguages, query: $flags.targetQuery)
                    Spacer()
                    if !prefs.offline {
                        Text(runner.hasAPIKey ? "Claude API key: saved in \(runner.keySource)" : "No Claude API key").foregroundStyle(runner.hasAPIKey ? Color.secondary : Color.orange)
                        Button("Set Key…") { flags.keyError = ""; flags.showKeySheet = true }
                    }
                }
                if !flags.keyInfo.isEmpty && !prefs.offline {
                    Text(flags.keyInfo).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                Toggle("Translate offline on this Mac (no Claude, no API key)", isOn: $prefs.offline).toggleStyle(.checkbox)
                if prefs.offline {
                    HStack {
                        Text("Offline model")
                        Picker("", selection: $prefs.translator) {
                            ForEach(OfflineTranslator.allCases) { Text($0.label).tag($0) }
                        }
                        .labelsHidden().frame(maxWidth: 320)
                        .disabled(Runner.machine != "arm64")
                    }
                    Text((Runner.machine == "arm64" ? prefs.translator : .nllb).hint
                         + (Runner.machine == "arm64" ? "" : " Hy-MT3 and TranslateGemma need Apple silicon.")
                         + " After the download no internet is needed.")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// The languages that are spoken in the audio: none ticked = every stretch of speech is detected among all of
    /// Whisper's languages; several = detected among those; one = forced.
    private var spokenSection: some View {
        GroupBox("Languages spoken in the audio (optional)") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Languages")
                    LanguageMenu(names: Languages.spokenAll, selection: $prefs.spoken, none: "Any (recognised automatically)",
                                 isPresented: $flags.showSpokenLanguages, query: $flags.spokenQuery)
                }
                Text(spokenHint).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack {
                    Text("Detection granularity")
                    Picker("", selection: $prefs.granularity) {
                        ForEach(Granularity.allCases) { g in Text(g.label).tag(g) }
                    }
                    .pickerStyle(.segmented).labelsHidden().frame(maxWidth: 380)
                }
                Text(prefs.granularity.hint + " Each chunk costs a full Whisper pass, so finer settings take longer.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// Several people on one microphone (or on the other side of a call): who spoke.
    private var speakersSection: some View {
        GroupBox("Who spoke") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Several people on one microphone: label who spoke (Speaker 1, Speaker 2 …; in a call both sides: You 1, You 2 … for several people at your microphone, Them 1, Them 2 … on the other side)", isOn: $prefs.speakers)
                    .toggleStyle(.checkbox)
                if prefs.speakers {
                    HStack {
                        Text("How many")
                        TextField("empty = find out", text: $prefs.speakerCount).frame(width: 140)
                        Text("Leave it empty unless the result is wrong: finding the number out did best in the test.").foregroundStyle(.secondary)
                    }
                    Text("Voices are told apart on this Mac (sherpa-onnx: pyannote segmentation and 3D-Speaker embeddings, 47 MB fetched at setup). Measured on a 10-minute six-voice test: all 6 found, 96 % of the sentences labelled right, 94 % of the speaker changes caught, about 0.4× the audio length on a 4-core CPU; two people talking at once are marked with a note. The lines are labelled when the transcription ends; you can give the speakers their real names afterwards (Name Speakers…).")
                        .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Toggle("Use pyannote instead (optional: better at telling voices apart, much slower on a CPU, about 2 GB of packages installed at setup; its models need a Hugging Face token once)", isOn: $prefs.pyannote)
                        .toggleStyle(.checkbox)
                    if prefs.pyannote {
                        HStack {
                            Text(runner.hasHFToken ? "Hugging Face token: saved" : "No Hugging Face token (needed for the one-time model download)")
                                .foregroundStyle(runner.hasHFToken ? Color.secondary : Color.orange)
                            Button("Set Token…") { flags.tokenError = ""; flags.showTokenSheet = true }
                            Text("Its packages are installed by \"Set Up Speech Engine Again…\" while this box is ticked.").foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var spokenHint: String {
        switch settings.spoken.count {
        case 0: return "None chosen: the language is recognised automatically for every stretch of speech (all ~100 languages), also when the same person changes language part-way: a stretch that sounds mixed is cut finer and each part detected on its own. The speaker label and the language are decided independently. Choose languages only to restrict detection (several) or to force one."
        case 1: return "One language: it is used for the whole recording, whatever is really spoken."
        default: return "Each stretch of speech is detected among these \(settings.spoken.count) languages, so the language can change during the meeting. A language that is not chosen still wins a stretch when Whisper is sure of it; the line is then marked with a ? (your cue to choose it). When Whisper is unsure, the likeliest chosen language is used."
        }
    }

    private var optionsSection: some View {
        DisclosureGroup("Options", isExpanded: $flags.showOptions) {
            Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
                GridRow {
                    Text("Audio track of a video").gridColumnAlignment(.trailing)
                    TextField("first (0); a number to pick another", text: $prefs.track).frame(maxWidth: 300)
                }
            }
            .padding(.top, 6)
            .disabled(prefs.mode == .translate)
        }
    }

    // MARK: - Actions

    private var startLabel: String {
        switch prefs.mode {
        case .files: return "Transcribe"
        case .record: return "Start Recording"
        case .translate: return "Translate"
        }
    }

    /// Start: a recording takes the recording slot (one at a time); a transcription or translation joins the queue, so a new
    /// interview can be recorded while the last one is still being processed.
    private var actionBar: some View {
        let blocked = prefs.mode == .record && recording
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button(startLabel) { runner.start(settings) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!engineReady || blocked || !problems.isEmpty)
                if !runner.queue.isIdle && prefs.mode != .record {
                    Text("joins the queue behind \(runner.queue.count) job\(runner.queue.count == 1 ? "" : "s")").foregroundStyle(.secondary)
                }
                Spacer()
            }
            if blocked {
                Text("A recording is in progress; stop it before starting another.").foregroundStyle(.secondary)
            } else if engineReady && !problems.isEmpty {
                Text(problems.joined(separator: " ")).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            if let error = runner.startupError { Text(error).foregroundStyle(.red) }
        }
    }

    // MARK: - Recordings in the data folder

    private var recordingsSection: some View {
        let busyFolders = runner.busyFolders          // being recorded, or queued: not offered a second time
        let unfinished = runner.recordings.filter { $0.unfinished && !busyFolders.contains($0.folder) }
        let stored = runner.recordings.filter { !$0.unfinished && !busyFolders.contains($0.folder) }
        return GroupBox("Recordings") {
            VStack(alignment: .leading, spacing: 6) {
                if !unfinished.isEmpty {
                    Text("Not transcribed yet (a cancelled or interrupted job):").foregroundStyle(.secondary)
                    ForEach(unfinished) { entry in
                        HStack {
                            Image(systemName: "waveform")
                            Text(entry.label + (entry.hasCheckpoint ? " — partly transcribed" : "")).lineLimit(1).truncationMode(.middle)
                            Spacer()
                            Button(entry.hasCheckpoint ? "Continue" : "Transcribe") { runner.transcribeRecording(entry, settings: settings) }
                                .disabled(!engineReady || keyMissing)
                                .help(keyMissing ? "Add your Claude API key to translate (Set Key…), or tick \"Translate offline\"" : "")
                            Button("Reveal") { reveal(entry.folder) }
                        }
                    }
                }
                if !stored.isEmpty {
                    DisclosureGroup("Stored meetings (\(stored.count))", isExpanded: $flags.showStored) {
                        ForEach(stored.prefix(30)) { entry in
                            HStack {
                                Image(systemName: "doc.text")
                                Text(entry.label).lineLimit(1).truncationMode(.middle)
                                Spacer()
                                Button("Translate") { runner.translateStored(entry, settings: settings) }
                                    .disabled(!engineReady || settings.targets.isEmpty || keyMissing)
                                    .help(keyMissing ? "Add your Claude API key to translate (Set Key…), or tick \"Translate offline\""
                                                     : "Into the languages chosen under \"Also translate to\"; finished translations are reused")
                                Button("Reveal") { reveal(entry.folder) }
                            }
                        }
                    }
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - The recording panel

    @ViewBuilder private func recordingSection(_ state: RunState) -> some View {
        GroupBox("Recording") {
            VStack(alignment: .leading, spacing: 8) {
                if state.phase == .recording {
                    recordingRow(state)
                    let text = state.progressText
                    if !text.isEmpty {
                        Text("Live: " + text).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
                    } else if prefs.live && !prefs.transcribeLater {
                        // Until the first line there is nothing else to see of the live transcription: say what it waits for.
                        Text(liveWaitText(state, short: false))
                            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    }
                    Text("Who spoke, the translation of the files, the names of the two sides and the output folder can still be changed: they apply when the recording stops."
                         + (prefs.live && !prefs.transcribeLater ? " The languages translated live are the ones chosen when the recording started. Changing the spoken languages or the granularity now makes the final pass redo what was transcribed live." : ""))
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                } else if state.phase == .working {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(state.stage.isEmpty ? "Saving the recording…" : state.stage)
                    }
                } else {
                    Text(state.outcome == .stopped ? "Recording cancelled: the audio stays in the data folder."
                         : state.outcome == .completed ? "Recording saved." : "The recording ended with an error.")
                }
                // The pane is there from the start of a live recording (empty until the first stretch of speech is closed),
                // so the transcript and its translation have their place on screen before the first line comes.
                if !state.lines.isEmpty || (state.phase == .recording && prefs.live && !prefs.transcribeLater) { transcriptPane(state) }
                resultRows(state)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder private func recordingRow(_ state: RunState) -> some View {
        HStack {
            if let since = runner.recordingSince {
                TimelineView(.periodic(from: since, by: 1)) { context in
                    HStack {
                        Image(systemName: runner.recordingPaused ? "pause.circle" : "record.circle")
                            .foregroundStyle(runner.recordingPaused ? Color.orange : Color.red)
                        Text((runner.recordingPaused ? "Paused " : "Recording ")
                             + Format.duration(Int(runner.recordedTime(at: context.date) ?? 0)))
                            .font(.system(.title3, design: .monospaced))
                    }
                }
            } else {
                Text("Recording…")
            }
            ProgressView(value: state.level).frame(width: 90)
                .help("Microphone level: it moves while you speak")
            Button(runner.recordingPaused ? "Resume" : "Pause") {
                if runner.recordingPaused { runner.resumeRecording() } else { runner.pauseRecording() }
            }
            .disabled(!runner.recordingReady)          // until the recorder is in place a pause would be lost, not shown
            .help("Nothing is recorded while paused; the recording and its transcript leave the pause out, and a note says where it was.")
            Button("Extend by 1 h") { runner.extendRecording() }
                .disabled(!runner.recordingReady)
            Spacer()
            Button("Stop Recording") { runner.stopRecording() }
            Button("Cancel") { flags.confirmCancelRecording = true }
                .confirmationDialog("Cancel the recording?", isPresented: $flags.confirmCancelRecording, titleVisibility: .visible) {
                    Button("Cancel Recording", role: .destructive) { runner.cancelRecording() }
                    Button("Keep Going", role: .cancel) { }
                } message: {
                    Text("The audio recorded so far stays in the data folder; nothing is transcribed.")
                }
        }
    }

    // MARK: - The processing panel: the running job, then the queue

    @ViewBuilder private func processingSection(_ state: RunState) -> some View {
        GroupBox(state.kind == .setup ? "Setup" : "Processing") {
            VStack(alignment: .leading, spacing: 8) {
                if state.phase == .finished {
                    if let label = runner.queue.current?.label { Text(label) }
                } else {
                    HStack {
                        if let label = runner.queue.current?.label { Text(label).font(.headline) }
                        Spacer()
                        if state.kind != .setup {
                            Button("Cancel") { flags.confirmCancelJob = true }
                                .confirmationDialog("Cancel this job?", isPresented: $flags.confirmCancelJob, titleVisibility: .visible) {
                                    Button("Cancel Job", role: .destructive) { runner.cancelProcessing() }
                                    Button("Keep Going", role: .cancel) { }
                                } message: {
                                    Text("What is already done stays on disk; a transcription continues from its checkpoint when it is run again. The jobs waiting behind it go on.")
                                }
                        }
                    }
                }
                if state.kind == .setup {
                    if !state.stage.isEmpty { Text(state.stage) }
                } else if state.phase == .starting {
                    Text("Starting…")
                } else if state.phase == .working {
                    workRows(state)
                }
                if !state.lines.isEmpty { transcriptPane(state) }
                resultRows(state)
                if !runner.queue.pending.isEmpty { queueRows }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var queueRows: some View {
        let pending = runner.queue.pending
        return VStack(alignment: .leading, spacing: 4) {
            Text("Waiting (\(pending.count))").font(.headline).padding(.top, 4)
            ForEach(Array(pending.enumerated()), id: \.element.id) { index, job in
                HStack {
                    Image(systemName: job.kind == .translate ? "character.book.closed" : "waveform")
                    Text(job.label).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button { runner.moveQueued(job.id, up: true) } label: { Image(systemName: "arrow.up") }
                        .buttonStyle(.borderless).disabled(index == 0).help("Earlier")
                    Button { runner.moveQueued(job.id, up: false) } label: { Image(systemName: "arrow.down") }
                        .buttonStyle(.borderless).disabled(index == pending.count - 1).help("Later")
                    Button("Remove") { runner.removeQueued(job.id) }
                }
            }
        }
    }

    /// Notes, written files, errors and the log of a job or a recording.
    @ViewBuilder private func resultRows(_ state: RunState) -> some View {
        ForEach(Array(state.notes.enumerated()), id: \.offset) { _, note in
            Label(note, systemImage: "exclamationmark.circle")
                .foregroundStyle(Color.orange).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
        }
        ForEach(Array(state.outputs.enumerated()), id: \.offset) { _, path in
            HStack {
                Image(systemName: "doc.text")
                Text(URL(fileURLWithPath: path).lastPathComponent).lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("Show") { reveal(path) }
                Button("Open") { NSWorkspace.shared.open(URL(fileURLWithPath: path)) }
            }
        }
        ForEach(Array(state.errors.enumerated()), id: \.offset) { _, failure in
            Text("\(URL(fileURLWithPath: failure.file).lastPathComponent): \(failure.message)")
                .foregroundStyle(.red).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
        }
        if !state.log.isEmpty {
            DisclosureGroup("Details", isExpanded: $flags.showLog) {
                Text(state.log.joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    @ViewBuilder private func workRows(_ state: RunState) -> some View {
        if state.fileCount > 0 {
            Text("File \(state.fileIndex) of \(state.fileCount): \(URL(fileURLWithPath: state.currentFile).lastPathComponent)")
                .lineLimit(1).truncationMode(.middle)
        }
        if state.chunks > 0 {
            ProgressView(value: state.fileFraction) {
                Text(state.stage.isEmpty ? (state.checks["engine"] == "mlx" ? "Transcribing on the GPU (MLX)" : "Transcribing on the CPU") : state.stage)
            }
        } else {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(state.stage.isEmpty ? (state.kind == .translate ? "Translating…" : "Working…") : state.stage)
            }
        }
        let text = state.progressText
        if !text.isEmpty {
            Text(text).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary).textSelection(.enabled)
        }
        if state.fileCount > 1 {
            ProgressView(value: state.overallFraction) { Text("All files") }
        }
    }

    // MARK: - Transcript pane (filled line by line while mt.py works)

    private static let speakerColors: [Color] = [.blue, .green, .purple, .orange, .pink, .teal, .indigo, .brown]

    private func speakerColor(_ name: String) -> Color {
        let hash = name.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0x7fffffff }
        return ContentView.speakerColors[hash % ContentView.speakerColors.count]
    }

    private func visibleLines(_ state: RunState) -> [TranscriptLine] {
        let needle = flags.transcriptFilter.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return state.lines }
        return state.lines.filter { ($0.note ?? $0.text).localizedCaseInsensitiveContains(needle) || ($0.speaker ?? "").localizedCaseInsensitiveContains(needle) }
    }

    /// The languages a translation can be shown in beside the transcript: the ones the lines carry (live, or after the
    /// final translation) and the ones this run translates into (fixed when it started: a language ticked afterwards is
    /// for the next job), in that order, without repeats.
    private func translationChoices(_ state: RunState) -> [String] {
        var seen: [String] = []
        for name in state.lines.flatMap({ $0.tr.keys.sorted() }) + state.targets where !seen.contains(name) { seen.append(name) }
        return seen
    }

    /// What the live transcription is doing before its first line: the model loading, waiting for speech, or stopped
    /// (mt.py reports a failure as the `live` check; a note says why).
    private func liveWaitText(_ state: RunState, short: Bool) -> String {
        if state.checks["live"] == "failed" {
            return "Live transcription stopped (the note above says why); the recording goes on and everything is transcribed when it stops."
        }
        if state.checks["engine"] == nil {
            return short ? "Loading the speech model… The first line appears a few seconds after a stretch of speech ends."
                         : "Live transcription: loading the speech model (some seconds; longer the first time)…"
        }
        return short ? "Waiting for speech: the first line appears a few seconds after a stretch of speech ends."
                     : "Live transcription: the speech model is loaded; the first line appears once a stretch of speech is closed by a pause, a few seconds after it was said."
    }

    /// The translations shown beside the transcript, one column each: the ones picked, else every one on offer.
    private func shownTranslations(_ state: RunState) -> [String] {
        return TranslationColumns.shown(choices: translationChoices(state), chosen: flags.shownTranslations)
    }

    @ViewBuilder private func transcriptPane(_ state: RunState) -> some View {
        let choices = translationChoices(state)
        let shown = shownTranslations(state)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Transcript so far: \(state.spokenLines) line\(state.spokenLines == 1 ? "" : "s")").font(.headline)
                if !choices.isEmpty {
                    LanguageMenu(names: choices, selection: Binding(get: { shown }, set: { flags.shownTranslations = $0 }),
                                 none: "No translation shown", isPresented: $flags.showTranslationMenu,
                                 query: $flags.translationQuery, width: 200)
                }
                Spacer()
                TextField("Find", text: $flags.transcriptFilter).frame(width: 160)
                Toggle("Follow", isOn: $flags.followTranscript).toggleStyle(.checkbox)
                Button("Copy") { copyTranscript(state) }
                if state.phase == .finished, let folder = state.folder ?? runner.lastFolder, !speakerLabels(state).isEmpty {
                    Button("Name Speakers…") {
                        flags.renameFolder = folder
                        flags.renameNames = Dictionary(uniqueKeysWithValues: speakerLabels(state).map { ($0, "") })
                        flags.showRename = true
                    }
                }
            }
            if !shown.isEmpty {
                // The columns' names, over columns of equal width (the rows below have the same layout).
                HStack(spacing: 6) {
                    Text("Transcript").frame(maxWidth: .infinity, alignment: .leading)
                    ForEach(shown, id: \.self) { language in
                        Divider()
                        Text(language).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .font(.caption.bold()).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 4)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        if state.lines.isEmpty {
                            Text(liveWaitText(state, short: true))
                                .font(.caption).foregroundStyle(.secondary).padding(4)
                        }
                        ForEach(Array(visibleLines(state).enumerated()), id: \.offset) { index, line in
                            transcriptRow(line, translations: shown).id(index)
                        }
                    }
                    .padding(4)
                }
                .frame(minHeight: 160, maxHeight: 360)
                .background(Color(nsColor: .textBackgroundColor))
                // Follow: the end of the transcript stays in view as lines come, change (the final pass refines them) and get
                // their translations.
                .onChange(of: state.lines) { _ in
                    if flags.followTranscript, let last = visibleLines(state).indices.last { proxy.scrollTo(last, anchor: .bottom) }
                }
            }
        }
    }

    /// One line: time, language, who spoke and the text, then each of `translations` beside it, all in columns of equal
    /// width, so the columns line up from row to row under their names (an ellipsis while a live translation is still to come).
    @ViewBuilder private func transcriptRow(_ line: TranscriptLine, translations: [String]) -> some View {
        if let note = line.note {
            Text("# [\(Format.duration(Int(line.t)))] \(note)")
                .font(.system(.caption, design: .monospaced)).italic().foregroundStyle(.secondary)
        } else {
            HStack(alignment: .top, spacing: 6) {
                HStack(alignment: .top, spacing: 6) {
                    Text(Format.duration(Int(line.t))).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    Text(line.tag).font(.system(.caption, design: .monospaced))
                        .foregroundStyle(line.tag.hasSuffix("?]") ? Color.orange : Color.secondary)
                    if let speaker = line.speaker {
                        Text(speaker).font(.caption.bold()).foregroundStyle(speakerColor(speaker))
                    }
                    Text(line.text).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                        .foregroundStyle(line.live ? Color.secondary : Color.primary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                ForEach(translations, id: \.self) { language in
                    Divider()
                    Text(line.tr[language] ?? "…").textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                        .foregroundStyle(line.tr[language] == nil ? Color.secondary : (line.live ? Color.secondary : Color.primary))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    /// The numbered labels speaker separation gave ("Speaker 1", "Them 2"), in order of appearance.
    private func speakerLabels(_ state: RunState) -> [String] {
        var seen: [String] = []
        for line in state.lines {
            if let s = line.speaker, s.range(of: #"^\S.* \d+$"#, options: .regularExpression) != nil, !seen.contains(s) { seen.append(s) }
        }
        return seen
    }

    private var renameSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Name the speakers").font(.headline)
            Text("The names go into the stored meeting and into the text files already written in its folder.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            ForEach(flags.renameNames.keys.sorted(), id: \.self) { label in
                HStack {
                    Text(label).frame(width: 110, alignment: .trailing)
                    TextField("name", text: Binding(get: { flags.renameNames[label] ?? "" }, set: { flags.renameNames[label] = $0 })).frame(width: 240)
                }
            }
            HStack {
                Spacer()
                Button("Cancel") { flags.showRename = false }
                Button("Apply") {
                    runner.renameSpeakers(folder: flags.renameFolder, names: flags.renameNames)
                    flags.showRename = false
                }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private var tokenSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Hugging Face token").font(.headline)
            Text("Needed once, to download the pyannote models (accept their conditions on huggingface.co first). Stored like the Claude key: in your Keychain, or in a file only you can read. Leave empty and press Save to remove it.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            SecureField("hf_…", text: $flags.tokenText).frame(width: 380)
            if !flags.tokenError.isEmpty {
                Text(flags.tokenError).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { flags.tokenText = ""; flags.tokenError = ""; flags.showTokenSheet = false }
                Button("Save") {
                    switch runner.saveHFToken(flags.tokenText) {
                    case .saved: flags.tokenText = ""; flags.tokenError = ""; flags.showTokenSheet = false
                    case .failed(let message): flags.tokenError = message
                    }
                }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private func copyTranscript(_ state: RunState) {
        let shown = shownTranslations(state)
        let text = state.lines.map { line -> String in
            if let note = line.note { return "# [\(Format.duration(Int(line.t)))] \(note)" }
            let who = line.speaker.map { "<\($0)> " } ?? ""
            // The shown translations beside the text, tab-separated, so the columns paste into a spreadsheet as they are shown.
            return TranslationColumns.copyLine("[\(Format.duration(Int(line.t)))] \(line.tag) \(who)\(line.text)",
                                               translations: shown, of: line.tr)
        }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private var alertTitle: String {
        switch runner.lastFinished?.outcome {
        case .completed?: return "Done"
        case .completedWithErrors(let n)?: return "Finished with \(n) problem\(n == 1 ? "" : "s")"
        case .stopped?: return "Cancelled"
        case .failed(let code)?: return "Failed (exit \(code))"
        case nil: return ""
        }
    }

    private var alertMessage: String {
        guard let state = runner.lastFinished else { return "" }
        var text = state.kind != .record ? "\(state.outputs.count) file\(state.outputs.count == 1 ? "" : "s") written."
            : state.outputs.isEmpty ? "Nothing was recorded." : "The recording is in the data folder."
        if !state.notes.isEmpty { text += "\n" + state.notes.joined(separator: "\n") }
        if let first = state.errors.first { text += "\n\(first.message)" }
        else if case .failed? = state.outcome, let last = state.log.last { text += "\n\(last)" }
        return text
    }

    // MARK: - Helpers

    /// Adds what was chosen or dropped: files as they are, folders replaced by the audio/video (or, when
    /// translating, text) files inside them. A folder can be big or on a slow disk, so it is read on a
    /// background queue; only the results are applied on the main queue. Call it from any thread.
    private func add(_ paths: [String], translate: Bool) {
        DispatchQueue.main.async {
            let token = flags.scan.begin()
            DispatchQueue.global(qos: .userInitiated).async {
                let found = MediaFiles.expand(paths, translate: translate)
                let folders = paths.contains { path in
                    var isDir: ObjCBool = false
                    return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
                }
                DispatchQueue.main.async {
                    // "Clear" pressed while this was reading: the user threw these files away, so drop them.
                    guard flags.scan.finish(token) else { return }
                    for path in found.files where !prefs.files.contains(path) { prefs.files.append(path) }
                    if found.truncated {
                        flags.notice = "A folder holds more than \(MediaFiles.limit) matching files (or too many to read); only \(MediaFiles.limit) were added."
                    } else if folders && found.files.isEmpty {
                        flags.notice = translate ? "No text files found in that folder." : "No audio or video files found in that folder."
                    } else {
                        flags.notice = ""
                    }
                }
            }
        }
    }

    private func addFiles() {
        let translate = prefs.mode == .translate
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.prompt = "Add"
        panel.message = translate ? "Choose transcript text files or folders of them"
            : "Choose audio or video files (any format ffmpeg reads) or folders that hold them"
        guard panel.runModal() == .OK else { return }
        add(panel.urls.map { $0.path }, translate: translate)
    }

    private func handleDrop(_ providers: [NSItemProvider]) -> Bool {
        guard prefs.mode != .record else { return false }
        let translate = prefs.mode == .translate
        for item in providers {
            _ = item.loadObject(ofClass: URL.self) { url, _ in
                guard let url = url, url.isFileURL else { return }
                add([url.path], translate: translate)
            }
        }
        return true
    }

    private var keySheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Claude API key").font(.headline)
            Text("Used only to translate. It is stored in your Keychain (or, if macOS refuses, in a file only you can read) and sent only to Anthropic with the transcript text. Leave empty and press Save to remove it.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            SecureField("sk-ant-…", text: $flags.keyText).frame(width: 380)
            if !flags.keyError.isEmpty {
                Text(flags.keyError).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { flags.keyText = ""; flags.keyError = ""; flags.showKeySheet = false }
                Button("Save") { saveKey() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 440)
    }

    private func saveKey() {
        switch runner.saveAPIKey(flags.keyText) {
        case .saved(_, let note):
            flags.keyInfo = note ?? ""
            flags.keyText = ""
            flags.keyError = ""
            flags.showKeySheet = false
        case .failed(let message):
            flags.keyError = message              // the sheet stays open so the key is not lost
        }
    }
}

private func reveal(_ path: String) {
    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
}

private func pickFolder(start: String, message: String) -> String? {
    let panel = NSOpenPanel()
    panel.prompt = "Choose"
    panel.message = message
    panel.canChooseFiles = false
    panel.canChooseDirectories = true
    panel.canCreateDirectories = true
    panel.allowsMultipleSelection = false
    if !start.isEmpty { panel.directoryURL = URL(fileURLWithPath: start) }
    return panel.runModal() == .OK ? panel.url?.path : nil
}

struct PathField: View {
    let path: String
    let placeholder: String

    var body: some View {
        Text(path.isEmpty ? placeholder : path)
            .lineLimit(1)
            .truncationMode(.middle)
            .foregroundStyle(path.isEmpty ? Color.secondary : Color.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(RoundedRectangle(cornerRadius: 5).fill(Color(nsColor: .controlBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 5).stroke(Color.secondary.opacity(0.3)))
    }
}
