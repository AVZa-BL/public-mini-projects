import SwiftUI
import AppKit
import Combine

// The window's telemetry objects (C5): each is its own ObservableObject, so a STAT (up to a few a second) or a LEVEL
// (once a second) re-evaluates only the views that show it, never ContentView or the transcript. Plain ObservableObjects
// (no @Observable, @State or @Environment): the Command Line Tools cannot expand SwiftUI's macro-backed wrappers.

/// Which session a telemetry view shows: the recording, the processing job, or (nothing runs) only the machine.
enum TelemetrySource: Equatable {
    case recording, processing, machine
}

/// The span the dashboard's charts show.
enum TelemetryWindow: String, CaseIterable, Identifiable {
    case twoMinutes = "2min", tenMinutes = "10min", session

    var id: String { return rawValue }

    var label: String {
        switch self {
        case .twoMinutes: return "2 min"
        case .tenMinutes: return "10 min"
        case .session: return "Session"
        }
    }

    /// nil: the whole session the ring buffers hold.
    var seconds: Double? {
        switch self {
        case .twoMinutes: return 120
        case .tenMinutes: return 600
        case .session: return nil
        }
    }
}

/// mt.py's telemetry (@@STAT) and the app's own samples, per session, in bounded buffers (Core `TelemetryModel`).
/// It takes every event at once but tells its views at most every 250 ms (a coalescing timer on the main queue).
/// All of it runs on the main thread.
final class TelemetryStore: ObservableObject {
    private(set) var recording = TelemetryModel()
    private(set) var processing = TelemetryModel()
    /// The app's samples since the launch: the machine cards while nothing runs.
    private(set) var machine = TelemetryModel()
    /// The latest list of the models in the Hugging Face cache (name, bytes), read by the sampler now and then.
    private(set) var modelsOnDisk: [(name: String, bytes: Int64)] = []

    private var recordingStart = Date()
    private var processingStart = Date()
    private var recordingEnd: Date? = nil        // set when the session ends: its charts stop at its end
    private var processingEnd: Date? = nil
    private let launch = Date()
    private var publishScheduled = false
    private var lastPublish: TimeInterval = 0
    static let publishInterval: TimeInterval = 0.25

    /// The STATs of one batch of mt.py's output, into the recording's or the job's telemetry.
    func ingest(_ stats: [Stat], toRecording: Bool) {
        guard !stats.isEmpty else { return }
        let at = Date().timeIntervalSince(toRecording ? recordingStart : processingStart)
        for stat in stats {
            if toRecording { recording.ingest(stat, at: at) } else { processing.ingest(stat, at: at) }
        }
        schedulePublish()
    }

    /// One sample of the app and the machine alone (no session, no mt.py figures): into the machine's telemetry.
    func ingest(_ sample: AppSample) {
        ingest(SamplerReading(machine: sample, recordingChild: nil, processingChild: nil), recordingActive: false, processingActive: false)
    }

    /// One sample of the sampler: the machine's into the machine model, and with each running session's own mt.py
    /// figures into that session (its x is the session's own clock).
    func ingest(_ reading: SamplerReading, recordingActive: Bool, processingActive: Bool) {
        let now = Date()
        var base = reading.machine
        base.at = now.timeIntervalSince(launch)
        machine.ingest(base)
        if recordingActive {
            var s = reading.machine
            s.at = now.timeIntervalSince(recordingStart)
            s.childCPU = reading.recordingChild?.cpu
            s.childFootprintMB = reading.recordingChild?.footprintMB
            recording.ingest(s)
        }
        if processingActive {
            var s = reading.machine
            s.at = now.timeIntervalSince(processingStart)
            s.childCPU = reading.processingChild?.cpu
            s.childFootprintMB = reading.processingChild?.footprintMB
            processing.ingest(s)
        }
        if let models = reading.modelsOnDisk { modelsOnDisk = models }
        schedulePublish()
    }

    /// A new recording (or job) starts with empty telemetry and its own clock.
    func reset(recording isRecording: Bool) {
        if isRecording {
            recording = TelemetryModel()
            recordingStart = Date()
            recordingEnd = nil
        } else {
            processing = TelemetryModel()
            processingStart = Date()
            processingEnd = nil
        }
        publishNow()
    }

    /// The session ended: its clock stops, so its charts keep showing its end instead of sliding into empty time.
    func end(recording isRecording: Bool) {
        if isRecording { recordingEnd = recordingEnd ?? Date() } else { processingEnd = processingEnd ?? Date() }
        schedulePublish()
    }

    func model(_ source: TelemetrySource) -> TelemetryModel {
        switch source {
        case .recording: return recording
        case .processing: return processing
        case .machine: return machine
        }
    }

    /// The session's clock now (seconds since its start): the right end of its charts.
    func now(_ source: TelemetrySource) -> Double {
        switch source {
        case .recording: return (recordingEnd ?? Date()).timeIntervalSince(recordingStart)
        case .processing: return (processingEnd ?? Date()).timeIntervalSince(processingStart)
        case .machine: return Date().timeIntervalSince(launch)
        }
    }

    private func schedulePublish() {
        guard !publishScheduled else { return }
        publishScheduled = true
        let delay = max(0, lastPublish + TelemetryStore.publishInterval - ProcessInfo.processInfo.systemUptime)
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in self?.publishNow() }
    }

    private func publishNow() {
        publishScheduled = false
        lastPublish = ProcessInfo.processInfo.systemUptime
        objectWillChange.send()
    }
}

/// The microphone's level (LEVEL, or the call recorder's own), once a second: only the meter views observe it.
final class LevelMeter: ObservableObject {
    @Published private(set) var level: Double = 0

    func set(_ value: Double) {
        let v = min(1, max(0, value.isFinite ? value : 0))
        if v != level { level = v }
    }
}

/// Reduce Motion and Increase Contrast, read from NSWorkspace and kept current by its notification (no @Environment,
/// which the app's CI bans in Sources/App).
final class DisplayPrefs: ObservableObject {
    static let shared = DisplayPrefs()

    @Published var reduceMotion: Bool
    @Published var increaseContrast: Bool
    private var observer: NSObjectProtocol?

    init() {
        let workspace = NSWorkspace.shared
        reduceMotion = workspace.accessibilityDisplayShouldReduceMotion
        increaseContrast = workspace.accessibilityDisplayShouldIncreaseContrast
        observer = workspace.notificationCenter.addObserver(forName: NSWorkspace.accessibilityDisplayOptionsDidChangeNotification,
                                                            object: nil, queue: .main) { [weak self] _ in self?.refresh() }
    }

    deinit {
        if let observer = observer { NSWorkspace.shared.notificationCenter.removeObserver(observer) }
    }

    private func refresh() {
        let workspace = NSWorkspace.shared
        if reduceMotion != workspace.accessibilityDisplayShouldReduceMotion { reduceMotion = workspace.accessibilityDisplayShouldReduceMotion }
        if increaseContrast != workspace.accessibilityDisplayShouldIncreaseContrast { increaseContrast = workspace.accessibilityDisplayShouldIncreaseContrast }
    }

    /// The width of a chart's lines: 2 pt, 2.5 pt with Increase Contrast.
    var lineWidth: CGFloat { return increaseContrast ? 2.5 : 2 }
    /// The opacity of the charts' grid lines.
    var gridOpacity: Double { return increaseContrast ? 1 : 0.6 }
}

/// What the window shows, remembered between launches (UserDefaults, as Prefs does; no @AppStorage). One for the app:
/// the View menu's commands and every window share it.
final class LayoutPrefs: ObservableObject {
    static let shared = LayoutPrefs()

    /// The Telemetry column (View ▸ Show Telemetry, ⌥⌘T). Shown by default.
    @Published var showTelemetry: Bool { didSet { UserDefaults.standard.set(showTelemetry, forKey: "showTelemetry") } }
    /// The Controls column (View ▸ Show Settings, ⌥⌘S).
    @Published var showControls: Bool { didSet { UserDefaults.standard.set(showControls, forKey: "showControls") } }
    @Published var window: TelemetryWindow { didSet { UserDefaults.standard.set(window.rawValue, forKey: "telemetryWindow") } }
    /// The time and the model of each translation, under it in the transcript.
    @Published var showTimings: Bool { didSet { UserDefaults.standard.set(showTimings, forKey: "showTranslationTimings") } }
    /// While a recording and a job run together: the dashboard shows the job instead of the recording.
    @Published var dashboardShowsJob = false
    /// A card the status bar asked to bring into view (its id), cleared once it has been scrolled to.
    @Published var focusCard: String? = nil
    /// Counts the status bar's requests, so that a second click on the same part scrolls again.
    @Published var focusRequest = 0

    init() {
        let d = UserDefaults.standard
        showTelemetry = d.object(forKey: "showTelemetry") == nil ? true : d.bool(forKey: "showTelemetry")
        showControls = d.object(forKey: "showControls") == nil ? true : d.bool(forKey: "showControls")
        window = TelemetryWindow(rawValue: d.string(forKey: "telemetryWindow") ?? "") ?? .tenMinutes
        showTimings = d.bool(forKey: "showTranslationTimings")
    }

    /// The status bar's parts: show the Telemetry column and bring the card into view.
    func open(card: String) {
        if !showTelemetry { showTelemetry = true }
        focusCard = card
        focusRequest &+= 1
    }
}

/// The window's colours: semantic NSColors, so light and dark mode and Increase Contrast follow by themselves. State
/// is never shown by colour alone: every coloured state also has a symbol and a word.
enum Palette {
    static let speech = Color(nsColor: .systemBlue)
    static let translation = Color(nsColor: .systemPurple)
    static let memory = Color(nsColor: .systemTeal)
    static let cpu = Color(nsColor: .systemOrange)
    static let gpu = Color(nsColor: .systemPink)
    static let ok = Color(nsColor: .systemGreen)
    /// Fills; text in the warning state uses `warningText` (contrast in light mode).
    static let warningFill = Color(nsColor: .systemYellow)
    static let warningText = Color(nsColor: .systemOrange)
    static let bad = Color(nsColor: .systemRed)
    static let grid = Color(nsColor: .separatorColor)

    static let speakers: [Color] = [.blue, .green, .purple, .orange, .pink, .teal, .indigo, .brown]

    /// The same colour for the same speaker, line after line.
    static func speaker(_ name: String) -> Color {
        let hash = name.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) & 0x7fffffff }
        return speakers[hash % speakers.count]
    }

    /// The colour of a lag level, for a fill or a line.
    static func fill(_ level: LagLevel) -> Color {
        switch level {
        case .ok: return ok
        case .slow: return warningFill
        case .behind: return bad
        }
    }

    /// The colour of a lag level for text and symbols.
    static func text(_ level: LagLevel) -> Color {
        switch level {
        case .ok: return ok
        case .slow: return warningText
        case .behind: return bad
        }
    }

    /// The SF Symbol of a lag level (ok, slow, behind).
    static func symbol(_ level: LagLevel) -> String {
        switch level {
        case .ok: return "checkmark.circle"
        case .slow: return "exclamationmark.triangle"
        case .behind: return "xmark.octagon"
        }
    }
}
