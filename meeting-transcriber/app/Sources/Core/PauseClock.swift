import Foundation

/// The clock of a recording that can be paused: wall time minus the time spent paused is the recorded time (the
/// engine counts samples; the window shows this), and a pause is told as a timed note placed where it happened.
struct PauseClock: Equatable {
    private(set) var pausedTotal: TimeInterval = 0
    private(set) var pausedSince: Date?

    var isPaused: Bool { return pausedSince != nil }

    mutating func pause(at now: Date = Date()) {
        if pausedSince == nil { pausedSince = now }
    }

    /// Ends the pause and returns how long it was (0 when not paused).
    @discardableResult
    mutating func resume(at now: Date = Date()) -> TimeInterval {
        guard let since = pausedSince else { return 0 }
        let length = max(0, now.timeIntervalSince(since))
        pausedTotal += length
        pausedSince = nil
        return length
    }

    /// How long the pause going on has lasted (0 when not paused); the clock stays paused.
    func currentPause(at now: Date = Date()) -> TimeInterval {
        return pausedSince.map { max(0, now.timeIntervalSince($0)) } ?? 0
    }

    /// Seconds of recording so far: the wall time since `started` without the pauses (the one going on included).
    func recorded(since started: Date, at now: Date = Date()) -> TimeInterval {
        let paused = pausedTotal + (pausedSince.map { max(0, now.timeIntervalSince($0)) } ?? 0)
        return max(0, now.timeIntervalSince(started) - paused)
    }

    /// The engine's wording for a pause: "[hh:mm:ss] Paused here for 3 min 5 s" (a timed note mt.py places at that time).
    static func note(recordedSeconds: Double, pausedFor: TimeInterval) -> String {
        let paused = Int(pausedFor.rounded())
        let length = paused >= 60 ? "\(paused / 60) min \(paused % 60) s" : "\(paused) s"
        let t = Int(recordedSeconds)
        return String(format: "[%02d:%02d:%02d] Paused here for %@", t / 3600, (t % 3600) / 60, t % 60, length)
    }
}
