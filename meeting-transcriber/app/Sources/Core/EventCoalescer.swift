import Foundation

/// Groups mt.py's events into batches, at most one batch per `interval`, so a burst (the final pass sends hundreds of
/// LINE events in a second) is one publish of the window's state instead of hundreds. Nothing is lost and the order is
/// kept: what is not due yet is held, and handed out with the next batch, at the trailing flush, or by `drain()`.
/// The first event after a quiet `interval` goes out at once (a live line is shown without delay).
struct EventCoalescer {
    let interval: TimeInterval
    private var pending: [MTEvent] = []
    private var lastFlush: TimeInterval? = nil

    init(interval: TimeInterval = 0.1) {
        self.interval = max(0, interval)
    }

    /// The events read so far plus `events`, when a batch is due (the last one went out `interval` or more ago); nil
    /// while they are held. Calling it with no events at the time `nextDue` gave is the trailing flush.
    mutating func add(_ events: [MTEvent], now: TimeInterval) -> [MTEvent]? {
        pending.append(contentsOf: events)
        guard !pending.isEmpty else { return nil }
        if let last = lastFlush, now - last < interval - 1e-6, now >= last { return nil }   // 1e-6: a timer set to nextDue
        lastFlush = now
        let batch = pending
        pending.removeAll(keepingCapacity: true)
        return batch
    }

    /// Everything held, at the end of the output (the process exited): nothing waits for a timer any more.
    mutating func drain() -> [MTEvent] {
        let batch = pending
        pending.removeAll()
        return batch
    }

    /// Seconds from `now` until the held events are due (0: due now), or nil when nothing is held: the delay of the
    /// trailing flush timer, which then calls `add([], now:)`.
    func nextDue(now: TimeInterval) -> TimeInterval? {
        guard !pending.isEmpty else { return nil }
        guard let last = lastFlush else { return 0 }
        return max(0, last + interval - now)
    }

    var held: Int { return pending.count }
}

/// Where each event goes: the transcript's state (`RunState`), the telemetry (`Stat`s, to their own store), or the level
/// meter. A STAT or a LEVEL never changes `RunState`, so a 1-10 Hz telemetry update never re-evaluates the transcript.
enum EventRouting {
    /// The order is kept within each part; of several levels the last one wins; ignored STATs are dropped.
    static func split(_ events: [MTEvent]) -> (state: [MTEvent], stats: [Stat], level: Double?) {
        var state: [MTEvent] = []
        var stats: [Stat] = []
        var level: Double? = nil
        state.reserveCapacity(events.count)
        for event in events {
            switch event {
            case .stat(let stat): if stat != .ignored { stats.append(stat) }
            case .level(let value): level = value
            default: state.append(event)
            }
        }
        return (state, stats, level)
    }
}
