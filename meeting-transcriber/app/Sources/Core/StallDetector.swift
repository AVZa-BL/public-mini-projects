import Foundation

/// Notices a track that has stopped delivering sound while the recording goes on. Silence in the room still delivers
/// buffers; a device that went away, or a capture that died without saying so, delivers none.
struct StallDetector {
    private var last: [String: Date] = [:]
    private var reported: Set<String> = []

    /// A buffer arrived for `track`.
    mutating func saw(_ track: String, at now: Date = Date()) {
        last[track] = now
        reported.remove(track)
    }

    /// The tracks that delivered nothing for `threshold` seconds while the recording has been going for at least that long;
    /// each is reported once until it delivers again.
    mutating func stalled(at now: Date = Date(), since started: Date, threshold: TimeInterval = 10) -> [String] {
        guard now.timeIntervalSince(started) >= threshold else { return [] }
        var out: [String] = []
        for (track, seen) in last.sorted(by: { $0.key < $1.key }) where now.timeIntervalSince(seen) >= threshold && !reported.contains(track) {
            reported.insert(track)
            out.append(track)
        }
        return out
    }

    mutating func register(_ track: String, at now: Date = Date()) {
        if last[track] == nil { last[track] = now }
    }
}
