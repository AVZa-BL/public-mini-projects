import Foundation

/// Bookkeeping for folder scans that run in the background while the user keeps working. Each scan takes a token
/// when it starts; "Clear" invalidates every scan that is still running, so a late result is dropped instead of
/// bringing back files the user threw away; and a job cannot start while a scan is still pending.
struct ScanTracker: Equatable {
    private(set) var generation = 0
    private(set) var pending = 0

    var busy: Bool { return pending > 0 }

    /// A scan starts; keep the token and hand it to `finish`.
    mutating func begin() -> Int {
        pending += 1
        return generation
    }

    /// A scan ends. True when its result is still wanted (nothing invalidated it meanwhile).
    mutating func finish(_ token: Int) -> Bool {
        pending = max(0, pending - 1)
        return token == generation
    }

    /// Scans that are running now will have their results dropped.
    mutating func invalidate() {
        generation += 1
    }
}
