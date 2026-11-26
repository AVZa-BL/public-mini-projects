import Foundation

/// The processing jobs (transcriptions, translations, the engine setup) in the order they were asked for. One runs at a
/// time; a recording is not a job here (it has its own slot in the window), so a new interview can be recorded while
/// the last one is still being transcribed.
struct QueuedJob: Equatable, Identifiable {
    let id: UUID
    let label: String
    let kind: RunState.Kind
    var folder: String? = nil        // the data folder the job works in (so the window does not offer it a second time)
}

struct JobQueue: Equatable {
    private(set) var pending: [QueuedJob] = []
    private(set) var current: QueuedJob? = nil

    var isIdle: Bool { return current == nil && pending.isEmpty }
    var count: Int { return pending.count + (current == nil ? 0 : 1) }
    /// The data folders of the running and waiting jobs.
    var folders: Set<String> {
        var set = Set(pending.compactMap { $0.folder })
        if let f = current?.folder { set.insert(f) }
        return set
    }

    mutating func add(_ job: QueuedJob) {
        pending.append(job)
    }

    /// Moves the first waiting job into the running slot (when that is free) and returns it.
    mutating func startNext() -> QueuedJob? {
        guard current == nil, !pending.isEmpty else { return nil }
        current = pending.removeFirst()
        return current
    }

    mutating func finishCurrent() {
        current = nil
    }

    /// Removes a waiting job; the running one is not touched (it is cancelled, not removed).
    @discardableResult
    mutating func remove(id: UUID) -> Bool {
        let before = pending.count
        pending.removeAll { $0.id == id }
        return pending.count != before
    }

    mutating func removeAll() {
        pending.removeAll()
    }

    /// Moves a waiting job one place up (earlier) or down.
    mutating func move(id: UUID, up: Bool) {
        guard let i = pending.firstIndex(where: { $0.id == id }) else { return }
        let j = up ? i - 1 : i + 1
        guard pending.indices.contains(j) else { return }
        pending.swapAt(i, j)
    }
}
