import Foundation

/// A fixed-capacity buffer that keeps the newest `capacity` elements: appending to a full buffer drops the oldest one.
/// The telemetry keeps its series in these, so a session of any length holds a bounded amount of memory.
struct RingBuffer<T>: Sequence {
    private var storage: [T] = []
    private var head = 0                 // once full: the index of the oldest element (the next one to overwrite)
    let capacity: Int

    init(capacity: Int) {
        self.capacity = Swift.max(1, capacity)
        storage.reserveCapacity(Swift.min(self.capacity, 1024))
    }

    mutating func append(_ x: T) {
        if storage.count < capacity {
            storage.append(x)
        } else {
            storage[head] = x
            head = (head + 1) % capacity
        }
    }

    mutating func removeAll() {
        storage.removeAll(keepingCapacity: true)
        head = 0
    }

    var count: Int { return storage.count }
    var isEmpty: Bool { return storage.isEmpty }

    /// The newest element.
    var last: T? {
        guard !storage.isEmpty else { return nil }
        return storage.count < capacity ? storage[storage.count - 1] : storage[(head + capacity - 1) % capacity]
    }

    /// The newest element that satisfies `predicate`, searched from the newest back, without copying the buffer.
    func newest(where predicate: (T) -> Bool) -> T? {
        let n = storage.count
        for k in 0..<n {
            let x = storage[n < capacity ? n - 1 - k : (head + capacity - 1 - k) % capacity]
            if predicate(x) { return x }
        }
        return nil
    }

    /// Oldest first.
    var elements: [T] {
        guard storage.count == capacity, head != 0 else { return storage }
        return Array(storage[head...]) + Array(storage[..<head])
    }

    func makeIterator() -> IndexingIterator<[T]> { return elements.makeIterator() }

    var underestimatedCount: Int { return storage.count }
}

extension RingBuffer: Equatable where T: Equatable {
    /// Same capacity and the same elements in the same order (however they lie in the storage).
    static func == (lhs: RingBuffer<T>, rhs: RingBuffer<T>) -> Bool {
        return lhs.capacity == rhs.capacity && lhs.count == rhs.count && lhs.elements == rhs.elements
    }
}
