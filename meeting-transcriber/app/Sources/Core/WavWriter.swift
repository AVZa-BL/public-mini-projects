import Foundation

/// A mono 16 kHz, 16-bit PCM WAV file written as samples arrive (the format mt.py reads and Whisper wants). The sizes in
/// the header are patched on `sync()` and `close()`, so a recording that is cut short is still a valid file.
final class WavWriter {
    static let sampleRate = 16000

    private let handle: FileHandle
    private let lock = NSLock()
    private var samples = 0
    private var failed = false

    /// Called once, from the thread that was writing, when a write fails (disk full, volume gone …). From then on nothing
    /// more is written and `sampleCount` stops growing, so it only ever counts samples that reached the file.
    var onFailure: ((Error) -> Void)?

    init(url: URL) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: WavWriter.header(dataBytes: 0))
    }

    var sampleCount: Int {
        lock.lock(); defer { lock.unlock() }
        return samples
    }

    var seconds: Double { return Double(sampleCount) / Double(WavWriter.sampleRate) }

    func append(_ chunk: [Int16]) {
        guard !chunk.isEmpty else { return }
        lock.lock()
        if failed { lock.unlock(); return }
        var little = chunk.map { $0.littleEndian }
        let data = little.withUnsafeMutableBytes { Data($0) }
        var problem: Error?
        do {
            try handle.write(contentsOf: data)
            samples += chunk.count
        } catch {
            failed = true
            problem = error
        }
        lock.unlock()
        if let problem = problem { onFailure?(problem) }       // outside the lock: the handler may look at the writer
    }

    var hasFailed: Bool {
        lock.lock(); defer { lock.unlock() }
        return failed
    }

    /// Silence, so that a track that started late lines up with the other one. Written in slices of one second: a long
    /// gap (the Mac slept for an hour) must not be built as one array of hundreds of megabytes.
    func appendSilence(seconds: Double) {
        var left = Int((max(0, seconds) * Double(WavWriter.sampleRate)).rounded())
        let slice = [Int16](repeating: 0, count: WavWriter.sampleRate)
        while left > 0 {
            let n = min(left, slice.count)
            append(n == slice.count ? slice : Array(slice.prefix(n)))
            left -= n
        }
    }

    /// Writes the sizes into the header (and leaves the write position at the end).
    func sync() {
        lock.lock(); defer { lock.unlock() }
        patchHeader()
    }

    func close() {
        lock.lock(); defer { lock.unlock() }
        patchHeader()
        try? handle.close()
    }

    private func patchHeader() {
        guard (try? handle.seek(toOffset: 0)) != nil else { return }
        try? handle.write(contentsOf: WavWriter.header(dataBytes: samples * 2))
        _ = try? handle.seekToEnd()
    }

    /// The 44-byte header of a PCM WAV file.
    static func header(dataBytes: Int) -> Data {
        func le32(_ v: Int) -> [UInt8] { return (0..<4).map { UInt8((v >> (8 * $0)) & 0xFF) } }
        func le16(_ v: Int) -> [UInt8] { return (0..<2).map { UInt8((v >> (8 * $0)) & 0xFF) } }
        var bytes: [UInt8] = []
        bytes += Array("RIFF".utf8) + le32(36 + dataBytes) + Array("WAVE".utf8)
        bytes += Array("fmt ".utf8) + le32(16) + le16(1) + le16(1) + le32(sampleRate) + le32(sampleRate * 2) + le16(2) + le16(16)
        bytes += Array("data".utf8) + le32(dataBytes)
        return Data(bytes)
    }
}
