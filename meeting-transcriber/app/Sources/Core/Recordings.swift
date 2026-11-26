import Foundation

/// What the data folder holds: recordings that were never transcribed (a cancelled or crashed job) and stored meetings
/// (with a transcript), so the window can offer to transcribe or translate them.
enum Recordings {
    struct Entry: Equatable, Identifiable {
        enum Kind: Equatable { case single(audio: String), call(mic: String, system: String) }
        let folder: String
        let name: String             // the folder name without its time stamp ("" for none)
        let date: Date?
        let kind: Kind
        let hasTranscript: Bool
        let hasCheckpoint: Bool      // a transcription was begun: the next run continues it
        var id: String { return folder }
        var unfinished: Bool { return !hasTranscript }
        var label: String {
            let when = date.map { Entry.formatter.string(from: $0) } ?? URL(fileURLWithPath: folder).lastPathComponent
            return name.isEmpty ? when : "\(name) — \(when)"
        }
        static let formatter: DateFormatter = {
            let f = DateFormatter(); f.dateStyle = .medium; f.timeStyle = .short; return f
        }()
    }

    private static let stamp: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"; f.locale = Locale(identifier: "en_US_POSIX"); return f
    }()

    /// "20261110-103728-weekly call" → (date, "weekly call").
    static func parse(folderName: String) -> (Date?, String) {
        guard folderName.count >= 15 else { return (nil, folderName) }
        let head = String(folderName.prefix(15))
        guard let date = stamp.date(from: head) else { return (nil, folderName) }
        var name = String(folderName.dropFirst(15))
        if name.hasPrefix("-") { name.removeFirst() }
        return (date, name)
    }

    /// Every folder in `data` that holds a recording, newest first.
    static func scan(data: URL, fileManager: FileManager = .default) -> [Entry] {
        guard let names = try? fileManager.contentsOfDirectory(atPath: data.path) else { return [] }
        var entries: [Entry] = []
        for folderName in names where !folderName.hasPrefix(".") {
            let folder = data.appendingPathComponent(folderName, isDirectory: true)
            func has(_ file: String) -> Bool { return fileManager.fileExists(atPath: folder.appendingPathComponent(file).path) }
            let kind: Entry.Kind
            if has("mic.wav") && has("system.wav") {
                kind = .call(mic: folder.appendingPathComponent("mic.wav").path, system: folder.appendingPathComponent("system.wav").path)
            } else if has("audio.wav") {
                kind = .single(audio: folder.appendingPathComponent("audio.wav").path)
            } else if has("transcript.json") {
                kind = .single(audio: "")                   // an imported file's transcript: nothing to transcribe again here
            } else {
                continue
            }
            let (date, name) = parse(folderName: folderName)
            let checkpoint = (try? fileManager.contentsOfDirectory(atPath: folder.path))?.contains { $0.hasPrefix("partial") && $0.hasSuffix(".json") } ?? false
            entries.append(Entry(folder: folder.path, name: name, date: date, kind: kind, hasTranscript: has("transcript.json"), hasCheckpoint: checkpoint))
        }
        return entries.sorted { ($0.date ?? .distantPast) > ($1.date ?? .distantPast) }
    }
}
