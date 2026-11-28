import Foundation

/// What a run is doing, built from the events of mt.py.
struct RunState: Equatable {
    enum Kind: Equatable { case transcribe, translate, record, setup }
    enum Phase: Equatable { case starting, recording, working, finished }
    enum Outcome: Equatable {
        case completed
        case completedWithErrors(Int)
        case stopped
        case failed(Int32)
    }
    struct FileError: Equatable {
        let file: String
        let message: String
    }

    var kind: Kind
    var phase: Phase = .starting
    var fileIndex = 0
    var fileCount = 0
    var currentFile = ""
    var chunk = 0
    var chunks = 0
    var stage = ""
    var outputs: [String] = []
    var errors: [FileError] = []
    var checks: [String: String] = [:]
    var log: [String] = []
    var notes: [String] = []      // notices worth showing with the result (the recording stops soon, a sleep, a lost device)
    /// The transcript so far, as mt.py reports it (reset per file). Setting it (a job seeded with a recording's lines)
    /// keeps the lines' ids when they are unique and set, and rebuilds the caches below (O(n)); `apply` updates them in O(1)
    /// for the usual events (a merge that takes a translation from the first line holding it looks for the next holder).
    var lines: [TranscriptLine] {
        get { return storedLines }
        set { replaceLines(newValue) }
    }
    var position: Double = 0      // seconds of audio done in the current track
    var duration: Double = 0      // seconds of audio in the current track
    var eta: Int? = nil           // seconds left, estimated by mt.py
    var speed: Double = 0         // audio seconds per wall-clock second, smoothed
    var level: Double = 0         // the microphone's loudness while recording, 0…1
    var folder: String? = nil     // the data folder a recording writes into
    var targets: [String] = []    // the languages this run translates into (live, or into files): what the pane can show
    var positionAt: Date? = nil
    var outcome: Outcome? = nil

    // Incremental caches of `lines` (C5): the window reads these instead of walking every line on every publish.
    /// +1 whenever `lines` changes (and only then): the transcript pane compares this Int instead of the lines.
    private(set) var revision = 0
    private(set) var spokenCount = 0                        // lines that are not notes
    private(set) var languageCounts: [String: Int] = [:]    // spoken lines per language code
    /// The languages the lines carry translations in, in order of first appearance (each line's sorted), each once.
    private(set) var translationKeys: [String] = []
    /// The numbered labels speaker separation gave ("Speaker 1", "Them 2"), in order of appearance.
    private(set) var speakerLabelList: [String] = []

    private struct LineKey: Hashable {
        let t: Double
        let isNote: Bool
    }
    private struct FirstSeen: Equatable {
        var count: Int        // lines holding it
        var first: Int        // the index of the first of them
    }
    private var storedLines: [TranscriptLine] = []
    private var index: [LineKey: [Int]] = [:]               // the lines at one time (usually 1-2): the merge's candidates
    private var translationSeen: [String: FirstSeen] = [:]
    private var speakerSeen: [String: FirstSeen] = [:]
    private var nextID = 1

    init(kind: Kind) {
        self.kind = kind
        if kind == .record { phase = .recording }
    }

    static let logLimit = 300

    /// 0…1 for the file being transcribed.
    var fileFraction: Double { return chunks > 0 ? min(1, Double(chunk) / Double(chunks)) : 0 }

    /// 0…1 over all files.
    var overallFraction: Double {
        guard fileCount > 0 else { return 0 }
        let finished = Double(max(0, fileIndex - 1))
        return min(1, (finished + fileFraction) / Double(fileCount))
    }

    mutating func apply(_ event: MTEvent, at now: Date = Date()) {
        switch event {
        case .file(let index, let count, let path):
            phase = .working
            fileIndex = index; fileCount = count; currentFile = path
            chunk = 0; chunks = 0
            stage = ""
            // The lines are reset per file of a batch. A one-file job keeps them: the lines transcribed live that seeded
            // the job, and a call's merged conversation, which mt.py sends before this event.
            if count > 1 { clearLines() }
            position = 0; duration = 0; eta = nil; speed = 0; positionAt = nil
        case .progress(let done, let total):
            phase = .working
            chunk = done; chunks = total       // the stage stays: it says what the bar counts (chunks, translation batches)
        case .position(let done, let total):
            if phase != .recording { phase = .working }      // live transcription reports positions while recording
            if let at = positionAt, done > position, now > at {
                let s = (done - position) / now.timeIntervalSince(at)
                speed = speed == 0 ? s : 0.7 * speed + 0.3 * s
            }
            position = done; duration = total; positionAt = now
        case .eta(let seconds):
            eta = seconds
        case .line(let line):
            // The final pass repeats the lines transcribed live (refined, with speaker labels): same time, same place.
            // Two live lines at the same time from different speakers (both sides of a call) are both kept.
            if let i = mergeIndex(for: line) {
                let old = storedLines[i]
                var merged = line
                merged.id = old.id
                // A line sent again without its translations keeps the ones shown, but only for the same text: a line the
                // final pass decoded anew (other languages, another granularity) must not show the translation of the old one.
                // So do their times, their models and the targets left for the end. The times and the targets left
                // are merged per target, not replaced: mt.py sends tr_ms (tr_skip) for the one target just translated
                // (left), while tr and tr_by carry every target so far.
                if merged.text == old.text {
                    if merged.tr.isEmpty { merged.tr = old.tr }
                    merged.trMs = old.trMs.merging(merged.trMs) { $1 }
                    if merged.trBy.isEmpty { merged.trBy = old.trBy }
                    merged.trSkipped.formUnion(old.trSkipped)
                    merged.trSkipped.subtract(merged.tr.keys)
                }
                if merged != old { replaceLine(at: i, with: merged) }
            } else {
                appendLine(line)
            }
        case .output(let path):
            outputs.append(path)
        case .failure(let file, let message):
            errors.append(FileError(file: file, message: message))
        case .done:
            break
        case .check(let name, let status):
            checks[name] = status
        case .stage(let text):
            stage = text
            chunk = 0; chunks = 0; eta = nil                 // a new stage counts something else
            if phase == .recording { phase = .working }      // mt.py announces a stage once the recording is saved
        case .ready:
            stage = "Ready"
        case .clear:
            clearLines()
        case .note(let text):
            notes.append(text)
            log.append(text)
        case .level(let value):
            level = value
        case .stat:
            break                                            // telemetry goes to its own store (EventRouting)
        case .folder(let path):
            folder = path
        case .text(let line):
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // mt.py counts recorded minutes ("  12 min") and redraws chunk counters; both are the window's job.
            if trimmed.isEmpty || trimmed.hasSuffix(" min") || trimmed.hasSuffix("chunks") { return }
            log.append(trimmed)
            if log.count > RunState.logLimit { log.removeFirst(log.count - RunState.logLimit) }
        }
    }

    /// Spoken lines only (no notes).
    var spokenLines: Int { return spokenCount }

    /// Languages of the lines so far as (code, percent), most frequent first.
    var languageShares: [(String, Int)] {
        let total = spokenCount
        guard total > 0 else { return [] }
        return languageCounts.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
            .map { ($0.key, Int((Double($0.value) * 100 / Double(total)).rounded())) }
    }

    /// The languages a translation can be shown in beside the transcript: the ones the lines carry (live, or after the
    /// final translation) and the ones this run translates into (fixed when it started: a language ticked afterwards is
    /// for the next job), in that order, without repeats.
    var translationChoices: [String] {
        var seen = translationKeys
        for name in targets where !seen.contains(name) { seen.append(name) }
        return seen
    }

    // MARK: the lines and their caches

    /// The line `line` refines: the last one at the same time and of the same kind whose speaker is the same, or a live
    /// one refined by a final line where one of the two has no speaker. The same as `lines.lastIndex(where:)` with that
    /// predicate, among the few lines at that time.
    private func mergeIndex(for line: TranscriptLine) -> Int? {
        guard let candidates = index[LineKey(t: line.t, isNote: line.isNote)] else { return nil }
        for i in candidates.reversed() {
            let other = storedLines[i]
            if other.speaker == line.speaker || (other.live && !line.live && (other.speaker == nil || line.speaker == nil)) { return i }
        }
        return nil
    }

    private static let speakerLabelPattern = try! NSRegularExpression(pattern: #"^\S.* \d+$"#)

    private static func isSpeakerLabel(_ s: String) -> Bool {
        return speakerLabelPattern.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
    }

    private mutating func appendLine(_ line: TranscriptLine) {
        var line = line
        line.id = nextID
        nextID += 1
        storedLines.append(line)
        let i = storedLines.count - 1
        index[LineKey(t: line.t, isNote: line.isNote), default: []].append(i)
        count(line, at: i, by: 1)
        revision += 1
    }

    private mutating func replaceLine(at i: Int, with line: TranscriptLine) {
        let old = storedLines[i]
        storedLines[i] = line              // the same time and kind: the index stays
        count(old, at: i, by: -1)
        count(line, at: i, by: 1)
        revision += 1
    }

    private mutating func clearLines() {
        nextID = 1                         // ids restart after CLEAR
        guard !storedLines.isEmpty else { return }
        storedLines = []
        resetCaches()
        revision += 1
    }

    private mutating func replaceLines(_ newLines: [TranscriptLine]) {
        var lines = newLines
        let ids = lines.map { $0.id }
        if ids.contains(0) || Set(ids).count != ids.count {
            for k in lines.indices { lines[k].id = k + 1 }
        }
        storedLines = lines
        nextID = (lines.map { $0.id }.max() ?? 0) + 1
        resetCaches()
        for (i, line) in lines.enumerated() {
            index[LineKey(t: line.t, isNote: line.isNote), default: []].append(i)
            count(line, at: i, by: 1)
        }
        revision += 1
    }

    private mutating func resetCaches() {
        index = [:]
        spokenCount = 0
        languageCounts = [:]
        translationSeen = [:]
        speakerSeen = [:]
        translationKeys = []
        speakerLabelList = []
    }

    /// Adds (`by` 1) or removes (-1) line `i`'s share of the counts.
    private mutating func count(_ line: TranscriptLine, at i: Int, by delta: Int) {
        if !line.isNote {
            spokenCount += delta
            let n = (languageCounts[line.lang] ?? 0) + delta
            languageCounts[line.lang] = n > 0 ? n : nil
        }
        var keysChanged = false
        for key in line.tr.keys {
            if seen(&translationSeen, key, at: i, by: delta, holds: { $0.tr[key] != nil }) { keysChanged = true }
        }
        if keysChanged { translationKeys = RunState.ordered(translationSeen) }
        if let speaker = line.speaker, RunState.isSpeakerLabel(speaker),
           seen(&speakerSeen, speaker, at: i, by: delta, holds: { $0.speaker == speaker }) {
            speakerLabelList = RunState.ordered(speakerSeen)
        }
    }

    /// Counts `name` in or out of line `i`; true when the order of first appearance may have changed.
    private func seen(_ table: inout [String: FirstSeen], _ name: String, at i: Int, by delta: Int,
                      holds: (TranscriptLine) -> Bool) -> Bool {
        if delta > 0 {
            if var entry = table[name] {
                entry.count += 1
                let earlier = i < entry.first
                if earlier { entry.first = i }
                table[name] = entry
                return earlier
            }
            table[name] = FirstSeen(count: 1, first: i)
            return true
        }
        guard var entry = table[name] else { return false }
        entry.count -= 1
        if entry.count <= 0 { table[name] = nil; return true }
        guard entry.first == i else { table[name] = entry; return false }
        // Line i was the first to hold it: the next one that does (usually the next line) is first now. Line i itself
        // is skipped: it is being counted out (its new content, if it holds the name again, is counted in after).
        var k = i + 1
        while k < storedLines.count && !holds(storedLines[k]) { k += 1 }
        entry.first = k
        table[name] = entry
        return true
    }

    private static func ordered(_ table: [String: FirstSeen]) -> [String] {
        return table.sorted { $0.value.first == $1.value.first ? $0.key < $1.key : $0.value.first < $1.value.first }.map { $0.key }
    }

    /// "1:12:30 of 3:58:00 (31 %) · 2.1× realtime · ETA 1 h 05 min · 312 lines · ru 60 % en 35 %".
    var progressText: String {
        var parts: [String] = []
        if duration > 0 {
            parts.append("\(Format.duration(Int(position))) of \(Format.duration(Int(duration))) (\(Int((position * 100 / duration).rounded())) %)")
        }
        if speed > 0 { parts.append(String(format: "%.1f× realtime", speed)) }
        if let eta = eta { parts.append("ETA " + Format.rough(eta)) }
        if spokenLines > 0 {
            parts.append("\(spokenLines) line\(spokenLines == 1 ? "" : "s")")
            let shares = languageShares.prefix(4).map { "\($0.0) \($0.1) %" }
            if !shares.isEmpty { parts.append(shares.joined(separator: " ")) }
        }
        return parts.joined(separator: " · ")
    }

    mutating func finish(exitStatus: Int32, stopRequested: Bool) {
        phase = .finished
        if stopRequested { outcome = .stopped }
        else if exitStatus == 0 { outcome = .completed }
        else if !errors.isEmpty && exitStatus == 1 { outcome = .completedWithErrors(errors.count) }
        else { outcome = .failed(exitStatus) }
    }
}

/// The translations shown beside the transcript, one column each.
enum TranslationColumns {
    /// `chosen` nil (nothing picked yet): every translation on offer; else the picked ones that are on offer, in the
    /// order they are offered (so the columns keep their places whatever order they were picked in).
    static func shown(choices: [String], chosen: [String]?) -> [String] {
        guard let chosen = chosen else { return choices }
        return choices.filter { chosen.contains($0) }
    }

    /// One line as text: the transcript's line, then a tab and each shown translation ("" while one is still to come).
    static func copyLine(_ text: String, translations: [String], of line: [String: String]) -> String {
        return ([text] + translations.map { line[$0] ?? "" }).joined(separator: "\t")
    }
}
