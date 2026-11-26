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
    var lines: [TranscriptLine] = []   // the transcript so far, as mt.py reports it (reset per file)
    var position: Double = 0      // seconds of audio done in the current track
    var duration: Double = 0      // seconds of audio in the current track
    var eta: Int? = nil           // seconds left, estimated by mt.py
    var speed: Double = 0         // audio seconds per wall-clock second, smoothed
    var level: Double = 0         // the microphone's loudness while recording, 0…1
    var folder: String? = nil     // the data folder a recording writes into
    var targets: [String] = []    // the languages this run translates into (live, or into files): what the pane can show
    var positionAt: Date? = nil
    var outcome: Outcome? = nil

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
            if count > 1 { lines = [] }
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
            if let i = lines.lastIndex(where: { $0.t == line.t && $0.isNote == line.isNote
                                                && ($0.speaker == line.speaker || ($0.live && !line.live && ($0.speaker == nil || line.speaker == nil))) }) {
                var merged = line
                // A line sent again without its translations keeps the ones shown, but only for the same text: a line the
                // final pass decoded anew (other languages, another granularity) must not show the translation of the old one.
                if merged.tr.isEmpty && merged.text == lines[i].text { merged.tr = lines[i].tr }
                lines[i] = merged
            } else {
                lines.append(line)
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
            lines = []
        case .note(let text):
            notes.append(text)
            log.append(text)
        case .level(let value):
            level = value
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
    var spokenLines: Int { return lines.filter { !$0.isNote }.count }

    /// Languages of the lines so far as (code, percent), most frequent first.
    var languageShares: [(String, Int)] {
        var counts: [String: Int] = [:]
        for line in lines where !line.isNote { counts[line.lang, default: 0] += 1 }
        let total = counts.values.reduce(0, +)
        guard total > 0 else { return [] }
        return counts.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
            .map { ($0.key, Int((Double($0.value) * 100 / Double(total)).rounded())) }
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
