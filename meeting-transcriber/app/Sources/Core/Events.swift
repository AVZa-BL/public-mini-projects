import Foundation

/// One line of output of mt.py. With MT_EVENTS=1 it prints machine-readable lines
/// "@@KIND<TAB>field<TAB>field…"; everything else is ordinary text (warnings, notes) shown as-is.
/// One transcript line as mt.py reports it live (@@LINE, JSON): a spoken line, or a note about the recording.
struct TranscriptLine: Equatable {
    var t: Double
    var end: Double?
    var lang: String
    var conf: Double
    var guess: Bool
    var speaker: String?
    var text: String
    var note: String?
    var live: Bool
    var tr: [String: String] = [:]     // the line's translations by target language ("Russian"), live or final

    var isNote: Bool { return note != nil }
    /// "[ru]", "[de?]": a question mark for a low confidence or a language outside the chosen list.
    var tag: String { return "[" + lang + (conf < 0.5 || guess ? "?" : "") + "]" }

    static func parse(json: String) -> TranscriptLine? {
        guard let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data), let dict = object as? [String: Any],
              let t = (dict["t"] as? NSNumber)?.doubleValue else { return nil }
        let note = dict["note"] as? String
        guard note != nil || dict["text"] is String else { return nil }
        let text = dict["text"] as? String ?? ""
        // `tr_of` is the text the translations were made from: a line whose text changed since (an echo trimmed from a
        // call's microphone line) carries translations of another text, which are not shown beside it.
        let translatedFrom = dict["tr_of"] as? String
        let tr = translatedFrom == nil || translatedFrom == text ? (dict["tr"] as? [String: String]) ?? [:] : [:]
        return TranscriptLine(t: t, end: (dict["end"] as? NSNumber)?.doubleValue, lang: dict["lang"] as? String ?? "",
                              conf: (dict["conf"] as? NSNumber)?.doubleValue ?? 1, guess: (dict["guess"] as? NSNumber)?.boolValue ?? false,
                              speaker: dict["speaker"] as? String, text: text, note: note,
                              live: (dict["live"] as? NSNumber)?.boolValue ?? false, tr: tr)
    }
}

enum MTEvent: Equatable {
    case file(index: Int, count: Int, path: String)
    case progress(done: Int, total: Int)
    case position(done: Double, total: Double)   // seconds of audio done, of the whole track
    case eta(Int)                                // seconds left, from the speed so far
    case line(TranscriptLine)
    case output(path: String)
    case failure(file: String, message: String)
    case done(failed: Int)
    case check(name: String, status: String)
    case stage(String)
    case ready
    case note(String)          // something the user should see at once (also in the final alert), not only in the log
    case level(Double)         // the microphone's loudness, 0…1, once a second while recording
    case folder(String)        // the data folder a recording writes into, as soon as it exists
    case clear                 // the lines shown so far are replaced by the ones that follow (a call, merged and echo-trimmed)
    case text(String)
}

enum EventParser {
    static func parse(_ rawLine: String) -> MTEvent {
        let line = rawLine.hasSuffix("\r") ? String(rawLine.dropLast()) : rawLine
        guard line.hasPrefix("@@") else {
            // A progress fragment mt.py redraws with "\r" and never ends ("   3 min", "12/40 chunks") can land in front of
            // an event written to the same pipe a moment later; the event is still one.
            if let at = line.range(of: "@@"), line[..<at.lowerBound].range(of: #"^\s*\d+( min|/\d+ chunks)$"#, options: .regularExpression) != nil {
                return parse(String(line[at.lowerBound...]))
            }
            return .text(line)
        }
        let f = line.dropFirst(2).split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
        func s(_ i: Int) -> String { return i < f.count ? f[i] : "" }
        func n(_ i: Int) -> Int { return Int(s(i)) ?? 0 }
        switch s(0) {
        case "FILE": return .file(index: n(1), count: n(2), path: s(3))
        case "PROG": return .progress(done: n(1), total: n(2))
        case "POS": return .position(done: Double(s(1)) ?? 0, total: Double(s(2)) ?? 0)
        case "ETA": return .eta(Int(Double(s(1)) ?? 0))
        case "LINE": if let line = TranscriptLine.parse(json: s(1)) { return .line(line) } else { return .text(line) }
        case "OUT": return .output(path: s(1))
        case "ERR": return .failure(file: s(1), message: s(2))
        case "DONE": return .done(failed: n(1))
        case "CHECK": return .check(name: s(1), status: s(2))
        case "STAGE": return .stage(s(1))
        case "READY": return .ready
        case "NOTE": return .note(s(1))
        case "LEVEL": return .level(min(1, max(0, Double(s(1)) ?? 0)))
        case "FOLDER": return .folder(s(1))
        case "CLEAR": return .clear
        default: return .text(line)
        }
    }
}

enum Format {
    /// "02:41", "1:02:41".
    static func duration(_ seconds: Int) -> String {
        let s = max(0, seconds)
        let h = s / 3600, m = (s % 3600) / 60, x = s % 60
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, x) : String(format: "%02d:%02d", m, x)
    }

    /// "1 h 05 min", "12 min", "40 s": for an estimate.
    static func rough(_ seconds: Int) -> String {
        let s = max(0, seconds)
        if s >= 3600 { return "\(s / 3600) h \(String(format: "%02d", (s % 3600) / 60)) min" }
        if s >= 60 { return "\(s / 60) min" }
        return "\(s) s"
    }
}
