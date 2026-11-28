import SwiftUI
import AppKit

// The transcript's rows (C2). The list is an Equatable view whose equality is the RunState's `revision` (an Int that
// changes only with the lines) plus what is shown of them, so a POS, a NOTE or any telemetry update does not
// re-evaluate it, and each row is Equatable too: of 2,000 lines only the one that changed is drawn again.

/// The lines of a recording or a job, filtered, one row each, following the end when asked.
struct TranscriptList: View, Equatable {
    let lines: [TranscriptLine]
    let revision: Int
    /// The session the lines belong to ("recording", "processing"): another session's list is another list.
    let session: String
    let translations: [String]          // the translation columns shown beside the transcript
    let filter: String
    let timings: Bool
    let follow: Bool
    let reduceMotion: Bool
    let emptyText: String

    static func == (a: TranscriptList, b: TranscriptList) -> Bool {
        // The first and the last line too: the next job's seeded RunState can have the same revision and count (O(1)).
        return a.revision == b.revision && a.lines.count == b.lines.count && a.session == b.session
            && a.lines.first == b.lines.first && a.lines.last == b.lines.last
            && a.translations == b.translations && a.filter == b.filter && a.timings == b.timings && a.follow == b.follow
            && a.reduceMotion == b.reduceMotion && a.emptyText == b.emptyText
    }

    private var visible: [TranscriptLine] {
        let needle = filter.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return lines }
        return lines.filter { ($0.note ?? $0.text).localizedCaseInsensitiveContains(needle) || ($0.speaker ?? "").localizedCaseInsensitiveContains(needle) }
    }

    var body: some View {
        let shown = visible
        let lastID = shown.last?.id
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if lines.isEmpty {
                        Text(emptyText)
                            .font(.caption).foregroundStyle(.secondary).padding(4)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    ForEach(shown, id: \.id) { line in
                        TranscriptRow(line: line, translations: translations, timings: timings).equatable()
                    }
                }
                .padding(4)
            }
            .background(Color(nsColor: .textBackgroundColor))
            // Follow: the end of the transcript stays in view as lines come, change (the final pass refines them) and get
            // their translations. An Int compare, not the lines.
            .onChange(of: revision) { _ in
                if follow, let id = lastID { scroll(proxy, to: id) }
            }
            .onChange(of: follow) { on in
                if on, let id = lastID { scroll(proxy, to: id) }
            }
            .onAppear {
                if follow, let id = lastID { proxy.scrollTo(id, anchor: .bottom) }
            }
        }
    }

    private func scroll(_ proxy: ScrollViewProxy, to id: Int) {
        if reduceMotion {
            proxy.scrollTo(id, anchor: .bottom)
        } else {
            withAnimation(.easeOut(duration: 0.15)) { proxy.scrollTo(id, anchor: .bottom) }
        }
    }
}

/// One line: time, language, who spoke and the text, then each of `translations` beside it, all in columns of equal
/// width, so the columns line up from row to row under their names.
struct TranscriptRow: View, Equatable {
    let line: TranscriptLine
    let translations: [String]
    let timings: Bool

    var body: some View {
        if let note = line.note {
            Text("# [\(Format.duration(Int(line.t)))] \(note)")
                .font(.system(.caption, design: .monospaced)).italic().foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            HStack(alignment: .top, spacing: 6) {
                HStack(alignment: .top, spacing: 6) {
                    Text(Format.duration(Int(line.t))).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    LanguageCapsule(line: line)
                    if let speaker = line.speaker {
                        Text(speaker).font(.caption.bold()).foregroundStyle(Palette.speaker(speaker))
                    }
                    Text(line.text).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                        .foregroundStyle(line.live ? Color.secondary : Color.primary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                ForEach(translations, id: \.self) { language in
                    Divider()
                    TranslationCell(line: line, language: language, timings: timings)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.vertical, 1)
        }
    }
}

/// The line's language as a small capsule, "[ru]": secondary when Whisper was sure (0.8 or more), primary between 0.5
/// and 0.8, and orange with a question mark for a guess (below 0.5, or a language outside the chosen list).
struct LanguageCapsule: View {
    let line: TranscriptLine

    var body: some View {
        let doubtful = line.conf < 0.5 || line.guess
        let tint: Color = doubtful ? Palette.warningText : (line.conf >= 0.8 ? Color.secondary : Color.primary)
        Text(line.tag)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(tint)
            .padding(.horizontal, 4)
            .background(Capsule().fill((doubtful ? Palette.warningFill : Color.secondary).opacity(doubtful ? 0.22 : 0.12)))
            .accessibilityLabel(LanguageCapsule.spoken(line))
    }

    /// "Russian, confidence 62 percent", and "a guess" when it is one.
    static func spoken(_ line: TranscriptLine) -> String {
        let name = Locale.current.localizedString(forLanguageCode: line.lang) ?? line.lang
        let percent = Int((min(1, max(0, line.conf)) * 100).rounded())
        return "\(name), confidence \(percent) percent" + (line.guess ? ", a guess" : "")
    }
}

/// One translation of a line: the text; "…" while it is still to come; "at the end" when the live translation left it
/// for the final pass (`tr_skip`); with Timings on, the time it took and the model that made it.
struct TranslationCell: View {
    let line: TranscriptLine
    let language: String
    let timings: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            if let text = line.tr[language] {
                Text(text).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    .foregroundStyle(line.live ? Color.secondary : Color.primary)
                if timings, let caption = TranslationCell.timing(ms: line.trMs[language], by: line.trBy[language]) {
                    Text(caption).font(.caption2).foregroundStyle(.secondary)
                }
            } else if line.trSkipped.contains(language) {
                Text("at the end").italic().foregroundStyle(.secondary)
                    .help("Not translated live (the live translation was behind, or the recording stopped first): the final pass translates this line.")
            } else {
                Text("…").foregroundStyle(.secondary)
                    .accessibilityLabel("translation still to come")
            }
        }
    }

    /// "0.9 s · Hy-MT3", "Hy-MT3", or nil when neither is known.
    static func timing(ms: Int?, by: String?) -> String? {
        let model = by.map(TranslationCell.modelName)
        switch (ms, model) {
        case let (ms?, model?): return Format.ms(ms) + " · " + model
        case let (ms?, nil): return Format.ms(ms)
        case let (nil, model?): return model
        default: return nil
        }
    }

    /// The model's name as the window says it: "hy" Hy-MT3, "gemma" TranslateGemma, "claude" Claude, an NLLB repo NLLB.
    static func modelName(_ by: String) -> String {
        let lower = by.lowercased()
        if lower == "hy" || lower.contains("hy-mt") { return "Hy-MT3" }
        if lower == "gemma" || lower.contains("translategemma") { return "TranslateGemma" }
        if lower == "claude" || lower.hasPrefix("claude") { return "Claude" }
        if lower.contains("nllb") { return "NLLB" }
        return by.split(separator: "/").last.map(String.init) ?? by
    }
}

/// The pane header's chips: how far behind the transcript is, and how many made-up lines were left out. They observe
/// the telemetry (at most 4 times a second), not the lines.
struct TranscriptChips: View {
    @ObservedObject var store: TelemetryStore
    let source: TelemetrySource

    var body: some View {
        let model = store.model(source)
        HStack(spacing: 6) {
            if let lag = model.transcriptLag {
                let level = LagLevel.transcript(lag)
                StateChip(symbol: Palette.symbol(level), text: "Lag " + TranscriptChips.seconds(lag), color: Palette.text(level))
                    .help("Seconds between the end of what was said and its line (\(level.word))")
                    .accessibilityLabel("Transcript lag \(TelemetrySummary.seconds(lag)), \(level.word)")
            }
            if model.madeUpTotal > 0 {
                StateChip(symbol: "eye.slash", text: "\(model.madeUpTotal) left out", color: .secondary)
                    .help("Lines Whisper made up on noise or silence (subtitle credits, outros repeated on silence) were left out by screen_made_up")
            }
        }
    }

    static func seconds(_ s: Double) -> String {
        return "\(Int(max(0, s).rounded())) s"
    }
}

/// A small capsule with a symbol and a word: a state is never told by colour alone.
struct StateChip: View {
    let symbol: String
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: symbol)
            Text(text).monospacedDigit()
        }
        .font(.caption)
        .foregroundStyle(color)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Capsule().fill(color.opacity(0.12)))
        .accessibilityElement(children: .combine)
    }
}
