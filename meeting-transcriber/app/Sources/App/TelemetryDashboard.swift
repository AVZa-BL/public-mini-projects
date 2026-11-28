import SwiftUI
import AppKit
import Charts
import Accessibility

// The Telemetry column (C3): cards of Swift Charts (macOS 13) over mt.py's @@STAT events and the app's own samples.
// It observes only the TelemetryStore (published at most every 250 ms), the LayoutPrefs and the DisplayPrefs; what it
// needs from the transcript comes as an Equatable `DashboardContext`, so a POS or a LINE does not redraw the charts.

/// The spoken lines per language, for the Languages card.
struct LanguageCount: Equatable, Hashable, Identifiable {
    var code: String
    var count: Int
    var id: String { return code }
}

/// What the dashboard takes from the window (values only).
struct DashboardContext: Equatable {
    var source: TelemetrySource = .machine
    var hasRecording = false
    var hasProcessing = false
    var trackNames: [String] = []          // the recording's tracks: "Microphone", or "You", "Them"
    var languages: [LanguageCount] = []
    var spokenLines = 0
    var jobLabel: String? = nil
    var chunk = 0                          // the job's chunks done and in all (RunState)
    var chunks = 0
    var fileFraction = 0.0
    var eta: Int? = nil
    var engineCheck: String? = nil         // RunState.checks["engine"]: "mlx", "ct2"
}

/// The span of the time charts: the chosen window ending now, or the whole session.
struct ChartSpan: Equatable {
    var window: Double?
    var now: Double

    /// Points per series a card draws: a column is about 300 pt wide, so 150 points is 2 pt each (C3 allows 300). Fewer
    /// marks keep a redraw of every card (at most 4 a second) cheap on the main thread.
    static let maxPoints = 150

    /// One metric's points in this span, downsampled (min and max per bucket kept) to `maxPoints`.
    func points(_ model: TelemetryModel, _ metric: Metric) -> [ChartPoint] {
        return model.series(metric, window: window, now: now, maxPoints: ChartSpan.maxPoints)
    }

    var domain: ClosedRange<Double> {
        let hi = max(now, 10)
        guard let w = window else { return 0...hi }
        let lo = max(0, hi - w)
        return lo...max(hi, lo + 10)
    }
}

// MARK: - The column

struct TelemetryColumn: View, Equatable {
    @ObservedObject var store: TelemetryStore
    @ObservedObject var layout: LayoutPrefs
    @ObservedObject var display: DisplayPrefs
    let context: DashboardContext
    /// A horizontal strip of cards above the transcript (narrow windows) instead of the column.
    var compact = false

    static func == (a: TelemetryColumn, b: TelemetryColumn) -> Bool {
        return a.context == b.context && a.compact == b.compact
    }

    var body: some View {
        let model = store.model(context.source)
        let span = ChartSpan(window: layout.window.seconds, now: store.now(context.source))
        if compact {
            VStack(alignment: .leading, spacing: 4) {
                header
                ScrollViewReader { proxy in
                    ScrollView(.horizontal, showsIndicators: true) {
                        HStack(alignment: .top, spacing: 10) { cards(model, span) }
                            .padding(.horizontal, 8)
                            .padding(.bottom, 6)
                    }
                    .onChange(of: layout.focusRequest) { _ in focus(proxy, anchor: .leading) }
                    .onAppear { DispatchQueue.main.async { focus(proxy, anchor: .leading) } }
                }
            }
        } else {
            VStack(spacing: 0) {
                header.padding(10)
                Divider()
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(spacing: 10) { cards(model, span) }
                            .padding(10)
                    }
                    // focusRequest changes on every click (the same card twice too); a request made while the column
                    // was hidden is taken when it appears (onChange does not fire for the first value).
                    .onChange(of: layout.focusRequest) { _ in focus(proxy, anchor: .top) }
                    .onAppear { DispatchQueue.main.async { focus(proxy, anchor: .top) } }
                }
            }
        }
    }

    /// Scrolls to the card the status bar asked for, if any, and clears the request.
    private func focus(_ proxy: ScrollViewProxy, anchor: UnitPoint) {
        guard let card = layout.focusCard else { return }
        if display.reduceMotion { proxy.scrollTo(card, anchor: anchor) }
        else { withAnimation(.easeInOut(duration: 0.2)) { proxy.scrollTo(card, anchor: anchor) } }
        DispatchQueue.main.async { layout.focusCard = nil }
    }

    private var header: some View {
        HStack(spacing: 8) {
            Label("Telemetry", systemImage: "waveform.path.ecg").font(.headline)
            Spacer(minLength: 4)
            if context.hasRecording && context.hasProcessing {
                Picker("", selection: $layout.dashboardShowsJob) {
                    Text("Recording").tag(false)
                    Text("Job").tag(true)
                }
                .pickerStyle(.segmented).labelsHidden().frame(width: 140)
                .help("Both run: show the recording's figures or the processing job's")
            }
            Picker("", selection: $layout.window) {
                ForEach(TelemetryWindow.allCases) { w in Text(w.label).tag(w) }
            }
            .pickerStyle(.segmented).labelsHidden().frame(width: 170)
            .help("The span the charts show (Session: as far back as the telemetry keeps)")
        }
    }

    private var width: CGFloat? { return compact ? 300 : nil }

    @ViewBuilder private func cards(_ model: TelemetryModel, _ span: ChartSpan) -> some View {
        switch context.source {
        case .recording:
            LiveHealthCard(model: model, span: span, context: context, display: display, width: width)
            SpeechCard(model: model, context: context, display: display, width: width)
            TranslatorCard(model: model, span: span, display: display, width: width, final: false)
            QueuesCard(model: model, span: span, context: context, display: display, width: width)
            MemoryCard(model: model, span: span, display: display, width: width)
            CPUCard(model: model, span: span, display: display, width: width)
            LanguagesCard(context: context, width: width)
            EventsCard(model: model, width: width)
        case .processing:
            ProgressCard(model: model, span: span, context: context, display: display, width: width)
            SpeechCard(model: model, context: context, display: display, width: width)
            TranslatorCard(model: model, span: span, display: display, width: width, final: true)
            MemoryCard(model: model, span: span, display: display, width: width)
            CPUCard(model: model, span: span, display: display, width: width)
            EventsCard(model: model, width: width)
        case .machine:
            MemoryCard(model: model, span: span, display: display, width: width)
            CPUCard(model: model, span: span, display: display, width: width)
            ModelsCard(models: store.modelsOnDisk, width: width)
        }
    }
}

// MARK: - Building blocks

/// A card: a title with a symbol over its content, on a material background.
struct Card<Content: View>: View {
    let title: String
    let symbol: String
    let id: String
    let width: CGFloat?
    let content: Content

    init(_ title: String, symbol: String, id: String, width: CGFloat?, @ViewBuilder content: () -> Content) {
        self.title = title
        self.symbol = symbol
        self.id = id
        self.width = width
        self.content = content()
    }

    var body: some View {
        framed
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Palette.grid.opacity(0.6), lineWidth: 0.5))
            .accessibilityElement(children: .contain)
            .accessibilityLabel(title)
            .id(id)
    }

    private var inner: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: symbol).font(.headline)
            content
        }
        .padding(12)
    }

    /// In the column: the column's width. In the strip of a narrow window: a fixed width and height, scrolling inside.
    @ViewBuilder private var framed: some View {
        if let width = width {
            ScrollView(.vertical) {
                inner.frame(width: width, alignment: .leading)
            }
            .frame(width: width, height: Card.stripHeight)
        } else {
            inner.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    static var stripHeight: CGFloat { return 250 }
}

/// "No telemetry from this engine": an engine card before (or without) any @@STAT.
struct NoTelemetry: View {
    var body: some View {
        Text("No telemetry from this engine yet.")
            .font(.caption).foregroundStyle(.secondary)
    }
}

/// A label and a value in a stats row.
struct StatValue: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value).font(.body.monospacedDigit())
            Text(label).font(.caption).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}

/// One big number of the Live health card, with its state as a symbol and a word.
struct HeroNumber: View {
    let title: String
    let seconds: Double?
    let level: LagLevel?
    let reduceMotion: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(seconds.map { "\(Int(max(0, $0).rounded()))" } ?? "–")
                    .font(.system(size: 30, weight: .semibold, design: .rounded)).monospacedDigit()
                    .contentTransition(reduceMotion ? .identity : .numericText())
                    .animation(reduceMotion ? nil : .default, value: seconds.map { Int(max(0, $0).rounded()) })
                Text("s").font(.caption).foregroundStyle(.secondary)
            }
            if let level = level {
                Label(level.word, systemImage: Palette.symbol(level))
                    .font(.caption).foregroundStyle(Palette.text(level))
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(title)
        .accessibilityValue(seconds.map { TelemetrySummary.seconds($0) + (level.map { ", " + $0.word } ?? "") } ?? "none yet")
    }
}

/// How a series is drawn: its colour, and dashed or solid (told in the legend too, not by colour alone).
struct SeriesStyle: Equatable {
    var color: Color
    var dashed = false
}

/// A horizontal line across a chart, with its label.
struct YRule: Equatable {
    var y: Double
    var label: String
    var color: Color = .secondary
}

final class HoverModel: ObservableObject {
    @Published var x: Double? = nil
}

/// The hover of a chart over time (C3) for the charts TimeSeriesChart does not draw (bars, points): the x under the
/// pointer goes into `hover`, and `HoverReadout.text` says each series' nearest value there.
struct HoverReadout: ViewModifier {
    @ObservedObject var hover: HoverModel

    func body(content: Content) -> some View {
        content.chartOverlay { proxy in
            GeometryReader { geometry in
                Rectangle().fill(Color.clear).contentShape(Rectangle())
                    .onContinuousHover(coordinateSpace: CoordinateSpace.local) { phase in
                        switch phase {
                        case .active(let location):
                            let origin = geometry[proxy.plotAreaFrame].origin
                            hover.x = proxy.value(atX: location.x - origin.x, as: Double.self)
                        case .ended:
                            hover.x = nil
                        }
                    }
            }
        }
    }

    /// "02:13 · 3.2×", or "02:13 · English 18.4 tok/s · German 17.0 tok/s" with `named`; " " when nothing is hovered.
    static func text(_ points: [ChartPoint], at x: Double?, unit: String, named: Bool = false) -> String {
        let near = ChartPoint.nearest(points, to: x)
        guard let x = x, !near.isEmpty else { return " " }
        return ([Format.duration(Int(max(0, x)))] + near.map { (named ? $0.series + " " : "") + ChartUnits.text($0.y, unit: unit) })
            .joined(separator: " · ")
    }
}

/// A chart over the session's time: lines, stacked areas, labelled rules, a hover readout (chartOverlay +
/// onContinuousHover, macOS 13), x labelled mm:ss with 5 ticks, an accessibility label, value and chart descriptor.
struct TimeSeriesChart: View {
    let label: String
    let summary: String
    var lines: [ChartPoint] = []
    var areas: [ChartPoint] = []           // stacked (`stacking: .standard`), or drawn over each other in order
    var stacked = true
    let styles: [String: SeriesStyle]
    let order: [String]                    // the legend's order
    let span: ChartSpan
    var yDomain: ClosedRange<Double>
    var rules: [YRule] = []
    let unit: String
    var height: CGFloat = 80
    @ObservedObject var display: DisplayPrefs
    @StateObject private var hover = HoverModel()

    private func style(_ series: String) -> SeriesStyle {
        return styles[series] ?? SeriesStyle(color: .secondary)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            legend
            chart
                .frame(height: height)
                .accessibilityLabel(label)
                .accessibilityValue(summary)
                .accessibilityChartDescriptor(SeriesDescriptor(title: label, unit: unit, points: areas + lines,
                                                               xRange: span.domain, yRange: yDomain))
            Text(readout ?? " ").font(.caption2.monospacedDigit()).foregroundStyle(.secondary).lineLimit(1)
                .accessibilityHidden(readout == nil)
        }
    }

    private var chart: some View {
        Chart {
            ForEach(Array(areas.enumerated()), id: \.offset) { _, p in
                AreaMark(x: .value("Time", p.x), y: .value(unit, p.y), series: .value("Series", p.series),
                         stacking: stacked ? .standard : .unstacked)
                    .foregroundStyle(style(p.series).color.opacity(0.35))
            }
            ForEach(Array(lines.enumerated()), id: \.offset) { _, p in
                LineMark(x: .value("Time", p.x), y: .value(unit, p.y), series: .value("Series", p.series))
                    .foregroundStyle(style(p.series).color)
                    .lineStyle(StrokeStyle(lineWidth: display.lineWidth, dash: style(p.series).dashed ? [4, 3] : []))
            }
            ForEach(Array(rules.enumerated()), id: \.offset) { _, r in
                RuleMark(y: .value("Limit", r.y))
                    .foregroundStyle(r.color.opacity(0.8))
                    .lineStyle(StrokeStyle(lineWidth: 0.75, dash: [2, 2]))
                    .annotation(position: .top, alignment: .leading) {
                        Text(r.label).font(.caption2).foregroundStyle(.secondary)
                    }
            }
            if let x = hover.x {
                RuleMark(x: .value("Time", x))
                    .foregroundStyle(Color.secondary.opacity(0.7))
                    .lineStyle(StrokeStyle(lineWidth: 0.75))
            }
        }
        .chartXScale(domain: span.domain)
        .chartYScale(domain: yDomain)
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisGridLine().foregroundStyle(Palette.grid.opacity(display.gridOpacity))
                AxisValueLabel {
                    if let s = value.as(Double.self) { Text(Format.duration(Int(s))) }
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: .automatic(desiredCount: 3)) { _ in
                AxisGridLine().foregroundStyle(Palette.grid.opacity(display.gridOpacity))
                AxisValueLabel()
            }
        }
        .chartOverlay { proxy in
            GeometryReader { geometry in
                Rectangle().fill(Color.clear).contentShape(Rectangle())
                    // CoordinateSpace (the enum) picks the macOS 13 overload; the macOS 14 one takes a CoordinateSpaceProtocol.
                    .onContinuousHover(coordinateSpace: CoordinateSpace.local) { phase in
                        switch phase {
                        case .active(let location):
                            let origin = geometry[proxy.plotAreaFrame].origin
                            hover.x = proxy.value(atX: location.x - origin.x, as: Double.self)
                        case .ended:
                            hover.x = nil
                        }
                    }
            }
        }
    }

    private var legend: some View {
        HStack(spacing: 10) {
            ForEach(order, id: \.self) { name in
                HStack(spacing: 4) {
                    LegendSwatch(style: style(name))
                    Text(name + (style(name).dashed ? " (dashed)" : "")).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .accessibilityHidden(true)
    }

    /// At the hovered time: each series' nearest value ("02:13 · Transcript lag 4 s · Translation lag 9 s").
    private var readout: String? {
        guard let x = hover.x else { return nil }
        var parts = [Format.duration(Int(max(0, x)))]
        let all = areas + lines
        for name in order {
            let points = all.filter { $0.series == name }
            guard let nearest = points.min(by: { abs($0.x - x) < abs($1.x - x) }) else { continue }
            parts.append("\(name) " + ChartUnits.text(nearest.y, unit: unit))
        }
        return parts.joined(separator: " · ")
    }
}

/// A short line in a series' colour, dashed when the series is.
struct LegendSwatch: View {
    let style: SeriesStyle

    var body: some View {
        Path { p in
            p.move(to: CGPoint(x: 0, y: 4))
            p.addLine(to: CGPoint(x: 14, y: 4))
        }
        .stroke(style.color, style: StrokeStyle(lineWidth: 2, dash: style.dashed ? [4, 3] : []))
        .frame(width: 14, height: 8)
    }
}

enum ChartUnits {
    /// "4 s", "6.1 GB", "38 %", "2.1×", "18.4 tok/s", "3".
    static func text(_ y: Double, unit: String) -> String {
        switch unit {
        case "s": return String(format: y < 10 ? "%.1f s" : "%.0f s", y)
        case "GB": return String(format: "%.1f GB", y)
        case "%": return String(format: "%.0f %%", y)
        case "×": return Format.rate(y, unit: "×")
        case "tok/s": return Format.rate(y, unit: "tok/s")
        default: return String(format: "%.0f", y)
        }
    }

    /// A y range from 0 that holds the values with some room: at least `floor`.
    static func upper(_ points: [ChartPoint], floor: Double, margin: Double = 1.15) -> Double {
        let top = points.map { $0.y }.max() ?? 0
        return max(floor, top * margin)
    }
}

/// VoiceOver's audio graph and data table for a time chart (AXChartDescriptor, macOS 12+).
struct SeriesDescriptor: AXChartDescriptorRepresentable {
    let title: String
    let unit: String
    let points: [ChartPoint]
    let xRange: ClosedRange<Double>
    let yRange: ClosedRange<Double>

    func makeChartDescriptor() -> AXChartDescriptor {
        let xAxis = AXNumericDataAxisDescriptor(title: "Time since the start", range: xRange, gridlinePositions: []) { value in
            Format.duration(Int(max(0, value)))
        }
        let unit = self.unit
        let yAxis = AXNumericDataAxisDescriptor(title: unit, range: yRange, gridlinePositions: []) { value in
            ChartUnits.text(value, unit: unit)
        }
        var names: [String] = []
        for p in points where !names.contains(p.series) { names.append(p.series) }
        let series = names.map { name in
            AXDataSeriesDescriptor(name: name, isContinuous: true,
                                   dataPoints: points.filter { $0.series == name }.map { AXDataPoint(x: $0.x, y: $0.y) })
        }
        return AXChartDescriptor(title: title, summary: nil, xAxis: xAxis, yAxis: yAxis, additionalAxes: [], series: series)
    }
}

// MARK: - 3.1 Live health

struct LiveHealthCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    let context: DashboardContext
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?

    var body: some View {
        Card("Live health", symbol: "heart.text.square", id: "health", width: width) {
            if !model.hasEngineTelemetry {
                NoTelemetry()
            } else {
                let translates = model.models["translator"] != nil || !model.translations.isEmpty || !model.queue.isEmpty
                HStack(alignment: .top, spacing: 18) {
                    HeroNumber(title: "Transcript lag", seconds: model.transcriptLag,
                               level: model.transcriptLag.map(LagLevel.transcript), reduceMotion: display.reduceMotion)
                    if translates {
                        HeroNumber(title: "Translation lag", seconds: model.translationLag,
                                   level: model.translationLag.map(LagLevel.translation), reduceMotion: display.reduceMotion)
                    }
                    Spacer(minLength: 0)
                }
                Text(backlog).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                let lag = span.points(model, .transcriptLag)
                let tr = span.points(model, .translationLag)
                TimeSeriesChart(label: "Transcript lag", summary: TelemetrySummary.lag(model), lines: lag + tr,
                                styles: ["Transcript lag": SeriesStyle(color: Palette.speech),
                                         "Translation lag": SeriesStyle(color: Palette.translation, dashed: true)],
                                order: translates ? ["Transcript lag", "Translation lag"] : ["Transcript lag"],
                                span: span, yDomain: 0...max(30, ((lag + tr).map { $0.y }.max() ?? 0) + 5),
                                rules: [YRule(y: 10, label: "slow", color: Palette.warningText), YRule(y: 30, label: "behind", color: Palette.bad)],
                                unit: "s", height: 100, display: display)
            }
        }
    }

    /// "Backlog: 0 chunks (You 0 · Them 1) · 3 translations queued · 19 at the end".
    private var backlog: String {
        var pending: [Int] = model.queue.last?.value.pending ?? []
        if pending.isEmpty {
            // No translator, so no queue samples: each track's latest chunk says how many are still pending.
            var byTrack: [Int: Int] = [:]
            for c in model.chunks.elements where c.value.src == "live" { byTrack[c.value.track] = c.value.pending }
            if let top = byTrack.keys.max() { pending = (0...top).map { byTrack[$0] ?? 0 } }
        }
        let total = pending.reduce(0, +)
        var text = "Backlog: \(total) chunk\(total == 1 ? "" : "s")"
        if pending.count > 1 {
            let named = pending.enumerated().map { i, n in "\(i < context.trackNames.count ? context.trackNames[i] : "Track \(i + 1)") \(n)" }
            text += " (" + named.joined(separator: " · ") + ")"
        }
        if let q = model.queue.last?.value {
            text += " · \(q.jobs) translation\(q.jobs == 1 ? "" : "s") queued"
        }
        if model.skippedTotal > 0 { text += " · \(model.skippedTotal) at the end" }
        return text
    }
}

// MARK: - 3.2 Speech engine

struct SpeechCard: View {
    let model: TelemetryModel
    let context: DashboardContext
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?
    @StateObject private var hover = HoverModel()

    var body: some View {
        Card("Speech engine", symbol: "waveform", id: "speech", width: width) {
            Text(SpeechCard.title(model.models["speech"], check: context.engineCheck)).font(.body)
                .fixedSize(horizontal: false, vertical: true)
            if let m = model.models["speech"], m.loadMs != nil || m.bytes != nil {
                Text([m.loadMs.map { "loaded in " + Format.ms($0) }, m.bytes.map { Format.bytes($0) + " of weights" }]
                        .compactMap { $0 }.joined(separator: " · "))
                    .font(.caption).foregroundStyle(.secondary)
            }
            if model.chunks.isEmpty {
                NoTelemetry()
            } else {
                let recent = Array(model.chunks.elements.suffix(60))
                let bars = recent.compactMap { c in c.value.rtf.map { ChartPoint(x: c.value.b, y: $0, series: "Speed") } }
                Chart {
                    ForEach(Array(bars.enumerated()), id: \.offset) { _, bar in
                        BarMark(x: .value("Chunk end", bar.x), y: .value("Speed", bar.y), width: .fixed(4))
                            .foregroundStyle(bar.y < 1.5 ? Palette.warningFill : Palette.speech)
                    }
                    RuleMark(y: .value("Real time", 1))
                        .foregroundStyle(Palette.bad.opacity(0.8))
                        .lineStyle(StrokeStyle(lineWidth: 0.75, dash: [2, 2]))
                        .annotation(position: .top, alignment: .leading) { Text("real time").font(.caption2).foregroundStyle(.secondary) }
                }
                .chartXAxis {
                    AxisMarks(values: .automatic(desiredCount: 5)) { value in
                        AxisGridLine().foregroundStyle(Palette.grid.opacity(display.gridOpacity))
                        AxisValueLabel { if let s = value.as(Double.self) { Text(Format.duration(Int(s))) } }
                    }
                }
                .chartYAxis {
                    AxisMarks(values: .automatic(desiredCount: 3)) { _ in
                        AxisGridLine().foregroundStyle(Palette.grid.opacity(display.gridOpacity))
                        AxisValueLabel()
                    }
                }
                .frame(height: 80)
                .modifier(HoverReadout(hover: hover))
                .accessibilityLabel("Speed of each chunk, times real time")
                .accessibilityValue(speedSummary(recent))
                Text(HoverReadout.text(bars, at: hover.x, unit: "×")).font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary).lineLimit(1).accessibilityHidden(hover.x == nil)
                Text("Bars: how many times faster than real time each chunk was decoded (yellow below 1.5×).")
                    .font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                HStack(alignment: .top, spacing: 14) {
                    StatValue(label: "median decode", value: model.medianDecodeMs.map(Format.ms) ?? "–")
                    StatValue(label: "chunks", value: "\(model.chunks.count)")
                    StatValue(label: "speech share", value: model.speechShare.map(Format.percent) ?? "–")
                }
                HStack(alignment: .top, spacing: 14) {
                    StatValue(label: "mean log-prob", value: meanLogProb.map { String(format: "%.2f", $0) } ?? "–")
                    StatValue(label: "fallbacks", value: "\(model.chunks.filter { ($0.value.temp ?? 0) > 0 }.count)")
                    StatValue(label: "made-up left out", value: "\(model.madeUpTotal)")
                }
            }
        }
    }

    private var meanLogProb: Double? {
        let values = model.chunks.compactMap { $0.value.lp }
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    private func speedSummary(_ chunks: [Timed<ChunkStat>]) -> String {
        let values = chunks.compactMap { $0.value.rtf }
        guard let last = values.last else { return "none yet" }
        let slow = values.filter { $0 < 1.5 }.count
        return "latest \(Format.rate(last, unit: "×")); \(slow) of \(values.count) below 1.5 times"
    }

    /// "Whisper large-v3-turbo · Apple GPU (MLX)", or "· CPU".
    static func title(_ m: ModelStat?, check: String?) -> String {
        let name = m.map { $0.name.split(separator: "/").last.map(String.init) ?? $0.name } ?? ""
        let shown = name.isEmpty ? "Whisper" : (name.lowercased().contains("whisper") ? name : "Whisper " + name)
        switch m?.engine ?? check {
        case "mlx"?: return shown + " · Apple GPU (MLX)"
        case "ct2"?: return shown + " · CPU"
        case let other?: return shown + " · " + other
        case nil: return shown
        }
    }
}

// MARK: - 3.3 Translator

struct TranslatorCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?
    let final: Bool                        // the processing job's (lines per second) rather than the recording's
    @StateObject private var tpsHover = HoverModel()
    @StateObject private var secondsHover = HoverModel()

    var body: some View {
        Card("Translator", symbol: "character.bubble", id: "translator", width: width) {
            let translator = model.models["translator"] ?? model.models["nllb"]
            if translator == nil && model.translations.isEmpty && model.queue.isEmpty {
                Text(model.hasEngineTelemetry ? "No translation in this session." : "No telemetry from this engine yet.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                HStack(spacing: 8) {
                    Text(TranslatorCard.name(translator)).font(.body).lineLimit(1).truncationMode(.middle)
                    Spacer(minLength: 4)
                    stateChip(translator)
                }
                if let m = translator, m.loadMs != nil || m.bytes != nil {
                    Text([m.loadMs.map { "loaded in " + Format.ms($0) }, m.bytes.map { Format.bytes($0) + " of weights" }]
                            .compactMap { $0 }.joined(separator: " · "))
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let m = translator, m.state == .downloading || m.state == .loading, let p = m.progress {
                    ProgressView(value: p) {
                        Text(m.state == .downloading ? "Downloading" : "Loading").font(.caption)
                    } currentValueLabel: {
                        Text(downloadText(m, progress: p)).font(.caption.monospacedDigit())
                    }
                }
                if let error = translator?.error, translator?.state == .failed {
                    Text(error).font(.caption).foregroundStyle(Palette.bad).fixedSize(horizontal: false, vertical: true)
                }
                charts
                counts
            }
        }
    }

    @ViewBuilder private var charts: some View {
        let tps = span.points(model, .tps)
        let seconds = span.points(model, .trSeconds)
        if !tps.isEmpty {
            Text("Tokens per second, each job").font(.caption).foregroundStyle(.secondary)
            Chart {
                ForEach(Array(tps.enumerated()), id: \.offset) { _, p in
                    PointMark(x: .value("Time", p.x), y: .value("tok/s", p.y))
                        .foregroundStyle(by: .value("Target", p.series))
                        .symbol(by: .value("Target", p.series))
                        .symbolSize(18)
                }
            }
            .chartXScale(domain: span.domain)
            .chartXAxis { timeAxis }
            .chartLegend(position: .bottom, alignment: .leading)
            .frame(height: 96)
            .modifier(HoverReadout(hover: tpsHover))
            .accessibilityLabel("Tokens per second of each translation")
            .accessibilityValue(model.medianTPS.map { "median " + Format.rate($0, unit: "tokens per second") } ?? "none yet")
            Text(HoverReadout.text(tps, at: tpsHover.x, unit: "tok/s", named: true)).font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary).lineLimit(1).accessibilityHidden(tpsHover.x == nil)
        }
        if !seconds.isEmpty {
            let median = TelemetryModel.median(seconds.map { $0.y }) ?? 0
            Text("Seconds per job (median \(ChartUnits.text(median, unit: "s")))").font(.caption).foregroundStyle(.secondary)
            Chart {
                ForEach(Array(seconds.enumerated()), id: \.offset) { _, p in
                    BarMark(x: .value("Time", p.x), y: .value("s", p.y), width: .fixed(3))
                        .foregroundStyle(Palette.translation)
                }
                RuleMark(y: .value("Median", median))
                    .foregroundStyle(Color.secondary)
                    .lineStyle(StrokeStyle(lineWidth: 0.75, dash: [2, 2]))
            }
            .chartXScale(domain: span.domain)
            .chartXAxis { timeAxis }
            .frame(height: 60)
            .modifier(HoverReadout(hover: secondsHover))
            .accessibilityLabel("Seconds each translation took")
            .accessibilityValue("median " + ChartUnits.text(median, unit: "s"))
            Text(HoverReadout.text(seconds, at: secondsHover.x, unit: "s")).font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary).lineLimit(1).accessibilityHidden(secondsHover.x == nil)
        }
    }

    private var timeAxis: some AxisContent {
        AxisMarks(values: .automatic(desiredCount: 5)) { value in
            AxisGridLine().foregroundStyle(Palette.grid.opacity(display.gridOpacity))
            AxisValueLabel { if let s = value.as(Double.self) { Text(Format.duration(Int(s))) } }
        }
    }

    @ViewBuilder private var counts: some View {
        let q = model.queue.last?.value
        HStack(alignment: .center, spacing: 14) {
            if final {
                let recent = model.translations.filter { $0.at >= span.now - 60 }
                let lines = recent.reduce(0) { $0 + max(1, $1.value.lines) }
                StatValue(label: "lines per second", value: String(format: "%.1f", Double(lines) / 60))
                StatValue(label: "translated", value: "\(model.translations.reduce(0) { $0 + max(1, $1.value.lines) })")
                StatValue(label: "median tok/s", value: model.medianTPS.map { String(format: "%.1f", $0) } ?? "–")
            } else {
                StatValue(label: "translated live", value: "\(q?.translated ?? model.translations.count)")
                StatValue(label: "at the end", value: "\(model.skippedTotal)")
                StatValue(label: "median tok/s", value: model.medianTPS.map { String(format: "%.1f", $0) } ?? "–")
                if let duty = model.duty {
                    Gauge(value: duty, in: 0...1) {
                        Text("Duty")
                    } currentValueLabel: {
                        Text(Format.percent(duty)).font(.caption2.monospacedDigit())
                    }
                    .gaugeStyle(.accessoryCircularCapacity)
                    .tint(duty > 0.8 ? Palette.warningText : Palette.translation)
                    .help("The translator's share of the last 60 s; it starts no new job above 80 %")
                    .accessibilityLabel("Translator duty")
                    .accessibilityValue(Format.percent(duty) + " of the last minute, cap 80 percent")
                    Text("cap 80 %").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func downloadText(_ m: ModelStat, progress: Double) -> String {
        var text = Format.percent(progress)
        if m.state == .downloading, let done = m.doneBytes, let total = m.totalBytes, total > 0 {
            text += " · " + Format.bytes(done) + " of " + Format.bytes(total)
        }
        return text
    }

    @ViewBuilder private func stateChip(_ m: ModelStat?) -> some View {
        let state = model.translatorState == .unknown ? (m?.state ?? .unknown) : model.translatorState
        let progress = (state == .downloading || state == .loading) ? m?.progress : nil
        let word = state.word + (progress.map { " " + Format.percent($0) } ?? "")
        StateChip(symbol: TranslatorCard.symbol(state), text: word, color: TranslatorCard.color(state))
    }

    static func name(_ m: ModelStat?) -> String {
        guard let m = m, !m.name.isEmpty else { return "Translator" }
        let short = m.name.split(separator: "/").last.map(String.init) ?? m.name
        return short + (m.engine.map { $0 == "api" ? " · online" : $0 == "mlx" ? " · Apple GPU" : " · CPU" } ?? "")
    }

    static func symbol(_ s: TranslatorState) -> String {
        switch s {
        case .ready: return "checkmark.circle"
        case .downloading: return "arrow.down.circle"
        case .checking: return "magnifyingglass"
        case .loading: return "hourglass"
        case .waiting: return "pause.circle"
        case .behind: return "exclamationmark.triangle"
        case .failed: return "xmark.octagon"
        case .nllb: return "memorychip"
        case .off, .stopped, .released, .unknown: return "minus.circle"
        }
    }

    static func color(_ s: TranslatorState) -> Color {
        switch s {
        case .ready: return Palette.ok
        case .behind, .waiting, .nllb: return Palette.warningText
        case .failed: return Palette.bad
        case .downloading, .checking, .loading: return Palette.translation
        case .off, .stopped, .released, .unknown: return .secondary
        }
    }
}

// MARK: - 3.4 Queues

struct QueuesCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    let context: DashboardContext
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?

    var body: some View {
        Card("Queues", symbol: "tray.full", id: "queues", width: width) {
            if model.queue.isEmpty {
                Text(model.hasEngineTelemetry ? "No translation queue in this session." : "No telemetry from this engine yet.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                let jobs = span.points(model, .jobs).map { ChartPoint(x: $0.x, y: $0.y, series: "Translation jobs") }
                let tracks = max(1, model.queue.last?.value.pending.count ?? 1)
                let names = (0..<tracks).map { i in i < context.trackNames.count ? context.trackNames[i] : "Track \(i + 1)" }
                let pending = (0..<tracks).flatMap { i in
                    span.points(model, .pending(track: i)).map { ChartPoint(x: $0.x, y: $0.y, series: names[i] + " chunks") }
                }
                TimeSeriesChart(label: "Queues", summary: queueSummary, lines: pending, areas: jobs, styles: QueuesCard.styles(names),
                                order: ["Translation jobs"] + names.map { $0 + " chunks" }, span: span,
                                yDomain: 0...ChartUnits.upper(jobs + pending, floor: 4), unit: "", height: 60, display: display)
            }
        }
    }

    /// Jobs in the translation colour; each track's pending chunks in the speech colour, the first solid, the others dashed.
    static func styles(_ names: [String]) -> [String: SeriesStyle] {
        var styles: [String: SeriesStyle] = ["Translation jobs": SeriesStyle(color: Palette.translation)]
        for (i, name) in names.enumerated() { styles[name + " chunks"] = SeriesStyle(color: Palette.speech, dashed: i > 0) }
        return styles
    }

    private var queueSummary: String {
        guard let q = model.queue.last?.value else { return "none yet" }
        return "\(q.jobs) translation jobs queued; chunks pending: " + q.pending.map(String.init).joined(separator: ", ")
    }
}

// MARK: - 3.5 Memory

struct MemoryCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?

    var body: some View {
        Card("Memory", symbol: "memorychip", id: "memory", width: width) {
            let mlx = MemoryCard.mlxAreas(model, span)
            let child = span.points(model, .childFootprint)
            let app = span.points(model, .appFootprint)
            let ramGB = model.app.last?.memTotalMB.map { Double($0) / 1024 }
            let recGB = model.sys.newest(where: { $0.value.gpuRecMB != nil })?.value.gpuRecMB.map { Double($0) / 1024 }
            if mlx.isEmpty && child.isEmpty && app.isEmpty {
                Text("No memory figures yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                let rules = MemoryCard.rules(recGB: recGB, ramGB: ramGB)
                let top = ramGB ?? ChartUnits.upper(mlx + child + app, floor: 1)
                TimeSeriesChart(label: "Memory", summary: TelemetrySummary.memory(model), lines: child + app, areas: mlx, stacked: false,
                                styles: ["MLX active + cache": SeriesStyle(color: Palette.memory.opacity(0.45)),
                                         "MLX active": SeriesStyle(color: Palette.memory),
                                         "mt.py": SeriesStyle(color: Palette.speech),
                                         "App": SeriesStyle(color: Palette.cpu, dashed: true)],
                                order: (mlx.isEmpty ? [] : ["MLX active", "MLX active + cache"]) + (child.isEmpty ? [] : ["mt.py"]) + ["App"],
                                span: span, yDomain: 0...max(top * 1.02, 0.5), rules: rules, unit: "GB", height: 120, display: display)
            }
            chips
        }
    }

    /// MLX's memory as two areas from the same samples (so their times line up): active + cache behind, active in front,
    /// which reads as active stacked under cache. Every k-th sample when there are more than the chart draws.
    static func mlxAreas(_ model: TelemetryModel, _ span: ChartSpan) -> [ChartPoint] {
        let lo = span.window.map { span.now - $0 } ?? -Double.infinity
        let samples = model.sys.elements.filter { $0.at >= lo && $0.value.mlxActiveMB != nil }
        guard !samples.isEmpty else { return [] }
        let step = max(1, (samples.count + ChartSpan.maxPoints - 1) / ChartSpan.maxPoints)
        var total: [ChartPoint] = [], active: [ChartPoint] = []
        for (k, s) in samples.enumerated() where k % step == 0 || k == samples.count - 1 {
            let a = Double(s.value.mlxActiveMB ?? 0) / 1024
            let c = Double(s.value.mlxCacheMB ?? 0) / 1024
            total.append(ChartPoint(x: s.at, y: a + c, series: "MLX active + cache"))
            active.append(ChartPoint(x: s.at, y: a, series: "MLX active"))
        }
        return total + active
    }

    /// The GPU's working-set limit (gpu_rec_mb) and the Mac's RAM, as lines across the chart.
    static func rules(recGB: Double?, ramGB: Double?) -> [YRule] {
        var rules: [YRule] = []
        if let rec = recGB { rules.append(YRule(y: rec, label: "GPU working-set limit", color: Palette.gpu)) }
        if let ram = ramGB { rules.append(YRule(y: ram, label: "RAM", color: Palette.bad)) }
        return rules
    }

    @ViewBuilder private var chips: some View {
        let sample = model.app.last
        let sys = model.sys.last?.value
        HStack(spacing: 6) {
            if let p = sample?.pressure {
                let level: LagLevel = p <= 0 ? .ok : p == 1 ? .slow : .behind
                StateChip(symbol: Palette.symbol(level), text: "Pressure " + TelemetrySummary.pressure(p), color: Palette.text(level))
                    .help("macOS's memory pressure: normal, warning or critical")
            }
            if let swap = sample?.swapUsedMB {
                StateChip(symbol: "arrow.left.arrow.right", text: "Swap " + Format.megabytes(swap), color: swap > 1024 ? Palette.warningText : .secondary)
                    .help("Swap in use on this Mac")
            }
        }
        HStack(spacing: 6) {
            if let used = sample?.memUsedMB, let total = sample?.memTotalMB {
                StateChip(symbol: "memorychip", text: "Used " + Format.megabytes(used) + " of " + Format.megabytes(total), color: .secondary)
                    .help("Memory used as Activity Monitor counts it: app memory, wired and compressed")
            }
            if let compressed = sample?.memCompressedMB {
                StateChip(symbol: "rectangle.compress.vertical", text: "Compressed " + Format.megabytes(compressed), color: .secondary)
            }
        }
        if let overflows = sys?.audioOverflows {
            StateChip(symbol: overflows > 0 ? "exclamationmark.triangle" : "checkmark.circle",
                      text: "\(overflows) audio overflow\(overflows == 1 ? "" : "s")", color: overflows > 0 ? Palette.bad : Palette.ok)
                .help("Audio blocks the recorder dropped because it was not read in time")
        }
        if let s = sys, s.mlxPeakMB != nil || s.mlxLimitMB != nil {
            Text([s.mlxPeakMB.map { "MLX peak " + Format.megabytes($0) }, s.mlxLimitMB.map { "limit " + Format.megabytes($0) },
                  s.rssMaxMB.map { "mt.py max RSS " + Format.megabytes($0) }].compactMap { $0 }.joined(separator: " · "))
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

// MARK: - 3.6 CPU, GPU and heat

struct CPUCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?

    var body: some View {
        Card("CPU, GPU and heat", symbol: "cpu", id: "cpu", width: width) {
            let child = span.points(model, .childCPU)
            let app = span.points(model, .appCPU)
            let gpu = span.points(model, .gpuBusy)
                .map { ChartPoint(x: $0.x, y: $0.y, series: "GPU busy (undocumented counter)") }
            if child.isEmpty && app.isEmpty && gpu.isEmpty {
                Text("No CPU figures yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                let top = gpu.isEmpty ? min(100, ChartUnits.upper(child + app, floor: 25, margin: 1.25)) : 100
                TimeSeriesChart(label: "CPU and GPU", summary: summary, lines: child + app + gpu,
                                styles: ["mt.py": SeriesStyle(color: Palette.cpu), "App": SeriesStyle(color: Palette.speech),
                                         "GPU busy (undocumented counter)": SeriesStyle(color: Palette.gpu, dashed: true)],
                                order: (child.isEmpty ? [] : ["mt.py"]) + ["App"] + (gpu.isEmpty ? [] : ["GPU busy (undocumented counter)"]),
                                span: span, yDomain: 0...top, unit: "%", height: 80, display: display)
                Text("CPU in percent of the whole Mac (all cores).").font(.caption2).foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                if let thermal = model.app.last?.thermal {
                    let level: LagLevel = thermal <= 1 ? .ok : thermal == 2 ? .slow : .behind
                    StateChip(symbol: level == .ok ? "thermometer" : "thermometer.sun", text: TelemetrySummary.thermal(thermal), color: Palette.text(level))
                        .help("macOS's thermal state: Nominal, Fair, Serious or Critical")
                }
                if model.sys.last?.value.yielding == true {
                    StateChip(symbol: "tortoise", text: "Yielding to the recording", color: Palette.warningText)
                        .help("This job sleeps after each chunk or line while a recording runs, so the live transcript keeps the GPU")
                }
            }
            if let s = model.sys.last?.value, s.dutySpeech != nil || s.dutyTr != nil || s.cutMs != nil {
                Text([s.dutySpeech.map { "speech " + Format.percent($0) }, s.dutyTr.map { "translation " + Format.percent($0) },
                      s.cutMs.map { String(format: "VAD cut %.0f ms", $0) }].compactMap { $0 }.joined(separator: " · ")
                     + (s.dutySpeech != nil || s.dutyTr != nil ? " of the worker's time" : ""))
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var summary: String {
        guard let a = model.app.last else { return "none yet" }
        var parts: [String] = []
        if let c = a.childCPU { parts.append("mt.py " + ChartUnits.text(c, unit: "%")) }
        if let c = a.appCPU { parts.append("app " + ChartUnits.text(c, unit: "%")) }
        if let g = a.gpuBusy { parts.append("GPU busy " + ChartUnits.text(g, unit: "%")) }
        parts.append("thermal " + TelemetrySummary.thermal(a.thermal))
        return parts.joined(separator: "; ")
    }
}

// MARK: - 3.7 Languages

struct LanguagesCard: View {
    let context: DashboardContext
    let width: CGFloat?

    var body: some View {
        Card("Languages", symbol: "globe", id: "languages", width: width) {
            let data = Array(context.languages.prefix(8))
            if data.isEmpty {
                Text("No lines yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                if #available(macOS 14, *) {
                    Chart(data) { item in
                        SectorMark(angle: .value("Lines", item.count), innerRadius: .ratio(0.6), angularInset: 1)
                            .foregroundStyle(by: .value("Language", item.code))
                    }
                    .chartLegend(position: .trailing, alignment: .center)
                    .frame(height: 110)
                    .help(summary(data))           // hover: every language's share (C3)
                    .accessibilityLabel("Languages of the lines")
                    .accessibilityValue(summary(data))
                } else {
                    Chart(data) { item in
                        BarMark(x: .value("Lines", item.count), y: .value("Language", item.code))
                            .foregroundStyle(Palette.speech)
                    }
                    .frame(height: CGFloat(max(1, data.count)) * 18 + 16)
                    .help(summary(data))
                    .accessibilityLabel("Languages of the lines")
                    .accessibilityValue(summary(data))
                }
                Text("\(context.spokenLines) line\(context.spokenLines == 1 ? "" : "s")").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func summary(_ data: [LanguageCount]) -> String {
        let total = max(1, context.spokenLines)
        return data.map { "\(Locale.current.localizedString(forLanguageCode: $0.code) ?? $0.code) \(Int((Double($0.count) * 100 / Double(total)).rounded())) percent" }
            .joined(separator: ", ")
    }
}

// MARK: - 3.8 Events

struct EventsCard: View {
    let model: TelemetryModel
    let width: CGFloat?

    var body: some View {
        Card("Events", symbol: "list.bullet.rectangle", id: "events", width: width) {
            let events = model.milestones.elements.reversed()
            if events.isEmpty {
                Text("Nothing yet: model loads, translations left for the end and pauses are listed here.")
                    .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(Array(events.enumerated()), id: \.offset) { _, event in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(Format.duration(Int(event.at))).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                        Text(event.value).font(.caption).fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }
}

// MARK: - The processing job's cards

struct ProgressCard: View {
    let model: TelemetryModel
    let span: ChartSpan
    let context: DashboardContext
    @ObservedObject var display: DisplayPrefs
    let width: CGFloat?

    var body: some View {
        Card("Progress", symbol: "chart.bar.xaxis", id: "progress", width: width) {
            if let label = context.jobLabel { Text(label).lineLimit(1).truncationMode(.middle) }
            if context.chunks > 0 {
                ProgressView(value: min(1, max(0, context.fileFraction)))
                HStack(spacing: 14) {
                    StatValue(label: "chunks", value: "\(context.chunk) of \(context.chunks)")
                    StatValue(label: "ETA", value: context.eta.map(Format.rough) ?? "–")
                    StatValue(label: "median decode", value: model.medianDecodeMs.map(Format.ms) ?? "–")
                }
            } else {
                Text("Waiting for the first chunk.").font(.caption).foregroundStyle(.secondary)
            }
            let speed = span.points(model, .rtf)
            if !speed.isEmpty {
                TimeSeriesChart(label: "Speed, times real time", summary: speed.last.map { "latest " + Format.rate($0.y, unit: "×") } ?? "",
                                lines: speed, styles: ["Speed": SeriesStyle(color: Palette.speech)], order: ["Speed"], span: span,
                                yDomain: 0...ChartUnits.upper(speed, floor: 2), rules: [YRule(y: 1, label: "real time", color: Palette.bad)],
                                unit: "×", height: 80, display: display)
            }
        }
    }
}

/// The models in the Hugging Face cache (the idle machine view).
struct ModelsCard: View {
    let models: [(name: String, bytes: Int64)]
    let width: CGFloat?

    var body: some View {
        Card("Models on disk", symbol: "externaldrive", id: "models", width: width) {
            if models.isEmpty {
                Text("No models in the Hugging Face cache yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(Array(models.prefix(12).enumerated()), id: \.offset) { _, model in
                    HStack {
                        Text(model.name).font(.caption).lineLimit(1).truncationMode(.middle)
                        Spacer(minLength: 6)
                        Text(Format.bytes(model.bytes)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                    }
                }
                Text("In all " + Format.bytes(models.reduce(0) { $0 + $1.bytes })).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - The status bar, the toolbar's capsule and the level meter

/// Always visible under the session (C2.4): "Whisper turbo · GPU | Hy-MT3 ready (load 6.2 s) | lag 3 s · translation
/// 8 s | MLX 6.1 GB · pressure normal | CPU 38 %". Each part opens its card.
struct TelemetryStatusBar: View {
    @ObservedObject var store: TelemetryStore
    let layout: LayoutPrefs
    let source: TelemetrySource
    let engineCheck: String?

    var body: some View {
        let m = store.model(source)
        HStack(spacing: 0) {
            if source != .machine {
                part(SpeechCard.title(m.models["speech"], check: engineCheck), card: "speech")
                if let translator = translatorText(m) { divider; part(translator, card: "translator") }
                if let lag = lagText(m) { divider; part(lag, card: source == .recording ? "health" : "progress") }
                divider
            }
            if let memory = memoryText(m) { part(memory, card: "memory"); divider }
            part(cpuText(m), card: "cpu")
            Spacer(minLength: 0)
        }
        .font(.caption.monospacedDigit())
        .lineLimit(1)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.bar)
    }

    private var divider: some View {
        Text(" | ").foregroundStyle(.tertiary)
    }

    private func part(_ text: String, card: String) -> some View {
        Button(action: { layout.open(card: card) }) {
            Text(text).foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help("Show the card in the Telemetry column")
    }

    private func translatorText(_ m: TelemetryModel) -> String? {
        guard let t = m.models["translator"] ?? m.models["nllb"] else { return nil }
        let short = t.name.split(separator: "/").last.map(String.init) ?? t.name
        let state = m.translatorState == .unknown ? t.state : m.translatorState
        var text = (short.isEmpty ? "Translator" : short) + " " + state.word.lowercased()
        if (state == .loading || state == .downloading), let p = t.progress { text += " " + Format.percent(p) }
        if let ms = t.loadMs, state == .ready { text += " (load " + Format.ms(ms) + ")" }
        return text
    }

    private func lagText(_ m: TelemetryModel) -> String? {
        guard let lag = m.transcriptLag else { return nil }
        var text = "lag " + TranscriptChips.seconds(lag)
        if let tr = m.translationLag { text += " · translation " + TranscriptChips.seconds(tr) }
        return text
    }

    private func memoryText(_ m: TelemetryModel) -> String? {
        var parts: [String] = []
        if let mlx = m.sys.last?.value.mlxActiveMB { parts.append("MLX " + Format.megabytes(mlx)) }
        if let p = m.app.last?.pressure { parts.append("pressure " + TelemetrySummary.pressure(p)) }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func cpuText(_ m: TelemetryModel) -> String {
        guard let a = m.app.last else { return "CPU –" }
        let cpu = (a.childCPU ?? 0) + (a.appCPU ?? 0)
        return "CPU " + ChartUnits.text(cpu, unit: "%")
    }
}

/// The toolbar's status capsule: "Recording 12:41 · lag 3 s", "Paused 03:10", "Transcribing · 41 %", "Idle".
struct ToolbarStatus: View {
    @ObservedObject var store: TelemetryStore
    let recording: Bool
    let paused: Bool
    let since: Date?
    let recordedTime: (Date) -> TimeInterval?
    let jobText: String?

    var body: some View {
        HStack(spacing: 6) {
            if recording, let since = since {
                Image(systemName: paused ? "pause.circle.fill" : "record.circle")
                    .foregroundStyle(paused ? Palette.warningText : Palette.bad)
                TimelineView(.periodic(from: since, by: 1)) { context in
                    Text((paused ? "Paused " : "Recording ") + Format.duration(Int(recordedTime(context.date) ?? 0)))
                        .monospacedDigit()
                }
                if let lag = store.recording.transcriptLag {
                    let level = LagLevel.transcript(lag)
                    Text("· lag " + TranscriptChips.seconds(lag)).monospacedDigit().foregroundStyle(Palette.text(level))
                    Image(systemName: Palette.symbol(level)).foregroundStyle(Palette.text(level)).accessibilityLabel(level.word)
                }
            } else if recording {
                Image(systemName: "record.circle").foregroundStyle(Palette.bad)
                Text("Starting the recording…")
            }
            if let job = jobText {
                if recording { Text("·").foregroundStyle(.tertiary) }
                Image(systemName: "gearshape.2").foregroundStyle(.secondary)
                Text(job).lineLimit(1).truncationMode(.middle)
            }
            if !recording && jobText == nil {
                Text("Idle").foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .padding(.horizontal, 10)
        .padding(.vertical, 3)
        .background(Capsule().fill(Color.secondary.opacity(0.12)))
        .accessibilityElement(children: .combine)
    }
}

/// The microphone's level bar: it alone observes the LevelMeter.
struct LevelMeterView: View {
    @ObservedObject var meter: LevelMeter

    var body: some View {
        ProgressView(value: meter.level)
            .frame(width: 90)
            .help("Microphone level: it moves while you speak")
            .accessibilityLabel("Microphone level")
            .accessibilityValue(Format.percent(meter.level))
    }
}
