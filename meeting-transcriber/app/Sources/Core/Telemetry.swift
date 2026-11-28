import Foundation

// Telemetry: what mt.py reports about its own work (@@STAT<TAB>kind<TAB>json, one JSON object of fewer than 20 fields) and
// what the app samples of the processes, kept in bounded buffers and reduced to the few numbers and series the window
// shows. Field names are snake_case on the wire and camelCase here. Units: `S` seconds (audio seconds for audio
// positions), `Ms` milliseconds, `MB` MiB, `Bytes` bytes; fractions are 0…1. A field that is missing is nil (or 0 / ""
// / [] for the few that are not optional); a field that is not known is ignored.

/// Reading the values of a parsed JSON object, whatever Foundation made of them (NSNumber, Bool, Int, Double).
enum TelemetryJSON {
    static func object(_ json: String) -> [String: Any]? {
        guard let data = json.data(using: .utf8), let object = try? JSONSerialization.jsonObject(with: data) else { return nil }
        return object as? [String: Any]
    }

    static func double(_ value: Any?) -> Double? {
        guard let value = value, !(value is String) else { return nil }
        var d: Double? = nil
        if let n = value as? NSNumber { d = n.doubleValue }
        else if let x = value as? Double { d = x }
        else if let x = value as? Int { d = Double(x) }
        guard let r = d, r.isFinite else { return nil }
        return r
    }

    static func int(_ value: Any?) -> Int? {
        guard let d = double(value), abs(d) < 9.0e15 else { return nil }
        return Int(d.rounded())
    }

    static func int64(_ value: Any?) -> Int64? {
        guard let d = double(value), abs(d) < 9.0e15 else { return nil }
        return Int64(d.rounded())
    }

    static func bool(_ value: Any?) -> Bool? {
        if let b = value as? Bool { return b }
        if let n = value as? NSNumber { return n.boolValue }
        return nil
    }

    static func string(_ value: Any?, limit: Int = 200) -> String? {
        guard let s = value as? String else { return nil }
        return s.count > limit ? String(s.prefix(limit)) : s
    }

    static func fraction(_ value: Any?) -> Double? {
        guard let d = double(value) else { return nil }
        return min(1, max(0, d))
    }

    static func ints(_ value: Any?) -> [Int] { return ((value as? [Any]) ?? []).compactMap { int($0) } }
    static func doubles(_ value: Any?) -> [Double] { return ((value as? [Any]) ?? []).compactMap { double($0) } }
    static func strings(_ value: Any?) -> [String] { return ((value as? [Any]) ?? []).compactMap { string($0) } }
}

/// The translator's state, as `queue.state` and `model.state` report it.
enum TranslatorState: String {
    case off, downloading, checking, waiting, loading, ready, behind, nllb, stopped, failed, released, unknown

    /// From the wire; a state this app does not know is `.unknown`.
    init(wire: String?) {
        self = TranslatorState(rawValue: (wire ?? "").lowercased()) ?? .unknown
    }

    /// The chip's words (a progress, if any, is added by the view).
    var word: String {
        switch self {
        case .off: return "Off"
        case .downloading: return "Downloading"
        case .checking: return "Checking files"
        case .waiting: return "Waiting for the other job"
        case .loading: return "Loading"
        case .ready: return "Ready"
        case .behind: return "Behind: newest first"
        case .nllb: return "NLLB (low memory)"
        case .stopped: return "Stopped"
        case .failed: return "Failed"
        case .released: return "Released"
        case .unknown: return "Unknown"
        }
    }
}

/// `chunk`: one decoded chunk (live: `src` "live", the final pass: "final").
struct ChunkStat: Equatable {
    var src: String = ""
    var track: Int = 0
    var i: Int = 0
    var a: Double = 0
    var b: Double = 0
    var decMs: Int = 0
    var rtf: Double?
    var lines: Int = 0
    var madeUp: Int = 0
    var pending: Int = 0
    var fedS: Double?
    var lagS: Double?
    var lang: String?
    var lp: Double?
    var temp: Double?
    var cr: Double?
}

/// `tr`: one finished translation job.
struct TranslationStat: Equatable {
    var src: String = ""
    var track: Int = 0
    var t: Double = 0
    var target: String = ""
    var model: String = ""
    var lines: Int = 0
    var ms: Int = 0
    var waitMs: Int?
    var prefillMs: Int?
    var inTok: Int?
    var outTok: Int?
    var tps: Double?
    var finish: String?
    var lagS: Double?
    var queue: Int?
}

/// `trskip`: lines whose translation was left for the final pass.
struct SkipStat: Equatable {
    var reason: String = ""
    var lines: Int = 0
    var targets: [String] = []
    var total: Int = 0
}

/// `queue`: the live translation queue, once a second while a translator exists.
struct QueueStat: Equatable {
    var state: TranslatorState = .unknown
    var jobs: Int = 0
    var oldestS: Double?
    var duty: Double?
    var translated: Int = 0
    var skipped: Int = 0
    var pending: [Int] = []      // chunks still to decode, per track
    var behindS: [Double] = []   // per track: fed − end of the oldest pending chunk (0 if none)
}

/// `model`: a model's state (`role` "speech", "translator", "nllb").
struct ModelStat: Equatable {
    var role: String = ""
    var name: String = ""
    var engine: String?
    var state: TranslatorState = .unknown
    var progress: Double?
    var doneBytes: Int64?
    var totalBytes: Int64?
    var loadMs: Int?
    var bytes: Int64?
    var error: String?
}

/// `sys`: mt.py's own figures, every 2 s.
struct SysStat: Equatable {
    var cpuUserS: Double?
    var cpuSysS: Double?
    var rssMaxMB: Int?
    var threads: Int?
    var mlxActiveMB: Int?
    var mlxCacheMB: Int?
    var mlxPeakMB: Int?
    var mlxLimitMB: Int?
    var gpuMemMB: Int?
    var gpuRecMB: Int?
    var diskFreeMB: Int?
    var cutMs: Double?
    var dutySpeech: Double?
    var dutyTr: Double?
    var audioOverflows: Int?
    var yielding: Bool?          // "yield" on the wire
}

extension ChunkStat {
    init(json d: [String: Any]) {
        self.init(src: TelemetryJSON.string(d["src"]) ?? "", track: TelemetryJSON.int(d["track"]) ?? 0, i: TelemetryJSON.int(d["i"]) ?? 0,
                  a: TelemetryJSON.double(d["a"]) ?? 0, b: TelemetryJSON.double(d["b"]) ?? 0, decMs: TelemetryJSON.int(d["dec_ms"]) ?? 0,
                  rtf: TelemetryJSON.double(d["rtf"]), lines: TelemetryJSON.int(d["lines"]) ?? 0, madeUp: TelemetryJSON.int(d["made_up"]) ?? 0,
                  pending: TelemetryJSON.int(d["pending"]) ?? 0, fedS: TelemetryJSON.double(d["fed_s"]), lagS: TelemetryJSON.double(d["lag_s"]),
                  lang: TelemetryJSON.string(d["lang"]), lp: TelemetryJSON.double(d["lp"]), temp: TelemetryJSON.double(d["temp"]),
                  cr: TelemetryJSON.double(d["cr"]))
    }
}

extension TranslationStat {
    init(json d: [String: Any]) {
        self.init(src: TelemetryJSON.string(d["src"]) ?? "", track: TelemetryJSON.int(d["track"]) ?? 0, t: TelemetryJSON.double(d["t"]) ?? 0,
                  target: TelemetryJSON.string(d["target"]) ?? "", model: TelemetryJSON.string(d["model"]) ?? "",
                  lines: TelemetryJSON.int(d["lines"]) ?? 0, ms: TelemetryJSON.int(d["ms"]) ?? 0, waitMs: TelemetryJSON.int(d["wait_ms"]),
                  prefillMs: TelemetryJSON.int(d["prefill_ms"]), inTok: TelemetryJSON.int(d["in_tok"]), outTok: TelemetryJSON.int(d["out_tok"]),
                  tps: TelemetryJSON.double(d["tps"]), finish: TelemetryJSON.string(d["finish"]), lagS: TelemetryJSON.double(d["lag_s"]),
                  queue: TelemetryJSON.int(d["queue"]))
    }
}

extension SkipStat {
    init(json d: [String: Any]) {
        self.init(reason: TelemetryJSON.string(d["reason"]) ?? "", lines: TelemetryJSON.int(d["lines"]) ?? 0,
                  targets: TelemetryJSON.strings(d["targets"]), total: TelemetryJSON.int(d["total"]) ?? 0)
    }
}

extension QueueStat {
    init(json d: [String: Any]) {
        self.init(state: TranslatorState(wire: d["state"] as? String), jobs: TelemetryJSON.int(d["jobs"]) ?? 0,
                  oldestS: TelemetryJSON.double(d["oldest_s"]), duty: TelemetryJSON.fraction(d["duty"]),
                  translated: TelemetryJSON.int(d["translated"]) ?? 0, skipped: TelemetryJSON.int(d["skipped"]) ?? 0,
                  pending: TelemetryJSON.ints(d["pending"]), behindS: TelemetryJSON.doubles(d["behind_s"]))
    }
}

extension ModelStat {
    init(json d: [String: Any]) {
        self.init(role: TelemetryJSON.string(d["role"]) ?? "", name: TelemetryJSON.string(d["name"]) ?? "", engine: TelemetryJSON.string(d["engine"]),
                  state: TranslatorState(wire: d["state"] as? String), progress: TelemetryJSON.fraction(d["progress"]),
                  doneBytes: TelemetryJSON.int64(d["done_bytes"]), totalBytes: TelemetryJSON.int64(d["total_bytes"]),
                  loadMs: TelemetryJSON.int(d["load_ms"]), bytes: TelemetryJSON.int64(d["bytes"]), error: TelemetryJSON.string(d["error"]))
    }
}

extension SysStat {
    init(json d: [String: Any]) {
        self.init(cpuUserS: TelemetryJSON.double(d["cpu_user_s"]), cpuSysS: TelemetryJSON.double(d["cpu_sys_s"]),
                  rssMaxMB: TelemetryJSON.int(d["rss_max_mb"]), threads: TelemetryJSON.int(d["threads"]),
                  mlxActiveMB: TelemetryJSON.int(d["mlx_active_mb"]), mlxCacheMB: TelemetryJSON.int(d["mlx_cache_mb"]),
                  mlxPeakMB: TelemetryJSON.int(d["mlx_peak_mb"]), mlxLimitMB: TelemetryJSON.int(d["mlx_limit_mb"]),
                  gpuMemMB: TelemetryJSON.int(d["gpu_mem_mb"]), gpuRecMB: TelemetryJSON.int(d["gpu_rec_mb"]),
                  diskFreeMB: TelemetryJSON.int(d["disk_free_mb"]), cutMs: TelemetryJSON.double(d["cut_ms"]),
                  dutySpeech: TelemetryJSON.fraction(d["duty_speech"]), dutyTr: TelemetryJSON.fraction(d["duty_tr"]),
                  audioOverflows: TelemetryJSON.int(d["audio_overflows"]), yielding: TelemetryJSON.bool(d["yield"]))
    }
}

/// One @@STAT event. A kind this app does not know, JSON that is not an object, or a line over `maxBytes` is `.ignored`:
/// dropped silently (not logged), so a newer engine never fills the window with text.
enum Stat: Equatable {
    case chunk(ChunkStat), translation(TranslationStat), skip(SkipStat), queue(QueueStat), model(ModelStat), sys(SysStat), ignored

    static let maxBytes = 4096

    static func parse(kind: String, json: String, byteCount: Int) -> Stat {
        guard byteCount <= maxBytes, json.utf8.count <= maxBytes, let d = TelemetryJSON.object(json) else { return .ignored }
        switch kind {
        case "chunk": return .chunk(ChunkStat(json: d))
        case "tr": return .translation(TranslationStat(json: d))
        case "trskip": return .skip(SkipStat(json: d))
        case "queue": return .queue(QueueStat(json: d))
        case "model": return .model(ModelStat(json: d))
        case "sys": return .sys(SysStat(json: d))
        default: return .ignored
        }
    }
}

/// What the app samples itself (C4, Darwin only): the app and the mt.py children, the machine.
/// `thermal`: ProcessInfo.ThermalState's raw value (0 nominal, 1 fair, 2 serious, 3 critical).
/// `pressure`: 0 normal, 1 warning, 2 critical (`TelemetrySummary.pressureLevel(dispatchRaw:)` converts a dispatch source's event).
/// CPU in percent of the whole machine; `gpuBusy` in percent (an undocumented counter).
struct AppSample: Equatable {
    var at: Double
    var appCPU: Double?
    var appFootprintMB: Int?
    var childCPU: Double?
    var childFootprintMB: Int?
    var thermal: Int?
    var pressure: Int?
    var swapUsedMB: Int?
    var memUsedMB: Int?
    var memCompressedMB: Int?
    var memTotalMB: Int?
    var gpuBusy: Double?
}

/// How far behind the transcript (or the translation) is, in three steps; the view adds a symbol and the word.
enum LagLevel: Equatable {
    case ok, slow, behind

    /// Seconds between the end of what was said and its line: under 10 ok, under 30 slow, else behind.
    static func transcript(_ s: Double) -> LagLevel { return s < 10 ? .ok : s < 30 ? .slow : .behind }

    /// Seconds between the end of a line and its translation: under 15 ok, under 45 slow, else behind.
    static func translation(_ s: Double) -> LagLevel { return s < 15 ? .ok : s < 45 ? .slow : .behind }

    var word: String {
        switch self {
        case .ok: return "ok"
        case .slow: return "slow"
        case .behind: return "behind"
        }
    }
}

struct ChartPoint: Equatable, Hashable {
    var x: Double          // seconds since the session start (arrival time)
    var y: Double
    var series: String
}

extension ChartPoint {
    /// For a hover readout: each series' point nearest to `x`, in the order the series first appear; none without `x`.
    static func nearest(_ points: [ChartPoint], to x: Double?) -> [ChartPoint] {
        guard let x = x else { return [] }
        var best: [String: ChartPoint] = [:], order: [String] = []
        for p in points {
            if let b = best[p.series] {
                if abs(p.x - x) < abs(b.x - x) { best[p.series] = p }
            } else {
                best[p.series] = p
                order.append(p.series)
            }
        }
        return order.compactMap { best[$0] }
    }
}

/// A series the dashboard can draw. Units of y: lags and `trSeconds` in seconds; `rtf` in ×; `tps` tokens per second;
/// `jobs` and `pending` counts; memory in GB (GiB); CPU and `gpuBusy` in percent; `duty` a fraction 0…1.
enum Metric: Hashable {
    case transcriptLag, translationLag, rtf, tps, trSeconds, jobs, pending(track: Int), mlxActive, mlxCache, childFootprint,
         appFootprint, childCPU, appCPU, gpuBusy, duty
}

struct Timed<T: Equatable>: Equatable {
    var at: Double
    var value: T
}

/// The telemetry of one session (the recording, or the processing job), in ring buffers of fixed capacity.
struct TelemetryModel: Equatable {
    static let capacities = (chunk: 512, translation: 1024, queue: 600, sys: 600, app: 600, milestones: 20)

    var chunks = RingBuffer<Timed<ChunkStat>>(capacity: TelemetryModel.capacities.chunk)
    var translations = RingBuffer<Timed<TranslationStat>>(capacity: TelemetryModel.capacities.translation)
    var queue = RingBuffer<Timed<QueueStat>>(capacity: TelemetryModel.capacities.queue)
    var sys = RingBuffer<Timed<SysStat>>(capacity: TelemetryModel.capacities.sys)
    var app = RingBuffer<AppSample>(capacity: TelemetryModel.capacities.app)
    var models: [String: ModelStat] = [:]      // by role
    var modelAt: [String: Double] = [:]        // when each role's latest model event arrived (translatorState)
    var skippedTotal = 0                       // translations left for the end (the latest `queue.skipped` or `trskip.total`)
    var madeUpTotal = 0                        // made-up lines left out (Σ chunk.made_up)
    var milestones = RingBuffer<Timed<String>>(capacity: TelemetryModel.capacities.milestones)
    private var speechSeconds: [String: Double] = [:]   // Σ(b − a) per "src track", over the whole session
    private var fedSeconds: [String: Double] = [:]      // the latest fed_s (live), or the furthest chunk end (final)

    init() {}

    /// `at`: the app's seconds since the session start when the event arrived.
    mutating func ingest(_ stat: Stat, at: Double) {
        switch stat {
        case .chunk(let c):
            chunks.append(Timed(at: at, value: c))
            madeUpTotal += max(0, c.madeUp)
            let key = "\(c.src) \(c.track)"
            speechSeconds[key, default: 0] += max(0, c.b - c.a)
            fedSeconds[key] = c.fedS ?? max(fedSeconds[key] ?? 0, c.b)
        case .translation(let t):
            translations.append(Timed(at: at, value: t))
        case .skip(let s):
            skippedTotal = s.total
            if s.lines > 0 {
                milestones.append(Timed(at: at, value: "\(s.lines) line\(s.lines == 1 ? "" : "s") left for the end" + TelemetryModel.reasonText(s.reason)))
            }
        case .queue(let q):
            let previous = queue.last?.value.state
            queue.append(Timed(at: at, value: q))
            skippedTotal = q.skipped
            if q.state != previous, let text = TelemetryModel.queueMilestone(q.state, previous: previous) {
                milestones.append(Timed(at: at, value: text))
            }
        case .model(let m):
            let previous = models[m.role]
            models[m.role] = m
            modelAt[m.role] = at
            if previous?.state != m.state || previous?.name != m.name, let text = TelemetryModel.modelMilestone(m) {
                milestones.append(Timed(at: at, value: text))
            }
        case .sys(let s):
            sys.append(Timed(at: at, value: s))
        case .ignored:
            break
        }
    }

    mutating func ingest(_ sample: AppSample) {
        app.append(sample)
    }

    /// Whether mt.py sent any telemetry at all (an older engine sends none: "No telemetry from this engine").
    var hasEngineTelemetry: Bool {
        return !chunks.isEmpty || !translations.isEmpty || !queue.isEmpty || !sys.isEmpty || !models.isEmpty
    }

    /// Seconds between what was fed and what is decoded: the latest chunk's lag, or the age of a chunk still pending
    /// (the latest queue sample's `behind_s`, any track) when that is larger.
    var transcriptLag: Double? {
        let chunkLag = chunks.newest(where: { $0.value.lagS != nil })?.value.lagS
        let pendingAge = queue.last?.value.behindS.max()
        return TelemetryModel.larger(chunkLag, pendingAge)
    }

    /// Seconds between the end of a line and its translation: the latest translation's lag, or the oldest queued job's
    /// when that is larger.
    var translationLag: Double? {
        let trLag = translations.newest(where: { $0.value.lagS != nil })?.value.lagS
        return TelemetryModel.larger(trLag, queue.last?.value.oldestS)
    }

    var medianDecodeMs: Int? {
        let values = chunks.map { $0.value.decMs }
        guard let m = TelemetryModel.median(values.map(Double.init)) else { return nil }
        return Int(m.rounded())
    }

    var medianTPS: Double? { return TelemetryModel.median(translations.compactMap { $0.value.tps }) }

    /// The translator's share of the last 60 s (latest queue sample).
    var duty: Double? { return queue.last?.value.duty }

    /// The translator's state: the newer of the latest queue sample's and the translator model's. A final model state
    /// (failed, stopped, nllb) always wins: mt.py rate-limits queue samples (a state change within 0.5 s of the last
    /// sample sends none) and sends none after the stop, and a queue sample from another thread can come after it.
    var translatorState: TranslatorState {
        let model = models["translator"]
        if let m = model, [.failed, .stopped, .nllb].contains(m.state) { return m.state }
        if let q = queue.last {
            if let m = model, m.state != .unknown, let at = modelAt["translator"], at > q.at { return m.state }
            return q.value.state
        }
        return model?.state ?? .unknown
    }

    /// Σ(b − a) over the chunks / audio fed, for the kind of pass (live or final) of the latest chunk, all tracks.
    var speechShare: Double? {
        guard let src = chunks.last?.value.src else { return nil }
        var speech = 0.0, fed = 0.0
        for (key, value) in speechSeconds where key.hasPrefix(src + " ") {
            speech += value
            fed += fedSeconds[key] ?? 0
        }
        guard fed > 0 else { return nil }
        return min(1, max(0, speech / fed))
    }

    /// The points of one metric with x in the last `window` seconds before `now` (nil: the whole session the buffers
    /// hold), at most `maxPoints` per series: a longer series is cut into buckets of equal count, each giving its
    /// lowest and its highest point, so spikes are kept.
    func series(_ metric: Metric, window: Double?, now: Double, maxPoints: Int = 300) -> [ChartPoint] {
        var points = rawPoints(metric)
        if let w = window { points = points.filter { $0.x >= now - w } }
        var order: [String] = []
        var bySeries: [String: [ChartPoint]] = [:]
        for p in points {
            if bySeries[p.series] == nil { order.append(p.series) }
            bySeries[p.series, default: []].append(p)
        }
        return order.flatMap { TelemetryModel.downsample(bySeries[$0] ?? [], maxPoints: maxPoints) }
    }

    // MARK: helpers

    private func rawPoints(_ metric: Metric) -> [ChartPoint] {
        func gb(_ mb: Int?) -> Double? { return mb.map { Double($0) / 1024 } }
        switch metric {
        case .transcriptLag:
            // The hero number over time: at each chunk or queue sample, the larger of the two as they stood then.
            var out: [ChartPoint] = []
            var chunkLag: Double? = nil, pendingAge: Double? = nil
            let c = chunks.elements, q = queue.elements
            var i = 0, j = 0
            while i < c.count || j < q.count {
                let x: Double
                if j >= q.count || (i < c.count && c[i].at <= q[j].at) {
                    x = c[i].at
                    if let lag = c[i].value.lagS { chunkLag = lag } else { i += 1; continue }
                    i += 1
                } else {
                    x = q[j].at
                    pendingAge = q[j].value.behindS.max()
                    j += 1
                }
                if let y = TelemetryModel.larger(chunkLag, pendingAge) { out.append(ChartPoint(x: x, y: y, series: "Transcript lag")) }
            }
            return out
        case .translationLag:
            var out: [ChartPoint] = []
            var trLag: Double? = nil, oldest: Double? = nil
            let t = translations.elements, q = queue.elements
            var i = 0, j = 0
            while i < t.count || j < q.count {
                let x: Double
                if j >= q.count || (i < t.count && t[i].at <= q[j].at) {
                    x = t[i].at
                    if let lag = t[i].value.lagS { trLag = lag } else { i += 1; continue }
                    i += 1
                } else {
                    x = q[j].at
                    oldest = q[j].value.oldestS
                    j += 1
                }
                if let y = TelemetryModel.larger(trLag, oldest) { out.append(ChartPoint(x: x, y: y, series: "Translation lag")) }
            }
            return out
        case .rtf:
            return chunks.compactMap { c in c.value.rtf.map { ChartPoint(x: c.at, y: $0, series: "Speed") } }
        case .tps:
            return translations.compactMap { t in t.value.tps.map { ChartPoint(x: t.at, y: $0, series: t.value.target) } }
        case .trSeconds:
            return translations.map { ChartPoint(x: $0.at, y: Double($0.value.ms) / 1000, series: $0.value.target) }
        case .jobs:
            return queue.map { ChartPoint(x: $0.at, y: Double($0.value.jobs), series: "Jobs") }
        case .pending(let track):
            return queue.compactMap { q in
                track >= 0 && track < q.value.pending.count ? ChartPoint(x: q.at, y: Double(q.value.pending[track]), series: "Track \(track + 1)") : nil
            }
        case .duty:
            return queue.compactMap { q in q.value.duty.map { ChartPoint(x: q.at, y: $0, series: "Duty") } }
        case .mlxActive:
            return sys.compactMap { s in gb(s.value.mlxActiveMB).map { ChartPoint(x: s.at, y: $0, series: "MLX active") } }
        case .mlxCache:
            return sys.compactMap { s in gb(s.value.mlxCacheMB).map { ChartPoint(x: s.at, y: $0, series: "MLX cache") } }
        case .childFootprint:
            return app.compactMap { a in gb(a.childFootprintMB).map { ChartPoint(x: a.at, y: $0, series: "mt.py") } }
        case .appFootprint:
            return app.compactMap { a in gb(a.appFootprintMB).map { ChartPoint(x: a.at, y: $0, series: "App") } }
        case .childCPU:
            return app.compactMap { a in a.childCPU.map { ChartPoint(x: a.at, y: $0, series: "mt.py") } }
        case .appCPU:
            return app.compactMap { a in a.appCPU.map { ChartPoint(x: a.at, y: $0, series: "App") } }
        case .gpuBusy:
            return app.compactMap { a in a.gpuBusy.map { ChartPoint(x: a.at, y: $0, series: "GPU busy") } }
        }
    }

    static func downsample(_ points: [ChartPoint], maxPoints: Int) -> [ChartPoint] {
        let limit = max(1, maxPoints)
        guard points.count > limit else { return points }
        if limit == 1 { return [points.max(by: { $0.y < $1.y })!] }
        let buckets = limit / 2
        var out: [ChartPoint] = []
        out.reserveCapacity(buckets * 2)
        for b in 0..<buckets {
            let lo = b * points.count / buckets, hi = (b + 1) * points.count / buckets
            guard lo < hi else { continue }
            var minIndex = lo, maxIndex = lo
            for k in lo..<hi {
                if points[k].y < points[minIndex].y { minIndex = k }
                if points[k].y > points[maxIndex].y { maxIndex = k }
            }
            if minIndex == maxIndex { out.append(points[minIndex]) }
            else { out.append(points[min(minIndex, maxIndex)]); out.append(points[max(minIndex, maxIndex)]) }
        }
        return out
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    private static func larger(_ a: Double?, _ b: Double?) -> Double? {
        switch (a, b) {
        case let (x?, y?): return max(x, y)
        case let (x?, nil): return x
        case let (nil, y?): return y
        default: return nil
        }
    }

    private static func reasonText(_ reason: String) -> String {
        switch reason {
        case "behind": return " (the translation was behind)"
        case "duty": return " (the translator's share of the GPU was used up)"
        case "stop": return " (the recording stopped)"
        case "off": return " (the translator is off)"
        case "slot": return " (another job holds the translator)"
        default: return ""
        }
    }

    private static func queueMilestone(_ state: TranslatorState, previous: TranslatorState?) -> String? {
        switch state {
        case .behind: return "Translation behind: newest lines first"
        case .waiting: return "Translation paused for the other job"
        case .ready where previous == .behind: return "Translation caught up"
        case .ready where previous == .waiting: return "Translation resumed"
        case .failed: return "Translator failed"
        case .nllb: return "Translating with NLLB (low memory)"
        default: return nil
        }
    }

    private static func modelMilestone(_ m: ModelStat) -> String? {
        let name = m.name.split(separator: "/").last.map(String.init) ?? m.role
        let shown = name.isEmpty ? m.role : name
        switch m.state {
        case .downloading: return "Downloading \(shown)"
        case .checking: return "Checking the files of \(shown)"
        case .loading: return "Loading \(shown)"
        case .ready: return m.loadMs.map { "\(shown) loaded in \(Format.ms($0))" } ?? "\(shown) ready"
        case .failed: return "\(shown) failed" + (m.error.map { ": \($0)" } ?? "")
        case .released: return "\(shown) released from memory"
        default: return nil
        }
    }
}

/// Words for VoiceOver (a chart's accessibility value).
enum TelemetrySummary {
    /// "3 seconds, steady; translation 8 seconds, rising"; a lag that is not ok says so ("35 seconds, behind, rising").
    static func lag(_ m: TelemetryModel) -> String {
        guard m.hasEngineTelemetry else { return "No telemetry from this engine" }
        var parts: [String] = []
        if let s = m.transcriptLag {
            let level = LagLevel.transcript(s)
            parts.append(seconds(s) + (level == .ok ? "" : ", " + level.word) + ", " + trend(m.series(.transcriptLag, window: nil, now: 0)))
        } else {
            parts.append("no transcript lag yet")
        }
        if let s = m.translationLag {
            let level = LagLevel.translation(s)
            parts.append("translation " + seconds(s) + (level == .ok ? "" : ", " + level.word) + ", " + trend(m.series(.translationLag, window: nil, now: 0)))
        }
        return parts.joined(separator: "; ")
    }

    /// "MLX 6.1 GB active, 512 MB cache; mt.py 7.2 GB; app 310 MB; pressure normal".
    static func memory(_ m: TelemetryModel) -> String {
        var parts: [String] = []
        if let s = m.sys.last?.value {
            var mlx: [String] = []
            if let a = s.mlxActiveMB { mlx.append(Format.megabytes(a) + " active") }
            if let c = s.mlxCacheMB { mlx.append(Format.megabytes(c) + " cache") }
            if !mlx.isEmpty { parts.append("MLX " + mlx.joined(separator: ", ")) }
        }
        if let a = m.app.last {
            if let c = a.childFootprintMB { parts.append("mt.py " + Format.megabytes(c)) }
            if let f = a.appFootprintMB { parts.append("app " + Format.megabytes(f)) }
            if a.pressure != nil { parts.append("pressure " + TelemetrySummary.pressure(a.pressure)) }
            if let swap = a.swapUsedMB, swap > 0 { parts.append("swap " + Format.megabytes(swap)) }
        }
        return parts.isEmpty ? "No memory figures yet" : parts.joined(separator: "; ")
    }

    /// DispatchSource.MemoryPressureEvent's raw value (normal 0x1, warning 0x2, critical 0x4) → AppSample.pressure.
    static func pressureLevel(dispatchRaw: UInt) -> Int {
        if dispatchRaw & 0x4 != 0 { return 2 }
        if dispatchRaw & 0x2 != 0 { return 1 }
        return 0
    }

    /// AppSample.pressure in words: "normal", "warning", "critical".
    static func pressure(_ level: Int?) -> String {
        switch level {
        case nil: return "unknown"
        case .some(let l) where l <= 0: return "normal"
        case .some(1): return "warning"
        default: return "critical"
        }
    }

    /// AppSample.thermal in words: "Nominal", "Fair", "Serious", "Critical".
    static func thermal(_ state: Int?) -> String {
        switch state {
        case nil: return "Unknown"
        case .some(let s) where s <= 0: return "Nominal"
        case .some(1): return "Fair"
        case .some(2): return "Serious"
        default: return "Critical"
        }
    }

    static func seconds(_ s: Double) -> String {
        let n = Int(max(0, s).rounded())
        return "\(n) second\(n == 1 ? "" : "s")"
    }

    /// "rising", "falling" or "steady": the latest point against the one about 30 s before it (or the first).
    static func trend(_ points: [ChartPoint]) -> String {
        guard let last = points.last, points.count > 1 else { return "steady" }
        let before = points.last(where: { $0.x <= last.x - 30 }) ?? points[0]
        let change = last.y - before.y
        let threshold = max(2, 0.2 * abs(before.y))
        return change > threshold ? "rising" : change < -threshold ? "falling" : "steady"
    }
}

enum CPUUsage {
    /// Percent of the whole machine (all `cores`) a process used between two samples of its CPU time (user + system, in
    /// nanoseconds) taken `wallSeconds` apart; 0 when the samples do not allow one (no time passed, a counter that went
    /// back after a restart).
    static func percent(prevNanos: UInt64, nanos: UInt64, wallSeconds: Double, cores: Int) -> Double {
        guard wallSeconds > 0, wallSeconds.isFinite, cores > 0, nanos >= prevNanos else { return 0 }
        let used = Double(nanos - prevNanos) / 1e9
        return min(100, max(0, used / (wallSeconds * Double(cores)) * 100))
    }
}

extension Format {
    /// "4.2 GB", "512 MB", "12 KB", "900 bytes" (units of 1024).
    static func bytes(_ b: Int64) -> String {
        let v = Double(max(0, b))
        if v >= 1024 * 1024 * 1024 { return String(format: "%.1f GB", v / (1024 * 1024 * 1024)) }
        if v >= 1024 * 1024 { return String(format: "%.0f MB", v / (1024 * 1024)) }
        if v >= 1024 { return String(format: "%.0f KB", v / 1024) }
        return "\(Int64(v)) bytes"
    }

    /// "6.1 GB", "512 MB" from MiB.
    static func megabytes(_ mb: Int) -> String {
        return mb >= 1024 ? String(format: "%.1f GB", Double(mb) / 1024) : "\(max(0, mb)) MB"
    }

    /// "850 ms", "6.2 s", "2 min".
    static func ms(_ ms: Int) -> String {
        let v = max(0, ms)
        if v < 1000 { return "\(v) ms" }
        if v < 60_000 { return String(format: "%.1f s", Double(v) / 1000) }
        return rough(v / 1000)
    }

    /// "41 %" from a fraction.
    static func percent(_ f: Double) -> String {
        guard f.isFinite else { return "– %" }
        return "\(Int((f * 100).rounded())) %"
    }

    /// "2.1×", "18.4 tok/s".
    static func rate(_ r: Double, unit: String) -> String {
        guard r.isFinite else { return "–" }
        let value = String(format: "%.1f", r)
        return unit == "×" ? value + "×" : value + " " + unit
    }
}
