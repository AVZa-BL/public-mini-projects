import Foundation
import Darwin
import IOKit

/// The CPU and memory of one session's mt.py processes, as the sampler read them.
struct ChildReading: Equatable {
    var cpu: Double?             // percent of the whole machine; nil at the first sample of every pid (no interval yet)
    var footprintMB: Int
}

/// One sample: the app and the machine (`machine`, whose `at` the store sets), each session's mt.py processes, and,
/// now and then, the models in the Hugging Face cache.
struct SamplerReading {
    var machine: AppSample
    var recordingChild: ChildReading?
    var processingChild: ChildReading?
    var modelsOnDisk: [(name: String, bytes: Int64)]? = nil
}

/// Samples the app, its mt.py processes and the machine on a background queue (C4): every 2 s while something runs,
/// every 10 s when nothing runs and the Telemetry column is shown, not at all otherwise. Darwin calls only, no root:
/// proc_pid_rusage (a process of the same user), host_statistics64, sysctl, the memory-pressure dispatch source and,
/// undocumented, the GPU's IORegistry counter (hidden when the key is missing).
final class SystemSampler {
    /// The pids of the recording's and the job's mt.py processes. Called on the sampler's queue: it must be thread-safe.
    var pids: () -> (recording: [Int32], processing: [Int32]) = { ([], []) }
    /// Called on the main queue with each sample.
    var onReading: ((SamplerReading) -> Void)?

    private let queue = DispatchQueue(label: "Meeting Transcriber system sampler", qos: .utility)
    private var timer: DispatchSourceTimer?
    private var interval: TimeInterval? = nil
    private var pressureSource: DispatchSourceMemoryPressure?
    private var pressureFromSource = 0
    private var previous: [Int32: (nanos: UInt64, at: TimeInterval)] = [:]
    private let timebaseNumer: UInt64
    private let timebaseDenom: UInt64
    private let cores = max(1, ProcessInfo.processInfo.activeProcessorCount)
    private var samples = 0
    private var gpuMisses = 0

    init() {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        timebaseNumer = UInt64(info.numer == 0 ? 1 : info.numer)
        timebaseDenom = UInt64(info.denom == 0 ? 1 : info.denom)
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: queue)
        source.setEventHandler { [weak self, weak source] in
            guard let self = self, let source = source else { return }
            self.pressureFromSource = TelemetrySummary.pressureLevel(dispatchRaw: source.data.rawValue)
        }
        source.resume()
        pressureSource = source
    }

    deinit {
        timer?.cancel()
        pressureSource?.cancel()
    }

    /// Seconds between samples; nil stops sampling. The first sample of a new cadence is taken at once.
    func setInterval(_ seconds: TimeInterval?) {
        queue.async { [weak self] in
            guard let self = self, self.interval != seconds else { return }
            self.interval = seconds
            self.timer?.cancel()
            self.timer = nil
            guard let seconds = seconds, seconds > 0 else { return }
            let t = DispatchSource.makeTimerSource(queue: self.queue)
            t.schedule(deadline: .now(), repeating: seconds, leeway: .milliseconds(Int(seconds * 100)))
            t.setEventHandler { [weak self] in self?.sample() }
            t.resume()
            self.timer = t
        }
    }

    // MARK: - One sample (on the queue)

    private func sample() {
        samples += 1
        let now = ProcessInfo.processInfo.systemUptime
        let groups = pids()
        var seen = Set<Int32>()
        func reading(_ list: [Int32]) -> ChildReading? {
            var cpu = 0.0, footprint: UInt64 = 0, any = false, anyCPU = false
            for pid in list where pid > 0 {
                seen.insert(pid)
                guard let r = usage(pid, now: now) else { continue }
                any = true
                if let c = r.cpu { cpu += c; anyCPU = true }      // a new pid has no CPU figure at its first sample
                footprint += r.footprint
            }
            return any ? ChildReading(cpu: anyCPU ? cpu : nil, footprintMB: Int(footprint >> 20)) : nil
        }
        let recording = reading(groups.recording)
        let processing = reading(groups.processing)
        let me = getpid()
        seen.insert(me)
        let app = usage(me, now: now)
        previous = previous.filter { seen.contains($0.key) }          // processes that ended

        var sample = AppSample(at: 0)
        sample.appCPU = app?.cpu
        sample.appFootprintMB = app.map { Int($0.footprint >> 20) }
        sample.thermal = ProcessInfo.processInfo.thermalState.rawValue
        sample.pressure = pressureLevel()
        sample.swapUsedMB = swapUsedMB()
        let memory = vmMemory()
        sample.memUsedMB = memory?.usedMB
        sample.memCompressedMB = memory?.compressedMB
        sample.memTotalMB = Int(ProcessInfo.processInfo.physicalMemory >> 20)
        sample.gpuBusy = gpuBusy()
        var out = SamplerReading(machine: sample, recordingChild: recording, processingChild: processing)
        if samples == 1 || samples % 30 == 0 { out.modelsOnDisk = SystemSampler.modelsOnDisk() }   // every 1-5 min
        DispatchQueue.main.async { [weak self] in self?.onReading?(out) }
    }

    /// CPU (percent of the machine since the last sample of this pid; nil at the first) and footprint of one process.
    private func usage(_ pid: Int32, now: TimeInterval) -> (cpu: Double?, footprint: UInt64)? {
        var info = rusage_info_v4()
        let result = withUnsafeMutablePointer(to: &info) { pointer -> Int32 in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
        }
        guard result == 0 else { return nil }
        // ri_user_time and ri_system_time count Mach absolute-time ticks (on Apple silicon not nanoseconds; to be checked
        // against `ps -o time` on the Mac): converted with the timebase.
        let ticks = info.ri_user_time &+ info.ri_system_time
        let nanos = UInt64(Double(ticks) * Double(timebaseNumer) / Double(timebaseDenom))
        var cpu: Double? = nil
        if let before = previous[pid] {
            cpu = CPUUsage.percent(prevNanos: before.nanos, nanos: nanos, wallSeconds: now - before.at, cores: cores)
        }
        previous[pid] = (nanos, now)
        return (cpu, info.ri_phys_footprint)
    }

    /// 0 normal, 1 warning, 2 critical: the kernel's level when sysctl gives it, else the dispatch source's last event.
    private func pressureLevel() -> Int {
        var level: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size, nil, 0) == 0, level > 0 {
            return TelemetrySummary.pressureLevel(dispatchRaw: UInt(level))      // 1, 2, 4 as the dispatch source's
        }
        return pressureFromSource
    }

    private func swapUsedMB() -> Int? {
        var swap = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        guard sysctlbyname("vm.swapusage", &swap, &size, nil, 0) == 0 else { return nil }
        return Int(swap.xsu_used >> 20)
    }

    /// Memory used as Activity Monitor counts it (app memory + wired + compressed) and the compressed part, in MiB.
    private func vmMemory() -> (usedMB: Int, compressedMB: Int)? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) { pointer -> kern_return_t in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard result == KERN_SUCCESS else { return nil }
        let page = UInt64(vm_kernel_page_size)
        let appPages = UInt64(stats.internal_page_count) &- min(UInt64(stats.purgeable_count), UInt64(stats.internal_page_count))
        let used = (appPages + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)) * page
        let compressed = UInt64(stats.compressor_page_count) * page
        return (Int(used >> 20), Int(compressed >> 20))
    }

    /// The GPU's busy percent from the IORegistry (IOAccelerator ▸ PerformanceStatistics ▸ "Device Utilization %", or
    /// "GPU Activity(%)"): undocumented, read without root as exelban/stats does; nil when no GPU reports it.
    private func gpuBusy() -> Double? {
        if gpuMisses >= 3 && samples % 30 != 0 { return nil }    // a Mac without the key: look again only now and then
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS else {
            gpuMisses += 1
            return nil
        }
        defer { IOObjectRelease(iterator) }
        var busy: Double? = nil
        var service = IOIteratorNext(iterator)
        while service != 0 {
            if let property = IORegistryEntryCreateCFProperty(service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0),
               let stats = property.takeRetainedValue() as? [String: Any],
               let value = (stats["Device Utilization %"] as? NSNumber) ?? (stats["GPU Activity(%)"] as? NSNumber) {
                busy = max(busy ?? 0, min(100, max(0, value.doubleValue)))
            }
            IOObjectRelease(service)
            service = IOIteratorNext(iterator)
        }
        gpuMisses = busy == nil ? gpuMisses + 1 : 0
        return busy
    }

    /// The models in the Hugging Face cache (HF_HUB_CACHE, HF_HOME/hub, or ~/.cache/huggingface/hub), largest first:
    /// each repo's blobs, the files themselves (the snapshots only link to them).
    static func modelsOnDisk() -> [(name: String, bytes: Int64)] {
        let env = ProcessInfo.processInfo.environment
        let hub: URL
        if let cache = env["HF_HUB_CACHE"], !cache.isEmpty {
            hub = URL(fileURLWithPath: cache)
        } else if let home = env["HF_HOME"], !home.isEmpty {
            hub = URL(fileURLWithPath: home).appendingPathComponent("hub")
        } else {
            hub = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        }
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: hub.path) else { return [] }
        var out: [(name: String, bytes: Int64)] = []
        for entry in entries where entry.hasPrefix("models--") {
            let blobs = hub.appendingPathComponent(entry).appendingPathComponent("blobs")
            guard let files = try? fm.contentsOfDirectory(at: blobs, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else { continue }
            var bytes: Int64 = 0
            for file in files {
                bytes += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
            guard bytes > 0 else { continue }
            let name = entry.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
            out.append((name, bytes))
        }
        return out.sorted { $0.bytes > $1.bytes }
    }
}
