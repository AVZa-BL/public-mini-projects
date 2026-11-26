// Diagnoses the capture of the computer's sound through a Core Audio process tap, outside the app: builds the tap and
// its aggregate device in several ways, runs each for a few seconds while something plays, and prints what the HAL
// reports (input streams, tap list, running state, IO callbacks, loudest sample). For a Mac on which the app's track
// stays silent although tap, aggregate and start all returned noErr.
//
//   cd meeting-transcriber/app
//   swiftc -O diagnose_tap.swift -o /tmp/diagnose_tap && /tmp/diagnose_tap        # play music or a call meanwhile
//
// macOS 14.2 or newer. The first run asks for the System Audio Recording permission for the Terminal (the tool waits for
// Enter after that), and the last four variants for the microphone: speak during those, so the microphone's loudest
// sample says whether it delivered sound beside the tap. Paste the whole output.
import Foundation
import CoreAudio
import AVFoundation

// MARK: - Core Audio lookups

func address(_ selector: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    return AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func string(of object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var value: CFString? = nil
    var size = UInt32(MemoryLayout<CFString?>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr, let found = value else { return nil }
    return found as String
}

func uint32(of object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
    var addr = address(selector, scope)
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr else { return nil }
    return value
}

func float64(of object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> Double? {
    var addr = address(selector)
    var value: Float64 = 0
    var size = UInt32(MemoryLayout<Float64>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value) == noErr else { return nil }
    return value
}

func objectList(of object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> [AudioObjectID] {
    var addr = address(selector, scope)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

func channels(of device: AudioObjectID, _ scope: AudioObjectPropertyScope) -> Int {
    var addr = address(kAudioDevicePropertyStreamConfiguration, scope)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(device, &addr, 0, nil, &size) == noErr, size > 0 else { return 0 }
    let list = AudioBufferList.allocate(maximumBuffers: max(1, (Int(size) - MemoryLayout<UInt32>.size) / MemoryLayout<AudioBuffer>.size))
    defer { free(list.unsafeMutablePointer) }      // allocate() uses calloc
    guard AudioObjectGetPropertyData(device, &addr, 0, nil, &size, list.unsafeMutablePointer) == noErr else { return 0 }
    return list.reduce(0) { $0 + Int($1.mNumberChannels) }
}

func defaultOutputUID() -> String? {
    var addr = address(kAudioHardwarePropertyDefaultOutputDevice)
    var device = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &device) == noErr else { return nil }
    return string(of: device, kAudioDevicePropertyDeviceUID)
}

func builtInOutputUID() -> String? {
    for id in objectList(of: AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDevices)
    where uint32(of: id, kAudioDevicePropertyTransportType) == kAudioDeviceTransportTypeBuiltIn && channels(of: id, kAudioObjectPropertyScopeOutput) > 0 {
        return string(of: id, kAudioDevicePropertyDeviceUID)
    }
    return nil
}

func ownProcessObject() -> AudioObjectID? {
    var pid = ProcessInfo.processInfo.processIdentifier
    var addr = address(kAudioHardwarePropertyTranslatePIDToProcessObject)
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = withUnsafePointer(to: &pid) { p in
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, UInt32(MemoryLayout<pid_t>.size), p, &size, &object)
    }
    return status == noErr && object != kAudioObjectUnknown ? object : nil
}

func name(ofUID uid: String?) -> String {
    guard let uid = uid else { return "none" }
    var addr = address(kAudioHardwarePropertyTranslateUIDToDevice)
    var device = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    var cf = uid as CFString
    let status = withUnsafePointer(to: &cf) { p in
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, UInt32(MemoryLayout<CFString>.size), p, &size, &device)
    }
    guard status == noErr, device != kAudioObjectUnknown else { return uid }
    return "\(string(of: device, kAudioObjectPropertyName) ?? "?") (\(uid))"
}

// MARK: - The counters the IO callbacks fill (globals: the C-function variant cannot capture)

/// The aggregate's input holds the tap's streams and, when the clock device is a duplex one (a headset, an interface),
/// that device's own microphone too. As the app does (SystemAudioTap.deliver), the tap's buffers are the last ones whose
/// channels add up to the tap's channel count; the headline peak is taken over those only, so a headset microphone
/// cannot make a silent tap look alive. Every buffer's own peak is printed as well.
final class Stats {
    let wantedChannels: Int
    let float32: Bool
    var callbacks = 0
    var buffers = 0
    var tapPeak: Float = 0
    var otherPeak: Float = 0
    var layoutMismatches = 0
    var firstLayout = ""

    init(wantedChannels: Int = 2, float32: Bool = true) {
        self.wantedChannels = wantedChannels
        self.float32 = float32
    }

    private func peak(of buffer: AudioBuffer) -> Float {
        guard float32, let data = buffer.mData, buffer.mDataByteSize >= 4 else { return 0 }
        let samples = data.assumingMemoryBound(to: Float.self)
        var p: Float = 0
        for i in 0..<Int(buffer.mDataByteSize) / 4 { p = max(p, abs(samples[i])) }
        return p
    }

    func take(_ input: UnsafePointer<AudioBufferList>) {
        callbacks += 1
        let list = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        buffers += list.count
        // The tap's buffers: from the end, until the channels add up.
        var tapIndices = Set<Int>()
        var channels = 0
        for index in stride(from: list.count - 1, through: 0, by: -1) where list[index].mDataByteSize > 0 {
            tapIndices.insert(index)
            channels += Int(list[index].mNumberChannels)
            if channels >= wantedChannels { break }
        }
        if channels != wantedChannels { layoutMismatches += 1; tapIndices = [] }
        var peaks: [String] = []
        for (index, buffer) in list.enumerated() {
            let p = peak(of: buffer)
            if tapIndices.contains(index) { tapPeak = max(tapPeak, p) } else { otherPeak = max(otherPeak, p) }
            if firstLayout.isEmpty { peaks.append("\(buffer.mNumberChannels)ch×\(buffer.mDataByteSize)B\(tapIndices.contains(index) ? " (tap)" : "")") }
        }
        if firstLayout.isEmpty { firstLayout = "\(list.count) buffer(s): " + peaks.joined(separator: ", ") }
    }

    var line: String {
        let peakText = float32 ? String(format: "%.4f", tapPeak) : "not computed (tap format is not float32)"
        var text = "\(callbacks) callbacks, \(buffers) buffers, tap peak \(peakText)"
        if otherPeak > 0 { text += String(format: ", other input buffers peak %.4f (the clock device's own microphone, not the tap)", otherPeak) }
        if layoutMismatches > 0 { text += ", \(layoutMismatches) callbacks with an unexpected layout" }
        return text + ", first callback: \(firstLayout.isEmpty ? "-" : firstLayout)"
    }
}

var functionStats = Stats()
let functionProc: AudioDeviceIOProc = { _, _, input, _, _, _, _ in
    functionStats.take(input)
    return noErr
}

// MARK: - One variant

struct Variant {
    let name: String
    let clock: String?          // sub-device UID the aggregate is clocked on
    let restrictTo: String?     // CATapDescription.deviceUID
    let autoStart: Bool
    let useFunction: Bool       // AudioDeviceCreateIOProcID (function) instead of WithBlock
    var drift = true            // kAudioSubTapDriftCompensationKey
    var settle = 0.0            // seconds between creating the aggregate and registering/starting the IOProc
}

/// A failed property read prints as n/a, never as a number that could be taken for a state.
func flag(_ value: UInt32?) -> String { return value.map(String.init) ?? "n/a" }

@available(macOS 14.2, *)
func run(_ v: Variant, seconds: Double) {
    print("\n== \(v.name)")
    print("   clock: \(name(ofUID: v.clock)); tap limited to: \(v.restrictTo.map { name(ofUID: $0) } ?? "no device (global)"); auto start: \(v.autoStart); IOProc: \(v.useFunction ? "function" : "block"); drift compensation: \(v.drift); settle: \(v.settle) s")
    guard let clock = v.clock else { print("   no output device to clock the aggregate on: skipped"); return }
    let desc = CATapDescription(stereoGlobalTapButExcludeProcesses: ownProcessObject().map { [$0] } ?? [])
    desc.name = "diagnose_tap"
    desc.isPrivate = true
    desc.muteBehavior = CATapMuteBehavior.unmuted
    if let uid = v.restrictTo { desc.deviceUID = uid }
    var tap = AudioObjectID(kAudioObjectUnknown)
    let created = AudioHardwareCreateProcessTap(desc, &tap)
    guard created == noErr else { print("   AudioHardwareCreateProcessTap failed: \(created)"); return }
    defer { AudioHardwareDestroyProcessTap(tap) }
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var formatAddr = address(kAudioTapPropertyFormat)
    let formatStatus = AudioObjectGetPropertyData(tap, &formatAddr, 0, nil, &size, &asbd)
    print("   tap \(tap): format status \(formatStatus): \(Int(asbd.mSampleRate)) Hz, \(asbd.mChannelsPerFrame) ch, \(asbd.mBitsPerChannel) bit, flags \(asbd.mFormatFlags); tap UID property: \(string(of: tap, kAudioTapPropertyUID) ?? "-"); description uuid: \(desc.uuid.uuidString)")

    let properties: [String: Any] = [
        kAudioAggregateDeviceNameKey: "diagnose_tap aggregate",
        kAudioAggregateDeviceUIDKey: "diagnose-tap." + desc.uuid.uuidString,
        kAudioAggregateDeviceMainSubDeviceKey: clock,
        kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: clock]],
        kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: desc.uuid.uuidString, kAudioSubTapDriftCompensationKey: v.drift]],
        kAudioAggregateDeviceTapAutoStartKey: v.autoStart,
        kAudioAggregateDeviceIsPrivateKey: true,
        kAudioAggregateDeviceIsStackedKey: false,
    ]
    var aggregate = AudioObjectID(kAudioObjectUnknown)
    let made = AudioHardwareCreateAggregateDevice(properties as CFDictionary, &aggregate)
    guard made == noErr else { print("   AudioHardwareCreateAggregateDevice failed: \(made)"); return }
    defer { AudioHardwareDestroyAggregateDevice(aggregate) }
    let inputStreams = objectList(of: aggregate, kAudioDevicePropertyStreams, kAudioObjectPropertyScopeInput)
    let outputStreams = objectList(of: aggregate, kAudioDevicePropertyStreams, kAudioObjectPropertyScopeOutput)
    func tapListCount() -> String {
        var tapListAddr = address(kAudioAggregateDevicePropertyTapList)
        var tapList: CFArray? = nil
        var tapListSize = UInt32(MemoryLayout<CFArray?>.size)
        let status = AudioObjectGetPropertyData(aggregate, &tapListAddr, 0, nil, &tapListSize, &tapList)
        guard status == noErr, let list = tapList else { return "status \(status)" }
        return "\(CFArrayGetCount(list)) tap(s)"
    }
    print("   aggregate \(aggregate): input streams \(inputStreams.count) (\(channels(of: aggregate, kAudioObjectPropertyScopeInput)) ch), output streams \(outputStreams.count) (\(channels(of: aggregate, kAudioObjectPropertyScopeOutput)) ch), tap list \(tapListCount()), nominal rate \(Int(float64(of: aggregate, kAudioDevicePropertyNominalSampleRate) ?? 0)), running before start \(flag(uint32(of: aggregate, kAudioDevicePropertyDeviceIsRunning)))")
    if v.settle > 0 {
        Thread.sleep(forTimeInterval: v.settle)
        print("   after \(v.settle) s settling: input streams \(objectList(of: aggregate, kAudioDevicePropertyStreams, kAudioObjectPropertyScopeInput).count), tap list \(tapListCount())")
    }

    let float32 = formatStatus == noErr && asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0 && asbd.mBitsPerChannel == 32
    let wanted = formatStatus == noErr ? Int(asbd.mChannelsPerFrame) : 2
    let stats = Stats(wantedChannels: wanted, float32: float32)
    functionStats = Stats(wantedChannels: wanted, float32: float32)
    var proc: AudioDeviceIOProcID?
    let queue = DispatchQueue(label: "diagnose_tap.io", qos: .userInteractive)
    let registered: OSStatus
    if v.useFunction {
        registered = AudioDeviceCreateIOProcID(aggregate, functionProc, nil, &proc)
    } else {
        registered = AudioDeviceCreateIOProcIDWithBlock(&proc, aggregate, queue) { _, input, _, _, _ in stats.take(input) }
    }
    guard registered == noErr, let ioProc = proc else { print("   IOProc registration failed: \(registered)"); return }
    defer { AudioDeviceDestroyIOProcID(aggregate, ioProc) }
    let started = AudioDeviceStart(aggregate, ioProc)
    print("   AudioDeviceStart: \(started)")
    guard started == noErr else { return }
    Thread.sleep(forTimeInterval: 1)
    print("   after 1 s: aggregate running \(flag(uint32(of: aggregate, kAudioDevicePropertyDeviceIsRunning))), running somewhere \(flag(uint32(of: aggregate, kAudioDevicePropertyDeviceIsRunningSomewhere))), tap list \(tapListCount()), callbacks so far \((v.useFunction ? functionStats : stats).callbacks)")
    Thread.sleep(forTimeInterval: seconds - 1)
    _ = AudioDeviceStop(aggregate, ioProc)
    queue.sync {}
    print("   result: \((v.useFunction ? functionStats : stats).line)")
}

// MARK: - The microphone beside the tap, as in the app (an AVAudioEngine input tap on the default input)

/// The app opens the microphone in one of two ways (CallRecorder.bringUpMicrophone): with Apple's voice processing
/// (echo cancellation, a VoiceProcessingIO unit that is also an output client of the current output device) when no
/// microphone is chosen in its settings, as a plain input unit when one is.
/// Beside the buffer count it keeps the loudest sample (0…1) the microphone delivered, so "the microphone ran" and "the
/// microphone had sound" are told apart (the app's capture.log prints the same two numbers), and it counts the engine's
/// configuration-change notifications: the app restarts its microphone on each one ("The microphone changed"), and a
/// Bluetooth headset that is the Mac's input and output raises one when its profile or sample rate changes, as when an
/// aggregate device is built on it.
final class MicrophoneBeside {
    let engine = AVAudioEngine()
    var buffers = 0
    var peak: Float = 0
    var configurationChanges = 0
    private var observer: NSObjectProtocol?
    private var format = ""
    func start(voiceProcessing: Bool) -> String {
        let input = engine.inputNode
        var mode = "plain input"
        if voiceProcessing {
            let on = (try? input.setVoiceProcessingEnabled(true)) != nil
            if on, #available(macOS 14.0, *) {
                input.voiceProcessingOtherAudioDuckingConfiguration =
                    AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
            }
            mode = on ? "voice processing on (echo cancellation, as the app with no chosen microphone)" : "voice processing refused, plain input"
        }
        engine.prepare()
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { return "no input device (\(format)), \(mode)" }
        self.format = "\(Int(format.sampleRate)) Hz \(format.channelCount) ch"
        input.installTap(onBus: 0, bufferSize: 4096, format: nil) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.buffers += 1
            guard let data = buffer.floatChannelData else { return }
            for channel in 0..<Int(buffer.format.channelCount) {
                for i in 0..<Int(buffer.frameLength) { self.peak = max(self.peak, abs(data[channel][i])) }
            }
        }
        observer = NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil) { [weak self] _ in
            self?.configurationChanges += 1
        }
        do { try engine.start() } catch { return "engine.start failed: \(error.localizedDescription), \(mode)" }
        return "running, \(self.format), \(mode)"
    }
    /// The counters so far, without stopping: "N buffers, peak P, K configuration changes (engine running/stopped)".
    var report: String {
        return "\(buffers) microphone buffers, peak \(String(format: "%.4f", peak)), \(configurationChanges) configuration change(s), engine \(engine.isRunning ? "running" : "NOT running")"
    }
    func stop() -> String {
        if let observer = observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? engine.inputNode.setVoiceProcessingEnabled(false)
        return report
    }
}

// MARK: - Main

setvbuf(stdout, nil, _IONBF, 0)      // every line reaches a pipe or file at once, even if a later step aborts
guard #available(macOS 14.2, *) else { print("macOS 14.2 or newer is needed"); exit(1) }
print("diagnose_tap on macOS \(ProcessInfo.processInfo.operatingSystemVersionString), pid \(ProcessInfo.processInfo.processIdentifier)")
let defaultOut = defaultOutputUID()
let builtIn = builtInOutputUID()
print("default output: \(name(ofUID: defaultOut)); built-in output: \(name(ofUID: builtIn)); own process object: \(ownProcessObject().map(String.init) ?? "none")")
// The clock the app uses (SystemAudioTap.buildAggregate): the built-in output where there is one, else the current output.
let appClock = builtIn ?? defaultOut
if builtIn == nil { print("no built-in output device found: the app would clock the aggregate on the current output, and so do A, B, E, H, I and G here") }
if defaultOut == builtIn { print("the default output is the built-in output: C repeats A (a second run of the same construction, as a reproducibility check)") }

// macOS asks for the System Audio Recording permission asynchronously, at the first tap; a variant that runs while the
// dialog is still open would report silence for a reason unrelated to its construction. One throwaway tap first, then a
// pause until the user is ready (only when a person is at the keyboard).
if #available(macOS 14.2, *) {
    let probe = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
    probe.isPrivate = true
    var probeTap = AudioObjectID(kAudioObjectUnknown)
    if AudioHardwareCreateProcessTap(probe, &probeTap) == noErr { AudioHardwareDestroyProcessTap(probeTap) }
}
if isatty(0) != 0 {
    print("\nIf macOS has just asked whether the Terminal may record the system audio, click Allow. Start the music or the call now, then press Enter.")
    _ = readLine()
}
print("Each variant runs 6 s (about 95 s in all); the last four run with the microphone open, in the app's two ways, beside the 1.9.10 and the 1.9.11 construction. Speak into the microphone during those.")
let seconds = 6.0
let variants = [
    Variant(name: "A  as the app 1.9.9 and 1.9.10: global tap, clock = \(builtIn == nil ? "current" : "built-in") output, auto start, block IOProc, drift compensation", clock: appClock, restrictTo: nil, autoStart: true, useFunction: false),
    Variant(name: "B  same as A, auto start off", clock: appClock, restrictTo: nil, autoStart: false, useFunction: false),
    Variant(name: "C  same as A, clock = default output (as the app from 1.9.11 on, with no chosen output)", clock: defaultOut, restrictTo: nil, autoStart: true, useFunction: false),
    Variant(name: "D  tap limited to the default output, clock = default output (as the app from 1.9.11 on, with that output chosen)", clock: defaultOut, restrictTo: defaultOut, autoStart: true, useFunction: false),
    Variant(name: "E  same as A, function IOProc instead of block", clock: appClock, restrictTo: nil, autoStart: true, useFunction: true),
    Variant(name: "F  clock = default output, auto start off, function IOProc (three changes against A)", clock: defaultOut, restrictTo: nil, autoStart: false, useFunction: true),
    Variant(name: "H  same as A, 2 s between creating the aggregate and starting it", clock: appClock, restrictTo: nil, autoStart: true, useFunction: false, settle: 2),
    Variant(name: "I  same as A, drift compensation off", clock: appClock, restrictTo: nil, autoStart: true, useFunction: false, drift: false),
]
for v in variants { if #available(macOS 14.2, *) { run(v, seconds: seconds) } }
// The microphone beside the tap: variant A (the 1.9.10 construction, clocked on the built-in output) and variant C (the
// 1.9.11 construction, clocked on the current output: with a Bluetooth headset, the very device the microphone is), each
// with the microphone open in the app's two ways. The microphone's counters are printed before the aggregate is built,
// 1.5 s after it, and at the end: a microphone that delivered sound before the aggregate and none after it, or whose
// engine stopped (a configuration change the app would answer with a restart), is the finding.
for (letter, index) in [("G", 0), ("J", 2)] {
    for voiceProcessing in [false, true] {
        let v = variants[index]
        print("\n== \(letter)\(voiceProcessing ? "2" : "1")  variant \(v.name.prefix(1)) again while the microphone is open through AVAudioEngine, \(voiceProcessing ? "with voice processing (the app with no chosen microphone)" : "as a plain input (the app with a chosen microphone)")")
        let microphone = MicrophoneBeside()
        print("   microphone: \(microphone.start(voiceProcessing: voiceProcessing))")
        Thread.sleep(forTimeInterval: 2)
        print("   microphone before the aggregate: \(microphone.report)")
        if #available(macOS 14.2, *) {
            DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) { print("   microphone 1.5 s after the aggregate was built: \(microphone.report)") }
            run(v, seconds: seconds)
        }
        print("   microphone at the end: \(microphone.stop())")
    }
}
print("\nDone. A variant with callbacks > 0 and tap peak > 0.01 while sound was playing is a working construction; one with callbacks > 0 and tap peak 0 gets the data but no sound (permission, or nothing was playing); callbacks 0 means the HAL never drove the aggregate. For G and J: a microphone whose buffers stop growing, or whose peak stays at 0 while you spoke, after the aggregate was built, is the microphone problem; a configuration change count above 0 is what makes the app say \"The microphone changed\".")
