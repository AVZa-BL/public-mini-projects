import Foundation
import AVFoundation
import AudioToolbox
import CoreAudio

/// One chosen input device recorded through its own HAL unit (kAudioUnitSubType_HALOutput) with input on and output off,
/// so no output device is opened. AVAudioEngine cannot do this on macOS: its input and output nodes are one unit bound to
/// the Mac's default devices (Apple developer forums threads 683348 and 71008), and on macOS 26.6 and 27.0.1 an engine
/// pointed at another input device drops it for the default one and delivers no buffer (llfbandit/record PRs 630 and 634);
/// in a 1.10.2 call on macOS 27.2 the chosen built-in microphone gave 0 buffers in 82 s. The order of the calls follows
/// Apple's Technical Note TN2091 (input I/O enabled and output I/O disabled before the device is set), as Chromium's
/// audio_low_latency_input.cc and OBS's mac-audio.c do. Buffers come in the device's own rate and channels, float32,
/// non-interleaved, and are handed on one at a time on this object's delivery queue.
final class DeviceMicrophone {
    enum Failure: LocalizedError {
        case step(String, OSStatus)
        case format

        var errorDescription: String? {
            switch self {
            case .step(let step, let status): return "\(step) failed (Core Audio error \(status))"
            case .format: return "the microphone's format could not be read"
            }
        }
    }

    /// The format the buffers have (the device's rate and channels).
    private(set) var format: AVAudioFormat
    private var unit: AudioUnit?
    private var scratch: AVAudioPCMBuffer
    private let onBuffer: (AVAudioPCMBuffer) -> Void
    private let delivery = DispatchQueue(label: "meeting-transcriber.device-microphone")
    private let lock = NSLock()
    private var running = false

    init(device: AudioObjectID, onBuffer: @escaping (AVAudioPCMBuffer) -> Void) throws {
        var description = AudioComponentDescription(componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
                                                    componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        guard let component = AudioComponentFindNext(nil, &description) else { throw Failure.step("finding the HAL unit", -1) }
        var made: AudioUnit?
        let created = AudioComponentInstanceNew(component, &made)
        guard created == noErr, let au = made else { throw Failure.step("making the HAL unit", created) }
        func set<T>(_ property: AudioUnitPropertyID, _ scope: AudioUnitScope, _ element: AudioUnitElement, _ value: T, _ step: String) throws {
            var copy = value
            let status = AudioUnitSetProperty(au, property, scope, element, &copy, UInt32(MemoryLayout<T>.size))
            if status != noErr {
                AudioComponentInstanceDispose(au)
                throw Failure.step(step, status)
            }
        }
        try set(kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, UInt32(1), "enabling the input")
        try set(kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, UInt32(0), "disabling the output")
        try set(kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, device, "choosing the device")
        var hardware = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let read = AudioUnitGetProperty(au, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &hardware, &size)
        let channels = hardware.mChannelsPerFrame
        let wanted: AVAudioFormat?
        if read != noErr || hardware.mSampleRate <= 0 || channels == 0 {
            wanted = nil
        } else if channels <= 2 {
            wanted = AVAudioFormat(standardFormatWithSampleRate: hardware.mSampleRate, channels: channels)
        } else if let layout = AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | channels) {
            wanted = AVAudioFormat(standardFormatWithSampleRate: hardware.mSampleRate, channelLayout: layout)
        } else {
            wanted = nil
        }
        guard let clientFormat = wanted else {
            AudioComponentInstanceDispose(au)
            throw Failure.format
        }
        // What the unit hands over: float32, non-interleaved, at the device's own rate (the HAL unit converts the sample
        // format, not the rate).
        try set(kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, clientFormat.streamDescription.pointee, "setting the format")
        var deviceFrames: UInt32 = 0
        size = UInt32(MemoryLayout<UInt32>.size)
        _ = AudioUnitGetProperty(au, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &deviceFrames, &size)
        let maxFrames = max(UInt32(4096), deviceFrames)
        try set(kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0, maxFrames, "setting the buffer size")
        guard let buffer = AVAudioPCMBuffer(pcmFormat: clientFormat, frameCapacity: maxFrames) else {
            AudioComponentInstanceDispose(au)
            throw Failure.format
        }
        self.format = clientFormat
        self.scratch = buffer
        self.onBuffer = onBuffer
        self.unit = au
        let callback = AURenderCallbackStruct(
            inputProc: { refCon, flags, timeStamp, bus, frames, _ in
                Unmanaged<DeviceMicrophone>.fromOpaque(refCon).takeUnretainedValue().render(flags, timeStamp, bus, frames)
            },
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque())
        var callbackCopy = callback
        let registered = AudioUnitSetProperty(au, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
                                              &callbackCopy, UInt32(MemoryLayout<AURenderCallbackStruct>.size))
        guard registered == noErr else {
            dispose()
            throw Failure.step("attaching to the input", registered)
        }
        let initialized = AudioUnitInitialize(au)
        guard initialized == noErr else {
            dispose()
            throw Failure.step("initializing the HAL unit", initialized)
        }
    }

    func start() throws {
        lock.lock(); defer { lock.unlock() }
        guard let au = unit else { throw Failure.step("starting the input", -1) }
        let started = AudioOutputUnitStart(au)
        guard started == noErr else { throw Failure.step("starting the input", started) }
        running = true
    }

    /// Stops the input and frees the unit. Never on the audio thread; safe to call twice.
    func stop() {
        lock.lock(); let au = unit; let wasRunning = running; unit = nil; running = false; lock.unlock()
        guard let au = au else { return }
        if wasRunning { _ = AudioOutputUnitStop(au) }
        _ = AudioUnitUninitialize(au)
        _ = AudioComponentInstanceDispose(au)
    }

    private func dispose() {
        if let au = unit { _ = AudioComponentInstanceDispose(au) }
        unit = nil
    }

    deinit { stop() }

    /// On the audio thread: the unit's input is rendered into the scratch buffer and a copy is handed on.
    private func render(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>, _ timeStamp: UnsafePointer<AudioTimeStamp>,
                        _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        guard lock.try() else { return noErr }                 // stopping just now: this cycle is dropped
        defer { lock.unlock() }
        guard running, let au = unit, frames <= scratch.frameCapacity else { return noErr }
        scratch.frameLength = frames
        let status = AudioUnitRender(au, flags, timeStamp, bus, frames, scratch.mutableAudioBufferList)
        guard status == noErr, let copy = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames),
              let source = scratch.floatChannelData, let target = copy.floatChannelData else { return status }
        copy.frameLength = frames
        for channel in 0..<Int(format.channelCount) {
            target[channel].update(from: source[channel], count: Int(frames))
        }
        let handler = onBuffer
        delivery.async { handler(copy) }
        return noErr
    }
}
