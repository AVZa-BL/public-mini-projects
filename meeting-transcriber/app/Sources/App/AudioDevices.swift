import Foundation
import CoreAudio

/// The sound devices of this Mac, as Core Audio lists them: every device with input channels (a microphone, a headset,
/// an interface) and every device with output channels (speakers, headphones, a display). A device is kept by its
/// UID, which stays the same across launches and replugging, and shown by its name.
enum AudioDevices {
    struct Device: Identifiable, Equatable {
        let uid: String
        let name: String
        var id: String { return uid }
    }

    enum Direction {
        case input, output

        var scope: AudioObjectPropertyScope {
            return self == .input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput
        }
    }

    /// The devices of one direction, in Core Audio's order (the built-in ones first, as a rule).
    static func list(_ direction: Direction) -> [Device] {
        return allDeviceIDs().compactMap { id in
            guard channels(of: id, direction) > 0, let uid = uid(of: id), !uid.isEmpty else { return nil }
            return Device(uid: uid, name: name(of: id) ?? uid)
        }
    }

    /// The Core Audio object of the device with this UID, if it is connected now.
    static func id(forUID uid: String) -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var device = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var cfUID = uid as CFString
        let status = withUnsafePointer(to: &cfUID) { pointer in
            AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                       UInt32(MemoryLayout<CFString>.size), pointer, &size, &device)
        }
        guard status == noErr, device != kAudioObjectUnknown else { return nil }
        return device
    }

    /// The UID of the device macOS uses now for this direction (System Settings > Sound).
    static func defaultUID(_ direction: Direction) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: direction == .input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var device = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &device) == noErr,
              device != kAudioObjectUnknown else { return nil }
        return uid(of: device)
    }

    /// Calls `changed` on `queue` whenever a device is connected or disconnected; the returned block keeps the listener
    /// (pass it to `stopWatching`).
    static func watch(queue: DispatchQueue, changed: @escaping () -> Void) -> AudioObjectPropertyListenerBlock? {
        var address = devicesAddress
        let block: AudioObjectPropertyListenerBlock = { _, _ in changed() }
        guard AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, block) == noErr else { return nil }
        return block
    }

    static func stopWatching(_ block: @escaping AudioObjectPropertyListenerBlock, queue: DispatchQueue) {
        var address = devicesAddress
        _ = AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, block)
    }

    /// Whether these two UIDs are one Bluetooth device's microphone and output (a headset, a speakerphone): both devices
    /// are Bluetooth, and they are one device object, or the pair their UIDs or names describe (TrackSilence.sameBluetoothDevice).
    static func sameBluetoothDevice(_ first: String, _ second: String) -> Bool {
        guard let a = id(forUID: first), let b = id(forUID: second), isBluetooth(a), isBluetooth(b) else { return false }
        if a == b { return true }
        return TrackSilence.sameBluetoothDevice(uids: (first, second), names: (name(of: a), name(of: b)))
    }

    /// Whether the device is connected over Bluetooth (classic or LE), from its transport type.
    static func isBluetooth(_ device: AudioObjectID) -> Bool {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyTransportType,
                                                 mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr else { return false }
        return value == kAudioDeviceTransportTypeBluetooth || value == kAudioDeviceTransportTypeBluetoothLE
    }

    // MARK: - Speaker or headphones

    /// What the output device is, as far as Core Audio says (OutputKind.of): its transport, the data source selected on
    /// its output side (an Intel Mac's built-in output: 'ispk' speakers, 'hdpn' headphones), its UID and the terminal
    /// types of its output streams. Reading these needs no permission.
    static func outputKind(of device: AudioObjectID) -> OutputKind {
        return OutputKind.of(transport: number(of: device, kAudioDevicePropertyTransportType) ?? 0, dataSource: dataSource(of: device),
                             uid: uid(of: device) ?? "", terminals: outputTerminalTypes(of: device))
    }

    /// The data source selected on the device's output side, when it has one.
    static func dataSource(of device: AudioObjectID) -> UInt32? {
        var address = dataSourceAddress
        guard AudioObjectHasProperty(device, &address) else { return nil }
        return number(of: device, kAudioDevicePropertyDataSource, scope: kAudioObjectPropertyScopeOutput)
    }

    /// The terminal types of the device's output streams. Only streams whose direction is output (0) count: macOS adds
    /// input streams to an output device while an app runs voice processing, as a call app does.
    private static func outputTerminalTypes(of device: AudioObjectID) -> [UInt32] {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: kAudioObjectPropertyScopeOutput,
                                                 mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return [] }
        var streams = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &streams) == noErr else { return [] }
        return streams.compactMap { stream in
            guard number(of: stream, kAudioStreamPropertyDirection) == 0 else { return nil }
            return number(of: stream, kAudioStreamPropertyTerminalType)
        }
    }

    /// Calls `changed` on `queue` when the Mac's current output changes (System Settings, the menu bar, headphones
    /// plugged into an Apple-silicon Mac, which makes them a device of their own); keep the block for `stopWatchingDefaultOutput`.
    static func watchDefaultOutput(queue: DispatchQueue, changed: @escaping () -> Void) -> AudioObjectPropertyListenerBlock? {
        var address = defaultOutputAddress
        let block: AudioObjectPropertyListenerBlock = { _, _ in changed() }
        guard AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, block) == noErr else { return nil }
        return block
    }

    static func stopWatchingDefaultOutput(_ block: @escaping AudioObjectPropertyListenerBlock, queue: DispatchQueue) {
        var address = defaultOutputAddress
        _ = AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, block)
    }

    /// Calls `changed` on `queue` when the device's output data source changes (headphones plugged into an Intel Mac's
    /// jack); nil when the device has no data source.
    static func watchDataSource(of device: AudioObjectID, queue: DispatchQueue, changed: @escaping () -> Void) -> AudioObjectPropertyListenerBlock? {
        var address = dataSourceAddress
        guard AudioObjectHasProperty(device, &address) else { return nil }
        let block: AudioObjectPropertyListenerBlock = { _, _ in changed() }
        guard AudioObjectAddPropertyListenerBlock(device, &address, queue, block) == noErr else { return nil }
        return block
    }

    static func stopWatchingDataSource(of device: AudioObjectID, _ block: @escaping AudioObjectPropertyListenerBlock, queue: DispatchQueue) {
        var address = dataSourceAddress
        _ = AudioObjectRemovePropertyListenerBlock(device, &address, queue, block)
    }

    private static let defaultOutputAddress = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDefaultOutputDevice,
                                                                         mScope: kAudioObjectPropertyScopeGlobal,
                                                                         mElement: kAudioObjectPropertyElementMain)
    private static let dataSourceAddress = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyDataSource,
                                                                      mScope: kAudioObjectPropertyScopeOutput,
                                                                      mElement: kAudioObjectPropertyElementMain)

    // MARK: - Core Audio lookups

    /// A UInt32 property of a device or a stream (the first item of an array property such as the selected data sources).
    private static func number(of object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                               scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr else { return nil }
        return value
    }

    private static let devicesAddress = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices,
                                                                   mScope: kAudioObjectPropertyScopeGlobal,
                                                                   mElement: kAudioObjectPropertyElementMain)

    private static func allDeviceIDs() -> [AudioObjectID] {
        var address = devicesAddress
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioObjectID](repeating: AudioObjectID(kAudioObjectUnknown), count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    /// How many channels the device has in this direction (0: it has none, so it is not a microphone / not an output).
    private static func channels(of device: AudioObjectID, _ direction: Direction) -> Int {
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration,
                                                 mScope: direction.scope, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(device, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let list = AudioBufferList.allocate(maximumBuffers: max(1, (Int(size) - MemoryLayout<UInt32>.size) / MemoryLayout<AudioBuffer>.size))
        defer { list.unsafeMutablePointer.deallocate() }
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, list.unsafeMutablePointer) == noErr else { return 0 }
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    static func uid(of device: AudioObjectID) -> String? {
        return string(of: device, kAudioDevicePropertyDeviceUID)
    }

    static func name(of device: AudioObjectID) -> String? {
        return string(of: device, kAudioObjectPropertyName)
    }

    private static func string(of device: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                                 mElement: kAudioObjectPropertyElementMain)
        var value: CFString? = nil
        var size = UInt32(MemoryLayout<CFString?>.size)
        guard AudioObjectGetPropertyData(device, &address, 0, nil, &size, &value) == noErr, let found = value else { return nil }
        return found as String
    }
}
