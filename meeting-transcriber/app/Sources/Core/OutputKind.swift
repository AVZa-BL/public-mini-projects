import Foundation

/// What the Mac's sound comes out of, as far as Core Audio says: a loudspeaker (the microphone hears through it what the
/// Mac plays, so a call needs no second track), headphones, or unknown. Core Audio gives a device's connection, not its
/// form, so most outputs are unknown: a Bluetooth device says nothing about being a speaker or headphones, a USB one
/// can be a headset or a desk speaker, a virtual or aggregate device can be anything. Unknown never unticks the
/// computer's-sound box by itself; the user's own untick teaches the app a speaker (OutputFollowing).
enum OutputKind: Equatable {
    case speaker, headphones, unknown

    /// A four-character code as Core Audio returns it (big-endian in a UInt32): 'bltn' is 0x626C746E.
    static func code(_ text: String) -> UInt32 {
        return text.utf8.reduce(UInt32(0)) { $0 << 8 | UInt32($1) }
    }

    // kAudioDeviceTransportType… (CoreAudio/AudioHardwareBase.h).
    static let builtIn = code("bltn"), hdmi = code("hdmi"), displayPort = code("dprt"), airPlay = code("airp")
    // Data sources of a built-in output (IOKit/audio/IOAudioTypes.h, kIOAudioOutputPortSubType…).
    static let internalSpeaker = code("ispk"), externalSpeaker = code("espk"), headphonesSource = code("hdpn")
    // Terminal types of an output stream that mean headphones: kAudioStreamTerminalTypeHeadphones (AudioHardwareBase.h),
    // and the USB Audio class's headphones (0x0302) and headset (0x0402) (IOKit/audio/IOAudioTypes.h).
    static let headphonesTerminals: Set<UInt32> = [code("hdph"), 0x0302, 0x0402]
    /// The UIDs Apple-silicon and T2 Macs give their built-in speakers and their headphone jack, each a device of its own
    /// (seen in device listings of such Macs; Apple calls the UID's content a black box, so they are a hint after the data source).
    static let builtInSpeakerUID = "BuiltInSpeakerDevice", builtInHeadphonesUID = "BuiltInHeadphoneOutputDevice"

    /// transport: kAudioDevicePropertyTransportType (0 when it could not be read); dataSource: the selected
    /// kAudioDevicePropertyDataSource on the output side, when the device has one (an Intel Mac's built-in output: 'ispk'
    /// speakers, 'hdpn' something in the jack); terminals: kAudioStreamPropertyTerminalType of its output streams.
    static func of(transport: UInt32, dataSource: UInt32?, uid: String, terminals: [UInt32]) -> OutputKind {
        if terminals.contains(where: { headphonesTerminals.contains($0) }) { return .headphones }
        switch transport {
        case builtIn:
            if let source = dataSource {
                if source == internalSpeaker || source == externalSpeaker { return .speaker }
                if source == headphonesSource { return .headphones }      // desk speakers in the jack look the same: the box stays
            }
            if uid == builtInSpeakerUID { return .speaker }
            if uid == builtInHeadphonesUID { return .headphones }
            return .unknown
        case hdmi, displayPort, airPlay:
            return .speaker                // a TV, a monitor's speakers, a HomePod; a monitor's own headphone jack is learned from a tick
        default:
            return .unknown
        }
    }
}

/// The box "Also record the sound this computer plays" following where the sound goes. While the sound plays through a
/// speaker the microphone hears both sides of a call, without echo cancellation in its own words, so the box is unticked
/// by itself; it is ticked again when the sound goes back to headphones, if it was unticked by itself. What the user
/// does with the box is remembered for the output the sound goes to at that moment: unticked = that output is a speaker
/// (how a Bluetooth speaker is learned), ticked = that output keeps two tracks (also for one Core Audio calls a speaker).
struct OutputFollowing: Equatable {
    var speakers: Set<String> = []        // outputs (key) the user unticked the box with
    var twoTracks: Set<String> = []       // outputs (key) the user ticked it with

    /// What a choice is remembered under: the UID, and the selected data source when the device has one, since an Intel
    /// Mac's built-in output is one device with one UID for its speakers and its jack ("<uid>#6973706b" for 'ispk').
    static func key(uid: String, dataSource: UInt32?) -> String {
        guard let source = dataSource else { return uid }
        return uid + "#" + String(source, radix: 16)
    }

    /// Whether the output (by its key) is a speaker: never when Core Audio says headphones (an untick made with them, for
    /// a recording without the other side, must not untick the box later for a call on them); else what the user chose
    /// with it, else Core Audio's word.
    func isSpeaker(uid: String, kind: OutputKind) -> Bool {
        if kind == .headphones { return false }
        if twoTracks.contains(uid) { return false }
        if speakers.contains(uid) { return true }
        return kind == .speaker
    }

    mutating func userChose(ticked: Bool, uid: String) {
        if ticked {
            speakers.remove(uid)
            twoTracks.insert(uid)
        } else {
            twoTracks.remove(uid)
            speakers.insert(uid)
        }
    }

    /// What the box becomes now that the sound goes to an output that is (or is not) a speaker; nil = it stays.
    /// untickedByApp: the box was unticked by itself, not by the user.
    static func box(ticked: Bool, untickedByApp: Bool, speaker: Bool) -> Bool? {
        if speaker && ticked { return false }
        if !speaker && !ticked && untickedByApp { return true }
        return nil
    }

    /// The line under the box after it was unticked by itself.
    static func untickedNote(output: String) -> String {
        return "Unticked by itself: the sound plays through “\(output)”, a speaker, so the microphone hears both sides and "
            + "records them as one track, in its own words. The box is ticked again when the sound goes to an output that is "
            + "not a speaker (headphones, or a device you have not unticked it with). If the call plays to other headphones, "
            + "choose them under “Take the computer's sound from”; tick the box yourself to record two tracks with “\(output)” "
            + "(remembered)."
    }
}
