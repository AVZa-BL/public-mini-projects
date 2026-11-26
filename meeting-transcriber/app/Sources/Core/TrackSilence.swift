import Foundation

/// A track that carried nothing audible for a whole call was not really recorded: nothing was playing, or the capture
/// delivered nothing. The engine says the same about its own tracks (mt.py `SILENT_PEAK`).
enum TrackSilence {
    /// The loudest sample (0…1) a silent track may have; the same value as the engine's.
    static let peak = 0.002

    /// The note for the computer's-sound track of a call of at least 10 s that got no buffer, or only silence; nil otherwise.
    static func systemNote(buffers: Int, peak: Double, seconds: Double) -> String? {
        guard seconds >= 10, buffers == 0 || peak < TrackSilence.peak else { return nil }
        return "The computer's sound was silent for the whole call: was anything playing, and is Meeting Transcriber allowed under System Settings > Privacy & Security > Screen & System Audio Recording?"
    }

    /// The note for the microphone track of a call of at least 10 s that got no buffer, or only silence: the microphone was not
    /// really recorded, whatever the speech model later says about it. `delivered` is quoted so the two cases can be told apart
    /// (no buffers: the engine delivered nothing; buffers with a peak of 0: it delivered zeros).
    static func microphoneNote(buffers: Int, peak: Double, seconds: Double) -> String? {
        guard seconds >= 10, buffers == 0 || peak < TrackSilence.peak else { return nil }
        let delivered = buffers == 0 ? "no sound reached it" : "\(buffers) buffers reached it, the loudest sample was \(String(format: "%.3f", peak))"
        return "Your microphone was silent for the whole call (\(delivered)): is the right microphone chosen, is it allowed under System Settings > Privacy & Security > Microphone, and is it not muted? capture.log in the recording's folder has the account."
    }

    /// The early note for a microphone that has delivered buffers but only zeros for `earlyAfter` seconds, or nil while there
    /// is nothing to say. With echo cancellation on, the app starts the microphone again without it (the note says so): a
    /// voice-processing unit that delivers zeros is the one state of the microphone the recording can get out of by itself.
    static func microphoneEarlyNote(seconds: Double, buffers: Int, peak: Double, echoCancelled: Bool) -> String? {
        guard seconds >= earlyAfter, buffers > 0, peak < TrackSilence.peak else { return nil }
        let counters = "\(buffers) buffers, loudest sample \(String(format: "%.3f", peak))"
        return echoCancelled
            ? "No sound has come from your microphone yet (\(counters)), with echo cancellation on; it is being started again without it. If your side still stays silent: is the microphone muted, and is the right one chosen?"
            : "No sound has come from your microphone yet (\(counters)): is it muted, is the right one chosen, and is Meeting Transcriber allowed under System Settings > Privacy & Security > Microphone?"
    }

    /// Whether a microphone that stalled (no buffer for 10 s) is started again without echo cancellation: when it is on
    /// and the input has not delivered one buffer since its latest start (the call's, or a restart's). A voice-processing
    /// unit that delivers nothing at all is the same dead end as one that delivers zeros (microphoneEarlyNote), which that
    /// rule cannot see: it needs a buffer. (The 0 buffers of a 1.9.11 call were the app's own drop of 3-channel buffers,
    /// fixed in 1.10.1, not this case: the engine did deliver.)
    static func restartsWithoutEchoCancellation(deliveredSinceStart: Bool, echoCancelled: Bool) -> Bool {
        return echoCancelled && !deliveredSinceStart
    }

    /// The note for a stalled microphone started again without echo cancellation (restartsWithoutEchoCancellation).
    static let nothingWithEchoCancellationNote = "No sound has come from your microphone for 10 s, with echo cancellation on; it is being started again without it."

    /// The note when the microphone and the output are one Bluetooth device (a headset, a speakerphone). Recording from its
    /// microphone switches it to its call mode (Apple: the quality stays reduced "until the microphone is no longer in
    /// use", support.apple.com/102217). Forcing a music rate (44.1 or 48 kHz) on the device in Audio MIDI Setup during the
    /// call silenced it, also after the recording, until its format was set back (a user's report, macOS 27.2, a Bluetooth
    /// speaker; Mozilla bug 1881609 describes the same with AirPods).
    static func sharedBluetoothNote(device: String) -> String {
        return "Your microphone is “\(device)”, the Bluetooth device the sound plays on: recording from it switches it to its call mode, so what it plays sounds worse until the recording ends. For full sound, choose another microphone as the input (System Settings > Sound > Input): the Mac's built-in one if it has one, or a wired or USB one, and keep “\(device)” as the output. Do not change its format in Audio MIDI Setup during a call: that has silenced it until the format was set back."
    }

    /// The device address in a Bluetooth device half's UID ("DE-61-B1-84-A1-50:input", "…:output"); nil for a UID of any
    /// other form.
    static func bluetoothAddress(_ uid: String) -> String? {
        for suffix in [":input", ":output"] where uid.hasSuffix(suffix) {
            let address = uid.dropLast(suffix.count)
            return address.isEmpty ? nil : String(address)
        }
        return nil
    }

    /// Whether two Bluetooth devices (both known to be Bluetooth, and not one device object) are one device's microphone
    /// and output: when both UIDs have the "<address>:input/output" form, the same address decides and nothing else; for
    /// a UID form not known, the same name.
    static func sameBluetoothDevice(uids: (String, String), names: (String?, String?)) -> Bool {
        if let a = bluetoothAddress(uids.0), let b = bluetoothAddress(uids.1) { return a == b }
        guard let first = names.0, let second = names.1, !first.isEmpty else { return false }
        return first == second
    }

    /// Seconds into a call after which a computer's-sound track that delivers only silence, while the microphone has sound,
    /// is pointed out: the one sign of a missing System Audio Recording permission, which cannot be asked about.
    static let earlyAfter: Double = 20

    /// The early note while the call goes on, or nil while there is nothing to say. `audioTap`: the capture runs through
    /// a Core Audio tap (macOS 14.2+, permission "System Audio Recording Only") rather than ScreenCaptureKit.
    static func earlyNote(seconds: Double, systemBuffers: Int, systemPeak: Double, microphonePeak: Double, audioTap: Bool) -> String? {
        guard seconds >= earlyAfter, systemBuffers > 0, systemPeak < peak, microphonePeak >= peak else { return nil }
        let place = audioTap ? "System Settings > Privacy & Security > Screen & System Audio Recording, in the \"System Audio Recording Only\" list"
                             : "System Settings > Privacy & Security > Screen & System Audio Recording"
        return "No sound has come from the computer yet, although your microphone has: if the other side is speaking, allow Meeting Transcriber under \(place), then start the call again (a build without the project's signing identity has to be allowed again after every update)."
    }
}
