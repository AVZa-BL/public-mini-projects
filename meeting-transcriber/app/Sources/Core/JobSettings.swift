import Foundation

/// The languages offered in the two drop-downs: the most used first, then the rest A-Z.
enum Languages {
    static let common = ["English", "Russian", "Spanish", "German", "French", "Italian", "Portuguese",
                         "Ukrainian", "Polish", "Turkish", "Arabic", "Hindi", "Chinese", "Japanese", "Korean"]

    /// Every language Whisper recognises (mt.py WHISPER_NAMES), each by a name mt.py's whisper_code accepts
    /// (tests/test_mt.py checks every one).
    static let spokenAll = menuOrder([
        "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Assamese", "Azerbaijani", "Bashkir", "Basque",
        "Belarusian", "Bengali", "Bosnian", "Breton", "Bulgarian", "Burmese", "Cantonese", "Catalan", "Chinese", "Croatian",
        "Czech", "Danish", "Dutch", "English", "Estonian", "Faroese", "Finnish", "French", "Galician", "Georgian", "German",
        "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew", "Hindi", "Hungarian", "Icelandic", "Indonesian",
        "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Korean", "Lao", "Latin", "Latvian", "Lingala",
        "Lithuanian", "Luxembourgish", "Macedonian", "Malagasy", "Malay", "Malayalam", "Maltese", "Maori", "Marathi",
        "Mongolian", "Nepali", "Norwegian", "Norwegian Nynorsk", "Occitan", "Pashto", "Persian", "Polish", "Portuguese",
        "Punjabi", "Romanian", "Russian", "Sanskrit", "Serbian", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian",
        "Somali", "Spanish", "Sundanese", "Swahili", "Swedish", "Tagalog", "Tajik", "Tamil", "Tatar", "Telugu", "Thai",
        "Tibetan", "Turkish", "Turkmen", "Ukrainian", "Urdu", "Uzbek", "Vietnamese", "Welsh", "Yiddish", "Yoruba"])

    /// The languages a transcript can be translated into: Whisper's without the three the offline model (NLLB-200) does
    /// not have (Breton, Hawaiian, Latin), with Traditional Chinese and Zulu, which it has, and the two Portuguese that
    /// TranslateGemma keeps apart (pt-BR, pt-PT; NLLB and Hy-MT3 write both as Portuguese); each by a name mt.py's nllb_code
    /// accepts (tests/test_mt.py checks every one). Claude translates into any of them too.
    static let targetAll = menuOrder(spokenAll.filter { !["Breton", "Hawaiian", "Latin"].contains($0) }
                                     + ["Traditional Chinese", "Zulu", "Brazilian Portuguese", "European Portuguese"])

    /// The translation drop-down's list: with Claude also the three the offline model lacks, which Claude translates
    /// into from their name (as the "Others" field allowed before 1.10.6).
    static func targets(offline: Bool) -> [String] {
        return offline ? targetAll : menuOrder(targetAll + ["Breton", "Hawaiian", "Latin"])
    }

    /// The common languages first (in their order), then the others A-Z, each once.
    static func menuOrder(_ names: [String]) -> [String] {
        let first = common.filter { names.contains($0) }
        let rest = Set(names).subtracting(first).sorted()
        return first + rest
    }

    /// The names that contain the search text (case and accents ignored); everything for an empty search.
    static func matching(_ query: String, in names: [String]) -> [String] {
        let wanted = query.trimmingCharacters(in: .whitespaces)
        guard !wanted.isEmpty else { return names }
        return names.filter { $0.range(of: wanted, options: [.caseInsensitive, .diacriticInsensitive]) != nil }
    }

    /// The chosen languages with names typed into the old "Others" field ("czech, Dutch") added, each spelled as in
    /// `names` when it is one of them (a name the list does not have is kept as typed), without duplicates.
    static func merged(_ chosen: [String], typed: String, names: [String]) -> [String] {
        return targets(checked: chosen, custom: typed).map { name in
            names.first { $0.caseInsensitiveCompare(name) == .orderedSame } ?? name
        }
    }

    /// The text of a drop-down's button: the chosen languages, the first three and how many more, or `none`.
    static func summary(_ chosen: [String], none: String) -> String {
        if chosen.isEmpty { return none }
        if chosen.count <= 3 { return chosen.joined(separator: ", ") }
        return chosen.prefix(3).joined(separator: ", ") + " +\(chosen.count - 3)"
    }

    /// Checked languages followed by the typed ones ("Czech, Dutch"), without empties or duplicates.
    static func targets(checked: [String], custom: String) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        let typed = custom.split(whereSeparator: { $0 == "," || $0 == ";" || $0 == "\n" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
        for name in checked + typed where !name.isEmpty && !seen.contains(name.lowercased()) {
            seen.insert(name.lowercased())
            result.append(name)
        }
        return result
    }
}

/// How finely speech is cut before each language decision (mt.py --granularity). Each chunk gets one language and
/// costs a full Whisper pass, so finer follows quick switches between languages but takes longer.
enum Granularity: String, CaseIterable, Identifiable {
    case auto, adaptive, coarse, normal, fine, finest
    var id: String { return rawValue }

    var label: String { return rawValue.prefix(1).uppercased() + rawValue.dropFirst() }

    var hint: String {
        switch self {
        case .auto: return "Finest when several languages are chosen (measured: on a 10-minute three-language dialogue normal lost 68 of 167 sentences, fine 18, finest 4), normal with one (it is forced), otherwise adaptive."
        case .adaptive: return "Cut like normal; a chunk whose language sounds mixed is cut again at 0.2 s pauses and each part detected on its own. Measured in automatic mode on the same dialogue: 11 sentences lost, every tag right, at about finest's time there and at normal's time on a single-language meeting."
        case .coarse: return "A pause of 1 s or more ends a chunk, chunks are at least 15 s: fewest language decisions and fastest, but quick switches between speakers are missed."
        case .normal: return "A pause of 0.7 s or more ends a chunk, chunks are at least 5 s."
        case .fine: return "A pause of 0.35 s or more, chunks of at least 2 s: follows quicker back-and-forth; slower."
        case .finest: return "A pause of 0.2 s or more, chunks of at least 1 s: follows rapid turn-taking (a 38 s test with 0.25 s gaps between speakers needed this); slowest, and a very short chunk gives the detector little to judge from."
        }
    }
}

/// A sound device the user chose: its Core Audio UID (the same across launches and replugging) and the name it had when
/// it was chosen (shown when it is not connected). An empty UID means the device macOS uses now (System Settings >
/// Sound), which follows a change of the default while the recording runs.
struct DeviceChoice: Equatable {
    var uid = ""
    var label = ""

    var isDefault: Bool { return uid.isEmpty }

    /// What to tell when the chosen device is not connected as a recording starts: the recording goes on with the
    /// Mac's current one. `what` names the kind ("microphone", "sound output"), `instead` what is used.
    func notConnectedNote(what: String, instead: String) -> String {
        let shown = label.isEmpty ? uid : label
        return "The \(what) “\(shown)” chosen in the settings is not connected; \(instead) is used."
    }

    /// How mt.py is told which input to record (`--device`), given the inputs connected now in Core Audio's order (the
    /// order PortAudio lists them in too): the device's name when no other input shares it, else `name#k` for the k-th
    /// input of that name (two identical USB microphones), which mt.py resolves among the same-named inputs; "" when the
    /// device is not connected (the recording then takes the Mac's current input and the window says so).
    func recorderName(among inputs: [(uid: String, name: String)]) -> String {
        guard let chosen = inputs.first(where: { $0.uid == uid }) else { return "" }
        let twins = inputs.filter { $0.name == chosen.name }
        guard twins.count > 1, let position = twins.firstIndex(where: { $0.uid == uid }) else { return chosen.name }
        return "\(chosen.name)#\(position + 1)"
    }
}

/// The local translation model (mt.py --translator): Hy-MT3 7B or TranslateGemma 12B on the Apple GPU, which take the
/// lines of their languages and leave the others to NLLB, or NLLB-200 on the CPU. The order of the cases is the menu's.
enum OfflineTranslator: String, CaseIterable, Identifiable {
    case hy, gemma, nllb
    var id: String { return rawValue }

    var label: String {
        switch self {
        case .hy: return "Hy-MT3 7B (4.2 GB, Apple silicon)"
        case .nllb: return "NLLB-200 (1.4 GB)"
        case .gemma: return "TranslateGemma 12B (7 GB, Apple silicon)"
        }
    }

    /// The model a user gets who never picked one (nothing stored under "translator"): Hy-MT3 on Apple silicon, NLLB on an
    /// Intel Mac. A stored choice (also "nllb") is kept; a value this version does not know counts as never picked.
    static func initial(stored: String?, appleSilicon: Bool) -> OfflineTranslator {
        if let stored = stored, let chosen = OfflineTranslator(rawValue: stored) { return chosen }
        return appleSilicon ? .hy : .nllb
    }

    var hint: String {
        switch self {
        case .hy: return "Tencent's Hy-MT3 7B on the Apple GPU: 102 languages (a line in one it lacks, such as Bosnian or Yoruba, goes to NLLB, which is fetched too). The first use downloads it (about 4.2 GB: a 4-bit copy converted by NetProjects from Tencent's weights, checked against its SHA-256); a recording does not wait for that: its live translation starts once the download is done. It needs about 4.5 GB of memory, about 6 GB beside the speech model while recording (estimates, not yet measured), so 16 GB is comfortable and 8 GB is tight. Its quality figures come from Tencent only; there is no independent evaluation yet. Its licence (Apache-2.0) allows commercial use. If it is not installed (the setup's Apple-GPU step failed), NLLB translates and a note says so."
        case .nllb: return "Meta's NLLB-200 on this Mac's CPU: about 200 languages, the first translation downloads it (about 1.4 GB). Less accurate than Claude, and its licence is non-commercial."
        case .gemma: return "Google's TranslateGemma 12B on the Apple GPU: 51 languages (the others go to NLLB, which is fetched too). The first use downloads it (about 7 GB); a recording does not wait for that: its live translation starts once the download is done. It needs about 7 GB of memory, 8.5 GB beside the speech model while recording, so 16 GB or more. Measured on an Apple M5 with 24 GB: 1.5–2.4 s per sentence, all five test sentences right where NLLB got four. Its licence (Gemma Terms of Use) allows commercial use. If it is not installed (the setup's Apple-GPU step failed), NLLB translates and a note says so."
        }
    }
}

/// What the user chose in the window, and how it becomes mt.py's arguments and environment.
struct JobSettings: Equatable {
    enum Mode: String, CaseIterable, Identifiable {
        case files, record, translate
        var id: String { return rawValue }
        var label: String {
            switch self {
            case .files: return "Audio / video files"
            case .record: return "Record"
            case .translate: return "Transcript files"
            }
        }
    }

    var mode: Mode = .files
    var files: [String] = []
    var outputDir = ""
    var targets: [String] = []
    var offline = false          // translate on this Mac with the local model instead of with Claude
    var translator = OfflineTranslator.nllb   // which local model: NLLB-200, Hy-MT3 7B or TranslateGemma 12B (both on the Apple GPU)
    var spoken: [String] = []    // languages spoken in the audio: none = detect any, one = forced, several = detect among them
    var granularity: Granularity = .auto
    var track = ""               // audio track of a video, empty = the first
    var recordName = ""
    var recordHours = 4          // the recording stops by itself after this many hours of audio (1...8)
    var transcribeLater = false  // record only; the recording stays in the data folder to be transcribed later
    var live = true              // transcribe while recording: the lines appear as the meeting goes on
    var plainLanguage = ""       // translate mode: the language of plain text files (offline translation needs it)
    var captureSystem = false    // record: also the sound this computer plays (the other side of a call), as a second track
    var microphone = DeviceChoice()      // record: which input is recorded (empty = the Mac's current one)
    var microphoneName = ""              // the chosen microphone's name as it is connected now ("" = not connected, or the default)
    var output = DeviceChoice()          // record a call: the output device whose sound is the other side (empty = the current one)
    var meName = "You"
    var themName = "Them"
    var speakers = false         // several people on one microphone (or on the other side of a call): label who spoke
    var speakerCount = ""        // how many, empty = find out
    var pyannote = false         // the optional pyannote backend instead of the built-in sherpa-onnx one

    /// The settings a recording's transcription uses when the recording stops: what the recording itself was started with
    /// (its name, length, live transcription, the second track) from the Start snapshot, everything that only matters
    /// afterwards (languages, granularity, who spoke, translation, output folder, the two names, "transcribe afterwards")
    /// from the form as it is now, so they can be changed while the recording runs.
    func atStop(now form: JobSettings?) -> JobSettings {
        guard var merged = form else { return self }
        merged.mode = mode
        merged.files = files
        merged.track = track
        merged.recordName = recordName
        merged.recordHours = recordHours
        merged.live = live
        merged.captureSystem = captureSystem
        merged.microphone = microphone
        merged.microphoneName = microphoneName
        merged.output = output
        return merged
    }

    /// Messages for what is missing; empty = ready to start.
    func validationProblems(isDirectory: (String) -> Bool = { path in
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
    }) -> [String] {
        var problems: [String] = []
        if mode != .record && files.isEmpty {
            problems.append(mode == .translate ? "Add the transcript files to translate." : "Add audio or video files.")
        }
        if mode == .translate && targets.isEmpty { problems.append("Choose at least one language to translate to.") }
        if !outputDir.isEmpty && !isDirectory(outputDir) { problems.append("The output folder does not exist.") }
        let t = track.trimmingCharacters(in: .whitespaces)
        if !t.isEmpty && (Int(t) == nil || Int(t)! < 0) { problems.append("The audio track must be a number from 0.") }
        if mode == .record && recordName.rangeOfCharacter(from: CharacterSet(charactersIn: "/\\:")) != nil {
            problems.append("The name cannot contain / \\ or : (it becomes a file name).")
        }
        let n = speakerCount.trimmingCharacters(in: .whitespaces)
        if speakers && mode != .translate && !n.isEmpty && (Int(n) == nil || Int(n)! < 1) {
            problems.append("The number of speakers must be a whole number from 1 (or empty to find out).")
        }
        return problems
    }

    /// Only Claude needs the key; the local model does not.
    var needsAPIKey: Bool { return !targets.isEmpty && !offline }

    /// Whether the recording translates its lines live as well (the targets chosen when it starts, offline or with Claude).
    var translatesLive: Bool { return live && !transcribeLater && !targets.isEmpty }

    /// The output folder when none is chosen: every meeting gets a subfolder there named after the hour it began
    /// (`26-11-22 14`), with the transcript, the translations and the recording (mt.py's MT_OUT).
    static var defaultOutputDir: String {
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents/Meeting Transcriber").path
    }

    /// The folder mt.py's results go to (MT_OUT): the chosen one, else the default.
    var resolvedOutputDir: String { return outputDir.isEmpty ? JobSettings.defaultOutputDir : outputDir }

    /// `--to`, `--offline` and `--translator` for a translation (also the live one).
    private var translationArguments: [String] {
        var args: [String] = []
        if offline && !targets.isEmpty {
            args.append("--offline")
            if translator != .nllb { args += ["--translator", translator.rawValue] }   // NLLB is mt.py's default
        }
        for target in targets { args += ["--to", target] }
        return args
    }

    /// What the language, granularity and translation options add to a transcription command (the output folder goes
    /// through the environment, MT_OUT, so that mt.py makes the meeting's subfolder in it).
    private func commonArguments(translating: Bool = false, track: String? = nil) -> [String] {
        var args: [String] = []
        if !translating {
            if !spoken.isEmpty { args += ["--language", spoken.joined(separator: ",")] }
            if granularity != .auto { args += ["--granularity", granularity.rawValue] }
            if speakers {
                let n = speakerCount.trimmingCharacters(in: .whitespaces)
                args += ["--speakers", n.isEmpty ? "auto" : n]
                if pyannote { args += ["--diarizer", "pyannote"] }
            }
        }
        if let t = track?.trimmingCharacters(in: .whitespaces), !t.isEmpty { args += ["--track", t] }
        args += translationArguments
        if translating, !plainLanguage.trimmingCharacters(in: .whitespaces).isEmpty {
            args += ["--from", plainLanguage.trimmingCharacters(in: .whitespaces)]
        }
        return args
    }

    /// Transcribes a recording that sits in its data folder (a cancelled or crashed job): the folder's checkpoint is continued.
    /// The files go to the output folder's subfolder for the recording's hour (MT_OUT in the environment).
    func transcribeRecordingArguments(scriptPath: String, entry: Recordings.Entry) -> [String] {
        switch entry.kind {
        case .single(let audio):
            return [scriptPath, "transcribe", audio, "--into", entry.folder]
                + ["--stem", entry.name.isEmpty ? "recording" : entry.name] + commonArguments()
        case .call(let mic, let system):
            let me = meName.trimmingCharacters(in: .whitespaces), them = themName.trimmingCharacters(in: .whitespaces)
            var args = [scriptPath, "call", mic, system, "--into", entry.folder]
                + ["--stem", entry.name.isEmpty ? "recording" : entry.name]
            if !me.isEmpty { args += ["--me", me] }
            if !them.isEmpty { args += ["--them", them] }
            return args + commonArguments()
        }
    }

    /// Gives the speakers of a stored meeting real names: `mt.py rename <folder> "Speaker 1=Anna" ...`.
    static func renameArguments(scriptPath: String, folder: String, names: [String: String]) -> [String] {
        return [scriptPath, "rename", folder] + names.keys.sorted().compactMap { label in
            let name = names[label]!.trimmingCharacters(in: .whitespaces)
            return name.isEmpty || name == label ? nil : "\(label)=\(name)"
        }
    }

    /// Translates a stored meeting from its transcript.json (and its cache) into the chosen languages, into the output
    /// folder's subfolder for its hour (MT_OUT).
    func translateStoredArguments(scriptPath: String, entry: Recordings.Entry) -> [String] {
        return [scriptPath, "translate", entry.folder] + translationArguments
    }

    /// The record-only command the app runs for a plain recording: the transcription is a separate, queued job.
    func recordingArguments(scriptPath: String) -> [String] {
        var args = [scriptPath, "record", "--hours", String(recordHours), "--no-transcribe"]
        let name = recordName.trimmingCharacters(in: .whitespaces)
        if !name.isEmpty { args += ["--name", name] }
        // mt.py knows the devices by name (PortAudio shows no UID; `name#k` tells twins apart, see DeviceChoice.recorderName);
        // a chosen microphone that is not connected is left out, so the recording takes the Mac's current input, and the
        // window says so.
        if !microphone.isDefault && !microphoneName.isEmpty { args += ["--device", microphoneName] }
        if live && !transcribeLater {                 // "Transcribe afterwards": nothing is transcribed, not even live
            args.append("--live")
            if !spoken.isEmpty { args += ["--language", spoken.joined(separator: ",")] }
            if granularity != .auto { args += ["--granularity", granularity.rawValue] }
            args += translationArguments              // the lines are translated live as well, beside the transcript
        }
        return args
    }

    /// The `mt.py call` command that transcribes a recorded call: the microphone file and the system-audio file.
    /// `into` is the recording's data folder, so the final pass continues from what was transcribed live.
    func callArguments(scriptPath: String, mic: String, system: String, into: String? = nil) -> [String] {
        let name = recordName.trimmingCharacters(in: .whitespaces)
        var args = [scriptPath, "call", mic, system, "--stem", name.isEmpty ? "recording" : name]
        if let into = into { args += ["--into", into] }
        let me = meName.trimmingCharacters(in: .whitespaces), them = themName.trimmingCharacters(in: .whitespaces)
        if !me.isEmpty { args += ["--me", me] }
        if !them.isEmpty { args += ["--them", them] }
        return args + commonArguments()
    }

    /// The `mt.py follow` command that transcribes the two files of a call while the app still writes them.
    func followArguments(scriptPath: String, mic: String, system: String, into: String) -> [String] {
        var args = [scriptPath, "follow", mic, system, "--into", into]
        let me = meName.trimmingCharacters(in: .whitespaces), them = themName.trimmingCharacters(in: .whitespaces)
        if !me.isEmpty { args += ["--me", me] }
        if !them.isEmpty { args += ["--them", them] }
        if !spoken.isEmpty { args += ["--language", spoken.joined(separator: ",")] }
        if granularity != .auto { args += ["--granularity", granularity.rawValue] }
        return args + translationArguments            // the lines are translated live as well, beside the transcript
    }

    func arguments(scriptPath: String) -> [String] {
        var args = [scriptPath]
        switch mode {
        case .files: args.append("transcribe")
        case .record: args.append("record")
        case .translate: args.append("translate")
        }
        if mode == .record {
            args += ["--hours", String(recordHours)]
            let name = recordName.trimmingCharacters(in: .whitespaces)
            if !name.isEmpty { args += ["--name", name] }
            if transcribeLater { args.append("--no-transcribe") }
            if live && !transcribeLater { args.append("--live") }     // record only: nothing is transcribed, not even live
        } else {
            args += files
        }
        return args + commonArguments(translating: mode == .translate, track: mode == .files ? track : nil)
    }

    /// `home` is where transcripts are stored (MT_HOME); `apiKey` is passed only when translating; MT_OUT is the output
    /// folder, in which mt.py makes one subfolder per meeting hour.
    func environment(home: String, apiKey: String?, hfToken: String? = nil) -> [String: String] {
        // No tqdm bars from Hugging Face downloads: each "\r" redraw was a publish of the window; the progress comes as
        // @@STAT model events instead.
        var env = ["MT_EVENTS": "1", "MT_HOME": home, "MT_OUT": resolvedOutputDir, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                   "HF_HUB_DISABLE_PROGRESS_BARS": "1"]
        if needsAPIKey, let key = apiKey, !key.isEmpty { env["ANTHROPIC_API_KEY"] = key }
        if speakers && pyannote, let token = hfToken, !token.isEmpty { env["HF_TOKEN"] = token }
        return env
    }
}
