// Core tests: event parsing, run state, settings → arguments, engine setup. Plain Foundation, so they run on Linux (CI) and macOS:
//   swiftc -o /tmp/core-tests meeting-transcriber/app/Sources/Core/*.swift meeting-transcriber/app/Tests/CoreTests/main.swift && /tmp/core-tests
import Foundation

var failures = 0
var passes = 0
func expect(_ condition: @autoclosure () -> Bool, _ name: String) {
    if condition() { passes += 1 } else { failures += 1; print("FAIL: \(name)") }
}

// MARK: parser
let t = "\t"
expect(EventParser.parse("@@FILE\(t)2\(t)5\(t)/a/b c.mp4") == .file(index: 2, count: 5, path: "/a/b c.mp4"), "FILE")
expect(EventParser.parse("@@PROG\(t)3\(t)10") == .progress(done: 3, total: 10), "PROG")
expect(EventParser.parse("@@OUT\(t)/x/call.transcript.txt") == .output(path: "/x/call.transcript.txt"), "OUT")
expect(EventParser.parse("@@ERR\(t)a.mp4\(t)a.mp4 has no audio track") == .failure(file: "a.mp4", message: "a.mp4 has no audio track"), "ERR")
expect(EventParser.parse("@@DONE\(t)1") == .done(failed: 1), "DONE")
expect(EventParser.parse("@@CHECK\(t)ffmpeg\(t)missing") == .check(name: "ffmpeg", status: "missing"), "CHECK")
expect(EventParser.parse("@@STAGE\(t)Loading the speech model") == .stage("Loading the speech model"), "STAGE")
expect(EventParser.parse("@@READY") == .ready, "READY")
expect(EventParser.parse("@@NOTE\(t)The recording stops in 10 minutes") == .note("The recording stops in 10 minutes"), "NOTE")
expect(EventParser.parse("@@CLEAR") == .clear, "CLEAR")
var cleared = RunState(kind: .transcribe)
cleared.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: nil, text: "a", note: nil, live: false)))
cleared.apply(.clear)
cleared.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: "You", text: "a", note: nil, live: false)))
expect(cleared.lines.count == 1 && cleared.lines[0].speaker == "You", "CLEAR replaces the lines shown so far")
expect(EventParser.parse("@@POS\(t)6.0\(t)70.5") == .position(done: 6.0, total: 70.5), "POS")
expect(EventParser.parse("@@ETA\(t)125") == .eta(125), "ETA")
let lineJSON = "{\"t\": 12.5, \"end\": 14.0, \"lang\": \"de\", \"conf\": 0.9, \"guess\": true, \"speaker\": \"Them\", \"text\": \"Danke.\"}"
let parsedLine = EventParser.parse("@@LINE\(t)\(lineJSON)")
if case .line(let l) = parsedLine {
    expect(l.t == 12.5 && l.end == 14.0 && l.lang == "de" && l.guess && l.speaker == "Them" && l.text == "Danke." && !l.live && l.tag == "[de?]", "LINE fields")
} else { expect(false, "LINE parses") }
if case .line(let n) = EventParser.parse("@@LINE\(t){\"t\": 20, \"note\": \"speech not recognised for 6 s\"}") {
    expect(n.isNote && n.t == 20 && n.note == "speech not recognised for 6 s", "LINE note (integer time)")
} else { expect(false, "LINE note parses") }
expect(EventParser.parse("@@LINE\(t){bad json") == .text("@@LINE\t{bad json"), "a broken LINE is plain text")
expect(TranscriptLine(t: 1, end: nil, lang: "ru", conf: 0.3, guess: false, speaker: nil, text: "x", note: nil, live: false).tag == "[ru?]", "low confidence tag")
// The translation beside a line: `tr` on the event, kept when the line is sent again without it (the final pass refines
// the text; the translation of the files comes after), replaced when it comes with new ones.
let translated = TranscriptLine.parse(json: #"{"t": 2.0, "lang": "ru", "conf": 1.0, "text": "Привет", "live": true, "tr": {"English": "Hello", "German": "Hallo"}, "tr_offline": true}"#)
expect(translated?.tr == ["English": "Hello", "German": "Hallo"] && translated?.live == true, "a line's translations are read")
expect(TranscriptLine.parse(json: #"{"t": 2.0, "lang": "ru", "conf": 1.0, "text": "x"}"#)?.tr.isEmpty == true, "no tr: none")
var withTr = RunState(kind: .record)
withTr.apply(.line(translated!))
withTr.apply(.line(TranscriptLine(t: 2, end: nil, lang: "ru", conf: 1, guess: false, speaker: nil, text: "Привет", note: nil, live: false)))
expect(withTr.lines.count == 1 && !withTr.lines[0].live && withTr.lines[0].tr["English"] == "Hello", "the same text sent again by the final pass keeps the translations shown")
withTr.apply(.line(TranscriptLine(t: 2, end: nil, lang: "ru", conf: 1, guess: false, speaker: nil, text: "Привет!", note: nil, live: false)))
expect(withTr.lines.count == 1 && withTr.lines[0].text == "Привет!" && withTr.lines[0].tr.isEmpty, "a line decoded anew does not keep the translation of its old text")
withTr.apply(.line(TranscriptLine(t: 2, end: nil, lang: "ru", conf: 1, guess: false, speaker: nil, text: "Привет!", note: nil, live: false, tr: ["English": "Hello!"])))
expect(withTr.lines[0].tr == ["English": "Hello!"], "the final translation replaces them")
let stale = TranscriptLine.parse(json: #"{"t": 3.0, "lang": "en", "conf": 1.0, "text": "fine by me", "tr": {"German": "wir treffen uns, mir recht"}, "tr_of": "we meet fine by me"}"#)
expect(stale?.tr.isEmpty == true, "translations of another text (tr_of differs: an echo trimmed) are not shown")
let current = TranscriptLine.parse(json: #"{"t": 3.0, "lang": "en", "conf": 1.0, "text": "fine by me", "tr": {"German": "mir recht"}, "tr_of": "fine by me"}"#)
expect(current?.tr == ["German": "mir recht"], "translations of this text are shown")
// Several translations side by side, one column each.
let offered = ["English", "German", "French"]
expect(TranslationColumns.shown(choices: offered, chosen: nil) == offered, "columns: all on offer until some are picked")
expect(TranslationColumns.shown(choices: offered, chosen: ["French", "English"]) == ["English", "French"], "columns: the picked ones, in offered order")
expect(TranslationColumns.shown(choices: offered, chosen: []).isEmpty, "columns: none picked, none shown")
expect(TranslationColumns.shown(choices: offered, chosen: ["Spanish", "German"]) == ["German"], "columns: a pick no longer on offer is left out")
expect(TranslationColumns.copyLine("[00:01] [ru] Привет", translations: ["English", "German"], of: ["English": "Hello"])
       == "[00:01] [ru] Привет\tHello\t", "copy: one tab-separated cell per shown translation, empty while still to come")
expect(TranslationColumns.copyLine("x", translations: [], of: ["English": "y"]) == "x", "copy: no translation shown, the text alone")
var noTargets = RunState(kind: .transcribe)
expect(noTargets.targets.isEmpty, "a run translates into nothing unless told")
noTargets.apply(.check(name: "live", status: "failed"))
expect(noTargets.checks["live"] == "failed", "a live transcription that stopped is a check the pane reads")
expect(Format.rough(3900) == "1 h 05 min" && Format.rough(125) == "2 min" && Format.rough(40) == "40 s", "rough durations")
expect(EventParser.parse("@@LEVEL\(t)0.37") == .level(0.37) && EventParser.parse("@@LEVEL\(t)7") == .level(1), "LEVEL, clamped")
expect(EventParser.parse("@@FOLDER\(t)/data/20261110-1200-x") == .folder("/data/20261110-1200-x"), "FOLDER")
var fl = RunState(kind: .record); fl.apply(.folder("/data/x"))
expect(fl.folder == "/data/x" && fl.phase == .recording, "a recording knows its folder from the start")
var lv = RunState(kind: .record); lv.apply(.level(0.5))
expect(lv.level == 0.5 && lv.phase == .recording, "the level is kept while recording")
var det = StallDetector()
let d0 = Date(timeIntervalSince1970: 0)
det.register("microphone", at: d0); det.register("system", at: d0)
det.saw("microphone", at: d0.addingTimeInterval(8)); det.saw("system", at: d0.addingTimeInterval(8))
expect(det.stalled(at: d0.addingTimeInterval(9), since: d0).isEmpty, "nothing stalled before the threshold")
det.saw("microphone", at: d0.addingTimeInterval(20))
expect(det.stalled(at: d0.addingTimeInterval(21), since: d0) == ["system"], "the system track delivered nothing for 13 s")
expect(det.stalled(at: d0.addingTimeInterval(26), since: d0).isEmpty, "reported once")
det.saw("system", at: d0.addingTimeInterval(27))
det.saw("microphone", at: d0.addingTimeInterval(27))
expect(det.stalled(at: d0.addingTimeInterval(40), since: d0) == ["microphone", "system"], "both again after delivering")
expect(EventParser.parse("note: x has 2 audio tracks\r") == .text("note: x has 2 audio tracks"), "text, CR stripped")
expect(EventParser.parse("@@WHAT\(t)x") == .text("@@WHAT\tx"), "unknown event is text")
expect(EventParser.parse("   3 min@@LEVEL\(t)0.25") == .level(0.25), "an event behind the minute counter's fragment is the event")
expect(EventParser.parse("12/40 chunks@@PROG\(t)13\(t)40") == .progress(done: 13, total: 40), "an event behind the chunk counter's fragment")
expect(EventParser.parse("see @@LINE in the docs") == .text("see @@LINE in the docs"), "other text with @@ inside stays text")
var oneFile = RunState(kind: .transcribe)
oneFile.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: "You", text: "live", note: nil, live: true)))
oneFile.apply(.file(index: 1, count: 1, path: "/d/mic.wav"))
expect(oneFile.lines.count == 1 && oneFile.phase == .working, "a one-file job keeps its lines at the FILE event")
oneFile.apply(.file(index: 1, count: 2, path: "/d/a.wav"))
expect(oneFile.lines.isEmpty, "a batch resets the lines per file")
expect(EventParser.parse("@@PROG\(t)bad") == .progress(done: 0, total: 0), "malformed numbers → 0")
expect(EventParser.parse("@@OUT\(t)/tab\tin/name.txt") == .output(path: "/tab"), "a tab inside a path is not expected (mt.py never prints one)")
expect(Format.duration(75) == "01:15" && Format.duration(3725) == "1:02:05", "duration format")

// MARK: run state
var s = RunState(kind: .transcribe)
expect(s.phase == .starting, "starts in starting")
s.apply(.progress(done: 2, total: 4))
expect(s.phase == .working && s.fileFraction == 0.5, "PROG before FILE (mt.py transcribes, then reports the file)")
s.apply(.file(index: 2, count: 4, path: "b.mp3"))
expect(s.chunk == 0 && s.chunks == 0 && s.currentFile == "b.mp3", "FILE resets the chunk counter")
s.apply(.progress(done: 1, total: 2))
expect(s.overallFraction == (1.0 + 0.5) / 4.0, "overall = finished files + current fraction")
s.apply(.output(path: "/o/b.transcript.txt"))
s.apply(.failure(file: "c.mp3", message: "boom"))
s.apply(.text("  12 min"))
s.apply(.text("3/9 chunks"))
s.apply(.text("warning: no speech found"))
expect(s.log == ["warning: no speech found"], "minute counters and chunk redraws stay out of the log")
s.finish(exitStatus: 1, stopRequested: false)
expect(s.outcome == .completedWithErrors(1) && s.phase == .finished && s.outputs == ["/o/b.transcript.txt"], "finish with a failed file")
var ok = RunState(kind: .translate); ok.finish(exitStatus: 0, stopRequested: false)
expect(ok.outcome == .completed, "finish ok")
var crashed = RunState(kind: .transcribe); crashed.finish(exitStatus: 1, stopRequested: false)
expect(crashed.outcome == .failed(1), "exit 1 without a reported file error is a failure")
var stopped = RunState(kind: .transcribe); stopped.finish(exitStatus: 15, stopRequested: true)
expect(stopped.outcome == .stopped, "stop → stopped")
expect(RunState(kind: .record).phase == .recording, "a recording starts in the recording phase")
var noted = RunState(kind: .record); noted.apply(.note("The recording stops in 10 minutes"))
expect(noted.notes == ["The recording stops in 10 minutes"] && noted.log == noted.notes && noted.phase == .recording, "a note is kept for the result and does not change the phase")
var rec = RunState(kind: .record); rec.apply(.progress(done: 1, total: 3))
expect(rec.phase == .working, "after the recording the work begins")
var rec2 = RunState(kind: .record); rec2.apply(.stage("Transcribing the recording"))
expect(rec2.phase == .working && rec2.stage == "Transcribing the recording", "the recording phase ends when the stage is announced")
var staged = RunState(kind: .transcribe)
staged.apply(.stage("Downloading the translation model"))
expect(staged.stage == "Downloading the translation model", "a stage is kept")
staged.apply(.file(index: 1, count: 1, path: "a.mp3"))
expect(staged.stage == "", "a new file clears the stage")
staged.apply(.stage("x")); staged.apply(.progress(done: 1, total: 2))
expect(staged.stage == "x", "progress keeps the stage: it says what the bar counts")
var prog = RunState(kind: .transcribe)
let t0 = Date(timeIntervalSince1970: 1000)
prog.apply(.position(done: 10, total: 100), at: t0)
prog.apply(.position(done: 30, total: 100), at: t0.addingTimeInterval(10))
expect(prog.position == 30 && prog.duration == 100 && prog.speed == 2.0, "speed from two positions: 20 s of audio in 10 s")
prog.apply(.position(done: 40, total: 100), at: t0.addingTimeInterval(20))
expect(abs(prog.speed - (0.7 * 2.0 + 0.3 * 1.0)) < 1e-9, "speed is smoothed")
prog.apply(.eta(3900))
prog.apply(.line(TranscriptLine(t: 1, end: 2, lang: "ru", conf: 0.9, guess: false, speaker: nil, text: "a", note: nil, live: false)))
prog.apply(.line(TranscriptLine(t: 3, end: 4, lang: "ru", conf: 0.9, guess: false, speaker: nil, text: "b", note: nil, live: false)))
prog.apply(.line(TranscriptLine(t: 5, end: 6, lang: "en", conf: 0.9, guess: false, speaker: nil, text: "c", note: nil, live: false)))
prog.apply(.line(TranscriptLine(t: 7, end: nil, lang: "", conf: 1, guess: false, speaker: nil, text: "", note: "n", live: false)))
expect(prog.spokenLines == 3 && prog.languageShares.map { "\($0.0) \($0.1)" } == ["ru 67", "en 33"], "lines and language shares (notes not counted)")
expect(prog.progressText == "00:40 of 01:40 (40 %) · 1.7× realtime · ETA 1 h 05 min · 3 lines · ru 67 % en 33 %", "progress text")
prog.apply(.stage("Translating to Russian (1 of 2)"))
expect(prog.eta == nil && prog.chunks == 0 && prog.lines.count == 4, "a new stage drops the estimate, keeps the lines")
prog.apply(.file(index: 2, count: 2, path: "b.mp3"))
expect(prog.lines.isEmpty && prog.position == 0 && prog.speed == 0, "a new file starts a new transcript")
var liveState = RunState(kind: .record)
liveState.apply(.position(done: 10, total: 30))
liveState.apply(.line(TranscriptLine(t: 2, end: 4, lang: "en", conf: 0.9, guess: false, speaker: "You", text: "hello", note: nil, live: true)))
expect(liveState.phase == .recording && liveState.lines.count == 1, "live events keep the recording phase")
var both = RunState(kind: .record)
both.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: "You", text: "a", note: nil, live: true)))
both.apply(.line(TranscriptLine(t: 0, end: nil, lang: "ru", conf: 1, guess: false, speaker: "Them", text: "b", note: nil, live: true)))
expect(both.lines.count == 2, "two live lines at the same time from both sides of a call are both kept")
both.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: "You", text: "A", note: nil, live: false)))
expect(both.lines.count == 2 && both.lines[0].text == "A" && !both.lines[0].live && both.lines[1].text == "b", "the final pass refines its own speaker's line")
both.apply(.line(TranscriptLine(t: 0, end: nil, lang: "en", conf: 1, guess: false, speaker: "You", text: "A2", note: nil, live: false)))
expect(both.lines.count == 2 && both.lines[0].text == "A2", "a final line replaces a final line of the same speaker")
var plain = RunState(kind: .record)
plain.apply(.line(TranscriptLine(t: 3, end: nil, lang: "en", conf: 1, guess: false, speaker: nil, text: "x", note: nil, live: true)))
plain.apply(.line(TranscriptLine(t: 3, end: nil, lang: "en", conf: 1, guess: false, speaker: "Speaker 1", text: "X", note: nil, live: false)))
expect(plain.lines.count == 1 && plain.lines[0].speaker == "Speaker 1", "a live line without a speaker is refined by a labelled final line")
liveState.apply(.line(TranscriptLine(t: 2, end: 4, lang: "en", conf: 0.9, guess: false, speaker: "You", text: "hello there", note: nil, live: false)))
expect(liveState.lines.count == 1 && liveState.lines[0].text == "hello there" && liveState.lines[0].speaker == "You", "the final pass replaces a live line at the same time")
liveState.apply(.line(TranscriptLine(t: 2, end: 4, lang: "en", conf: 0.9, guess: false, speaker: "Boris", text: "x", note: nil, live: false)))
expect(liveState.lines.count == 2, "a different speaker at the same time is another line")
var many = RunState(kind: .transcribe)
for i in 0..<(RunState.logLimit + 20) { many.apply(.text("line \(i)")) }
expect(many.log.count == RunState.logLimit && many.log.last == "line \(RunState.logLimit + 19)", "log is bounded, newest kept")

// MARK: languages
expect(Languages.targets(checked: ["English"], custom: "Czech, english ;Dutch,, ") == ["English", "Czech", "Dutch"], "targets: dedupe, trim, order")
expect(Languages.targets(checked: [], custom: "").isEmpty, "targets: none")
// The two drop-downs: the common languages first, then the rest A-Z, each once.
expect(Languages.spokenAll.count == 100 && Set(Languages.spokenAll).count == 100, "languages: Whisper's 100, each once")
expect(Array(Languages.spokenAll.prefix(Languages.common.count)) == Languages.common, "languages: the common ones first")
let rest = Array(Languages.spokenAll.dropFirst(Languages.common.count))
expect(rest == rest.sorted() && rest.first == "Afrikaans", "languages: then the others A-Z")
expect(!Languages.targetAll.contains("Latin") && !Languages.targetAll.contains("Breton") && !Languages.targetAll.contains("Hawaiian")
       && Languages.targetAll.contains("Traditional Chinese") && Languages.targetAll.contains("Zulu")
       && Languages.targetAll.contains("Brazilian Portuguese") && Languages.targetAll.contains("European Portuguese")
       && Languages.targetAll.count == 101,
       "languages: the targets are those the offline model translates into")
expect(Languages.targets(offline: true) == Languages.targetAll && Languages.targets(offline: false).count == 104
       && Languages.targets(offline: false).contains("Latin") && Languages.targets(offline: false).first == "English",
       "languages: with Claude also Breton, Hawaiian and Latin, in menu order")
expect(Languages.matching("", in: ["English", "German"]) == ["English", "German"], "search: empty shows all")
expect(Languages.matching("  nor ", in: Languages.spokenAll) == ["Norwegian", "Norwegian Nynorsk"], "search: part of a name, any case")
expect(Languages.matching("ÉNG", in: ["English"]) == ["English"], "search: accents and case ignored")
expect(Languages.merged(["English"], typed: "czech, Klingon ;english", names: Languages.targetAll) == ["English", "Czech", "Klingon"],
       "merge: typed names spelled as in the list, an unknown one kept, no duplicates")
expect(Languages.merged([], typed: "", names: Languages.spokenAll).isEmpty, "merge: nothing typed, nothing chosen")
expect(Languages.summary([], none: "Any") == "Any" && Languages.summary(["English", "Russian"], none: "Any") == "English, Russian"
       && Languages.summary(["A", "B", "C", "D", "E"], none: "") == "A, B, C +2", "summary: the chosen ones, then +n")

// MARK: settings → arguments
var j = JobSettings()
expect(j.validationProblems().count == 1, "files mode without files")
j.files = ["/m/a b.mp3", "/m/c.mov"]
expect(j.validationProblems().isEmpty, "files mode ready")
expect(j.arguments(scriptPath: "/e/mt.py") == ["/e/mt.py", "transcribe", "/m/a b.mp3", "/m/c.mov"], "plain transcribe")
j.outputDir = "/out"; j.spoken = ["ru"]; j.track = "1"; j.targets = ["English", "Brazilian Portuguese"]
expect(j.arguments(scriptPath: "/e/mt.py") == ["/e/mt.py", "transcribe", "/m/a b.mp3", "/m/c.mov",
                                              "--language", "ru", "--track", "1", "--to", "English", "--to", "Brazilian Portuguese"], "transcribe with everything (the output folder goes through MT_OUT)")
expect(j.environment(home: "/h", apiKey: nil)["MT_OUT"] == "/out", "the chosen output folder is MT_OUT: mt.py makes the meeting's hour subfolder in it")
expect(JobSettings().environment(home: "/h", apiKey: nil)["MT_OUT"] == JobSettings.defaultOutputDir
       && JobSettings.defaultOutputDir.hasSuffix("/Documents/Meeting Transcriber"), "no folder chosen: the default under Documents")
expect(j.validationProblems(isDirectory: { _ in false }) == ["The output folder does not exist."], "missing output folder")
j.track = "x"
expect(j.validationProblems(isDirectory: { _ in true }).count == 1, "a bad audio track is the one problem (language names are checked by mt.py before it starts)")
j.track = "1"; j.spoken = ["ru", "English", "Brazilian Portuguese"]; j.targets = []; j.outputDir = ""
expect(j.arguments(scriptPath: "m") == ["m", "transcribe", "/m/a b.mp3", "/m/c.mov", "--language", "ru,English,Brazilian Portuguese", "--track", "1"], "several spoken languages are passed as one comma-separated list")
j.granularity = .finest
expect(j.arguments(scriptPath: "m") == ["m", "transcribe", "/m/a b.mp3", "/m/c.mov", "--language", "ru,English,Brazilian Portuguese", "--granularity", "finest", "--track", "1"], "granularity goes after the languages")
j.granularity = .auto
expect(!j.arguments(scriptPath: "m").contains("--granularity"), "auto granularity adds nothing (mt.py decides)")
j.spoken = []
expect(!j.arguments(scriptPath: "m").contains("--language"), "no spoken language: detect any")
expect(Granularity.allCases.map { $0.label } == ["Auto", "Adaptive", "Coarse", "Normal", "Fine", "Finest"], "granularity labels")
j.granularity = .adaptive
expect(j.arguments(scriptPath: "m").contains("adaptive"), "granularity: adaptive is passed as such")
j.granularity = .auto
expect(Granularity.allCases.allSatisfy { !$0.hint.isEmpty }, "every granularity has a hint")
var tr = JobSettings(); tr.mode = .translate; tr.files = ["/t/a.transcript.txt"]
expect(tr.validationProblems() == ["Choose at least one language to translate to."], "translate needs a target")
tr.targets = ["Spanish"]; tr.spoken = ["ru", "en"]; tr.track = "2"; tr.granularity = .fine
expect(tr.arguments(scriptPath: "m") == ["m", "translate", "/t/a.transcript.txt", "--to", "Spanish"], "translate ignores language, granularity and track")
var r = JobSettings(); r.mode = .record; r.recordName = " weekly "; r.recordHours = 2; r.targets = ["English"]; r.outputDir = "/out"
expect(r.validationProblems(isDirectory: { _ in true }).isEmpty, "record needs no files")
expect(r.arguments(scriptPath: "m") == ["m", "record", "--hours", "2", "--name", "weekly", "--live", "--to", "English"], "record arguments")
r.spoken = ["ru", "en"]
expect(r.arguments(scriptPath: "m") == ["m", "record", "--hours", "2", "--name", "weekly", "--live", "--language", "ru,en", "--to", "English"], "record with spoken languages")
expect(r.translatesLive, "a live recording with a target translates live")
var noLive = r; noLive.transcribeLater = true
var noTarget = r; noTarget.targets = []
expect(!noLive.translatesLive && !noTarget.translatesLive, "record only, or no target: no live translation")
var off = JobSettings(); off.files = ["/m/a.mp3"]; off.targets = ["English", "Russian"]; off.offline = true
expect(off.arguments(scriptPath: "m") == ["m", "transcribe", "/m/a.mp3", "--offline", "--to", "English", "--to", "Russian"], "offline translation argument")
expect(!off.needsAPIKey && off.environment(home: "/h", apiKey: "sk-x")["ANTHROPIC_API_KEY"] == nil, "offline needs and sends no API key")
var gem = off; gem.translator = .gemma
expect(gem.arguments(scriptPath: "m") == ["m", "transcribe", "/m/a.mp3", "--offline", "--translator", "gemma", "--to", "English", "--to", "Russian"], "TranslateGemma is passed as the offline translator")
var gemOnline = gem; gemOnline.offline = false
expect(!gemOnline.arguments(scriptPath: "m").contains("--translator"), "with Claude the offline model is not passed")
var gemLive = gem; gemLive.mode = .record
expect(gemLive.recordingArguments(scriptPath: "m").suffix(7) == ["--offline", "--translator", "gemma", "--to", "English", "--to", "Russian"], "the live translation uses it too")
expect(OfflineTranslator.allCases.allSatisfy { !$0.label.isEmpty && !$0.hint.isEmpty }, "every offline model is described")
var offNone = JobSettings(); offNone.files = ["/m/a.mp3"]; offNone.offline = true
expect(offNone.arguments(scriptPath: "m") == ["m", "transcribe", "/m/a.mp3"], "offline without languages adds nothing")
var offTr = JobSettings(); offTr.mode = .translate; offTr.files = ["/t/a.txt"]; offTr.targets = ["German"]; offTr.offline = true
expect(offTr.arguments(scriptPath: "m") == ["m", "translate", "/t/a.txt", "--offline", "--to", "German"], "offline translate of a transcript")
let env = r.environment(home: "/h", apiKey: "sk-x")
expect(env["MT_EVENTS"] == "1" && env["MT_HOME"] == "/h" && env["ANTHROPIC_API_KEY"] == "sk-x", "environment with a key")
expect(JobSettings().environment(home: "/h", apiKey: "sk-x")["ANTHROPIC_API_KEY"] == nil, "no key is passed when nothing is translated")

// MARK: engine
let engine = Engine(home: URL(fileURLWithPath: "/Users/me"))
expect(engine.python.path == "/Users/me/Library/Application Support/Meeting Transcriber/venv/bin/python", "engine python path")
expect(engine.data.path.hasSuffix("Meeting Transcriber/data"), "engine data path")
expect(Engine.pythonVersion("Python 3.12.4\n") == [3, 12, 4] && Engine.pythonVersion("Python 3.9.6") == [3, 9, 6], "python version")
expect(Engine.pythonVersion("Python 3.13.0rc1") == [3, 13, 0], "python pre-release suffix")
expect(Engine.pythonVersion("command not found") == nil, "not a python version")
expect(Engine.isSupported([3, 9, 6]) && Engine.isSupported([3, 13]) && !Engine.isSupported([3, 8, 18]) && !Engine.isSupported([2, 7]), "supported pythons")
let steps = engine.setupSteps(basePython: "/usr/bin/python3", engineFolder: "/App/Resources/engine")
expect(steps.count == 4 && steps[0] == ["/usr/bin/python3", "-m", "venv", engine.venv.path], "setup without a bundled runtime: venv first")
expect(steps[2].suffix(2) == ["-r", "/App/Resources/engine/requirements.txt"] && steps[3].suffix(2) == ["/App/Resources/engine/mt.py", "prepare"], "setup: packages, then the model")
let bundled = URL(fileURLWithPath: "/App/Resources/python/arm64", isDirectory: true)
let bundledSteps = engine.setupSteps(basePython: nil, engineFolder: "/App/Resources/engine", bundledRuntime: bundled)
expect(bundledSteps.count == 5 && bundledSteps[0] == ["/usr/bin/ditto", "/App/Resources/python/arm64", engine.runtime.path], "setup with a bundled runtime: copy it first")
expect(bundledSteps[1] == [engine.runtimePython.path, "-m", "venv", engine.venv.path] && engine.runtimePython.path.hasSuffix("Meeting Transcriber/runtime/bin/python3"), "the venv comes from the copied runtime")
let resources = URL(fileURLWithPath: "/App/Resources", isDirectory: true)
expect(Engine.bundledRuntime(resources: resources, arch: "arm64", fileExists: { $0 == "/App/Resources/python/arm64/bin/python3" })?.path == "/App/Resources/python/arm64", "bundled runtime found for the architecture")
expect(Engine.bundledRuntime(resources: resources, arch: "x86_64", fileExists: { $0 == "/App/Resources/python/arm64/bin/python3" }) == nil, "no runtime for another architecture")
expect(Engine.bundledRuntime(resources: nil, arch: "arm64") == nil, "no resources, no runtime")
let mlxSteps = engine.setupSteps(basePython: nil, engineFolder: "/E", bundledRuntime: bundled, mlx: true)
expect(mlxSteps.count == 6 && mlxSteps[4] == [engine.python.path, "-m", "pip", "install", "-r", "/E/requirements-mlx.txt"] && mlxSteps[5].last == "prepare", "the Apple-GPU engine is installed after the packages, before prepare")
expect(Engine.isOptionalStep(mlxSteps[4]) && !Engine.isOptionalStep(mlxSteps[3]) && !Engine.isOptionalStep(mlxSteps[5]), "only the Apple-GPU step is optional")
let pyaSteps = engine.setupSteps(basePython: nil, engineFolder: "/E", bundledRuntime: bundled, mlx: true, pyannote: true)
expect(pyaSteps.count == 7 && pyaSteps[5] == [engine.python.path, "-m", "pip", "install", "-r", "/E/requirements-pyannote.txt"] && Engine.isOptionalStep(pyaSteps[5]), "pyannote is an optional step before prepare")
var sp = JobSettings(); sp.files = ["/a.mp3"]; sp.speakers = true
expect(sp.arguments(scriptPath: "m") == ["m", "transcribe", "/a.mp3", "--speakers", "auto"], "speakers: find out how many")
sp.speakerCount = " 3 "; sp.pyannote = true
expect(sp.arguments(scriptPath: "m") == ["m", "transcribe", "/a.mp3", "--speakers", "3", "--diarizer", "pyannote"], "speakers: a count and the optional backend")
expect(sp.environment(home: "/h", apiKey: nil, hfToken: "hf_x")["HF_TOKEN"] == "hf_x" && sp.environment(home: "/h", apiKey: nil)["HF_TOKEN"] == nil, "the Hugging Face token goes only with pyannote")
sp.pyannote = false
expect(sp.environment(home: "/h", apiKey: nil, hfToken: "hf_x")["HF_TOKEN"] == nil, "no token without pyannote")
sp.speakerCount = "two"
expect(sp.validationProblems(isDirectory: { _ in true }) == ["The number of speakers must be a whole number from 1 (or empty to find out)."], "a bad speaker count is reported")
var spCall = JobSettings(); spCall.mode = .record; spCall.captureSystem = true; spCall.speakers = true
expect(spCall.callArguments(scriptPath: "m", mic: "a", system: "b").suffix(2) == ["--speakers", "auto"], "a call separates the other side")
expect(spCall.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe", "--live"], "the record-only process does not separate speakers (the queued transcription does)")
expect(JobSettings.renameArguments(scriptPath: "m", folder: "/d", names: ["Speaker 2": " Boris ", "Speaker 1": "Anna", "Speaker 3": "", "Them 1": "Them 1"]) == ["m", "rename", "/d", "Speaker 1=Anna", "Speaker 2=Boris"], "only real new names are passed")
expect(engine.isReady(appVersion: "1.3.0", fileExists: { _ in true }, readMarker: { "1.3.0\n" }), "ready: same version")
expect(!engine.isReady(appVersion: "1.4.0", fileExists: { _ in true }, readMarker: { "1.3.0" }), "not ready: set up by another version")
expect(!engine.isReady(appVersion: "1.3.0", fileExists: { $0 != engine.marker.path }, readMarker: { "1.3.0" }), "not ready: no marker")

expect(Engine.ffmpegAvailable(fileExists: { $0 == "/opt/homebrew/bin/ffmpeg" || $0 == "/usr/local/bin/ffprobe" }), "ffmpeg found in Homebrew folders")
expect(!Engine.ffmpegAvailable(fileExists: { $0 == "/opt/homebrew/bin/ffmpeg" }), "ffprobe missing")

// MARK: folders and files to work on
let fm = FileManager.default
func tempDir(_ name: String) -> URL {
    let url = fm.temporaryDirectory.appendingPathComponent("mt-\(name)-\(UUID().uuidString)", isDirectory: true)
    try? fm.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}
do {
    let root = tempDir("media")
    func touch(_ rel: String) {
        let url = root.appendingPathComponent(rel)
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? Data("x".utf8).write(to: url)
    }
    for f in ["a.mp3", "sub/b.MOV", "sub/deep/c.flac", ".hidden.mp3", "notes.txt", "sub/cover.jpg", "sub/x.transcript.txt", "odd.xyz"] { touch(f) }
    try? fm.createDirectory(at: root.appendingPathComponent("emptydir"), withIntermediateDirectories: true)
    let names = { (r: MediaFiles.Result) in r.files.map { URL(fileURLWithPath: $0).lastPathComponent } }
    let audio = MediaFiles.expand([root.path], translate: false)
    expect(names(audio) == ["a.mp3", "b.MOV", "c.flac"], "folder: audio and video at any depth, nothing else, sorted by path")
    expect(!audio.truncated, "folder: not truncated")
    expect(names(MediaFiles.expand([root.path], translate: true)) == ["notes.txt", "x.transcript.txt"], "folder: text files for translating")
    expect(MediaFiles.expand([root.path + "/odd.xyz"], translate: false).files == [root.path + "/odd.xyz"], "a single chosen file is taken whatever its extension")
    expect(MediaFiles.expand([root.path + "/missing.mp3"], translate: false).files.isEmpty, "a missing path is ignored")
    expect(MediaFiles.expand([root.path + "/emptydir"], translate: false).files.isEmpty, "an empty folder adds nothing")
    let mixed = MediaFiles.expand([root.path + "/a.mp3", root.path + "/sub"], translate: false)
    expect(names(mixed) == ["a.mp3", "b.MOV", "c.flac"], "files and folders together")
    expect(MediaFiles.matches("/x/Y.M4A", translate: false) && !MediaFiles.matches("/x/y.txt", translate: false), "extension match ignores case")
}
do {
    let big = tempDir("big")
    for i in 0..<(MediaFiles.limit + 5) { try? Data("x".utf8).write(to: big.appendingPathComponent(String(format: "f%04d.mp3", i))) }
    let r = MediaFiles.expand([big.path], translate: false)
    expect(r.files.count == MediaFiles.limit && r.truncated, "a huge folder is capped and says so")
    expect(r.files == r.files.sorted(), "a capped result is still sorted")
    let two = MediaFiles.expand([big.path, big.path], translate: false)
    expect(two.files.count == MediaFiles.limit && two.truncated, "the cap covers all chosen folders together")
}
do {
    let lots = tempDir("visit")
    for i in 0..<30 { try? Data("x".utf8).write(to: lots.appendingPathComponent("n\(i).dat")) }
    try? Data("x".utf8).write(to: lots.appendingPathComponent("z.mp3"))
    let cut = MediaFiles.expand([lots.path], translate: false, visitLimit: 10)
    expect(cut.truncated, "a scan that looks at too many entries gives up and says so")
    let whole = MediaFiles.expand([lots.path], translate: false)
    expect(!whole.truncated && whole.files.count == 1, "the same folder is fine within the visit limit")
}

// MARK: scans in the background
do {
    var tracker = ScanTracker()
    expect(!tracker.busy, "scans: idle at first")
    let a = tracker.begin()
    expect(tracker.busy && tracker.pending == 1, "scans: busy while one runs")
    expect(tracker.finish(a), "scans: a normal result is wanted")
    expect(!tracker.busy, "scans: idle again")
    let b = tracker.begin()
    tracker.invalidate()
    let c = tracker.begin()
    expect(tracker.pending == 2, "scans: two overlap")
    expect(!tracker.finish(b), "scans: a result that was cleared meanwhile is dropped")
    expect(tracker.busy, "scans: still busy until the newer scan ends")
    expect(tracker.finish(c), "scans: a scan started after the clear is wanted")
    expect(!tracker.busy, "scans: idle after both")
    var odd = ScanTracker()
    _ = odd.finish(0)
    expect(odd.pending == 0, "scans: a stray finish never goes below zero")
}

// MARK: a call: two tracks
do {
    var c = JobSettings(); c.mode = .record; c.captureSystem = true; c.recordName = " standup "
    c.spoken = ["ru", "en"]; c.granularity = .finest; c.targets = ["English"]; c.offline = true; c.outputDir = "/out"
    c.meName = " Anna "; c.themName = "Client"
    expect(c.callArguments(scriptPath: "m", mic: "/d/mic.wav", system: "/d/system.wav") == [
        "m", "call", "/d/mic.wav", "/d/system.wav", "--stem", "standup", "--me", "Anna", "--them", "Client",
        "--language", "ru,en", "--granularity", "finest", "--offline", "--to", "English"], "call arguments")
    expect(c.followArguments(scriptPath: "m", mic: "a", system: "b", into: "/d") == ["m", "follow", "a", "b", "--into", "/d", "--me", "Anna", "--them", "Client",
        "--language", "ru,en", "--granularity", "finest", "--offline", "--to", "English"], "the follower translates live into the ticked languages")
    var plain = JobSettings(); plain.mode = .record
    expect(plain.callArguments(scriptPath: "m", mic: "a", system: "b") == ["m", "call", "a", "b", "--stem", "recording", "--me", "You", "--them", "Them"], "call defaults")
    expect(plain.arguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--live"], "a plain recording is still mt.py record, 4 h by default, live")
    var laterLive = JobSettings(); laterLive.mode = .record; laterLive.transcribeLater = true
    expect(laterLive.arguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe"], "record only never transcribes live")
    var later = JobSettings(); later.mode = .record; later.transcribeLater = true; later.recordHours = 8; later.live = false
    expect(later.arguments(scriptPath: "m") == ["m", "record", "--hours", "8", "--no-transcribe"], "record only, up to 8 h")
    var liveRec = JobSettings(); liveRec.mode = .record
    expect(liveRec.arguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--live"], "live transcription is on by default")
    var follow = JobSettings(); follow.mode = .record; follow.spoken = ["ru", "en"]; follow.granularity = .fine; follow.meName = "Anna"
    expect(follow.followArguments(scriptPath: "m", mic: "a", system: "b", into: "/d") == ["m", "follow", "a", "b", "--into", "/d", "--me", "Anna", "--them", "Them", "--language", "ru,en", "--granularity", "fine"], "follow arguments")
    expect(follow.callArguments(scriptPath: "m", mic: "a", system: "b", into: "/d").prefix(7) == ["m", "call", "a", "b", "--stem", "recording", "--into"], "call --into")
    var ro = JobSettings(); ro.mode = .record; ro.recordName = " weekly "; ro.recordHours = 2; ro.spoken = ["ru", "en"]; ro.granularity = .fine; ro.targets = ["English"]; ro.outputDir = "/out"
    expect(ro.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "2", "--no-transcribe", "--name", "weekly", "--live", "--language", "ru,en", "--granularity", "fine", "--to", "English"], "the app only records; the live pass gets the languages and the live translation; the queued transcription gets the rest")
    var rlater = JobSettings(); rlater.mode = .record; rlater.transcribeLater = true; rlater.spoken = ["ru"]
    expect(rlater.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe"], "record only: no live pass either")
    var roff = JobSettings(); roff.mode = .record; roff.live = false; roff.spoken = ["ru"]
    expect(roff.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe"], "without live transcription, only the recording")
    var mic = JobSettings(); mic.mode = .record; mic.transcribeLater = true
    mic.microphone = DeviceChoice(uid: "AppleUSBAudioEngine:Jabra:1", label: "Jabra Evolve2"); mic.microphoneName = "Jabra Evolve2"
    expect(mic.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe", "--device", "Jabra Evolve2"], "the chosen microphone goes to mt.py by its name")
    mic.microphoneName = ""
    expect(mic.recordingArguments(scriptPath: "m") == ["m", "record", "--hours", "4", "--no-transcribe"], "a chosen microphone that is not connected is left out (the window says so)")
    expect(mic.microphone.notConnectedNote(what: "microphone", instead: "the Mac's current sound input")
           == "The microphone \u{201C}Jabra Evolve2\u{201D} chosen in the settings is not connected; the Mac's current sound input is used.", "the note for a missing device")
    expect(DeviceChoice(uid: "BuiltInSpeakerDevice").notConnectedNote(what: "sound output", instead: "the Mac's current output").hasPrefix("The sound output \u{201C}BuiltInSpeakerDevice\u{201D}"), "without a label, the UID is named")
    expect(DeviceChoice().isDefault && !mic.microphone.isDefault, "an empty UID is the Mac's current device")
    let inputs = [(uid: "builtin", name: "MacBook Pro Microphone"), (uid: "usb-a", name: "USB Audio"), (uid: "usb-b", name: "USB Audio")]
    expect(DeviceChoice(uid: "builtin").recorderName(among: inputs) == "MacBook Pro Microphone", "a name shared with no other input goes as it is")
    expect(DeviceChoice(uid: "usb-a").recorderName(among: inputs) == "USB Audio#1" && DeviceChoice(uid: "usb-b").recorderName(among: inputs) == "USB Audio#2",
           "two inputs of one name: the k-th of that name, so mt.py can tell them apart")
    expect(DeviceChoice(uid: "gone").recorderName(among: inputs) == "", "not connected: nothing to pass")
    var badName = JobSettings(); badName.mode = .record; badName.recordName = "Q3/Q4 review"
    expect(badName.validationProblems(isDirectory: { _ in true }) == ["The name cannot contain / \\ or : (it becomes a file name)."], "a slash in the name is refused before the job starts")
    expect(c.needsAPIKey == false && c.validationProblems(isDirectory: { _ in true }).isEmpty, "a call needs no files")
}
do {
    let dir = tempDir("wav")
    let url = dir.appendingPathComponent("t.wav")
    do {
        let w = try WavWriter(url: url)
        w.append([1, -2, 300])
        w.appendSilence(seconds: 0.5)
        w.sync()
        expect(w.sampleCount == 3 + 8000, "wav: sample count")
        w.appendSilence(seconds: 2.25)
        expect(w.sampleCount == 3 + 8000 + 36000, "wav: silence written in slices adds up")
        w.append([Int16](repeating: 0, count: 0))
        expect(abs(w.seconds - (3.0 + 8000.0 + 36000.0) / 16000.0) < 1e-9, "wav: seconds")
        w.append([7])
        w.close()
    } catch { expect(false, "wav: could not write \(error)") }
    let bytes = [UInt8]((try? Data(contentsOf: url)) ?? Data())
    func u32(_ at: Int) -> Int { return (0..<4).reduce(0) { $0 | Int(bytes[at + $1]) << (8 * $1) } }
    func u16(_ at: Int) -> Int { return Int(bytes[at]) | Int(bytes[at + 1]) << 8 }
    expect(bytes.count == 44 + (3 + 8000 + 36000 + 1) * 2, "wav: header plus the samples")
    expect(String(decoding: bytes[0..<4], as: UTF8.self) == "RIFF" && String(decoding: bytes[8..<12], as: UTF8.self) == "WAVE"
           && String(decoding: bytes[36..<40], as: UTF8.self) == "data", "wav: markers")
    expect(u32(4) == bytes.count - 8 && u32(40) == bytes.count - 44, "wav: sizes are patched on close")
    expect(u16(20) == 1 && u16(22) == 1 && u32(24) == 16000 && u32(28) == 32000 && u16(32) == 2 && u16(34) == 16, "wav: mono 16 kHz 16-bit PCM")
    expect(Int16(truncatingIfNeeded: u16(44)) == 1 && Int16(truncatingIfNeeded: u16(46)) == -2 && Int16(truncatingIfNeeded: u16(48)) == 300, "wav: samples are little-endian")
}

do {
    let dir = tempDir("wavfail")
    do {
        let w = try WavWriter(url: dir.appendingPathComponent("f.wav"))
        var reported = 0
        w.onFailure = { _ in reported += 1 }
        w.append([1, 2, 3])
        w.close()                                            // the file is gone from under the writer: the next write fails
        w.append([4, 5])
        w.append([6])
        expect(w.hasFailed && reported == 1, "wav failure: reported once")
        expect(w.sampleCount == 3, "wav failure: only samples that reached the file are counted")
    } catch { expect(false, "wav failure: could not write \(error)") }
}

// MARK: recordings in the data folder
do {
    let data = tempDir("data")
    func make(_ name: String, _ files: [String]) {
        let folder = data.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for f in files { try? Data("x".utf8).write(to: folder.appendingPathComponent(f)) }
    }
    make("20261110-103728-prn", ["audio.wav"])
    make("20261109-090000-weekly call", ["mic.wav", "system.wav", "partial.1.json"])
    make("20261108-120000", ["audio.wav", "transcript.json"])
    make("20261107-080000-talk", ["transcript.json"])
    make("notes", ["readme.txt"])
    let entries = Recordings.scan(data: data)
    expect(entries.map { $0.name } == ["prn", "weekly call", "", "talk"], "newest first, names without the time stamp")
    expect(entries[0].unfinished && !entries[0].hasCheckpoint, "a plain recording without a transcript is unfinished")
    if case .call(let mic, let system) = entries[1].kind { expect(mic.hasSuffix("mic.wav") && system.hasSuffix("system.wav") && entries[1].hasCheckpoint, "a call with a checkpoint") } else { expect(false, "call kind") }
    expect(!entries[2].unfinished && !entries[3].unfinished, "stored meetings")
    expect(Recordings.parse(folderName: "20261110-103728-prn").1 == "prn" && Recordings.parse(folderName: "weird").0 == nil, "folder name parsing")
    var js = JobSettings(); js.spoken = ["ru"]; js.targets = ["English"]; js.offline = true; js.meName = "Anna"
    expect(js.transcribeRecordingArguments(scriptPath: "m", entry: entries[0]) == ["m", "transcribe", entries[0].folder + "/audio.wav", "--into", entries[0].folder, "--stem", "prn", "--language", "ru", "--offline", "--to", "English"], "transcribe a left-over recording, continuing its folder, named after it")
    expect(js.transcribeRecordingArguments(scriptPath: "m", entry: entries[1]).prefix(8) == ["m", "call", entries[1].folder + "/mic.wav", entries[1].folder + "/system.wav", "--into", entries[1].folder, "--stem", "weekly call"], "continue a call")
    expect(js.transcribeRecordingArguments(scriptPath: "m", entry: entries[2]).prefix(7) == ["m", "transcribe", entries[2].folder + "/audio.wav", "--into", entries[2].folder, "--stem", "recording"], "a recording without a name is called recording")
    var chosen = js; chosen.outputDir = "/out"
    expect(!chosen.transcribeRecordingArguments(scriptPath: "m", entry: entries[0]).contains("--out")
           && chosen.environment(home: "/h", apiKey: nil)["MT_OUT"] == "/out", "the output folder is never an argument: mt.py takes it from MT_OUT and makes the hour subfolder")
    expect(js.translateStoredArguments(scriptPath: "m", entry: entries[2]) == ["m", "translate", entries[2].folder, "--offline", "--to", "English"], "translate a stored meeting")
    var pl = JobSettings(); pl.mode = .translate; pl.files = ["/t/a.txt"]; pl.targets = ["German"]; pl.offline = true; pl.plainLanguage = " Russian "
    expect(pl.arguments(scriptPath: "m") == ["m", "translate", "/t/a.txt", "--offline", "--to", "German", "--from", "Russian"], "the language of plain text is passed")
}

// MARK: the queue of processing jobs (a recording has its own slot beside it)
do {
    var q = JobQueue()
    expect(q.isIdle && q.count == 0 && q.startNext() == nil && q.folders.isEmpty, "an empty queue is idle")
    let a = QueuedJob(id: UUID(), label: "Transcribing A", kind: .transcribe, folder: "/d/a")
    let b = QueuedJob(id: UUID(), label: "Translating B", kind: .translate, folder: "/d/b")
    let c = QueuedJob(id: UUID(), label: "Transcribing C", kind: .transcribe)
    q.add(a); q.add(b); q.add(c)
    expect(!q.isIdle && q.count == 3 && q.current == nil, "jobs wait until one is started")
    expect(q.startNext() == a && q.current == a && q.pending == [b, c], "the first job starts, in order")
    expect(q.startNext() == nil && q.current == a, "one job runs at a time")
    expect(q.folders == ["/d/a", "/d/b"], "the folders of the running and waiting jobs")
    q.move(id: c.id, up: true)
    expect(q.pending == [c, b], "a waiting job moves up")
    q.move(id: c.id, up: true)
    expect(q.pending == [c, b], "the first waiting job stays first")
    q.move(id: c.id, up: false)
    expect(q.pending == [b, c], "and moves down again")
    q.move(id: a.id, up: false)
    expect(q.pending == [b, c], "the running job is not in the waiting list")
    expect(!q.remove(id: a.id) && q.current == a, "the running job is not removed (it is cancelled)")
    expect(q.remove(id: b.id) && q.pending == [c] && q.count == 2, "a waiting job is removed")
    expect(!q.remove(id: b.id), "removing it again changes nothing")
    q.finishCurrent()
    expect(q.current == nil && !q.isIdle, "the slot is free while a job waits")
    expect(q.startNext() == c && q.pending.isEmpty && q.folders.isEmpty, "the next job starts after the first finishes")
    q.finishCurrent()
    expect(q.isIdle, "idle again")
    q.add(a); q.add(b); q.removeAll()
    expect(q.isIdle, "removeAll drops the waiting jobs")
}

// MARK: key file
do {
    let dir = tempDir("keyfile")
    let file = KeyFile(url: dir.appendingPathComponent("nested/api-key"))
    expect(file.read() == nil, "key file: nothing yet")
    do { try file.write("sk-ant-test-1") } catch { expect(false, "key file: write failed \(error)") }
    expect(file.read() == "sk-ant-test-1", "key file: read back")
    let mode = (try? fm.attributesOfItem(atPath: file.url.path))?[.posixPermissions] as? NSNumber
    expect(mode?.intValue == 0o600, "key file: only the owner can read it")
    do { try file.write("sk-ant-test-2") } catch { expect(false, "key file: rewrite failed \(error)") }
    expect(file.read() == "sk-ant-test-2", "key file: overwritten")
    let again = (try? fm.attributesOfItem(atPath: file.url.path))?[.posixPermissions] as? NSNumber
    expect(again?.intValue == 0o600, "key file: still 0600 after a rewrite")
    try? Data("  sk-ant-padded \n".utf8).write(to: file.url)
    expect(file.read() == "sk-ant-padded", "key file: whitespace trimmed")
    try? Data("  \n".utf8).write(to: file.url)
    expect(file.read() == nil, "key file: blank is no key")
    file.delete()
    expect(file.read() == nil && !fm.fileExists(atPath: file.url.path), "key file: deleted")
}

// MARK: a computer's-sound track that was silent for the whole call
expect(TrackSilence.systemNote(buffers: 0, peak: 0, seconds: 120) != nil, "silent track: no buffer at all")
expect(TrackSilence.systemNote(buffers: 500, peak: 0.001, seconds: 120) != nil, "silent track: only silence")
expect(TrackSilence.systemNote(buffers: 500, peak: 0.3, seconds: 120) == nil, "silent track: sound came")
expect(TrackSilence.systemNote(buffers: 0, peak: 0, seconds: 5) == nil, "silent track: a call under 10 s says nothing")
expect(TrackSilence.systemNote(buffers: 0, peak: 0, seconds: 120)!.contains("Screen & System Audio Recording"), "silent track: names the permission")
expect(TrackSilence.peak == 0.002, "silent track: the engine's threshold")
expect(TrackSilence.microphoneNote(buffers: 0, peak: 0, seconds: 120)?.contains("no sound reached it") == true, "silent microphone: no buffer at all")
expect(TrackSilence.microphoneNote(buffers: 335, peak: 0.001, seconds: 120)?.contains("335 buffers reached it, the loudest sample was 0.001") == true, "silent microphone: zeros, quoted")
expect(TrackSilence.microphoneNote(buffers: 335, peak: 0.055, seconds: 120) == nil, "silent microphone: a quiet voice is sound")
expect(TrackSilence.microphoneNote(buffers: 0, peak: 0, seconds: 5) == nil, "silent microphone: a call under 10 s says nothing")
expect(TrackSilence.microphoneNote(buffers: 0, peak: 0, seconds: 120)!.contains("Privacy & Security > Microphone"), "silent microphone: names the permission")
expect(TrackSilence.microphoneEarlyNote(seconds: 25, buffers: 90, peak: 0.0, echoCancelled: true)?.contains("started again without it") == true, "early microphone note: a retry without echo cancellation")
expect(TrackSilence.microphoneEarlyNote(seconds: 25, buffers: 90, peak: 0.0, echoCancelled: false)?.contains("muted") == true, "early microphone note: plain input, nothing to retry")
expect(TrackSilence.microphoneEarlyNote(seconds: 10, buffers: 90, peak: 0.0, echoCancelled: true) == nil, "early microphone note: not before 20 s")
expect(TrackSilence.microphoneEarlyNote(seconds: 25, buffers: 0, peak: 0.0, echoCancelled: true) == nil, "early microphone note: no buffers is a stall, not silence")
expect(TrackSilence.microphoneEarlyNote(seconds: 25, buffers: 90, peak: 0.05, echoCancelled: true) == nil, "early microphone note: sound came")
expect(TrackSilence.restartsWithoutEchoCancellation(deliveredSinceStart: false, echoCancelled: true), "stalled microphone: not one buffer since its start with echo cancellation, it goes")
expect(!TrackSilence.restartsWithoutEchoCancellation(deliveredSinceStart: false, echoCancelled: false), "stalled microphone: a plain input is only started again")
expect(!TrackSilence.restartsWithoutEchoCancellation(deliveredSinceStart: true, echoCancelled: true), "stalled microphone: one that delivered since its start keeps it")
expect(TrackSilence.nothingWithEchoCancellationNote.contains("without it"), "stalled microphone: the note says what is done")
expect(TrackSilence.bluetoothAddress("DE-61-B1-84-A1-50:input") == "DE-61-B1-84-A1-50"
       && TrackSilence.bluetoothAddress("DE-61-B1-84-A1-50:output") == "DE-61-B1-84-A1-50", "bluetooth: the address of each half")
expect(TrackSilence.bluetoothAddress("AppleUSBAudioEngine:Vendor:Product:1:2") == nil && TrackSilence.bluetoothAddress(":input") == nil,
       "bluetooth: other UID forms, and an empty address, have none")
expect(TrackSilence.sameBluetoothDevice(uids: ("DE-61-B1-84-A1-50:input", "DE-61-B1-84-A1-50:output"), names: ("TG659", "TG659")), "bluetooth: one address")
expect(!TrackSilence.sameBluetoothDevice(uids: ("AA-11-22-33-44-55:input", "DE-61-B1-84-A1-50:output"), names: ("AirPods", "AirPods")),
       "bluetooth: two addresses are two devices, whatever the names")
expect(TrackSilence.sameBluetoothDevice(uids: ("headset-in", "headset-out"), names: ("Q20i", "Q20i")), "bluetooth: a UID form not known falls back to the name")
expect(!TrackSilence.sameBluetoothDevice(uids: ("headset-in", "headset-out"), names: ("Q20i", nil)), "bluetooth: no name, no match")
do {
    let note = TrackSilence.sharedBluetoothNote(device: "TG659")
    expect(note.contains("“TG659”") && note.contains("System Settings > Sound > Input") && note.contains("Audio MIDI Setup"),
           "bluetooth: the note names the device, the remedy and the rate to leave alone")
    expect(note.contains("if it has one") && note.contains("wired or USB"), "bluetooth: the note does not assume a built-in microphone")
}
expect(TrackSilence.earlyNote(seconds: 25, systemBuffers: 300, systemPeak: 0.0, microphonePeak: 0.3, audioTap: true)?.contains("System Audio Recording Only") == true, "early note: the tap's permission")
expect(TrackSilence.earlyNote(seconds: 25, systemBuffers: 300, systemPeak: 0.0, microphonePeak: 0.3, audioTap: false)?.contains("Only") == false, "early note: screen recording wording")
expect(TrackSilence.earlyNote(seconds: 10, systemBuffers: 300, systemPeak: 0.0, microphonePeak: 0.3, audioTap: true) == nil, "early note: not before 20 s")
expect(TrackSilence.earlyNote(seconds: 25, systemBuffers: 0, systemPeak: 0.0, microphonePeak: 0.3, audioTap: true) == nil, "early note: no buffers is a stall, not silence")
expect(TrackSilence.earlyNote(seconds: 25, systemBuffers: 300, systemPeak: 0.0, microphonePeak: 0.0, audioTap: true) == nil, "early note: a silent microphone too says nothing")
expect(TrackSilence.earlyNote(seconds: 25, systemBuffers: 300, systemPeak: 0.2, microphonePeak: 0.3, audioTap: true) == nil, "early note: sound came")

// MARK: a pausable recording's clock
do {
    var clock = PauseClock()
    let t0 = Date(timeIntervalSince1970: 1_000)
    expect(!clock.isPaused && clock.recorded(since: t0, at: t0.addingTimeInterval(100)) == 100, "pause clock: runs")
    clock.pause(at: t0.addingTimeInterval(100))
    clock.pause(at: t0.addingTimeInterval(110))                        // a second pause while paused changes nothing
    expect(clock.isPaused && clock.recorded(since: t0, at: t0.addingTimeInterval(160)) == 100, "pause clock: frozen while paused")
    expect(clock.resume(at: t0.addingTimeInterval(160)) == 60 && !clock.isPaused, "pause clock: the pause length")
    expect(clock.resume(at: t0.addingTimeInterval(170)) == 0, "pause clock: resuming twice is nothing")
    expect(clock.currentPause(at: t0.addingTimeInterval(170)) == 0, "pause clock: no pause going on")
    clock.pause(at: t0.addingTimeInterval(180))
    expect(clock.currentPause(at: t0.addingTimeInterval(190)) == 10 && clock.isPaused, "pause clock: the pause going on, still paused")
    clock.resume(at: t0.addingTimeInterval(190))
    expect(clock.recorded(since: t0, at: t0.addingTimeInterval(200)) == 130 && clock.pausedTotal == 70, "pause clock: runs on without the pauses")
    expect(PauseClock.note(recordedSeconds: 750.4, pausedFor: 185.2) == "[00:12:30] Paused here for 3 min 5 s", "pause note: minutes")
    expect(PauseClock.note(recordedSeconds: 3, pausedFor: 7.6) == "[00:00:03] Paused here for 8 s", "pause note: seconds")
}

// MARK: the settings a recording's transcription takes when it stops
do {
    var started = JobSettings(); started.mode = .record; started.recordName = "weekly"; started.recordHours = 2; started.live = false
    started.captureSystem = true; started.spoken = ["ru"]; started.targets = ["English"]; started.speakers = false; started.outputDir = ""
    started.microphone = DeviceChoice(uid: "mic-1", label: "Jabra"); started.microphoneName = "Jabra"; started.output = DeviceChoice(uid: "out-1", label: "Speakers")
    var form = JobSettings(); form.mode = .files; form.recordName = "typed later"; form.recordHours = 8; form.live = true
    form.captureSystem = false; form.spoken = ["ru", "en"]; form.granularity = .finest; form.targets = ["German"]; form.offline = true
    form.speakers = true; form.speakerCount = "3"; form.pyannote = true; form.outputDir = "/out"; form.meName = "Anna"; form.themName = "Boris"
    form.transcribeLater = true; form.microphone = DeviceChoice(uid: "mic-2", label: "AirPods"); form.output = DeviceChoice()
    let s = started.atStop(now: form)
    expect(s.mode == .record && s.recordName == "weekly" && s.recordHours == 2 && s.live == false && s.captureSystem == true
           && s.microphone.uid == "mic-1" && s.microphoneName == "Jabra" && s.output.uid == "out-1",
           "at stop: what the recording was started with stays (the devices too)")
    expect(s.spoken == ["ru", "en"] && s.granularity == .finest && s.targets == ["German"] && s.offline && s.speakers && s.speakerCount == "3"
           && s.pyannote && s.outputDir == "/out" && s.meName == "Anna" && s.themName == "Boris" && s.transcribeLater,
           "at stop: what only matters afterwards comes from the form")
    expect(started.atStop(now: nil) == started, "at stop: no form, the snapshot")
}

// MARK: the kind of code-signing identity (does macOS keep the permissions across builds?)
expect(SigningIdentity.kind(ofRequirement: "designated => cdhash H\"3f2a…\"") == .adHoc, "signing: ad-hoc is a cdhash")
expect(SigningIdentity.kind(ofRequirement: "identifier \"com.github.avza-bl.meeting-transcriber\" and certificate leaf = H\"ab…\"") == .stable, "signing: a certificate is stable")
expect(SigningIdentity.kind(ofRequirement: "identifier \"x\" and anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6]") == .stable, "signing: an Apple chain is stable")
expect(SigningIdentity.kind(ofRequirement: nil) == .unknown && SigningIdentity.kind(ofRequirement: "  ") == .unknown, "signing: nothing read")
expect(SigningIdentity.warning(for: .adHoc)?.contains("make_signing_identity.sh") == true, "signing: the ad-hoc warning says what to run")
expect(SigningIdentity.warning(for: .stable) == nil && SigningIdentity.warning(for: .unknown) == nil, "signing: no warning otherwise")

// MARK: AppInfo
expect(AppInfo.versionString(info: ["CFBundleShortVersionString": "1.3.0", "CFBundleVersion": "7", "MeetingTranscriberGitCommit": "abc"])
       == "Meeting Transcriber 1.3.0 (build 7, commit abc)", "appinfo: full")
expect(AppInfo.versionString(info: nil) == "Meeting Transcriber 0.0.0", "appinfo: defaults")

// MARK: speaker or headphones (the computer's-sound box follows where the sound goes)
expect(OutputKind.code("bltn") == 0x626C746E && OutputKind.code("ispk") == 0x6973706B && OutputKind.code("hdph") == 0x68647068,
       "output: four-character codes as Core Audio returns them")
let builtIn = OutputKind.builtIn
expect(OutputKind.of(transport: builtIn, dataSource: OutputKind.code("ispk"), uid: "BuiltInSpeakerDevice", terminals: []) == .speaker,
       "output: an Intel Mac's built-in output on its speakers")
expect(OutputKind.of(transport: builtIn, dataSource: OutputKind.code("hdpn"), uid: "BuiltInSpeakerDevice", terminals: []) == .headphones,
       "output: the data source wins over the UID (something in the jack)")
expect(OutputKind.of(transport: builtIn, dataSource: nil, uid: "BuiltInSpeakerDevice", terminals: []) == .speaker,
       "output: an Apple-silicon Mac's speakers, a device of their own")
expect(OutputKind.of(transport: builtIn, dataSource: nil, uid: "BuiltInHeadphoneOutputDevice", terminals: []) == .headphones,
       "output: an Apple-silicon Mac's headphone jack")
expect(OutputKind.of(transport: builtIn, dataSource: OutputKind.code("line"), uid: "x", terminals: []) == .unknown
       && OutputKind.of(transport: builtIn, dataSource: nil, uid: "AppleHDAEngineOutput:1B,0,1,1:0", terminals: [OutputKind.code("spkr")]) == .unknown,
       "output: a built-in output that says neither, or only a speaker terminal type, is unknown")
for transport in ["hdmi", "dprt", "airp"] {
    expect(OutputKind.of(transport: OutputKind.code(transport), dataSource: nil, uid: "x", terminals: []) == .speaker, "output: \(transport) is a speaker")
}
expect(OutputKind.of(transport: OutputKind.code("hdmi"), dataSource: nil, uid: "x", terminals: [OutputKind.code("hdph")]) == .headphones,
       "output: a headphones terminal type is never a speaker")
for transport in ["blue", "blea", "usb ", "virt", "grup", "thun"] {
    expect(OutputKind.of(transport: OutputKind.code(transport), dataSource: nil, uid: "00-02-3C-99-0B-7C:output", terminals: [OutputKind.code("spkr")]) == .unknown,
           "output: \(transport) does not say speaker or headphones")
}
expect(OutputKind.of(transport: OutputKind.code("usb "), dataSource: nil, uid: "x", terminals: [0x0402]) == .headphones, "output: a USB headset")
expect(OutputKind.of(transport: 0, dataSource: nil, uid: "", terminals: []) == .unknown, "output: nothing could be read")

var follow = OutputFollowing()
let tg = "00-02-3C-99-0B-7C:output", mac = "BuiltInSpeakerDevice", phones = "BuiltInHeadphoneOutputDevice"
expect(!follow.isSpeaker(uid: tg, kind: .unknown) && follow.isSpeaker(uid: mac, kind: .speaker), "follow: Core Audio's word until the user chose")
follow.userChose(ticked: false, uid: tg)
expect(follow.isSpeaker(uid: tg, kind: .unknown) && follow.speakers == [tg], "follow: an untick with a Bluetooth speaker teaches it")
follow.userChose(ticked: true, uid: mac)
expect(!follow.isSpeaker(uid: mac, kind: .speaker) && follow.twoTracks == [mac], "follow: a tick with the Mac's speakers keeps two tracks there")
follow.userChose(ticked: true, uid: tg)
expect(!follow.isSpeaker(uid: tg, kind: .unknown) && follow.speakers.isEmpty && follow.twoTracks == [mac, tg], "follow: a tick undoes a learned speaker")
expect(OutputFollowing.box(ticked: true, untickedByApp: false, speaker: true) == false, "follow: a speaker unticks the box")
expect(OutputFollowing.box(ticked: false, untickedByApp: true, speaker: false) == true, "follow: headphones tick it again when it was unticked by itself")
expect(OutputFollowing.box(ticked: false, untickedByApp: false, speaker: false) == nil, "follow: a box the user unticked stays unticked")
expect(OutputFollowing.box(ticked: false, untickedByApp: true, speaker: true) == nil
       && OutputFollowing.box(ticked: true, untickedByApp: false, speaker: false) == nil, "follow: nothing to change")
expect(!OutputFollowing().isSpeaker(uid: phones, kind: .headphones), "follow: headphones are not a speaker")
var unplugged = OutputFollowing()
unplugged.userChose(ticked: false, uid: phones)
expect(!unplugged.isSpeaker(uid: phones, kind: .headphones), "follow: an untick made with headphones never makes them a speaker")
// An Intel Mac's built-in output is one device, one UID, for its speakers and its jack: choices are kept per data source.
let hda = "AppleHDAEngineOutput:1B,0,1,1:0"
let hdaPhones = OutputFollowing.key(uid: hda, dataSource: OutputKind.headphonesSource)
let hdaSpeakers = OutputFollowing.key(uid: hda, dataSource: OutputKind.internalSpeaker)
expect(hdaSpeakers == hda + "#6973706b" && hdaPhones != hdaSpeakers && OutputFollowing.key(uid: tg, dataSource: nil) == tg,
       "follow: the key is the UID, with the data source when there is one")
var intel = OutputFollowing()
intel.userChose(ticked: true, uid: hdaPhones)
expect(intel.isSpeaker(uid: hdaSpeakers, kind: .speaker), "follow: a tick with headphones in an Intel Mac's jack leaves its speakers a speaker")
intel.userChose(ticked: false, uid: hdaSpeakers)
expect(!intel.isSpeaker(uid: hdaPhones, kind: .headphones) && intel.isSpeaker(uid: hdaSpeakers, kind: .speaker),
       "follow: an untick on an Intel Mac's speakers leaves its jack headphones")
expect(OutputFollowing.untickedNote(output: "TG659").contains("“TG659”"), "follow: the note names the output")

// MARK: data folders the app makes itself (a call): named and numbered as mt.py names its own
do {
    var when = DateComponents()
    when.year = 2026; when.month = 11; when.day = 26; when.hour = 15; when.minute = 20; when.second = 11
    let date = Calendar(identifier: .gregorian).date(from: when)!            // local time, as the stamp is
    expect(DataFolder.stamp(date) == "20261126-152011", "data folder: the stamp is mt.py's, local time")
    expect(DataFolder.name(stamp: "S", name: "weekly call") == "S-weekly call" && DataFolder.name(stamp: "S", name: "") == "S",
           "data folder: the first of a second")
    expect(DataFolder.name(stamp: "S", name: " x ", number: 3) == "S-x-3" && DataFolder.name(stamp: "S", name: "", number: 2) == "S-recording-2",
           "data folder: numbered as mt.py numbers (-recording-2 without a name)")
    let data = tempDir("datafolder").appendingPathComponent("data", isDirectory: true)    // not there yet: made
    let first = try? DataFolder.make(in: data, name: "call", date: date)
    let second = try? DataFolder.make(in: data, name: "call", date: date)
    let third = try? DataFolder.make(in: data, name: "call", date: date)
    expect([first, second, third].map { $0?.lastPathComponent } == ["20261126-152011-call", "20261126-152011-call-2", "20261126-152011-call-3"],
           "data folder: a second and a third call of one name in one second get folders of their own")
    var isDir: ObjCBool = false
    expect(fm.fileExists(atPath: third?.path ?? "", isDirectory: &isDir) && isDir.boolValue, "data folder: it is made")
    // An mt.py job made the plain one first (here a file in its place does the same): the call takes the next number.
    expect(fm.createFile(atPath: data.appendingPathComponent("20261126-152011-talk").path, contents: Data()), "data folder: setup, a file in the plain folder's place")
    expect((try? DataFolder.make(in: data, name: "talk", date: date))?.lastPathComponent == "20261126-152011-talk-2",
           "data folder: a name another job holds is skipped, never shared")
    let nameless1 = try? DataFolder.make(in: data, name: "", date: date)
    let nameless2 = try? DataFolder.make(in: data, name: "  ", date: date)
    expect(nameless1?.lastPathComponent == "20261126-152011" && nameless2?.lastPathComponent == "20261126-152011-recording-2",
           "data folder: without a name")
    let parsed = Recordings.parse(folderName: "20261126-152011-call-2")
    expect(parsed.0 == date && parsed.1 == "call-2" && Recordings.parse(folderName: "20261126-152011-recording-2").1 == "recording-2",
           "data folder: Recordings reads a numbered one back, date and name")
    let notAFolder = tempDir("datafolder-file").appendingPathComponent("data")
    expect(fm.createFile(atPath: notAFolder.path, contents: Data()), "data folder: setup, a file where the data folder goes")
    // A dangling link with the plain name counts as taken (mkdir's EEXIST), as in mt.py, not as an error.
    let linked = data.appendingPathComponent("20261126-152011-link")
    if (try? fm.createSymbolicLink(atPath: linked.path, withDestinationPath: "/nonexistent/mt-datafolder")) != nil {
        expect((try? DataFolder.make(in: data, name: "link", date: date))?.lastPathComponent == "20261126-152011-link-2",
               "data folder: a dangling link with the name is skipped too")
    }
    expect((try? DataFolder.make(in: notAFolder, name: "call", date: date)) == nil, "data folder: a data folder that is a file is an error, not a loop")
}

print("\(passes) passed, \(failures) failed")
exit(failures == 0 ? 0 : 1)
