# Meeting Transcriber — overview

A tool that turns speech into text, in many languages at once, and translates the text on request.

- **In:** a recording made by the tool (microphone, optionally also the sound the computer plays, for online calls) **or any audio or video file** that ffmpeg can read.
- **Out:** a plain-text **transcript** in which **every line carries its language**, and, if asked, one **translation file per language**.
- **Where it runs:** a native **Mac app** (macOS 13 or newer), a **command-line tool** (`mt.py`; macOS, Linux and, since 1.11.0, Windows 10/11 x64 from PowerShell; see *Requirements*), or a **Claude Code session** in this folder.
- **Where the speech is processed:** on your own computer. Only translation with Claude sends text (not audio) over the internet; offline translation sends nothing.

```text
[00:00:04] [en] <You> Let's start with the budget.
[00:00:09] [ru] <Them> Хорошо, я подготовил цифры.
[00:00:15] [de?] <Them> Danke, das passt.
```

`[xx]` is the language of the line; `[xx?]` marks a guess (low confidence, or a language you did not list). `<You>` / `<Them>` (or `<Speaker 1>` …) say who spoke when that was asked for.

This page describes **what the project does and what it does not do**. Command details are in [README.md](README.md); the version described here is the one in [VERSION](VERSION). Claims are marked **measured**, **tested** or **not tested** where the project's own documentation says so; anything unmarked is a description of how the code is designed to behave.

## 1. At a glance

| | |
|---|---|
| **Job** | record or import audio → full multilingual transcript (language per line) → translations on request. Nothing else. |
| **Speech recognition** | Whisper `large-v3-turbo` (about 99 languages) through [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU; on Apple silicon, [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on the GPU when installed |
| **Who spoke** | optional: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) speaker diarization; optionally [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| **Translation** | Claude (needs an API key and internet), or offline: NLLB-200 (about 90 languages) or, on Apple silicon, Hy-MT3 7B (102 languages; the app's default there) or TranslateGemma 12B (51 languages) |
| **Input** | any audio/video format ffmpeg reads (mp3, m4a, wav, flac, ogg, opus, aac, wma, aiff, mp4, mov, m4v, mkv, webm, avi …); the microphone; the computer's own sound |
| **Output** | `<name>.transcript.txt`, `<name>.<Language>.txt` per translation, and the recording (`<name>.wav`, or `<name>.mic.wav` + `<name>.system.wav` for a call) |
| **Longest recording** | 4 hours by default, up to 8, counted in recorded audio (not wall-clock) |
| **Languages spoken** | automatic per stretch of speech among ~100, or restricted to a list you give |

## 2. Three ways to use it

1. **The Mac app.** Add files or whole folders (dropped in or picked; every audio/video file at any depth is taken), record, choose options, press **Transcribe**. Results appear as text files with *Show* and *Open* buttons. The app bundles its own Python runtime; the first launch runs **Set Up Speech Engine** once (about 2 GB in all, a few minutes, needs internet; nothing else has to be installed).
2. **The command line.** `python3 mt.py <command>` (on Windows `.\.venv\Scripts\python.exe mt.py <command>`); commands: `transcribe`, `translate`, `record`, `call`, `follow`, `show`, `list`, `rename`, `devices`, `prepare`.
3. **A Claude Code session** opened in this folder: it follows the project's instructions file and runs the same `mt.py` for you. Recording needs a microphone, so it works only when Claude Code runs on your own machine; transcribing files works anywhere.

The Mac app is a window around the same `mt.py` (bundled inside it), so both produce the same files.

## 3. Features

### 3.1 Input
- **Audio and video files**, one or many in one run; from a video only the **audio track** is used (`--track N` chooses one when there are several; by default the first, with a note). A video without sound is reported as an error.
- **A batch continues after a bad file**: the unreadable file is reported and the exit code is 1 at the end.
- **Microphone recording** with a chosen input device (by index, exact name, or a unique part of the name).
- **Two-track call recording** (§3.4).
- **Two finished recordings** can be combined later with `call mic.wav system.wav`.
- **Plain text** can be translated without a transcript format (`translate`); lines without the `[time] [lang]` prefix are translated as they are.

### 3.2 Transcription
- **Local speech recognition**; offline after the first model download.
- **A language per stretch of speech.** Speech is cut at pauses into chunks and the language of each chunk is detected in the same pass that decodes it, so detection costs nothing extra.
- **Automatic or restricted language choice.** With no list, any of the ~100 Whisper languages; with a list (`--language ru,en,de`), each chunk is detected among those only; with one language it is forced for the whole file.
- **Guess marking.** If Whisper is at least 70 % sure a chunk is in a language you did *not* list, the words are kept in that language and the tag gets a `?`.
- **Granularity** (`--granularity coarse|normal|fine|finest|adaptive|auto`) sets how finely speech is cut for the language decision, trading speed for following fast language switches (§5).
- **Notes.** Lines starting with `# ` are notes, not speech (stretches not recognised, how much was decoded, paused periods, capture problems). They are kept in every translation file.
- **Whisper's invented text is removed.** On silence, music or noise Whisper tends to write subtitle credits ("Subtitles by the Amara.org community" and similar). 82 patterns remove lines that are only such credits, in any language; standalone outros are dropped only under strict conditions, and short real phrases ("Thank you.") are never dropped by text alone. A note says how many lines were left out.
- **Hardware engines:** CPU (int8) everywhere; Apple GPU through MLX on Apple silicon; on Windows an optional NVIDIA GPU (`requirements-cuda-windows.txt`, `MT_DEVICE=auto|cpu|cuda`; when CUDA cannot load or decode, the CPU is used with a note). The project's measurements are CPU only.
- **Choosing languages in the Mac app:** *Languages spoken in the audio* and *Also translate to* are searchable drop-downs. The spoken list has all 100 languages Whisper recognises; the offline translation list has 101 names (Traditional Chinese, Zulu and Brazilian/European Portuguese included), and with Claude Breton, Hawaiian and Latin are offered too. Other NLLB languages by code (for example `ibo_Latn`), and any language for Claude, stay available on the command line (`--to`).

### 3.3 Recording
- **Length:** 4 h by default, up to 8 h, measured in recorded audio, so a sleeping computer does not use it up. A notice comes 10 minutes before the stop; **Extend by 1 h** (app) or `kill -USR1 <pid>` (terminal) moves the limit.
- **Pause / Resume** (app, or `kill -USR2 <pid>`): nothing is recorded meanwhile; the transcript leaves the pause out and a timed note says where and for how long. **On Windows**, which has no such signals, set `MT_CONTROL=stdin` and type `pause`, `resume`, `extend`, `stop` or `cancel` plus Enter (Ctrl+C also stops).
- **Transcribe while recording** (on by default in the app; `--live` on the command line): each stretch of speech is decoded as soon as a pause closes it, so at the end only the last stretch is left. If the computer is slower than the meeting, the window says how far behind it is; the final result is complete either way.
- **The live transcript never waits for the translation (since 1.13.0).** One thread owns the GPU and schedules the work: a closed stretch of speech always goes first, and the translation runs in between, one token at a time (or one 256 MB slice of the model's load, or one NLLB batch), so a line appears at most about a second later than without translation. When the translation falls more than 30 s behind, the older stretches are left to the final pass (marked *at the end* in the window) and the live translation goes back to the newest; it also leaves a fifth of each minute free. When the recording stops, the translation stops at once (the final pass makes the same text with the same model) and the live transcript is finished for at most 90 s. The window's telemetry column shows the lags, the queues, the models' states and the memory as it goes.
- **Record only, transcribe later** (`--no-transcribe`).
- **Two things at once** (app): one recording slot plus a queue of processing jobs, so a new interview can be recorded while the previous one is still being transcribed; the queue can be reordered, jobs removed or cancelled.
- **Sleep handling:** while a job runs the Mac does not sleep by itself (the command line uses `caffeinate`); a closed lid still sleeps it, and the time asleep is reported as a note and filled with silence so timestamps stay right.
- **Level bar and timer** show that sound is arriving.

### 3.4 Online calls: your side and the other side
- Two tracks at once: **your microphone** and **the sound the computer plays** (the other participants). Each is transcribed separately, merged by time, and every line is labelled `<You>` or `<Them>` (names can be changed).
- **macOS 14.2 or newer:** a Core Audio process tap, which needs only the *System Audio Recording Only* permission (no screen recording, no monthly re-approval prompt). **macOS 13–14.1:** ScreenCaptureKit with the *Screen & System Audio Recording* permission.
- **Device choice:** the input device and the output whose sound counts as the other side, chosen by Core Audio UID, so the choice survives relaunch and replugging.
- **Speakers vs headphones:** while the sound plays through a loudspeaker the microphone also hears the other side, so the app unticks the second track by itself and records the microphone alone; for headphones it ticks it again (only if it unticked it itself). Built-in speakers, HDMI, DisplayPort and AirPlay are recognised; a Bluetooth or USB speaker is learned from your own untick.
- **Echo handling:** Apple's echo cancellation on the Mac's current microphone; whatever still comes through is trimmed from your lines (words shared with a simultaneous line of the other side are removed; a line that is only echo is dropped; `--keep-echo` keeps everything).
- **Self-recovery:** a changed input device, a dead capture, a track that delivers nothing for 10 s or only zeros for 20 s are each handled (restart, fall back to the microphone alone) and reported as a note. The call never fails as a whole because one side could not be captured. A `capture.log` is written line by line into the recording's folder for diagnosis.
- **Command line:** `record --system-device N` records a second input as the other side (a *Monitor of …* source on Linux, BlackHole on macOS, a "Stereo Mix" input or a virtual audio cable on Windows; capturing the computer's sound directly on Windows is planned for a Windows app that does not exist yet).

### 3.5 Who spoke (`--speakers`)
- `--speakers auto` labels lines `<Speaker 1>`, `<Speaker 2>` … in order of appearance; in a call, `<Them 1>` … for the other side and `<You 1>` … if several people share your microphone.
- A number (`--speakers 3`) counts the other side; automatic counting did better in the project's test, so give a number only when the result is wrong.
- **Overlap** of half a second or more becomes a note (`overlap 00:04:00–00:04:05 Speaker 1 / Speaker 2`).
- **Naming:** `rename <id or folder> "Speaker 1=Anna"` (app: *Name Speakers…*) updates the stored transcript and the text files already written.
- **Measured** on a 10-minute synthetic six-voice English/Russian/German test: 6 of 6 speakers found, 96 % of sentences labelled right, diarization error rate 9.4 % (with the number forced to 6: 81 % and 25 %). Synthetic voices, one CPU; real recordings may differ.
- **pyannote** (`--diarizer pyannote`) is an optional alternative: about 2 GB of packages, Python 3.10+, a gated model on huggingface.co that needs your acceptance of its conditions and a Hugging Face token, slower on a CPU. **Not measured** by the project.

### 3.6 Translation
- **Any number of target languages** per run (`--to English --to Russian`); each becomes its own file, written as soon as it is ready, so a failure in the third language never hides the first two.
- **Per-line source language:** each line is translated from the language it is tagged with, so mixed-language transcripts work; lines already in the target language are left alone. Labels (`<Them>`) and notes are kept.
- **Four translators:**

| Translator | Needs | Languages | Quality / notes |
|---|---|---|---|
| **Claude** (default for the command line) | `ANTHROPIC_API_KEY`, internet; text is sent to the Claude API | any Claude handles | the project's reference quality |
| **NLLB-200** (`--offline`) | a 1.4 GB model, once; then no internet | about 90 common languages (by name or code) | below Claude's; literal at times; **non-commercial licence** |
| **Hy-MT3 7B** (`--offline --translator hy`, the app's default on Apple silicon) | Apple silicon, `mlx-lm`, a 4.2 GB download, about 6 GB beside the speech model (estimate) | 102 (in this app 92 of 101 targets); other lines fall back to NLLB | vendor figures only, no independent evaluation yet; Apache-2.0, commercial use allowed |
| **TranslateGemma 12B** (`--offline --translator gemma`) | Apple silicon, `mlx-lm`, 16 GB memory advised, a 7 GB download | 51 languages; other lines fall back to NLLB | better than NLLB on the project's 5-sentence spot check (a spot check, not a benchmark); commercial use allowed by its terms |

- **Live translation, side by side** (`record --live --to Russian`, app: *Also translate to*): each line is translated as it appears, without ever holding up the transcript (since 1.13.0, §3.3); a line the Mac could not translate in time says *at the end* and is translated by the final pass. Since 1.11.0 the pane shows the text and one column per translation, with a picker for which columns to show; *Copy* puts them on the clipboard tab-separated for a spreadsheet.
- **One 7B translator per Mac** (since 1.13.0): Hy-MT3 or TranslateGemma is loaded by one process at a time. A job translating with it pauses after its current line when a recording wants to translate live, and goes on from where it stopped afterwards; a job beside a live recording uses the GPU at most half the time. Below 12 GB of memory (16 GB for TranslateGemma) the live translation uses NLLB and the chosen model translates at the end.
- **Reuse:** finished translations are cached and reused; a translation made by one model is reused only by the same model.
- The Mac app translates **offline by default** (untick *Translate offline on this Mac* to use Claude).

### 3.7 Files and folders
- **Output folder** (app default `~/Documents/Meeting Transcriber`, or `MT_OUT` on the command line): each meeting goes into a subfolder named after the hour it began (`26-11-22 14`). A second meeting in the same hour is numbered (`recording-2.*`); nothing is overwritten.
- **Next to the audio** when no output folder is set (`call.m4a` → `call.transcript.txt`); with `--out` and several same-named files (for example one `audio_only.m4a` per Zoom folder), the parent folder's name is put in front.
- **Stored meetings:** working copies (`transcript.json`, checkpoints, `capture.log`) live in the app's data folder; `list`, `show [id] --to <Language>` and the app's *Recordings* box (Transcribe, Continue, Translate, Reveal) read from there.
- **Recording is kept** beside the text (on an APFS disk as a clone that takes no extra space).

### 3.8 Robustness
- **Nothing finished is lost.** Every finished chunk is written to a checkpoint; a run that is killed, crashes or loses power continues from it when started again with the same file and settings. A different model, language list or granularity starts over and says so.
- **A chunk the model chokes on** becomes a `# … chunk failed` note; the rest goes on.
- `transcript.json` is written whole or not at all; offline translations are saved sentence by sentence.
- **Cancel** keeps the audio (nothing is transcribed) during a recording; during a transcription what is done stays.
- **Early checks:** language names, devices and API-key presence are checked before a long job starts.

### 3.9 Privacy and safety of the app's own data
- The API key is stored in the macOS Keychain (if macOS refuses, in a file only you can read).
- Speech recognition, speaker separation and offline translation never send audio or text anywhere. Downloads happen at setup or first use (speech model, ffmpeg if missing, diarization model of about 47 MB, translation models).
- The Mac app is signed with one self-signed identity created by `app/make_signing_identity.sh`, so macOS keeps its permissions across rebuilds. A build without that identity is ad-hoc signed and macOS forgets the permissions at every update.

## 4. Recipes

```bash
# transcribe a file, translate to two languages
python3 mt.py transcribe call.m4a --to English --to Russian

# a meeting in three languages, speakers named afterwards
python3 mt.py transcribe meeting.m4a --language ru,en,de --speakers auto
python3 mt.py rename "<id or folder>" "Speaker 1=Anna" "Speaker 2=Boris"

# no internet, no API key
python3 mt.py transcribe call.m4a --to English --offline

# record the microphone for up to 2 hours, transcribing live, translating live
python3 mt.py record --name weekly --hours 2 --live --to English

# an online call from two finished recordings
python3 mt.py call mic.wav system.wav --me Anna --them Team

# a video with several audio tracks: the second one
python3 mt.py transcribe lecture.mkv --track 1

# continue an interrupted transcription
python3 mt.py transcribe talk.m4a --into data/<id>
```

## 5. Language switching: what it catches and what it cannot

The language is decided **once per chunk of speech**, so how speech is cut decides how quickly a switch is noticed. **Measured** on a 10-minute synthetic English/Russian/German meeting (6 voices, 167 sentences, gaps of 0.3–1 s, `--language en,ru,de`, `large-v3-turbo`, 4-core 2.1 GHz Xeon CPU):

| `--granularity` | Sentences lost | Word error rate | Lines tagged right | Time ÷ audio length |
|---|---|---|---|---|
| `normal` | 68 of 167 | 50 % | 92 % | 0.33 |
| `fine` | 18 | 21 % | 96 % | 0.68 |
| `finest` | 4 | 11 % | 100 % | 0.99 |

`auto` (the default) means `finest` when several languages are listed, `normal` when one is, and `adaptive` (finer cuts only where a chunk looks mixed) otherwise. One run per setting, synthetic voices, so treat the figures as indications (the project estimates about ±5 % on the times).

## 6. Requirements and footprint

- **Mac app:** macOS 13 or newer. The Apple-GPU engine and the call-sound process tap need newer systems (GPU parts: Apple silicon, macOS 14+, Python 3.10+; process tap: macOS 14.2+).
- **Command line (macOS, Linux):** Python 3.9 or newer (CI tests 3.9 and 3.12); `pip install -r requirements.txt` (includes `static-ffmpeg`, which fetches ffmpeg when the system has none). Recording needs `libportaudio2` on Linux.
- **Command line (Windows, since 1.11.0):** Windows 10 or 11 on x64, PowerShell, Python 3.12 and the Visual C++ runtime (the README installs both with `winget`); an NVIDIA GPU is optional. Windows on Arm works only through x64 Python under emulation and was not measured.
- **Disk:** about 2 GB for the speech engine at setup; about 3 GB of model cache including the offline translator (1.4 GB); Hy-MT3 adds 4.2 GB, TranslateGemma 7 GB; pyannote about 2 GB.
- **Memory:** the speech model about 1.6 GB; a 3-hour recording is about 350 MB of 16 kHz WAV (115 MB per hour per track); transcribing holds audio as float32 (about 230 MB per hour). A Mac with 8 GB can do a 4-hour file if other large programs are closed. A live recording with offline translation holds both models (about 3 GB); Hy-MT3 about 4.5 GB (about 6 GB beside the speech model; estimates, not measured; the weights on disk are 4.2 GB for Hy-MT3 and about 1.6 GB for the speech model, 5.8 GB together); TranslateGemma about 7 GB (8.5 GB beside the speech model). Since 1.13.0 a recording process caps MLX's buffer cache at 1 GB (`MT_MLX_CACHE_MB`), and only one process on the Mac holds a 7B translator at a time.
- **Time:** about 5 s per chunk on the measured CPU whatever its length, because Whisper pads each chunk to 30 s, so time grows with the number of chunks. An Apple-silicon GPU is expected to be several times faster (**not measured**).

## 7. Limitations

### Recognition and language
- **A language change inside a sentence is not caught** at any setting (0 of 10 in the project's test at `normal`/`fine`, 1 of 10 at `finest`): Whisper's output carries no language per word, so the part in the other language shares its sentence's tag or is dropped.
- **A forced single language** produces wrong text for any other language in the file.
- **Very short utterances** give the detector little to judge from; a language list helps.
- **Some minority-language stretches stay invisible** to the automatic mixed-chunk trigger (5 chunks with a dozen English words inside a Russian or German turn in the test).
- **Whisper has no Zulu** as a spoken language (it can be a translation target).
- **Quiet, distant, noisy or unknown-language speech** may be reported as "speech not recognised" for that stretch.
- **Word errors exist.** In the project's own synthetic tests the best setting still had an 11 % word error rate; real audio has not been benchmarked.

### Speakers
- "Who spoke" is a **guess from voices**: alike voices (the same person on two devices, two similar speakers) can be merged or split. Without `--speakers`, nothing is labelled. Two people talking at once are reported as an overlap note, not separated into two lines.

### Calls
- The other side is recorded **only if you ask**; otherwise it is the microphone alone.
- **There is no way for the app to ask macOS whether the System Audio Recording permission exists**; without it the other side's track is silent, and the app can only notice that after about 20 s.
- **One Bluetooth device as both microphone and output** (a headset) switches to its call mode while recording and sounds worse until it ends; use the Mac's built-in microphone as the input to avoid this.
- **Echo cancellation works only with the Mac's current input**; with a microphone chosen by name it is off (a note says so). For a call on loudspeakers the better choice is then the microphone alone, which is what the app does by itself for speakers it recognises (this reading is an inference from the documentation, not a separate test).
- **Not tested:** the chosen output device (tap limited to one output), a change of output during a call, a chosen microphone through its own input unit in real calls, `--system-device` with real devices, calls on Linux, and calls on Windows (no real Windows PC has run the tool).
- **Measured on one Mac only** (macOS 27.2, a Bluetooth headset and a Bluetooth speaker): the call-sound capture behaviour. Other Macs, other macOS versions and other devices may differ; the project documents the failures it saw on that Mac and how the code now handles them.
- **Consent:** recording other people requires their consent in many jurisdictions.

### Translation
- **Offline translation is below Claude's quality.** NLLB works one sentence at a time, can be literal ("Guten Tag zusammen" became "Have a good day together" in a test), writes Chinese with ASCII punctuation, and a wrong language tag gives a wrong translation. Names, numbers and idioms deserve a check.
- **NLLB's weights are non-commercial (CC-BY-NC 4.0), and Meta calls NLLB-200 a research model**, not meant for certified translation. Use Claude or another model for commercial or legally binding work. Hy-MT3's licence (Apache-2.0) and TranslateGemma's terms allow commercial use.
- **Translation with Claude sends the transcript text to the Claude API** and costs API usage.
- **TranslateGemma** is limited to 51 languages, to Apple silicon, and to machines with about 16 GB of memory; a whole 3-hour meeting takes about an hour per language at the end (estimated, **not measured**), less with live translation.
- **Hy-MT3**'s quality figures are Tencent's own (no independent evaluation yet, none per language pair for Russian or German); it runs on Apple silicon only, its memory and speed on a Mac are **not measured**, and its 4-bit weights are a third party's copy (NetProjects), pinned and checked by SHA-256.
- **Live translation can fall behind** on a slow Mac with several targets: the transcript stays live, but many lines are then translated only at the end. The one-token-at-a-time scheduling of 1.13.0 is proven on stand-ins (the Python tests and the lag driver), **not yet on a real Apple GPU**.
- **Offline translation speed:** about 0.4 s per two-sentence line on a 4-core CPU, roughly 15–25 minutes per language for a 3-hour meeting (measured on Linux x86, not on a Mac).

### Platform and scope
- **The app is macOS only; there is no Windows window app yet.** The command line's file transcription is tested on Linux (the project's CI).
- **Windows is new and only partly proven (1.11.0).** `mt.py` has **not been run on a real Windows PC**. On a Windows CI runner (no sound device, no GPU) the Windows tests run, and a second job runs the real speech engine for `prepare`, `transcribe`, `devices`, `--help` and `--version` only; `call`, `translate`, `show`, `list`, `rename`, `--speakers`, `record` and the GPU path are covered by tests that fake the hardware, not by that job. Names and paths must obey Windows rules (no `< > " | ? *`, no reserved names such as `CON`, at most 64 characters, and the data folder path short enough for Windows' 260-character limit unless long paths are enabled), `--device` matches only WASAPI devices, and Windows Security's *Controlled folder access* can refuse writes to Documents. Recording on Linux or Windows depends on PortAudio and your devices.
- **Video: only the audio track** is used; no picture, no subtitles, no extra audio file saved.
- **Output is plain text**: no subtitle formats (a search of `mt.py` found no SRT or VTT writer), no summaries, no action items, no cloud storage, no live captions for other people. The project deliberately does one job.
- **The macOS parts have limited verification.** The Linux continuous-integration job runs the Python tests and the Swift core-logic tests; the macOS build and smoke test run on a macOS runner. The recording, permission and audio-device behaviour on real Macs rests on the project's own measurements on one machine.
- **Time and speed depend on hardware**; the figures above come from one 4-core CPU.

## 8. Licences and attributions

- Speech model: OpenAI Whisper weights as published for `large-v3-turbo`, run through faster-whisper (SYSTRAN) or mlx-whisper. Diarization: sherpa-onnx with pyannote segmentation 3.0 and 3D-Speaker ERes2Net embeddings. Offline translation: NLLB-200 (Meta, CC-BY-NC 4.0) converted for CTranslate2 by OpenNMT; Hy-MT3 7B (Tencent, Apache-2.0; the 4-bit MLX copy by NetProjects); TranslateGemma (Google; Gemma Terms of Use). The ffmpeg binaries fetched by `static-ffmpeg` are GPL builds of [FFmpeg](https://ffmpeg.org).
- Each component keeps its own licence; check them before commercial use. **This repository's own licence is not yet stated**: until a LICENSE file is added, the default copyright rules apply.

## 9. Where to look next

- [README.md](README.md): every command, option and behaviour in detail, including the measurements.
- `tests/test_mt.py` and `app/Tests/CoreTests/main.swift`: the Python and Swift test suites.
