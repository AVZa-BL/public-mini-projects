# meeting-transcriber — record, transcribe, translate

Records a meeting (4 hours by default, up to 8, counted in recorded audio; a notice comes 10 minutes before it stops and you can extend it) **or loads any audio file**, writes a full transcript where **every line carries its language**
(`[ru]`, `[en]`, `[de?]` — the `?` marks a low-confidence guess), and translates it to any language on request. The transcript and each translation are saved as separate text files.
Use the **Mac app**, a terminal, or a Claude Code session in this folder (the instructions it follows are in `CLAUDE.md`, which is kept only in the development repository).

```text
[00:00:04] [en] Let's start with the budget.
[00:00:09] [ru] Хорошо, я подготовил цифры.
[00:00:15] [de?] Danke, das passt.
```

## How it works

| Step | Engine | Runs |
|------|--------|------|
| Speech → text, language per ~30 s speech chunk | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper `large-v3-turbo`, ~99 languages) on the CPU; on Apple silicon [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on the GPU (`MT_ENGINE=auto`, the default, picks it when installed) | locally, offline after the first model download |
| Who spoke (`--speakers`) | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) speaker diarization: pyannote segmentation 3.0 + 3D-Speaker ERes2Net embeddings (47 MB, fetched at setup); optionally [pyannote.audio](https://github.com/pyannote/pyannote-audio) | locally |
| Translation (default) | Claude API | needs `ANTHROPIC_API_KEY` and internet |
| Translation (`--offline`) | NLLB-200 (Meta) on CTranslate2; on Apple silicon also Hy-MT3 7B (Tencent) or TranslateGemma 12B (Google) through mlx-lm | locally; downloads the model once (1.4 GB, 4.2 GB or 7 GB), then no internet and no key |

Claude itself does not take audio input, so a speech model does the transcription; translation is Claude's, or the local model's when you choose offline.

## The Mac app (no Terminal needed)

A native window for everything below: add audio or video files **or whole folders** (use *Add Files or Folders…* or drop them in; a folder
brings in the audio and video files inside it, at any depth), record, pick languages, press **Transcribe**;
each result is a text file with **Show** and **Open** buttons. macOS 13 or newer.

**Choosing languages** (since 1.10.6). *Languages spoken in the audio* and *Also translate to* are drop-downs: the button names
what is chosen (the first three, then "+n"), and its list has a search field, the most used languages at the top and the rest
A–Z, with the chosen ones ticked and highlighted; it stays open while several are picked (*Clear*, *Done*). The spoken list has
every language Whisper recognises (100). The translation list, offline, has the 101 languages `mt.py` knows by name for the
offline models: Whisper's without Breton, Hawaiian and Latin, plus Traditional Chinese, Zulu, and Brazilian and European
Portuguese, which TranslateGemma keeps apart (Hy-MT3 and NLLB translate both as Portuguese). With Claude (*Translate offline* unticked) Breton, Hawaiian and Latin are offered
too. Each name is one `mt.py` accepts (`tests/test_mt.py` checks every one against `whisper_code` and `nllb_code`). The *Others*
text fields are gone: languages typed there in an earlier version are added to the choice once, at the first launch. Other NLLB
languages, by their code (`ibo_Latn`), and any language for Claude remain available on the command line (`--to`).

```bash
./meeting-transcriber/app/build_app.sh --install      # builds, then copies to ~/Applications (needs the Xcode Command Line Tools)
open ~/Applications/"Meeting Transcriber.app"
```

Add `--arch arm64` for Apple silicon only (faster). CI also builds it: open the pull request's *Checks* tab and download the
`Meeting-Transcriber-*` artifact (ad-hoc signed: the first launch of a download needs *System Settings > Privacy & Security > Open Anyway*).

**First launch:** press **Set Up Speech Engine** once. The app copies its own Python runtime (CPython 3.12 from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone), bundled for Apple silicon and Intel) into
`~/Library/Application Support/Meeting Transcriber/`, creates a private environment there, installs faster-whisper, and downloads
the Whisper model and, when the Mac has no ffmpeg, ffmpeg and ffprobe (about 2 GB in all, a few minutes; internet is needed only
for this). Nothing else has to be installed: no Python, no Homebrew, no Command Line Tools. The ffmpeg binaries come from the
`static-ffmpeg` package (GPL builds of [FFmpeg](https://ffmpeg.org)); a Homebrew ffmpeg is used when present. For translation with Claude press **Set Key…** and paste your API key (kept in the Keychain; if macOS refuses it, the window says so and keeps it in a file only you can read, `~/Library/Application Support/Meeting Transcriber/api-key`), or keep **Translate offline on this Mac** ticked (the default: nothing leaves the Mac; untick it to translate with Claude)
(no key; see *Offline translation* below). On Apple silicon the offline model is Hy-MT3 7B unless you chose another (*Offline
model*); on an Intel Mac it is NLLB.
**Output folder.** Every meeting's files go to the folder chosen under *Output folder* (default `~/Documents/Meeting Transcriber`),
into a subfolder named after the hour the meeting began, `yy-mm-dd hh` (`26-11-22 14`): the transcript (`<name>.transcript.txt`),
one file per translation (`<name>.<Language>.txt`) and the recording itself (`<name>.wav`, or a call's `<name>.mic.wav` and
`<name>.system.wav`; on an APFS disk a clone that takes no extra space, elsewhere a copy). Two meetings of one hour share the
subfolder; the second one's files are numbered (`recording-2.*`) and a note says so, so nothing is overwritten; a rerun or a later
translation of a meeting keeps its own name there, and *Name Speakers…* changes those files too. The app passes the folder to
`mt.py` as `MT_OUT`; on the command line `export MT_OUT=~/Documents/Meetings` does the same for every command (`--out` still names
an exact folder). The working copies (the audio as recorded, `transcript.json`, the checkpoints, `capture.log`) stay in
`~/Library/Application Support/Meeting Transcriber/data/<id>/`. The window's **Recordings**
box lists what is there: recordings that were never transcribed (a cancelled or interrupted job) with a **Transcribe** or **Continue**
button (a transcription that was begun goes on from its checkpoint), and stored meetings with **Translate** (into the ticked languages,
beside the transcript; finished translations are reused) and **Reveal**. In *Transcript files* mode a **Language of plain text files**
field names the language of text without `[tags]`, which offline translation needs.

**Two things at once.** The window has a **Recording** panel and a **Processing** panel. A recording has its own slot (one at a
time: the microphone, or the microphone plus the sound the Mac plays); transcriptions and translations are **processing jobs** that
run one after another in a queue. When a recording stops, its transcription joins the queue, so a new interview can be recorded
while the last one is still being transcribed, and files or stored meetings can be queued behind it (the waiting jobs are listed
with Remove and reorder buttons; **Cancel** on the running job lets the next one start; Quit cancels everything). The app records
with `mt.py record --no-transcribe --live` and queues `mt.py transcribe audio.wav --into <folder> --stem <name>` (or `call … --into`),
which continues from what was transcribed live. A job that runs beside a recording takes half the cores (`MT_THREADS`), so the live
transcript keeps up; with both running, two speech models are in memory (about 3.5 GB with `large-v3-turbo`).

**The window and its telemetry** (since 1.13.0). The window has three columns (at least 980 × 660 points, 1280 × 860 at first):
**Controls** (every setting, as before; *View > Show Settings*, ⌥⌘S, hides it), **Session** (the recording or the job: its
header, the transcript, the processing strip and a status bar) and **Telemetry** (cards with charts of what `mt.py` reports;
shown by default; *View > Show Telemetry*, ⌥⌘T, hides it, and the choice is kept). The status bar is always there, for example
`whisper-large-v3-turbo · Apple GPU (MLX) | Hy-MT3-7B-MLX-4bit ready (load 6.2 s) | lag 3 s · translation 8 s | MLX 6.1 GB ·
pressure normal | CPU 38 %`; each part opens its card. The transcript's **lag chip** is the number of seconds between the end
of what was said and its line on screen: green with a tick below 10 s, yellow with a triangle from 10 s, red with a stop sign from 30 s (the translation's lag: 15 s and
45 s). A translation cell says *at the end* when that line is left to the translation at the end (see *While it works*), and
*Timings* shows how long each translation took and which model made it. The cards: live health (both lags over time),
the speech engine (speed per chunk, × real time), the translator (its state: downloading, checking files, waiting for the other
job, loading, ready, behind; tokens per second; its share of the last minute), the queues, memory (MLX's own figures, the
footprint of `mt.py` and of the app, memory pressure, swap, audio blocks the recorder dropped), CPU, GPU and heat, languages,
and the last events in plain words. The GPU's busy share comes from a counter Apple does not document (labelled so). Nothing of
the telemetry is written into a transcript or a file: `mt.py` sends it, with `MT_EVENTS=1` (which the app sets), as lines
`@@STAT<tab>kind<tab>{json}` on stdout (kinds `chunk`, `tr`, `trskip`, `queue`, `model`, `sys`: per decoded chunk, per finished
translation, per skip, every second, per change of a model's state, every 2 s; never per token), and it lives in the window
only. To see it on the command line, write the events to a file and read it in a second terminal (a `| grep` behind `record`
would be ended by the Ctrl+C that stops the recording, and `mt.py` would then fail on the closed pipe):
`MT_EVENTS=1 python3 mt.py record --live --to German --offline --translator hy > events.log 2>&1`, and
`tail -f events.log | grep --line-buffered -E '^@@(STAT|LINE)'`.

The app is a front end for `mt.py` (bundled inside it), so it produces exactly the files described under *Use*.
The same `VERSION` file versions the app, the command-line tool and the launcher entry. Bump it when you change either.

## Setup (command line)

```bash
cd meeting-transcriber
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes static-ffmpeg: ffmpeg/ffprobe are fetched once if the system has none
brew install ffmpeg                      # optional: a system ffmpeg is used when present
export ANTHROPIC_API_KEY=sk-ant-...      # only for translation
```

## Windows (command line)

Since 1.11.0 `mt.py` itself is meant to run on Windows 10 and 11 (x64), from PowerShell. It has not been run on a real
Windows PC yet. On a Windows CI runner (Windows Server 2025, no sound device, no GPU) the tests run. A second CI job,
*Engine on Windows* (`tools/win_e2e.py`), is meant to run the real speech engine there (faster-whisper on CTranslate2,
Whisper `tiny`, on the CPU); whether it passes is shown by the latest run of the workflow `meeting-transcriber.yml` in the
development repository (CI runs only there; the public copy of this folder has none). It installs
`requirements.txt` into a venv like the lines below, but with setup-python's `python` for `py -3.12` and without the
two `winget` lines, runs `mt.py prepare` (ffmpeg fetched by static-ffmpeg), and transcribes two English sentences from a
file with a Cyrillic name, with stdout through a pipe and no `PYTHONUTF8`: once plainly, and twice with
`MT_CONTROL=stdin` and `@@` events. The sentences are spoken by Windows' own speech synthesizer, or, when the runner has
no English voice, come from a committed recording of them (`tools/win_e2e_sentence.flac`, made with espeak-ng); each
transcript must hold words of both, every line tagged `en`, with no failed chunk. Only `prepare`, `transcribe`,
`devices` (with no sound device), `--help` and `--version` are run by that job: `call`, `translate`, `show`, `list`,
`rename`, `--speakers`, `record`, the GPU and the `MT_DEVICE` fallback in the table below are not.

| | On Windows today |
|---|---|
| `transcribe`, `call`, `translate` (Claude or `--offline`), `show`, `list`, `rename`, `--speakers` | as on the Mac |
| `record` from a microphone | yes; Pause, Extend, Stop and Cancel as lines on stdin (below), or Ctrl+C to stop |
| the computer's sound (`--system-device`) | only through a "Stereo Mix" input or a virtual audio cable; capturing it directly comes with the Windows app |
| an NVIDIA GPU | optional (`requirements-cuda-windows.txt`); without it, or when CUDA fails, the CPU, with a note |
| a window app | not yet: its technology is chosen after a phone test |
| Windows on Arm | only x64 Python under emulation (CTranslate2 has no arm64 wheel); not measured |

Install, in *Windows PowerShell* (from Start), in a checkout of this repository. The installers ask for administrator
rights (the Python launcher for all users, the Visual C++ runtime); afterwards close every PowerShell window and open a
new one, so that `py` is on its PATH. Calling `.\.venv\Scripts\python.exe` by its path needs no change of the execution
policy (`Activate.ps1` would):

```powershell
winget install --id Python.Python.3.12 -e --source winget --accept-source-agreements --accept-package-agreements
winget install --id Microsoft.VCRedist.2015+.x64 -e --source winget --accept-source-agreements --accept-package-agreements
#   Python 3.12.10 (or the installer from python.org, "Add python.exe to PATH"); the Visual C++ runtime (MSVCP140.dll)
#   that ctranslate2 and onnxruntime need
cd meeting-transcriber
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt               # static-ffmpeg fetches ffmpeg once if the PC has none
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda-windows.txt  # optional, an NVIDIA GPU: cuBLAS and NVRTC, about 0.63 GB
.\.venv\Scripts\python.exe mt.py prepare
$env:ANTHROPIC_API_KEY = "sk-ant-..."                  # only for translation with Claude
```

Use it as under *Use* below, with `.\.venv\Scripts\python.exe mt.py` for `python3 mt.py`:

```powershell
.\.venv\Scripts\python.exe mt.py transcribe "$HOME\Downloads\meeting.m4a" --to English --offline
.\.venv\Scripts\python.exe mt.py devices               # every device once per host API; --device matches the WASAPI ones
$env:MT_CONTROL = "stdin"
.\.venv\Scripts\python.exe mt.py record --name weekly --live --offline --to English
#   then type pause, resume, extend, stop or cancel and press Enter (what you type appears only then); Ctrl+C also stops
```

**Checking a PC first (`tools/win_probe.py`).** A one-time report of what matters for this app on that PC: Windows
version, Python, the Visual C++ runtime, Smart App Control, Controlled folder access, the microphone privacy switches,
the sound devices, the computer's sound (loopback), headphones or speaker, NVIDIA/CUDA, ffmpeg. Nothing is uploaded;
the home folder, account and computer names are replaced by placeholders. With `--interactive` it asks you to speak,
then to play some sound, then to stop it (Enter each time), and loads Whisper `tiny` (about 75 MB) on the CPU and the GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install soundcard pyaudiowpatch pycaw comtypes      # the probe's optional extras
.\.venv\Scripts\python.exe tools\win_probe.py --interactive --fetch-ffmpeg --out "$([Environment]::GetFolderPath('Desktop'))\probe.json"
```

**Pause, Extend, Stop and Cancel without signals (`MT_CONTROL=stdin`).** Windows has no `SIGUSR1`/`SIGUSR2`, and there a
`kill` is `TerminateProcess`: nothing is saved, the WAV header stays up to 30 s stale. With `MT_CONTROL=stdin` set, `mt.py`
reads one command per line on stdin (case and surrounding spaces do not matter; any other line is a warning on stderr):

| Line | During `record` | During `follow` | During `transcribe`, `call`, `translate` |
|---|---|---|---|
| `pause` / `resume` | pauses / resumes (as `kill -USR2`); a second `pause` or `resume` changes nothing | ignored | ignored |
| `extend` | one hour more (as `kill -USR1`) | ignored | ignored |
| `stop` | ends as Ctrl+C does: saved, then transcribed if that was asked | finishes the current chunk and ends | as Ctrl+C |
| `cancel` | ends as `SIGTERM` does: the audio is kept, nothing is transcribed | as `stop` | ends at once as from `SIGTERM` (exit code 15 on Windows); `--into` continues it later |
| end of stdin | as `stop` | as `stop` | as `cancel` |

The end of stdin counts so that a window app that crashes still leaves a saved recording and no job running on its own; a
front end therefore keeps the pipe open until `mt.py` has ended. Without `MT_CONTROL`, `mt.py` never reads stdin (the Mac
app gives it `/dev/null` and uses the signals, which keep working on macOS and Linux either way). A front end also gets:
`@@` events in UTF-8 with `\n` line ends whatever the code page; ffmpeg and ffprobe started without a console window and
without stdin; the PC kept awake while a long command runs (`SetThreadExecutionState`, not a closed lid).

Other differences on Windows:
- **Names** (`--name`, `--stem`) must also be valid Windows names: no `< > " | ? *` or control characters, no dot or space
  at the end, not `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9` (also with an extension), and at most 64
  characters, so that the longest path under the data folder stays within Windows' 260 characters
  (keep `MT_HOME`, or the checkout's `data` folder, at most about 85 characters long; a job whose longest file there
  would not fit is refused before anything starts, naming that path, unless long paths are switched on). macOS and
  Linux keep their rules (only `/ \ :` are refused).
- **Devices:** Windows lists every device once per host API (MME, DirectSound, WASAPI, WDM-KS); `--device` and
  `--system-device` match the *Windows WASAPI* ones only, opened with Windows' own conversion to 16 kHz mono.
- **GPU:** `MT_DEVICE=auto` (the default) tries CUDA when an NVIDIA GPU is counted and falls back to the CPU (int8) with a
  note when the model cannot load or decode there; `MT_DEVICE=cpu` never tries it; `MT_DEVICE=cuda` makes a CUDA failure an
  error (to test the GPU). On Linux the same holds once `MT_DEVICE` is set (`MT_DEVICE=auto` switches the check and the
  fallback on); without it a counted GPU is used as before 1.11.0, with no warm-up, and a CUDA fault is an error.
- **Saving:** a file that Defender, the indexer or a sync client holds open for a moment is replaced on the next try (5 tries
  in 0.75 s); a lasting refusal, and a refused file or folder in the output folder (Documents is protected), names
  *Controlled folder access* (Windows Security › Virus & threat protection › Ransomware protection).

## Use

**Audio or video file in, two kinds of text file out.** Any file ffmpeg can read works: audio (mp3, m4a, wav, flac, ogg,
opus, aac, wma, aiff ...) and video (mp4, mov, m4v, mkv, webm, avi ...). From a video only the **audio track** is
taken; the picture is ignored and no extra audio file is saved.

```bash
python3 mt.py transcribe call.m4a                                  # -> call.transcript.txt next to the audio
python3 mt.py transcribe call.m4a --to English --to Russian        # + call.English.txt and call.Russian.txt
python3 mt.py transcribe IMG_1234.MOV --to English                 # a video: its sound is transcribed
python3 mt.py transcribe lecture.mkv --track 1                     # a video with several audio tracks: the second one
python3 mt.py transcribe data/<id>/audio.wav --into data/<id> --stem standup   # a stored recording: continue it, name the files
python3 mt.py transcribe *.mp3 --out ~/Documents/transcripts       # several files, results in one folder
python3 mt.py translate call.transcript.txt --to Spanish           # load a transcript, write call.Spanish.txt
python3 mt.py transcribe call.m4a --to English --offline           # translate on this Mac: no Claude, no API key
python3 mt.py record --name weekly --to English                    # microphone; files land in data/<id>/
```

```text
call.transcript.txt     [00:00:12] [ru] Добрый день, коллеги! ...
call.English.txt        [00:00:12] [ru] Good afternoon, colleagues! ...     (tag = language of the original line)
```

- The transcript and each translation are **separate files**; the transcript is written first, so it is kept even if a translation fails.
- `translate` also accepts any plain text: lines without the `[time] [lang]` prefix are translated as they are.
  It refuses to overwrite its own input.
- A video with several audio tracks (for example a screen recording with microphone and system sound) uses the **first**
  track and says so; pick another with `--track N` (0 = first). A video without sound is reported as an error.
- A batch goes on after a bad file (an unreadable file is reported, the exit code is 1 at the end).
- `--to` without `ANTHROPIC_API_KEY` (and without `--offline`) stops at once, before the long transcription starts.
- Existing result files with the same name are overwritten (`a.mp3` and `a.wav` in one folder share `a.transcript.txt`; use `--out` or rename).
  In the output folder's hour subfolders (`MT_OUT`, the app) only a rerun of the same input replaces its files (a hidden
  `.meeting-transcriber.json` there records which input each name belongs to); another meeting's files of the same name are kept
  and the new ones numbered (`a-2.*`). A transcript file that already sits in the output folder is translated beside itself.
  With `--out` and several files of the same name in one run (a Zoom folder per meeting, each holding `audio_only.m4a`), the second
  and later ones get their parent folder's name in front (`b-audio_only.transcript.txt`), and a line says so.
- Also: `python3 mt.py show [id] --to English` prints a stored transcript, `list`, `devices`; `--language` sets the spoken languages (below);
  `record --hours N` (default 4, at most 8, measured in recorded audio, so a sleeping computer does not use it up; a notice 10 minutes
  before the stop; `kill -USR1 <pid>` adds an hour; `kill -USR2 <pid>` pauses, and again resumes: nothing is recorded meanwhile, the
  recording and its transcript leave the pause out, and a timed note `# [hh:mm:ss] Paused here for N min S s` marks the place;
  with `MT_CONTROL=stdin` (on Windows the only way) the lines `extend`, `pause`, `resume`, `stop`, `cancel` on stdin do the same, see *Windows* above),
  `--no-transcribe` (only record; transcribe the files later), `--device <index|name>` picks the microphone (an input's index from
  `devices`, its exact name, a part of its name that fits one input only, or `NAME#2` for the second of two inputs with the same
  name; anything else stops with the list before the recording starts; the default is the system's current input). Every note a
  recording gives (a pause, a gap,
  a stop at the limit) goes into its own transcription too; `--note "[hh:mm:ss] text"` places a note at that time instead of the top.
  The name (`--name`) becomes part of the folder and file names, so it cannot contain `/` or `:`. Stored transcripts (`transcript.json`) live in `data/<id>/`, which is git-ignored.
  A data id is the time a job began and its name (`20261126-152011-weekly`); another `mt.py` job of the same name begun in the same second (two short clips both called `x.wav`, a recording and a queued job) gets `-2`, `-3` …, and one without a name `-recording-2` (since 1.11.1; before, it failed with "File exists"). The app numbers the folder of a call it records the same way (since 1.11.2; before, a call begun in the same second as an `mt.py` job of its name wrote into that job's folder).

## Permissions and updates (the app's signing identity)

macOS ties a permission grant (microphone; *Screen & System Audio Recording*; *System Audio Recording Only*) to the signed
code's **designated requirement**. An ad-hoc signature (`codesign --sign -`, what `build_app.sh` and the CI release used so far)
has no stable identity: its requirement is the build's own hash, so **after every rebuild or update the entry in System
Settings still shows as allowed but no longer matches the app**, and the permission does not count (the app then reports that
macOS sees no permission although you granted it). The fix is one self-signed code-signing certificate that every build carries:
the requirement becomes `identifier "com.github.avza-bl.meeting-transcriber" and certificate leaf = H"…"`, the same for every
build, and the grant survives. No Apple developer account is needed; the certificate matters only to this Mac's privacy database,
Gatekeeper is unchanged (right-click → Open on a downloaded app, as before).

- **Once, on your Mac:** `./app/make_signing_identity.sh` creates the certificate "Meeting Transcriber Signing" and puts it into a
  keychain of its own (`~/Library/Keychains/meeting-transcriber-signing.keychain-db`, holding nothing else, so an export can never
  carry another key of yours; macOS may ask for your login password once). From then on `build_app.sh` (and the launcher, which runs it when it
  rebuilds an app from the checkout) signs with it; the build prints the designated requirement so you can see `certificate leaf`
  in it. The first signed build is a new identity to macOS: allow the permissions one last time, and later builds keep them.
- **Releases built by GitHub Actions** stay ad-hoc signed, and lose the permissions at every update, until the two repository
  secrets `MT_SIGNING_P12` (the certificate and key as base64 of a `.p12`) and `MT_SIGNING_P12_PASSWORD` exist; the script prints
  the exact `gh secret set` commands (`./app/make_signing_identity.sh --export` prints them again). With the secrets the CI job
  signs every release with the same identity, and the job summary says so.
- The app tells you which kind of build it is: an ad-hoc build shows an orange line in *Record* ("This build is ad-hoc signed…").
- **A grant stuck on an old identity:** remove the app's entry under System Settings > Privacy & Security > Screen & System Audio
  Recording (−) and allow it again, or in Terminal: `tccutil reset ScreenCapture com.github.avza-bl.meeting-transcriber` for the
  screen capture, `tccutil reset AudioCapture com.github.avza-bl.meeting-transcriber` for the audio tap's permission,
  `tccutil reset Microphone …` for the microphone; then start the recording again.
- The other apps of this repository (launcher, mirror, legal, economy) are ad-hoc signed the same way; the same certificate
  and secret could serve them.

## Calls: your microphone and the other side

For an online call two things are recorded at once, as **two tracks**: your microphone (you) and the sound the computer plays
(the other participants). Each track is transcribed on its own, the lines are merged by time, and every line says who spoke:

```text
[00:00:00] [ru] <Them> Добрый день, коллеги! Сегодня мы обсудим бюджет на следующий квартал.
[00:00:07] [en] <You> Thank you. I have a question about the budget.
[00:00:13] [de] <Them> Ich habe die Zahlen vorbereitet und schicke sie nach dem Treffen.
```

- **Mac app:** in *Record*, tick **Also record the sound this computer plays**. **Which devices:** the *Microphone* menu picks the
  input that is recorded (your side; also for a plain recording) and *Take the computer's sound from* the output device whose sound
  is the other side (the speakers or headphones the call is played to), each by its Core Audio UID, so the choice survives relaunches
  and replugging; "the Mac's current input/output" (the default) follows System Settings > Sound, as before. A chosen device that is
  not connected when the recording starts is reported as a note and the current one is used; a chosen microphone that goes away
  mid-recording hands over to the current input with a note, a chosen output that is disconnected hands the capture over to the
  current output with a note; with a chosen output the capture does not follow a change of the default output (that is the point
  of choosing). A chosen output on which the capture device does not run (1.5 s after the start the HAL reports it not running:
  measured on one Mac for an output that was not the Mac's current one) is given up with a note and the capture is built again on
  the current output, the construction measured to work. The lists refresh by themselves when a device is plugged in or out (and on *Refresh*). A plain recording (no
  computer's sound) is made by `mt.py record`, which knows the inputs by name: the app passes the chosen microphone's name, or
  `NAME#k` for the k-th of two inputs with the same name (two identical USB microphones), k counted in Core Audio's device order,
  which is the order PortAudio lists them in too (an assumption from PortAudio's Core Audio host, not tested with twins). On macOS 14.2 or newer the
  sound is taken from a
  **Core Audio process tap** (a tap on every process but the app itself, limited to the chosen output device when one is chosen,
  read through a private aggregate device clocked by the chosen output, else the Mac's current one. Measured once, on one Mac
  (macOS 27.2, a Bluetooth headset) with `app/diagnose_tap.swift`: an aggregate clocked on the idle built-in speakers while the
  sound went to the headset, as 1.9.9 and 1.9.10 built it, started without an error, reported itself not running and never
  delivered a buffer; clocked on the current output, the headset in its 16 kHz hands-free profile, it ran. Whether a capture
  clocked on a chosen output that is not the current one runs was not measured: the app looks 1.5 s after the start and says so
  if the device did not run. The buffers come at the clock's sample rate, not at the rate the tap states, and are labelled so),
  which needs only the **System Audio Recording Only** permission: no Screen Recording, no picture, none of
  macOS 15's monthly re-approval prompts. macOS asks the first time a signed build starts the capture; there is no way for the app to
  ask whether the permission exists, so without it the track is silent, and the app says so after 20 s of a silent computer's sound
  while your microphone has sound, and again at the end. On macOS 13-14.1, or when the tap cannot be made, ScreenCaptureKit is used,
  which needs the full *Screen & System Audio Recording* permission (`MT_SYSTEM_CAPTURE=sck` in the environment, or the default
  `systemCapture` = `sck`, forces it for a comparison) and takes the sound of every output: a chosen output device does not apply
  to it, and a note says so. The capture is built anew (tap and aggregate device both, the one remedy reported for a tap that
  runs but delivers zeros) when the clock device's sample rate changes, once when the track has carried only silence for 20 s
  while your microphone had sound, and, with no chosen output, when the current output changes; each time a note says so, and
  the listeners move to the new clock device. 1.5 s after each start the app looks whether the capture device reports itself
  running and, if not, says on which output it did not start. The microphone permission is asked for as well.
  Allow the app in System Settings > Privacy & Security, then reopen it. A build without the project's signing identity is a
  new identity to macOS at every rebuild or update: macOS keeps the old entry listed as allowed but does not honour it (see
  *Permissions and updates* above for the lasting fix; remove the old entry (−) and allow the app again meanwhile). When the
  computer's sound cannot be captured (no permission, no display, a ScreenCaptureKit error) the call still
  goes on with the microphone alone, and the reason is a warning, not an error; a call whose computer's-sound track stayed silent
  for its whole length (nothing was playing, or the capture delivered nothing) gets a warning too, as does a track in which no
  speech was found at transcription time. You can rename the two sides (default `You` and `Them`).
- **The box follows where the sound goes** (since 1.10.5). While the Mac plays through a loudspeaker, the microphone hears both
  sides of a call, and a chosen microphone, recorded without echo cancellation, takes them down in its own words, which were
  the better ones in a call with the built-in microphone and a Bluetooth speaker. Two tracks would then hold the other side
  twice, so the app unticks **Also record the sound this computer plays** by itself and records the microphone alone, with a
  line under the box that names the output. When the sound goes to an output that is not a speaker (headphones, or a device
  you have not unticked it with: AirPods are as unknown to macOS as a Bluetooth speaker), it ticks the box again, but only if it
  unticked it itself: a box you unticked stays off. The output it looks at is the one chosen under *Take the computer's sound
  from*, else the Mac's current output; that menu stays shown while the box is unticked by itself, so a call that plays to
  other headphones can be pointed there without a tick. It looks again when the current output changes, a device is plugged
  in or out, the jack of an Intel Mac changes between speakers and headphones, and a recording ends; never during a recording,
  whose tracks were fixed at Start (also not in a window opened again while one runs).
  - **What macOS recognises as a speaker:** the built-in speakers (the data source `'ispk'` or `'espk'` of an Intel Mac's
    built-in output, IOKit's `IOAudioTypes.h`; the device with UID `BuiltInSpeakerDevice` on Apple-silicon and T2 Macs, which
    give the speakers and the jack devices of their own, a UID seen in device listings of such Macs and not documented by
    Apple), and outputs over HDMI, DisplayPort and AirPlay (`kAudioDeviceTransportType…`, `AudioHardwareBase.h`).
  - **Never a speaker:** the headphone jack (`'hdpn'`, or the UID `BuiltInHeadphoneOutputDevice`; desk speakers in the jack
    look the same, so the box stays ticked) and any output whose stream says it is headphones (terminal type `'hdph'`, or the
    USB headphones and headset types 0x0302 and 0x0402).
  - **Learned from you:** Core Audio gives a device's connection, not its form, so Bluetooth, USB, virtual and aggregate outputs
    are unknown and never untick the box by themselves. Untick the box once while the sound plays through such a speaker (a
    Bluetooth speaker such as the TG659) and the app remembers it by its UID; from then on the box is unticked by itself
    whenever that speaker is the output. Ticking the box while an output is current keeps two tracks with that output from then
    on, also for one macOS calls a speaker (a monitor whose own headphone jack is used). A choice is kept by the output's UID
    and, when it has one, its selected data source: an Intel Mac's speakers and jack are one device with one UID. An untick
    made while macOS says headphones is never taken as a speaker. Reading the Bluetooth device class would need the Bluetooth
    permission, and speakers often call themselves headsets, so the app does not.
  - **Not tested on a Mac yet:** which data source and UID each Mac reports, and that the listeners fire for every change
    (macOS 15 to 27); with "Every output" a call app that plays to an output other than the Mac's current one is not seen.
- **Command line:** `python3 mt.py record --system-device N` records a second input as the other side (`python3 mt.py devices`
  lists the inputs, with `>` and `<` at the system's current input and output; `--device` and `--system-device` take an index, an
  exact name or a part of a name that fits one input only, and stop with the list otherwise; Linux needs `libportaudio2` for
  `record` and `devices`): a *Monitor of …* source on Linux (PulseAudio or PipeWire;
  if PortAudio does not list it, start with `PULSE_SOURCE=<sink>.monitor python3 mt.py record --system-device pulse`), BlackHole on
  macOS (make a Multi-Output device of BlackHole + your speakers in Audio MIDI Setup, play the call to it, record BlackHole), *Stereo
  Mix* or a virtual cable on Windows. `python3 mt.py call mic.wav system.wav` transcribes two finished recordings; `--me` and `--them`
  name the sides.
- **If something goes wrong during a call**: when the input device changes (headphones plugged in or out, a headset connected) the
  app restarts the microphone on the new device and fills the gap with silence, so both tracks stay on one timeline. When the
  capture of the computer's sound dies, it is started once more; if that fails your microphone goes on alone (as it does when
  the capture could not start at all). A track that delivers
  nothing for 10 s is reported (and the microphone restarted). A microphone that delivers buffers but only zeros for 20 s is
  reported too (the stall check cannot see that; the level bar can: it stays flat), and when echo cancellation was on the
  microphone is started once more without it, the one remedy the recording has by itself (added in 1.9.12 after a call in
  which the transcript found no speech on the microphone; a headset whose noise suppression sends pure zeros while you are
  quiet gets the same treatment, once, and the note says so). A microphone that stayed silent for the whole call is reported at
  the end with its counters (how many buffers, the loudest sample), like the computer's sound. Every channel of the
  microphone is averaged into its track, however many there are: until 1.10.1 a buffer with more than two channels was
  dropped (Apple's `AVAudioFormat(commonFormat:sampleRate:channels:interleaved:)` "fails (returns nil)" for more than 2
  channels, `AVAudioFormat.h`), and with echo cancellation on, macOS 27.2 hands the microphone over in 3 channels (seen with a
  Bluetooth headset and a Bluetooth speaker), so the track came out as zeros of the right length. A microphone with echo
  cancellation on that has not delivered one buffer for 10 s since it was (re)started is started again without it. Every such event is a **note**: it
  is shown in the progress box and the final alert, and written at the top of the transcript (`# [00:00:00] …`); every restart of
  the microphone, with its reason and the input format afterwards, is in `capture.log`. The recording ends by itself only when
  neither side can be recorded any more. A level bar next to the timer moves while you speak. What Apple's audio engine *raises* (an
  Objective-C exception, which Swift cannot catch and which ended the app in 1.9.6 and 1.9.7 at the microphone tap) is caught in
  a small Objective-C file (`app/Sources/App/ObjCCatch.m`) and reported with the engine's own reason: the recording fails or goes
  on without that side, the app stays up. The microphone is set up and restarted on one queue, so two threads never touch the
  engine at once. Should an exception still end the app, its name, reason and stack are written to
  `~/Library/Logs/Meeting Transcriber/uncaught-exception.txt` (macOS's own report leaves the reason out): send that file.
  Every call recording also writes **`capture.log`** into its folder, line by line as the call goes (since 1.10.1; before, only
  at Stop, so a Stop that never finished left none): the build's code signature (stable identity or ad-hoc),
  the devices, how the tap and its aggregate device were built (format, clock device), when the microphone's first buffer came,
  and every 10 s how many buffers each track received and their loudest sample (for a track with several channels the loudest
  sample of each, and any buffers that could not be written), whether the microphone's audio engine runs, plus the tap's own counters (callbacks seen, buffers handed on, unexpected
  layouts, and whether the capture device reports itself running; the setup line gives the rate the tap states against the
  rate the buffers come at, and a line 1.5 s after each start says whether the device ran). A silent computer's-sound track
  is then explained from that file: no callbacks with "capture device NOT running" means the HAL never drove it (the 1.9.9 and
  1.9.10 case), callbacks with an unexpected layout mean the buffers were not recognised, callbacks full of zeros mean macOS
  withheld the System Audio Recording permission (it never says so). The note at the end of such a call quotes the counters.
  When the counters say *no callbacks* (with 1.9.9 and 1.9.10 on one Mac, macOS 27.2, whenever the sound went to a Bluetooth
  headset: the tool showed the same construction not running and the current-output clock running, so 1.9.11 clocks on the
  current output; a real call with 1.9.11 on that Mac then delivered the other side, 77 callbacks in the first 1.5 s, see *Tested*
  below), the stand-alone tool
  `app/diagnose_tap.swift` finds the construction that works on that Mac: it builds the tap and its aggregate device in
  twelve ways (clock on the built-in or the current output, tap limited to one device or not, automatic start or not, block or
  function callback, a pause before the start, drift compensation off, and four times with the microphone open in the app's
  two ways, plain input and with echo cancellation, beside the 1.9.10 construction and beside the 1.9.11 one), runs each for
  6 s while something plays, and prints the device's input streams, running state, callbacks and the loudest sample of the
  tap's own buffers; for the four microphone variants also the microphone's buffers, its loudest sample and the number of
  configuration changes the audio engine reported, before the aggregate was built, 1.5 s after and at the end. Run it from
  Terminal: `cd meeting-transcriber/app && swiftc -O diagnose_tap.swift -o /tmp/diagnose_tap && /tmp/diagnose_tap` (the first
  run asks for the System Audio Recording permission for Terminal; the tool then waits for Enter, so start the music or the
  call before pressing it; the microphone is asked for at the last four variants, speak during those; needs the Command Line
  Tools), and send the output. A variant with callbacks and a tap peak above 0.01 is the one the app should use; callbacks
  with a tap peak of 0 mean the data flows but macOS withholds the sound; no callbacks mean the capture device never ran. A
  microphone whose buffers stop growing, or whose peak stays at 0 while you spoke, once the aggregate exists is a microphone
  that the capture of the computer's sound silenced; a configuration change is what the app answers with "The microphone changed".
- **Echo:** the microphone's echo cancellation (Apple's voice processing) keeps what the Mac plays off your track where macOS
  allows it, so a call on the speakers no longer puts the other side on your side; headphones are cleaner still. It works with
  the Mac's current input only: with a microphone chosen in the *Microphone* menu it is off, and a note says so (pointing
  Apple's voice processor at another device made the engine abort the app in 1.9.6). A chosen microphone is recorded through
  its own input unit (`app/Sources/App/DeviceMicrophone.swift`: a HAL unit with input on and output off, in the order of
  Apple's TN2091), not through the audio engine: the engine's input and output are one unit bound to the Mac's default
  devices, and pointed at another input it gave no buffer at all (a 1.10.2 call on macOS 27.2, the built-in microphone chosen
  and a Bluetooth speaker as output: 0 buffers in 82 s; others saw the same on macOS 26.6 and 27.0.1, llfbandit/record PRs 630
  and 634). If the chosen device stops delivering and cannot be opened again, the Mac's current input takes over. Whatever still
  comes through is trimmed from your lines: the words a line shares with a line of the other side spoken at the same moment are
  removed, and only a line that is nothing but echo is dropped (`--keep-echo` keeps everything). Apple's voice processing
  also lowers ("ducks") what other apps play while it runs (an Apple engineer: "its volume will be significantly reduced.
  This is intentional", developer forums thread 829753): the app asks for the smallest ducking, and since 1.10.2 again after
  every restart of the microphone (a configuration change 0.7 s into a call restarts it; in a 1.10.1 call the sound the Mac
  played could not be heard after that), and `capture.log` says which ducking the voice processor reports. If the other side
  is still too quiet to hear, choose the microphone by name in the *Microphone* menu: echo cancellation is then off, and
  nothing is lowered.
- **Sleep:** while a job runs the Mac does not go to sleep by itself (the command line uses `caffeinate` the same way); a closed
  lid still sleeps it, and the time asleep is reported as a note. A plain recording fills such a gap with silence so the
  timestamps stay right.
- **Cancel** while recording keeps the two audio files in the data folder and transcribes nothing; Stop transcribes them.
- **Transcribe afterwards** (the app) or `--no-transcribe` (the command line) only records: the folder is shown as the result and the
  files can be transcribed later (`transcribe <folder>/audio.wav --into <folder>`, or `call <folder>/mic.wav <folder>/system.wav
  --into <folder>`, so they are screened as a recording). A recording stops by itself after the chosen hours of recorded audio (4 by
  default, up to 8), with a notice 10 minutes before; **Extend by 1 h** (or `kill -USR1 <pid>` for the command line) moves the limit.
- **Pause** / **Resume** (or `kill -USR2 <pid>` for the command line): nothing is recorded while paused, the timer and the limit stand
  still, the recording and its transcript leave the pause out, and a timed note says where it was and for how long (`# [00:12:30]
  Paused here for 3 min 5 s`). Stop while paused ends the recording there.
- **Settings while a recording runs:** the languages, granularity, who spoke, translation, the names of the two sides, the output folder
  and "Transcribe afterwards" can still be changed — they are read when the recording stops and apply to its transcription (changing the
  languages or the granularity of a live recording makes the final pass redo what was transcribed live, and says so). The name, the
  length, "Transcribe while recording", the languages translated live (the ones ticked at the start), the second track and the two
  devices are locked, because the recording was started with them. The engine
  setup ("Set Up Speech Engine Again…") waits until the recording and the queued jobs are done: it rebuilds the environment every
  running process uses.
- **Translations** keep the labels (`<Them>` stays on the translated line). Languages and granularity apply to both tracks. With
  languages chosen under *Also translate to*, the follower translates every line live as well (`follow --to`), so the other side's
  words appear translated beside the transcript while the call goes on.
- **Every output:** with *Take the computer's sound from* left at "Every output" (the default), the tap takes the sound of every app to
  every output device (Apple's `CATapDescription.deviceUID` "will have a value if this tap only taps a specific hardware device";
  the app sets none), both channels mixed to one; a chosen device limits the track to that device. The microphone track takes every
  channel of the input, mixed to one. `capture.log` says which. Not measured on a Mac: whether a global tap clocked on the current
  output delivers what goes to a *second* output device at the same time (a Multi-Output setup); the 1.9.11 measurement covered
  the current output only.
- **One Bluetooth device as microphone and output** (a headset, a speakerphone): recording from its microphone switches it to
  its call mode, and what it plays sounds worse until the recording ends (Apple: the quality stays reduced "until the
  microphone is no longer in use", support.apple.com/102217); a note says so when a call starts. For full sound make the Mac's
  built-in microphone the input (System Settings > Sound > Input) and keep the Bluetooth device as the output. Do not force a
  music rate (44.1 or 48 kHz) on the device in Audio MIDI Setup during a call: on one Mac (macOS 27.2, a Bluetooth speaker) that
  silenced it, also after the recording, until its format was set back to 16 kHz (Mozilla bug 1881609 reports the same with
  AirPods). The capture of the computer's sound waits for nothing it cannot finish: its Core Audio work runs on its own queues,
  and Stop waits for it at most 3 s (in 1.10.0, after such a rate change, a rebuild of the capture never returned, the
  computer's-sound track stopped and Stop never finished; the audio files were kept). A WAV whose header counts less than the
  file holds (a recording whose Stop never finished: the headers are patched every 30 s and at Stop) is read to its end:
  `mt.py` passes ffmpeg `-ignore_length` for such a file, never for a WAV whose extra bytes are a metadata chunk.
- **Tested here** on a synthetic call (English on the microphone, Russian and German on the other side, with a quieter delayed echo
  of the other side on the microphone): the merged transcript had the right speaker and language on every line, and the three echoed
  lines were dropped. **Tested on one Mac** (macOS 27.2, a Bluetooth headset as input and output): 1.9.6 and 1.9.7 ended in an
  exception from the audio engine at the microphone tap, which 1.9.8 catches and reports instead of dying; 1.9.8 to 1.9.10
  recorded the microphone but the computer's-sound track stayed silent, 0 callbacks (see capture.log above); 1.9.11 recorded
  the other side (capture device running, 77 callbacks in 1.5 s, a `<Them>` line transcribed and translated), but that call's
  transcript found no speech on the microphone, and whether the microphone delivered sound was not measured (1.9.12 reports it:
  the early note, the note at the end and `capture.log`; `diagnose_tap` J1/J2 measure the microphone beside the 1.9.11
  construction). With a Bluetooth speaker as input and output, 1.9.11 logged the microphone's input as "16000.0 Hz 3 ch" with
  echo cancellation on and 0 buffers for 446 s, and a 1.10.0 call wrote a microphone track of zeros for its whole 223 s: the
  3-channel buffers were dropped by the app itself (fixed in 1.10.1, see *Calls* above), whether the device's microphone
  delivered sound is not yet measured. **Not tested:** the output device chosen in the app (the tap limited to one output), a chosen microphone
  through its own input unit (1.10.3; the engine's way gave 0 buffers), a change of output during a call, the `--system-device` recording with real devices, and calls on
  Windows or Linux.
- **Consent:** recording the other side of a call requires their consent in many places.

## Who spoke: several people on one microphone (`--speakers`)

An interview or a meeting in one room has several people on one microphone. `--speakers auto` (the app: **Several people on one
microphone** under *Who spoke*) labels every line with who spoke: `<Speaker 1>`, `<Speaker 2>` … in order of appearance. In a call
it applies to both sides: the other side becomes `<Them 1>`, `<Them 2>` …, and several people at your microphone (a room on your
side of the call) become `<You 1>`, `<You 2>` … (since 1.10.2); one voice found at your microphone stays `<You>`. A number
(`--speakers 3`) counts the other side; your side's number is always found by itself. Give the number only when the
result is wrong: finding it out did better in the test. Two people talking at once for half a second or more become a note
(`# [00:04:00] overlap 00:04:00–00:04:05 Speaker 1 / Speaker 2`). Afterwards give them their real names:

```bash
python3 mt.py transcribe interview.m4a --language en,ru --speakers auto
python3 mt.py rename <id or folder> "Speaker 1=Anna" "Speaker 2=Boris"     # the app: Name Speakers… under the transcript
```

The names are kept in `transcript.json` (every later file uses them) and put into the text files already in the folder.

How it works: the whole recording is first separated into speaker turns (sherpa-onnx, on the CPU, about 0.4× the audio length on
one core), a chunk for Whisper never spans a speaker change (turns shorter than 1 s join their neighbour), and each line gets the
speaker whose turns overlap it most. With **Transcribe while recording** the lines are labelled at the end, when the whole recording is
known: the stretches transcribed live were cut without knowing the speakers, so the final pass cuts those that span a change of
speaker at the turn and decodes the pieces again (a note says how many; before 1.9.12 such a stretch kept one label for both
people), and reuses the rest. The number of speakers is found by clustering the voices (`CLUSTER_THRESHOLD` 0.9 in `mt.py`).

Measured on the 10-minute six-voice English/Russian/German test (synthetic voices, a hard case: two voices per language, alike),
4-core Xeon:

| | speakers found | sentences labelled right | speaker changes caught | DER |
|---|---|---|---|---|
| `--speakers auto` | 6 of 6 | 96 % | 78 of 83 | 9.4 % |
| `--speakers 6` | 6 | 81 % | 78 of 83 | 25 % |
| a 3 s overlap of two voices | both reported, 0.02 s of false overlap elsewhere | | | |

**pyannote** (`--diarizer pyannote`; the app: *Use pyannote instead*): the optional, better-known backend. Its packages are
about 2 GB (`requirements-pyannote.txt`, Python 3.10 or newer; the app installs them at setup while the box is ticked), its models
are gated on huggingface.co (accept the conditions of `pyannote/speaker-diarization-community-1` once and give a Hugging Face token
with *Set Token…* or `HF_TOKEN`), and on a CPU it is much slower than sherpa. Not measured here.

## Languages spoken in the recording (`--language`)

**By default the language is recognised automatically**: each stretch of speech (a chunk between pauses) is detected among
all ~100 Whisper languages from the same pass that decodes it, so detection costs nothing extra, and a chunk that sounds mixed
is cut finer (see *Granularity*, `adaptive`) so that a person who switches language part-way gets a tag per part. The speaker
label and the language tag are decided independently (the speaker from the voice, the language per stretch of speech), so one
speaker can switch languages, and with `--speakers` a chunk never spans a speaker change. Giving the languages is optional:
with several, each chunk is detected **among those only**; with one, it is forced:

```bash
python3 mt.py transcribe meeting.m4a --language ru,en,de          # a Russian, English and German meeting
python3 mt.py transcribe call.m4a --language "Russian, English"   # names work too, in any case
python3 mt.py transcribe talk.m4a --language ru                   # one language: forced for the whole file
python3 mt.py transcribe call.m4a                                 # nothing given: any of the ~100 Whisper languages
```

In the Mac app it is the **Languages spoken in the audio (optional)** drop-down: leave it at "Any" for automatic recognition, or
choose the languages that occur (all of Whisper's, with a search field) to restrict detection to them.

- **Several languages:** Whisper detects the language of each chunk as before, and a chosen language wins unless Whisper is sure
  (70 % or more) that the chunk is in a language you did **not** list: then that language is kept and the line is tagged with a `?`,
  e.g. `[de?]`, so the words stay German instead of being turned by Whisper into a listed language; the `?` is your cue to add the
  language. When Whisper is unsure, the likeliest chosen language is used, and below 50 % the runner-up is decoded as well and the
  better fit kept. Tested on a synthetic English, Russian and German recording: all three listed gave all three tags; with German
  left out, its line became `[de?]` in German.
- **One language:** forced for the whole recording, whatever is really spoken (a recording that switches languages comes out
  wrong). **None (the default):** all languages, detected per chunk; below 50 % Whisper's own second choice is decoded as well and
  the better fit kept, and the line is tagged `?`.
- **Cost:** every chunk is encoded once (the language probabilities and the decoding share that pass); only a chunk Whisper is
  unsure about (below 50 %) is decoded a second time, in the runner-up language.
- Names and codes are checked before anything starts. Whisper has no Zulu (it can be a translation language, not a spoken one).

## While it works, and what the transcript tells you

The app shows the transcript **line by line as it is decoded** (with a search field, a Follow toggle and Copy), and above it the
position in the audio, the speed (× realtime), the estimated time left, the number of lines and the share of each language so far;
a translation shows the target being worked on and its batches.

With **Transcribe while recording** (on by default; `record --live` on the command line) the transcript fills in during the
meeting: every stretch of speech that a pause follows is decoded as soon as it is closed and written to the recording's checkpoint,
so when the recording stops only the last stretch is left to do before the translation. A stretch is closed once a pause of the
granularity's length follows it and it is at least the granularity's chunk minimum long (5 s with no language ticked or one, 1 s
with several); a shorter one is held until the next stretch joins it or, since 1.9.12, until 3 s of silence have followed it, so a
short answer appears a few seconds after it was said rather than when that side next speaks. The speech model is loaded when the
recording starts (the first line waits for it: some seconds, longer the first time); until the first line the Recording box says
what it waits for (the model, then a closed stretch of speech), and the window scrolls to the box when the recording starts and
when the first line arrives (the box sits under the settings, below the fold at the window's default size). A speech model that
cannot be loaded ends the live transcription with a note; the recording goes on and everything is transcribed at the end. If the
computer is slower than the meeting, the header says how far behind the live transcript is; the result at the end is complete either
way. Before 1.9.12 a plain recording (no computer's sound) showed no live line, no level and no note in the app at all: `mt.py`
redrew its minute counter on the same pipe without a line end, and every event behind such a fragment was read as text. For a call the app runs `mt.py follow mic.wav system.wav --into <folder>` beside its own capture,
and the final `call --into <folder>` continues from what it did. Lines starting with `# ` are **notes**, not speech, and are kept in
every translation file: `# [00:12:03] speech not recognised for 24 s` marks a stretch the speech model gave up on (quiet, far away,
noisy, or a language it does not know), and a note at the end says how much of the speech found was decoded. A `[de?]` tag marks a
guess: a language outside your list that Whisper was sure of, or a low confidence.

**The live transcript never waits for the translation** (since 1.13.0). Up to 1.12.0 one thread decoded a stretch and then
translated its every line into every target before it decoded the next one, and it loaded the translation model at the first
line: with Hy-MT3 and two targets the transcript froze for the load and then fell further behind the longer the meeting went
(in the lag driver, at 6 s a line and target, 256 s behind at the median after ten minutes). Now the one thread that uses the GPU
decodes a closed stretch whenever there is one and translates in between, **one token at a time**, so a line appears at most
about a second (one look for closed stretches, plus one token) later than it would without translation, however long the
translation takes. The translation model starts loading when the recording starts, in slices of 256 MB between two stretches;
its download and the check of its files run beside the work. When the Mac cannot translate as fast as people speak, a line
whose translation is more than **30 s behind** is not translated live: it is marked *at the end* in the window and translated by
the final pass with the same model, so the files are the same either way. The live translator also leaves a fifth of each
minute free (it starts no new line while it worked more than 48 s of the last 60). When the recording stops, the translation
stops at once (what is left goes to the final pass) and the live transcript is finished for at most 90 s, so the stop never
waits for a translation; what is not finished then is transcribed by the final pass. *Live: x of y* now counts the audio
that is **decoded**, not the audio cut into stretches, so it shows the real backlog. The program's own messages about this go
to the log (*Details*) and the telemetry, never into the transcript's notes.

**What Whisper makes up on noise is left out** (since 1.10.3). Whisper learned from subtitled video, so on silence, music or
noise it writes the credits of those subtitles: "Субтитры сделал DimaTorzok", "Редактор субтитров А.Синецкая Корректор
А.Егорова", "Subtitles by the Amara.org community", "Untertitel im Auftrag des ZDF" (openai/whisper discussions 928 and 1873;
Barański et al. 2025). Its own no-speech score does not catch them (about 0 on large-v3-turbo, SYSTRAN/faster-whisper#1128).
`mt.py` drops a line that is nothing but such a credit or a sound caption ("[Music]", "♪♪", "♪ Musik ♪"), in any language,
and cuts one off the start or end of a real line; a meeting's outro left in front of it is screened again on its own
("Спасибо за внимание. Субтитры сделал DimaTorzok" at speaking speed keeps "Спасибо за внимание."). A German broadcaster
named after "Untertitel" or "Untertitelung" with "des" or "der" or a year ("Untertitel der ARD.", "Untertitelung des ZDF,
2020") is such a credit in any file, so a real "Untertitel der ARD." is lost too; "Untertitel für ARD und ZDF." and
"Untertitel vom ZDF." stay. A question is never a sound caption, a broadcaster's credit or a video's phrase ("Music
playing?", "SWR 2021?", "Untertitel des ZDF?", "Thanks for watching?", "Субтитры сделал Иван?"), and sung words between ♪
stay ("♪ Happy birthday ♪"; only one word, or two with a music word, is a caption). Since 1.11.4, in a recording or a call (and in the app's queued transcription
of one, `transcribe <folder>/audio.wav --into <folder>`), the phrases of a published video are dropped by their text too:
"Thanks for watching", "Thank you for watching", "Danke fürs Zuschauen", "Продолжение следует", "Fortsetzung folgt",
"Подпишитесь на мой канал", "Like and subscribe", "Ставьте лайки", "See you in the next video", a broadcaster's tag ("SWR
2021", "ZDF, 2021"), and a subtitle credit naming any person ("Субтитры сделал Иван.") or thanking for the subtitles
("Спасибо за субтитры"); one at the end of a real line is cut off ("Хорошо, договорились. Продолжение следует..." keeps
"Хорошо, договорились."). So a real "Thank you for watching." at the end of a demo, a real "Продолжение следует." at the
end of a topic, a real "Голосуем в чате. Ставьте лайки." (kept as "Голосуем в чате."), a real "Welcher Sender? ZDF, 2021."
(kept as "Welcher Sender?"), or a video played through the computer's sound loses those lines or sentences; this is
accepted to keep them out of the live window. A channel plug counts as a video's only for "my channel"
("Abonnez-vous à ma chaîne"); a bare one, or "our" or "the" channel, is a work channel or a button too. In an imported file,
which may be such a video, they are dropped only with a sign that nobody said them; there that includes a decode Whisper
itself rejected (still below its own thresholds, average log-probability -1.0 or compression ratio 2.4, after its
temperature fallback), the reason "a video's phrase Whisper itself rejected". What a meeting says as well ("Спасибо за
внимание", "Bis zum nächsten Mal", "See you tomorrow", "До новых встреч", "Subscribe to the channel", "Captions by Friday") is
dropped only with such a sign, and a rejected decode is none (mumbled speech is still said): the line stands alone in its
stretch of audio spread over five seconds or more of near-silence, or Whisper timed it five seconds or more past the end of the
audio (into the silence it pads a chunk with; real words of a short chunk end up to about 1.6 s past it). An outro sentence
at the end of a real line timed that far past its audio is cut off the same way. Video phrases and the outros of 1.11.3 are
also dropped when they come three times or more on one track, now while recording as well: the lines already shown are
then taken out of the window again, and the final pass does not show them again. Goodbyes several people say ("See you
tomorrow.", "Bis zum nächsten Mal.") are never counted that way. Short thanks ("Спасибо.", "Thank you.", "Vielen Dank.",
"Дякую.") are never dropped by their text, only alone and timed five seconds or more past their audio. Such lines are never
shown, translated or written, and one note at the end of the track says how many were left out. What still gets through:
fluent or garbled sentences Whisper writes on wordless voice, song or murmur ("I don't understand.", "Gut, gut, gut, gut."),
and a short thanks or goodbye at speaking speed; no measured sign tells them apart from real speech. The patterns (139,
collected from those reports and from other projects' filters) are in `mt.py` above `segment_lines`, and the tests check them
against real sentences that must stay.

**Live translation, side by side.** With languages chosen under *Also translate to* (`record --live --to Russian`, `follow --to
Russian`, `--offline` for the local model), every line is translated as soon as it is shown: the line appears first, its
translation a moment later beside it. The transcript pane shows the text and, beside it, every translation, one column each, under
a row with the column names (since 1.11.0; before, one translation at a time). Its drop-down picks which translations are shown
(all until some are picked; *Clear* leaves the text alone), and *Copy* puts the shown columns on the clipboard tab-separated, so
they paste into a spreadsheet as shown. Several languages are translated one after the other. The pane is there from the start
of a live recording, before the first line; *Follow* (on by default) keeps the end in view
as lines come, change and get their translations. Offline, the translation model (1.4 GB for NLLB, about 4.5 GB for Hy-MT3, 7 GB
for TranslateGemma) is loaded beside the speech model, so a live recording with offline translation holds both (about 3 GB with
NLLB, about 6 GB with Hy-MT3, an estimate, 8.5 GB with TranslateGemma; with Claude only the speech model). A translation that fails
(no network, a model that will not load) ends the live translation with a note and the recording goes on; the files are
translated at the end as before. The live translations are kept with the lines in the checkpoint and the stored meeting, and the
final pass reuses them instead of translating those lines again (only when they were made the same way: offline live and Claude
at the end are not mixed); the lines of the tail, decoded after the stop, are translated with the files, and the window gets every
line again with its translation when a file is written. Not measured: how far the offline translation lags behind on a Mac
(each line is a sentence-by-sentence pass of NLLB on the CPU, beside Whisper).

## If a run is cut short

Nothing done is lost. While a file is transcribed, every finished chunk is written down in its data folder (`partial.json` and
`partial.jsonl` next to the audio); a run that is killed, crashes or loses power is continued from there when it is started again
with the same file and settings: `python3 mt.py transcribe talk.m4a --into data/<id>` (a data id or a folder; the app's
"Unfinished recordings" list does the same). Another model, language list or granularity starts over, and says so. A chunk the
speech model chokes on becomes a `# [hh:mm:ss] chunk failed (…)` note and the rest goes on. `transcript.json` is written whole or
not at all. An offline translation writes every finished sentence down too (`translations/` in the data folder) and reuses them on
a rerun; each finished translation file is announced and saved as soon as it exists, so a failure in the third language never hides
the first two. In the app, **Cancel** asks first: during a recording the audio stays in the data folder and nothing is transcribed
(the command line gets the same from `kill -TERM <pid>`; Ctrl+C or Stop still transcribes); during a transcription what is done
stays and the next run continues from it.

## Granularity: how finely the language is followed (`--granularity`)

The language is decided once per **chunk** of speech, so how speech is cut into chunks sets how quickly a language change is noticed.
A chunk ends at a pause of at least a given length once it is at least a given number of seconds long (never above 30 s):

| `--granularity` | Pause that ends a chunk | Chunk at least | Use it for |
|---|---|---|---|
| `coarse` | 1 s | 15 s | one language, or long monologues: fewest chunks, fastest |
| `normal` | 0.7 s | 5 s | the earlier default |
| `fine` | 0.35 s | 2 s | meetings that switch language between speakers |
| `finest` | 0.2 s | 1 s | rapid turn-taking, slowest |
| `adaptive` | 0.7 s, then 0.2 s inside a chunk that sounds mixed | 5 s, pieces of 1 s | automatic language recognition: finer cuts only where a switch is likely |
| `auto` (default) | | | `finest` when several `--language` are given, `normal` with one (it is forced), else `adaptive` |

```bash
python3 mt.py transcribe interview.m4a --language en,ru --granularity finest
```

Tested on synthetic English, Russian and German speech with the language changing at every sentence (18 sentences, `--language ru,en,de`, the `small` model):

| Gaps between speakers | `coarse` / `normal` | `fine` | `finest` |
|---|---|---|---|
| 0.5 s (42 s of audio) | 2 chunks, **every line tagged `de`** | 17 chunks, every line right, one 1.5 s sentence lost | not run |
| 0.25 s (38 s of audio) | 2 chunks, mostly `ru` and `de` | 4 chunks, still wrong | 20 chunks, **every line right** (a sentence with a comma split in two) |

Measured again on a 10-minute synthetic meeting (6 voices, English/Russian/German, 167 sentences, gaps between turns 0.3-1 s, 14 % of
them under 0.4 s, 10 sentences switching language in the middle), `--language en,ru,de`, `large-v3-turbo`, on a 4-core 2.1 GHz Xeon:

| `--granularity` | Chunks | Sentences lost | Word error rate | Lines tagged right | Time (realtime factor) |
|---|---|---|---|---|---|
| `normal` | 35 | 68 of 167 (English nearly absent: it sits in Russian/German-majority chunks) | 50 % | 92 % | 0.33 |
| `fine` | 83 | 18 | 21 % | 96 % | 0.68 |
| `finest` | 121 | 4 (each right before a mid-sentence switch) | 11 % | 100 % | 0.99 |

A 2-minute exchange with every gap exactly 0.25 s: `fine` and `normal` give 5 chunks, every line tagged `de`; `finest` gives 29 chunks,
every tag right (word error rate 8 %). That is why `auto` means `finest` as soon as several languages are given.

- **`adaptive` (what `auto` means when no language is given):** chunks are cut like `normal`; when the language probabilities of a
  chunk look mixed (the best language below 99 %, or the second above 3 %), the chunk is cut again at `finest` pauses and every piece
  is detected and decoded on its own; a chunk that is sure, or has no pause to cut at, is decoded as it is. The extra cost is one
  Whisper pass per piece, only for the chunks that looked mixed (plus a VAD pass over the chunk, well under a second).
  Measured on the same 10-minute dialogue in automatic mode (no `--language`) and on an English-only twin of it (same voices, turns
  and gaps; 185 sentences):

  | Granularity, no `--language` | Chunks (split) | Sentences lost | WER | Tags right | Realtime factor, dialogue | Realtime factor, English only |
  |---|---|---|---|---|---|---|
  | `normal` | 35 (0) | 68 of 167 | 50 % | 92 % | 0.33 | 0.35 |
  | `adaptive` (99 % / 3 %) | 35 (22 split into 86 pieces) | 11 | 15 % | 100 % | 0.96 | 0.35 (1 chunk split) |
  | `fine` | 83 | 17 | 21 % | 96 % | 0.66 | 0.56 |
  | `finest` | 121 | 4 | 11 % | 100 % | 0.95 | — |

  The thresholds have to be that tight because Whisper's chunk-level probabilities understate a minority language badly (median
  94 % / 4.6 % on chunks that really hold two languages): at 80 % / 15 % the trigger caught a quarter of the mixed chunks and lost 48
  sentences. At 99 % / 3 % it catches 79 % of them; 5 chunks with a dozen English words inside a Russian or German turn stay invisible
  to any threshold, and a switch in the middle of a sentence with no pause is still not cut (0 of 10, as at every setting). On a
  single-language meeting the rule costs almost nothing (one false alarm in 35 chunks, 1.02× `normal`); on a meeting that mixes
  languages in every chunk it costs about what `finest` costs. **Uncertainty:** one run per setting (about ±5 % on the times), the
  thresholds were chosen on the dialogue's own probabilities, and the false-alarm rate was checked only on clean synthetic English
  (top probability 97-100 %): real recordings with noise or accents will trip the 99 % threshold more often, which costs time
  (at most about `finest`'s), not text. `--granularity normal` or `fine` is always available when time matters more.

- **Why `normal` fails there:** a pause shorter than its threshold never splits the speech, so the whole exchange is one 30 s window
  with one language. The chunk boundaries alone force 40 % of the words into the wrong tag at `normal`, 12 % at `fine`, 4 % at `finest`:
  the detector is not the problem, the chunking is.
- **Cost:** every chunk is a full Whisper pass (it is padded to 30 s), so the time grows with the number of chunks, not with their length:
  with the single-pass engine about 4.7-5.1 s per chunk for `large-v3-turbo` (int8, 4 threads) on that CPU, 10 s before the first one
  (audio decode, VAD, model load). `finest` makes about 12 chunks per minute of dense dialogue, so a 3-hour meeting is roughly 2,000 chunks.
  The ETA in the window is chunks left × the median chunk time so far (after 5 chunks; before the first one, the time remembered from the
  last run on this computer). An Apple-silicon Mac with the GPU engine should be several times faster (not measured here).
- **Mid-sentence switches** are not caught at any setting: the VAD never splits inside the sentence, so the part in the other
  language is dropped (0 of 10 both parts right at `normal` and `fine`, 1 of 10 at `finest`).
- **Limits:** a very short chunk gives the detector little to judge from (a language list helps), and a language change **inside**
  a chunk, or inside a sentence, is still not seen: Whisper's output carries no language per word, so no setting can follow that.

## Offline translation (`--offline`)

Works on `transcribe`, `translate`, `record` and `show`, and as a checkbox in the Mac app. Each line is translated from the
language it is tagged with, so mixed-language transcripts work; lines already in the target language are left as they are.

- **Model:** NLLB-200 distilled 1.3B (Meta), converted for CTranslate2 by the OpenNMT project (`MT_NLLB_MODEL` can name another,
  for example `JustFrederik/nllb-200-distilled-600M-ct2-int8`, half the size and noticeably worse: on a test it dropped half
  of a long sentence). The first use downloads it (about 1.4 GB; the app and `record`/`transcribe --offline` fetch it before a
  long job starts, not after); after that it runs without internet. The models live in the Hugging Face cache
  (`~/.cache/huggingface/hub`, about 3 GB with the speech model and NLLB; Hy-MT3 adds 4.2 GB, TranslateGemma 7 GB) — delete that folder to free the space; the engine downloads again when needed.
  On Apple silicon the app uses [Hy-MT3 7B](#hy-mt3-7b---translator-hy-the-apps-default-on-apple-silicon) unless you pick
  another model, and [TranslateGemma 12B](#translategemma-12b---translator-gemma) can be chosen; both leave the lines of a
  language they lack to NLLB. The command line uses NLLB unless `--translator` says otherwise.
- **Languages:** about 90 common ones by English name or two-letter code (`English`, `ru`, `Brazilian Portuguese`, `Cantonese`, ...),
  or any NLLB code such as `ces_Latn`. An unknown name stops the run before anything starts and says so.
- **Quality:** below Claude's. NLLB works on single sentences, so each line is split into sentences first; it can still be literal
  (the German "Guten Tag zusammen" became "Have a good day together"), and Chinese comes out with ASCII punctuation. A wrong
  language tag (`[de?]`) gives a wrong translation. Names, numbers and idioms deserve a check.
- **Speed:** about 0.4 s per two-sentence line on a 4-core CPU, so a 3-hour meeting is roughly 15–25 minutes **per target
  language** (measured on Linux x86; not measured on a Mac).
- **Licence and fitness:** the NLLB weights are CC-BY-NC 4.0, **non-commercial**, and Meta calls NLLB-200 a research model, not
  meant for document translation or certified translations. Fine for your own notes; use Claude translation (or another model)
  for commercial or legally binding work. Sentences over 200 tokens are cut at clause punctuation before translation, and a
  translation that comes back much shorter than its sentence is counted in the result ("may be incomplete").
- `translate` of plain text without `[time] [lang]` tags needs `--from LANGUAGE` with `--offline`.

### Hy-MT3 7B (`--translator hy`), the app's default on Apple silicon

On a Mac with Apple silicon, `--offline --translator hy` (on any command that translates) or `MT_TRANSLATOR=hy` uses Tencent's
[Hy-MT3-7B](https://huggingface.co/tencent/Hy-MT3-7B) (released 24 Nov 2026) instead of NLLB, on the GPU through `mlx-lm`. In
the app it is the *Offline model* for anyone who has not chosen one; a choice made before (TranslateGemma or NLLB) is kept.

- **Languages:** 102 (the model card); in `mt.py` 89 of Whisper's languages plus Traditional Chinese. Breton, Bosnian, Faroese,
  Hausa, Hawaiian, Latin, Lingala, Maori, Occitan, Shona and Yoruba go to NLLB, which is fetched too, and the note at the end
  says how many lines did; so does a target outside them (in the app 92 of the 101 offline targets are Hy-MT3's). Kazakh and
  Mongolian are sent as the Cyrillic varieties (the card's plain Kazakh and Mongolian are other scripts). Both Portuguese targets
  are written as Portuguese.
- **Quality, honestly:** every figure so far is Tencent's own, scored by AI judges (report
  [arXiv 2611.27673](https://arxiv.org/abs/2611.27673): FLORES-200 and WildMTBench; its 4-bit Q4_K_M stays within 0.6 points of
  bf16, Table 8, but that is llama.cpp's 4-bit, not MLX's). **No independent evaluation exists yet** (the model came out after
  WMT26), and there are no figures per language pair for Russian or German. This project's own check: 8 made-up meeting lines,
  Tencent's 7B GGUF on a CPU, 1 of 8 wrong against 3–4 of 8 for NLLB (a spot check, not a benchmark). The pinned MLX copy ran
  through `mt.py` on a Linux CPU (MLX 0.32.5): download and checks, load, six lines (ru→en three times, de→en, en→ru twice; one
  of them through `mt.py translate --offline --translator hy`) all right and all ended at the end token, a Yoruba line sent to
  NLLB, 8–14 minutes a line with MLX working on one core, 4.3–4.4 GiB at peak in that process (a CPU, not the Apple GPU: says
  nothing about a Mac's speed or memory). Not measured on a Mac.
- **The copy and the download:** about 4.2 GB, once. The weights are NetProjects' 4-bit MLX copy of Tencent's weights
  ([`netprojects/Hy-MT3-7B-MLX-4bit`](https://huggingface.co/netprojects/Hy-MT3-7B-MLX-4bit) at commit `dca6fb3`), the tokenizer
  files Tencent's own (commit `38d6c3c`), because the copy's `tokenizer_config.json` needs transformers 5 and the app pins 4.57.6.
  Each file is fetched at its pinned commit into the Hugging Face cache and checked against its SHA-256 before use (the 4.2 GB
  file once; a file that does not match is downloaded again, then the run stops and says which folder to delete).
  `MT_HY_MODEL` names another copy, a repository id or a folder, which is used as it is (no checksum, its own tokenizer): for
  example your own `mlx_lm.convert --hf-path tencent/Hy-MT3-7B -q --q-bits 4 --mlx-path ~/models/hy-mt3-7b-4bit`, a 15 GB
  download. A copy whose `tokenizer_config.json` names `TokenizersBackend` stops with a message to put Tencent's in its place.
- **Decoding:** greedy, with the card's repetition penalty (1.05 over the last 20 tokens), not the card's sampling: the same line
  always gives the same text, live, at the end and in a rerun. In a 10-line test meant to provoke loops (Tencent's GGUF on a
  CPU) no setting looped, and sampling changed half of the lines from run to run. The token limit of a request depends on the
  target's script (Burmese takes about 8 times the tokens of English); a line that still reaches it is kept and counted in the
  note ("ran to the token limit").
- **Cost:** about 4.5 GB of memory, about 6 GB beside the speech model while a recording is translated live (estimates from the
  same-size Hy-MT2-7B, 4.53 GB at peak; not measured); the run says so on a Mac with less than 12 GB. Speed not measured on a Mac.
- **Live, since 1.13.0:** the model is loaded when a recording starts, in steps between two stretches of speech, while the
  transcript goes on (the window's Translator card shows *Checking files*, *Loading xx %*, *Ready* and the load time). **One
  7B translator at a time on the Mac:** TranslateGemma and Hy-MT3 take a slot (a lock file in `~/Library/Caches/meeting-transcriber/`)
  while loaded; when a recording wants to translate live while an earlier job translates with the model, the job lets it go
  after its current line ("Paused: the recording in progress translates live (one translation model at a time)"), waits, and
  goes on from where it stopped when the recording ends; a job that finds the slot taken waits ("Waiting for the live
  translation of the recording in progress"). A job beside a live recording also pauses after each chunk or line as long as it
  took (at most 2 s), so it uses the GPU at most half the time. On a Mac with less than 12 GB (16 GB for TranslateGemma) the
  live translation uses NLLB instead and Hy-MT3 translates at the end. In a recording process MLX's buffer cache is capped at
  1 GB (`MT_MLX_CACHE_MB` changes it).
- **Licence:** Apache-2.0; Tencent's LICENSE.txt: "does not impose any additional restrictions beyond those specified in the
  license". Commercial use is allowed (NLLB's licence does not allow it).
- **Downloads, setup, live use:** as TranslateGemma's (below): fetched before a job, beside a recording without waiting, the same
  `requirements-mlx.txt` (no other package), NLLB with a note on an Intel Mac or without `mlx-lm`. **How it is called:** the
  card's plain prompt ("Translate the following text into English. Note that you should only output the translated result
  without any additional explanation:" and the text), one line per request, no system prompt; its end token `<|eos|>` is
  added as a stop token.

### TranslateGemma 12B (`--translator gemma`)

On a Mac with Apple silicon, the app's *Offline model* menu (or `--offline --translator gemma` on any command that
translates, or `MT_TRANSLATOR=gemma`) uses Google's TranslateGemma 12B instead of NLLB, on the GPU through `mlx-lm`.

- **Measured** on an Apple M5 with 24 GB, five meeting sentences (ru→en twice, de→en, en→ru, ru→de): all five right,
  1.5–2.4 s each, 17.6–19.1 tokens/s, 6.8 GB at peak. NLLB, on the same five sentences, got four: "send … to legal first"
  became «первым отправим» ("we will be the first to send") and «в юридическую» (cut off). Five sentences are a spot check,
  not a benchmark; Google reports MetricX 3.60 and COMET 83.5 for the 12B on WMT24++ (55 language pairs).
- **Languages:** the 51 it was evaluated on (WMT24++: English and ar bg bn ca cs da de el es et fa fi fil fr gu he hi hr hu id
  is it ja kn ko lt lv ml mr nl no pa pl pt ro ru sk sl sr sv sw ta te th tr uk ur vi zh zu; `Brazilian Portuguese`, `European
  Portuguese` and `Traditional Chinese` are their own targets). A line in any other language, or a target outside the 51,
  goes to NLLB, and the note at the end says how many lines did. A text longer than 1,500 characters is sent in pieces of whole
  sentences (the model takes 2K tokens of input).
- **Cost:** a 7 GB download the first time, about 7 GB of memory, 8.5 GB beside
  the speech model while a recording is translated live: 16 GB of memory or more is advised, and the run says so on a smaller
  Mac. One line per request, so a whole meeting translated at the end takes about an hour per language for 3 hours of
  speech (estimated from the per-line time, not measured); with live translation on, those lines are done during the
  meeting and only the tail is left.
- **Downloads:** a transcription or translation job fetches TranslateGemma (when a target is one of its languages) and NLLB
  (for the lines in a language TranslateGemma lacks, which Whisper can tag in any meeting) before it starts. A recording,
  and the follower beside a call, never wait for a download: the live worker fetches the models beside the work, and live
  translation starts once they are here (the lines before that are translated at the end). If NLLB cannot be had (no
  network), TranslateGemma's lines are still translated; the others stay in the original, the note says how many, and
  the translation is not kept as finished, so the next run tries again.
- **Setup:** the app's Apple-GPU setup step installs `mlx-lm` 0.29.1 with `transformers` 4.57.6 (`requirements-mlx.txt`), the
  combination the measurement ran with; on the command line `pip install -r requirements-mlx.txt`. Without them, or on an Intel
  Mac, NLLB translates instead and a note says why (a recording is never refused over the translation model). The model is `mlx-community/translategemma-12b-it-4bit`
  (`MT_GEMMA_MODEL` names another MLX copy, the 4B or the 27B); the download needs no Hugging Face login.
- **How it is called:** the model's chat template takes one message whose content is a list of one item with `type`,
  `source_lang_code`, `target_lang_code` and `text` (a plain prompt is refused); the answer ends with `<end_of_turn>`, which its
  MLX copy does not name as a stop token, so `mt.py` adds it (without it the model writes on to the token limit).
  `mlx-lm` 0.29.1 makes its GPU stream when it is first imported, and MLX 0.32 or later lets a stream be used only on the
  thread that made it; the live translation runs on another thread than the final pass, so `mt.py` gives `mlx-lm` a stream
  of the thread that generates (reproduced and checked with a small model on MLX 0.29.3, 0.31.2, 0.32.0 and 0.32.4 on a CPU;
  `mlx-lm` 0.31.3 made the same change upstream). The model itself is loaded on the live worker's thread, between two chunks.
- **Licence:** the Gemma Terms of Use, which allow commercial use (NLLB's do not). A translation made by one model is reused at
  the end only by the same model: switching models translates again.

## Limits (honest)

- **Who spoke** is a guess from the voices: alike voices (the same person on two devices, two similar speakers) can be merged or
  split; a wrong count is corrected by giving `--speakers N`, wrong names by `rename`. Without `--speakers` nothing is labelled.
- **Language tags are per stretch of speech, not per word.** Speech is cut at pauses into chunks, and Whisper detects the
  language once per chunk; with automatic recognition a chunk that sounds mixed is cut again at 0.2 s pauses and each piece
  detected on its own. A switch at a pause is caught; a switch in the middle of a sentence, with no pause at all, still shares
  the tag of its sentence, and very short utterances are guessed less reliably.
- **Speed depends on hardware.** On a 4-core 2.1 GHz Xeon (CPU engine, `large-v3-turbo` int8) a 10-minute three-language meeting
  took 0.33× its length at `normal`, 0.68× at `fine` and 0.99× at `finest`; every chunk costs about 5 s whatever its length, because
  Whisper pads it to 30 s. The GPU engine on Apple silicon is used when installed (not measured here); CUDA when available.
  `MT_WHISPER_MODEL=small` is 1.6× faster than the two-pass turbo was but worse everywhere (word error rate 55 %, hallucinated lines);
  the single-pass turbo is faster than `small` and better, so there is no reason to switch.
- **The other side of a call** is recorded only when you ask for it (see *Calls* above); otherwise it is the microphone only.
  A computer's-sound track that could not be captured, stayed silent or could not be transcribed costs you that side only: the
  transcript holds your side and a note saying what happened (the job then ends with "Finished with 1 problem" rather than "Done"
  when the track could not be transcribed).
- A 3-hour recording is ~350 MB of 16 kHz WAV (115 MB per hour per track). Transcribing holds the audio as float32 (about
  230 MB per hour) plus the speech model (about 1.6 GB); the voice-activity detector runs on 10-minute windows, so a 4-hour file
  needs about 1 GB for its audio and no large extra copies. A Mac with 8 GB can do it; close other large programs.
- **Live translation on a slow Mac:** with several targets and a dense meeting the live translator may not keep up; then many
  lines are marked *at the end* and translated only by the final pass (the transcript itself stays live). The Translator card
  shows how many and why (its share of the minute).
- **GPU busy** in the telemetry is an IORegistry counter that Apple does not document; take it as a hint.
- Recording people requires their consent in many jurisdictions.

## Version

`VERSION` holds the version (`python3 mt.py --version`, and the app's version), now 1.13.0; the launcher, another project of the development repository, shows it with the date of the last change.
Bump it when you change the tool or the app.

## Tests

```bash
pip install pytest numpy && python3 -m pytest tests                    # mt.py (needs ffmpeg for the audio-format tests)
# 1.13.0: the live scheduler's tests (T1-T16 of the design) start at "the live transcript never waits for a translation" in
# tests/test_mt.py: the stall reproduced with real threads (a slow translation, the first model load), the lag on a virtual
# clock over the lag driver's grid, the stop, the steps' text against 1.12.0's one-shot path, one thread for MLX, the LLM slot
# and the GPU yield, the low-memory fallback and the @@STAT events; the app's CoreTests have a "Telemetry" section
# tests/test_windows.py covers the Windows parts (faked where needed) and runs for real on the Windows CI job, where the
# tests that send POSIX signals skip; on Windows run them with $env:PYTHONUTF8 = "1"
swiftc -module-name CoreTests app/Sources/Core/*.swift app/Tests/CoreTests/main.swift -o /tmp/mt-core-tests && /tmp/mt-core-tests   # the app's logic
```
