#!/usr/bin/env python3
"""tools/win_e2e.py: the end-to-end check of mt.py's Windows install, for the CI job "Engine on Windows" on the
windows-latest runner. After the install of README "Windows (command line)" (a venv, requirements.txt) it runs what
that section tells a user to run, with the real speech engine, and prints one PASS or FAIL line per check; any FAIL
makes the exit code 1. Each step is its own command, so that each is its own step in the job's log:

    python tools\\win_e2e.py imports              imports every native package of requirements.txt, each in a child,
                                                 then all of them in one child in the order of a --speakers run
    python tools\\win_e2e.py speech WORK          writes the input WORK\\встреча.*: SENTENCES spoken by Windows' own speech
                                                 synthesizer (SAPI through System.Speech) as встреча.wav, else the
                                                 committed clip CLIP (the same sentences, espeak-ng) as встреча.flac;
                                                 WORK\\speech.json says which. Either way it is speech, so the later
                                                 steps check what the engine decoded, never only exit codes
    python tools\\win_e2e.py prepare WORK         mt.py prepare: ffmpeg fetched by static-ffmpeg, Whisper tiny downloaded;
                                                 once more after a failure that looks like a network error (INFRA lines)
    python tools\\win_e2e.py transcribe WORK      mt.py transcribe of the input with its stdout and stderr through pipes
    python tools\\win_e2e.py control WORK         the same twice with MT_CONTROL=stdin and MT_EVENTS=1: stdin kept open to
                                                 the end, then stdin closed as soon as @@DONE is read
    python tools\\win_e2e.py cli WORK             mt.py --help, --version and devices (the runner has no sound device)

A transcript passes when: no note says a chunk failed, speech was not recognised or no speech was found; every line is
speech or a note; every speech line is tagged en or en?; at least MIN_WORDS_EACH content word of each of the two
sentences is there (so every chunk was decoded); and mt.py's stderr holds no traceback and no failed chunk.

Run it with the Python that has requirements.txt installed (the venv's): mt.py runs as a child of that same Python
(sys.executable), from mt.py's folder. Every mt.py child gets the environment a user's PC gives it: no PYTHONUTF8 or
PYTHONIOENCODING (a pipe then gets the ANSI code page, cp1252 on the runner), no ANTHROPIC_API_KEY, and no MT_OUT,
MT_EVENTS, MT_CONTROL, MT_DEVICE or MT_ENGINE unless a step sets them; then MT_WHISPER_MODEL=tiny (about 75 MB),
MT_HOME=WORK\\home\\data (the data, the speaker models and ffmpeg stay under WORK), and PATH without the folders that hold
ffmpeg or ffprobe, so that static-ffmpeg fetches them as on a PC that has none (--system-ffmpeg keeps them). Nothing is
translated (no --to: Claude needs a key, the offline model is 1.4 GB).

Each step has a time budget (STEP_BUDGET, --budget): every child gets the time left, so a hang ends with this tool's
FAIL line and RESULT line, before the workflow step's timeout-minutes (at least the budget plus STEP_MARGIN_S; a test
compares them) kills the step.

Python 3.9 or newer, standard library only.
"""
import argparse
import array
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

WIN = sys.platform == "win32"
HERE = Path(__file__).resolve().parent
MT = HERE.parent / "mt.py"
INPUT_STEM = "встреча"          # "meeting" in Russian: a Cyrillic name for the paths, the stdout pipe and the @@ events
SAPI_FILE = "speech-sapi.wav"   # what PowerShell writes (an ASCII name), renamed to встреча.wav by Python
SENTENCES = ("Hello, this is a test of the meeting transcriber on Windows, reading a file with a Cyrillic name.",
             "The weather is nice today, and the coffee is hot.")
SENTENCE = " ".join(SENTENCES)
PAUSE_MS = 1200                 # between the two sentences: longer than mt.py's 700 ms split (GRANULARITY normal, mt.py:27)
# The input when Windows cannot speak: SENTENCES as spoken by espeak-ng 1.51 (voice en-us, its default 175 words a
# minute), the first sentence padded with 1.2 s of silence and the second with 0.5 s, 16 kHz mono 16-bit FLAC (10.08 s,
# about 150 KB). The first sentence is over 5 s, mt.py's MIN_S (mt.py:23), so mt.py cuts the clip into two chunks:
#   espeak-ng -v en-us -w s1.wav "<SENTENCES[0]>"; espeak-ng -v en-us -w s2.wav "<SENTENCES[1]>"
#   ffmpeg -i s1.wav -i s2.wav -filter_complex "[0:a]aresample=16000,apad=pad_dur=1.2[a];[1:a]aresample=16000,
#     apad=pad_dur=0.5[b];[a][b]concat=n=2:v=0:a=1[out]" -map "[out]" -map_metadata -1 -fflags +bitexact
#     -flags:a +bitexact -ac 1 -sample_fmt s16 -c:a flac -compression_level 12 win_e2e_sentence.flac   (ffmpeg 6.1.1)
# Whisper tiny on Linux (faster-whisper 1.2.1, CTranslate2 4.9.1) made of it: "[00:00:00] [en] Hello, this is the first
# time I'm eating transcriber on Windows, reading a file with a similic name." and "[00:00:06] [en] The weather is nice
# today, and the coffee is hot.", in two chunks.
CLIP = HERE / "win_e2e_sentence.flac"
CLIP_SHA256 = "0efeec649d19478112684ac2a193e4abf33b971905e9d7f61953197586a4a7be"
MIN_WORDS_EACH = 1              # at least this many content words of each sentence must be in the transcript
STOP_WORDS = frozenset({"this", "that", "with", "from", "have", "were", "there", "their", "what", "when", "your"})
SPEECH_TAGS = frozenset({"en", "en?"})   # the language tag every speech line must have (en? is a low-confidence en)
RATE = 16000                    # mt.py:21, the rate mt.py decodes at (and the synthesizer is asked for)
SPEECH_PEAK = 0.01              # a synthesized file whose loudest sample stays below this is no speech
# Left out of every mt.py child's environment: what would make it differ from a user's PC (PYTHONUTF8 and
# PYTHONIOENCODING would hide the cp1252 pipe, PYTHONLEGACYWINDOWSSTDIO changes the console streams), what would reach
# the network for translation (ANTHROPIC_API_KEY), and mt.py's own settings, which the steps set themselves.
CHILD_ENV_DROP = frozenset({"PYTHONUTF8", "PYTHONIOENCODING", "PYTHONLEGACYWINDOWSSTDIO", "ANTHROPIC_API_KEY", "MT_OUT",
                            "MT_EVENTS", "MT_CONTROL", "MT_DEVICE", "MT_ENGINE", "MT_HOME", "MT_WHISPER_MODEL"})
# A speech line of a transcript file: "[hh:mm:ss] [xx] text" or "[xx?]", a speaker as "<Name> " before the text (LINE at
# mt.py:3401, fmt at mt.py:3405); a line whose own text starts with "<" gets a zero-width space in front (mt.py:3400).
SPEECH_LINE = re.compile(r"^\[\d+:\d\d:\d\d\] \[([^\]?\s]+\??)\] (?:<[^<>]+> )?(.*)$")
NOTE_LINE = re.compile(r"^# (?:\[\d+:\d\d:\d\d\] )?(.*)$")      # NOTE_LINE at mt.py:3402
# Notes that say speech was lost: a chunk whose decode raised (mt.py:1743), a chunk or piece that gave nothing
# (mt.py:2434, 2440), a file in which the VAD found no speech (mt.py:1805).
BAD_NOTE = re.compile(r"chunk failed|speech not recognised|No speech was found")
BAD_STDERR = ("Traceback", "warning: chunk")      # a crash, or a chunk that failed (mt.py:1742) while the run went on
# What a network failure looks like on mt.py's stderr: static-ffmpeg's fetch (mt.py:1067, then the exit at mt.py:4024),
# the speaker models (urllib, mt.py:1484, not fatal), the Whisper model (huggingface_hub: requests or httpx errors, and
# LocalEntryNotFoundError when the Hub cannot be reached and nothing is cached).
INFRA = re.compile(r"could not be fetched|ConnectError|ConnectionError|ConnectTimeout|ReadTimeout|TimeoutError|timed out"
                   r"|getaddrinfo failed|Name or service not known|Temporary failure in name resolution|Max retries "
                   r"exceeded|RemoteDisconnected|IncompleteRead|SSLError|ProxyError|URLError|LocalEntryNotFoundError"
                   r"|HfHubHTTPError|HTTP Error (?:429|5\d\d)|\b(?:429|5\d\d) (?:Server Error|Too Many Requests)")
# Seconds each step may take in all; the workflow gives each step timeout-minutes of at least this plus STEP_MARGIN_S.
STEP_BUDGET = {"imports": 240, "speech": 120, "prepare": 600, "transcribe": 240, "control": 330, "cli": 120}
STEP_MARGIN_S = 60
IMPORT_S = 60                   # one import child at most
SAPI_S = 50                     # one PowerShell at most (two are tried)
CONTROL_RUN_S = 150             # one of the two control runs at most
RETRY_MIN_S = 180               # prepare runs once more only with at least this much of its budget left
# Each mt.py import, with what its version line shows: the native library inside where there is one.
IMPORTS = (
    ("ctranslate2", "m.__version__ + ', CUDA devices: ' + str(m.get_cuda_device_count())"),
    ("faster_whisper", "m.__version__"),
    ("onnxruntime", "m.__version__ + ', ' + m.get_device()"),
    ("av", "m.__version__"),
    ("tokenizers", "m.__version__"),
    ("numpy", "m.__version__"),
    ("huggingface_hub", "m.__version__"),
    ("hf_xet", "getattr(m, '__version__', 'imported')"),
    ("sherpa_onnx", "getattr(m, '__version__', 'imported')"),
    ("soundfile", "m.__version__ + ', libsndfile ' + m.__libsndfile_version__"),
    ("sounddevice", "m.__version__ + ', ' + m.get_portaudio_version()[1]"),
    ("static_ffmpeg", "'imported'"),
    ("anthropic", "m.__version__"),
)
# The same in one process, in the order a transcription with --speakers loads them: faster_whisper (with av and
# ctranslate2) first (mt.py:1675, 1684), then sherpa_onnx for the speakers (diarize, mt.py:1692 -> 1548), and only then
# onnxruntime, which faster-whisper's VAD imports when its first session is made (cut_chunks, mt.py:1696 -> 1821;
# faster_whisper/vad.py get_vad_model). Two of them ship a DLL named onnxruntime.dll (sherpa_onnx\lib and
# onnxruntime\capi), and Windows gives a later load of that name the one loaded first; so after sherpa_onnx one silero
# VAD session is made and run, as cut_chunks does. No sherpa diarization session is made (its models come with prepare):
# this says nothing about whether --speakers works, only whether the VAD still works once sherpa_onnx is loaded.
ONE_PROCESS = r"""
import sys
import faster_whisper, faster_whisper.vad, ctranslate2, av, sherpa_onnx
import numpy
from faster_whisper.vad import VadOptions, get_speech_timestamps
spans = get_speech_timestamps(numpy.zeros(16000, dtype=numpy.float32), VadOptions())
import onnxruntime, soundfile, sounddevice
loaded = 'not looked up (not Windows)'
if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleHandleW.restype = wintypes.HMODULE
    kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = kernel.GetModuleHandleW('onnxruntime.dll')
    loaded = 'none loaded'
    if handle:
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetModuleFileNameW(handle, buffer, len(buffer))
        loaded = buffer.value
print('silero VAD session after sherpa_onnx: ' + str(len(spans)) + ' speech spans in 1 s of silence; '
      'onnxruntime.dll in this process: ' + loaded)
"""
ONE_PROCESS_LABEL = ("faster_whisper, ctranslate2, av, sherpa_onnx, then a silero VAD session, then onnxruntime, "
                     "soundfile, sounddevice (imports and the VAD only: no speaker-separation session)")
SAPI_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
  Add-Type -AssemblyName System.Speech
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try {
    $all = @($s.GetInstalledVoices())
    foreach ($v in $all) {
      Write-Output ('VOICE ' + $v.VoiceInfo.Name + ' | ' + $v.VoiceInfo.Culture.Name + ' | enabled: ' + $v.Enabled)
    }
    $english = @($all | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*' })
    if ($english.Count -eq 0) { throw ('no enabled English voice among ' + $all.Count + ' installed') }
    $s.SelectVoice($english[0].VoiceInfo.Name)
    Write-Output ('USING ' + $s.Voice.Name)
    $format = [System.Speech.AudioFormat.SpeechAudioFormatInfo]::new(16000,
      [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $prompt = New-Object System.Speech.Synthesis.PromptBuilder
    $prompt.AppendText(@@TEXT1@@)
    $prompt.AppendBreak([TimeSpan]::FromMilliseconds(@@PAUSE@@))
    $prompt.AppendText(@@TEXT2@@)
    $s.SetOutputToWaveFile(@@PATH@@, $format)
    $s.Speak($prompt)
    $s.SetOutputToNull()
  } finally { $s.Dispose() }
} catch {
  Write-Output ('ERROR ' + $_.Exception.Message)
  exit 1
}
"""


def say(text=""):
    print(text, flush=True)


def annotate(kind, text):
    """A GitHub Actions annotation (shown on the run's summary page) when running there; elsewhere nothing. The message
    is escaped as workflow commands want it (% CR LF)."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        say(f"::{kind}::" + text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A"))


class Report:
    """The PASS and FAIL lines of one step, and its exit code."""
    def __init__(self, step):
        self.step, self.passed, self.failed = step, 0, 0

    def ok(self, text):
        self.passed += 1
        say(f"PASS: {text}")

    def fail(self, text):
        self.failed += 1
        say(f"FAIL: {text}")
        annotate("error", f"win_e2e {self.step}: {text}")

    def check(self, condition, good, bad=None):
        if condition:
            self.ok(good)
        else:
            self.fail(bad or f"not so: {good}")
        return bool(condition)

    def end(self):
        say(f"RESULT {self.step}: {self.passed} passed, {self.failed} failed")
        return 1 if self.failed else 0


class Clock:
    """A step's time budget: the seconds left for its next child (at least 1, at most `cap`)."""
    def __init__(self, seconds):
        self.seconds = seconds
        self.end = time.monotonic() + seconds

    def left(self, cap=None):
        seconds = max(1.0, self.end - time.monotonic())
        return min(seconds, cap) if cap else seconds


# ---------- the pure parts (tests/test_win_e2e.py runs them on Linux) ----------
def content_words(sentence):
    """The words of `sentence` a transcript is checked for: lower case, in order, once each, at least 4 letters and not a
    stop word (short and common words would match by chance)."""
    words = []
    for word in re.findall(r"[a-z']+", sentence.lower()):
        if len(word) >= 4 and word not in STOP_WORDS and word not in words:
            words.append(word)
    return words


def transcript_parts(transcript):
    """A transcript file as (speech, notes, other): speech as (language tag as written, text) per speech line, without
    time and speaker; notes without "# " and their time; other: the lines that are neither (mt.py writes none)."""
    speech, notes, other = [], [], []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line:
            continue
        note = NOTE_LINE.match(line)
        if note:
            notes.append(note.group(1))
            continue
        m = SPEECH_LINE.match(line)
        if m:
            speech.append((m.group(1), m.group(2).replace("\u200b", "")))
        else:
            other.append(line)
    return speech, notes, other


def speech_text(transcript):
    """What was said in a transcript file: its speech lines without time, language tag and speaker. Notes ("# ...") are
    left out: they are mt.py's words, not the speaker's."""
    return "\n".join(text for _, text in transcript_parts(transcript)[0])


def words_found(sentence, transcript):
    """The content words of `sentence` that the transcript's speech holds as whole words, ignoring case."""
    text = speech_text(transcript).lower()
    return [w for w in content_words(sentence) if re.search(rf"(?<![a-z']){re.escape(w)}(?![a-z'])", text)]


def stderr_problems(text):
    """The lines of a child's stderr that show a crash or a failed chunk."""
    return [l.strip() for l in re.split(r"[\r\n]+", text or "") if any(b in l for b in BAD_STDERR)]


def infra_lines(text):
    """The lines of a child's stderr that look like a network failure, not like a fault of mt.py."""
    return [l.strip() for l in re.split(r"[\r\n]+", text or "") if l.strip() and INFRA.search(l)]


def ffmpeg_names(windows=WIN, pathext=None):
    """The file names under which a PATH search finds ffmpeg or ffprobe: on Windows with every PATHEXT extension
    (shutil.which, which mt.py uses at mt.py:1053, tries them all)."""
    if not windows:
        return ("ffmpeg", "ffprobe")
    exts = [e.lower() for e in (pathext or os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD").split(";") if e]
    return tuple(base + ext for base in ("ffmpeg", "ffprobe") for ext in exts)


def ffmpeg_free_path(value, names=None, isfile=os.path.isfile, sep=os.pathsep):
    """PATH without the folders that hold ffmpeg or ffprobe, and the folders left out: mt.py then finds neither and
    static-ffmpeg fetches both (mt.py:1049, ensure_ffmpeg), as on a PC that has none."""
    names = names or ffmpeg_names()
    kept, removed = [], []
    for folder in (value or "").split(sep):
        if folder and any(isfile(os.path.join(folder, name)) for name in names):
            removed.append(folder)
        else:
            kept.append(folder)
    return sep.join(kept), removed


def mt_env(base, home, model="tiny", extra=None, keep_ffmpeg=False):
    """The environment of an mt.py child (see the module's docstring) and the PATH folders left out of it. Names are
    compared in upper case: Windows' are case-insensitive."""
    env = {k: v for k, v in base.items() if k.upper() not in CHILD_ENV_DROP}
    env["MT_WHISPER_MODEL"] = model
    env["MT_HOME"] = str(Path(home) / "data")
    removed = []
    if not keep_ffmpeg:
        key = next((k for k in env if k.upper() == "PATH"), "PATH")
        env[key], removed = ffmpeg_free_path(env.get(key, ""))
    env.update(extra or {})
    return env, removed


def parse_events(text):
    """The @@ events of mt.py's stdout (mt.py:80, ev: "@@KIND<TAB>field<TAB>..." one per line) as (kind, [fields])."""
    events = []
    for line in text.split("\n"):
        if line.startswith("@@"):
            kind, *fields = line[2:].split("\t")
            events.append((kind, fields))
    return events


def same_path(a, b):
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def transcript_of(src):
    """Where `mt.py transcribe <src>` writes the transcript without --out and MT_OUT: <stem>.transcript.txt beside the
    input (write_outputs at mt.py:3529, the folder from result_folder at mt.py:3670 with the input's folder, mt.py:4141)."""
    src = Path(src)
    return src.with_name(src.stem + ".transcript.txt")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wav_summary(path):
    """{rate, channels, bits, seconds, peak} of a PCM WAV (peak as a fraction of full scale; 16-bit only, else 0)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width, count = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        data = w.readframes(count)
    peak = 0.0
    if width == 2 and len(data) >= 2:
        samples = array.array("h")
        samples.frombytes(data[: len(data) // 2 * 2])
        if sys.byteorder == "big":
            samples.byteswap()
        peak = max(abs(x) for x in samples) / 32768
    return {"rate": rate, "channels": channels, "bits": width * 8, "seconds": round(count / rate, 2) if rate else 0.0,
            "peak": round(peak, 4)}


def ps_quote(text):
    """A PowerShell single-quoted string literal."""
    return "'" + text.replace("'", "''") + "'"


def tail(data, lines=4):
    """The last lines of a child's stderr on one line (a traceback's last frame and its error)."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else (data or "")
    return " | ".join([l.strip() for l in text.splitlines() if l.strip()][-lines:]) or "no output"


# ---------- running things ----------
def workdir(a):
    return Path(a.work).resolve()


def child_env(a, work, extra=None):
    env, removed = mt_env(os.environ, work / "home", a.model, extra, a.system_ffmpeg)
    return env, removed


class Child:
    """A child with stdout and stderr through pipes, each read to its end by a thread (so neither pipe fills up): stdout
    kept, and passed line by line to on_line(child, raw) when given; stderr kept and shown in the log as it comes, a
    progress line ("\\r" between its updates) as its last update. wait() never blocks past its timeout plus 10 s, also
    when a grandchild still holds a pipe."""
    def __init__(self, cmd, env, cwd=None, stdin_pipe=False, on_line=None, label="stderr"):
        self.on_line, self.label = on_line, label
        self.out, self.err = [], []
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.threads = [threading.Thread(target=f, daemon=True) for f in (self._read_out, self._read_err)]
        for t in self.threads:
            t.start()

    def _read_out(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.out.append(raw)
            if self.on_line:
                self.on_line(self, raw)

    def _read_err(self):
        for raw in iter(self.proc.stderr.readline, b""):
            self.err.append(raw)
            shown = [s for s in raw.decode("utf-8", "replace").rstrip("\r\n").split("\r") if s.strip()]
            if shown:
                say(f"  {self.label}| {shown[-1]}")

    def wait(self, timeout):
        """The exit code, or None when the child was still running after `timeout` seconds (it is then killed)."""
        try:
            code = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for t in self.threads:
            t.join(timeout=10)
        return code

    def stdout(self):
        return b"".join(self.out)

    def stderr(self):
        return b"".join(self.err).decode("utf-8", "replace")


def start_mt(a, args, env, **kw):
    """mt.py <args> as a user runs it: from its folder, stdout and stderr through pipes, stdin empty unless a pipe."""
    return Child([sys.executable, str(a.mt), *args], env, cwd=str(Path(a.mt).parent), **kw)


def run_mt(a, args, env, timeout):
    """mt.py <args> to its end (a short command): from its folder, stdin empty, stdout and stderr through pipes."""
    return subprocess.run([sys.executable, str(a.mt), *args], cwd=str(Path(a.mt).parent), env=env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def show(label, data):
    """Bytes of a child's stream, line by line, as text in the log."""
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else (data or "")
    for line in text.splitlines():
        say(f"  {label}| {line}")


def say_infra(step, text):
    """INFRA lines for what looks like a network failure in a child's stderr; returns them."""
    lines = infra_lines(text)
    for line in lines:
        say(f"INFRA: {line}")
    if lines:
        annotate("warning", f"win_e2e {step}: a network failure on mt.py's stderr (INFRA lines in the log): {lines[0]}")
    return lines


def check_utf8(r, data, what):
    """stdout as text, with a PASS when it is valid UTF-8 with "\\n" line ends (mt.py's utf8_streams, mt.py:3861)."""
    try:
        text = data.decode("utf-8")
        r.ok(f"{what} is valid UTF-8 ({len(data)} bytes)")
    except UnicodeDecodeError as e:
        r.fail(f"{what} is not UTF-8 ({e}): the pipe got another encoding")
        text = data.decode("utf-8", "replace")
    r.check(b"\r\n" not in data, f"{what} ends its lines with \\n only", f"{what} has \\r\\n line ends")
    return text


def check_stderr(r, text, what):
    bad = stderr_problems(text)
    r.check(not bad, f"{what}: no traceback and no failed chunk", f"{what}: " + " | ".join(bad[:3]))


def pipe_encoding(env):
    """What a Python child's stdout is on a pipe with this environment, before any program switches it."""
    try:
        done = subprocess.run([sys.executable, "-c", "import sys; print(sys.stdout.encoding)"], env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, timeout=60)
        return done.stdout.decode("ascii", "replace").strip() or "?"
    except Exception as e:                                 # only reported
        return f"? ({type(e).__name__})"


def say_pipe_encoding(env):
    enc = pipe_encoding(env)
    say(f"INFO: a Python child's stdout on a pipe here, before mt.py switches its own to UTF-8: {enc}")
    if WIN and enc.lower().replace("-", "") in ("utf8", "utf8sig"):
        annotate("warning", "a pipe is UTF-8 here, not the ANSI code page: this run does not exercise a cp1252 pipe")


def load_speech(r, work):
    """speech.json of the speech step, or None (a FAIL)."""
    try:
        info = json.loads((work / "speech.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        r.fail(f"no speech.json in {work} ({type(e).__name__}): run the speech step first")
        return None
    if not (isinstance(info.get("input"), str) and isinstance(info.get("sentences"), list) and info["sentences"]):
        r.fail(f"speech.json in {work} names no input or no sentences: run the speech step again")
        return None
    say(f"INFO: the input {info['input']} was made by: {info.get('source')}"
        + (f" ({info['voice']})" if info.get("voice") else f" ({info.get('reason')})"))
    return info


def check_content(r, info, transcript):
    """What the engine decoded: no lost speech, nothing but speech and notes, English tags, words of every sentence."""
    speech, notes, other = transcript_parts(transcript)
    lost = [n for n in notes if BAD_NOTE.search(n)]
    r.check(not lost, f"no note says speech was lost ({len(notes)} note(s) in all)",
            "the transcript says speech was lost: " + " | ".join(lost))
    r.check(not other, "every line is speech or a note",
            f"{len(other)} line(s) are neither speech nor a note: {other[:2]}")
    tags = sorted({tag for tag, _ in speech})
    r.check(speech and all(tag in SPEECH_TAGS for tag, _ in speech),
            f"{len(speech)} speech line(s), every one tagged en or en?: {', '.join(tags)}",
            f"{len(speech)} speech line(s) with the tags {tags}: every one must be en or en?")
    for n, sentence in enumerate(info["sentences"], 1):
        found, words = words_found(sentence, transcript), content_words(sentence)
        r.check(len(found) >= MIN_WORDS_EACH,
                f"sentence {n}: {len(found)} of its {len(words)} words are in the transcript: {', '.join(found)}",
                f"sentence {n}: {len(found)} of its words ({', '.join(words)}) are in the transcript, at least "
                f"{MIN_WORDS_EACH} expected (was its chunk decoded?)")


def check_transcript(r, expected):
    """The transcript file beside the input: PASS when it is there and readable as UTF-8; returns its text."""
    if not r.check(expected.is_file(), f"the transcript is beside the input: {expected}",
                   f"no transcript at {expected}"):
        return ""
    try:
        text = expected.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        r.fail(f"the transcript cannot be read as UTF-8: {e}")
        return ""
    show("transcript", text)
    return text


# ---------- the steps ----------
def synthesize(path, sentences, clock):
    """Windows: `sentences` (two) spoken by an English SAPI voice into `path` (16 kHz mono PCM16) with PAUSE_MS between
    them, through Windows PowerShell 5.1 (System.Speech is a .NET Framework assembly), else PowerShell 7. Returns (voice
    used or None, voices listed, the reason when no file was written)."""
    script = (SAPI_SCRIPT.replace("@@PATH@@", ps_quote(str(path))).replace("@@TEXT1@@", ps_quote(sentences[0]))
              .replace("@@TEXT2@@", ps_quote(sentences[1])).replace("@@PAUSE@@", str(PAUSE_MS)))
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    reasons, voices = [], []
    for shell in ("powershell.exe", "pwsh.exe"):
        try:
            done = subprocess.run([shell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                  stdin=subprocess.DEVNULL, capture_output=True, timeout=clock.left(SAPI_S))
        except (OSError, subprocess.TimeoutExpired) as e:
            reasons.append(f"{shell}: {type(e).__name__}: {e}")
            continue
        lines = done.stdout.decode("utf-8", "replace").splitlines()
        voices = voices or [l[len("VOICE "):] for l in lines if l.startswith("VOICE ")]
        using = next((l[len("USING "):] for l in lines if l.startswith("USING ")), None)
        error = next((l[len("ERROR "):] for l in lines if l.startswith("ERROR ")), None)
        if done.returncode == 0 and using and Path(path).is_file():
            return using, voices, None
        reasons.append(f"{shell}: exit {done.returncode}: {error or tail(done.stderr)}")
    return None, voices, "; ".join(reasons)


def step_speech(a):
    r = Report("speech")
    work = workdir(a)
    work.mkdir(parents=True, exist_ok=True)
    for name in (INPUT_STEM + ".wav", INPUT_STEM + CLIP.suffix, SAPI_FILE):
        if (work / name).exists():
            (work / name).unlink()
    info = {"sentence": SENTENCE, "sentences": list(SENTENCES)}
    target = None
    if WIN:
        raw = work / SAPI_FILE
        voice, voices, reason = synthesize(raw, SENTENCES, a.clock)
        info["voices"] = voices
        say("INFO: speech voices (System.Speech): " + ("; ".join(voices) if voices else "none listed"))
        if voice:
            try:
                summary = wav_summary(raw)
            except (OSError, EOFError, wave.Error) as e:   # Python's wave cannot read it: ffmpeg may, the words decide
                summary = None
                say(f"INFO: {raw.name} could not be read with Python's wave module ({e}); it is used as it is")
            if summary and (summary["seconds"] < 1 or summary["peak"] < SPEECH_PEAK):
                reason = f"{voice} wrote no speech: {summary}"
            else:
                target = work / (INPUT_STEM + ".wav")
                os.replace(raw, target)
                info.update(source="sapi", voice=voice)
                say(f"SPEECH: synthesized with {voice}: {SENTENCE!r}")
                say(f"INFO: {target}: {summary}")
    else:
        reason = "not Windows: no System.Speech here"
    if target is None:
        target = work / (INPUT_STEM + CLIP.suffix)
        try:
            data = CLIP.read_bytes()
        except OSError as e:
            r.fail(f"the committed clip {CLIP} cannot be read: {e}")
            return r.end()
        digest = hashlib.sha256(data).hexdigest()
        r.check(digest == CLIP_SHA256, f"the committed clip {CLIP.name} is intact (sha256 {digest[:16]}...)",
                f"{CLIP.name} has sha256 {digest}, not {CLIP_SHA256}: it was changed or damaged")
        target.write_bytes(data)
        info.update(source="clip", reason=reason)
        say(f"SPEECH: FALLBACK to the committed clip {CLIP.name} (espeak-ng, the same sentences) because {reason}; "
            "the later steps check its words all the same")
        annotate("notice", f"win_e2e speech: the committed clip instead of Windows' speech synthesizer ({reason})")
    info.update(input=target.name, file=str(target))
    r.check(target.is_file() and target.stat().st_size > 44, f"the input is written under its Cyrillic name: {target} "
            f"({target.stat().st_size if target.is_file() else 0} bytes)", f"no input file at {target}")
    (work / "speech.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return r.end()


def step_imports(a):
    r = Report("imports")
    env = {k: v for k, v in os.environ.items() if k.upper() not in CHILD_ENV_DROP}
    for name, expr in IMPORTS:
        code = f"import importlib; m = importlib.import_module({name!r}); print({expr})"
        timeout = a.clock.left(IMPORT_S)
        try:
            done = subprocess.run([sys.executable, "-c", code], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            r.fail(f"import {name}: no answer in {timeout:.0f} s")
            continue
        out = done.stdout.decode("utf-8", "replace").strip()
        r.check(done.returncode == 0, f"import {name}: {out}",
                f"import {name} failed (exit {done.returncode}): {tail(done.stderr)}")
    timeout = a.clock.left(2 * IMPORT_S)
    try:
        done = subprocess.run([sys.executable, "-c", ONE_PROCESS], env=env, stdin=subprocess.DEVNULL,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        r.fail(f"in one process ({ONE_PROCESS_LABEL}): no answer in {timeout:.0f} s")
        return r.end()
    out = done.stdout.decode("utf-8", "replace").strip()
    r.check(done.returncode == 0, f"in one process: {ONE_PROCESS_LABEL}: {out}",
            f"in one process ({ONE_PROCESS_LABEL}) it failed (exit {done.returncode}): {tail(done.stderr)}")
    return r.end()


def step_prepare(a):
    r = Report("prepare")
    work = workdir(a)
    work.mkdir(parents=True, exist_ok=True)
    env, removed = child_env(a, work)
    if a.system_ffmpeg:
        say("INFO: PATH kept as it is (--system-ffmpeg): a system ffmpeg is used when there is one")
    elif removed:
        say("INFO: PATH without the folders that hold ffmpeg or ffprobe: " + "; ".join(removed))
    else:
        say("INFO: no folder on PATH holds ffmpeg or ffprobe (as on a PC without them)")
    for attempt in (1, 2):
        started = time.monotonic()
        child = start_mt(a, ["prepare"], env)
        timeout = a.clock.left()
        code = child.wait(timeout)
        if code is None:
            r.fail(f"mt.py prepare did not end in {timeout:.0f} s")
            return r.end()
        say(f"INFO: mt.py prepare (attempt {attempt}) took {time.monotonic() - started:.1f} s, exit code {code}")
        show("stdout", child.stdout())
        network = say_infra("prepare", child.stderr())
        if code != 0 and network and attempt == 1 and a.clock.left() >= RETRY_MIN_S + a.retry_wait:
            say(f"INFO: mt.py prepare failed on what looks like a network failure: it runs once more in {a.retry_wait:g} s")
            time.sleep(a.retry_wait)
            continue
        break
    if code != 0 and network:
        say("INFRA: the failure below looks like a network failure, not like a fault of mt.py (see the INFRA lines)")
    out = child.stdout().decode("utf-8", "replace").splitlines()
    r.check(code == 0, "mt.py prepare: exit code 0", f"mt.py prepare: exit code {code}")
    r.check("ffmpeg: ok" in out, "mt.py prepare says ffmpeg: ok", "mt.py prepare does not say ffmpeg: ok")
    r.check("engine: ct2" in out, "mt.py prepare says engine: ct2 (faster-whisper on CTranslate2)",
            "mt.py prepare does not say engine: ct2")
    r.check("ready" in out, "mt.py prepare says ready", "mt.py prepare does not say ready")
    check_stderr(r, child.stderr(), "mt.py prepare's stderr")
    if not a.system_ffmpeg:
        home = work / "home" / "ffmpeg"                   # FFMPEG_DIR, mt.py:71: beside the data folder (MT_HOME)
        binaries = ("ffmpeg.exe", "ffprobe.exe") if WIN else ("ffmpeg", "ffprobe")
        crumbs = sorted(home.glob("*/installed.crumb"))
        fetched = [c for c in crumbs if all((c.parent / b).is_file() for b in binaries)]
        crumb = fetched[0].read_text(encoding="utf-8", errors="replace").strip() if fetched else ""
        r.check(fetched, f"static-ffmpeg fetched {' and '.join(binaries)} into {fetched[0].parent if fetched else ''}: "
                f"{crumb}", f"no ffmpeg and ffprobe fetched under {home}")
    models = work / "home" / "models"                     # MODELS_DIR, mt.py:72: fetched by prepare, not fatal there
    found = sorted(p for p in models.rglob("*") if p.is_file()) if models.is_dir() else []
    say("INFO: speaker-separation models: " + (", ".join(f"{p.relative_to(models)} ({p.stat().st_size // 1024} KiB)"
                                                       for p in found) or "none fetched"))
    return r.end()


def step_transcribe(a):
    r = Report("transcribe")
    work = workdir(a)
    info = load_speech(r, work)
    if info is None:
        return r.end()
    src = work / info["input"]
    expected = transcript_of(src)
    if expected.exists():
        expected.unlink()
    env, _ = child_env(a, work)
    say_pipe_encoding(env)
    started = time.monotonic()
    child = start_mt(a, ["transcribe", str(src)], env)
    timeout = a.clock.left()
    code = child.wait(timeout)
    if code is None:
        r.fail(f"mt.py transcribe did not end in {timeout:.0f} s")
        return r.end()
    say(f"INFO: mt.py transcribe took {time.monotonic() - started:.1f} s")
    show("stdout", child.stdout())
    say_infra("transcribe", child.stderr())
    r.check(code == 0, "mt.py transcribe: exit code 0", f"mt.py transcribe: exit code {code}")
    out = check_utf8(r, child.stdout(), "stdout (a pipe)")
    r.check(any(same_path(line.strip(), expected) for line in out.split("\n") if line.strip()),
            f"stdout names the transcript, Cyrillic intact: {expected}", f"stdout does not name {expected}")
    check_stderr(r, child.stderr(), "mt.py's stderr (a pipe)")
    check_content(r, info, check_transcript(r, expected))
    return r.end()


def control_run(a, r, info, work, folder_name, close_on_done):
    """mt.py transcribe of a copy of the input in WORK\\<folder_name> with MT_CONTROL=stdin and MT_EVENTS=1; stdin a pipe
    that gets the line pause once and then stays open, or is closed as soon as @@DONE is read (close_on_done). Returns
    (exit code or None, the child, the input, its transcript, whether mt.py still ran when stdin was closed on @@DONE)."""
    folder = work / folder_name
    folder.mkdir(exist_ok=True)
    src = folder / info["input"]
    shutil.copyfile(work / info["input"], src)
    expected = transcript_of(src)
    if expected.exists():
        expected.unlink()
    env, _ = child_env(a, work, {"MT_CONTROL": "stdin", "MT_EVENTS": "1"})
    closed = {}

    def on_line(child, raw):
        if close_on_done and raw.startswith(b"@@DONE") and "running" not in closed:
            closed["running"] = child.proc.poll() is None
            try:
                child.proc.stdin.close()
            except OSError:
                pass
    started = time.monotonic()
    child = start_mt(a, ["transcribe", str(src)], env, stdin_pipe=True, on_line=on_line)
    # One command the reader must answer: during transcribe "pause" does nothing but say so on stderr (Control.dispatch,
    # mt.py:405). The pipe then stays open, written to no more, until mt.py has ended or @@DONE is read.
    try:
        child.proc.stdin.write(b"pause\n")
        child.proc.stdin.flush()
    except (OSError, ValueError) as e:
        r.fail(f"the line pause could not be written to mt.py's stdin: {e}")
    timeout = a.clock.left(CONTROL_RUN_S)
    code = child.wait(timeout)
    if code is None:
        r.fail(f"mt.py ({folder_name}) still runs after {timeout:.0f} s with MT_CONTROL=stdin: the control channel "
               "keeps it alive, or it hangs")
    try:
        child.proc.stdin.close()
    except OSError:
        pass
    say(f"INFO: mt.py transcribe in {folder_name} (MT_CONTROL=stdin, MT_EVENTS=1) took {time.monotonic() - started:.1f} s")
    show("stdout", child.stdout())
    say_infra("control", child.stderr())
    return code, child, src, expected, closed.get("running")


def step_control(a):
    r = Report("control")
    work = workdir(a)
    info = load_speech(r, work)
    if info is None:
        return r.end()
    say("INFO: run 1: stdin stays open until mt.py has ended")
    code, child, src, expected, _ = control_run(a, r, info, work, "control", close_on_done=False)
    if code is not None:
        r.check(code == 0, "run 1: exit code 0, ended by itself with stdin still open (the reader did not keep it "
                "alive)", f"run 1: exit code {code}" + (" (SIGTERM's: its control channel saw cancel or the end of "
                                                       "stdin, mt.py:399)" if code in (15, -15) else ""))
    text = check_utf8(r, child.stdout(), "the @@ events (a pipe)")
    events = parse_events(text)
    r.check(any(k == "FILE" and len(f) == 3 and same_path(f[2], src) for k, f in events),
            f"@@FILE names the Cyrillic input: {src}", f"no @@FILE names {src}")
    r.check(any(k == "OUT" and f and same_path(f[0], expected) for k, f in events),
            f"@@OUT names the transcript: {expected}", f"no @@OUT names {expected}")
    lines = [f[0] for k, f in events if k == "LINE" and f]
    bad = []
    for line in lines:
        try:
            json.loads(line)
        except ValueError:
            bad.append(line)
    r.check(not bad, f"{len(lines)} @@LINE events, each one JSON", f"{len(bad)} @@LINE events are not JSON: {bad[:2]}")
    r.check(events and events[-1] == ("DONE", ["0"]), "the last event is @@DONE 0",
            f"the last event is {events[-1] if events else 'missing'}, not @@DONE 0")
    r.check("'pause' does nothing now" in child.stderr(),
            "the line pause on stdin was read and answered on stderr (the control reader works on a Windows pipe)"
            if WIN else "the line pause on stdin was read and answered on stderr",
            "no answer on stderr to the line pause: the control channel did not read stdin")
    check_stderr(r, child.stderr(), "run 1: mt.py's stderr")
    check_content(r, info, check_transcript(r, expected))

    # A front end that closes the pipe as soon as it reads @@DONE (README: "a front end therefore keeps the pipe open
    # until mt.py has ended"; mt.py:3992-3995 makes an end of stdin after the work harmless all the same).
    say("INFO: run 2: stdin is closed as soon as @@DONE is read")
    code, child, _, _, running = control_run(a, r, info, work, "control-close", close_on_done=True)
    if running is None:
        r.fail("run 2: no @@DONE was read, so stdin was never closed on it")
    elif running:
        say("INFO: run 2: mt.py still ran when stdin was closed on @@DONE: the end of stdin after the work was exercised")
    else:
        say("INFO: run 2: mt.py had already ended when @@DONE was read: the end of stdin after the work was not exercised")
        annotate("notice", "win_e2e control: mt.py had ended before stdin was closed on @@DONE, so the end of stdin "
                 "after the work (mt.py:3992-3995) was not exercised in this run")
    if code is not None:
        r.check(code == 0, "run 2: stdin closed on @@DONE, exit code 0 (not 15)", f"run 2: exit code {code}"
                + (" (SIGTERM's: the end of stdin after @@DONE was taken as cancel, mt.py:3992-3995)"
                   if code in (15, -15) else ""))
    events = parse_events(child.stdout().decode("utf-8", "replace"))
    r.check(events and events[-1] == ("DONE", ["0"]), "run 2: the last event is @@DONE 0",
            f"run 2: the last event is {events[-1] if events else 'missing'}, not @@DONE 0")
    check_stderr(r, child.stderr(), "run 2: mt.py's stderr")
    return r.end()


def step_cli(a):
    r = Report("cli")
    work = workdir(a)
    work.mkdir(parents=True, exist_ok=True)
    env, _ = child_env(a, work)
    try:
        done = run_mt(a, ["--help"], env, a.clock.left())
        out = done.stdout.decode("utf-8", "replace")
        r.check(done.returncode == 0 and "transcribe" in out and "devices" in out,
                "mt.py --help: exit code 0, it lists transcribe and devices",
                f"mt.py --help: exit code {done.returncode}: {tail(done.stderr)}")
        version = (Path(a.mt).parent / "VERSION").read_text(encoding="utf-8").strip()
        done = run_mt(a, ["--version"], env, a.clock.left())
        out = done.stdout.decode("utf-8", "replace").strip()
        r.check(done.returncode == 0 and out == f"meeting-transcriber {version}", f"mt.py --version: {out}",
                f"mt.py --version: exit code {done.returncode}, {out!r} (VERSION says {version})")
        done = run_mt(a, ["devices"], env, a.clock.left())
    except subprocess.TimeoutExpired as e:
        r.fail(f"{' '.join(map(str, e.cmd[2:]))} did not end in {e.timeout:.0f} s")
        return r.end()
    show("stdout", done.stdout)
    show("stderr", done.stderr)
    err = done.stderr.decode("utf-8", "replace")
    listed = [l for l in done.stdout.decode("utf-8", "replace").splitlines() if l.strip()]
    say(f"INFO: mt.py devices listed {len(listed)} device(s)")
    r.check(done.returncode == 0, "mt.py devices: exit code 0", f"mt.py devices: exit code {done.returncode}")
    r.check("Traceback" not in err, "mt.py devices: no traceback", "mt.py devices: a traceback on stderr")
    return r.end()


STEPS = {"imports": step_imports, "speech": step_speech, "prepare": step_prepare, "transcribe": step_transcribe,
         "control": step_control, "cli": step_cli}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):              # the log shows Cyrillic paths whatever the code page
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=list(STEPS))
    p.add_argument("work", nargs="?", default=None, help="the work folder (every step but imports)")
    p.add_argument("--mt", default=str(MT), help="mt.py to run (default: the one beside tools/)")
    p.add_argument("--model", default="tiny", help="MT_WHISPER_MODEL for mt.py (default: tiny, about 75 MB)")
    p.add_argument("--budget", type=float, default=None, help="seconds the whole step may take (default: STEP_BUDGET "
                   "of the step, below the workflow step's timeout-minutes)")
    p.add_argument("--retry-wait", type=float, default=30.0, help="seconds before prepare runs once more after a network "
                   "failure (default: 30)")
    p.add_argument("--system-ffmpeg", action="store_true", help="keep the folders on PATH that hold ffmpeg (by default "
                   "they are left out, so that static-ffmpeg fetches it, as on a PC without one)")
    a = p.parse_args(argv)
    if a.step != "imports" and not a.work:
        p.error(f"{a.step} needs the work folder")
    a.clock = Clock(a.budget or STEP_BUDGET[a.step])
    say(f"INFO: {a.step}: Python {sys.version.split()[0]} at {sys.executable}, {sys.platform}, mt.py {a.mt}, "
        f"budget {a.clock.seconds:g} s")
    return STEPS[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
