#!/usr/bin/env python3
"""meeting-transcriber: record or load audio -> multilingual transcript (text file) -> translations (text files)."""
import argparse
import collections
import functools
import hashlib
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import datetime
from types import SimpleNamespace
from difflib import SequenceMatcher
from pathlib import Path

RATE = 16000
CHUNK_S = 30          # Whisper's native window: a chunk never exceeds it
MIN_S = 5             # a chunk closes at the next pause once it is this long (language is detected per chunk)
# How finely speech is cut before each language decision: (a chunk closes at the next pause once it is this many
# seconds long, a pause of this many milliseconds splits speech). Finer follows quick back-and-forth between
# languages, costs more time (every chunk is a full Whisper window) and gives the detector less audio to judge from.
GRANULARITY = {"coarse": (15, 1000), "normal": (MIN_S, 700), "fine": (2, 350), "finest": (1, 200)}
# adaptive: cut like normal, then re-cut a chunk at the finest settings when its language probabilities look mixed (a
# switch inside the chunk): the finer cuts, and their cost, go only where a switch is likely. See decode_chunk.
GRANULARITY["adaptive"] = GRANULARITY["normal"]
AUTO_GRANULARITY = ("adaptive", "finest")  # "auto": adaptive with no spoken language given, finest when several are chosen (one: normal)
LIVE_CLOSE_S = 3      # live: a stretch of speech shorter than the chunk minimum is decoded once this much silence has followed it
# Adaptive thresholds, measured (10-min en/ru/de dialogue + its English-only twin, turbo): Whisper's chunk-level probabilities
# understate a minority language badly (median 0.94 / 0.046 on truly mixed chunks), so the trigger has to be tight: at 0.99 / 0.03
# it fires on 79 % of the mixed chunks and on 1 of 35 single-language ones; at 0.8 / 0.15 it caught 25 % and lost 48 sentences.
SPLIT_CONF = 0.99     # adaptive: a chunk whose best language is below this, ...
SPLIT_SECOND = 0.03   # ... or whose second-best is above this, is re-cut at the finest settings
# Measured (PR0, 10-min three-language dialogue, large-v3-turbo): normal loses 68 of 167 sentences (WER 50 %), fine 18 (21 %),
# finest 4 (11 %); with 0.25 s gaps between speakers only finest tags anything right. finest costs 1.5x the chunks of fine.
# Adaptive, automatic mode (step 0 of 1.9.2): 11 lost, WER 15 %, every tag right, at 0.96x realtime on that dialogue (finest:
# 0.95x) and 0.35x on its English-only twin (normal: 0.35x, fine: 0.56x).


def resolve_granularity(granularity, languages):
    """'auto' -> 'finest' with several spoken languages (they alternate, often within 0.3 s), 'normal' with one (it is forced:
    nothing to detect), else 'adaptive' (detected, re-cut where a chunk sounds mixed); a preset name stays."""
    if granularity in (None, "auto"):
        return AUTO_GRANULARITY[1] if len(languages) > 1 else "normal" if languages else AUTO_GRANULARITY[0]
    if granularity not in GRANULARITY:
        raise ValueError(f"unknown granularity {granularity!r}; use auto, " + ", ".join(GRANULARITY))
    return granularity

LOW_CONF = 0.5        # below this the language tag is shown as a guess: [ru?]
UNLISTED_CONF = 0.7   # a language outside the chosen list wins a chunk when Whisper is at least this sure of it
ETA_AFTER = 5         # chunks decoded before an estimate of the remaining time is given (the per-chunk time varies by ~6 %)
BATCH = 80            # lines per translation request (keeps the reply well inside max_tokens)
DATA = Path(os.environ.get("MT_HOME", Path(__file__).parent / "data"))
VERSION = (Path(__file__).parent / "VERSION").read_text().strip()
WHISPER = os.environ.get("MT_WHISPER_MODEL", "large-v3-turbo")
CLAUDE = os.environ.get("MT_CLAUDE_MODEL", "claude-sonnet-5-5")
# Offline translation: NLLB-200 (Meta, ~200 languages) converted for CTranslate2 by the OpenNMT project. The licence is
# CC-BY-NC 4.0 (non-commercial). MT_NLLB_MODEL can name a smaller one, e.g. JustFrederik/nllb-200-distilled-600M-ct2-int8.
NLLB = os.environ.get("MT_NLLB_MODEL", "OpenNMT/nllb-200-distilled-1.3B-ct2-int8")
# The second offline translator, TranslateGemma 12B on the Apple GPU (see gemma_code); --translator gemma chooses it.
GEMMA = os.environ.get("MT_GEMMA_MODEL", "mlx-community/translategemma-12b-it-4bit")
# The third, Hy-MT3 7B on the Apple GPU (see hy_code); --translator hy chooses it. Unset, MT_HY_MODEL means the pinned and
# checked files of HY_FILES; set, a Hugging Face repo id or a local folder, loaded as it is.
HY_DEFAULT = "netprojects/Hy-MT3-7B-MLX-4bit"
HY = os.environ.get("MT_HY_MODEL") or HY_DEFAULT
TRANSLATOR = os.environ.get("MT_TRANSLATOR", "nllb")     # the offline translator: nllb, gemma or hy
EVENTS = os.environ.get("MT_EVENTS") == "1"   # the Mac app sets this: machine-readable "@@KIND<TAB>fields" lines on stdout
# The output folder (the app's "Output" box): without --out, every result goes to a subfolder of it named after the hour
# the meeting began (`26-11-22 14`: yy-mm-dd hh), the transcript, the translations and the recording's audio together.
OUT = Path(os.environ["MT_OUT"]).expanduser() if os.environ.get("MT_OUT") else None
FFMPEG_DIR = DATA.parent / "ffmpeg"          # where the engine keeps ffmpeg and ffprobe when the system has none
MODELS_DIR = DATA.parent / "models"          # the speaker-separation models (about 47 MB, fetched once)
MAX_HOURS = 8         # the longest recording; the default is 4 h
SILENT_PEAK = 0.002   # a track whose loudest sample stays below this was silent (nothing playing, wrong device)
REOPEN_S = 5          # seconds between attempts to reopen a microphone that stopped
NOTICE_S = 600        # a notice this long before a recording stops at its limit
EXTEND_S = 3600       # what one SIGUSR1 (the app's "Extend") adds to a running recording


# One event line at a time on stdout: the live worker's thread and the main thread both send events, and print() writes
# a line and its newline separately, so two of them at once could glue ('@@CLEAR@@LEVEL\t0.25', which the app reads as
# text and loses the CLEAR). ev() writes the whole line in one write(), which is what fixes that on CPython's own stdout;
# the lock keeps a line whole on a stdout that writes in pieces too (a wrapper, a free-threaded build), as the test
# test_events_of_two_threads_never_glue_into_one_line checks. Reentrant: a signal handler runs on the main thread,
# possibly while it is inside ev().
EV_LOCK = threading.RLock()


def ev(kind, *fields):
    if kind == "DONE":
        # The work is done: from here on an end of stdin (a front end that closes the pipe on @@DONE) or a late stop or
        # cancel on the control channel changes nothing, as main()'s finally makes it once run() returns. Put in place
        # before the line is printed, so that a close answering it cannot come first and end the process as SIGTERM.
        CONTROL.listen({verb: (lambda: None) for verb in (*CONTROL_VERBS, "eof")})
    clean = [re.sub(r"[\t\r\n]", " ", str(f)) for f in fields]   # only these split a line; other spaces (U+202F ...) are kept
    line = "@@" + "\t".join([kind, *clean]) + "\n"
    with EV_LOCK:                                      # the whole line in one write, then flushed (see EV_LOCK)
        sys.stdout.write(line)
        sys.stdout.flush()


def note(text):
    """Something the user should see at once (not just in a log): an event for the app, a stderr line otherwise."""
    if EVENTS:
        ev("NOTE", text)
    else:
        print(text, file=sys.stderr)


def check_name(name, what="name"):
    """A recording name or stem becomes a folder and file name, so it cannot hold path separators. On Windows it must also
    be a name Windows accepts (windows_name_problem); elsewhere the rule stays this one, the same as the Mac app's
    (JobSettings.swift validationProblems)."""
    if name and re.search(r"[/\\:\x00]", name):
        sys.exit(f"The {what} {name!r} cannot contain / \\ or : (it becomes a file name). Nothing was started.")
    problem = windows_name_problem(name) if name and sys.platform == "win32" else None
    if problem:
        sys.exit(f"The {what} {name!r} {problem} (it becomes a folder and file name). Nothing was started.")
    return name


# ---------- Windows ----------
# The longest paths mt.py builds under the data folder are an offline translation's log,
#   <DATA>\<yyyymmdd-hhmmss>-<name>\translations\<NLLB code>-<model, / as _>-<16 hex>.jsonl
# (len(DATA) + len(name) + 103 characters with the default NLLB model), and a recording's translation, which stays in
# its data folder without --out or MT_OUT,
#   <DATA>\<yyyymmdd-hhmmss>-<name>\<name>.<Language, as a file name>.txt
# (len(DATA) + 2 * len(name) + len(Language) + 23). Without long paths switched on, Windows allows 259 (MAX_PATH is 260
# with the closing NUL). A name of at most 64 with --to "Brazilian Portuguese" (20) leaves 88 for the data folder: the
# app's %LOCALAPPDATA%\Meeting Transcriber\data with a 20-character user name is 68, this folder's data\ under
# C:\Users\<20 characters>\mini-projs about 65. Any other case (a longer data folder, --to or model) is refused before
# anything starts (windows_path_problem). Counted in UTF-16 units, as Windows counts.
WIN_NAME_MAX = 64
# Device names Windows reserves in every folder, also with an extension ("NUL.txt" is NUL); ¹ ² ³ count as digits there
# (learn.microsoft.com/windows/win32/fileio/naming-a-file).
WIN_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL"} | {f"{port}{n}" for port in ("COM", "LPT") for n in "123456789¹²³"})


def utf16_len(text):
    return len(text.encode("utf-16-le")) // 2


def windows_name_problem(name):
    """What makes `name` unusable as a Windows file or folder name (as words for the message), or None: the characters
    Windows reserves (< > " | ? * and control characters; / \\ : are refused everywhere already), a dot or space at the
    end (Explorer and most programs cannot open such a folder), a reserved device name, more than WIN_NAME_MAX."""
    if re.search(r'[<>"|?*\x00-\x1f]', name):
        return 'cannot contain < > " | ? * or control characters on Windows'
    if name[-1] in ". ":
        return "cannot end with a dot or a space on Windows"
    if name.split(".")[0].rstrip(" ").upper() in WIN_RESERVED:
        return "is a device name on Windows (CON, PRN, AUX, NUL, COM1 to COM9, LPT1 to LPT9, also with an extension)"
    if utf16_len(name) > WIN_NAME_MAX:
        return (f"is {utf16_len(name)} characters long; on Windows at most {WIN_NAME_MAX} fit (a path there is limited "
                "to 260 characters)")
    return None


def windows_folder_name(name):
    """A data folder's name part on Windows: cut to WIN_NAME_MAX (an input file's own name can be 255 characters, which the
    data folder's path would not hold) without a dot or space at the end. Elsewhere the name as it is."""
    if sys.platform != "win32" or not name:
        return name
    while utf16_len(name) > WIN_NAME_MAX:
        name = name[:-1]
    return name.rstrip(". ") or "x"


def child_options():
    """Keyword arguments for every child process mt.py starts (ffprobe, ffmpeg, sysctl, cp, caffeinate): no stdin, so a
    child never reads the control channel's commands (MT_CONTROL=stdin), and on Windows no console window of its own
    (started from a window app, which has no console, each console program would open one). Nothing a child of mt.py
    does reads stdin: ffmpeg runs with -nostdin, and the Mac app gives mt.py /dev/null as stdin anyway."""
    options = {"stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return options


def replace_file(src, dst):
    """os.replace, the last step of every save that is whole or absent. On Windows it fails with PermissionError while
    another program holds the target open for a moment (Defender, the search indexer, a sync client): there it is tried
    5 times, 50 to 400 ms apart (0.75 s in all), and a lasting refusal is said, then raised. Elsewhere plain os.replace."""
    if sys.platform != "win32":
        return os.replace(src, dst)
    for wait in (0.05, 0.1, 0.2, 0.4, None):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if wait is None:
                windows_refused("replace", dst)
                raise
            time.sleep(wait)


def windows_refused(what, path):
    """The note for a PermissionError on Windows: who may be holding the file, and Controlled folder access, which
    protects Documents, Desktop and Pictures (where the output folder usually is) from programs it does not know."""
    note(f"Windows refused to {what} {path}: another program keeps it open, or Controlled folder access blocks this "
         "program (Windows Security > Virus & threat protection > Ransomware protection > Allow an app through Controlled "
         "folder access)")


def write_output(path, text):
    """A text file of the output folder, in UTF-8. On Windows a refusal is said (windows_refused), then raised; elsewhere
    path.write_text as it was."""
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        if sys.platform == "win32":
            windows_refused("write", path)
        raise


def keep_awake(cmd):
    """Keeps the computer awake while a long command runs (not a closed lid): caffeinate on macOS, on Windows the
    execution state of the main thread (ES_CONTINUOUS | ES_SYSTEM_REQUIRED, cleared at exit; the system also clears it
    when the process ends). A failure changes nothing else."""
    if cmd not in LONG_COMMANDS:
        return
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **child_options())
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import atexit
            import ctypes
            state = ctypes.windll.kernel32.SetThreadExecutionState
            state.argtypes, state.restype = [ctypes.c_uint32], ctypes.c_uint32
            state(0x80000000 | 0x00000001)               # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            atexit.register(state, 0x80000000)           # ES_CONTINUOUS alone: idle sleep is allowed again
        except Exception:
            pass


def total_memory():
    """The computer's memory in bytes, or None: sysctl on macOS, GlobalMemoryStatusEx on Windows (which has no
    os.sysconf), else os.sysconf."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5,
                                 **child_options()).stdout.strip()
            return int(out) if out.isdigit() else None
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):          # MEMORYSTATUSEX: two DWORDs, then seven DWORDLONGs (64 bytes)
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                    (field, ctypes.c_uint64) for field in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                           "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                           "ullAvailExtendedVirtual")]
            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            return int(status.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else None
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None


def wasapi_api(sd):
    """On Windows the index of the WASAPI host API, else None. Windows lists every device once per host API (MME, DirectSound,
    WASAPI, WDM-KS; MME cuts names to 31 characters), so a name there would match three or four inputs: devices are
    matched among the WASAPI ones only (the current endpoints with their full names)."""
    if sys.platform != "win32":
        return None
    try:
        for k, api in enumerate(sd.query_hostapis()):
            if "wasapi" in str(api.get("name", "")).lower():
                return k
    except Exception:
        pass
    return None


def stream_options(sd, device):
    """Extra arguments for sd.InputStream: on Windows a WASAPI device is opened with auto_convert, which lets Windows' own
    mixer convert its shared-mode format (often 48 kHz stereo) to the 16 kHz mono recorded here; without it such an open
    fails. The default input (None) is PortAudio's default, an MME device, which converts by itself. Elsewhere none."""
    if sys.platform != "win32" or device is None:
        return {}
    try:
        api = wasapi_api(sd)
        if api is not None and sd.query_devices(device).get("hostapi") == api:
            return {"extra_settings": sd.WasapiSettings(auto_convert=True)}
    except Exception:
        pass
    return {}


def windows_text(windows, other):
    """A message in its Windows wording on Windows, as it was elsewhere."""
    return windows if sys.platform == "win32" else other


def ffmpeg_hint():
    """How to install ffmpeg when the engine could not fetch it (winget's package of the gyan.dev builds on Windows)."""
    return windows_text("Windows: winget install --id Gyan.FFmpeg -e", "macOS: brew install ffmpeg")


# ---------- the control channel (MT_CONTROL=stdin) ----------
# Windows has no SIGUSR1 or SIGUSR2, and there os.kill() with SIGINT or SIGTERM is TerminateProcess: nothing is cleaned up,
# the WAV header stays up to 30 s stale, no @@DONE comes. With MT_CONTROL=stdin a front end writes one command per line
# to mt.py's stdin instead (case and surrounding spaces do not matter; any other line is a warning on stderr):
#   pause    a running recording pauses (the app's Pause, SIGUSR2); a pause while paused does nothing
#   resume   a paused recording goes on (SIGUSR2 again); a resume while recording does nothing
#   extend   a recording's limit grows by EXTEND_S (SIGUSR1)
#   stop     a recording ends as on Ctrl+C (SIGINT): the files are closed and saved, then transcribed if that was asked;
#            follow finishes its current chunk and ends; any other command gets SIGINT's KeyboardInterrupt
#   cancel   a recording ends as on SIGTERM: the files are kept, nothing is transcribed; follow as on stop; any other
#            command (transcribe, call, translate) ends at once as from SIGTERM, which it has no handler for, after a
#            note naming the cause: what is saved whole-or-absent stays so, and transcribe/call --into goes on from the
#            checkpoint
# The end of stdin (the front end closed the pipe, or died) counts as stop while a recording or follow runs, as cancel
# otherwise: a crashed window still leaves a saved recording and no orphan job. A Windows console has no end that can
# be typed (a line of Ctrl+Z comes back as an empty read): there type stop or press Ctrl+C. In a console the line typed
# shows only once Enter is pressed (see stdin_reader). The record loop and follow act on these at their next turn (0.5 s
# and 1 s); the reader thread itself only sets their flags, or does what the signal's default would. Without MT_CONTROL
# nothing reads stdin: the Mac app's /dev/null means nothing, and it keeps using the signals, which work as before
# either way.
CONTROL_VERBS = ("pause", "resume", "extend", "stop", "cancel")
LONG_COMMANDS = ("record", "call", "transcribe", "translate", "follow")    # kept awake, and read the control channel


def control_wanted():
    return os.environ.get("MT_CONTROL", "").strip().lower() == "stdin"


# Windows errors after which a pipe gives nothing more, so the end of stdin: ERROR_BROKEN_PIPE (the writer closed it and
# all was read), ERROR_BAD_PIPE and ERROR_PIPE_NOT_CONNECTED (a named pipe whose other end disconnected: npfs's NpPeek
# returns STATUS_INVALID_PIPE_STATE or STATUS_PIPE_DISCONNECTED, ReactOS drivers/filesystems/npfs/fsctrl.c), ERROR_NO_DATA
# (being closed), ERROR_INVALID_HANDLE (stdin itself closed).
PIPE_ENDED = (109, 230, 233, 232, 6)


def stdin_reader(fd):
    """A function that returns the next bytes of fd, b"" at its end (the control channel's reader thread calls it).

    Elsewhere a plain os.read. On Windows no read is left waiting on a pipe or a console (two exceptions, marked in the
    code: a "pipe" that cannot be peeked at, and 4096 console keys without a line end). Run 45796780693 (windows-latest,
    "Engine on Windows"): mt.py transcribe with MT_CONTROL=stdin and an open stdin pipe stopped after @@FOLDER and never
    decoded while the reader thread sat in os.read on stdin; without MT_CONTROL the same command took 3.7 s. Run
    45805995625 (windows-latest, the report test in tests/test_windows.py) showed what waits behind a read left pending
    on a stdin pipe: a child process that inherits that stdin (it hung 10 s and came back at once when stdin was
    closed), while os.fstat(0), os.isatty(0), sys.stdin.isatty() and a child started with child_options() (stdin NUL)
    came back at once. So mt.py's own children were not the cause (they all get child_options()); which library starts
    one that inherits stdin is not known. Windows serializes the I/O on a synchronous handle, which stdin is
    (PeekNamedPipe's own page: it "can block thread execution the same way any I/O function can when called on a
    synchronous handle in a multi-threaded application"; CPython issue 78961; libuv src/win/pipe.c). So a pipe is polled
    with PeekNamedPipe and read only for the bytes already in it; a console is read only once a line's end is in its
    input buffer (PeekConsoleInputW); anything else (a file, the NUL device) is read directly, which never waits. The
    cost in a console: it echoes what is typed only while a read takes it in (ENABLE_ECHO_INPUT, SetConsoleMode's page),
    so the line typed shows when Enter is pressed. With a read pending on a console, os.fstat, os.isatty and a
    child_options() child came back at once in the same run; a child inheriting the console was not tried, so the
    console is still read this way."""
    def plain():
        try:
            return os.read(fd, 4096)
        except OSError:
            return b""
    if sys.platform != "win32":
        return plain
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)      # its own function objects: argtypes stay local
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(fd))
        k32.GetFileType.argtypes, k32.GetFileType.restype = [wintypes.HANDLE], wintypes.DWORD
        k32.GetConsoleMode.argtypes, k32.GetConsoleMode.restype = [wintypes.HANDLE, wintypes.LPDWORD], wintypes.BOOL
        kind, mode = k32.GetFileType(handle), wintypes.DWORD()
        console = kind == 2 and bool(k32.GetConsoleMode(handle, ctypes.byref(mode)))   # FILE_TYPE_CHAR: or NUL
    except Exception:
        return plain
    if kind == 3:                                   # FILE_TYPE_PIPE: an anonymous or a named pipe, or a socket
        k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD,
                                      wintypes.LPDWORD, wintypes.LPDWORD]
        k32.PeekNamedPipe.restype = wintypes.BOOL
        available = wintypes.DWORD()
        peeked = warned = plainly = False

        def pipe():
            nonlocal peeked, warned, plainly
            errors = 0
            while not plainly:
                if k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
                    peeked, errors = True, 0
                    if available.value:
                        try:
                            return os.read(fd, min(available.value, 65536))    # there: returns at once
                        except OSError:
                            return b""
                    time.sleep(0.05)
                    continue
                error = ctypes.get_last_error()
                if error in PIPE_ENDED:
                    return b""
                if not peeked:
                    # Not one peek worked: no pipe npfs serves (GetFileType calls a socket a pipe too, and PeekNamedPipe
                    # is not expected to work on one). Read as before, a plain read left waiting.
                    plainly = True
                    break
                # Any other error is not taken for the end, which would cancel a transcription: it may pass (ReactOS's
                # PeekNamedPipe allocates on every call and fails with ERROR_NOT_ENOUGH_MEMORY when that fails,
                # dll/win32/kernel32/client/file/npipe.c). Looked at again, ever more slowly (up to 1 s), and said once.
                errors += 1
                if not warned:
                    warned = True
                    print(f"warning: MT_CONTROL=stdin: Windows error {error} looking at the stdin pipe; still listening",
                          file=sys.stderr, flush=True)
                time.sleep(min(0.05 * errors, 1.0))
            return plain()
        return pipe
    if not console:
        return plain

    class KeyEvent(ctypes.Structure):
        _fields_ = [("down", wintypes.BOOL), ("repeat", wintypes.WORD), ("key", wintypes.WORD),
                    ("scan", wintypes.WORD), ("char", wintypes.WCHAR), ("state", wintypes.DWORD)]

    class Record(ctypes.Structure):                               # INPUT_RECORD; KEY_EVENT_RECORD is its largest member
        _fields_ = [("type", wintypes.WORD), ("event", KeyEvent)]

    k32.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
    k32.GetNumberOfConsoleInputEvents.restype = wintypes.BOOL
    for name in ("PeekConsoleInputW", "ReadConsoleInputW"):
        getattr(k32, name).argtypes = [wintypes.HANDLE, ctypes.POINTER(Record), wintypes.DWORD, wintypes.LPDWORD]
        getattr(k32, name).restype = wintypes.BOOL
    count, got, window = wintypes.DWORD(), wintypes.DWORD(), 4096

    def line_waiting():
        """Whether a line's end is in the console's input buffer (True), not yet (False), or the buffer cannot be looked
        at (None). Drops a buffer of nothing but other events (focus, mouse, size), which a line read would skip anyway,
        so that it does not grow while nobody types."""
        if not k32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(count)):
            return None
        if not count.value:
            return False
        records = (Record * min(count.value, window))()
        if not k32.PeekConsoleInputW(handle, records, len(records), ctypes.byref(got)):
            return None
        seen = records[:got.value]
        # A cooked read ends on the character \r alone (microsoft/terminal src/host/readDataCooked.cpp,
        # UNICODE_CARRIAGERETURN), not on the key: a VK_RETURN whose character is \n (Ctrl+Enter's, it seems), which
        # conhost skips (src/host/stream.cpp, GetChar), ends none, and a read started on it would wait for Enter.
        if any(r.type == 1 and r.event.down and r.event.char == "\r" for r in seen):
            return True
        keys = any(r.type == 1 for r in seen)
        if keys and got.value == window:
            # A full window of keys and no line end in it (some 2000 characters typed or pasted without Enter): read
            # anyway, or a line end beyond it would never be seen. The read reaches that at once, or waits for Enter.
            return True
        if seen and not keys:
            k32.ReadConsoleInputW(handle, records, got.value, ctypes.byref(got))
        return False

    def console_line():
        empty = failed = 0
        while True:
            waiting = line_waiting()
            failed = failed + 1 if waiting is None else 0
            if failed >= 1000:
                # 100 s in a row the buffer could not be looked at: the console is gone (a console has no end of its
                # own; a pipe's end comes as one of PIPE_ENDED). As the reader before did after 1000 failed reads.
                return b""
            if waiting:
                try:
                    data = os.read(fd, 4096)                       # a line is complete: returns at once
                except OSError:
                    data = b""
                if data:
                    return data
                # A line that starts with Ctrl+Z comes back empty (conhost sets the count to 0: microsoft/terminal,
                # src/server/ApiDispatchers.cpp, ProcessControlZ): not the end of a console, which is read on.
                empty += 1
                if empty >= 1000:
                    return b""
            time.sleep(0.1)
    return console_line


class Control:
    """The control channel's reader: a daemon thread turns stdin lines into calls of the handlers the running phase
    registered with listen() (record() and follow(); their handlers only set flags), else into what the signal does."""
    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.handlers = {}
        self.thread = None

    def listen(self, handlers):
        """Puts `handlers` ({verb or "eof": callable}) in place and returns the ones before, to be put back the same way."""
        with self.lock:
            before, self.handlers = self.handlers, dict(handlers)
        return before

    def start(self, fd=None):
        """Starts reading `fd` (stdin by default) once."""
        import threading
        if self.thread is not None:
            return
        if fd is None:
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, OSError, ValueError):     # pythonw.exe has no stdin at all
                print("warning: MT_CONTROL=stdin, but this process has no stdin to read commands from", file=sys.stderr)
                return
        self.thread = threading.Thread(target=self._read, args=(fd,), daemon=True, name="mt-control")
        self.thread.start()

    def _read(self, fd):
        # os.read on the descriptor, not sys.stdin: a daemon thread blocked in a buffered read holds that reader's lock,
        # and the interpreter then aborts at exit ("could not acquire lock for <_io.BufferedReader name='<stdin>'> at
        # interpreter shutdown", exit status 134: tried on Linux with Python 3.12).
        # On Windows stdin_reader leaves no read waiting on a pipe or a console (see there).
        read = stdin_reader(fd)
        pending = b""
        while True:
            data = read()
            if not data:
                break
            *lines, pending = (pending + data).split(b"\n")
            for raw in lines:
                self.command(raw.decode("utf-8", "replace"))
        if pending.strip():                                   # a last line without a line end still counts
            self.command(pending.decode("utf-8", "replace"))
        self.dispatch("eof")

    def command(self, line):
        verb = line.strip().lstrip("\ufeff").strip().lower()    # a UTF-8 BOM in front of the first line (a .NET writer)
        if not verb:
            return
        if verb not in CONTROL_VERBS:
            print(f"warning: unknown control command {line.strip()!r} ignored; the commands are "
                  + ", ".join(CONTROL_VERBS), file=sys.stderr, flush=True)
            return
        self.dispatch(verb)

    def dispatch(self, verb):
        with self.lock:
            handler = self.handlers.get(verb)
        if handler is not None:
            handler()
        elif verb == "stop":
            import _thread
            _thread.interrupt_main()                          # SIGINT's effect: KeyboardInterrupt in the main thread
        elif verb in ("cancel", "eof"):
            # Said first: the signal's end leaves no trace, and a front end that forgot to give stdin a pipe (NUL, a
            # closed one) would otherwise see only exit code 15.
            note(("cancel on stdin" if verb == "cancel" else "stdin ended") + " (MT_CONTROL=stdin): mt.py ends at once, "
                 "as on SIGTERM")
            self.terminate()
        else:                                                 # SIGUSR1/SIGUSR2 are ignored outside record() too
            print(f"warning: {verb!r} does nothing now: only a running recording can be paused, resumed or extended",
                  file=sys.stderr, flush=True)

    @staticmethod
    def terminate():
        """What SIGTERM does to a command that has no handler for it: the process ends at once, by that signal (on
        Windows os.kill of oneself is TerminateProcess with exit code 15, what a front end's os.kill(pid, SIGTERM) gives).
        What was printed is flushed first; nothing else runs, exactly as with the signal."""
        import signal
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        os.kill(os.getpid(), signal.SIGTERM)


CONTROL = Control()


# ---------- storage ----------
def _dir(rec_id):
    if rec_id in (None, "latest"):
        runs = sorted(p for p in DATA.glob("*/transcript.json"))
        if not runs:
            sys.exit("No transcripts yet.")
        return runs[-1].parent
    d = DATA / rec_id
    if not (d / "transcript.json").exists():
        sys.exit(f"No transcript '{rec_id}'. Try: mt.py list")
    return d


def _load(d):
    return json.loads((d / "transcript.json").read_text(encoding="utf-8"))


def _save(d, doc):
    """transcript.json, written whole or not at all (a crash in the middle never leaves a half file)."""
    tmp = d / "transcript.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    replace_file(tmp, d / "transcript.json")


class RecordingCancelled(Exception):
    """SIGTERM/SIGHUP during a recording (the app's Cancel): the files are closed and kept, nothing is transcribed.
    `folder` is where they are, set by record() before it re-raises (None when the signal came before the folder existed)."""
    folder = None


class WavFile:
    """A mono 16 kHz PCM16 WAV written as it arrives, whose header is patched on every flush: a recording that is cut short
    (kill, crash, a dying computer) is still a valid file up to the last flush."""
    def __init__(self, path, rate=RATE):
        self.path, self.rate, self.samples = Path(path), rate, 0
        self.f = open(self.path, "wb")
        self.f.write(self.header(0))

    def header(self, samples):
        data = samples * 2
        return (b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, self.rate, self.rate * 2, 2, 16)
                + b"data" + struct.pack("<I", data))

    def write(self, block):
        import numpy as np
        data = np.ascontiguousarray(block, dtype="<i2").tobytes()
        self.f.write(data)
        self.samples += len(data) // 2

    def flush(self):
        self.f.flush()
        pos = self.f.tell()
        self.f.seek(0)
        self.f.write(self.header(self.samples))
        self.f.seek(pos)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        if not self.f.closed:
            self.flush()
            self.f.close()


# ---------- record ----------
def _limit_text(seconds):
    return f"{seconds / 3600:g} h" if seconds % 3600 == 0 else f"{seconds / 60:g} min"


def recording_controls(limit):
    """How a running recording is stopped, extended and paused, for the line it starts with: the signals on macOS and
    Linux (as before), the control channel's lines on Windows, which has no SIGUSR1/SIGUSR2."""
    stops = f"stops by itself after {_limit_text(limit)}"
    typed = "type pause, resume, extend (an hour more), stop or cancel and press Enter"
    if sys.platform == "win32":
        if control_wanted():
            return f"{stops}; {typed}; Ctrl+C also stops"
        return f"Ctrl+C to stop, {stops}; to pause or extend it, start mt.py with MT_CONTROL=stdin and type the commands"
    pid = os.getpid()
    text = f"Ctrl+C to stop, {stops}; kill -USR1 {pid} adds an hour, kill -USR2 {pid} pauses and resumes"
    return text + (f"; or {typed}" if control_wanted() else "")


def pause_note(recorded_s, paused_s):
    """The note for a pause: where in the recording it was (a timed note, placed there in the transcript) and how long."""
    paused = int(round(paused_s))
    length = f"{paused // 60} min {paused % 60} s" if paused >= 60 else f"{paused} s"
    return f"[{hms(recorded_s)}] Paused here for {length}"


def resolve_device(spec, what="input"):
    """Turns `--device` / `--system-device` into what sounddevice takes: an index as it is (checked to be an input), a name by
    an exact match among the input devices (the same case first, then any case), else a part of a name that fits exactly
    one of them (so `--device "MacBook Pro Microphone"` and `--device jabra` both work); `name#k` is the k-th input of
    exactly that name, for two identical devices (the app sends this when the chosen microphone has a twin). None stays
    None (the default input). Anything that does not name exactly one device stops here with the list, before the
    recording folder is made. On Windows only the WASAPI host API's inputs count (wasapi_api), indexes too."""
    if spec is None:
        return None
    import sounddevice as sd
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]
    api = wasapi_api(sd)
    if api is not None:
        devices = [(i, d) for i, d in devices if d.get("hostapi") == api]
    listing = "\n".join(f"  {i:>3}  {d['name']}" for i, d in devices) or "  (none)"
    text = str(spec).strip()
    if text.isdigit():
        if int(text) not in {i for i, _ in devices}:
            sys.exit(f"{text} is not an input device. The input devices are (index  name):\n{listing}\nNothing was started.")
        return int(text)
    twin = re.fullmatch(r"(.+)#(\d+)", text)
    if twin and int(twin.group(2)) >= 1:
        same = [i for i, d in devices if d["name"].strip() == twin.group(1).strip()]
        if len(same) >= int(twin.group(2)):
            return same[int(twin.group(2)) - 1]
        sys.exit(f"{text!r} asks for input number {twin.group(2)} of the name {twin.group(1).strip()!r}, but {len(same)} "
                 f"input{'s' if len(same) != 1 else ''} of that name {'are' if len(same) != 1 else 'is'} connected (index  name):\n{listing}\nNothing was started.")
    same = [i for i, d in devices if d["name"].strip() == text]
    if len(same) == 1:
        return same[0]
    exact = same or [i for i, d in devices if d["name"].strip().lower() == text.lower()]
    if len(exact) == 1:
        return exact[0]
    partial = [i for i, d in devices if text.lower() in d["name"].lower()]
    if len(partial) == 1:
        return partial[0]
    if not exact and not partial:
        sys.exit(f"No {what} device is called {text!r}. The input devices are (index  name):\n{listing}\nNothing was started.")
    found = exact or partial
    sys.exit(f"{len(found)} input devices match {text!r}; give the index instead, or NAME#2 for the second input of a name (index  name):\n"
             + "\n".join(f"  {i:>3}  {dict(devices)[i]['name']}" for i in found) + "\nNothing was started.")


def record(name, hours, device, system_device=None, live=None, notes=None, check=None):
    """Records the microphone (device) to mic.wav, and with system_device (a loopback or monitor input that carries what
    the computer plays: the other side of a call) also to system.wav, both from the same moment. Returns (folder, mic, system).
    The recording stops after `hours` of recorded audio (not wall-clock time: a sleeping computer does not use up the
    limit); NOTICE_S before that a notice is given, and SIGUSR1 (the app's "Extend") adds EXTEND_S. SIGUSR2 (the app's
    "Pause") pauses the recording and, sent again, resumes it: nothing is written meanwhile (the recording and its
    transcript have one timeline without the pause) and a timed note says where and for how long. `live`, when given,
    is a dict of transcription settings (language, granularity, me, them): speech is transcribed while the recording
    goes on, so that only the tail is left to do at the end. `notes`, when given, collects every note the recording gave,
    for the transcription that follows. With MT_CONTROL=stdin the lines pause, resume, extend, stop and cancel do what
    SIGUSR2, SIGUSR1, SIGINT and SIGTERM do (see Control); the signals keep working beside them. `check` goes to _new_dir
    (main's Windows path check, for a folder numbered -2 …)."""
    import signal
    extensions = [0]
    gate = [False]                             # closed while paused: the capture callbacks drop their blocks at once
    toggles = []                               # when each pause/resume came, for the bookkeeping in the loop
    collected = notes if notes is not None else []

    def tell(text):
        collected.append(text)
        note(text)

    def on_extend(*_):
        extensions[0] += 1

    def on_pause(*_):
        gate[0] = not gate[0]                  # takes effect in the next audio callback: nothing said after Pause is queued
        toggles.append(time.monotonic())

    def capture(q):
        return lambda x, *_, q=q: None if gate[0] else q.put(x.copy())

    def on_cancel(signum, _):
        raise RecordingCancelled(signal.Signals(signum).name)

    # The control channel (MT_CONTROL=stdin; see Control): its reader thread only sets what the loop below acts on at its
    # next turn. Pause and resume close and open the gate at once, as the SIGUSR2 handler does, but say which: a second
    # pause is not a resume. Extend, stop and cancel wait in `commands` for the loop (stop leaves it the way Ctrl+C does,
    # cancel raises RecordingCancelled there); the end of stdin is a stop.
    commands = []

    def set_gate(closed):
        def act():
            if gate[0] != closed:
                gate[0] = closed
                toggles.append(time.monotonic())
        return act
    # The handlers go in before the slow imports: an "Extend" sent while the recorder is still starting must count, not kill it
    # (main() ignores SIGUSR1 even earlier), and a Cancel must save what there is.
    previous = {}
    for sig_name, handler in (("SIGUSR1", on_extend), ("SIGUSR2", on_pause), ("SIGTERM", on_cancel), ("SIGHUP", on_cancel)):
        if hasattr(signal, sig_name):
            previous[sig_name] = signal.signal(getattr(signal, sig_name), handler)
    previous_control = CONTROL.listen({"pause": set_gate(True), "resume": set_gate(False),
                                       "extend": lambda: commands.append("extend"), "stop": lambda: commands.append("stop"),
                                       "cancel": lambda: commands.append("cancel"), "eof": lambda: commands.append("stop")})
    cancelled = None
    import sounddevice as sd

    def release():                                 # before the folder exists: the handlers back as the caller had them
        for sig_name, handler in previous.items():
            signal.signal(getattr(signal, sig_name), handler)
        CONTROL.listen(previous_control)

    if "cancel" in commands:                       # cancelled while starting, before the folder exists (as SIGTERM then)
        release()
        raise RecordingCancelled("cancel on stdin")
    try:
        d = _new_dir(name, check)                  # another recording of that name in the same second takes -2
    except BaseException:                          # no folder (refused by `check`, or DATA unusable): nothing was recorded
        release()
        raise
    if EVENTS:
        ev("FOLDER", d)                            # the app keeps a folder being recorded out of its lists
    mic_wav = d / ("mic.wav" if system_device is not None else "audio.wav")
    sys_wav = d / "system.wav" if system_device is not None else None
    last_sync = time.time()
    tracks = [(mic_wav, device)] + ([(sys_wav, system_device)] if sys_wav else [])
    queues = [queue.Queue() for _ in tracks]
    limit = int(round(hours * 3600))         # recorded seconds
    written = [0] * len(tracks)              # samples written per track; the recording's clock is the longer one
    noticed = False
    peaks, level_sq, level_n, last_level = [0.0] * len(tracks), 0.0, 0, time.monotonic()   # the loudest sample per track
    t0 = time.monotonic()
    last_block = [t0] * len(tracks)          # when each track last delivered sound
    stalled, reopened, next_reopen = False, 0, 0.0
    paused, pause_began, paused_total = False, 0.0, 0.0   # wall seconds spent paused: taken out of every wall-clock comparison
    print(f"Recording to {d} ({recording_controls(limit)})", file=sys.stderr)
    files, streams, worker = [], [], None
    if live is not None:
        allowed = parse_languages(live.get("language"))
        names = [live.get("me", "You"), live.get("them", "Them")] if sys_wav else [None]
        live_tracks = [LiveTrack(str(wav), d, f".{k}" if sys_wav else "", wav, allowed, live.get("granularity", "auto"),
                                 speaker_name(names[k - 1], ("You", "Them")[k - 1]) if sys_wav else None, engine_name())
                       for k, (wav, _) in enumerate(tracks, 1)]
        worker = LiveWorker(live_tracks, targets=live.get("targets") or (), offline=live.get("offline", False))
        worker.start()
    try:
        for (wav, dev), q in zip(tracks, queues):
            files.append(WavFile(wav))
            streams.append(sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=dev, callback=capture(q),
                                          **stream_options(sd, dev)))
        for stream in streams:
            stream.start()
        while max(written) < limit * RATE:
            time.sleep(0.5)
            now = time.monotonic()
            while toggles:                             # "Pause" / "Resume" (SIGUSR2): the gate closed or opened at that moment
                at = toggles.pop(0)
                if not paused:
                    paused, pause_began = True, at
                    print("paused", file=sys.stderr)
                else:
                    paused = False
                    paused_total += at - pause_began
                    last_block = [now] * len(tracks)   # the stall check starts afresh: nothing was expected meanwhile
                    tell(pause_note(max(written) / RATE, at - pause_began))
                    print("recording", file=sys.stderr)
            stopping = False
            while commands:                            # the control channel's extend, stop and cancel (MT_CONTROL=stdin)
                verb = commands.pop(0)
                if verb == "extend":
                    extensions[0] += 1                 # counted here, on the loop's thread: applied below, as SIGUSR1's
                elif verb == "cancel":
                    raise RecordingCancelled("cancel on stdin")
                else:
                    stopping = True
            if stopping:
                break                                  # as Ctrl+C: the finally below closes, drains and saves
            arrived = [False] * len(tracks)
            for i, (f, q) in enumerate(zip(files, queues)):
                while not q.empty():                   # only what the open gate let through (nothing from the pause)
                    block = q.get()
                    f.write(block)
                    if worker:
                        worker.tracks[i].feed(block)
                    arrived[i] = True
                    written[i] += len(block)
                    import numpy as np
                    samples = np.asarray(block, dtype="float32") / 32768.0
                    peaks[i] = max(peaks[i], float(np.max(np.abs(samples))) if samples.size else 0.0)
                    if i == 0:
                        level_sq += float(np.sum(samples * samples)); level_n += samples.size
            if paused:
                if EVENTS and now - last_level >= 1:
                    ev("LEVEL", 0.0)
                    level_sq, level_n, last_level = 0.0, 0, now
                continue
            now = time.monotonic()
            for i in range(len(tracks)):
                if arrived[i]:
                    last_block[i] = now
            if arrived[0]:
                stalled = False
            if EVENTS and now - last_level >= 1 and level_n:
                ev("LEVEL", round(min(1.0, (level_sq / level_n) ** 0.5 * 4), 3))      # a speaking voice reaches about 0.25 RMS
                level_sq, level_n, last_level = 0.0, 0, now
            # A microphone that stops delivering (unplugged, the headset that was the input gone): say so and try to reopen
            # it a few times, REOPEN_S apart, whether or not anything arrives meanwhile.
            quiet = now - last_block[0]
            if not arrived[0] and quiet > 5:
                if not stalled:
                    stalled = True
                    tell(f"No sound has come from the microphone for {int(quiet)} s" + (": trying to reopen it" if reopened < 3 else ""))
                if reopened < 3 and now >= next_reopen:
                    reopened += 1
                    next_reopen = now + REOPEN_S
                    try:
                        streams[0].stop(); streams[0].close()
                        streams[0] = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", device=device,
                                                    callback=capture(queues[0]), **stream_options(sd, device))
                        streams[0].start()
                    except Exception as e:
                        tell(f"The microphone could not be reopened: {e}" + (f"; trying again in {REOPEN_S} s" if reopened < 3 else ""))
            # Each track's clock is the samples written to it. When the wall clock has run far ahead of a track while nothing
            # arrived on it (the computer slept, that device went away), that track's gap is filled with silence, so the
            # timelines stay right; a track that goes on delivering is not touched.
            for i, f in enumerate(files):
                gap = (now - t0 - paused_total) * RATE - written[i]
                if not arrived[i] and gap > 2 * RATE:
                    import numpy as np
                    silence = np.zeros((int(gap), 1), dtype="int16")
                    f.write(silence)
                    if worker:
                        worker.tracks[i].feed(silence)
                    written[i] += int(gap)
                    which = "" if len(files) == 1 else (" on the microphone" if i == 0 else " on the computer's sound")
                    tell(f"A gap of {int(gap / RATE)} s with no sound{which} (the computer slept, or the device went away) was filled with silence")
            if extensions[0]:
                limit += EXTEND_S * extensions[0]
                extensions[0] = 0
                noticed = False
                tell(f"Recording extended: it now stops after {_limit_text(limit)}")
            if not noticed and limit > NOTICE_S and max(written) >= (limit - NOTICE_S) * RATE:
                noticed = True
                left = NOTICE_S // 60 and f"{NOTICE_S // 60} minutes" or f"{NOTICE_S} seconds"
                tell(f"The recording stops in {left} ({_limit_text(limit)} limit)")
            if time.time() - last_sync > 30:          # the header is patched too: the file is valid up to here if the app dies
                for f in files:
                    f.flush()
                last_sync = time.time()
            if not EVENTS:                             # the window has its own timer; and a fragment without a line end
                print(f"\r{max(written) // RATE // 60:>4} min", end="", file=sys.stderr)   # would glue itself to the next event
        else:
            tell(f"Recording stopped at the {_limit_text(limit)} limit")
    except KeyboardInterrupt:
        pass
    except RecordingCancelled as e:
        cancelled = e
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)
        # The loop no longer reads `commands`: from here on (the live tail can take a while) a cancel on stdin is what
        # SIGTERM now is, the end of the process at once (Control.dispatch with no handler for it); stop, the end of
        # stdin, pause, resume and extend change nothing.
        CONTROL.listen({verb: (lambda: None) for verb in ("pause", "resume", "extend", "stop", "eof")})
        if toggles and not paused:                     # a Pause that came just before the stop
            paused, pause_began = True, toggles[0]
        if paused:                                     # stopped while paused: the pause is still told
            tell(pause_note(max(written) / RATE, time.monotonic() - pause_began))
        for stream in streams:
            for step in (stream.stop, stream.close):
                try:
                    step()
                except Exception as e:                 # a vanished device must not keep the files from being closed
                    print(f"warning: {e}", file=sys.stderr)
        for i, (f, q) in enumerate(zip(files, queues)):
            while not q.empty():
                block = q.get()
                f.write(block)
                if worker:
                    worker.tracks[i].feed(block)
            f.close()
        if worker:
            if EVENTS:
                ev("STAGE", "Finishing what was transcribed live")
            worker.stop()
        CONTROL.listen(previous_control)           # last: an end of stdin while the live tail was finished changed nothing
    print(f"\nSaved {d}", file=sys.stderr)
    if written[0] > RATE * 10 and peaks[0] < SILENT_PEAK:
        tell("The recording is silent: was the right microphone chosen, and is it allowed under "
             + windows_text("Settings > Privacy & security > Microphone (Let desktop apps access your microphone)?",
                            "Privacy & Security?"))
    if sys_wav and written[1] > RATE * 10 and peaks[1] < SILENT_PEAK:
        tell("The computer's sound was silent for the whole recording: was anything playing, and is the right loopback device "
             "chosen (see `devices`" + windows_text("; on Windows a Stereo Mix input or a virtual audio cable", "") + ")?")
    if cancelled:
        cancelled.folder = d                      # main() names the folder the audio was kept in
        raise cancelled
    return d, mic_wav, sys_wav


# ---------- transcribe ----------
def group_spans(spans, rate=RATE, min_s=MIN_S, max_s=CHUNK_S):
    """Merge VAD speech spans (in samples) into chunks of min_s..max_s seconds, split at pauses."""
    chunks = []
    for s in spans:
        if chunks and chunks[-1][1] - chunks[-1][0] < min_s * rate and s["end"] - chunks[-1][0] <= max_s * rate:
            chunks[-1][1] = s["end"]
        else:
            chunks.append([s["start"], s["end"]])
    return chunks


def audio_tracks(src):
    """Number of audio tracks in the file (0 for a silent video); None when ffprobe cannot tell."""
    ensure_ffmpeg()
    try:
        done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                               "-of", "csv=p=0", str(src)], capture_output=True, encoding="utf-8", errors="replace",
                              **child_options())   # ffprobe writes UTF-8 (a path in an error); a cp1252 decode would fail
    except FileNotFoundError:
        return None
    return len(done.stdout.split()) if done.returncode == 0 else None


def cut_short_wav(src):
    """Whether src is a WAV in the layout WavFile and the app's WavWriter write (one 16-byte PCM fmt chunk, then the data,
    a 44-byte header) whose header counts less sound than the file holds: a recording cut short before its header was
    patched (the app's headers are patched every 30 s and at Stop; in 1.10.0 a Stop that never finished left one stale).
    ffmpeg stops at the header's count, so such a file is read with -ignore_length. A file whose extra bytes begin like a
    chunk (an imported WAV's LIST or id3 metadata after its data) is not one: those bytes are not sound."""
    try:
        with open(src, "rb") as f:
            head = f.read(48)
            if (len(head) < 44 or head[:4] != b"RIFF" or head[8:16] != b"WAVEfmt " or head[36:40] != b"data"
                    or struct.unpack("<IH", head[16:22]) != (16, 1)):
                return False
            counted = struct.unpack("<I", head[40:44])[0]
            size = os.path.getsize(src)
            if 44 + counted >= size:
                return False
            f.seek(44 + counted)
            after = f.read(4)
        return not (len(after) == 4 and all(48 <= c <= 57 or 65 <= c <= 90 or 97 <= c <= 122 or c == 32 for c in after))
    except (OSError, struct.error):
        return False


def load_audio(src, track=None):
    """The sound of any audio or video file ffmpeg can read -> mono 16 kHz float32 samples.

    A video's picture is dropped. With several audio tracks the first is used unless track (0-based) says otherwise."""
    import numpy as np

    n = audio_tracks(src)
    if n == 0:
        raise RuntimeError(f"{src} has no audio track")
    if track is not None and n is not None and track >= n:
        raise RuntimeError(f"{src} has {n} audio track(s); --track {track} does not exist (numbering starts at 0)")
    if n and n > 1 and track is None:
        print(f"note: {src} has {n} audio tracks; using track 0 (choose with --track N, from 0 to {n - 1})",
              file=sys.stderr)
    ensure_ffmpeg()
    whole = ["-ignore_length", "1"] if cut_short_wav(src) else []      # read past a stale header to the end of the file
    try:
        raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *whole, "-i", str(src), "-map", f"0:a:{track or 0}",
                              "-vn", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"], capture_output=True, **child_options())
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not available: the engine could not fetch it (no internet?) and the system has none "
                           f"({ffmpeg_hint()})")
    if raw.returncode:
        raise RuntimeError(f"ffmpeg could not read {src}: {raw.stderr.decode(errors='replace').strip()[:300]}")
    return np.frombuffer(raw.stdout, np.float32)


@functools.lru_cache(maxsize=1)
def cpu_threads():
    """All cores; on Apple silicon the performance cores only (the efficiency cores slow a shared workload down).
    MT_THREADS overrides it: the app sets it to half the cores for a job that runs beside a live recording."""
    if os.environ.get("MT_THREADS", "").isdigit() and int(os.environ["MT_THREADS"]) > 0:
        return int(os.environ["MT_THREADS"])
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, timeout=5,
                                 **child_options())
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
    except Exception:
        pass
    return os.cpu_count() or 4


# ---------- the GPU: CUDA through CTranslate2, with the CPU to fall back on ----------
def compute_device():
    """MT_DEVICE: auto (the default: CUDA when CTranslate2 counts an NVIDIA GPU, the CPU when CUDA then fails to load or
    to decode; see gpu_checked), cuda (CUDA only: a failure is an error, not the fallback; for testing a GPU), cpu (never
    CUDA)."""
    choice = os.environ.get("MT_DEVICE", "auto").strip().lower() or "auto"
    if choice not in ("auto", "cuda", "cpu"):
        sys.exit(f"MT_DEVICE={choice!r} is not available; use auto, cuda or cpu. Nothing was started.")
    return choice


def gpu_checked():
    """Whether a CUDA model is tried at load (a warm-up decode or translation) and, with MT_DEVICE=auto, replaced by the
    CPU when that fails: on Windows, where a driver without CUDA 12 or cuBLAS is common, and wherever MT_DEVICE is set.
    On macOS and Linux without MT_DEVICE the models load exactly as before 1.11.0 (CUDA with no warm-up when a GPU is
    counted, a CUDA fault an error); the Mac counts no GPU, so it never gets here."""
    return sys.platform == "win32" or bool(os.environ.get("MT_DEVICE", "").strip())


_NVIDIA_HANDLES = []          # the DLL folders added and the libraries loaded, kept for the life of the process


def site_packages():
    """This Python's own site-packages folders (a venv's, the user's when it is in use, the installation's), absolute:
    where pip puts the nvidia-* wheels."""
    import site
    import sysconfig
    folders = []
    for get in (lambda: site.getsitepackages(), lambda: [site.getusersitepackages()] if site.ENABLE_USER_SITE else [],
                lambda: [sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]]):
        try:
            folders += [f for f in get() if f and os.path.isabs(f) and f not in folders]
        except Exception:
            pass
    return folders


@functools.lru_cache(maxsize=1)
def nvidia_libraries():
    """Windows: the CUDA libraries pip installs from requirements-cuda-windows.txt (site-packages\\nvidia\\<lib>\\bin) made
    findable for CTranslate2, which loads cuBLAS by name (cublas64_12.dll) only when it first computes on the GPU, from
    folders Windows does not search. Each bin folder goes onto the DLL search path (os.add_dll_directory) and in front of
    PATH (for a library that searches the classic way), and cuBLASLt, cuBLAS and NVRTC (which cuBLAS requires) are loaded
    by full path at once: a later load by name then finds the module already in the process (os.add_dll_directory alone
    covers only loads that ask for it; critic.md 3.7). NVRTC's .alt.dll twin is not loaded. cuDNN is not loaded: the
    Windows ctranslate2.dll 4.9.1 names no cuDNN library or function (its Conv1D has its own GPU code), so the dispatcher
    cudnn64_9.dll its wheel ships stays unused. Only this Python's own site-packages folders are searched, never the
    current folder or another sys.path entry: a DLL found there would be loaded into the process (a planted
    cublas64_12.dll in a downloads folder). Every step may fail (no wheels, a missing dependency): nothing is said here,
    a CUDA model that then cannot load falls back to the CPU with a note. Returns the names of the libraries loaded."""
    if sys.platform != "win32":
        return []
    loaded = []
    try:
        import ctypes
        folders = []
        for entry in site_packages():
            root = Path(entry) / "nvidia"
            if root.is_dir():
                folders += [b for b in sorted(root.glob("*/bin")) if b.is_dir() and b not in folders]
        for folder in folders:
            try:
                _NVIDIA_HANDLES.append(os.add_dll_directory(str(folder)))
            except Exception:
                pass
        if folders:
            os.environ["PATH"] = os.pathsep.join([str(f) for f in folders] + [os.environ.get("PATH", "")])
        seen = set()
        for pattern in ("cublasLt64_12.dll", "cublas64_12.dll", "nvrtc64_12*.dll", "nvrtc-builtins64_12*.dll"):
            for folder in folders:
                for dll in sorted(folder.glob(pattern)):
                    if dll.name.lower() in seen or dll.name.lower().endswith(".alt.dll"):
                        continue
                    seen.add(dll.name.lower())
                    try:
                        _NVIDIA_HANDLES.append(ctypes.WinDLL(str(dll)))
                        loaded.append(dll.name)
                    except OSError:
                        pass
    except Exception:
        pass
    return loaded


def import_ctranslate2():
    if sys.platform == "win32" and compute_device() != "cpu":
        nvidia_libraries()
    import ctranslate2
    return ctranslate2


def cuda_wanted(ctranslate2):
    """Whether a model is first tried on CUDA: MT_DEVICE=cuda, or auto with a GPU that CTranslate2 counts."""
    choice = compute_device()
    if choice != "auto":
        return choice == "cuda"
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def gpu_failed(what, error):
    """A model could not use the GPU: a note (it then runs on the CPU), or with MT_DEVICE=cuda the error itself."""
    if compute_device() == "cuda":
        raise error
    first = (str(error).strip().splitlines() or [""])[0][:200]
    note(f"The GPU could not be used for {what} ({type(error).__name__}: {first}); it runs on the CPU instead "
         "(MT_DEVICE=cpu skips the attempt)")


def warm_up_whisper(model):
    """One decode of a second of silence: CTranslate2 loads cuBLAS and runs its first CUDA kernels only when it first
    computes, so a GPU without a working CUDA 12 / cuBLAS fails here, where the CPU can still take over, rather than in
    the middle of a job."""
    import numpy as np
    segments, _ = model.transcribe(np.zeros(RATE, dtype=np.float32), language="en", beam_size=1, vad_filter=False,
                                   condition_on_previous_text=False)
    list(segments)


@functools.lru_cache(maxsize=1)
def get_model():
    """The Whisper model on CTranslate2: on CUDA in float16 when a GPU is counted (MT_DEVICE), on the CPU in int8 when that
    fails to load or to decode (with a note; gpu_checked), and as always on the CPU in int8 otherwise."""
    ctranslate2 = import_ctranslate2()
    from faster_whisper import WhisperModel

    def load(**kw):
        try:
            return WhisperModel(WHISPER, local_files_only=True, cpu_threads=cpu_threads(), **kw)   # no network when the model is already here
        except Exception as e:
            # On CUDA it is fetched only when it is not downloaded yet (huggingface_hub's LocalEntryNotFoundError is a
            # FileNotFoundError): a GPU that fails otherwise leaves the model to the CPU at once, not after a request to
            # the Hub that fails the same way (or waits for its timeout offline).
            if kw.get("device") == "cuda" and not isinstance(e, FileNotFoundError):
                raise
            return WhisperModel(WHISPER, cpu_threads=cpu_threads(), **kw)
    if cuda_wanted(ctranslate2):
        if not gpu_checked():
            return load(compute_type="float16")         # as before 1.11.0: CTranslate2's own choice of device, CUDA
        try:
            model = load(device="cuda", compute_type="float16")
            warm_up_whisper(model)
            return model
        except Exception as e:
            gpu_failed("the speech model", e)
            return load(device="cpu", compute_type="int8")
    if compute_device() == "cpu":
        return load(device="cpu", compute_type="int8")
    return load(compute_type="int8")                    # as before: CTranslate2's own choice of device, with no GPU the CPU


@functools.lru_cache(maxsize=None)
def ensure_ffmpeg():
    """ffmpeg and ffprobe on PATH: the system's when present, else the ones the engine fetches once (the static-ffmpeg
    package, about 80 MB, kept in FFMPEG_DIR beside the data folder). Returns True when both can be used."""
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    try:
        from static_ffmpeg import run
    except ImportError:
        return False
    try:
        if not FFMPEG_DIR.exists():
            text = "Fetching ffmpeg and ffprobe (once, about 80 MB)"
            ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
        # The archive holds a platform folder (darwin_arm64, darwin, ...) that static-ffmpeg unpacks beside download_dir,
        # then looks for the binaries inside download_dir: so download_dir must be that platform folder.
        ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise(download_dir=str(FFMPEG_DIR / run.get_platform_key()))
    except Exception as e:
        print(f"warning: ffmpeg could not be fetched: {e}", file=sys.stderr)
        return False
    for tool in (ffmpeg, ffprobe):
        folder = os.path.dirname(tool)
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _decode_signals(temperature, compression_ratio):
    """The temperature the decode ended at and its compression ratio, when the engine reports them (screen_made_up reads
    them: a decode that still fails Whisper's thresholds at a temperature above 0 is one Whisper itself rejected)."""
    out = {}
    if temperature is not None:
        out["temp"] = float(temperature)
    if compression_ratio is not None:
        out["cr"] = float(compression_ratio)
    return out


def _segment(s):
    return {"start": float(s.start), "end": float(getattr(s, "end", s.start)), "text": s.text,
            "logprob": float(getattr(s, "avg_logprob", 0.0) or 0.0),
            **_decode_signals(getattr(s, "temperature", None), getattr(s, "compression_ratio", None))}


class CT2Engine:
    """faster-whisper on CTranslate2 (the CPU, or CUDA when present). A chunk is encoded once: the language probabilities
    and the decoding both come from that one encoder pass (faster-whisper's own transcribe() encodes twice when it has to
    detect the language). When the library's internals are not as expected, transcribe() is used as before."""
    name = "ct2"

    def __init__(self, model):
        self.model = model
        self.single_pass = False
        try:
            import dataclasses
            import inspect
            from faster_whisper.audio import pad_or_trim
            from faster_whisper.transcribe import Tokenizer, TranscriptionOptions, get_suppressed_tokens

            self._tokenizer, self._options, self._suppressed, self._pad = Tokenizer, TranscriptionOptions, get_suppressed_tokens, pad_or_trim
            self._defaults = {k: v.default for k, v in inspect.signature(model.transcribe).parameters.items()}
            self._fields = [f.name for f in dataclasses.fields(TranscriptionOptions)]
            self.single_pass = (all(hasattr(model, a) for a in ("feature_extractor", "encode", "generate_segments", "hf_tokenizer"))
                                and hasattr(model.model, "detect_language")
                                and all(n in self._defaults or n == "temperatures" for n in self._fields))
        except Exception:
            self.single_pass = False

    def analyse(self, chunk, detect=True):
        """What a chunk needs before decoding. `.probs` = [(language, probability), ...] best first (None without detection)."""
        if not self.single_pass:
            state = SimpleNamespace(chunk=chunk, probs=None, decoded=None, segments=None)
            if detect:
                segs, info = self.model.transcribe(chunk, language=None, vad_filter=False, condition_on_previous_text=False)
                probs = info.all_language_probs or [(info.language, info.language_probability)]
                state.probs = sorted(probs, key=lambda x: -x[1])
                state.decoded, state.segments = info.language, [_segment(s) for s in segs]
            return state
        features = self.model.feature_extractor(chunk)
        # Exactly the window generate_segments() encodes for its first (and, for a chunk of at most 30 s, only) pass:
        # the last frame is left out (content_frames = frames - 1), then padded to 30 s.
        window = min(self.model.feature_extractor.nb_max_frames, features.shape[-1] - 1)
        enc = self.model.encode(self._pad(features[..., :window]))
        probs = None
        if detect:
            if getattr(self.model.model, "is_multilingual", True):
                probs = [(token[2:-2], prob) for token, prob in self.model.model.detect_language(enc)[0]]
            else:
                probs = [("en", 1.0)]
        return SimpleNamespace(chunk=chunk, features=features, enc=enc, probs=probs)

    def decode(self, state, language):
        """The segments of a chunk decoded in `language`: [{start, end, text, logprob}, ...], times within the chunk."""
        if not self.single_pass:
            if state.segments is not None and state.decoded == language:
                return state.segments
            segs, _ = self.model.transcribe(state.chunk, language=language, vad_filter=False, condition_on_previous_text=False)
            return [_segment(s) for s in segs]
        tokenizer = self._tokenizer(self.model.hf_tokenizer, getattr(self.model.model, "is_multilingual", True),
                                    task="transcribe", language=language)
        values = dict(self._defaults, condition_on_previous_text=False)
        opts = {}
        for name in self._fields:
            if name == "temperatures":
                t = values["temperature"]
                opts[name] = list(t) if isinstance(t, (list, tuple)) else [t]
            elif name == "suppress_tokens":
                opts[name] = self._suppressed(tokenizer, values[name]) if values[name] else values[name]
            else:
                opts[name] = values[name]
        segments = self.model.generate_segments(state.features, tokenizer, self._options(**opts), False, state.enc)
        return [_segment(s) for s in segments]


# The Apple-GPU engine: mlx-whisper on Apple silicon (macOS 14 or newer, Python 3.10 or newer; requirements-mlx.txt).
# MT_WHISPER_MODEL names the size for both engines; MT_MLX_MODEL names an MLX repository outright.
MLX_MODELS = {"large-v3-turbo": "mlx-community/whisper-large-v3-turbo", "turbo": "mlx-community/whisper-large-v3-turbo",
              "large-v3": "mlx-community/whisper-large-v3-mlx", "medium": "mlx-community/whisper-medium-mlx",
              "small": "mlx-community/whisper-small-mlx", "base": "mlx-community/whisper-base-mlx",
              "tiny": "mlx-community/whisper-tiny-mlx"}


def mlx_repo():
    named = os.environ.get("MT_MLX_MODEL")
    if named:
        return named
    return WHISPER if "/" in WHISPER else MLX_MODELS.get(WHISPER, f"mlx-community/whisper-{WHISPER}")


def mlx_available():
    """Apple silicon with the mlx-whisper package installed."""
    import importlib
    import platform

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        importlib.import_module("mlx_whisper")
        return True
    except Exception:
        return False


def engine_name():
    """Which engine a run uses (without loading it): a checkpoint written by one engine is not continued by another.
    MT_ENGINE=auto (the default) takes the Apple GPU when it can, else the CPU; ct2 or mlx force one."""
    choice = os.environ.get("MT_ENGINE", "auto")
    if choice == "auto":
        return "mlx" if mlx_available() else "ct2"
    if choice == "mlx" and not mlx_available():
        sys.exit("MT_ENGINE=mlx needs Apple silicon and the mlx-whisper package (pip install -r requirements-mlx.txt).")
    if choice not in ("ct2", "mlx"):
        sys.exit(f"MT_ENGINE={choice!r} is not available; use auto, ct2 or mlx.")
    return choice


class MLXEngine:
    """mlx-whisper on the Apple GPU. The language probabilities come from one encoder pass over the chunk's mel
    spectrogram; mlx_whisper.transcribe() decodes it (with its own encoder pass, cheap on the GPU)."""
    name = "mlx"

    def __init__(self, repo):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper import audio as mlx_audio
        from mlx_whisper.transcribe import ModelHolder

        self.repo, self.mx, self.lib, self.audio = repo, mx, mlx_whisper, mlx_audio
        self.model = ModelHolder.get_model(repo, mx.float16)

    def analyse(self, chunk, detect=True):
        probs = None
        if detect:
            mel = self.audio.log_mel_spectrogram(chunk, n_mels=self.model.dims.n_mels)
            mel = self.audio.pad_or_trim(mel, self.audio.N_FRAMES, axis=-2).astype(self.mx.float16)
            _, table = self.model.detect_language(mel)
            probs = sorted(table.items(), key=lambda x: -x[1])
        return SimpleNamespace(chunk=chunk, probs=probs)

    def decode(self, state, language):
        result = self.lib.transcribe(state.chunk, path_or_hf_repo=self.repo, language=language,
                                     condition_on_previous_text=False, verbose=None)
        return [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"],
                 "logprob": float(s.get("avg_logprob", 0.0) or 0.0),
                 **_decode_signals(s.get("temperature"), s.get("compression_ratio"))} for s in result.get("segments", [])]


@functools.lru_cache(maxsize=None)
def get_engine():
    engine = MLXEngine(mlx_repo()) if engine_name() == "mlx" else CT2Engine(get_model())
    if EVENTS:
        ev("CHECK", "engine", engine.name)
    return engine


def release_engine():
    """Frees the speech model (about 1.6 GB) before another large model is loaded."""
    import gc
    get_engine.cache_clear()
    get_model.cache_clear()
    holder = getattr(sys.modules.get("mlx_whisper.transcribe"), "ModelHolder", None)
    if holder is not None:                     # mlx-whisper keeps its last model in a class attribute: let it go too
        holder.model = None
        holder.model_path = None
    gc.collect()


def model_identity(engine):
    """The model a checkpoint was made with: the MLX repository for the GPU engine, else the Whisper model name."""
    return mlx_repo() if engine == "mlx" else WHISPER


# The language codes Whisper knows (faster_whisper.tokenizer._LANGUAGE_CODES; a test compares them).
WHISPER_LANGS = frozenset(
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha haw he hi hr ht hu hy id is "
    "it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk "
    "sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue".split())


# Whisper's own English names for its languages (openai/whisper's LANGUAGES table), with its aliases, so every language
# Whisper supports can be chosen by name, not only the ones that can also be translated.
WHISPER_NAMES = {
    "en": "english", "zh": "chinese|mandarin", "de": "german", "es": "spanish|castilian", "ru": "russian", "ko": "korean",
    "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan|valencian",
    "nl": "dutch|flemish", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi",
    "fi": "finnish", "vi": "vietnamese", "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech",
    "ro": "romanian|moldavian|moldovan", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori",
    "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
    "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili", "gl": "galician", "mr": "marathi",
    "pa": "punjabi|panjabi", "si": "sinhala|sinhalese", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi",
    "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole|haitian", "ps": "pashto|pushto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese",
    "sa": "sanskrit", "lb": "luxembourgish|letzeburgesch", "my": "myanmar|burmese", "bo": "tibetan", "tl": "tagalog",
    "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir",
    "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}
_WHISPER_BY_NAME = {re.sub("[^a-z]", "", alias): code for code, names in WHISPER_NAMES.items() for alias in names.split("|")}


def whisper_code(name):
    """'ru', 'Russian', 'Latin', 'Brazilian Portuguese' (or an NLLB code like 'rus_Cyrl') -> the code Whisper uses; None if unknown."""
    key = (name or "").strip().lower()
    if key in WHISPER_LANGS:
        return key
    if re.sub("[^a-z]", "", key) in _WHISPER_BY_NAME:
        return _WHISPER_BY_NAME[re.sub("[^a-z]", "", key)]
    code = nllb_code(name)
    if code == "zho_Hant":
        return "zh"                                  # Whisper has one Chinese
    for alias in (NLLB_LANGS.get(code) or "").split("|"):
        if alias in WHISPER_LANGS:
            return alias
    return None


def parse_languages(spec):
    """'ru, English;de' -> ['ru', 'en', 'de'] (Whisper codes, no repeats). Empty -> []. Raises ValueError on unknown names."""
    names = [n.strip() for n in re.split(r"[,;\n]", spec or "") if n.strip()]
    codes, unknown = [], []
    for name in names:
        code = whisper_code(name)
        if not code:
            unknown.append(name)
        elif code not in codes:
            codes.append(code)
    if unknown:
        raise ValueError("Whisper does not know " + ", ".join(map(repr, unknown)) + ". Use a name or code such as "
                         "English, ru, Brazilian Portuguese (up to the ~100 languages Whisper supports).")
    return codes


def best_allowed(probs, allowed):
    """The most probable of the allowed languages: probs is Whisper's [(code, probability), ...] for all languages.
    The probability returned is the language's own (not renormalised), so a chunk spoken in a language that is not
    in the list gets a low value and the line is tagged as a guess."""
    table = dict(probs or [])
    best = max(allowed, key=lambda code: table.get(code, 0.0))
    return best, round(table.get(best, 0.0), 2)


def choose_language(probs, allowed):
    """(language, confidence, guess) for a chunk from Whisper's probabilities (best first) and the chosen languages.
    With several chosen languages a language outside the list still wins when Whisper is sure of it (UNLISTED_CONF):
    the words are kept in their own language and the line is tagged as a guess, instead of being decoded, that is
    translated by Whisper, in a listed language. When Whisper is unsure, the likeliest listed language is used."""
    if not probs:
        return (allowed[0] if allowed else "en"), 0.0, True
    top, p = probs[0]
    if len(allowed) > 1 and top not in allowed:
        if p >= UNLISTED_CONF:
            return top, round(p, 2), True
        lang, conf = best_allowed(probs, allowed)
        return lang, conf, False
    return top, round(p, 2), False


def runner_up(probs, allowed, lang):
    """The likeliest of the chosen languages other than `lang`, or None."""
    table = dict(probs or [])
    others = [c for c in allowed if c != lang]
    return max(others, key=lambda c: table.get(c, 0.0)) if others else None


def mean_logprob(segs):
    return sum(s["logprob"] for s in segs) / len(segs) if segs else float("-inf")


def hms(seconds):
    t = int(seconds)
    return f"{t // 3600:02}:{t % 3600 // 60:02}:{t % 60:02}"


class Checkpoint:
    """What a transcription has done so far, in its data folder: partial<tag>.json (what the work is: the source and the
    settings, and the chunk list) and partial<tag>.jsonl (one record per finished chunk). A rerun with the same fingerprint
    goes on from the last finished chunk; anything else starts over. Removed once transcript.json is written."""
    def __init__(self, folder, tag=""):
        self.meta = Path(folder) / f"partial{tag}.json"
        self.log = Path(folder) / f"partial{tag}.jsonl"
        self.handle = None
        self.was_live = False                          # resume() found a live checkpoint (reused or not)

    @staticmethod
    def _same(a, b, live):
        if live:                                       # a live checkpoint was written while the file still grew
            a, b = dict(a, size=None, mtime=None), dict(b, size=None, mtime=None)
        return a == b

    def resume(self, fingerprint):
        """-> (chunks, {chunk index: lines}, covered) of an earlier run with the same fingerprint, else (None, {}, 0).
        `covered` is how many samples a live run had cut into chunks; the rest of the audio is still to be cut."""
        try:
            meta = json.loads(self.meta.read_text(encoding="utf-8"))
            self.was_live = bool(meta.get("live"))
            if not self._same(meta.get("fingerprint"), fingerprint, meta.get("live")):
                earlier = meta.get("fingerprint") or {}
                changed = sorted(k for k in set(earlier) | set(fingerprint or {}) if earlier.get(k) != (fingerprint or {}).get(k)
                                 and not (meta.get("live") and k in ("size", "mtime")))
                note("Starting over: the earlier run in this folder used another " + (", ".join(changed) or "setting")
                     + ", so its chunks cannot be reused")
                return None, {}, 0
            done = {}
            if self.log.exists():
                for raw in self.log.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(raw)
                        if Checkpoint.is_failure(rec["lines"]):
                            done.pop(rec["i"], None)   # a chunk the model choked on is tried again, not counted as done
                        else:
                            done[rec["i"]] = rec["lines"]
                    except Exception:
                        pass                           # a record cut short by the crash
            chunks = [tuple(c) for c in meta["chunks"]]
            return chunks, done, (meta.get("covered", 0) if meta.get("live") else -1)
        except Exception:
            return None, {}, 0

    @staticmethod
    def is_failure(lines):
        """The lines a chunk got when the model raised: one note starting with 'chunk failed'."""
        return len(lines) == 1 and str(lines[0].get("note", "")).startswith("chunk failed")

    def start(self, fingerprint, chunks, live=False, covered=0, keep_log=False):
        self.meta.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.meta.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"fingerprint": fingerprint, "chunks": [list(c) for c in chunks], "live": live,
                                   "covered": covered}), encoding="utf-8")
        replace_file(tmp, self.meta)
        if not live and not keep_log and self.log.exists():
            self.log.unlink()

    def record(self, i, lines):
        if self.handle is None:
            self.handle = open(self.log, "a", encoding="utf-8")
        self.handle.write(json.dumps({"i": i, "lines": lines}, ensure_ascii=False) + "\n")
        self.handle.flush()

    def finish(self):
        if self.handle:
            self.handle.close()
            self.handle = None
        for p in (self.meta, self.log):
            if p.exists():
                p.unlink()

    def finish_log_only(self):
        """Closes the log but keeps the checkpoint: the final pass continues from it."""
        if self.handle:
            self.handle.close()
            self.handle = None


def fingerprint(src, allowed, granularity, speaker, engine, track=None):
    try:
        st = Path(src).stat()
        size, mtime = st.st_size, int(st.st_mtime)
    except OSError:
        size = mtime = None
    return {"source": str(src), "size": size, "mtime": mtime, "model": model_identity(engine), "engine": engine,
            "languages": list(allowed), "granularity": granularity, "speaker": speaker, "track": track, "format": 1}


def _speed_file():
    return DATA / ".speed.json"


def load_speed(key):
    """Seconds per chunk this computer needed last time with this engine and model, or None."""
    try:
        return float(json.loads(_speed_file().read_text(encoding="utf-8"))[key])
    except Exception:
        return None


def save_speed(key, seconds):
    try:
        data = json.loads(_speed_file().read_text(encoding="utf-8")) if _speed_file().exists() else {}
    except Exception:
        data = {}
    data[key] = round(seconds, 3)
    try:
        _speed_file().parent.mkdir(parents=True, exist_ok=True)
        _speed_file().write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


# ---------- who spoke: speaker separation (diarization) ----------
# Measured (PR0, the 10-min six-voice three-language dialogue, 4-core Xeon, sherpa-onnx 2.0.1): segmentation pyannote-3.0 +
# 3D-Speaker ERes2Net embeddings. Number of speakers found by clustering (threshold 0.9): 6 of 6, DER 9.4 %, 96 % of the sentences
# labelled right, 78 of 83 speaker changes caught; thresholds 0.5/0.6/0.7/0.8 found 23/17/9/7 speakers (DER 37/23/11/10 %); with
# the count given instead: DER 25 %, 81 % right (the clustering then cannot use the threshold). 0.42x realtime on one thread,
# about 500 MB. The English CAM++ embeddings told the voices apart far worse (DER 62 %). Overlapping speech (3 s, two voices) is
# reported as two speakers, with 0.02 s of false overlap elsewhere. Synthetic voices are a hard case: real voices differ more.
SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_MODEL = ("sherpa-onnx-pyannote-segmentation-3-0", f"{SHERPA_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMBEDDING_MODEL = ("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
                   f"{SHERPA_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"
DIARIZERS = ("sherpa", "pyannote")
CLUSTER_THRESHOLD = 0.9   # how different two voices must be to count as two speakers when their number is not given (measured: 0.9 best)
TURN_JOIN_S = 0.5         # same speaker again within this: one turn
TURN_MIN_S = 1.0          # a shorter turn does not cut a chunk (it joins its neighbour)
OVERLAP_MIN_S = 0.5       # two speakers at once for at least this long: a note in the transcript


def _download(url, dest):
    """Fetches url into dest (a temporary file beside it first, so a cut-off download never looks complete)."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        shutil_copy(r, f)
    replace_file(part, dest)


def shutil_copy(src, dst):
    import shutil
    shutil.copyfileobj(src, dst, 1 << 20)


def ensure_diarization_models():
    """The two sherpa-onnx models in MODELS_DIR, fetched once (about 47 MB): the segmentation model (pyannote 3.0, MIT) and
    the speaker-embedding model (3D-Speaker ERes2Net, Apache-2.0). Returns their paths."""
    import tarfile
    seg = MODELS_DIR / SEGMENTATION_MODEL[0] / "model.onnx"
    emb = MODELS_DIR / EMBEDDING_MODEL[0]
    if not (seg.exists() and emb.exists()):
        text = "Fetching the speaker-separation models (once, about 47 MB)"
        ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
    if not seg.exists():
        archive = MODELS_DIR / (SEGMENTATION_MODEL[0] + ".tar.bz2")
        _download(SEGMENTATION_MODEL[1], archive)
        member = f"{SEGMENTATION_MODEL[0]}/model.onnx"
        with tarfile.open(archive, "r:bz2") as tar:
            info = tar.getmember(member)                       # only this one file, never arbitrary paths from the archive
            if not info.isfile() or ".." in info.name:
                raise RuntimeError(f"unexpected archive content: {info.name}")
            seg.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(info) as src, open(seg.with_suffix(".part"), "wb") as dst:
                shutil_copy(src, dst)
            replace_file(seg.with_suffix(".part"), seg)
        archive.unlink()
    if not emb.exists():
        _download(EMBEDDING_MODEL[1], emb)
    return seg, emb


def diarize(audio, speakers="auto", backend="sherpa"):
    """Who spoke when: [(start, end, k)] in seconds, speakers numbered 0.. in order of first appearance, sorted by start.
    speakers: "auto" (found by clustering) or how many there are. backend: sherpa (built in) or pyannote (optional, slower
    on a CPU, better; needs its packages and a Hugging Face token once)."""
    import numpy as np
    n = -1 if speakers in (None, "auto", "") else int(speakers)
    samples = np.asarray(audio, dtype="float32")
    if backend == "pyannote":
        raw = _diarize_pyannote(samples, n)
    elif backend == "sherpa":
        raw = _diarize_sherpa(samples, n)
    else:
        raise ValueError(f"unknown diarizer {backend!r}; use " + " or ".join(DIARIZERS))
    order = {}
    out = []
    for start, end, k in sorted(raw):
        out.append((round(float(start), 3), round(float(end), 3), order.setdefault(k, len(order))))
    return out


def _diarize_sherpa(samples, n):
    try:
        import sherpa_onnx
    except ImportError:
        raise RuntimeError("speaker separation needs the sherpa-onnx package: pip install -r requirements.txt")
    seg, emb = ensure_diarization_models()
    threads = cpu_threads()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg)), num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=n, threshold=CLUSTER_THRESHOLD),
        min_duration_on=0.3, min_duration_off=0.5)
    if not cfg.validate():
        raise RuntimeError("the speaker-separation models could not be loaded")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def progress(done, total):
        if EVENTS:
            ev("PROG", int(done), int(total))
        return 0
    result = sd.process(samples, callback=progress).sort_by_start_time()
    return [(r.start, r.end, int(r.speaker)) for r in result]


def _diarize_pyannote(samples, n):
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        raise RuntimeError("the pyannote backend needs its packages: pip install -r requirements-pyannote.txt "
                           "(and a Hugging Face token in HF_TOKEN for the one-time model download)")
    token = os.environ.get("HF_TOKEN") or None
    pipeline = Pipeline.from_pretrained(PYANNOTE_PIPELINE, token=token)
    if pipeline is None:
        raise RuntimeError(f"{PYANNOTE_PIPELINE} could not be loaded: accept its conditions on huggingface.co and set HF_TOKEN")
    output = pipeline({"waveform": torch.from_numpy(samples)[None], "sample_rate": RATE}, **({"num_speakers": n} if n > 0 else {}))
    annotation = getattr(output, "speaker_diarization", output)           # pyannote 4 wraps the annotation
    labels = {}
    return [(turn.start, turn.end, labels.setdefault(label, len(labels)))
            for turn, _, label in annotation.itertracks(yield_label=True)]


def turn_cuts(segments):
    """The moments the speaker changes, in seconds: a chunk must not span one. Gaps under TURN_JOIN_S between the same speaker
    do not count; a turn shorter than TURN_MIN_S makes no cut (it joins its neighbour)."""
    turns = []
    for start, end, k in segments:
        if turns and turns[-1][2] == k and start - turns[-1][1] <= TURN_JOIN_S:
            turns[-1][1] = max(turns[-1][1], end)
        else:
            turns.append([start, end, k])
    cuts = []
    for prev, cur in zip(turns, turns[1:]):
        if cur[1] - cur[0] >= TURN_MIN_S and prev[1] - prev[0] >= TURN_MIN_S:
            cuts.append(round((prev[1] + cur[0]) / 2, 3))
    return cuts


def split_at(chunks, cuts, rate=RATE, margin_s=0.5):
    """Chunks (sample pairs) split at the cuts (seconds) that fall at least margin_s inside them."""
    out = []
    for a, b in chunks:
        pieces, start = [], a
        for c in cuts:
            at = int(c * rate)
            if start + margin_s * rate < at < b - margin_s * rate:
                pieces.append((start, at))
                start = at
        pieces.append((start, b))
        out.extend(pieces)
    return out


def speaker_label(k, prefix="Speaker"):
    return f"{prefix} {k + 1}"


def label_speakers(lines, segments, prefix="Speaker"):
    """Gives every spoken line the speaker whose segments overlap it most (by [t, end]; a line without end counts 2 s).
    A line no segment touches keeps no label."""
    out = []
    for line in lines:
        if "text" not in line or line.get("t") is None:
            out.append(line)
            continue
        t0, t1 = line["t"], line.get("end") or line["t"] + 2.0
        totals = {}                                    # speaker -> seconds of the line that are theirs, over all their turns
        for start, end, k in segments:
            if start >= t1:
                break
            overlap = min(end, t1) - max(start, t0)
            if overlap > 0:
                totals[k] = totals.get(k, 0.0) + overlap
        new = dict(line)
        if totals:
            best = max(totals, key=lambda k: (totals[k], -k))   # the most time; the earlier speaker on a tie
            new["speaker"] = speaker_label(best, prefix)
        out.append(new)
    return out


def overlap_notes(segments, prefix="Speaker"):
    """A note for every stretch of OVERLAP_MIN_S or more in which two speakers talk at once."""
    notes = []
    for i, (s1, e1, k1) in enumerate(segments):
        for s2, e2, k2 in segments[i + 1:]:
            if s2 >= e1:
                break
            if k2 == k1:
                continue
            start, end = max(s1, s2), min(e1, e2)
            if end - start >= OVERLAP_MIN_S:
                a, b = speaker_label(min(k1, k2), prefix), speaker_label(max(k1, k2), prefix)
                notes.append({"t": round(start, 1), "note": f"overlap {hms(start)}\u2013{hms(end)} {a} / {b}"})
    return notes


def with_notes(lines, notes):
    """Spoken lines and notes together, in time order (a note sorts before a line at the same second)."""
    return sorted(lines + notes, key=lambda l: (l.get("t") if l.get("t") is not None else -1, "text" in l))


def transcribe_lines(src, language=None, track=None, granularity="auto", speaker=None, position=None, checkpoint=None,
                     speakers=None, diarizer="sherpa", label="Speaker", context="media"):
    """The lines of one audio file (speaker, when given, is put on every line). granularity: auto or a key of GRANULARITY, how finely speech is cut into chunks. language: None = Whisper detects each chunk among all its languages; one language = forced for the whole file;
    several (a list or 'ru,en,de') = detected per chunk, but only among these (a language Whisper is sure of still wins, tagged as a guess).
    position: (track, tracks) of a call, for the progress. A chunk in which nothing was recognised becomes a note line, so the
    transcript shows where speech was lost; a note at the end says how much. context: "recording" (a recording or a
    call) or "media" (an imported file), for screen_made_up."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    allowed = parse_languages(language) if isinstance(language, str) or language is None else list(language)
    forced = allowed[0] if len(allowed) == 1 else None

    audio = load_audio(src, track)
    total = len(audio) / RATE
    resolved = resolve_granularity(granularity, allowed)
    min_s, pause_ms = GRANULARITY[resolved]
    engine = get_engine()
    cp = Checkpoint(checkpoint, f".{position[0]}" if position else "") if checkpoint else None
    fp = fingerprint(src, allowed, resolved, speaker, engine_name(), track)    # the same source as a live track's
    chunks, done, covered = cp.resume(fp) if cp else (None, {}, 0)
    # A live run whose chunks cannot be reused (its languages or granularity were changed): the window still shows its
    # lines, which this run decodes again; it starts empty (a call's tracks are sent again after the merge).
    restart_window = bool(cp and cp.was_live and chunks is None and EVENTS and position is None)
    segments = None
    if speakers is not None:
        if EVENTS:
            ev("STAGE", "Separating speakers" + (f" (track {position[0]} of {position[1]})" if position else ""))
        segments = diarize(audio, speakers, diarizer)
        found = len({k for _, _, k in segments})
        note(f"{found} speaker{'s' if found != 1 else ''} found" + (f" on track {position[0]}" if position else ""))
    if chunks is None:
        chunks = cut_chunks(audio, 0, pause_ms, min_s)
        if segments:                                   # a chunk never spans a speaker change
            chunks = split_at(chunks, turn_cuts(segments))
        if cp:
            cp.start(fp, chunks)
    else:
        if covered >= 0:                               # a live run: its chunks stay, the tail is still to be cut
            tail = cut_chunks(audio, covered, pause_ms, min_s)
            rewritten = False
            if segments:                               # the live run knew no speakers: its chunks may span a change of speaker
                cuts = turn_cuts(segments)
                split, tail = split_at(chunks, cuts), split_at(tail, cuts)
                if split != chunks:
                    kept = {split.index(chunks[i - 1]) + 1: lines for i, lines in done.items()
                            if 0 < i <= len(chunks) and chunks[i - 1] in split}
                    redo = len(done) - len(kept)
                    chunks, done = split, kept
                    if redo:
                        note(f"{redo} chunk{'s' if redo != 1 else ''} transcribed live span a change of speaker and are decoded again")
                    if cp:                             # the log is written anew: its records are numbered by the new list
                        cp.start(fp, chunks + tail)
                        for i, lines_done in sorted(done.items()):
                            cp.record(i, lines_done)
                        rewritten = True
            chunks = chunks + tail
            if cp and not rewritten:
                cp.start(fp, chunks, keep_log=True)    # the full list, as a normal checkpoint; the finished chunks stay
        if done:
            note(f"Going on from an earlier run: {len(done)} of {len(chunks)} chunks were already done")
    lines, started, unrecognised, speech, failed, decoded, made = [], time.monotonic(), 0.0, 0.0, 0, 0, 0
    hidden = []                                        # the made-up lines left out so far: they count toward a repeated outro
    speed_key = f"{engine.name}:{model_identity(engine.name)}"
    chunk_times, last_tick = [], time.monotonic()
    if restart_window:
        ev("CLEAR")
    if EVENTS and len(chunks) > len(done):
        remembered = load_speed(speed_key)         # how long a chunk took last time: an estimate before the first one is done
        if remembered is not None:
            ev("ETA", round(remembered * (len(chunks) - len(done))))
    for i, (a, b) in enumerate(chunks, 1):
        speech += (b - a) / RATE
        if i in done:
            new = done[i]
        else:
            try:
                new = decode_chunk(engine, audio[a:b], a, allowed, forced, speaker, adaptive=resolved == "adaptive",
                                   context=context)
            except Exception as e:                     # one bad chunk must not cost the hours already done
                failed += 1
                print(f"warning: chunk {i} at {hms(a / RATE)} failed: {type(e).__name__}: {e}", file=sys.stderr)
                new = [{"t": round(a / RATE, 1), "note": f"chunk failed ({type(e).__name__}: {e})"}]
            if cp:
                cp.record(i, new)
            decoded += 1
            chunk_times.append(time.monotonic() - last_tick)
        last_tick = time.monotonic()
        screened = screen_made_up(new, context=context)   # also the lines of a checkpoint written before the screening existed
        new = without_made_up(screened)
        hidden.extend(l for l in screened if l.get("made_up"))
        made += len(screened) - len(new)
        for line in new:                               # the chunk, or the pieces of a re-cut chunk, that gave nothing
            lost = re.match(r"speech not recognised for (\d+) s", str(line.get("note", "")))
            if lost:
                unrecognised += int(lost.group(1))
        lines.extend(new)
        if EVENTS:
            ev("PROG", i, len(chunks))
            ev("POS", round(b / RATE, 1), round(total, 1))
            for line in new:
                ev("LINE", json.dumps(line, ensure_ascii=False))
            if decoded >= ETA_AFTER and i < len(chunks):
                ev("ETA", round(sorted(chunk_times)[len(chunk_times) // 2] * (len(chunks) - i)))   # the median: start-up is in chunk 1
        else:
            print(f"\r{i}/{len(chunks)} chunks", end="", file=sys.stderr)
    if not EVENTS:
        print(file=sys.stderr)
    if len(chunk_times) >= ETA_AFTER:
        save_speed(speed_key, sorted(chunk_times)[len(chunk_times) // 2])
    if failed and failed == len(chunks):
        raise RuntimeError(f"every chunk failed; the last error is in the log")
    lines, again = drop_repeated_outros(lines, left_out=hidden)
    made += again
    if made:
        text = made_up_note(made, position)
        lines.append({"t": round(total, 1), "note": text})
        note(text)
    if unrecognised:
        text = (f"decoded {round(100 * (speech - unrecognised) / speech)} % of the speech found"
                f" ({round(unrecognised)} s not recognised" + (f" on track {position[0]}" if position else "") + ")")
        lines.append({"t": round(total, 1), "note": text})
        note(text)
    if segments is not None:
        lines = with_notes(label_speakers(lines, segments, label), overlap_notes(segments, label))
    # The window gets every line again when the labels came at the end, or when outros it was already shown were left out
    # (a call's track does not: the call sends its merged lines again once both tracks are done).
    if EVENTS and (segments is not None or (again and position is None)):
        ev("CLEAR")
        for line in lines:
            ev("LINE", json.dumps(line, ensure_ascii=False))
    if not any("text" in l for l in lines):
        text = no_speech_text(src, position)
        lines.append({"t": round(total, 1), "note": text})
        note(text)
    return lines


def no_speech_text(src, position=None):
    """The note for a file, or a call track (position = (track, tracks)), in which nothing was spoken."""
    if position and position[0] == 2:
        return "No speech was found in the computer's sound: the other side is missing from the transcript (was anything playing?)"
    if position:
        return "No speech was found in your microphone: your side is missing from the transcript"
    return f"No speech was found in {Path(src).name}"


VAD_WINDOW_S = 600    # the VAD looks at the audio in windows of this length (its copies of a 4 h file would take 2 GB)


def speech_spans(audio, start, pause_ms, window_s=VAD_WINDOW_S):
    """Speech spans of audio[start:] as {start, end} in samples of the whole audio. The VAD runs on windows of
    window_s; a span that reaches a window's end is left to the next window, which starts where the last whole span
    ended, so no stretch of speech is cut in two."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(max_speech_duration_s=CHUNK_S, min_silence_duration_ms=pause_ms, min_speech_duration_ms=250)
    spans, pos, size = [], start, window_s * RATE
    while pos < len(audio):
        end = min(len(audio), pos + size)
        found = get_speech_timestamps(audio[pos:end], options)
        last = end >= len(audio)
        if not last:
            margin = int(RATE * pause_ms / 1000)
            whole = [sp for sp in found if sp["end"] + margin < end - pos]
            partial = [sp for sp in found if sp["end"] + margin >= end - pos]     # reaches the window's end: next window's
            if partial and partial[0]["start"] > 0:
                found, next_pos = whole, pos + partial[0]["start"]
            elif partial:                                   # speech from the window's start to its end: keep it as it is
                next_pos = end
            else:
                next_pos = pos + whole[-1]["end"] if whole else end
            if next_pos <= pos:
                next_pos = end
        else:
            next_pos = end
        spans.extend({"start": sp["start"] + pos, "end": sp["end"] + pos} for sp in found)
        pos = next_pos
    return spans


def cut_chunks(audio, start, pause_ms, min_s):
    """The chunks of audio[start:] (sample offsets in the whole audio): speech spans found by the VAD, grouped."""
    if start >= len(audio):
        return []
    return [tuple(c) for c in group_spans(speech_spans(audio, start, pause_ms), min_s=min_s)]


# ---------- live: transcribing while the recording goes on ----------
class LiveTrack:
    """One growing track: new samples are fed in, and once a stretch of speech is closed (a pause follows it, and the
    chunk is long enough) it is handed out as a chunk. The chunks and lines go to a live checkpoint, which the final
    pass continues from, so at the end only the tail is still to be decoded."""
    def __init__(self, name, folder, tag, src, allowed, granularity, speaker, engine_name, position=None):
        import numpy as np

        self.name, self.speaker, self.position = name, speaker, position
        self.allowed = allowed
        self.forced = allowed[0] if len(allowed) == 1 else None
        self.granularity = resolve_granularity(granularity, allowed)
        self.min_s, self.pause_ms = GRANULARITY[self.granularity]
        # Only the audio from `base` on is kept (what the pending chunks and the next cut need): chunk offsets are absolute
        # in the whole track, so a recording of any length holds about a minute of samples here, not all of them.
        import threading

        self.audio = np.zeros(0, dtype="float32")
        self.base = 0                                  # the track position of audio[0]
        self.blocks = []                               # fed, not yet joined onto audio (one join per cut, not per block)
        self.handoff = threading.Lock()                # feed() runs on the recording's thread, _settle() on the worker's
        self.fed = 0                                   # samples fed in all
        self.covered = 0                               # samples cut into chunks so far
        self.chunks = []                               # (a, b) handed out
        self.pending = []                              # chunks waiting for the worker
        self.shown = []                                # the lines shown in the window so far (LiveWorker.resend)
        self.outro_count = collections.Counter()       # outros decoded so far, and the shown copies (mark_repeated_outros)
        self.shown_outros = collections.defaultdict(list)
        self.outro_chunks = {}                         # chunk index -> (its lines, how many were marked) while a shown outro is in it
        self.withdrawn = False                         # a shown line was marked made up since the window was last sent
        self.cp = Checkpoint(folder, tag)
        # The final pass fingerprints without a speaker (a call's labels come from the merge), so the live one does too.
        self.fp = dict(fingerprint(src, allowed, resolve_granularity(granularity, allowed), None, engine_name), size=None, mtime=None)
        self.cp.start(self.fp, [], live=True, covered=0)

    def feed(self, block):
        import numpy as np

        samples = np.asarray(block).reshape(-1).astype("float32") / 32768.0
        with self.handoff:
            self.blocks.append(samples)
            self.fed += len(samples)

    def _settle(self):
        """Joins the fed blocks onto the kept audio and drops what no chunk needs any more (in steps of 30 s, so the
        copy is not made at every call). The blocks are taken under the lock, so a block fed during the copy is not lost."""
        import numpy as np

        with self.handoff:
            blocks, self.blocks = self.blocks, []
        if blocks:
            self.audio = np.concatenate([self.audio] + blocks)
        needed = min([self.covered] + [a for a, _ in self.pending])
        if needed - self.base >= RATE * 30:
            self.audio = self.audio[needed - self.base:].copy()
            self.base = needed

    def cut(self, final=False):
        """Closes the chunks that can be closed: speech that ends a pause before the end of what was fed (or everything,
        when final). Returns the number of new chunks."""
        self._settle()
        end = self.base + len(self.audio)              # the track position just past the audio kept
        if end - self.covered < RATE * (0.5 if final else 2):
            return 0
        new = [(a + self.base, b + self.base) for a, b in cut_chunks(self.audio, self.covered - self.base, self.pause_ms, self.min_s)]
        if not new:
            if not final and end - self.covered > RATE * 60:   # a minute without speech: nothing to come back to
                self.covered = end - RATE * 2
            return 0
        if not final:
            keep = end - int(RATE * self.pause_ms / 1000) - RATE // 2
            closed = [(a, b) for a, b in new if b <= keep]
            # The last chunk may still grow when the next span arrives (group_spans joins a short chunk with what follows),
            # so a short one is held back; but only while the silence after it is short. Held back for good, a short answer
            # on one side of a call ("Yes, Friday works") would not appear until that side spoke again, and a short test
            # would show no line at all (1.9.11).
            if closed and closed[-1] == new[-1] and (closed[-1][1] - closed[-1][0]) < self.min_s * RATE \
                    and end - closed[-1][1] < LIVE_CLOSE_S * RATE:
                closed = closed[:-1]
            new = closed
        if not new:
            return 0
        self.chunks.extend(new)
        self.pending.extend(new)
        self.covered = new[-1][1]
        self.cp.start(self.fp, self.chunks, live=True, covered=self.covered)
        return len(new)

    def decode(self, engine, translate=None):
        """Decodes one pending chunk; returns its lines or None. `translate`, when given, is called with the lines once
        they are shown and adds their translations (`tr`): the lines go out again with them, and the checkpoint records
        them with the translations, so the final pass has those too."""
        if not self.pending:
            return None
        self._settle()
        a, b = self.pending.pop(0)
        i = self.chunks.index((a, b)) + 1
        try:
            lines = decode_chunk(engine, self.audio[a - self.base:b - self.base], a, self.allowed, self.forced, self.speaker,
                                 adaptive=self.granularity == "adaptive", context="recording")
        except Exception as e:
            lines = [{"t": round(a / RATE, 1), "note": f"chunk failed ({type(e).__name__}: {e})"}]
        if mark_repeated_outros(self, lines):          # an outro come again and again: its shown copies are withdrawn too
            self.withdrawn = True
            self._record_withdrawn()
        shown = without_made_up(lines)                 # what Whisper made up on noise: kept for the final count, never shown
        if any(is_outro(l) for l in shown):            # a copy that may be withdrawn later: its chunk is written again then
            vars(self).setdefault("outro_chunks", {})[i] = (lines, sum(1 for l in lines if l.get("made_up")))
        vars(self).setdefault("shown", []).extend(shown)
        if EVENTS:
            for line in shown:
                ev("LINE", json.dumps(dict(line, live=True), ensure_ascii=False))
        if translate and any("text" in l for l in shown):
            translate(shown)                           # the line is on screen already; its translation follows
            if EVENTS:
                for line in shown:
                    if "tr" in line:
                        ev("LINE", json.dumps(dict(line, live=True), ensure_ascii=False))
        self.cp.record(i, lines)
        if EVENTS:
            ev("POS", round(self.covered / RATE, 1), round(self.fed / RATE, 1))
        return shown

    def _record_withdrawn(self):
        """Writes the earlier chunks whose shown outro was just withdrawn to the checkpoint again, with the mark: resume
        keeps the last record of a chunk, so the final pass neither shows nor sends them again, and its note still
        counts each copy once. A chunk is forgotten here once it holds no unmarked outro. The chunk being decoded is
        not among them yet (decode adds it after this call, and writes it at its end)."""
        chunks = vars(self).setdefault("outro_chunks", {})
        for j, (lines, marked) in list(chunks.items()):
            now = sum(1 for l in lines if l.get("made_up"))
            if now > marked:
                self.cp.record(j, lines)
                chunks[j] = (lines, now)
            if not any(is_outro(l) and not l.get("made_up") for l in lines):
                del chunks[j]

    def close(self):
        self.cp.finish_log_only()


class LiveWorker:
    """A thread that decodes the closed chunks of one or more live tracks, round-robin, as long as the recording runs.
    With `targets`, every decoded line is translated into them as well (offline with the local model beside the speech
    model, or with Claude), so the window shows the translation beside the transcript while the meeting goes on."""
    def __init__(self, tracks, targets=(), offline=False, source=None):
        import threading

        self.tracks = tracks
        self.targets, self.offline, self.source = list(targets), offline, source
        self.translation_off = False                   # set after a failure: said once, the rest is translated at the end
        self.models_here = threading.Event()           # the translation models are on disk (fetched in the background)
        self.models_loaded = False
        self.waiting_told = False
        self.stop_flag = threading.Event()
        self.engine_ready = threading.Event()          # set once the speech model is loaded (until then nothing is written)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.lines = 0
        self.failed = None                             # the error that ended the worker early, if one did

    def start(self):
        self.thread.start()

    def fetch_models(self):
        """On a thread of its own: the translation models downloaded if they are not here yet (7 GB for TranslateGemma,
        4.2 GB for Hy-MT3 the first time), so that neither the recording nor the live transcript waits for them. A chunk
        decoded meanwhile gets no live translation; the final pass translates it."""
        try:
            for repo, size in translation_models(self.targets, self.offline):
                try:
                    fetch_model(repo, size, stage=False)
                except Exception as e:
                    if repo == NLLB and llm_on():      # TranslateGemma's or Hy-MT3's lines go on; NLLB's wait for the end
                        print(f"NLLB could not be fetched ({type(e).__name__}); the lines {llm_label()} does not cover "
                              "are translated at the end", file=sys.stderr)
                        continue
                    raise
        except Exception as e:
            self.translation_off = True
            note(f"Live translation is off ({type(e).__name__}: {e}): the translation model could not be fetched; "
                 "the lines are transcribed live and translated when the recording ends")
        finally:
            self.models_here.set()

    def load_models(self):
        """On the worker's own thread, between two chunks (the model's GPU work stays on one thread): the model that
        takes most lines, TranslateGemma or Hy-MT3 when a target is one of its languages, else NLLB."""
        if llm_on() and any(llm_code(t) for t in self.targets):
            if TRANSLATOR == "hy" and HY == HY_DEFAULT:
                fetch_hy(stage=False, refetch=False)   # checked here first: a file to download again waits for the final pass
            get_llm(keep_engine=True)
        else:
            get_nllb(keep_engine=True)

    def translate(self, lines):
        """Adds the targets' translations to the lines of one chunk; a failure (no network, the model would not load)
        ends the live translation with a note, not the recording."""
        if not self.targets or self.translation_off:
            return
        if self.offline and not self.models_loaded:
            if not self.models_here.is_set():
                if not self.waiting_told:
                    self.waiting_told = True
                    print("Live translation starts once the translation model is downloaded; until then the lines are "
                          "translated at the end", file=sys.stderr)
                return
            if self.translation_off:
                return
            try:
                self.load_models()
                self.models_loaded = True
            except Exception as e:                     # no memory, a broken download: the transcript goes on live
                self.translation_off = True
                note(f"Live translation is off ({type(e).__name__}: {e}): the translation model could not be loaded; "
                     "the lines are transcribed live and translated when the recording ends")
                return
        try:
            translate_live(lines, self.targets, self.offline, self.source)
        except Exception as e:
            self.translation_off = True
            note(f"Live translation stopped ({type(e).__name__}: {e}); the translation is made when the recording ends")

    def run(self):
        try:
            engine = get_engine()
            if self.targets and self.offline:          # the translation models: here already, or fetched beside the work
                import threading
                if all(model_present(repo) for repo, _ in translation_models(self.targets, self.offline)):
                    self.models_here.set()
                else:
                    threading.Thread(target=self.fetch_models, daemon=True).start()
            self.engine_ready.set()
            if self.stop_flag.is_set():                # stopped while the model loaded: the final pass does everything
                return
            while True:
                worked = False
                for track in self.tracks:
                    track.cut()
                    lines = track.decode(engine, translate=self.translate) if self.targets else track.decode(engine)
                    if lines is not None:
                        worked = True
                        self.lines += sum(1 for l in lines if "text" in l)
                    if any(getattr(t, "withdrawn", False) for t in self.tracks):
                        self.resend()
                if not worked:
                    if self.stop_flag.is_set():
                        for track in self.tracks:      # the recording is over: what is closed now is decoded too
                            track.cut(final=False)
                        if not any(t.pending for t in self.tracks):
                            break
                    else:
                        self.stop_flag.wait(1.0)
        except Exception as e:                         # the model could not be loaded, the VAD failed …: said, not silent
            self.failed = e
            self.engine_ready.set()
            if EVENTS:                                 # the window stops saying that the model is loading
                ev("CHECK", "live", "failed")
            note(f"Live transcription stopped ({type(e).__name__}: {e}); the recording goes on and everything is transcribed at the end")

    def resend(self):
        """The window again without the lines withdrawn since they were shown (an outro said again and again): @@CLEAR,
        then every track's shown lines that stand, by time, with their translations. One worker sends every @@LINE of a
        recording, so one CLEAR covers both tracks of a call."""
        kept = []
        for track in self.tracks:
            track.withdrawn = False
            track.shown = [l for l in getattr(track, "shown", []) if not l.get("made_up")]
            kept.extend(track.shown)
        if EVENTS:
            ev("CLEAR")
            for line in in_time_order(kept):
                ev("LINE", json.dumps(dict(line, live=True), ensure_ascii=False))

    def stop(self, wait=120):
        """Ends the worker and waits for it: its checkpoint is handed to the final pass only once nothing writes to it. A
        worker still loading the speech model has written nothing and is not waited for (the final pass does the work)."""
        self.stop_flag.set()
        self.thread.join(wait)
        if self.thread.is_alive() and not self.engine_ready.is_set():
            note("The speech model was still loading when the recording stopped: nothing was transcribed live, everything is transcribed now")
        else:
            while self.thread.is_alive():              # a slow computer: the current chunk still decodes; say so and wait on
                print("Still finishing what was transcribed live", file=sys.stderr)
                self.thread.join(wait)
        for track in self.tracks:
            track.close()


def looks_mixed(probs):
    """Adaptive chunking: do a chunk's language probabilities suggest more than one language in it?"""
    if not probs:
        return False
    second = probs[1][1] if len(probs) > 1 else 0.0
    return probs[0][1] < SPLIT_CONF or second > SPLIT_SECOND


def decode_unit(engine, state, allowed, forced):
    """Language decision and decoding of one analysed unit (a chunk, or a piece of one): (segments, lang, conf, guess)."""
    if forced:
        lang, conf, guess = forced, 1.0, False
    else:
        lang, conf, guess = choose_language(state.probs, allowed)
    segs = engine.decode(state, lang)
    if not forced and not guess and conf < LOW_CONF:
        # Not sure: decode in the runner-up as well (the likeliest other chosen language, or without a list Whisper's own
        # second choice) and keep what Whisper found likelier.
        other = runner_up(state.probs, allowed, lang) if len(allowed) > 1 else \
            (state.probs[1][0] if not allowed and state.probs and len(state.probs) > 1 else None)
        if other:
            alt = engine.decode(state, other)
            if mean_logprob(alt) > mean_logprob(segs):
                segs, lang = alt, other
    return segs, lang, conf, guess


# ---------- what Whisper makes up on noise ----------
# Whisper learned from subtitled video: on silence, music or noise it writes the credits and outros of those subtitles
# ("Субтитры сделал DimaTorzok", "Редактор субтитров А.Синецкая Корректор А.Егорова", "Subtitles by the Amara.org
# community": openai/whisper discussions 928, 1873 and 2608; Barański et al. 2025, "Investigation of Whisper ASR
# Hallucinations Induced by Non-Speech Audio"). Its own no-speech score does not catch them: such a line is decoded with
# confidence, and large-v3-turbo reports a no-speech probability of about 0 even for silence (SYSTRAN/faster-whisper#1128).
# MADE_UP_ALWAYS (and MADE_UP_CAPTION): text that is essentially never real speech (subtitle credits, Amara.org, named
# subtitle authors, sound captions); MADE_UP_MEDIA: the phrases of a published video ("Thanks for watching", channel
# plugs, broadcaster tags, "Продолжение следует"): in a recording or a call they are dropped like a credit (and cut
# off the end of a real line), in an imported file (which may be such a video) only with a sign that nobody said them;
# MADE_UP_DOUBTFUL: outros and plugs people really say in meetings too ("Спасибо за внимание", "Subscribe to the
# channel", "Captions by Friday"), dropped only with a sign that nobody said them; MADE_UP_MEETING: the same, but never
# counted as repeated (goodbyes several people say: "Bis zum nächsten Mal", "See you tomorrow"); MADE_UP_SHORT: short thanks and goodbyes ("Thank you.", "Спасибо."), never
# dropped by their text, only alone in their chunk and timed by Whisper past the end of the audio (made_up). The signs:
# stretched over near-silence (_stretched), timed far past the end of the chunk's audio (OVERHANG_S), a decode that still
# fails Whisper's own thresholds after its temperature fallback (_failed, for MADE_UP_MEDIA in an imported file only),
# or 3 times or more on one track (MADE_UP_MEDIA and MADE_UP_DOUBTFUL). Every
# pattern matches a whole sentence (^...$), never a part of a longer one. The lists were collected from the reports and
# from the filters of other projects (Vexa, hex, Speech-Translate, BeniaBot, NeuroMita, Lapsus, Sharrnah, tapeback) and
# checked against real sentences in the tests.
MADE_UP_ALWAYS = [re.compile(p, re.I) for p in (
    # Субтитры сделал/создавал/делал/добавил DimaTorzok; bare DimaTorzok; 'канал Субтитры сделал DimaTo...; repeated
    # without a break; run into 'Продолжение следует' ('Субтитры сделал DimaTorzok, Продолжение следует...')
    r"^[\W_]*(?:(?:канал\W+)?(?:субтитры\W+)?(?:\w+\W+)?dima\W?torzok[\W_]*)+(?:продолжение\W+следует[\W_]*)?$",
    # Субтитры подогнал «Симон»!
    r"^[\W_]*субтитры\W+подогнал\W+симон[\W_]*$",
    # Редактор субтитров А.Синецкая/А.Семкин/М.Лосева/Е.Воинова Корректор А.Егорова/А.Кулакова; Редакто...; also run
    # into 'Продолжение следует...' without a sentence break
    r"^[\W_]*(?:редактор|субтитры)\W+субтитров\W+(?-i:[А-ЯЁ])(?:\.\s?|\s)(?-i:[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)(?:\W+корректор\W+(?-i:[А-ЯЁ])(?:\.\s?|\s)(?-i:[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?))?(?:\W+продолжение\W+следует)?[\W_]*$",
    # Корректор А.Егорова / Корректор А.Кулакова as a separate segment (the tail of the editor credit); also glued
    # without the initial, 'Корректор .Кулакова' (tapeback _quality.py): the dot right before the surname, never a
    # plain 'Корректор Егорова' or 'Корректор. Егорова.' (a proofreader named in an editorial meeting)
    r"^[\W_]*корректор(?:\W+(?-i:[А-ЯЁ])(?:\.\s?|\s)|\s?\.)(?:егорова|кулакова)[\W_]*$",
    # Субтитры А.Синецкая (NeuroMita whisper_recognizer.py)
    r"^[\W_]*субтитры\W+(?-i:[А-ЯЁ])\.\s?синецкая[\W_]*$",
    # Субтитры и корректура Оксаны Каменской (NeuroMita)
    r"^[\W_]*субтитры\W+и\W+корректура\W+оксаны\W+каменской[\W_]*$",
    # www.northstarit.co.uk (openai/whisper discussion 928)
    r"^[\W_]*(?:www\W+)?northstarit\W+co\W+uk[\W_]*$",
    # This video is a derivative work of the Touhou Project (openai/whisper discussion 1873)
    r"^[\W_]*this\W+video\W+is\W+a\W+derivative\W+work\W+of\W+the\W+touhou\W+project[\W_]*$",
    # Спасибо за субтитры Алексею Дубровскому!
    r"^[\W_]*спасибо\W+за\W+субтитры\W+алексею\W+дубровскому[\W_]*$",
    # Субтитрувальниця Оля Шор, also with 'Дякую за перегляд!' in front in the same segment
    r"^[\W_]*(?:дякую\W+за\W+перегляд\W+)?субтитрувальниц[яі]\W+оля\W+шор[\W_]*$",
    # Upper-case Russian sound captions: ДИНАМИЧНАЯ МУЗЫКА, ВЕСЕЛАЯ МУЗЫКА, АПЛОДИСМЕНТЫ, СМЕХ. Case-se...
    r"^[\W_]*(?-i:(?:[А-ЯЁ]+\s)?(?:МУЗЫКА|АПЛОДИСМЕНТЫ|СМЕХ|ПЕРЕСТРЕЛКА|ВЗРЫВ|ВЫСТРЕЛЫ?|ГРОМ|ПЛАЧ|КРИК|СТУК|ГУДОК|СИРЕНА|ЛАЙ\sСОБАК))[\W_]*$",
    # Amara.org credits in about 15 languages: Subtitles by the Amara.org community; Sous-titres réalis...
    r"^[\W_]*(?:[\w'’]+\W+(?:e|i|and|und|et|y|и|ja)\W+|a\W+)?(?:小[编編])?(?:subtit\w*|sous\W?titr\w*|untertitel\w*|sottotitol\w*|subt[ií]t[uo]l\w*|legenda\w*|napis\w*|ondertitel\w*|undertext\w*|felirat\w*|субтитр\w*|титр\w*|字幕|자막|transcri\w*|tłumacz\w*|traduc\w*|übersetz\w*|caption\w*)(?:\W+(?:by|the|par|para|per|la|de|d|des|du|die|från|von|der|pela|pelo|da|do|das|przez|door|dalla|dal|por|del|en|español|réalisés?|créés?|stworzone|wykonane|robione|ingediend|gemaakt|creati|realizados|creados|hechos|aufgrund|community|comm?unauté|comunidade|comunidad|comunitat|comunità|społeczność|gemeenschap|gemeinschaft|közösség\w*|készítette|az|a|e|i|und|and|et|y|и|сообществом|создавались|созданы|сделаны|subtítulos|legendas|napisy)\b|\W*(?:由|字幕|社[区區群])){0,6}?\W*amara\W*org(?:\W+(?:community|comm?unauté|gemeinschaft|gemenskapen|gemeenschap|comunidade|comunidad|comunitat|comunità|społeczność|közösség\w*|сообщества?)\b|\W*(?:社[区區群](?:提供)?|提供))?[\W_]*$",
    # Amara.org (optionally plus 'community' in a few languages) as the whole segment
    r"^[\W_]*(?:www\W+)?amara\W*org(?:\W+(?:community|communauté|gemeinschaft|gemeenschap|comunidade|comunidad|comunità|społeczność|közösség|сообщество))?[\W_]*$",
    # 由 Amara.org 社群提供的字幕
    r"^[\W_]*由\W*amara\W*org\W*社[群区區]提供的字幕[\W_]*$",
    # Untertitel im Auftrag des ZDF (für funk), 2017-2021; Untertitelung des ZDF für funk, 2017; Untertitelung des ZDF.;
    # Untertitelung. BR 2018: 'des/der <broadcaster>' or '<broadcaster> <year>' (a plain 'Untertitel für ARD und ZDF.' or
    # 'Untertitel vom ZDF.' is said at work); not asked ('Untertitel des ZDF?' is a question in a meeting)
    r"^[\W_]*untertitel(?:ung)?(?:\W+im\W+auftrag)?\W+(?:(?:des|der)\W+(?:ard|zdf|ndr|wdr|swr|br|mdr|hr|rbb|funk|deutschen\W+welle)|(?:ard|zdf|ndr|wdr|swr|br|mdr|hr|rbb|funk|deutschen\W+welle)\W+(?:19|20)\d\d)(?:\W+(?:des|der|von|vom|für|und|ard|zdf|ndr|wdr|swr|br|mdr|hr|rbb|funk|deutschen\W+welle|(?:19|20)\d\d))*(?:(?!\?)[\W_])*$",
    # Untertitel von Stephanie Geiges
    r"^[\W_]*untertitel(?:ung)?\W+(?:von\W+)?stephanie\W+geiges[\W_]*$",
    # Copyright WDR 2021; (C) TV GELDERLAND 2021; © BF-WATCH TV 2021 (© is removed by the end-stripping)
    r"^[\W_]*(?:copyright|\(?c\)?)\W+(?:wdr|swr|ndr|zdf|ard|br|mdr|tv\W+gelderland|bf\W*watch\W*tv)(?:\W+(?:19|20)\d\d)?[\W_]*$",
    # BF-WATCH TV 2021
    r"^[\W_]*bf\W*watch\W*tv(?:\W+(?:19|20)\d\d)?[\W_]*$",
    # Sous-titrage Société Radio-Canada; Sous-titrage ST' 501; Sous-titres par SousTitreur.com. 'J'écou...
    r"^[\W_]*sous\W?titr(?:age|es?)(?:\W+(?:par|de|la|société|réalisés?|fait|fr))*\W+(?:radio\W?canada|st\W*501|soustitreur\W*com)[\W_]*$",
    # ❤️ par SousTitreur.com (the emoji is removed by the end-stripping)
    r"^[\W_]*(?:par\W+)?soustitreur\W*com[\W_]*$",
    # Sottotitoli (e revisione) a cura di QTSS
    r"^[\W_]*sottotitoli(?:\W+e\W+revisione)?\W+a\W+cura\W+di\W+qtss[\W_]*$",
    # Napisy by Jacek Makarewicz
    r"^[\W_]*napisy\W+(?:by|przez|wykonał|zrobił|stworzył)\W+jacek\W+makarewicz[\W_]*$",
    # Danske tekster af Nicolai Winther; Tekstet av / Norsk teksting av Nicolai Winther
    r"^[\W_]*(?:danske\W+tekster|norsk\W+teksting|tekster|tekstet|teksting|tekstning)\W+(?:af|av|ved)\W+nicolai\W+winther[\W_]*$",
    # Titulky vytvořil JohnyX
    r"^[\W_]*titulky\W+(?:vytvořil|vytvořila|přeložil|připravil)\W+johny\W?x[\W_]*$",
    # Altyazı M.K.; Altyazı M.; Çeviri ve Altyazı M.K. ('Altyazı mı açayım?' is kept)
    r"^[\W_]*(?:çeviri\W+ve\W+)?altyaz[ıi]\W+m(?:\W*k)?[\W_]*$",
    # 'abone ol' repeated 3 or more times as the whole segment
    r"^[\W_]*(?:abone\W+ol\W*){3,}[\W_]*$",
    # ترجمة نانسي قنقر / ترجمة من صفحة نانسي قنقر
    r"^[\W_]*ترجمة\W+(?:من\W+صفحة\W+)?نانسي\W+قنقر[\W_]*$",
    # ترجمة من صفحة <name> (a translation credit naming a Facebook page)
    r"^[\W_]*ترجمة\W+من\W+صفحة(?:\W+\w+){1,3}[\W_]*$",
    # 자막 제공 및 광고를 포함하고 있습니다, including its looped form. '이 영상은 유료 광고를 포함하고 있습니다' is kept.
    r"^[\W_]*(?:자막\W*제공\W*및\W*)+광고를\W*포함하고\W*있습니다[\W_]*$",
    # 请不吝点赞 订阅 转发 打赏支持明镜与点点栏目 (simplified and traditional), and the bare channel name. '我在看明镜与点点' and '...
    r"^[\W_]*(?:[请請]不吝)?(?:[点點][赞贊讚]\W*)?(?:[订訂][阅閱]\W*)?(?:[转轉][发發]\W*)?(?:打[赏賞]\W*)?(?:支持)?明[镜鏡](?:与|與)[点點]{2}(?:[栏欄]目?)?[\W_]*$",
    # 謝謝觀看 記得訂閱 按讚 分享 打賞支持明鏡(與點點)
    r"^[\W_]*(?:[谢謝]{2}[观觀][看赏賞]\W*)?(?:[记記]得|[请請])?(?:[订訂][阅閱]\W*)?(?:按[赞讚贊]\W*)?(?:分享\W*)?(?:[转轉][发發]\W*)?打[赏賞]支持明[镜鏡](?:[与與](?:[点點]{1,2}(?:[栏欄]目?)?)?)?[\W_]*$",
    # 字幕志愿者 杨茜茜优优独播剧场; 优优独播剧场. '字幕志愿者很多' is kept.
    r"^[\W_]*(?:字幕志[愿願]者\W*\w{0,4}?\W*)?[优優]{2}[独獨]播[剧劇][场場][\W_]*$",
    # 字幕提供者 李宗盛; 字幕by索兰娅. Only these known names: '字幕制作很难' is kept.
    r"^[\W_]*字幕(?:提供者|by|志[愿願]者)\W*(?:李宗盛|索[兰蘭][娅婭]|[杨楊]茜茜)[\W_]*$",
    # MING PAO CANADA | MING PAO TORONTO ('MING PAO reported it first.' is kept)
    r"^[\W_]*ming\W+pao(?:\W+(?:canada|toronto|news))*(?:\W+ming\W+pao(?:\W+(?:canada|toronto))?)*[\W_]*$",
    # English credits with the names actually observed: Subtitles by Steamteamextra; Captions by NicoSu...
    r"^[\W_]*(?:subtitles?|subtitled|subtitling|captions?|captioned|captioning|closed\W+captions?(?:ing)?|transcription|transcribed|transcript|translation|translated)(?:\W+(?:and|&)\W+(?:subtitles?|translation|captions?|transcription))?(?:\W+(?:by|provided\W+by|made\W+by|created\W+by|done\W+by))?\W+(?:the\W+)?(?:steam\W*team\W*extra|nico\W?subs|cotter\W+captioning\W+services|get\W*transcribed\W+com|kris\W*brandhagen\W+com|imperial\W+news\W+network|muhsen|eso(?:\W+(?:translation|translated)\W+by)?|(?:https?\W+|www\W+)+otter\W+ai|casting\W?words|emily\W+beynon)[\W_]*$",
    # www.mooji.org as the whole segment ('Visit www.mooji.org for more.' is kept)
    r"^[\W_]*(?:www\W+)?mooji\W+org[\W_]*$",
)]
MADE_UP_MEDIA = [re.compile(p, re.I) for p in (
    # Thanks for watching! / Thank you (so much) for watching (my video) (until the end) (, please subs...
    r"^[\W_]*(?:(?:so|well|okay|ok|and)\W+)?(?:thanks|thank\W+you)(?:\W+(?:so|very)\W+much)?(?:\W+(?:all|everyone|guys))?\W+for\W+watching(?:\W+(?:my|this|the|our)\W+(?:video|channel|stream|episode))?(?:\W+(?:everyone|everybody|guys|all))?(?:\W+(?:until|till|to)\W+the\W+end)?(?:\W+(?:and\W+)?(?:please\W+|don\W?t\W+forget\W+to\W+|remember\W+to\W+)?(?:like\W+and\W+)?subscribe)?(?:\W+bye(?:\W+bye)?)?(?:\W+(?:and\W+)?(?:i\W*ll\W+)?see\W+you\W+(?:in\W+the\W+next\W+(?:one|video)|next\W+time))?[\W_]*$",
    # Like and subscribe; Subscribe to my channel (a bare 'Subscribe (to the channel)' is said at work too: MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:(?:please|don\W?t\W+forget\W+to|remember\W+to)\W+)?(?:like\W+(?:and|&)\W+subscribe(?:\W+to\W+(?:my|our|the)\W+channel)?|subscribe\W+to\W+my\W+channel)[\W_]*$",
    # This is the end of the video, remember to like and subscribe (the bare 'This is the end of the video' is said of a
    # clip shown in a meeting: MADE_UP_DOUBTFUL)
    r"^[\W_]*this\W+is\W+the\W+end\W+of\W+the\W+video\W+remember\W+to\W+like\W+and\W+subscribe[\W_]*$",
    # Sous-titres par <Name> (dozens of names in the HF dataset, mostly under ht/ln/mg language tags)
    r"^[\W_]*sous\W?titres?\W+par\W+(?-i:[A-ZÀ-Ý])[\w.'-]*(?:\W+(?:et\W+)?(?-i:[A-ZÀ-Ý])[\w.'-]*){0,3}[\W_]*$",
    # ترجمة من قبل <name> ('translation by <name>')
    r"^[\W_]*ترجمة\W+من\W+قبل(?:\W+\w+){1,3}[\W_]*$",
    # Altyazıya teşekkür ederim / teşekkürler ('thanks for the subtitles')
    r"^[\W_]*altyazıya\W+teşekkür(?:\W+ederim|ler)[\W_]*$",
    # Продолжение следует / Продолжение следует... (not asked: 'Продолжение следует?')
    r"^[\W_]*продолжение\W+следует(?:\W+продолжение\W+следует)*(?:(?!\?)[\W_])*$",
    # Спасибо за просмотр / Спасибо за субтитры ('Спасибо за внимание' is said in meetings: MADE_UP_DOUBTFUL)
    r"^[\W_]*спасибо\W+за\W+(?:просмотр(?:\W+(?:этого\W+)?видео)?|субтитры)(?:\W+(?:и\W+)?до\W+новых\W+встреч)?[\W_]*$",
    # Подписывайтесь на мой канал (и ставьте лайки); Подписывайтесь на канал и ставьте лайки (without 'мой' or the
    # likes it is said at work too, 'Подпишись на канал': MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:подписывайтесь|подпишись|подписывайся|не\W+забудьте\W+подписаться)(?:\W+на\W+мой\W+канал(?:\W+и\W+ставьте\W+лайки?)?|(?:\W+на\W+(?:наш\W+)?канал)?(?:\W+(?:и\W+)?(?:ставьте\W+лайки?|жмите\W+(?:на\W+)?колокольчик)){1,2})[\W_]*$",
    # Ставьте лайк(и)
    r"^[\W_]*ставь(?:те)?\W+лайки?[\W_]*$",
    # Смотрите продолжение в следующей / во второй / в 4 части (видео)
    r"^[\W_]*смотрите\W+продолжение\W+во?\W+(?:\w+\W+)?част[иь](?:\W+видео)?[\W_]*$",
    # Субтитры <verb> <1-3 capitalised words> with an unknown name (DimaTorzok and «Симон» are in the a...
    r"^[\W_]*субтитры\W+(?:сделал|сделала|делал|делала|создал|создала|создавал|создавала|добавил|добавила|подогнал|подготовил|подготовила|выполнил|перев[её]л)\W+(?-i:[A-ZА-ЯЁ])[\w.-]*(?:\W+(?-i:[A-ZА-ЯЁ])[\w.-]*){0,2}[\W_]*$",
    # Дякую за перегляд!
    r"^[\W_]*дякую\W+за\W+перегляд(?:\W+(?:і\W+)?до\W+зустрічі)?[\W_]*$",
    # Підписуйтесь на мій канал ('… на (наш) канал' is a work channel too: MADE_UP_DOUBTFUL)
    r"^[\W_]*підписуйтесь\W+на\W+мій\W+канал[\W_]*$",
    # Субтитрувальниця <Name> with a name other than Оля Шор
    r"^[\W_]*субтитрувальни(?:ця|к)\W+(?-i:[А-ЯІЇЄҐ])\w*(?:\W+(?-i:[А-ЯІЇЄҐ])\w*)?[\W_]*$",
    # Vielen Dank fürs / für's Zuschauen; Danke fürs Ansehen; Das war's für heute. Vielen Dank fürs Zus...
    r"^[\W_]*(?:das\W+war\W*s\W+(?:für\W+heute|es)\W+)?(?:vielen\W+)?danke?(?:\W+schön)?\W+f(?:ü|ue)r\W*s?\W+(?:das\W+)?(?:zuschauen|zusehen|ansehen)(?:\W+und\W+bis\W+zum\W+nächsten\W+mal)?(?:\W+tschüss)?[\W_]*$",
    # Abonniere meinen Kanal ('Abonniert den Kanal', 'Bitte abonnieren': MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:abonniere|abonniert)\W+meinen\W+kanal[\W_]*$",
    # SWR 2021; ZDF, 2021 (not asked: 'SWR 2021?')
    r"^[\W_]*(?:swr|wdr|ndr|zdf|br|mdr)\W*(?:19|20)\d\d(?:(?!\?)[\W_])*$",
    # Bedankt voor het kijken
    r"^[\W_]*bedankt\W+voor\W+het\W+kijken[\W_]*$",
    # TV Gelderland 2021 (the copyright-prefixed form is always dropped)
    r"^[\W_]*tv\W+gelderland(?:\W+(?:19|20)\d\d)?[\W_]*$",
    # Merci d'avoir regardé (cette vidéo)
    r"^[\W_]*merci\W+d\W*avoir\W+regardé(?:\W+(?:cette|la)\W+vidéo)?[\W_]*$",
    # Abonnez-vous à ma chaîne; N'oubliez pas de vous abonner à ma chaîne (bare, 'la' or 'notre': MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:abonnez\W*vous|n\W*oubliez\W+pas\W+de\W+vous\W+abonner)\W+à\W+ma\W+chaîne[\W_]*$",
    # ¡Gracias por ver (el vídeo)!
    r"^[\W_]*(?:muchas\W+)?gracias\W+por\W+ver(?:\W+el\W+v[ií]deo)?[\W_]*$",
    # ¡Suscríbete a mi canal!; ¡Suscríbete y dale like! (bare or 'al canal': MADE_UP_DOUBTFUL)
    r"^[\W_]*suscr[ií]bete\W+(?:a\W+mi\W+canal|y\W+dale\W+(?:like|me\W+gusta))[\W_]*$",
    # Grazie per la visione / per aver guardato
    r"^[\W_]*grazie\W+per\W+(?:la\W+visione|aver\W+guardato(?:\W+il\W+video)?)[\W_]*$",
    # Iscriviti al mio canale; Non dimenticare di iscriverti al mio canale (bare or 'al canale': MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:iscriviti|non\W+dimenticare\W+di\W+iscriverti)\W+al\W+mio\W+canale[\W_]*$",
    # Obrigado/Obrigada por assistir
    r"^[\W_]*obrigad[oa]\W+por\W+assistir(?:\W+ao\W+v[ií]deo)?[\W_]*$",
    # Inscreva-se no meu canal ('no canal': MADE_UP_DOUBTFUL)
    r"^[\W_]*inscreva\W*se\W+no\W+meu\W+canal[\W_]*$",
    # nu uitati sa da-ti like si subscribe
    r"^[\W_]*nu\W+uita[tț]i\W+s[aă]\W+(?:da\W*[tț]i|dați)\W+like\W+(?:și|si)\W+subscribe[\W_]*$",
    # Dziękuję/Dzięki za oglądanie ('za uwagę' is said in meetings: MADE_UP_DOUBTFUL)
    r"^[\W_]*dzi[ęe]k(?:uj[ęe]|i)\W+za\W+(?:oglądanie|obejrzenie)[\W_]*$",
    # Subskrybuj mój kanał (bare or 'nasz kanał': MADE_UP_DOUBTFUL)
    r"^[\W_]*(?:za)?subskrybuj\W+mój\W+kanał[\W_]*$",
    # İzlediğiniz için teşekkür ederim (Python re.I equates İ/i and I/ı, tested)
    r"^[\W_]*izlediğiniz\W+için\W+teşekkür(?:\W+eder(?:im|iz)|ler)[\W_]*$",
    # (Kanalıma) abone olmayı unutmayın
    r"^[\W_]*(?:kanalıma\W+|kanalımıza\W+)?abone\W+olmayı\W+unutmayın[\W_]*$",
    # Bir sonraki videoda görüşmek üzere
    r"^[\W_]*bir\W+sonraki\W+videoda\W+görüşmek\W+üzere[\W_]*$",
    # Ευχαριστώ που παρακολουθήσατε
    r"^[\W_]*ευχαριστώ\W+(?:πολύ\W+)?που\W+παρακολουθήσατε[\W_]*$",
    # ご視聴ありがとうございました/ます; 最後までご視聴いただきありがとうございました
    r"^[\W_]*(?:最後まで)?ご視聴(?:いただき|頂き)?(?:本当に)?ありがとうございま(?:した|す)[\W_]*$",
    # (高評価と)チャンネル登録(を)お願いします
    r"^[\W_]*(?:高評価と)?チャンネル登録(?:を)?(?:よろしく)?お願いします[\W_]*$",
    # (また)次(回)の動画でお会いしましょう
    r"^[\W_]*(?:また)?次(?:回)?の動画でお会いしましょう[\W_]*$",
    # 見てくれてありがとう
    r"^[\W_]*見てくれてありがとう(?:ございました)?[\W_]*$",
    # 시청해(  )주셔서 감사합니다
    r"^[\W_]*시청해\s?주셔서\s?감사합니다[\W_]*$",
    # 구독과 좋아요 부탁드려요; 구독해주세요; 구독과 좋아요는 사랑입니다
    r"^[\W_]*(?:구독과\s?좋아요\s?(?:부탁드려요|부탁드립니다|는\s?사랑입니다)|구독해\s?주세요)[\W_]*$",
    # MBC 뉴스 이덕영/김재경/이재경/김지경입니다; 이 시각 세계였습니다
    r"^[\W_]*(?:mbc\s?뉴스\s?\w{2,4}입니다|이\s?시각\s?세계였습니다)[\W_]*$",
    # 谢谢观看 (欢迎订阅我的频道 / 下集再见); 感谢收看
    r"^[\W_]*(?:谢谢|感谢|謝謝|感謝)(?:大家)?(?:观看|收看|觀看)(?:\W*(?:欢迎|歡迎)?(?:订阅|訂閱)(?:我的)?(?:频道|頻道))?(?:\W*下集再[见見])?[\W_]*$",
    # (请)订阅我的频道
    r"^[\W_]*(?:[请請])?(?:订阅|訂閱)(?:我的)?(?:频道|頻道)[\W_]*$",
    # شكرا(ً) على المشاهدة / للمشاهدة
    r"^[\W_]*شكرا\W*(?:على\W+المشاهدة|للمشاهدة|لكم\W+على\W+المشاهدة)[\W_]*$",
    # اشتركوا في القناة
    r"^[\W_]*(?:اشتركوا|اشترك)\W+في\W+القناة[\W_]*$",
    # תודה שצפיתם / תודה על הצפייה / הירשמו לערוץ
    r"^[\W_]*(?:תודה\W+(?:שצפיתם|על\W+הצפייה)|הירשמו\W+לערוץ)[\W_]*$",
    # (I'll / I can / we'll) see you in the next video / episode (made up 4 times on voice grains in the 1.11.4 harvest;
    # '… in the next one' is the next meeting too: MADE_UP_MEETING)
    r"^[\W_]*(?:(?:and|so|okay|ok|but|well)\W+)?(?:i\W*ll|i\W+will|we\W*ll|i\W+can|we\W+can)?\W*see\W+you\W+(?:all\W+|guys\W+)?in\W+the\W+next\W+(?:video|episode)[\W_]*$",
    # Подпишитесь / Подписаться на мой канал (HF sachaarbonel, tinglongwu; without 'мой': MADE_UP_MEETING)
    r"^[\W_]*(?:подпишитесь|подписаться)\W+на\W+мой\W+канал[\W_]*$",
    # Ставьте лайки и подписывайтесь (на канал) (Lapsus, HF sachaarbonel; the other order is above)
    r"^[\W_]*ставьте\W+лайки?\W+(?:и\W+)?подписывайтесь(?:\W+на\W+(?:мой\W+|наш\W+)?канал)?[\W_]*$",
    # Подпишись на канал, ставь лайк (и жми колокольчик) (HF sachaarbonel)
    r"^[\W_]*подпишись\W+на\W+канал\W+ставь\W+лайк(?:\W+и\W+жми\W+колокольчик)?[\W_]*$",
    # (Всем привет и) добро пожаловать на мой канал; Hello everyone, welcome to my channel; Hallo zusammen und willkommen
    # auf meinem Kanal (HF sachaarbonel, AGH, Sharrnah)
    r"^[\W_]*(?:всем\W+привет\W+и\W+)?добро\W+пожаловать\W+на\W+(?:мой|наш)\W+канал[\W_]*$",
    r"^[\W_]*(?:hello|hi)\W+(?:everyone|everybody|guys)\W+(?:and\W+)?welcome\W+(?:back\W+)?to\W+(?:my|our)\W+channel[\W_]*$",
    r"^[\W_]*welcome\W+(?:back\W+)?to\W+my\W+channel[\W_]*$",
    r"^[\W_]*hallo\W+zusammen\W+und\W+willkommen\W+auf\W+meinem\W+kanal[\W_]*$",
    # Смотрите другие видео; Спасибо, что смотрели; Благодарим за субтитры (HF sachaarbonel, tinglongwu)
    r"^[\W_]*смотрите\W+другие\W+видео[\W_]*$",
    r"^[\W_]*спасибо\W+что\W+(?:смотрели|посмотрели\W+(?:это\W+|наше\W+|мо[её]\W+)?видео)[\W_]*$",
    r"^[\W_]*благодарим\W+за\W+субтитры[\W_]*$",
    # Субтитры подготовлены / сделаны / созданы <1-3 capitalised words>, a credit (Lapsus; bare, or with a lower-case
    # tail such as 'для второго ролика', it is a status said at work: MADE_UP_MEETING)
    r"^[\W_]*субтитры\W+(?:подготовлены|сделаны|созданы)\s+(?-i:[A-ZА-ЯЁ])[\w.-]*(?:\s+(?-i:[A-ZА-ЯЁ])[\w.-]*){0,2}[\W_]*$",
    # Ich hoffe, euch hat das Video gefallen; (Wenn ja, dann) lasst mir einen Daumen da (und abonniert meinen Kanal) (Sharrnah)
    r"^[\W_]*ich\W+hoffe\W+euch\W+hat\W+das\W+video\W+gefallen[\W_]*$",
    r"^[\W_]*(?:wenn\W+ja\W+)?(?:dann\W+)?lasst\W+mir\W+(?:einen\W+)?daumen\W+da(?:\W+und\W+abonniert\W+meinen\W+kanal)?[\W_]*$",
    # Fortsetzung folgt (tinglongwu); Продовження буде (the Ukrainian 'Продолжение следует'); not asked ('…?')
    r"^[\W_]*fortsetzung\W+folgt(?:(?!\?)[\W_])*$",
    r"^[\W_]*(?:wir\W+sehen\W+uns\W+im|bis\W+zum)\W+nächsten\W+video[\W_]*$",
    r"^[\W_]*продовження\W+буде(?:(?!\?)[\W_])*$",
    # Sound captions written without brackets: (Piano) music continues / playing; Bell rings; Birds chirping (discussion
    # 1873); not asked ('Music playing?', 'Phone rings?')
    r"^[\W_]*(?:(?:soft|piano|upbeat|gentle|dramatic)\W+)?music\W+(?:continues|playing|plays|fades)(?:(?!\?)[\W_])*$",
    r"^[\W_]*(?:bell|phone|doorbell)\W+rings(?:(?!\?)[\W_])*$",
    r"^[\W_]*birds?\W+(?:chirp|chirps|chirping|singing)(?:(?!\?)[\W_])*$",
)]
MADE_UP_DOUBTFUL = [re.compile(p, re.I) for p in (
    # I'm not sure what I'm doing here
    r"^[\W_]*i\W*m\W+not\W+sure\W+what\W+i\W*m\W+doing\W+here[\W_]*$",
    # This is the end of the video (also said of a clip shown in a meeting; with the plug after it: MADE_UP_MEDIA)
    r"^[\W_]*this\W+is\W+the\W+end\W+of\W+the\W+video(?:(?!\?)[\W_])*$",
    # Спасибо за внимание
    r"^[\W_]*спасибо\W+за\W+внимание[\W_]*$",
    # Dziękuję/Dzięki za uwagę
    r"^[\W_]*dzi[ęe]k(?:uj[ęe]|i)\W+za\W+uwagę[\W_]*$",
    # Subtitles/Captions/Transcribed by <1-3 capitalised words> (unknown names, Transcribed by Otter.ai; also 'Captions by
    # Friday', 'Transcription by Whisper' in a meeting, so not dropped by its text)
    r"^[\W_]*(?:subtitles?|captions?|closed\W+captions?(?:ing)?|transcribed|transcription)\W+(?:by|provided\W+by|made\W+by)\W+(?:(?-i:[A-Z])[\w.-]*)(?:\W+(?-i:[A-Z])[\w.-]*){0,2}[\W_]*$",
    # Please subscribe; Subscribe (to our/the channel) (a button, a Slack or Teams channel; the plugs are MADE_UP_MEDIA)
    r"^[\W_]*(?:(?:please|don\W?t\W+forget\W+to|remember\W+to)\W+)?(?:like\W+(?:and|&)\W+)?subscribe(?:\W+to\W+(?:my|our|the)\W+channel)?[\W_]*$",
    # Подписывайтесь / Подпишись на (наш) канал; ПОДПИШИСЬ!; Не забудьте подписаться
    r"^[\W_]*(?:подписывайтесь|подпишись|подписывайся|не\W+забудьте\W+подписаться)(?:\W+на\W+(?:мой\W+|наш\W+)?канал)?(?:\W+и\W+ставьте\W+лайки?)?[\W_]*$",
    # Abonniert den / unseren Kanal; Bitte abonnieren
    r"^[\W_]*(?:(?:abonniere|abonniert)\W+(?:meinen|unseren|den)\W+kanal|bitte\W+abonnieren)[\W_]*$",
    # The bare, 'our' and 'the' channel plug in Ukrainian, French, Spanish, Italian, Portuguese, Polish (a work channel
    # too; the 'my channel' forms are MADE_UP_MEDIA)
    r"^[\W_]*підписуйтесь\W+на\W+(?:наш\W+)?канал[\W_]*$",
    r"^[\W_]*(?:abonnez\W*vous|n\W*oubliez\W+pas\W+de\W+vous\W+abonner)(?:\W+à\W+(?:la|notre)\W+chaîne)?[\W_]*$",
    r"^[\W_]*suscr[ií]bete(?:\W+al\W+canal)?[\W_]*$",
    r"^[\W_]*(?:iscriviti|non\W+dimenticare\W+di\W+iscriverti)(?:\W+al\W+canale)?[\W_]*$",
    r"^[\W_]*inscreva\W*se\W+no\W+canal[\W_]*$",
    r"^[\W_]*(?:za)?subskrybuj(?:\W+nasz\W+kanał)?[\W_]*$",
)]
# Phrases a meeting says as well, first listed in 1.11.4: dropped like MADE_UP_DOUBTFUL only with a sign that nobody said
# them, but never for coming again: several people on one track say "See you tomorrow." at the end of a call.
MADE_UP_MEETING = [re.compile(p, re.I) for p in (
    # Спасибо за внимание и до новых встреч; До новых встреч (Lapsus, HF sachaarbonel)
    r"^[\W_]*спасибо\W+за\W+внимание\W+и\W+до\W+новых\W+встреч[\W_]*$",
    r"^[\W_]*до\W+новых\W+встреч[\W_]*$",
    r"^[\W_]*(?:дякую\W+за\W+увагу|до\W+зустрічі)[\W_]*$",
    # (Okay,) (I'll) see you next time / soon / later / next week / tomorrow (the 1.11.4 harvest: 'I'll see you next time.')
    r"^[\W_]*(?:(?:okay|ok|so|and|well|alright|all\W+right)\W+)?(?:i\W*ll|we\W*ll|i\W+will)?\W*see\W+you\W+(?:all\W+|guys\W+)?(?:next\W+time|soon|later|next\W+week|tomorrow)[\W_]*$",
    # Bis zum nächsten Mal (a goodbye several people say, like 'Until next time')
    r"^[\W_]*bis\W+zum\W+nächsten\W+mal[\W_]*$",
    # Das war's für heute (Sharrnah)
    r"^[\W_]*das\W+war\W*s\W+für\W+heute[\W_]*$",
    # (So) until next time (, bye for now) (AGH)
    r"^[\W_]*(?:so\W+)?until\W+next\W+time(?:\W+bye(?:\W+for\W+now)?)?[\W_]*$",
    # (I'll) see you in the next one (the next meeting as well as the next video)
    r"^[\W_]*(?:(?:and|so|okay|ok|but|well)\W+)?(?:i\W*ll|i\W+will|we\W*ll|i\W+can|we\W+can)?\W*see\W+you\W+(?:all\W+|guys\W+)?in\W+the\W+next\W+one[\W_]*$",
    # Подпишитесь / Подписаться на (наш) канал (a work channel too)
    r"^[\W_]*(?:подпишитесь|подписаться)\W+на\W+(?:наш\W+)?канал[\W_]*$",
    # Субтитры подготовлены / сделаны / созданы, with nothing after it (a status: 'the subtitles are done')
    r"^[\W_]*субтитры\W+(?:подготовлены|сделаны|созданы)[\W_]*$",
)]
# Short thanks, goodbyes and one-word captions: the commonest lines Whisper makes up on non-speech ('Thank you.' 24.76 %
# of large-v3's, Barański et al. 2025, Table III; turbo: 'Vielen Dank', 'Thank you', 'Musik'), and real replies as often.
MADE_UP_SHORT = [re.compile(p, re.I) for p in (
    r"^[\W_]*(?:thank\W+you|thanks)(?:\W+(?:so|very)\W+much)?(?:\W+(?:all|everyone|guys))?[\W_]*$",
    r"^[\W_]*(?:(?:большое|огромное)\W+)?спасибо(?:\W+(?:большое|огромное|всем))?[\W_]*$",
    r"^[\W_]*всем\W+спасибо[\W_]*$",
    r"^[\W_]*(?:(?:щиро|дуже)\W+)?дякую(?:\W+(?:вам|всім|дуже))?[\W_]*$",
    r"^[\W_]*(?:vielen\W+)?dank(?:e)?(?:\W+(?:schön|sehr))?[\W_]*$",
    r"^[\W_]*(?:muchas\W+)?gracias[\W_]*$",
    r"^[\W_]*(?:muito\W+)?obrigad[oa][\W_]*$",
    r"^[\W_]*merci(?:\W+beaucoup)?[\W_]*$",
    r"^[\W_]*grazie(?:\W+mille)?[\W_]*$",
    r"^[\W_]*(?:bye|bye\W+bye|tschüss|(?:всем\W+)?пока|пока\W+пока|до\W+свидания|до\W+побачення|бувайте)[\W_]*$",
    r"^[\W_]*(?:музыка|musik|music)[\W_]*$",
)]
# A whole line of sound captions in brackets or asterisks: [Music], [APPLAUSE], (Musik), * Musik *, *Klopfen*, ♪♪, ♪ Musik ♪
# (discussion 1873; marmalade). Whisper does not write speech inside brackets; it does write sung words between ♪, so
# between ♪ only one word counts, or two when one is a music word ('♪ Sanfte Musik ♪'): '♪ Happy birthday ♪' and
# '♪ Happy birthday to you ♪' are a team singing on a call.
_W = r"[^\s\[\]()*♪]+"
_MUSIC = r"(?i:music|musik|музыка|musique|m[uú]sica)"
MADE_UP_CAPTION = re.compile(r"^\s*(?:(?:[\[(*]\s*[^\[\]()*♪]{1,40}?\s*[\])*]|♪\s*(?:" + _MUSIC + r"\s+" + _W + r"|" + _W + r"\s+"
                             + _MUSIC + r"|" + _W + r")?\s*♪)\s*)+$"
                             r"|^\s*[♪♫¶]+\s*$")
# A line timed this far past the end of its audio: Whisper timed words in the zero padding. Lines made up on noise end at
# the end of Whisper's 30 s window (5.8-29 s past the audio in the 1.11.4 harvest and noise runs); real words of a
# 1-1.4 s chunk can end at a round second after it (up to 1.62 s past the audio measured: 'барсук' on a 1.37 s chunk).
OVERHANG_S = 5.0


def _made_up_kind(sentence):
    """'always', 'media', 'doubtful', 'meeting', 'short' or None for one sentence (NFKC, spaces collapsed)."""
    plain = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", sentence)).strip()
    if not plain:
        return None
    if MADE_UP_CAPTION.match(plain) or any(p.match(plain) for p in MADE_UP_ALWAYS):
        return "always"
    # A question is never a video's phrase ('Thanks for watching?', 'Субтитры сделал Иван?': asked in a meeting)
    asked = "?" in re.search(r"[\W_]*$", plain).group()
    if not asked and any(p.match(plain) for p in MADE_UP_MEDIA):
        return "media"
    if any(p.match(plain) for p in MADE_UP_DOUBTFUL):
        return "doubtful"
    if any(p.match(plain) for p in MADE_UP_MEETING):
        return "meeting"
    if any(p.match(plain) for p in MADE_UP_SHORT):
        return "short"
    return None


def _sentences(text):
    """A line's sentences: cut after . ! ? … followed by a space (and after 。！？), not after an initial ("А. Синецкая")."""
    parts, start = [], 0
    for m in re.finditer(r"[.!?…]+\s+|[。！？]+", text):
        if text[m.start()] == "." and re.search(r"(?:^|\s)\w$", text[start:m.start()]):
            continue                                   # a single letter and a dot: an initial, not a sentence end
        parts.append(text[start:m.end()].strip())
        start = m.end()
    if text[start:].strip():
        parts.append(text[start:].strip())
    return [p for p in parts if p]


def _sentence_kinds(text):
    """The kinds of a line's sentences; the last two count as one credit when joined they are one ("Untertitelung. BR
    2018", which _sentences cuts in two)."""
    parts = _sentences(text)
    kinds = [_made_up_kind(s) for s in parts] or [None]
    if len(parts) >= 2 and _made_up_kind(" ".join(parts[-2:])) == "always":
        kinds[-2:] = ["always", "always"]
    return kinds


def _stretched(line):
    """Whether a line lasts at least 5 s and 2.5 times as long as its words take to say (about 12 letters a second;
    5 characters a second for Han, Kana and Hangul): a short phrase Whisper spread over near-silence. Not 3 s: on a
    1-1.4 s chunk Whisper ends real words at a round 2 or 3 s after their start ('\u0431\u0430\u0440\u0441\u0443\u043a' at 1.1-4.1 s, 1.11.4); the
    outros it made up on noise lasted about 30 s."""
    seconds = (line.get("end") or line["t"]) - line["t"]
    text = line.get("text", "")
    wide = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))
    letters = len(re.findall(r"\w", text)) - wide
    return seconds >= 5.0 and seconds >= 2.5 * (letters / 12 + wide / 5)


def _overhangs(line, audio_end):
    """Whether Whisper timed the line OVERHANG_S or more past the end of its chunk's audio, into the padding: words where
    there was no sound (see OVERHANG_S)."""
    return audio_end is not None and (line.get("end") or line["t"]) - audio_end >= OVERHANG_S


def _failed(line):
    """Whether the line's decode still failed Whisper's own thresholds (average log-probability -1.0, compression ratio
    2.4) after the temperature fallback: the engine kept a result it had rejected. Only lines just decoded carry these
    values (_lp, _temp, _cr: segment_lines), never a line read from a checkpoint."""
    if (line.get("_temp") or 0.0) <= 0:
        return False
    return (line.get("_lp") is not None and line["_lp"] < -1.0) or (line.get("_cr") or 0.0) > 2.4


def made_up(line, alone=True, audio_end=None, context="media"):
    """Why a spoken line is one Whisper made up on noise, or None. alone: it is the only spoken line of its chunk;
    audio_end: where the audio of its chunk (or piece) ends, in seconds (None: not known, as for a checkpoint's lines);
    context: "recording" (a recording or a call: a video's outro or channel plug is dropped by its text) or "media" (an
    imported file, which may be such a video: only with a sign that nobody said it). A meeting's outro and a short thanks
    are never dropped by their text alone."""
    outro = ("media", "doubtful", "meeting")
    text = line.get("text") or ""
    kinds = _sentence_kinds(text)
    whole = _made_up_kind(text)
    if whole == "always" or (all(k in ("always", "media") for k in kinds) and "always" in kinds):
        return "a subtitle credit"
    media = whole == "media" or all(k == "media" for k in kinds)
    if context == "recording" and media:
        return "a video's outro or channel plug"
    over, stretched = _overhangs(line, audio_end), alone and _stretched(line)
    if media and (stretched or over):
        return "an outro over near-silence"
    if media and _failed(line):
        return "a video's phrase Whisper itself rejected"
    # A meeting's outro ("Спасибо за внимание") that was mumbled is still said: a failed decode is no sign for it
    if (whole in outro or all(k in outro for k in kinds)) and (stretched or over):
        return "an outro over near-silence"
    short = whole == "short" or (all(k in ("short", *outro) for k in kinds) and "short" in kinds)
    if short and alone and over:
        return "thanks over near-silence"
    return None


def _trim_credits(text, over=False, recording=False):
    """A real line without the credit sentences Whisper put before or after it ("Доклад окончен. Субтитры сделал …").
    over: the line runs past the end of its audio, so an outro sentence at its end comes off too. recording: a
    recording or a call, where a video's phrase at its end comes off without that sign (as a whole line of one is
    dropped by its text there); a meeting's outro and a short thanks still need it."""
    parts = _sentences(text)
    while len(parts) > 1:
        last = _made_up_kind(parts[-1])
        if last == "always":
            parts.pop()
        elif len(parts) > 2 and _made_up_kind(" ".join(parts[-2:])) == "always":
            del parts[-2:]                             # a credit cut in two ("Untertitelung." + "BR 2018", itself a video's tag)
        elif (over and last in ("media", "doubtful", "meeting")) or (recording and last == "media"):
            parts.pop()
        else:
            break
    while len(parts) > 1 and _made_up_kind(parts[0]) == "always":
        parts.pop(0)
    return " ".join(parts) if parts else text


def screen_made_up(lines, audio_end=None, context="media"):
    """The lines of one chunk (or piece), with those Whisper made up on noise marked (`made_up`: why) and credit
    sentences trimmed off the ends of real ones. The marked lines are kept in the checkpoint (they are counted at the
    end) but never shown, translated or written. audio_end: where the chunk's audio ends, in seconds (decode_chunk
    passes it; see made_up). The values only a fresh decode has (_lp, _temp, _cr) are removed. Running it again on the
    same lines changes nothing."""
    spoken = [l for l in lines if "text" in l]
    for line in spoken:
        if line.get("made_up"):
            continue
        why = made_up(line, alone=len(spoken) == 1, audio_end=audio_end, context=context)
        if not why:
            trimmed = _trim_credits(line["text"], _overhangs(line, audio_end), context == "recording")
            if trimmed != line["text"]:                # what is left is screened too ('Спасибо.' timed 28 s past its audio)
                line["text"] = trimmed
                why = made_up(line, alone=len(spoken) == 1, audio_end=audio_end, context=context)
        if why:
            line["made_up"] = why
    for line in lines:
        for key in ("_lp", "_temp", "_cr"):
            line.pop(key, None)
    return lines


def without_made_up(lines):
    return [l for l in lines if not l.get("made_up")]


def outro_key(line):
    """What makes two outros the same, for counting them (case, punctuation and spacing aside)."""
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", line["text"]).casefold()).strip()


def is_outro(line):
    """An outro or a video's phrase, which a real meeting or video says once, not again and again (a short thanks is
    not one: three real "Thank you." on one track are normal)."""
    return "text" in line and _made_up_kind(line["text"]) in ("doubtful", "media")


REPEATED = "an outro said again and again"


def drop_repeated_outros(lines, times=3, left_out=()):
    """The track's lines without the outros that come `times` times or more (Whisper's "Продолжение следует" every 30 s
    on noise, openai/whisper discussion 1873; a real video says its outro once), and how many were left out.
    left_out: the made-up lines already taken out of `lines` (an outro spread over near-silence); they count toward the
    `times` but are not counted again among those left out."""
    counts = collections.Counter(outro_key(l) for l in [*lines, *left_out] if is_outro(l))
    repeated = {k for k, n in counts.items() if n >= times}
    kept = [l for l in lines if not (is_outro(l) and outro_key(l) in repeated)]
    return kept, len(lines) - len(kept)


def mark_repeated_outros(track, lines, times=3):
    """drop_repeated_outros while recording, for one live track and the lines of its latest chunk: the outros are counted
    by outro_key (the hidden ones too, as left_out is at the end); the copy that reaches `times` and every one after it
    are marked made up, and so are the earlier copies not yet marked. Returns whether a copy shown in an earlier chunk
    was marked: the window is then sent its lines again (LiveWorker.resend); copies earlier in these same lines were
    never shown, so marking them needs no resend. The earlier chunks' copies were written to the checkpoint unmarked:
    LiveTrack.decode writes those chunks again with the mark, so the final pass neither shows nor sends them."""
    state = vars(track)                                # made on first use (a track built without __init__, as in a test)
    counts = state.setdefault("outro_count", collections.Counter())
    shown = state.setdefault("shown_outros", collections.defaultdict(list))
    here = {id(line) for line in lines}                # this chunk's lines: not on screen yet
    withdrawn = False
    for line in lines:
        if not is_outro(line):
            continue
        k = outro_key(line)
        counts[k] += 1
        if counts[k] >= times:
            if not line.get("made_up"):
                line["made_up"] = REPEATED
            for earlier in shown.pop(k, []):
                earlier["made_up"] = REPEATED
                withdrawn = withdrawn or id(earlier) not in here
        elif not line.get("made_up"):
            shown[k].append(line)
    return withdrawn


def made_up_note(count, position=None):
    return (f"{count} line{'s' if count != 1 else ''} that Whisper makes up on noise (subtitle credits, outros said again and "
            f"again) {'was' if count == 1 else 'were'} left out" + (f" on track {position[0]}" if position else ""))


def segment_lines(segs, a, lang, conf, guess, speaker):
    """The lines of decoded segments that start at sample a. What the engine reported of each decode (log-probability,
    temperature, compression ratio) goes along as _lp, _temp, _cr for screen_made_up, which removes them."""
    new = []
    for s in segs:
        if s["text"].strip():
            line = {"t": round(a / RATE + s["start"], 1), "end": round(a / RATE + s["end"], 1), "lang": lang, "conf": conf,
                    "text": s["text"].strip()}
            if guess:
                line["guess"] = True
            if speaker:
                line["speaker"] = speaker
            for key, signal in (("logprob", "_lp"), ("temp", "_temp"), ("cr", "_cr")):
                if s.get(key) is not None:
                    line[signal] = s[key]
            new.append(line)
    return new


def decode_chunk(engine, chunk, a, allowed, forced, speaker, adaptive=False, context="media"):
    """The lines of one chunk starting at sample a: language decision, decoding, notes for a chunk without a result.
    adaptive: when the chunk's language probabilities look mixed (looks_mixed), it is re-cut at the finest settings
    (pauses of 200 ms, pieces of at least 1 s) and, when that gives two or more pieces, each piece gets its own language
    decision and decoding, so a switch inside the chunk gets its own tag; with one piece the chunk is decoded as it is.
    The re-cut is a VAD pass over at most 30 s (well under a second); each piece is one more Whisper pass.
    context: "recording" (a recording or a call) or "media" (an imported file), for screen_made_up."""
    state = engine.analyse(chunk, detect=forced is None)
    if adaptive and not forced and looks_mixed(state.probs):
        spans = speech_spans(chunk, 0, GRANULARITY["finest"][1])
        pieces = [tuple(p) for p in group_spans(spans, min_s=GRANULARITY["finest"][0])]
        if len(pieces) >= 2:
            new = []
            for ps, pe in pieces:
                piece = engine.analyse(chunk[ps:pe], detect=True)
                segs, lang, conf, guess = decode_unit(engine, piece, allowed, forced)
                lines = screen_made_up(segment_lines(segs, a + ps, lang, conf, guess, speaker), (a + pe) / RATE, context)
                if not lines:                          # a piece that gave nothing is said, like a chunk that gave nothing
                    lines = [{"t": round((a + ps) / RATE, 1), "note": f"speech not recognised for {round((pe - ps) / RATE)} s"}]
                new.extend(lines)
            return new
    segs, lang, conf, guess = decode_unit(engine, state, allowed, forced)
    new = screen_made_up(segment_lines(segs, a, lang, conf, guess, speaker), (a + len(chunk)) / RATE, context)
    if not new:
        new.append({"t": round(a / RATE, 1), "note": f"speech not recognised for {round(len(chunk) / RATE)} s"})
    return new


def header_notes(notes):
    """Notes given from outside (--note): at the top of the transcript, or, with a leading [hh:mm:ss], at that time."""
    out = []
    for n in notes or []:
        if not n or not n.strip():
            continue
        m = re.match(r"\[(\d+):(\d\d):(\d\d)\]\s+(.+)$", n.strip())
        if m:
            h, mi, sec, text = m.groups()
            out.append({"t": int(h) * 3600 + int(mi) * 60 + int(sec), "note": text})
        else:
            out.append({"t": 0, "note": n})
    return out


def _keep_outputs(d, doc):
    """A meeting transcribed again in its data folder keeps the record of where its files were written (doc["outputs"]),
    so the new files replace them under the same name and `rename` still finds them."""
    try:
        earlier = _load(d).get("outputs") if (d / "transcript.json").exists() else None
    except Exception:
        earlier = None
    if earlier:
        doc["outputs"] = earlier


def in_time_order(lines):
    """Lines by time; a stable sort, so what shares a second keeps its order (notes before the speech at that second)."""
    return sorted(lines, key=lambda l: l["t"])


def screening_context(src, d):
    """"recording" for a meeting this program recorded (record transcribing itself, or the app's queued job: the
    audio.wav in its own data folder), else "media": an imported file, which may be a published video, whose outros and
    channel plugs are then really said (screen_made_up)."""
    try:
        return "recording" if Path(src).name == "audio.wav" and Path(src).resolve().parent == Path(d).resolve() else "media"
    except (OSError, RuntimeError, TypeError):
        return "media"


def transcribe(src, d, language=None, track=None, granularity="auto", notes=None, speakers=None, diarizer="sherpa"):
    """The transcript of one file, saved in the data folder d; a run cut short is continued from its checkpoint there.
    speakers ("auto" or a number) labels who spoke."""
    if EVENTS:
        ev("FOLDER", d)
    lines = in_time_order(header_notes(notes) + transcribe_lines(src, language, track, granularity, checkpoint=d, speakers=speakers,
                                                                 diarizer=diarizer, context=screening_context(src, d)))
    doc = {"source": str(src), "whisper": WHISPER, "lines": lines, "translations": {}}
    _keep_outputs(d, doc)
    _save(d, doc)
    Checkpoint(d).finish()
    return doc


# ---------- a call: the microphone and the other side, as two tracks ----------
def follow(paths, d, language=None, granularity="auto", me="You", them="Them", targets=(), offline=False):
    """Transcribes growing WAV files (a recording in progress, written by the app) while they grow: every closed stretch
    of speech is decoded and written to the live checkpoint in d, which `call`/`transcribe --into d` continue from.
    With targets every line is translated as well, live (LiveWorker). Ends on SIGTERM/SIGINT once the current chunk is done,
    and likewise on stop, cancel or the end of stdin with MT_CONTROL=stdin."""
    import signal
    import numpy as np

    allowed = parse_languages(language)
    names = [me, them]
    tracks, handles, done = [], [], []
    for k, path in enumerate(paths, 1):
        tag = f".{k}" if len(paths) > 1 else ""
        speaker = speaker_name(names[k - 1], ("You", "Them")[k - 1]) if len(paths) > 1 else None
        tracks.append(LiveTrack(str(path), d, tag, path, allowed, granularity, speaker, engine_name()))
        handles.append(None)
    worker = LiveWorker(tracks, targets=targets, offline=offline)
    worker.start()
    stop = []

    def on_stop(*_):
        stop.append(True)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_stop)
    # The control channel (MT_CONTROL=stdin): stop, cancel and the end of stdin end it as those signals do. Kept to the
    # end, as the signal handlers are: a front end closing stdin while the tail is decoded changes nothing.
    CONTROL.listen({"stop": on_stop, "cancel": on_stop, "eof": on_stop})
    offsets = [44] * len(paths)
    while not stop:
        for k, path in enumerate(paths):
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > offsets[k]:
                with open(path, "rb") as f:
                    f.seek(offsets[k])
                    data = f.read((size - offsets[k]) // 2 * 2)
                offsets[k] += len(data)
                tracks[k].feed(np.frombuffer(data, dtype="<i2"))
        time.sleep(1.0)
    for k, path in enumerate(paths):                   # whatever came after the last read
        try:
            size = os.path.getsize(path)
            if size > offsets[k]:
                with open(path, "rb") as f:
                    f.seek(offsets[k])
                    tracks[k].feed(np.frombuffer(f.read((size - offsets[k]) // 2 * 2), dtype="<i2"))
        except OSError:
            pass
    worker.stop()
    if EVENTS:
        ev("DONE", 0)


def _plain(text):
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _overlaps(a, b, window=2.0):
    """Whether two lines were spoken at the same time: their intervals overlap (a line without an end is given 2 s)."""
    a0, a1 = a["t"], a.get("end", a["t"] + window)
    b0, b1 = b["t"], b.get("end", b["t"] + window)
    return min(a1, b1) - max(a0, b0) > 0 or abs(a0 - b0) <= window


def trim_echo(line, others, min_chars=12):
    """A microphone line heard the speakers: the words it shares with a line of the other side spoken at the same time
    are the echo. Returns the line without them, or None when nothing of its own (at least min_chars) is left."""
    words = line["text"].split()
    plain = [_plain(w) for w in words]
    keep = [True] * len(words)
    for other in others:
        if not _overlaps(line, other):
            continue
        theirs = [p for p in (_plain(w) for w in other["text"].split()) if p]
        if theirs and _plain(line["text"]) == _plain(other["text"]):
            return None                                 # the whole line is the other side
        matcher = SequenceMatcher(None, plain, theirs, autojunk=False)
        for block in matcher.get_matching_blocks():
            span = plain[block.a:block.a + block.size]
            if block.size >= 2 or len("".join(span)) >= min_chars:
                for k in range(block.a, block.a + block.size):
                    keep[k] = False
    if all(keep):
        return line
    rest = " ".join(w for w, k in zip(words, keep) if k)
    if len(_plain(rest)) < min_chars:
        return None
    return dict(line, text=rest)


def is_echo(line, others, window=2.0, ratio=0.7, min_chars=12):
    """True when the microphone line repeats a line of the other side that starts within `window` seconds of it: the
    speakers were heard again by the microphone, at the same moment. A real answer that repeats what was just said comes
    after the other side has finished, so it starts later; and short texts are never called echo (a real "yes" stays)."""
    mine = _plain(line["text"])
    if len(mine) < min_chars:
        return False
    for other in others:
        if abs(other["t"] - line["t"]) > window:
            continue
        theirs = _plain(other["text"])
        if len(theirs) < min_chars:
            continue
        # Only text the other side also said counts: a microphone line that continues with words of its own is kept.
        if mine in theirs or SequenceMatcher(None, mine, theirs).ratio() >= ratio:
            return True
    return False


def speaker_name(name, default):
    """A speaker name for the transcript: no angle brackets (they mark the name) and no line breaks."""
    clean = re.sub(r"\s+", " ", re.sub(r"[<>]", " ", name or "")).strip()
    return clean or default


def merge_call(mine, theirs, me="You", them="Them", echo=True, keep_labels=False):
    """The two tracks' lines in time order, each with its speaker. With echo, microphone lines that only repeat the other
    side (heard from the speakers) are dropped: use headphones to avoid it altogether. keep_labels: the other side's lines
    already say who spoke (<Them 1>, <Them 2>, and <You 1>, <You 2> for several people at the microphone: speaker separation
    on each track)."""
    theirs = [dict(l, speaker=(l.get("speaker") if keep_labels and l.get("speaker") else speaker_name(them, "Them"))) for l in theirs]
    mine = [dict(l, speaker=(l.get("speaker") if keep_labels and l.get("speaker") else speaker_name(me, "You"))) for l in mine]
    if echo:
        spoken = [l for l in theirs if "text" in l]
        trimmed = []
        for l in mine:
            kept = l if "text" not in l else trim_echo(l, spoken)
            if kept is not None:
                trimmed.append(kept)
        mine = trimmed
    return sorted(mine + theirs, key=lambda l: l["t"])      # a stable sort: at the same second the microphone comes first


def one_voice_plain(lines):
    """The lines of a microphone separated into speakers: when one voice (or none) was found there, its lines lose the
    number, so the person alone at the microphone stays <You>, as without separation."""
    voices = {l["speaker"] for l in lines if "text" in l and l.get("speaker")}
    if len(voices) > 1:
        return lines
    return [{k: v for k, v in l.items() if k != "speaker"} for l in lines]


def transcribe_call(mic, system, d, me="You", them="Them", language=None, granularity="auto", echo=True, notes=None,
                    speakers=None, diarizer="sherpa", problems=None):
    """Transcribes the microphone track and the system-audio track, labels who spoke and merges them by time. With
    speakers, the other side is separated into <Them 1>, <Them 2> ... and the microphone too, for several people at it
    (a room on your side of the call): <You 1>, <You 2> ...; one voice found there stays <You>. A number of speakers counts
    the other side; your side's number is always found by itself. A second
    track that cannot be transcribed (an empty or damaged file) does not cost the first: the transcript holds your side
    and a note, the problem is reported (and appended to `problems`, when given) so the command ends with exit 1."""
    if EVENTS:
        ev("FOLDER", d)
        ev("STAGE", "Transcribing your microphone (track 1 of 2)")
    mine = transcribe_lines(mic, language, None, granularity, position=(1, 2), checkpoint=d, context="recording",
                            **({} if speakers is None else dict(speakers="auto", diarizer=diarizer, label=speaker_name(me, "You"))))
    if speakers is not None:
        mine = one_voice_plain(mine)
    if EVENTS:
        ev("STAGE", "Transcribing the other side (track 2 of 2)")
    extra = []
    try:
        theirs = transcribe_lines(system, language, None, granularity, position=(2, 2), checkpoint=d, context="recording",
                                  speakers=speakers, diarizer=diarizer, label=speaker_name(them, "Them"))
    except Exception as e:                                    # the other side is lost, your side is not
        theirs = []
        text = f"The computer's sound could not be transcribed ({type(e).__name__}: {e}); only your side is in the transcript"
        extra = [{"t": 0, "note": text}]
        note(text)
        ev("ERR", system, text) if EVENTS else print(f"error: {system}: {text}", file=sys.stderr)
        if problems is not None:
            problems.append(text)
    lines = in_time_order(header_notes(notes) + extra + merge_call(mine, theirs, me, them, echo, keep_labels=speakers is not None))
    if EVENTS:                      # the window saw each track as it was decoded; now the merged, echo-trimmed conversation
        ev("CLEAR")
        for line in lines:
            ev("LINE", json.dumps(line, ensure_ascii=False))
    doc = {"source": f"{mic} + {system}", "whisper": WHISPER, "lines": lines, "translations": {}}
    _keep_outputs(d, doc)
    _save(d, doc)
    for tag in (".1", ".2"):
        Checkpoint(d, tag).finish()
    return doc


# ---------- translate ----------
# NLLB language code -> names and codes it is known by (the codes are the ones Whisper puts on a line).
NLLB_LANGS = {
    "eng_Latn": "en|english", "rus_Cyrl": "ru|russian", "spa_Latn": "es|spanish", "deu_Latn": "de|german",
    "fra_Latn": "fr|french", "ita_Latn": "it|italian",
    "por_Latn": "pt|portuguese|brazilian portuguese|european portuguese", "ukr_Cyrl": "uk|ukrainian",
    "pol_Latn": "pl|polish", "tur_Latn": "tr|turkish", "arb_Arab": "ar|arabic", "hin_Deva": "hi|hindi",
    "zho_Hans": "zh|chinese|mandarin|simplified chinese", "zho_Hant": "traditional chinese",
    "yue_Hant": "yue|cantonese", "jpn_Jpan": "ja|japanese", "kor_Hang": "ko|korean", "nld_Latn": "nl|dutch",
    "ces_Latn": "cs|czech", "swe_Latn": "sv|swedish", "dan_Latn": "da|danish",
    "nob_Latn": "no|nb|norwegian|norwegian bokmal", "nno_Latn": "nn|norwegian nynorsk", "fin_Latn": "fi|finnish",
    "ell_Grek": "el|greek", "heb_Hebr": "he|hebrew", "hun_Latn": "hu|hungarian", "ron_Latn": "ro|romanian",
    "bul_Cyrl": "bg|bulgarian", "srp_Cyrl": "sr|serbian", "hrv_Latn": "hr|croatian", "slk_Latn": "sk|slovak",
    "slv_Latn": "sl|slovenian", "lit_Latn": "lt|lithuanian", "lvs_Latn": "lv|latvian", "est_Latn": "et|estonian",
    "bel_Cyrl": "be|belarusian", "kat_Geor": "ka|georgian", "hye_Armn": "hy|armenian", "azj_Latn": "az|azerbaijani",
    "kaz_Cyrl": "kk|kazakh", "uzn_Latn": "uz|uzbek", "pes_Arab": "fa|persian|farsi", "urd_Arab": "ur|urdu",
    "ben_Beng": "bn|bengali", "tam_Taml": "ta|tamil", "tel_Telu": "te|telugu", "tha_Thai": "th|thai",
    "vie_Latn": "vi|vietnamese", "ind_Latn": "id|indonesian", "zsm_Latn": "ms|malay",
    "tgl_Latn": "tl|tagalog|filipino", "swh_Latn": "sw|swahili", "afr_Latn": "af|afrikaans", "cat_Latn": "ca|catalan",
    "glg_Latn": "gl|galician", "eus_Latn": "eu|basque", "isl_Latn": "is|icelandic", "cym_Latn": "cy|welsh",
    "khk_Cyrl": "mn|mongolian", "mkd_Cyrl": "mk|macedonian", "bos_Latn": "bs|bosnian", "als_Latn": "sq|albanian",
    "tgk_Cyrl": "tg|tajik", "mar_Deva": "mr|marathi", "guj_Gujr": "gu|gujarati", "pan_Guru": "pa|punjabi",
    "kan_Knda": "kn|kannada", "mal_Mlym": "ml|malayalam", "npi_Deva": "ne|nepali", "sin_Sinh": "si|sinhala",
    "khm_Khmr": "km|khmer", "lao_Laoo": "lo|lao", "mya_Mymr": "my|burmese", "amh_Ethi": "am|amharic",
    "som_Latn": "so|somali", "yor_Latn": "yo|yoruba", "hau_Latn": "ha|hausa", "zul_Latn": "zu|zulu",
    "mlt_Latn": "mt|maltese", "ltz_Latn": "lb|luxembourgish", "jav_Latn": "jv|jw|javanese",
    # The other Whisper languages the model knows (Breton, Hawaiian and Latin are not in NLLB-200).
    "asm_Beng": "as|assamese", "bak_Cyrl": "ba|bashkir", "bod_Tibt": "bo|tibetan", "fao_Latn": "fo|faroese",
    "hat_Latn": "ht|haitian creole|haitian", "lin_Latn": "ln|lingala", "plt_Latn": "mg|malagasy", "mri_Latn": "mi|maori",
    "oci_Latn": "oc|occitan", "pbt_Arab": "ps|pashto", "san_Deva": "sa|sanskrit", "snd_Arab": "sd|sindhi",
    "sna_Latn": "sn|shona", "sun_Latn": "su|sundanese", "tuk_Latn": "tk|turkmen", "tat_Cyrl": "tt|tatar",
    "ydd_Hebr": "yi|yiddish",
}
_BY_NAME = {re.sub("[^a-z]", "", alias): code for code, names in NLLB_LANGS.items() for alias in names.split("|")}


def nllb_code(name):
    """'Russian', 'ru', 'Brazilian Portuguese' or a raw NLLB code ('ces_Latn') -> NLLB code; None if unknown."""
    name = (name or "").strip()
    if re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", name):
        return name
    key = re.sub("[^a-z]", "", name.lower())
    return _BY_NAME.get(key) or _NLLB_VARIANTS.get(key)


# The regional codes TranslateGemma's template uses as targets (pt-BR, zh-TW, fil), for NLLB, which takes the lines
# TranslateGemma leaves: a target given so must be one both models know.
_NLLB_VARIANTS = {"ptbr": "por_Latn", "ptpt": "por_Latn", "zhtw": "zho_Hant", "zhhant": "zho_Hant", "zhcn": "zho_Hans",
                  "zhhans": "zho_Hans", "fil": "tgl_Latn"}


def model_present(repo):
    """Whether a Hugging Face model is in the local cache, or `repo` is a local folder (MT_HY_MODEL, MT_GEMMA_MODEL), which
    is used as it is (nothing is downloaded)."""
    if repo == HY_DEFAULT and HY == HY_DEFAULT:
        return hy_present()                          # pinned files from two repositories, each looked up on its own
    if Path(repo).expanduser().is_dir():
        return True                                  # huggingface_hub takes no path as a repo id (HFValidationError)
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, local_files_only=True)
        return repo != NLLB or _nllb_weights_here(path)
    except Exception:
        return False


def _nllb_weights_here(path):
    """Whether NLLB's snapshot folder holds its weights: a download cut short (no network, HTTP 429) leaves the folder with
    the small files only, which snapshot_download(local_files_only=True) still returns."""
    return os.path.isfile(os.path.join(path, "model.bin"))


def fetch_model(repo, size, stage=True):
    """The model in the local cache, downloaded once when it is not; raises when it cannot be fetched. `stage` says so as
    a stage of the job; during a recording it is a plain line (a stage would end the window's Recording panel)."""
    if repo == HY_DEFAULT and HY == HY_DEFAULT:
        fetch_hy(stage)                              # also when present: its files are checked before a long job
        return
    if model_present(repo):
        return
    from huggingface_hub import snapshot_download

    text = f"Fetching the translation model {repo} first (once, {size}), so that it is here when it is needed"
    ev("STAGE", text) if EVENTS and stage else print(text, file=sys.stderr)
    snapshot_download(repo)


def translation_models(targets, offline):
    """(repo, size) of the local models a translation into `targets` needs: TranslateGemma or Hy-MT3 when it is chosen
    and a target is one of its languages; NLLB in every case (beside either, for the lines in a language it lacks)."""
    if not offline or not targets:
        return []
    models = [(llm_repo(), llm_size())] if llm_on() and any(llm_code(t) for t in targets) else []
    return models + [(NLLB, "about 1.4 GB")]


def get_nllb(keep_engine=False):
    """The local translation model (downloaded once, about 1.4 GB, then used without internet) and its tokenizer. The
    speech model is let go first to make room, unless keep_engine: a live translation runs beside the recording's
    transcription and needs both (about 3 GB together)."""
    if not keep_engine:
        release_engine()                               # the speech model is not needed any more: make room
    return _load_nllb()


@functools.lru_cache(maxsize=1)
def _load_nllb():
    ctranslate2 = import_ctranslate2()
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    try:
        path = snapshot_download(NLLB, local_files_only=True)
        if not _nllb_weights_here(path):
            raise FileNotFoundError(os.path.join(path, "model.bin"))      # a download cut short: the rest is fetched
    except Exception:
        text = f"Downloading the translation model {NLLB} (first time only, about 1.4 GB)"
        ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
        path = snapshot_download(NLLB)
    if not cuda_wanted(ctranslate2):
        translator = ctranslate2.Translator(path, device="cpu", intra_threads=cpu_threads())
    elif not gpu_checked():
        translator = ctranslate2.Translator(path, device="cuda", intra_threads=cpu_threads())    # as before 1.11.0
    else:
        # CUDA with the model's own compute type, as before; a load or a first translation that fails (no cuBLAS ...)
        # leaves the work to the CPU, int8, with a note (gpu_checked).
        try:
            translator = ctranslate2.Translator(path, device="cuda", intra_threads=cpu_threads())
            translator.translate_batch([["eng_Latn", "▁a", "</s>"]], target_prefix=[["fra_Latn"]], beam_size=1,
                                       max_decoding_length=2)
        except Exception as e:
            gpu_failed("the translation model", e)
            translator = ctranslate2.Translator(path, device="cpu", compute_type="int8", intra_threads=cpu_threads())
    return translator, Tokenizer.from_file(os.path.join(path, "tokenizer.json"))


# ---------- TranslateGemma: the second offline translator (Apple silicon, through mlx-lm) ----------
# Google's TranslateGemma 12B (January 2026, Gemma 3 based), 4-bit for MLX: 6.8 GB at peak and 17.6-19.1 tokens/s on
# an Apple M5 with 24 GB, measured with five meeting sentences (ru/de/en, 1.5-2.4 s each, all five right; NLLB got four).
# Its licence (Gemma Terms of Use) allows commercial use. MT_GEMMA_MODEL names another MLX copy (the 4B or the 27B).
# The languages it was evaluated on: the 55 of WMT24++ (google/wmt24pp, English to each), 51 distinct languages with
# English. Its template knows 541 codes, but a line goes to it only when both its language and the target are among
# these 51; any other line goes to NLLB. Codes as Whisper writes them (Filipino is Whisper's "tl").
GEMMA_LANGS = frozenset(
    "en ar bg bn ca cs da de el es et fa fi tl fr gu he hi hr hu id is it ja kn ko lt lv ml mr nl no pa pl pt ro ru "
    "sk sl sr sv sw ta te th tr uk ur vi zh zu".split())
GEMMA_VARIANTS = {"brazilianportuguese": "pt-BR", "europeanportuguese": "pt-PT", "traditionalchinese": "zh-TW",
                  "zhhant": "zh-TW", "zhtw": "zh-TW", "zhohant": "zh-TW", "ptbr": "pt-BR", "ptpt": "pt-PT",
                  "zhhans": "zh", "zhcn": "zh", "zhohans": "zh", "filipino": "tl", "fil": "tl"}


def gemma_code(name):
    """'Russian', 'ru', 'Brazilian Portuguese', 'Traditional Chinese', 'zul_Latn' -> the code TranslateGemma's template
    takes ('ru', 'pt-BR', 'zh-TW', 'zu'); None for a language it was not evaluated on."""
    key = re.sub("[^a-z]", "", (name or "").strip().lower())
    if key in GEMMA_VARIANTS:
        return GEMMA_VARIANTS[key]
    code = whisper_code(name)
    if not code:                                     # a language Whisper does not have (Zulu): through the NLLB table
        aliases = (NLLB_LANGS.get(nllb_code(name)) or "").split("|")
        code = next((a for a in aliases if re.fullmatch(r"[a-z]{2,3}", a)), None)
    return code if code in GEMMA_LANGS else None


def mlx_lm_available():
    """Apple silicon with the mlx-lm package installed: TranslateGemma and Hy-MT3 need it (the app's Apple-GPU setup step
    installs it). Looked up without importing it: mlx-lm makes its generation stream when it is imported (see
    _thread_stream)."""
    import importlib.util
    import platform

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        return importlib.util.find_spec("mlx_lm") is not None
    except Exception:
        return False


_STREAM_THREAD = []          # the thread mlx-lm's generation stream belongs to (see _thread_stream)


def _thread_stream():
    """mlx-lm 0.29.1 makes its generation stream when it is imported (mlx_lm/generate.py:217) and uses it for every
    generation; MLX 0.32 and later let a stream be used only on the thread that made it ("There is no Stream(gpu, 0) in
    current thread"), and the live translation runs on another thread than the final pass. Before a generation on a
    thread other than the stream's, a stream of that thread is put in its place (mlx-lm 0.31.3 made the same change
    upstream with a thread-local stream). Measured with MLX 0.32.4 on a CPU; with older MLX it changes nothing."""
    import threading

    me = threading.get_ident()
    if _STREAM_THREAD and _STREAM_THREAD[0] == me:
        return
    import importlib

    import mlx.core as mx

    # The module itself: `import mlx_lm.generate` would give the function of that name, which mlx_lm/__init__.py
    # imports over it, and a stream set there is read by nothing.
    generation = importlib.import_module("mlx_lm.generate")
    if hasattr(generation, "generation_stream"):
        generation.generation_stream = mx.new_stream(mx.default_device())
    _STREAM_THREAD[:] = [me]


def get_gemma(keep_engine=False):
    """TranslateGemma and its tokenizer (downloaded once, about 7 GB). The speech model is let go first, unless
    keep_engine (a live translation: both stay loaded, about 8.5 GB together)."""
    if not keep_engine:
        release_engine()
    return _load_gemma()


@functools.lru_cache(maxsize=1)
def _load_gemma():
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    try:
        snapshot_download(GEMMA, local_files_only=True)
    except Exception:
        text = f"Downloading the translation model {GEMMA} (first time only, about 7 GB)"
        ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
        snapshot_download(GEMMA)
    model, tok = load(GEMMA)
    # The model ends its answer with <end_of_turn>; the MLX copy's config.json names no stop token, so mlx-lm would stop
    # only at <eos> and run on to the token limit (measured: with this added, every test line ended with "stop").
    tok.add_eos_token("<end_of_turn>")
    return model, tok


def release_gemma():
    import gc
    clear = getattr(_load_gemma, "cache_clear", None)
    if clear:
        clear()
    gc.collect()


# ---------- Hy-MT3: the third offline translator (Apple silicon, through mlx-lm) ----------
# Tencent's Hy-MT3-7B (24 Nov 2026, https://huggingface.co/tencent/Hy-MT3-7B), 102 languages, Apache-2.0 ("does not impose
# any additional restrictions beyond those specified in the license", its LICENSE.txt). The weights are NetProjects' 4-bit
# MLX copy (affine, group size 64, converted from Tencent's revision f83bea8; weights and tokenizer unchanged up to 38d6c3c):
# its tensors match what mlx-lm 0.29.1 builds for hunyuan_v1_dense, name for name and shape for shape. Its own
# tokenizer_config.json names TokenizersBackend, a class of transformers 5 (the app pins 4.57.6), so the tokenizer files
# are Tencent's own, which load and give the same ids. The quality figures so far are Tencent's (no independent evaluation
# yet); its memory is not measured here (about 4.5 GB is an estimate from the same-size Hy-MT2-7B, 4.53 GB at peak).
HY_FILES = (   # (repo, commit, file, bytes, sha256): everything the default needs, each fetched and checked on its own
    (HY_DEFAULT, "dca6fb32e6d61d2657ed714b572307216df2cb5c", "config.json", 1721,
     "be19177093a8e49915052d463e28d78cc14b4124731a17b3549bc608bfdeb9d0"),
    (HY_DEFAULT, "dca6fb32e6d61d2657ed714b572307216df2cb5c", "model.safetensors", 4221806239,
     "81176df138f651ced6f827956d8c767dcbf8a99399f3a2040ad3dbfb2ef7ccc1"),
    (HY_DEFAULT, "dca6fb32e6d61d2657ed714b572307216df2cb5c", "LICENSE.txt", 9673,
     "974b62424d5627567283a4808181293e3ea53894cea7e75341d8e9f5f1b874cc"),
    (HY_DEFAULT, "dca6fb32e6d61d2657ed714b572307216df2cb5c", "NOTICE", 447,
     "84f60c1684c896850a314f36f877b0e1ec306916d8c51a11c0a35f5d8fa89e38"),
    ("tencent/Hy-MT3-7B", "38d6c3c31d6b4dbdfc208e0621b822ae7bf6aeb2", "tokenizer.json", 16388221,
     "4185389a69284fdd3ebf737c6120bbeee36cbed9763151fa5bac5986a39e05d4"),
    ("tencent/Hy-MT3-7B", "38d6c3c31d6b4dbdfc208e0621b822ae7bf6aeb2", "tokenizer_config.json", 37763,
     "1fffdb809549222938f4a732645dd984c7c0b2973d592c2d01e05425274317da"),
    ("tencent/Hy-MT3-7B", "38d6c3c31d6b4dbdfc208e0621b822ae7bf6aeb2", "special_tokens_map.json", 431,
     "c6571f2ee67c36fa93d7f688ee6fd43f5b84784e895f3e6a4eb8c2657379758e"),
    ("tencent/Hy-MT3-7B", "38d6c3c31d6b4dbdfc208e0621b822ae7bf6aeb2", "chat_template.jinja", 662,
     "788ac16c5d7bfefc28655928ad524c8f378a44cb24d24fb125d6a5859b167677"),
)
HY_CHECKED = "hy-mt3-checked.json"    # in MODELS_DIR: {blob path: [size, mtime_ns, sha256]} of the big file once checked
# The languages of the card's tables that Whisper tags lines with (89 of its 100) and Traditional Chinese, by the English
# name the prompt takes, spelled as on the card. Whisper's Kazakh and Mongolian are written in Cyrillic, the card's plain
# kk and mn are the Arabic and the traditional scripts; Whisper's no and jw are the card's nb and jv. Not on the card, so
# NLLB's: Breton, Bosnian, Faroese, Hausa, Hawaiian, Latin, Lingala, Maori, Occitan, Shona, Yoruba.
HY_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fi": "Finnish", "fr": "French", "gl": "Galician", "gu": "Gujarati", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian", "kk": "Cyrillic Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "lb": "Luxembourgish", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mk": "Macedonian", "ml": "Malayalam", "mn": "Cyrillic Mongolian", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik",
    "th": "Thai", "tk": "Turkmen", "tl": "Filipino", "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yue": "Cantonese", "zh": "Chinese", "zh-TW": "Traditional Chinese",
}
# Hy-MT3 has one Portuguese (as NLLB has); whisper_code("zho_Hant") is zh (Whisper has one Chinese), hence zhohant here.
HY_VARIANTS = {"traditionalchinese": "zh-TW", "zhhant": "zh-TW", "zhtw": "zh-TW", "zhohant": "zh-TW", "zhhans": "zh",
               "zhcn": "zh", "zhohans": "zh", "brazilianportuguese": "pt", "europeanportuguese": "pt", "ptbr": "pt",
               "ptpt": "pt", "filipino": "tl", "fil": "tl"}
# The targets of at most 3 tokens per English token (measured on one sentence in 12 languages: German 1.6, Russian 1.8,
# Ukrainian 2.5, Arabic 2.6, Chinese 0.9, Japanese 1.8, Korean 2.1; the others of the same scripts by inference); every
# other target gets 9 (measured: Burmese 8.1, Hindi 4.5, Thai 3.5).
HY_DENSE_OK = frozenset("en de fr es it pt nl ca gl eu af lb mt cy sq ro hu cs sk sl hr sr bs pl lt lv et fi sv da no nn is "
                        "tr az uz tk id ms tl jw su vi sw so mg ht ru uk be bg mk kk mn tg tt ba el ar fa ur ps sd yi he "
                        "zh zh-TW yue ja ko".split())
HY_NO_SPACES = frozenset("zh zh-TW yue ja th lo km my".split())     # written without spaces between sentences (NO_SPACES)
HY_PIECE = 1500       # characters per request: about 350 English tokens, which with 9 times that as output fit its 8K context


def hy_code(name):
    """'Russian', 'ru', 'Kazakh', 'Brazilian Portuguese', 'Traditional Chinese', 'zho_Hant' -> the code of HY_NAMES ('ru',
    'kk', 'pt', 'zh-TW'); None for a language Hy-MT3 does not support (Yoruba, Bosnian) or one Whisper does not know."""
    key = re.sub("[^a-z]", "", (name or "").strip().lower())
    if key in HY_VARIANTS:
        return HY_VARIANTS[key]
    code = whisper_code(name)
    return code if code in HY_NAMES else None


def hy_pair(lang, target, source=None):
    """(source code, target code) when Hy-MT3 takes a line in `lang` (or `source` for an untagged one) into `target`,
    else None."""
    src, tgt = hy_code(lang or source), hy_code(target)
    return (src, tgt) if src and tgt else None


def hy_prompt(text, tgt):
    """The model card's plain "Default Translation" prompt in English, with the target's English name; it names no source
    language, and the model has no system prompt."""
    return (f"Translate the following text into {HY_NAMES[tgt]}. Note that you should only output the translated result "
            f"without any additional explanation:\n\n{text}")


def hy_budget(n_src, tgt):
    """max_tokens for a piece of n_src tokens into tgt: room for the target's script (measured: Burmese 8.1x English, Hindi
    4.5x, Thai 3.5x, Arabic 2.6x, Ukrainian 2.5x), capped at the card's 4096."""
    return min(4096, 64 + (3 if tgt in HY_DENSE_OK else 9) * n_src)


def hy_present():
    """Whether every one of HY_FILES is in the Hugging Face cache at its commit (nothing is downloaded). Looked up file by
    file: snapshot_download(local_files_only=True) gives the snapshot folder without looking at the files in it."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo, name, revision=commit), str) for repo, commit, name, _, _ in HY_FILES)
    except Exception:
        return False


def fetch_hy(stage=True, refetch=True):
    """{file: path} of HY_FILES in the Hugging Face cache, each one downloaded when it is missing and checked against its
    size and SHA-256 (huggingface_hub checks neither). `stage` as in fetch_model; `refetch` as in _check_hy_file."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = {name: try_to_load_from_cache(repo, name, revision=commit) for repo, commit, name, _, _ in HY_FILES}
    if not all(isinstance(path, str) for path in cached.values()):
        text = "Downloading the translation model Hy-MT3 7B (first time only, about 4.2 GB)"
        ev("STAGE", text) if EVENTS and stage else print(text, file=sys.stderr)
    paths = {}
    for repo, commit, name, size, sha256 in HY_FILES:
        path = cached[name] if isinstance(cached[name], str) else hf_hub_download(repo, name, revision=commit)
        paths[name] = _check_hy_file(repo, commit, name, size, sha256, path, stage, refetch)
    return paths


_HY_REFETCHED = set()     # (repo, name) of the files downloaded again in this run: a second mismatch is not fetched a third time


def _check_hy_file(repo, commit, name, size, sha256, path, stage=True, refetch=True):
    """`path` when its size and SHA-256 are the pinned ones; otherwise it is downloaded once more (once per run) and
    checked again, and an error tells what to delete when it still does not match. Without `refetch` (the live worker's
    thread, where a 4.2 GB download would hold up the live transcript) a mismatch is an error at once: the live
    translation is off and the final pass, off that thread, downloads the file again."""
    from huggingface_hub import hf_hub_download

    got = _hy_digest(path, size, sha256, stage)
    if got != sha256 and not refetch:
        raise RuntimeError(f"The Hy-MT3 file {name} from {repo} does not match its checksum; it is downloaded again when "
                           "the recording is translated at the end")
    if got != sha256 and (repo, name) not in _HY_REFETCHED:
        _HY_REFETCHED.add((repo, name))
        print(f"The Hy-MT3 file {name} from {repo} does not match its checksum; downloading it again", file=sys.stderr)
        path = hf_hub_download(repo, name, revision=commit, force_download=True)
        got = _hy_digest(path, size, sha256, stage)
    if got != sha256:
        raise RuntimeError(f"The Hy-MT3 file {name} from {repo} does not match its checksum (expected {sha256}, got {got}). "
                           f"Delete the folder ~/.cache/huggingface/hub/models--{repo.replace('/', '--')} and translate "
                           "again, or choose another offline model.")
    return path


def _hy_digest(path, size, sha256, stage=True):
    """The SHA-256 of a file (or the size it has, when that is not `size`). The 4.2 GB file is hashed once: after it
    matched, its path, size and modification time are recorded in MODELS_DIR, and a file that still has them is not read
    again; the small files (17 MB) are hashed at every load."""
    real = os.path.realpath(path)
    st = os.stat(real)
    if st.st_size != size:
        return f"a file of {st.st_size} bytes instead of {size}"
    big = size > 2**26
    record = MODELS_DIR / HY_CHECKED
    checked = {}
    if big:
        try:
            checked = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            checked = {}
        if checked.get(real) == [st.st_size, st.st_mtime_ns, sha256]:
            return sha256
        text = "Checking the downloaded translation model (once)"
        ev("STAGE", text) if EVENTS and stage else print(text, file=sys.stderr)
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            h.update(block)
    got = h.hexdigest()
    if big and got == sha256:
        try:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            checked[real] = [st.st_size, st.st_mtime_ns, sha256]
            record.write_text(json.dumps(checked), encoding="utf-8")
        except OSError:
            pass                                     # not recorded: it is hashed again next time, nothing worse
    return got


def get_hy(keep_engine=False):
    """Hy-MT3 and its tokenizer (downloaded once, about 4.2 GB). The speech model is let go first, unless keep_engine (a
    live translation: both stay loaded, about 6 GB together, an estimate)."""
    if not keep_engine:
        release_engine()
    return _load_hy()


@functools.lru_cache(maxsize=1)
def _load_hy():
    if HY == HY_DEFAULT:
        from mlx_lm.utils import load_model, load_tokenizer

        # Downloaded (a stage of the job) only when a file is missing; a recording's live worker has fetched them already,
        # so there it is only the check, which says nothing unless the big file was never checked.
        files = fetch_hy(stage=not hy_present())
        model, config = load_model(Path(files["model.safetensors"]).parent)
        tok = load_tokenizer(Path(files["tokenizer.json"]).parent, eos_token_ids=config.get("eos_token_id"))
    else:
        from mlx_lm import load

        folder = Path(HY).expanduser()
        if not folder.is_dir() and not model_present(HY):
            text = f"Downloading the translation model {HY} (first time only)"
            ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
        try:
            model, tok = load(str(folder) if folder.is_dir() else HY)
        except ValueError as e:
            if "TokenizersBackend" not in str(e):
                raise
            raise RuntimeError(f"{HY}: its tokenizer_config.json needs transformers 5 ('TokenizersBackend'); put Tencent's "
                               "tokenizer_config.json (https://huggingface.co/tencent/Hy-MT3-7B/blob/main/tokenizer_config.json) "
                               "in its place, or unset MT_HY_MODEL") from e
    # The answer ends with <|eos|>. Tencent's own config.json names only that one (a copy converted from it too), the
    # generation config also <|extra_5|>: both stop the generation, the second when the vocabulary has it.
    tok.add_eos_token("<|eos|>")
    extra = tok.convert_tokens_to_ids("<|extra_5|>")
    if isinstance(extra, int) and extra != getattr(tok, "unk_token_id", None):
        tok.add_eos_token("<|extra_5|>")
    return model, tok


def release_hy():
    import gc
    clear = getattr(_load_hy, "cache_clear", None)
    if clear:
        clear()
    gc.collect()


def hy_translate(items, tgt, keep_engine=False):
    """(texts, cut): the texts of `items` ((source code, text) pairs, codes from hy_code) in `tgt`, one request per text
    (a long text in pieces of whole sentences), and how many of them had a piece that ran to its token limit (kept: it is
    the best there is). Greedy, with the card's repetition penalty (1.05 over the last 20 tokens) and not its sampling: the
    same line always gives the same text, live, at the end and in a rerun (in a 10-line test meant to provoke loops no
    setting looped, and sampling changed half the lines from run to run). mlx-lm's penalty window holds the last prompt
    token and the reply, not the source text, so a name or a number copied from it is not penalised."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors

    model, tok = get_hy(keep_engine)
    _thread_stream()
    decoding = dict(logits_processors=make_logits_processors(repetition_penalty=1.05, repetition_context_size=20))
    out, cut = [], 0
    for src, text in items:
        done, short = [], False
        for p in _pieces(text, HY_PIECE):
            ids = tok.apply_chat_template([{"role": "user", "content": hy_prompt(p, tgt)}], add_generation_prompt=True)
            budget = hy_budget(len(tok.encode(p, add_special_tokens=False)), tgt)
            reply, finish = "", None
            for r in stream_generate(model, tok, prompt=ids, max_tokens=budget, **decoding):
                reply += r.text
                finish = r.finish_reason
            short = short or finish == "length"
            done.append(reply.split("<|eos|>")[0].strip() or p)
        cut += short
        out.append(("" if tgt in HY_NO_SPACES else " ").join(done))
    return out, cut


def _pieces(text, limit):
    """A text in pieces of at most `limit` characters, of whole sentences where it can (a longer sentence at words): one
    request each to TranslateGemma or Hy-MT3."""
    if len(text) <= limit:
        return [text]
    units = []
    for sentence in sentences(text):
        sentence = sentence.strip()
        while len(sentence) > limit:                 # one sentence too long for a request: cut at the last space before the limit
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            units.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            units.append(sentence)
    pieces, piece = [], ""
    for unit in units:
        if piece and len(piece) + 1 + len(unit) > limit:
            pieces.append(piece)
            piece = ""
        piece = f"{piece} {unit}" if piece else unit
    if piece:
        pieces.append(piece)
    return pieces


GEMMA_PIECE = 1500    # characters per request: the model takes 2K tokens of input; a longer text goes sentence by sentence


def gemma_translate(items, tgt, keep_engine=False):
    """The texts of `items` ((source code, text) pairs, codes from gemma_code) in `tgt`, one request per text (a long
    text in pieces of whole sentences)."""
    from mlx_lm import generate

    model, tok = get_gemma(keep_engine)
    _thread_stream()
    out = []
    for src, text in items:
        done = []
        for p in _pieces(text, GEMMA_PIECE):
            msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": src, "target_lang_code": tgt, "text": p}]}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
            reply = generate(model, tok, prompt=ids, max_tokens=min(2048, 2 * len(ids) + 64))
            done.append(reply.split("<end_of_turn>")[0].strip() or p)
        out.append(("" if tgt.split("-")[0] in ("zh", "ja", "th") else " ").join(done))
    return out


def gemma_pair(lang, target, source=None):
    """(source code, target code) when TranslateGemma takes a line in `lang` (or `source` for an untagged one) into
    `target`, else None."""
    src, tgt = gemma_code(lang or source), gemma_code(target)
    return (src, tgt) if src and tgt else None


def same_language(src, tgt):
    """A line already in the target: pt and pt-BR count as the same (as NLLB's codes do); zh and zh-TW do not (Whisper's
    one Chinese is turned into Traditional Chinese, as NLLB does)."""
    if src == tgt:
        return True
    return src.split("-")[0] == tgt.split("-")[0] and src.split("-")[0] != "zh"


# TranslateGemma and Hy-MT3 go the same ways (a line each, the rest to NLLB, a log, the live pass); these say which of the
# two the run has chosen. They read TRANSLATOR when called (--translator sets it in main(), the tests with monkeypatch).
LLMS = ("gemma", "hy")       # the offline translators that run on the Apple GPU through mlx-lm


def llm_on():
    return TRANSLATOR in LLMS


def llm_repo():
    return HY if TRANSLATOR == "hy" else GEMMA


def llm_label():
    return "Hy-MT3" if TRANSLATOR == "hy" else "TranslateGemma"


def llm_outside():
    """The words for a line the model does not take ("1 in a language … go to NLLB")."""
    return "Hy-MT3 does not support" if TRANSLATOR == "hy" else "TranslateGemma was not evaluated on"


def llm_size():
    return "about 4.2 GB" if TRANSLATOR == "hy" else "about 7 GB"


def llm_code(name):
    return hy_code(name) if TRANSLATOR == "hy" else gemma_code(name)


def llm_pair(lang, target, source=None):
    return hy_pair(lang, target, source) if TRANSLATOR == "hy" else gemma_pair(lang, target, source)


def llm_model_name():
    """The model's name in a log's file name: a local folder (MT_HY_MODEL) by its own name and 8 characters of its full
    path's hash, as the full path could make the file name longer than a file system allows, and two folders of the same
    name (a 4-bit and an 8-bit copy) must not reuse each other's lines."""
    folder = Path(HY).expanduser()
    if TRANSLATOR == "hy" and folder.is_dir():
        return f"{folder.name}-{hashlib.sha1(str(folder.resolve()).encode('utf-8')).hexdigest()[:8]}"
    return llm_repo()


def get_llm(keep_engine=False):
    return get_hy(keep_engine) if TRANSLATOR == "hy" else get_gemma(keep_engine)


def release_llm():
    release_hy() if TRANSLATOR == "hy" else release_gemma()


def llm_translate(items, tgt, keep_engine=False):
    """(texts, cut) as hy_translate, with the chosen model (TranslateGemma's replies are never counted as cut)."""
    if TRANSLATOR == "hy":
        return hy_translate(items, tgt, keep_engine)
    return gemma_translate(items, tgt, keep_engine), 0


def line_translator(lang, target, offline, source=None):
    """Which model translates a line into `target`: "claude", NLLB, TranslateGemma or Hy-MT3 (which give a line outside
    their languages to NLLB). A live translation is reused at the end only when the same model made it."""
    if not offline:
        return "claude"
    return llm_repo() if llm_on() and llm_pair(lang, target, source) else NLLB


SENTENCE_END =re.compile(r"(?<=[.!?\u2026])\s+|(?<=[\u3002\uff01\uff1f])")      # . ! ? ... and CJK 。！？
NO_SPACES = {"zho_Hans", "zho_Hant", "yue_Hant", "jpn_Jpan", "tha_Thai", "khm_Khmr", "lao_Laoo", "mya_Mymr"}


# Words after which a full stop does not end a sentence (en, de, ru); a single letter (an initial) and a number also do not.
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "jr", "sr", "inc", "ltd", "co", "fig", "approx", "dept",
                 "e.g", "i.e", "a.m", "p.m", "u.s", "d.c", "z.b", "bzw", "usw", "ca", "evtl", "ggf", "nr", "str", "tel", "u.a", "d.h",
                 "т.е", "т.д", "т.п", "т.к", "г", "ул", "д", "тел", "др", "см", "напр", "им", "стр"}


def sentences(text):
    """NLLB drops or cuts sentences when a line holds several, so lines are translated one sentence at a time. A full
    stop after an abbreviation, an initial or a number, or one followed by a lowercase word, does not end a sentence."""
    pieces, start = [], 0
    for m in SENTENCE_END.finditer(text):
        before = text[start:m.start()]
        after = text[m.end():]
        last = before.rstrip().split()[-1].rstrip(".!?\u2026").lower() if before.strip() else ""
        if before.rstrip().endswith(".") and (
                last in ABBREVIATIONS or len(last) == 1 or last.replace(",", "").isdigit() or (after[:1].islower())):
            continue
        if before.strip():
            pieces.append(before)
        start = m.end()
    tail = text[start:]
    if tail.strip():
        pieces.append(tail)
    return [p for p in pieces if p.strip()] or [text]


def tidy(text, tgt):
    """NLLB sometimes puts punctuation in tokens of its own ("Hello , my friends ."); close those gaps. French keeps its
    space before ? ! : ; and the languages written without spaces are left alone."""
    return text if tgt in NO_SPACES or tgt == "fra_Latn" else re.sub(r"\s+([,.;:!?\u2026])", r"\1", text)


def _translation_log(checkpoint, tgt, lines):
    """Where the pieces of a translation are kept as they are made (a rerun after a crash reuses them), or None."""
    if not checkpoint:
        return None
    # The key holds each line's language ("Gift" in English is not "Gift" in German) and its sentences as this version cuts
    # them: a log from a version that cut sentences differently is not reused (its records are numbered by sentence).
    digest = hashlib.sha1(json.dumps([[l.get("lang"), sentences(l["text"]) if "text" in l else None] for l in lines],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    folder = Path(checkpoint) / "translations"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / translation_log_name(tgt, NLLB, digest)


def translation_log_name(tgt, model, digest):
    """The file name of a translation's log in the data folder's translations\\ (NLLB's, TranslateGemma's, Hy-MT3's)."""
    return f"{tgt}-{model.replace('/', '_')}-{digest}.jsonl"


def translate_offline(doc, target, source=None, checkpoint=None, live=None, done_label="translated live"):
    """Translate every line with the local model. A line is read as the language it is tagged with (source for
    untagged lines); lines already in the target language, or of an unknown language, stay as they are. With a checkpoint
    folder every finished batch is written down, so a run cut short goes on where it stopped. `live` holds the lines
    translated already ({line index: text}, live, or by TranslateGemma or Hy-MT3: `done_label` says which in the note):
    they are taken as they are, and the model is loaded only when a line is left for it."""
    tgt = nllb_code(target)
    lines = doc["lines"]
    live = live or {}
    log = _translation_log(checkpoint, tgt, lines)
    earlier = {}
    if log and log.exists():
        for raw in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(raw)
                earlier[(rec["i"], rec["n"])] = rec["text"]
            except Exception:
                pass
    out = [l.get("text") for l in lines]         # a note line has no text and stays None
    parts = {}                                   # line index -> translated sentences
    groups, unknown, untagged = {}, 0, 0         # source language -> [(line index, sentence number, sentence)]
    for i, l in enumerate(lines):
        if "text" not in l:
            continue
        if i in live:
            out[i] = live[i]
            continue
        src = nllb_code(l.get("lang") or source)
        if not src:
            unknown += 1
            untagged += not l.get("lang")         # a line without a language tag: --from would have named it
        elif src != tgt:
            pieces = sentences(l["text"])
            parts[i] = [earlier.get((i, n)) for n in range(len(pieces))]
            groups.setdefault(src, []).extend((i, n, piece) for n, piece in enumerate(pieces) if (i, n) not in earlier)
    if earlier:
        note(f"Going on from an earlier run: {len(earlier)} translated sentences reused")
    translator, tok = get_nllb() if groups else (None, None)       # nothing left for the model: it is not loaded
    # A sentence longer than the model likes is cut at clause punctuation (then by length) and translated in pieces.
    expanded = {}
    piece_total = {}                                 # (line, sentence) -> how many pieces it was cut into
    for src, items in groups.items():
        for i, n, piece in items:
            cut = split_long(piece, tok)
            piece_total[(i, n)] = len(cut)
            for k, part in enumerate(cut):
                expanded.setdefault(src, []).append((i, n, k, part))
    groups = expanded
    pieces_out = {}                                  # (line, sentence) -> {k: text}
    short = set()                                    # lines whose translation came back much shorter than their text
    batches = sum((len(items) + 15) // 16 for items in groups.values())
    done = 0
    handle = open(log, "a", encoding="utf-8") if log else None
    for src, items in groups.items():
        for start in range(0, len(items), 16):
            batch = items[start:start + 16]
            done += 1
            if EVENTS:
                ev("PROG", done, batches)
            encoded = [tok.encode(piece, add_special_tokens=False).tokens[:400] for _, _, _, piece in batch]
            tokens = [[src] + e + ["</s>"] for e in encoded]
            longest = max(len(e) for e in encoded)
            results = translator.translate_batch(tokens, target_prefix=[[tgt]] * len(batch), beam_size=4,
                                                 max_decoding_length=min(512, 2 * longest + 16))
            for (i, n, k, piece), e, r in zip(batch, encoded, results):
                hyp = r.hypotheses[0][1:]                                  # [1:] drops the target-language token
                ids = [tok.token_to_id(t) for t in hyp]
                text = tidy(tok.decode([x for x in ids if x is not None], skip_special_tokens=True).strip(), tgt) or piece
                if len(e) >= 8 and len(hyp) < 0.4 * len(e):
                    short.add(i)
                pieces_out.setdefault((i, n), {})[k] = text
            # join the pieces of a sentence once all of them are translated (only the sentences this batch touched)
            for (i, n) in dict.fromkeys((i, n) for i, n, _, _ in batch):
                parts_k = pieces_out[(i, n)]
                if parts[i][n] is None and len(parts_k) == piece_total[(i, n)]:
                    joined = ("" if tgt in NO_SPACES else " ").join(parts_k[k] for k in sorted(parts_k))
                    parts[i][n] = joined
                    del pieces_out[(i, n)]
                    if handle:
                        handle.write(json.dumps({"i": i, "n": n, "text": joined}, ensure_ascii=False) + "\n")
            if handle:
                handle.flush()
    if handle:
        handle.close()
    for i, pieces in parts.items():
        out[i] = ("" if tgt in NO_SPACES else " ").join(p if p is not None else "" for p in pieces)
    translated = len(parts)
    same = sum(1 for i, l in enumerate(lines) if "text" in l and i not in live and nllb_code(l.get("lang") or source) == tgt)
    note(f"Translated {translated} line(s) into {target}" + (f"; {len(live)} {done_label}" if live else "")
         + (f"; {same} already in {target}" if same else "")
         + (f"; {unknown} of unknown language kept as they are" + (" (give the language with --from)" if untagged and not source else
                                                                   " (a language the offline model does not have)" if not untagged else "") if unknown else "")
         + (f"; {len(short)} came back much shorter than the original and may be incomplete" if short else ""))
    return out


def split_long(piece, tok, limit=200):
    """A sentence over `limit` tokens is cut at clause punctuation near the middle, then by length, so nothing is
    dropped by the model's input limit (it was trained on at most 512 tokens and tends to shorten long inputs)."""
    if len(tok.encode(piece, add_special_tokens=False).tokens) <= limit:
        return [piece]
    cuts = [m.end() for m in re.finditer(r"[,;:\u2014\u2013)]\s+|\s[-\u2013]\s", piece)]
    middle = len(piece) / 2
    if cuts:
        at = min(cuts, key=lambda c: abs(c - middle))
        if 0 < at < len(piece):
            return split_long(piece[:at].rstrip(), tok, limit) + split_long(piece[at:].lstrip(), tok, limit)
    words = piece.split()
    if len(words) < 2:
        return [piece]
    half = len(words) // 2
    return split_long(" ".join(words[:half]), tok, limit) + split_long(" ".join(words[half:]), tok, limit)


def claude_translate(client, target, texts):
    """One request to Claude for a list of texts: ({n: translation}, complete); a text that did not come back is missing
    from the dict (the caller keeps the original and says so)."""
    import anthropic

    body = "\n".join(f"{n}|{t}" for n, t in enumerate(texts))
    try:
        msg = client.messages.create(
            model=CLAUDE, max_tokens=16000,
            messages=[{"role": "user", "content":
                       f"Translate each numbered line into {target}. Output exactly one line per input, "
                       f"in the form N|translation, nothing else. If a line is already in {target}, "
                       f"copy it unchanged.\n\n{body}"}])
    except anthropic.APIConnectionError as e:
        # The SDK's own text is just "Connection error."; the real reason (DNS, certificate, proxy, timeout) is the cause.
        why = e.__cause__ or e
        raise RuntimeError(f"cannot reach the Claude API: {type(why).__name__}: {why}. The transcript is saved; "
                           f"check the network, VPN or proxy and run the translation again, or use --offline") from e
    got = dict(re.findall(r"^(\d+)\|(.*)$", msg.content[0].text, re.M))
    cut_off = msg.stop_reason == "max_tokens"
    if cut_off and got:                              # the reply stopped at the token limit: its last line may be cut short
        got.pop(max(got, key=int))
    return got, not cut_off and all(str(n) in got for n in range(len(texts)))


def translated_live(lines, target, offline, source=None):
    """{line index: text} of the lines a live run translated into `target` with the model the final pass would use for
    them (`tr_by`: Claude, NLLB, TranslateGemma or Hy-MT3; before 1.10.0's TranslateGemma, `tr_offline` alone) and whose
    text is still the one translated (`tr_of`): a line the final pass changed (echo trimmed, decoded again) is translated
    anew."""
    def made_by(l):
        return (l.get("tr_by") or {}).get(target) or (NLLB if l.get("tr_offline") else "claude")
    return {i: l["tr"][target] for i, l in enumerate(lines)
            if "text" in l and target in (l.get("tr") or {}) and l.get("tr_of") == l["text"]
            and made_by(l) == line_translator(l.get("lang"), target, offline, source)}


def _llm_log(checkpoint, tgt, lines):
    """Where TranslateGemma's or Hy-MT3's lines are kept as they are made (a rerun after a crash reuses them), or None."""
    if not checkpoint:
        return None
    digest = hashlib.sha1(json.dumps([[l.get("lang"), l.get("text")] for l in lines], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    folder = Path(checkpoint) / "translations"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / translation_log_name(tgt, llm_model_name(), digest)


def translate_llm_doc(doc, target, source=None, checkpoint=None, live=None):
    """Every line with TranslateGemma or Hy-MT3 (the chosen one), one line per request; a line already in the target
    stays as it is; a line of a language outside the model's (or untagged without --from) goes to NLLB, which the note
    counts. With a checkpoint folder each translated line is written down, so a run cut short goes on where it stopped.
    `live` holds the lines the model translated live ({line index: text})."""
    lines, live = doc["lines"], dict(live or {})
    tgt = llm_code(target)
    if not tgt:                                       # a target outside the model's languages: all of it is NLLB's
        note(f"{target} is not among {llm_label()}'s languages: NLLB translates it")
        return translate_offline(doc, target, source, checkpoint, live=live) if live else translate_offline(doc, target, source, checkpoint)
    log = _llm_log(checkpoint, tgt, lines) if tgt else None
    earlier = {}
    if log and log.exists():
        for raw in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(raw)
                earlier[rec["i"]] = rec["text"]
            except Exception:
                pass                                  # a record cut short by the crash
    done, todo, same, reused = dict(live), [], 0, 0
    for i, l in enumerate(lines):
        if "text" not in l or i in done:
            continue
        pair = llm_pair(l.get("lang"), target, source)
        if not pair:
            continue                                  # NLLB's, below
        if same_language(*pair):
            done[i] = l["text"]
            same += 1
        elif i in earlier:
            done[i] = earlier[i]
            reused += 1
        else:
            todo.append((i, pair[0]))
    if reused:
        note(f"Going on from an earlier run: {reused} lines translated by {llm_label()} reused")
    if todo:
        release_engine()                              # once: the speech model is not needed any more
    handle = open(log, "a", encoding="utf-8") if log and todo else None
    cut = 0                                           # lines with a piece that ran to the token limit (Hy-MT3)
    try:
        for k, (i, src) in enumerate(todo, 1):
            if EVENTS:
                ev("PROG", k, len(todo))
            texts, n = llm_translate([(src, lines[i]["text"])], tgt, keep_engine=True)
            done[i], cut = texts[0], cut + n
            if handle:
                handle.write(json.dumps({"i": i, "text": done[i]}, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        if handle:
            handle.close()
    spoken = sum(1 for l in lines if "text" in l)
    rest = spoken - len(done)
    if rest and not (todo or reused or live or same):  # not "Translated 0 line(s) … with Hy-MT3"
        note(f"None of the {rest} line(s) is in one of {llm_label()}'s languages: NLLB translates them into {target}")
    else:
        note(f"Translated {len(todo) + reused} line(s) into {target} with {llm_label()}"
             + (f"; {len(live)} translated live" if live else "") + (f"; {same} already in {target}" if same else "")
             + (f"; {rest} in a language {llm_outside()} go to NLLB" if rest else "")
             + (f"; {cut} line(s) ran to the token limit and may be cut short or repeat" if cut else ""))
    if not rest:
        return [done.get(i, l.get("text")) for i, l in enumerate(lines)]
    release_llm()                                     # room for NLLB
    try:
        return translate_offline(doc, target, source, checkpoint, live=done,
                                 done_label=f"done before (by {llm_label()}, live, or already in {target})")
    except Exception as e:                            # NLLB not here (no network): the model's lines are not lost
        note(f"NLLB could not translate the {rest} line(s) {llm_label()} does not cover ({type(e).__name__}: {e}); they are "
             f"kept in the original, and the translation is made again next time")
        doc.setdefault("_incomplete", set()).add(target)
        return [done.get(i, l.get("text")) for i, l in enumerate(lines)]


def translate(doc, target, offline=False, source=None, checkpoint=None):
    """The target's text for every line (None for a note), cached in the doc by language code and model. A line
    translated live (`tr` on the line, made the same way) is not translated again."""
    if offline and llm_on():
        key = f"{llm_code(target) or nllb_code(target) or target} [{llm_repo()}]"
    else:
        key = f"{nllb_code(target) or target} [{NLLB}]" if offline else target
    cached = doc["translations"].get(key)
    if cached:
        return cached
    done = translated_live(doc["lines"], target, offline, source)
    if offline and llm_on():
        out = translate_llm_doc(doc, target, source, checkpoint, live=done)
        if target not in (doc.pop("_incomplete", None) or ()):     # a part left in the original is not kept as done
            doc["translations"][key] = out
        return out
    if offline:
        out = translate_offline(doc, target, source, checkpoint, live=done) if done else translate_offline(doc, target, source, checkpoint)
        doc["translations"][key] = out
        return out
    import anthropic

    client, complete = anthropic.Anthropic(), True
    out = [None] * len(doc["lines"])
    for i, text in done.items():
        out[i] = text
    spoken = [(i, l) for i, l in enumerate(doc["lines"]) if "text" in l and i not in done]    # note lines are not translated
    for start in range(0, len(spoken), BATCH):
        batch = spoken[start:start + BATCH]
        if EVENTS:
            ev("PROG", start // BATCH + 1, (len(spoken) + BATCH - 1) // BATCH)
        got, whole = claude_translate(client, target, [l["text"] for _, l in batch])
        if not whole:
            complete = False
            missing = sum(1 for n in range(len(batch)) if str(n) not in got)
            note(f"{missing} line(s) came back untranslated from Claude and were kept in the original; this translation was not "
                 "cached, run it again to retry")
        for n, (i, l) in enumerate(batch):
            out[i] = got.get(str(n), l["text"])
    if done:
        note(f"{len(done)} line(s) were translated into {target} live and are reused")
    if complete:
        doc["translations"][key] = out
    return out


def nllb_translate(items, tgt, translator, tok):
    """The texts of `items` ((source code, text) pairs) in `tgt`, with the local model: sentence by sentence, long
    sentences in pieces (as translate_offline cuts them), in batches of 16."""
    jobs, shape = [], {}                             # (item, sentence, piece, source, text); (item, sentence) -> pieces
    for k, (src, text) in enumerate(items):
        for n, sentence in enumerate(sentences(text)):
            cut = split_long(sentence, tok)
            shape[(k, n)] = len(cut)
            jobs.extend((k, n, p, src, piece) for p, piece in enumerate(cut))
    results = {}
    for start in range(0, len(jobs), 16):
        batch = jobs[start:start + 16]
        encoded = [tok.encode(piece, add_special_tokens=False).tokens[:400] for *_, piece in batch]
        tokens = [[src] + e + ["</s>"] for (_, _, _, src, _), e in zip(batch, encoded)]
        longest = max(len(e) for e in encoded)
        for (k, n, p, _, piece), r in zip(batch, translator.translate_batch(tokens, target_prefix=[[tgt]] * len(batch), beam_size=4,
                                                                           max_decoding_length=min(512, 2 * longest + 16))):
            hyp = r.hypotheses[0][1:]                                      # [1:] drops the target-language token
            ids = [tok.token_to_id(t) for t in hyp]
            results[(k, n, p)] = tidy(tok.decode([x for x in ids if x is not None], skip_special_tokens=True).strip(), tgt) or piece
    joiner = "" if tgt in NO_SPACES else " "
    out = []
    for k, (src, text) in enumerate(items):
        count = sum(1 for (kk, _) in shape if kk == k)
        out.append(joiner.join(joiner.join(results[(k, n, p)] for p in range(shape[(k, n)])) for n in range(count)))
    return out


def _nllb_texts(lines, target, source=None):
    """translate_texts with NLLB: a line already in the target stays as it is, one of a language NLLB lacks gets None."""
    tgt = nllb_code(target)
    sources = [nllb_code(l.get("lang") or source) for l in lines]
    todo = [(i, src) for i, src in enumerate(sources) if src and src != tgt]
    out = [l["text"] if src == tgt else None for l, src in zip(lines, sources)]
    if todo:
        translator, tok = get_nllb(keep_engine=True)
        for (i, _), text in zip(todo, nllb_translate([(src, lines[i]["text"]) for i, src in todo], tgt, translator, tok)):
            out[i] = text
    return out


_NLLB_FAILED = []            # live: NLLB could not be loaded beside TranslateGemma or Hy-MT3 once; not tried again each chunk


def translate_texts(lines, target, offline=False, source=None):
    """The `target` text of each line (a dict with text and lang) right away, for the live transcript, or None for a
    line that got none: offline with the local model (a line already in the target stays as it is; one of a language
    the model does not have gets None, so the final pass counts it in its note), else one request to Claude (a line
    missing from the reply, or cut off at its end, gets None, so the final pass sends it again). With --translator
    gemma or hy, the lines that model covers go to it and the others to NLLB."""
    if offline and llm_on():
        out = [None] * len(lines)
        todo, rest = [], []
        for i, l in enumerate(lines):
            pair = llm_pair(l.get("lang"), target, source)
            if not pair:
                rest.append(i)
            elif same_language(*pair):
                out[i] = l["text"]
            else:
                todo.append((i, pair))
        texts, cut = llm_translate([(src, lines[i]["text"]) for i, (src, _) in todo], llm_code(target),
                                   keep_engine=True) if todo else ([], 0)
        for (i, _), text in zip(todo, texts):
            out[i] = text
        if cut:
            print(f"{cut} line(s) ran to the token limit and may be cut short or repeat", file=sys.stderr)
        if rest and not _NLLB_FAILED:                # the lines outside the model's languages: NLLB, beside it
            try:
                for i, text in zip(rest, _nllb_texts([lines[i] for i in rest], target, source)):
                    out[i] = text
            except Exception as e:                   # NLLB not here (no network): those lines wait for the final pass
                _NLLB_FAILED.append(e)
                print(f"NLLB could not be loaded ({type(e).__name__}: {e}); the lines {llm_label()} does not cover are "
                      "translated at the end", file=sys.stderr)
        return out
    if offline:
        return _nllb_texts(lines, target, source)
    import anthropic

    got, _ = claude_translate(anthropic.Anthropic(), target, [l["text"] for l in lines])
    return [got.get(str(n)) for n in range(len(lines))]


def translate_live(lines, targets, offline=False, source=None):
    """Adds `tr` ({target: text}), `tr_of` (the text that was translated), `tr_by` (the model: "claude", NLLB,
    TranslateGemma or Hy-MT3) and `tr_offline` to the spoken lines of one decoded chunk, one target after another; a
    line that got no translation into a target has none for it."""
    spoken = [l for l in lines if "text" in l]
    if not spoken:
        return
    for target in targets:
        for l, text in zip(spoken, translate_texts(spoken, target, offline, source)):
            if text is None:
                continue
            l.setdefault("tr", {})[target] = text
            l["tr_of"] = l["text"]
            l.setdefault("tr_by", {})[target] = line_translator(l.get("lang"), target, offline, source)
            l["tr_offline"] = bool(offline)


# ---------- output ----------
ZWSP = "\u200b"      # a zero-width space marks a line whose own text starts with "<" (see fmt)
LINE = re.compile(r"^\[(\d+):(\d\d):(\d\d)\] \[([^\]?\s]+)(\?)?\] (?:<([^<>]+)> )?(.*)$")
NOTE_LINE = re.compile(r"^# (?:\[(\d+):(\d\d):(\d\d)\] )?(.*)$")      # a note: not speech, never translated


def fmt(line, text=None, names=None):
    if "note" in line:                            # a note about the recording or the transcription, kept in every file
        body = line["note"]
        if names:                                 # an overlap note names its speakers too: "… Speaker 1 / Speaker 10" (whole names only)
            alt = "|".join(re.escape(k) for k in sorted(names, key=len, reverse=True))
            body = re.sub(rf"(?<= )(?:{alt})(?= / )|(?<=/ )(?:{alt})$", lambda m: names[m.group(0)], body)
        return "# " + (f"[{hms(line['t'])}] " if line.get("t") is not None else "") + body
    if line["t"] is None:                         # a plain line of a loaded text without time and language
        return text or line["text"]
    t = int(line["t"])
    tag = line["lang"] + ("?" if line["conf"] < LOW_CONF or line.get("guess") else "")
    speaker = (names or {}).get(line.get("speaker"), line.get("speaker"))   # "Speaker 1" renamed to a real name
    who = f"<{speaker}> " if speaker else ""                                  # who spoke
    body = text or line["text"]
    if not who and body.startswith("<"):
        body = ZWSP + body          # Whisper's own "<laughs>" must not be read as a speaker name
    return f"[{t // 3600:02}:{t % 3600 // 60:02}:{t % 60:02}] [{tag}] {who}{body}"


def parse_transcript(text):
    """Load a transcript text: our own '[hh:mm:ss] [xx] text' lines (a call's also '<Name> ' before the text); any other non-empty line is kept as plain text."""
    lines = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        n = NOTE_LINE.match(raw)
        if n:
            h, mi, sec, text = n.groups()
            lines.append({"t": int(h) * 3600 + int(mi) * 60 + int(sec) if h else None, "note": text})
            continue
        m = LINE.match(raw)
        if m:
            h, mi, sec, lang, guess, speaker, body = m.groups()
            body = body.removeprefix(ZWSP)
            line = {"t": int(h) * 3600 + int(mi) * 60 + int(sec), "lang": lang, "conf": 0.0 if guess else 1.0, "text": body}
            if speaker:
                line["speaker"] = speaker
            lines.append(line)
        else:
            lines.append({"t": None, "lang": None, "conf": 1.0, "text": raw})
    return {"lines": lines, "translations": {}}


def render(doc, texts=None):
    names = doc.get("speakers") or {}
    return "\n".join(fmt(l, t, names) for l, t in zip(doc["lines"], texts or [None] * len(doc["lines"]))) + "\n"


def rename_speakers(d, mapping):
    """Gives the speakers of a stored meeting real names ({"Speaker 1": "Anna"}): kept in transcript.json for every later
    file, and put into the text files already in the folder (<Speaker 1> becomes <Anna> there)."""
    doc = _load(d)
    names = dict(doc.get("speakers") or {})
    shown = {v: k for k, v in names.items()}                 # a name already given can be renamed again
    changes = []                                             # (old name in the files, new name)
    for label, name in mapping.items():
        name = speaker_name(name, "")
        if not name:
            raise ValueError(f"no name for {label}")
        key = shown.get(label, label)
        changes.append((names.get(key, key), name))
        names[key] = name
    doc["speakers"] = names
    _save(d, doc)
    changed = []
    # One pass over every file with all the renames at once: swapping two names ("Anna=Boris" "Boris=Anna") must not rename
    # the already renamed text a second time.
    names_alt = "|".join(re.escape(old) for old, _ in sorted(changes, key=lambda c: -len(c[0])))
    # A name as the speaker tag "<Name> ", or in an overlap note "… Name / Name" (first or second place).
    pattern = re.compile(rf"(?<=<)(?:{names_alt})(?=> )|(?<= )(?:{names_alt})(?= / )|(?<=/ )(?:{names_alt})(?=$)", re.M)
    table = dict(changes)
    elsewhere = [Path(f) for out in (doc.get("outputs") or {}).values() for f in out.get("files", [])]    # the output folder's
    files = sorted(d.glob("*.txt")) + [p for p in elsewhere if p.suffix == ".txt" and p.parent != d and p.exists()]
    for path in files:
        text = path.read_text(encoding="utf-8")
        new = pattern.sub(lambda m: table[m.group(0)], text) if changes else text
        if new != text:
            write_output(path, new)
            changed.append(path)
            announce(path)
    return changed


def announce(path):
    """A file is written: the app learns of it at once (a later failure never hides it), the terminal gets its path."""
    ev("OUT", path) if EVENTS else print(path, flush=True)


def translation_file_name(stem, lang):
    """<stem>.<Language>.txt: the language as it was asked for, made a safe file name."""
    name = re.sub(r"[^\w-]+", "_", lang).strip("_") or "translation"
    return f"{stem}.{name}.txt"


def write_outputs(doc, out_dir, stem, targets, transcript=True, protect=None, offline=False, source=None, data=None, audio=(),
                  owner=None):
    """<stem>.transcript.txt and one <stem>.<Language>.txt per target; returns the paths written. Each file is announced
    as soon as it exists. With a data folder the doc (and so every finished translation) is saved there after each target,
    and an offline translation cut short is continued from its pieces. `audio` lists (file, suffix) pairs of the recording
    to keep beside the text files (`<stem>.wav`; a call's `<stem>.mic.wav` and `<stem>.system.wav`) when the output
    folder is the shared one (MT_OUT): there a meeting's folder holds everything about it. In that folder the stem is
    claimed (claim_stem; `owner` names the input, by default the doc's source or the data folder) so two meetings of one
    hour keep their files apart, and the files written are remembered in the doc (`outputs`), for a rerun and for `rename`."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        if sys.platform == "win32":
            windows_refused("make the folder", out_dir)
        raise
    beside_input = protect is not None and Path(protect).resolve().parent == out_dir.resolve()
    if OUT is not None and _within(out_dir, OUT) and not beside_input:
        stem = claim_stem(out_dir, stem, doc, owner or source_id(doc, data))
    paths = []

    def remember():
        if data is not None:
            earlier = (doc.get("outputs") or {}).get(str(out_dir)) or {}
            before = (earlier.get("files") or []) if earlier.get("stem") == stem else []
            files = list(dict.fromkeys(before + [str(p) for p in paths]))   # a later translation adds to the list
            doc.setdefault("outputs", {})[str(out_dir)] = {"stem": stem, "files": files}
            _save(data, doc)

    if transcript:
        paths.append(out_dir / f"{stem}.transcript.txt")
        write_output(paths[-1], render(doc))                    # written first: kept even if a translation fails
        announce(paths[-1])
    if OUT is not None:
        for src, suffix in audio:
            kept = keep_audio(src, out_dir / f"{stem}{suffix}", data)
            if kept:
                announce(kept)
                paths.append(kept)
    if paths:
        remember()
    shown = {}            # per line: every translation sent with it so far (the window replaces a line's translations)
    for k, lang in enumerate(targets, 1):
        path = out_dir / translation_file_name(stem, lang)
        if EVENTS:
            ev("STAGE", f"Translating to {lang}" + (f" ({k} of {len(targets)})" if len(targets) > 1 else ""))
        if protect and path.resolve() == protect.resolve():
            print(f"skipped {path}: it is the input file", file=sys.stderr)
            continue
        texts = translate(doc, lang, offline, source, checkpoint=data)
        write_output(path, render(doc, texts))
        announce(path)
        paths.append(path)
        remember()
        if EVENTS:                                   # the window shows the translation beside each line, as it does live
            for i, (line, text) in enumerate(zip(doc["lines"], texts)):
                if "text" in line and text is not None:
                    if i not in shown:               # the live ones of this text, not those of an older text
                        shown[i] = dict(line.get("tr") or {}) if line.get("tr_of") == line["text"] else {}
                    shown[i][lang] = text            # with the targets of this job before it
                    ev("LINE", json.dumps(dict(line, tr=shown[i], tr_of=line["text"]), ensure_ascii=False))
    return paths


def show(d, target, offline=False):
    doc = _load(d)
    texts = translate(doc, target, offline) if target else None
    if target:
        _save(d, doc)
    print(render(doc, texts), end="")


def _need_key(targets, offline=False):
    """Fail at once, before a long transcription, when the translation cannot work."""
    global TRANSLATOR
    if offline and targets and llm_on() and not mlx_lm_available():
        # A recording must not be refused over the translation: NLLB translates instead, and the note says why.
        note(f"{llm_label()} is not available here (it needs Apple silicon and the mlx-lm package: pip install -r "
             "requirements-mlx.txt, or in the app Set Up Speech Engine Again… if its Apple-GPU step failed); NLLB translates instead")
        TRANSLATOR = "nllb"
    if offline:
        unknown = [t for t in targets if not nllb_code(t)]
        raw = [nllb_code(t) for t in targets if nllb_code(t) and nllb_code(t) not in NLLB_LANGS]
        if raw:
            try:                                     # a code such as xxx_Latn passes the pattern; the model's own vocabulary decides
                from huggingface_hub import snapshot_download
                from tokenizers import Tokenizer
                vocab = Tokenizer.from_file(os.path.join(snapshot_download(NLLB, local_files_only=True), "tokenizer.json")).get_vocab()
                unknown += [code for code in raw if code not in vocab]
            except Exception:
                pass                                 # the model is not downloaded yet: nothing to check against
        if unknown:
            sys.exit(f"Offline translation does not know {', '.join(map(repr, unknown))}. Use a name or code such as "
                     f"English, ru, Brazilian Portuguese, or an NLLB code like ces_Latn. Nothing was started.")
    elif targets and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Translation needs ANTHROPIC_API_KEY (" + windows_text('$env:ANTHROPIC_API_KEY="sk-ant-..." in PowerShell',
                                                                    "export ANTHROPIC_API_KEY=sk-ant-...") + "), or use --offline. "
                 "Nothing was started.")


def _data_path(into, name, stamp=None, n=1):
    """The data folder _data_dir makes or reuses, without making it: `<yyyymmdd-hhmmss>-<name>`, or with n > 1 the n-th
    one of that second (`…-<name>-2`; `…-recording-2` without a name). On Windows name and number together stay within
    WIN_NAME_MAX, the name cut as far as needed. A numbered folder can still be up to 4 characters longer than the first
    one, which the job's windows_path_problem checked before it began, so _new_dir checks it again (`check`)."""
    if into:
        return Path(into) if Path(into).is_dir() else DATA / into
    name = windows_folder_name(name)                 # an input file's own long name must not make the path too long there
    if n > 1:
        # A nameless folder is read as "recording" (translate's stem, the app's list and --stem): numbered it stays that,
        # as `recording-2` in the output folder's hour (claim_stem), where a bare `-2` would make the name "2".
        name = name or "recording"
        suffix = f"-{n}"
        if sys.platform == "win32":
            while utf16_len(name + suffix) > WIN_NAME_MAX:
                name = name[:-1]
        name += suffix
    return DATA / ((stamp or datetime.now().strftime("%Y%m%d-%H%M%S")) + (f"-{name}" if name else ""))


def _new_dir(name, check=None):
    """A new data folder. Another job that began in the same second under the same name (a batch of short clips with one
    file name in two folders; two processes, such as a recording and a queued transcription; on Windows also two long
    names that are the same in their first WIN_NAME_MAX characters) takes `-2`, `-3` … of that second instead of failing
    with FileExistsError. mkdir without exist_ok decides, so two jobs never share one folder; the app makes the folder of
    a call it records itself, by the same names and numbers and the same rule (app/Sources/Core/DataFolder.swift, since
    1.11.2), so a call and an mt.py job do not share one either. `check`, when given,
    is the job's Windows path check (it raises): a numbered folder is up to 4 characters longer than the first one, which
    the job checked before it began, so it is checked again, and on a refusal removed again, before anything uses it."""
    DATA.mkdir(parents=True, exist_ok=True)          # a DATA that is a file or a dangling link fails here, under its name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        d = _data_path(None, name, stamp, n)
        try:
            d.mkdir()                                # without parents: a FileExistsError is this folder, never DATA
        except FileExistsError:
            continue
        if check and n > 1:                          # the first one is the folder the job checked
            try:
                check(d)
            except BaseException:
                d.rmdir()
                raise
        return d
    raise FileExistsError(f"{_data_path(None, name, stamp)}: 999 data folders of that name in one second")


def _data_dir(into, name, check=None):
    """The data folder of a job: --into names an existing folder or a data id (a run cut short is continued there);
    otherwise a new folder (_new_dir, `check` as there)."""
    if not into:
        return _new_dir(name, check)
    d = _data_path(into, name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def windows_long_paths():
    """Whether Windows lets programs use paths over 259 characters (LongPathsEnabled = 1; Python is long-path aware)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return winreg.QueryValueEx(key, "LongPathsEnabled")[0] == 1
    except Exception:
        return False


def windows_path_problem(folder, targets, offline, stem=None):
    """Windows without long paths: what makes a file this job writes under its data folder `folder` too long (as words:
    the path and its length), or None. The longest are an offline translation's log in translations\\ and, when the text
    files stay in the data folder (a recording without --out or MT_OUT; `stem`), <stem>.<Language>.txt. WIN_NAME_MAX
    keeps them within 259 characters for a short data folder and a short --to; this check covers any other (a checkout
    deep in OneDrive, a long MT_HOME, MT_NLLB_MODEL or --to). Elsewhere, and with long paths on, None."""
    if sys.platform != "win32" or windows_long_paths():
        return None
    folder = Path(os.path.abspath(folder))
    files = []
    if offline and targets:                          # zho_Hant: every NLLB code has 8 characters
        files += [folder / "translations" / translation_log_name("zho_Hant", model, "0" * 16)
                  for model in ((NLLB, llm_model_name()) if llm_on() else (NLLB,))]
    if stem:
        files += [folder / f"{stem}.transcript.txt"] + [folder / translation_file_name(stem, t) for t in targets]
    longest = max(files, key=lambda p: utf16_len(str(p)), default=None)
    if longest is None or utf16_len(str(longest)) <= 259:
        return None
    return (f"{longest} would be {utf16_len(str(longest))} characters long, and without long paths Windows allows 259: "
            "use a shorter --name, --stem or --to, a data folder with a shorter path (MT_HOME), or switch long paths on "
            "(LongPathsEnabled)")


def hour_stamp(data=None):
    """The subfolder name of a meeting in the output folder: the hour it began, as `yy-mm-dd hh` (`26-11-22 14`), read from
    its data folder's time stamp (`20261122-140512-weekly`), so a transcription queued later lands with its recording; a
    job without one (a file transcribed now) takes the current hour."""
    when = None
    m = re.match(r"(\d{8}-\d{4})\d{2}", Path(data).name) if data is not None else None
    if m:
        try:
            when = datetime.strptime(m.group(1), "%Y%m%d-%H%M")
        except ValueError:
            pass
    return (when or datetime.now()).strftime("%y-%m-%d %H")


def result_folder(explicit, default, data=None):
    """Where the text files of a job go: --out when given; else, with MT_OUT set (the app's output folder), that folder's
    subfolder for the hour the meeting began (hour_stamp); else `default` (next to the input, or the data folder)."""
    if explicit:
        return Path(explicit)
    if OUT is None:
        return Path(default)
    return OUT / hour_stamp(data)


OWNERS = ".meeting-transcriber.json"     # in an hour folder: which input each stem's files were made from


def source_id(doc, fallback=None):
    """What a meeting's files are made from, the same on every run: the input file(s) as absolute paths (a call's two
    tracks joined by " + "), else `fallback` (the data folder, the transcript file)."""
    source = doc.get("source")
    if source:
        return " + ".join(str(Path(p).resolve()) for p in str(source).split(" + "))
    return str(Path(fallback).resolve()) if fallback else None


def _owners(out_dir):
    try:
        return json.loads((out_dir / OWNERS).read_text(encoding="utf-8"))
    except Exception:
        return {}


def claim_stem(out_dir, stem, doc, owner=None):
    """The stem of this meeting's files in out_dir, when meetings share a folder (two recordings in one hour): the stem its
    earlier files there carry (a later translation: doc["outputs"]); else `stem`, when no file of that name is there or
    those files were made from the same input (`owner`, a rerun: they are replaced, as next to an input file); else
    `stem-2`, `stem-3` … so another meeting's files are not overwritten. The choice is written down in the folder (OWNERS)."""
    earlier = (doc.get("outputs") or {}).get(str(out_dir))
    if earlier and earlier.get("stem"):
        return earlier["stem"]
    owners = _owners(out_dir)
    candidate, k = stem, 1
    while any(out_dir.glob(glob_escape(candidate) + ".*")) and not (owner and owners.get(candidate) == owner):
        k += 1
        candidate = f"{stem}-{k}"
    if candidate != stem:
        note(f"Another meeting's files in {out_dir} are called {stem}.*; this one's are called {candidate}.*")
    if owner and owners.get(candidate) != owner:
        owners[candidate] = owner
        try:
            tmp = out_dir / (OWNERS + ".tmp")
            tmp.write_text(json.dumps(owners, ensure_ascii=False, indent=1), encoding="utf-8")
            replace_file(tmp, out_dir / OWNERS)
        except OSError:
            pass                                     # without it a rerun numbers its files: nothing is overwritten either way
    return candidate


def _within(path, folder):
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def glob_escape(text):
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def keep_audio(src, dest, data):
    """The recording's audio beside its transcript in the output folder: `src` (a file of the data folder `data`, not a
    file the user gave) becomes `dest`, cloned on APFS (`cp -c`: no second copy of the bytes), else copied. Returns dest,
    or None when nothing was done (not a recording of ours, or already there)."""
    src, dest = Path(src), Path(dest)
    try:
        if data is None or src.resolve().parent != Path(data).resolve() or not src.is_file():
            return None
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return None
        if sys.platform == "darwin":
            done = subprocess.run(["cp", "-c", str(src), str(dest)], capture_output=True, **child_options())
            if done.returncode == 0:
                return dest
        import shutil
        shutil.copy2(src, dest)
        return dest
    except Exception as e:                           # the text files matter more: a failed copy is a line, not a failure
        print(f"The recording could not be copied to {dest} ({type(e).__name__}: {e}); it stays in {src.parent}", file=sys.stderr)
        return None


def unique_stems(files):
    """One result-file stem per input, for a batch written into one folder: a file keeps its own name unless an earlier
    file of the batch has it; then its parent folder's name goes in front (`b-audio`), and if that is taken too (by an
    input really called so, or by an earlier prefixed one) a counter follows (`b-audio-2`). Every stem of the batch,
    original or made, is reserved before any is handed out, so no two results share a name."""
    paths = [Path(f) for f in files]
    originals = collections.Counter(p.stem for p in paths)
    taken = set(originals)                                   # every input's own name is spoken for
    out, used = [], set()
    for p in paths:
        stem = p.stem
        if stem in used or originals[stem] > 1 and stem in used:
            stem = f"{p.parent.name}-{p.stem}" if p.parent.name else f"{p.stem}-2"
            n = 2
            while stem in taken or stem in used:
                stem, n = (f"{p.parent.name}-{p.stem}-{n}" if p.parent.name else f"{p.stem}-{n}"), n + 1
            taken.add(stem)
        used.add(stem)
        out.append(stem)
    return out


def _each(files, work):
    """Run work(path) for every file; report a failure and go on with the next. Exit code 1 if any failed."""
    failed = 0
    for n, f in enumerate(files, 1):
        if EVENTS:
            ev("FILE", n, len(files), f)
        try:
            work(Path(f))                            # every file written is announced as it is written
        except Exception as e:                       # one bad file must not stop the rest of a batch
            failed += 1
            ev("ERR", f, e) if EVENTS else print(f"error: {f}: {e}", file=sys.stderr)
    if EVENTS:
        ev("DONE", failed)
    if failed:
        sys.exit(1)


GRANULARITY_HELP = ("how finely speech is cut for language detection (each chunk gets one language and costs a full Whisper pass): "
                    "coarse (pauses of 1 s), normal (0.7 s), fine (0.35 s), finest (0.2 s: rapid turn-taking, slowest); "
                    "adaptive = normal, but a chunk that sounds mixed is re-cut at 0.2 s and each part detected on its own; "
                    "auto (default) = finest when several --language are given, else adaptive")
LANGUAGE_HELP = ("languages spoken, comma-separated (ru,en,de or Russian,English): each speech chunk is detected among "
                 "these only; one language forces it; omit to detect any of ~100")


def preflight(hours=0.0, tracks=1, offline_targets=(), fetch=True):
    """Before a long job: enough disk for the recording, the offline translation model present (fetched now rather than
    after hours of work), and a word about memory on a small machine. Notes, not refusals, except for the disk."""
    import shutil

    if hours:
        need = int(hours * 3600 * RATE * 2 * tracks)
        try:
            free = shutil.disk_usage(DATA if DATA.exists() else DATA.parent).free
        except OSError:
            free = None
        if free is not None and free < 2 * need:
            sys.exit(f"Not enough free disk for {_limit_text(int(hours * 3600))} of recording: {need // 2**20} MB are needed "
                     f"(twice that to be safe), {free // 2**20} MB are free. Nothing was started.")
    if offline_targets:
        # TranslateGemma or Hy-MT3 when a target is one of its languages, and NLLB in every case: beside either it takes
        # the lines of a language that model lacks, which Whisper may tag in any meeting. A recording does not wait for
        # them (fetch=False): its live worker fetches them beside the work, and its transcription before translating.
        for model, size in (translation_models(offline_targets, True) if fetch else []):
            try:
                fetch_model(model, size)
            except Exception as e:                   # no internet: the transcript must not be lost over the translation
                why = f"{type(e).__name__}: {e}" if isinstance(e, RuntimeError) else type(e).__name__   # our own advice
                print(f"The translation model could not be fetched now ({why}); it is tried again when the "
                      "translation starts, and the transcript is written first either way", file=sys.stderr)
    total = total_memory()
    if total and total < 16 * 2**30 and offline_targets and TRANSLATOR == "gemma":
        print(f"This computer has {total // 2**30} GB of memory: TranslateGemma needs about 7 GB (8.5 GB beside the speech "
              "model while recording); with less than 16 GB the Mac may slow down, and --translator nllb needs 1.4 GB", file=sys.stderr)
    # 12 GB for Hy-MT3: about 6 GB with the speech model is three quarters of an 8 GB Mac and well inside 16 GB (estimates).
    if total and total < 12 * 2**30 and offline_targets and TRANSLATOR == "hy":
        print(f"This computer has {total // 2**30} GB of memory: Hy-MT3 needs about 4.5 GB (about 6 GB beside the speech model "
              "while recording; estimates); with less than 12 GB the Mac may slow down, and --translator nllb needs 1.4 GB",
              file=sys.stderr)
    if total and total < 12 * 2**30 and hours >= 2:
        # Advice about the machine, not about the recording: a plain line (the app's log), not a note, which the app would
        # put at the top of the transcript with the recording's own notes.
        print(f"This computer has {total // 2**30} GB of memory: a recording this long needs about 3 GB to transcribe; close other big programs",
              file=sys.stderr)


def _speakers(a):
    """--speakers as transcribe_lines wants it: None (off), "auto", or a count; checked before anything long starts."""
    value = getattr(a, "speakers", None)
    if value is None:
        return None
    if value == "auto":
        return "auto"
    if not value.isdigit() or int(value) < 1:
        sys.exit(f"--speakers takes a number of speakers or auto, not {value!r}. Nothing was started.")
    return int(value)


def _check_spoken(spec):
    """Fail at once, before a long job, when a language name is unknown."""
    try:
        parse_languages(spec)
    except ValueError as e:
        sys.exit(f"{e} Nothing was started.")


def utf8_streams():
    """Windows: stdout and stderr in UTF-8. A Windows pipe gets the ANSI code page (cp1252 ...), where an @@LINE with
    Cyrillic raised UnicodeEncodeError and ended the job; and UTF-8 is the default only from Python 3.15 (PEP 686).
    stdout replaces what UTF-8 cannot carry (a lone surrogate from an undecodable file name) instead of failing, and keeps
    "\\n" as the line end (the @@ protocol is one event per line). macOS and Linux keep the streams Python gives them, as
    before: there the locale (or PYTHONIOENCODING, which the Mac app sets to utf-8) decides, a Latin-1 terminal keeps
    getting Latin-1 and an undecodable file name its own bytes (surrogateescape). A stream that cannot be reconfigured
    is left as it is."""
    if sys.platform != "win32":
        return
    for stream, options in ((sys.stdout, dict(errors="replace", newline="\n")), (sys.stderr, dict(errors="backslashreplace"))):
        try:
            stream.reconfigure(encoding="utf-8", **options)
        except Exception:                                # None (pythonw.exe), or a stream without reconfigure()
            pass


def main():
    global TRANSLATOR                                    # --translator chooses the offline translator for this run
    utf8_streams()
    import signal
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)    # "Extend" before record() has its handler: ignored, not fatal (the default)
        signal.signal(signal.SIGUSR2, signal.SIG_IGN)    # "Pause" likewise
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="meeting-transcriber " + VERSION)
    sub = p.add_subparsers(dest="cmd", required=True)
    to = dict(action="append", metavar="LANGUAGE", default=[],
              help="also write a translation file (repeat for several): English, Russian, Spanish ...")
    note_arg = dict(action="append", metavar="TEXT", default=[], dest="notes",
                    help="a note to put at the top of the transcript (repeatable): what the app knows about the recording")
    offline = dict(action="store_true", help="translate on this computer instead of with Claude (no API key; "
                   "the first time it downloads a 1.4 GB model; quality is below Claude's)")
    translator_arg = dict(choices=["nllb", "gemma", "hy"], default=None,
                          help="the offline translator: nllb (default; 1.4 GB, about 200 languages, CPU), gemma "
                               "(TranslateGemma 12B: Apple silicon and mlx-lm, 7 GB, 51 languages) or hy (Hy-MT3 7B: "
                               "Apple silicon and mlx-lm, 4.2 GB, 102 languages); lines outside gemma's or hy's languages "
                               "go to nllb")
    r = sub.add_parser("record", help="record from the microphone, then transcribe")
    r.add_argument("--name", help="a short name for the recording (becomes part of the folder and file names)")
    r.add_argument("--hours", type=float, default=4, help=f"stop by itself after this much recorded audio (default 4, at most {MAX_HOURS}; "
                   "a notice comes 10 minutes before, and " + windows_text("the line extend on stdin, with MT_CONTROL=stdin,",
                                                                            "kill -USR1 <pid>") + " adds an hour)")
    r.add_argument("--device", help="the microphone to record: an input device's index, its exact name, a part of its name that "
                   "fits one device only, or NAME#2 for the second of two inputs with the same name (see: devices); "
                   "default: the system's current input")
    r.add_argument("--language", metavar="LANGUAGES", help=LANGUAGE_HELP)
    r.add_argument("--granularity", choices=["auto", *GRANULARITY], default="auto", help=GRANULARITY_HELP)
    speakers_arg = dict(metavar="N|auto", help="label who spoke (several people on one microphone; in a call both "
                        "sides, You 1 … and Them 1 …): how many speakers (in a call: on the other side; yours are found by "
                        "themselves), or auto to find out")
    diarizer_arg = dict(choices=list(DIARIZERS), default="sherpa", help="how speakers are told apart: sherpa (built in) or "
                        "pyannote (optional, better, slower; pip install -r requirements-pyannote.txt and a Hugging Face token in HF_TOKEN)")
    r.add_argument("--speakers", **speakers_arg)
    r.add_argument("--diarizer", **diarizer_arg)
    r.add_argument("--no-transcribe", action="store_true", help="only record; transcribe later (transcribe <folder>/audio.wav "
                   "--into <folder>, or call <folder>/mic.wav <folder>/system.wav --into <folder>)")
    r.add_argument("--system-device", metavar="DEVICE", help="a second input that carries what the computer plays (the other "
                   "side of a call): a monitor source on Linux, BlackHole or similar on macOS, Stereo Mix on Windows; "
                   "an index, an exact name or a part of a name that fits one device only (see: devices); "
                   "recorded as its own track, so every line says who spoke")
    r.add_argument("--me", default="You", help="name for your microphone in the transcript (default: You)")
    r.add_argument("--them", default="Them", help="name for the other side (default: Them)")
    r.add_argument("--keep-echo", action="store_true", help="keep microphone lines that only repeat the other side (heard from the speakers)")
    r.add_argument("--live", action="store_true", help="transcribe while recording (the lines appear as the meeting goes on; at the "
                   "end only the tail is left to do; the speech model is loaded at the start); with --to every line is "
                   "translated live as well")
    r.add_argument("--out", help="folder for the text files (default: the recording's folder in data/, or with MT_OUT set a "
                   "subfolder of that folder named after the hour the recording began, yy-mm-dd hh, holding the audio too)")
    r.add_argument("--to", **to), r.add_argument("--offline", **offline), r.add_argument("--note", **note_arg)
    r.add_argument("--translator", **translator_arg)
    c = sub.add_parser("call", help="transcribe a call from two recordings: your microphone and the other side (system audio)")
    c.add_argument("mic", help="the microphone recording")
    c.add_argument("system", help="the recording of what the computer played")
    c.add_argument("--out", help="folder for the text files (default: next to the microphone file)")
    c.add_argument("--into", metavar="FOLDER", help="the data folder to use (a run cut short goes on from its checkpoint there)")
    c.add_argument("--stem", default="call", help="name of the result files (default: call)")
    c.add_argument("--me", default="You", help="name for your microphone in the transcript (default: You)")
    c.add_argument("--them", default="Them", help="name for the other side (default: Them)")
    c.add_argument("--keep-echo", action="store_true", help="keep microphone lines that only repeat the other side")
    c.add_argument("--language", metavar="LANGUAGES", help=LANGUAGE_HELP)
    c.add_argument("--granularity", choices=["auto", *GRANULARITY], default="auto", help=GRANULARITY_HELP)
    c.add_argument("--speakers", **speakers_arg)
    c.add_argument("--diarizer", **diarizer_arg)
    c.add_argument("--to", **to), c.add_argument("--offline", **offline), c.add_argument("--note", **note_arg)
    c.add_argument("--translator", **translator_arg)
    t = sub.add_parser("transcribe", help="transcribe audio (or video) files into text files")
    t.add_argument("files", nargs="+", help="audio or video files: mp3, m4a, wav, flac, ogg, opus, mp4, mov, mkv, webm, avi ... (a video's sound is used)")
    t.add_argument("--out", help="folder for the text files (default: next to each audio file)")
    t.add_argument("--into", metavar="FOLDER", help="the data folder to use (a run cut short goes on from its checkpoint there)")
    t.add_argument("--stem", help="name of the text files (one file only; default: the audio file's name)")
    t.add_argument("--language", metavar="LANGUAGES", help=LANGUAGE_HELP)
    t.add_argument("--granularity", choices=["auto", *GRANULARITY], default="auto", help=GRANULARITY_HELP)
    t.add_argument("--speakers", **speakers_arg)
    t.add_argument("--diarizer", **diarizer_arg)
    t.add_argument("--track", type=int, help="audio track of a video with several (0 = first, the default)")
    t.add_argument("--to", **to), t.add_argument("--offline", **offline), t.add_argument("--note", **note_arg)
    t.add_argument("--translator", **translator_arg)
    x = sub.add_parser("translate", help="translate transcript text files (the output of transcribe, or any text)")
    x.add_argument("files", nargs="+")
    x.add_argument("--out", help="folder for the translations (default: next to each file)")
    x.add_argument("--to", **{**to, "required": True}), x.add_argument("--offline", **offline)
    x.add_argument("--translator", **translator_arg)
    x.add_argument("--from", dest="source", metavar="LANGUAGE",
                   help="language of lines that carry no [time] [lang] tag (needed with --offline for plain text)")
    s = sub.add_parser("show", help="print a stored transcript, optionally translated")
    s.add_argument("id", nargs="?", help="transcript id (default: latest)")
    s.add_argument("--to", metavar="LANGUAGE", help="e.g. English, Russian, Spanish"), s.add_argument("--offline", **offline)
    s.add_argument("--translator", **translator_arg)
    fo = sub.add_parser("follow", help="transcribe one or two WAV files while they are still being written (the app's calls)")
    fo.add_argument("files", nargs="+", help="mic.wav [system.wav]")
    fo.add_argument("--into", metavar="FOLDER", required=True, help="the data folder of the recording")
    fo.add_argument("--language", metavar="LANGUAGES", help=LANGUAGE_HELP)
    fo.add_argument("--granularity", choices=["auto", *GRANULARITY], default="auto", help=GRANULARITY_HELP)
    fo.add_argument("--me", default="You"), fo.add_argument("--them", default="Them")
    fo.add_argument("--to", **{**to, "help": "also translate every line live, into this language (repeat for several)"})
    fo.add_argument("--offline", **offline)
    fo.add_argument("--translator", **translator_arg)
    sub.add_parser("prepare", help="check ffmpeg and download the speech model (once, before first use)")
    rn = sub.add_parser("rename", help="name the speakers of a stored meeting: rename <id|folder> \"Speaker 1=Anna\" ...")
    rn.add_argument("id", help="a data id or folder with transcript.json")
    rn.add_argument("names", nargs="+", metavar="LABEL=NAME", help="\"Speaker 1=Anna\" (or \"Them 2=Boris\" for a call)")
    sub.add_parser("list", help="list stored transcripts")
    sub.add_parser("devices", help="list the audio devices (the inputs for --device and --system-device, by index or name)")
    a = p.parse_args()
    if getattr(a, "translator", None):
        TRANSLATOR = a.translator
    if a.cmd in ("prepare", "show", *LONG_COMMANDS):
        compute_device()                                 # an unknown MT_DEVICE stops here, before anything long starts
    if control_wanted() and a.cmd in LONG_COMMANDS:
        CONTROL.start()                                  # as early as possible: a stop or cancel while starting counts
    keep_awake(a.cmd)                                    # the computer stays awake while this process runs (not a closed lid)
    try:
        run(a)
    finally:
        # The work is done (or failed): from here on an end of stdin, say a front end closing the pipe on @@DONE, must not
        # turn the exit status into SIGTERM's, nor a late stop raise KeyboardInterrupt while the interpreter exits.
        CONTROL.listen({verb: (lambda: None) for verb in (*CONTROL_VERBS, "eof")})


def run(a):
    """The command main() parsed into `a`."""
    if a.cmd == "list":
        for x in sorted(DATA.glob("*/transcript.json")):
            print(x.parent.name)
    elif a.cmd == "rename":
        mapping = {}
        for item in a.names:
            label, _, name = item.partition("=")
            if not label.strip() or not name.strip():
                sys.exit(f"{item!r}: use LABEL=NAME, for example \"Speaker 1=Anna\". Nothing was changed.")
            mapping[label.strip()] = name.strip()
        folder = Path(a.id)
        rename_speakers(folder if (folder / "transcript.json").exists() else _dir(a.id), mapping)
        if EVENTS:
            ev("DONE", 0)
    elif a.cmd == "prepare":
        import shutil
        ok = ensure_ffmpeg()
        try:
            ensure_diarization_models()
        except Exception as e:                       # no internet, or the package is missing: said, not fatal (fetched at first use)
            note(f"The speaker-separation models could not be fetched now ({e}); they are fetched when first needed")
        ev("CHECK", "ffmpeg", "ok" if ok else "missing") if EVENTS else print("ffmpeg:", "ok" if ok else
                                                                             f"MISSING (could not be fetched; {ffmpeg_hint().split(': ', 1)[1]})")
        if not ok:
            sys.exit(f"ffmpeg and ffprobe are needed and could not be fetched. Connect to the internet and try again, or install them ({ffmpeg_hint()}).")
        ev("STAGE", f"Loading the speech model {WHISPER} (the first time it is downloaded, about 1.6 GB)") if EVENTS else print("loading model", WHISPER)
        get_model()
        if engine_name() == "mlx":                   # the Apple-GPU model as well; the CPU one above stays as the fallback
            from huggingface_hub import snapshot_download
            text = f"Loading the Apple-GPU speech model {mlx_repo()} (the first time it is downloaded, about 1.6 GB)"
            ev("STAGE", text) if EVENTS else print(text, file=sys.stderr)
            snapshot_download(mlx_repo())
        ev("CHECK", "engine", engine_name()) if EVENTS else print("engine:", engine_name())
        ev("READY") if EVENTS else print("ready")
    elif a.cmd == "follow":
        _need_key(a.to, a.offline)
        _check_spoken(a.language)
        preflight(0, 2, a.to if a.offline else (), fetch=False)          # beside a call: the live worker fetches
        follow([Path(f) for f in a.files], _data_dir(a.into, "call"), a.language, a.granularity, a.me, a.them,
               targets=a.to, offline=a.offline)
    elif a.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        print("\nrecord --device <index or name> picks the microphone (a part of the name is enough when it fits one device); "
              "--system-device likewise for the input that carries what the computer plays. The ones marked > and < are the "
              "system's current input and output." + windows_text(
                  " Windows lists every device once per host API (the name after the comma); --device and --system-device "
                  "take the Windows WASAPI ones only.", ""), file=sys.stderr)
    elif a.cmd == "show":
        _need_key([a.to] if a.to else [], a.offline)
        show(_dir(a.id), a.to, a.offline)
    elif a.cmd == "record":
        if not 0 < a.hours <= MAX_HOURS:
            sys.exit(f"--hours must be above 0 and at most {MAX_HOURS}. Nothing was started.")
        check_name(a.name)
        translating = not a.no_transcribe or a.live          # record only: no translation follows, unless the live one
        if translating:
            _need_key(a.to, a.offline)
        _check_spoken(a.language)
        stays = not a.no_transcribe and not a.out and OUT is None    # the text files are written into the data folder

        def refuse(folder):                                # now, and again on a numbered folder (_new_dir)
            problem = windows_path_problem(folder, a.to if translating else [], a.offline,
                                           (a.name or "recording") if stays else None)
            if problem:
                sys.exit(f"{problem}. Nothing was started.")
        refuse(_data_path(None, a.name))
        device, system_device = resolve_device(a.device), resolve_device(a.system_device, "computer's-sound")   # before any download
        preflight(a.hours, 2 if a.system_device else 1, a.to if a.offline and translating else (), fetch=False)
        live = {"language": a.language, "granularity": a.granularity, "me": a.me, "them": a.them,
                "targets": a.to, "offline": a.offline} if a.live else None
        rec_notes = []
        try:
            d, wav, system_wav = record(a.name, a.hours, device, system_device, live, notes=rec_notes, check=refuse)
        except RecordingCancelled as e:
            if e.folder is None:                           # cancelled before the folder was made: nothing was recorded
                note(f"Recording cancelled ({e}) before it began; nothing was recorded")
            else:
                note(f"Recording cancelled ({e}); the audio is kept in {e.folder}, nothing is transcribed")
                announce(e.folder)
            if EVENTS:
                ev("DONE", 0)
            return
        if a.no_transcribe:
            ev("OUT", d) if EVENTS else print(d)
            if EVENTS:
                ev("DONE", 0)
        else:
            if EVENTS:
                ev("STAGE", "Transcribing the recording")
            problems = []
            notes = (a.notes or []) + rec_notes            # what happened while recording goes into the transcript too
            if system_wav:
                doc = transcribe_call(wav, system_wav, d, a.me, a.them, a.language, a.granularity, not a.keep_echo, notes,
                                      speakers=_speakers(a), diarizer=a.diarizer, problems=problems)
            else:
                doc = transcribe(wav, d, a.language, granularity=a.granularity, notes=notes, speakers=_speakers(a), diarizer=a.diarizer)
            audio = [(wav, ".mic.wav"), (system_wav, ".system.wav")] if system_wav else [(wav, ".wav")]
            _each([wav], lambda _: write_outputs(doc, result_folder(a.out, d, d), a.name or "recording", a.to,
                                                         offline=a.offline, data=d, audio=audio))
            if problems:
                sys.exit(1)
    elif a.cmd == "call":
        check_name(a.stem, "stem")
        _need_key(a.to, a.offline)
        _check_spoken(a.language)
        preflight(0, 2, a.to if a.offline else ())
        for f in (a.mic, a.system):
            if not Path(f).is_file():
                sys.exit(f"No such file: {f}")
        mic = Path(a.mic)

        def refuse(folder):                                # now, and again on a numbered folder (_new_dir)
            problem = windows_path_problem(folder, a.to, a.offline)
            if problem:
                sys.exit(f"{problem}. Nothing was started.")
        refuse(_data_path(a.into, a.stem))
        d = _data_dir(a.into, a.stem, refuse)
        problems = []
        doc = transcribe_call(mic, Path(a.system), d, a.me, a.them, a.language, a.granularity, not a.keep_echo, a.notes,
                              speakers=_speakers(a), diarizer=a.diarizer, problems=problems)
        _each([mic], lambda _: write_outputs(doc, result_folder(a.out, mic.parent, d), a.stem, a.to, offline=a.offline, data=d,
                                             audio=[(mic, ".mic.wav"), (Path(a.system), ".system.wav")]))
        if problems:
            sys.exit(1)
    elif a.cmd == "transcribe":
        _need_key(a.to, a.offline)
        _check_spoken(a.language)
        preflight(0, 1, a.to if a.offline else ())

        if (a.into or a.stem) and len(a.files) > 1:
            sys.exit("--into and --stem name one data folder and one file, so they work for one file at a time. Nothing was started.")
        check_name(a.stem, "stem")

        # With one output folder, two files of the same name (a Zoom folder per meeting, each with audio_only.m4a) would
        # write the same <stem>.transcript.txt: every file gets a stem of its own, decided for the whole batch up front.
        stems = dict(zip(a.files, unique_stems(a.files))) if a.out and not a.stem else {}

        def work(src):
            stem = a.stem or stems.get(str(src), src.stem)
            if stem != src.stem:
                print(f"note: {src} shares its name with another file; its files are called {stem}.*", file=sys.stderr)

            def refuse(folder):                            # now, and again on a numbered folder (_new_dir)
                problem = windows_path_problem(folder, a.to, a.offline)
                if problem:
                    raise RuntimeError(f"{problem}; nothing was transcribed")     # this file only: the batch goes on
            refuse(_data_path(a.into, stem))
            d = _data_dir(a.into, stem, refuse)
            doc = transcribe(src, d, a.language, a.track, a.granularity, notes=a.notes, speakers=_speakers(a), diarizer=a.diarizer)
            return write_outputs(doc, result_folder(a.out, src.parent, d), stem, a.to, offline=a.offline, data=d, audio=[(src, ".wav")])
        _each(a.files, work)
    elif a.cmd == "translate":
        _need_key(a.to, a.offline)

        def work(src):
            if src.is_dir() or (DATA / str(src)).is_dir():           # a stored meeting: its transcript.json and its cache
                folder = src if src.is_dir() else DATA / str(src)
                if not (folder / "transcript.json").exists():
                    raise RuntimeError(f"{folder} holds no transcript.json")
                doc = _load(folder)
                stem = re.sub(r"^\d{8}-\d{6}-?", "", folder.name) or "recording"
                return write_outputs(doc, result_folder(a.out, folder, folder), stem, a.to, transcript=False,
                                     offline=a.offline, source=a.source, data=folder)
            doc = parse_transcript(src.read_text(encoding="utf-8-sig"))
            if not doc["lines"]:
                raise RuntimeError("no text in the file")
            stem = src.stem[:-len(".transcript")] if src.stem.endswith(".transcript") else src.stem
            # A transcript in the output folder already (a meeting's own file) is translated beside itself, under its name;
            # any other file goes to this hour's folder, under a name of its own there (claim_stem, owner = the file).
            here = OUT is not None and not a.out and _within(src, OUT)
            return write_outputs(doc, src.parent if here else result_folder(a.out, src.parent), stem, a.to, transcript=False,
                                 protect=src, offline=a.offline, source=a.source, owner=str(src.resolve()))
        _each(a.files, work)


if __name__ == "__main__":
    main()
