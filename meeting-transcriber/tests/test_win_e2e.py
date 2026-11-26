"""tools/win_e2e.py: the end-to-end check of the Windows install (the CI job "Engine on Windows"). Its real run needs the
engine and Windows; what is tested here is that it imports with the standard library only, its pure parts (the checks
of a transcript, the environment an mt.py child gets, PATH without ffmpeg, the @@ events, the committed clip), that the
workflow gives each step more time than the tool's budget for it, and its steps against a stand-in for mt.py, which
answers as mt.py does (or wrongly, to see the step fail)."""
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOL = Path(__file__).parent.parent / "tools" / "win_e2e.py"
WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "meeting-transcriber.yml"
OPTIONAL = ["numpy", "faster_whisper", "ctranslate2", "onnxruntime", "sounddevice", "soundfile", "sherpa_onnx", "av",
            "static_ffmpeg", "anthropic"]


def load():
    spec = importlib.util.spec_from_file_location("win_e2e", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2e = load()


def run_tool(*args, timeout=120):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, encoding="utf-8",
                          errors="replace", timeout=timeout, stdin=subprocess.DEVNULL, env=quiet_env())


def quiet_env(**extra):
    """The tests' environment for the tool: without GITHUB_ACTIONS, so that the tool's FAIL lines in a failed test's
    message are not taken for annotations of this job."""
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_ACTIONS"}
    env.update(extra)
    return env


def test_the_file_compiles():
    py_compile.compile(str(TOOL), doraise=True)


def test_importing_loads_no_optional_package():
    code = (f"import importlib.util, json, sys; spec = importlib.util.spec_from_file_location('e', {str(TOOL)!r}); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"print(json.dumps(sorted(set(sys.modules) & set({OPTIONAL!r}))))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_the_sentence_words_that_are_checked():
    assert e2e.SENTENCE == " ".join(e2e.SENTENCES) and len(e2e.SENTENCES) == 2
    assert e2e.content_words(e2e.SENTENCES[0]) == ["hello", "test", "meeting", "transcriber", "windows", "reading",
                                                   "file", "cyrillic", "name"]
    assert e2e.content_words(e2e.SENTENCES[1]) == ["weather", "nice", "today", "coffee"]
    assert e2e.content_words("This is the one, the ONE.") == []      # short and stop words only: nothing to check


def test_the_committed_clip():
    data = e2e.CLIP.read_bytes()
    assert data[:4] == b"fLaC" and 50_000 < len(data) < 200_000
    assert hashlib.sha256(data).hexdigest() == e2e.CLIP_SHA256 == e2e.sha256_of(e2e.CLIP)


def test_speech_text_keeps_what_was_said_only():
    transcript = textwrap.dedent("""\
        # Recorded on a test runner: hello meeting windows
        # [00:00:03] Paused here for 1 s
        [00:00:00] [en] Hello, this is a test.
        [00:00:04] [en?] <Speaker 1> The weather is nice.
        [00:00:07] [en] \u200b<laughs> coffee
        """)
    assert e2e.speech_text(transcript) == "Hello, this is a test.\nThe weather is nice.\n<laughs> coffee"


def test_the_parts_of_a_transcript():
    transcript = ("# [00:00:06] chunk failed (RuntimeError: x)\n"
                  "[00:00:00] [en] Hello.\n"
                  "[00:00:02] [ru?] <Them 1> Привет.\n"
                  "\n"
                  "Hello without time or tag\n"
                  "#not a note\n")
    speech, notes, other = e2e.transcript_parts(transcript)
    assert speech == [("en", "Hello."), ("ru?", "Привет.")]
    assert notes == ["chunk failed (RuntimeError: x)"]
    assert other == ["Hello without time or tag", "#not a note"]
    assert all(e2e.BAD_NOTE.search(n) for n in ("chunk failed (x)", "speech not recognised for 4 s",
                                                 "No speech was found in встреча.flac"))
    assert not e2e.BAD_NOTE.search("2 speakers found")


def test_words_found_ignores_case_notes_and_parts_of_words():
    transcript = ("# hello meeting windows today\n"                    # a note is mt.py's text, never counted
                  "[00:00:00] [en] HELLO there, a TESTING of the Meeting.\n")
    assert e2e.words_found(e2e.SENTENCES[0], transcript) == ["hello", "meeting"]   # "testing" is not "test"
    assert e2e.words_found(e2e.SENTENCE, "# hello meeting\n") == []
    assert e2e.words_found(e2e.SENTENCES[1], "[00:00:01] [en] Whatever.\n") == []


def test_what_on_stderr_is_a_problem_or_the_network():
    stderr = ("\r1/2 chunkswarning: chunk 2 at 00:00:06 failed: RuntimeError: x\n\r2/2 chunks\n"
              "Traceback (most recent call last):\n"
              "warning: 'pause' does nothing now\n")
    assert e2e.stderr_problems(stderr) == ["1/2 chunkswarning: chunk 2 at 00:00:06 failed: RuntimeError: x",
                                           "Traceback (most recent call last):"]
    assert e2e.stderr_problems("loading model tiny\n\r1/1 chunks\n") == []
    for line in ("warning: ffmpeg could not be fetched: 503 Server Error: Service Unavailable for url: https://x",
                 "httpx.ConnectError: [Errno 11001] getaddrinfo failed",
                 "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded",
                 "huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file",
                 "urllib.error.HTTPError: HTTP Error 502: Bad Gateway"):
        assert e2e.infra_lines("x\n" + line + "\n") == [line], line
    assert e2e.infra_lines("Traceback (most recent call last):\nValueError: bad chunk\n") == []


def test_ffmpeg_names_follow_pathext_on_windows():
    assert e2e.ffmpeg_names(windows=False) == ("ffmpeg", "ffprobe")
    assert e2e.ffmpeg_names(windows=True, pathext=".EXE;.CMD") == ("ffmpeg.exe", "ffmpeg.cmd", "ffprobe.exe",
                                                                   "ffprobe.cmd")


def test_path_without_the_folders_that_hold_ffmpeg(tmp_path):
    a, b, c = (tmp_path / n for n in ("a", "b", "c"))
    for folder in (a, b, c):
        folder.mkdir()
    (b / "ffmpeg").write_text("")
    (c / "ffprobe").write_text("")
    value = os.pathsep.join([str(a), str(b), "", str(c)])
    kept, removed = e2e.ffmpeg_free_path(value, names=("ffmpeg", "ffprobe"))
    assert kept.split(os.pathsep) == [str(a), ""]
    assert removed == [str(b), str(c)]
    assert e2e.ffmpeg_free_path("", names=("ffmpeg",)) == ("", [])


def test_the_environment_of_an_mt_child(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / e2e.ffmpeg_names()[0]).write_text("")
    (tmp_path / "tools").mkdir()                                       # a folder without ffmpeg stays
    base = {"PATH": os.pathsep.join([str(tmp_path / "bin"), str(tmp_path / "tools")]), "PYTHONUTF8": "1",
            "pythonioencoding": "utf-8", "ANTHROPIC_API_KEY": "sk-ant-x", "MT_OUT": "/out", "MT_EVENTS": "1",
            "MT_CONTROL": "stdin", "MT_DEVICE": "cuda", "MT_ENGINE": "mlx", "MT_HOME": "/elsewhere", "KEEP": "yes"}
    env, removed = e2e.mt_env(base, tmp_path / "home", "tiny", {"MT_CONTROL": "stdin"})
    assert removed == [str(tmp_path / "bin")]
    assert env["PATH"] == str(tmp_path / "tools")
    assert env["MT_WHISPER_MODEL"] == "tiny" and env["MT_HOME"] == str(tmp_path / "home" / "data")
    assert env["MT_CONTROL"] == "stdin" and env["KEEP"] == "yes"
    for name in ("PYTHONUTF8", "pythonioencoding", "ANTHROPIC_API_KEY", "MT_OUT", "MT_EVENTS", "MT_DEVICE", "MT_ENGINE"):
        assert name not in env
    env, removed = e2e.mt_env(base, tmp_path / "home", keep_ffmpeg=True)
    assert removed == [] and env["PATH"] == base["PATH"]


def test_events_and_the_transcript_beside_the_input():
    text = "Downloading x -> y\n@@FILE\t1\t1\tC:\\a\\встреча.wav\n@@LINE\t{\"t\": 0}\n@@DONE\t0\n"
    assert e2e.parse_events(text) == [("FILE", ["1", "1", "C:\\a\\встреча.wav"]), ("LINE", ['{"t": 0}']),
                                      ("DONE", ["0"])]
    assert e2e.transcript_of(Path("w") / "встреча.flac") == Path("w") / "встреча.transcript.txt"
    assert e2e.same_path("a/./b", "a/b")


def test_a_powershell_literal():
    assert e2e.ps_quote("it's C:\\x") == "'it''s C:\\x'"


def test_the_clock_of_a_step():
    clock = e2e.Clock(100)
    assert 99 < clock.left() <= 100 and clock.left(30) == 30
    assert e2e.Clock(0).left() == 1.0                                 # a child always gets a second


def test_the_workflow_gives_each_step_more_time_than_its_budget():
    """Each step that runs win_e2e.py has a timeout-minutes of at least its budget plus the margin (so the tool's FAIL
    and RESULT lines come before the step is killed), and the job's timeout-minutes is above all of them together."""
    text = WORKFLOW.read_text(encoding="utf-8")
    job = text.split("\n  meeting-transcriber-engine-windows:\n", 1)[1]
    job_minutes = int(re.search(r"^    timeout-minutes: (\d+)$", job, re.M).group(1))
    seen, total = {}, 0
    for step in job.split("\n      - name: ")[1:]:
        minutes = re.search(r"^        timeout-minutes: (\d+)$", step, re.M)
        total += int(minutes.group(1)) if minutes else 0
        tool = re.search(r"tools\\win_e2e\.py (\w+)", step)
        if tool:
            assert minutes, f"the step that runs win_e2e.py {tool.group(1)} has no timeout-minutes"
            seen[tool.group(1)] = int(minutes.group(1))
        if "upload-artifact" in step:
            assert "if: always()" in step                               # also after a step that timed out
    assert set(seen) == set(e2e.STEP_BUDGET)
    for name, minutes in seen.items():
        assert minutes * 60 >= e2e.STEP_BUDGET[name] + e2e.STEP_MARGIN_S, name
    assert job_minutes >= total + 5                                    # checkout and setup-python have none of their own


def test_a_step_needs_the_work_folder():
    done = run_tool("transcribe")
    assert done.returncode == 2 and "needs the work folder" in done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows the speech step asks the speech synthesizer")
def test_the_speech_step_falls_back_to_the_committed_clip(tmp_path):
    done = run_tool("speech", tmp_path)
    assert done.returncode == 0, done.stdout
    assert "SPEECH: FALLBACK to the committed clip" in done.stdout and "RESULT speech: 2 passed, 0 failed" in done.stdout
    info = json.loads((tmp_path / "speech.json").read_text(encoding="utf-8"))
    assert info["source"] == "clip" and info["input"] == "встреча.flac" and info["sentences"] == list(e2e.SENTENCES)
    assert (tmp_path / "встреча.flac").read_bytes() == e2e.CLIP.read_bytes()


# A stand-in for mt.py: prepare says what mt.py says; transcribe writes <stem>.transcript.txt beside the input and names
# it on stdout (or, with MT_EVENTS=1, as @@ events); with MT_CONTROL=stdin it reads stdin as mt.py does, answers "pause"
# on stderr, waits for that line before it goes on (so its life does not depend on the machine's speed), stays 0.5 s
# after @@DONE (so a front end that closes stdin on @@DONE does so while it runs), and ends with 15 when stdin ends
# before the work is done, but not after it (as mt.py:3992-3995). FAKE_MODE makes it answer wrongly.
FAKE = r'''
import os, sys, threading, time
from pathlib import Path
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", newline="\n")
mode = os.environ.get("FAKE_MODE", "")
assert os.environ.get("MT_WHISPER_MODEL") == "tiny" and "PYTHONUTF8" not in os.environ
paused, finished = threading.Event(), threading.Event()
FIRST = "[00:00:00] [en] Hello, this is a test of the meeting transcriber.\n"
GOOD = FIRST + "[00:00:06] [en] The weather is nice today.\n"
TRANSCRIPTS = {
    "chunk-failed": FIRST + "# [00:00:06] chunk failed (RuntimeError: CTranslate2 crashed)\n",
    "wrong-language": FIRST.replace("[en]", "[ru?]") + "[00:00:06] [en] The weather is nice today.\n",
    "one-sentence": FIRST + "[00:00:06] [en] Thank you.\n",
    "no-speech": "# [00:00:10] No speech was found in встреча.flac\n",
    "untagged": GOOD + "the weather without time or tag\n",
}
TRACEBACK = "Traceback (most recent call last):\n  File \"mt.py\", line 1, in <module>\nOSError: PortAudio library not found\n"
STDERR = {"chunk-failed": "\r1/2 chunkswarning: chunk 2 at 00:00:06 failed: RuntimeError: CTranslate2 crashed\n\r2/2 chunks\n",
          "traceback": TRACEBACK}

def read():
    while True:
        data = os.read(0, 4096)
        if not data:
            if finished.is_set() and mode != "eof-15":
                return
            os._exit(15)
        if b"pause" in data:
            print("warning: 'pause' does nothing now: only a running recording can be paused", file=sys.stderr, flush=True)
            paused.set()

cmd = sys.argv[1]
if cmd == "--help":
    print("usage: mt.py {record,call,transcribe,translate,show,follow,prepare,rename,list,devices}")
elif cmd == "--version":
    print("meeting-transcriber " + Path(__file__).with_name("VERSION").read_text().strip())
elif cmd == "devices":
    if mode == "traceback":
        print(TRACEBACK, file=sys.stderr)
        sys.exit(1)
    print("")
elif cmd == "prepare":
    marker = Path(__file__).with_name("prepared-once")
    if mode == "flaky-network" and not marker.exists():
        marker.write_text("")
        print("warning: ffmpeg could not be fetched: HTTPSConnectionPool(host='github.com', port=443): Max retries "
              "exceeded", file=sys.stderr)
        sys.exit("ffmpeg and ffprobe are needed and could not be fetched. Connect to the internet and try again.")
    if mode == "broken":
        print(TRACEBACK + "ValueError: a bug", file=sys.stderr)
        sys.exit(1)
    print("ffmpeg: ok\nloading model tiny\nengine: ct2\nready")
elif cmd == "transcribe":
    src = Path(sys.argv[2])
    control = os.environ.get("MT_CONTROL") == "stdin"
    if control:
        threading.Thread(target=read, daemon=True).start()
    events = os.environ.get("MT_EVENTS") == "1"
    if events:
        print(f"@@FILE\t1\t1\t{src}", flush=True)
    if control:
        paused.wait(30)
    sys.stderr.write(STDERR.get(mode, "\r1/2 chunks\r2/2 chunks\n"))
    sys.stderr.flush()
    out = src.with_name(src.stem + ".transcript.txt")
    if mode != "no-file":
        out.write_text(TRANSCRIPTS.get(mode, GOOD), encoding="utf-8")
    finished.set()
    if events:
        print('@@LINE\t{"t": 0.0, "text": "Hello"}', flush=True)
        print(f"@@OUT\t{out}", flush=True)
        print("@@DONE\t0", flush=True)
    else:
        sys.stdout.write(f"{out}" + ("\r\n" if mode == "crlf" else "\n"))
        sys.stdout.flush()
    if control:
        time.sleep(0.5)
'''


@pytest.fixture
def fake(tmp_path):
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / "mt.py").write_text(FAKE, encoding="utf-8")
    (folder / "VERSION").write_text("9.9.9\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    shutil.copyfile(e2e.CLIP, work / "встреча.flac")
    (work / "speech.json").write_text(json.dumps({"sentence": e2e.SENTENCE, "sentences": list(e2e.SENTENCES),
                                                  "source": "clip", "reason": "test", "input": "встреча.flac"}),
                                      encoding="utf-8")
    return folder / "mt.py", work


def run_step(step, mt, work, mode="", *extra):
    env = quiet_env(FAKE_MODE=mode, PYTHONUTF8="1")       # PYTHONUTF8 is set here to see that the child loses it
    return subprocess.run([sys.executable, str(TOOL), step, str(work), "--mt", str(mt), "--system-ffmpeg", *extra],
                          env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                          stdin=subprocess.DEVNULL)


def test_the_transcribe_step(fake):
    mt, work = fake
    done = run_step("transcribe", mt, work)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "RESULT transcribe: 11 passed, 0 failed" in done.stdout, done.stdout
    assert "PASS: stdout names the transcript, Cyrillic intact" in done.stdout
    assert "PASS: 2 speech line(s), every one tagged en or en?: en" in done.stdout
    assert "PASS: sentence 1: 4 of its 9 words are in the transcript: hello, test, meeting, transcriber" in done.stdout
    assert "PASS: sentence 2: 3 of its 4 words are in the transcript: weather, nice, today" in done.stdout
    assert "PASS: mt.py's stderr (a pipe): no traceback and no failed chunk" in done.stdout


@pytest.mark.parametrize("mode, failures", [
    ("no-file", ["FAIL: no transcript at"]),
    ("crlf", ["FAIL: stdout (a pipe) has \\r\\n line ends"]),
    # one of two chunks failed inside the engine: mt.py goes on and exits 0 (mt.py:1740-1743)
    ("chunk-failed", ["FAIL: the transcript says speech was lost: chunk failed (RuntimeError: CTranslate2 crashed)",
                      "FAIL: mt.py's stderr (a pipe): 1/2 chunkswarning: chunk 2 at 00:00:06 failed",
                      "FAIL: sentence 2: 0 of its words"]),
    ("wrong-language", ["FAIL: 2 speech line(s) with the tags ['en', 'ru?']: every one must be en or en?"]),
    ("one-sentence", ["FAIL: sentence 2: 0 of its words (weather, nice, today, coffee) are in the transcript"]),
    ("no-speech", ["FAIL: the transcript says speech was lost: No speech was found", "FAIL: 0 speech line(s)",
                   "FAIL: sentence 1: 0 of its words"]),
    ("untagged", ["FAIL: 1 line(s) are neither speech nor a note: ['the weather without time or tag']"]),
    ("traceback", ["FAIL: mt.py's stderr (a pipe): Traceback (most recent call last):"]),
])
def test_the_transcribe_step_fails_when_mt_answers_wrongly(fake, mode, failures):
    mt, work = fake
    done = run_step("transcribe", mt, work, mode)
    assert done.returncode == 1, done.stdout
    for failure in failures:
        assert failure in done.stdout, done.stdout


def test_the_control_step_keeps_stdin_open_then_closes_it_on_done(fake):
    mt, work = fake
    done = run_step("control", mt, work)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS: run 1: exit code 0, ended by itself with stdin still open" in done.stdout
    assert "PASS: the line pause on stdin was read and answered on stderr" in done.stdout
    assert "PASS: the last event is @@DONE 0" in done.stdout
    assert "INFO: run 2: mt.py still ran when stdin was closed on @@DONE" in done.stdout
    assert "PASS: run 2: stdin closed on @@DONE, exit code 0 (not 15)" in done.stdout
    assert "RESULT control: 18 passed, 0 failed" in done.stdout, done.stdout
    assert (work / "control" / "встреча.transcript.txt").is_file()
    assert (work / "control-close" / "встреча.transcript.txt").is_file()


def test_the_control_step_fails_when_the_end_of_stdin_after_done_cancels(fake):
    mt, work = fake
    done = run_step("control", mt, work, "eof-15")
    assert done.returncode == 1, done.stdout
    assert "PASS: run 1: exit code 0" in done.stdout
    assert "FAIL: run 2: exit code 15 (SIGTERM's: the end of stdin after @@DONE was taken as cancel" in done.stdout


@pytest.mark.parametrize("mode, failure", [("no-file", "FAIL: no transcript at"),
                                           ("chunk-failed", "FAIL: run 1: mt.py's stderr: 1/2 chunkswarning: chunk 2")])
def test_the_control_step_fails_when_mt_answers_wrongly(fake, mode, failure):
    mt, work = fake
    done = run_step("control", mt, work, mode)
    assert done.returncode == 1 and failure in done.stdout, done.stdout


def test_the_prepare_step(fake):
    mt, work = fake
    done = run_step("prepare", mt, work)
    assert done.returncode == 0, done.stdout
    assert "RESULT prepare: 5 passed, 0 failed" in done.stdout and "INFRA:" not in done.stdout


def test_the_prepare_step_runs_once_more_after_a_network_failure(fake):
    mt, work = fake
    done = run_step("prepare", mt, work, "flaky-network", "--retry-wait", "0")
    assert done.returncode == 0, done.stdout
    assert "INFRA: warning: ffmpeg could not be fetched" in done.stdout
    assert "INFO: mt.py prepare (attempt 2)" in done.stdout and "RESULT prepare: 5 passed, 0 failed" in done.stdout


def test_the_prepare_step_does_not_run_again_after_a_bug(fake):
    mt, work = fake
    done = run_step("prepare", mt, work, "broken", "--retry-wait", "0")
    assert done.returncode == 1, done.stdout
    assert "FAIL: mt.py prepare: exit code 1" in done.stdout and "FAIL: mt.py prepare's stderr: Traceback" in done.stdout
    assert "INFRA:" not in done.stdout and "attempt 2" not in done.stdout


def test_the_cli_step(fake):
    mt, work = fake
    done = run_step("cli", mt, work)
    assert done.returncode == 0, done.stdout
    assert "PASS: mt.py --version: meeting-transcriber 9.9.9" in done.stdout
    done = run_step("cli", mt, work, "traceback")
    assert done.returncode == 1
    assert "FAIL: mt.py devices: exit code 1" in done.stdout and "FAIL: mt.py devices: a traceback" in done.stdout


def test_a_step_without_speech_json_fails(fake):
    mt, work = fake
    (work / "speech.json").unlink()
    done = run_step("transcribe", mt, work)
    assert done.returncode == 1 and "FAIL: no speech.json" in done.stdout
