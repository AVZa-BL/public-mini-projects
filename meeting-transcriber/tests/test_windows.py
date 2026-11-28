"""mt.py on Windows (W2 and W3 of the plan): the control channel on stdin (MT_CONTROL=stdin), UTF-8 output, the Windows
name rules, keeping the PC awake, child processes without stdin or console, the memory check, the os.replace retry, the
WASAPI device choice, the CUDA -> CPU fallback and the NVIDIA DLL preload. Nearly everything here runs on Linux and macOS
as well (Windows is faked where a test needs it) and for real on the Windows CI job; what needs a real Windows (a console
of its own, cp1252 pipes, the report on a read left pending) skips elsewhere. No test sends a signal to pytest's process:
on Windows os.kill(pid, SIGINT or SIGTERM) is TerminateProcess and would end pytest."""
import importlib.util
import io
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import mt
from test_mt import SecondEngine, StampedStream, _content_vad, _fake_recording, _seconds, _speech

MT = Path(__file__).parent.parent / "mt.py"
QUIET = {verb: (lambda: None) for verb in (*mt.CONTROL_VERBS, "eof")}


@pytest.fixture
def control(monkeypatch):
    """A fresh control channel (mt.CONTROL for this test) reading a pipe, with MT_CONTROL=stdin. What the channel does
    when nothing listens (SIGTERM's and SIGINT's default) is recorded instead of done: either would end pytest."""
    import _thread
    r, w = os.pipe()
    channel = mt.Control()
    monkeypatch.setattr(mt, "CONTROL", channel)
    monkeypatch.setenv("MT_CONTROL", "stdin")
    terminated, interrupted = [], []
    monkeypatch.setattr(mt.Control, "terminate", staticmethod(lambda: terminated.append(True)))
    monkeypatch.setattr(_thread, "interrupt_main", lambda *a: interrupted.append(True))
    channel.start(fd=r)
    closed = []

    def send(text):
        os.write(w, text.encode())

    def close():
        if not closed:
            closed.append(True)
            os.close(w)
    yield SimpleNamespace(send=send, close=close, channel=channel, terminated=terminated, interrupted=interrupted)
    channel.listen(QUIET)
    close()
    channel.thread.join(5)
    os.close(r)


# ---------- the control channel: its lines ----------
def test_after_done_is_printed_an_end_of_stdin_or_a_late_cancel_changes_nothing(control, capsys):
    """A front end that closes stdin as soon as it reads @@DONE: the end of stdin must not end mt.py as SIGTERM (exit 15)
    in the moment between that line and main()'s finally. ev("DONE") quiets the channel before it prints the line."""
    seen = []
    control.channel.listen({"eof": lambda: seen.append("eof"), "cancel": lambda: seen.append("cancel")})
    mt.ev("DONE", 0)
    assert capsys.readouterr().out == "@@DONE\t0\n"
    control.send("cancel\n")
    control.close()
    control.channel.thread.join(5)
    assert not control.channel.thread.is_alive()
    assert seen == [] and control.terminated == [] and control.interrupted == []


def test_control_lines_are_trimmed_and_case_blind_unknown_ones_are_warned_and_a_last_line_needs_no_line_end(capsys):
    got = []
    channel = mt.Control()
    channel.listen({verb: (lambda verb=verb: got.append(verb)) for verb in (*mt.CONTROL_VERBS, "eof")})
    r, w = os.pipe()
    channel.start(fd=r)
    os.write(w, b"\xef\xbb\xbf  Pause \r\nRESUME\n\nbogus\nextend\ncanc")      # a UTF-8 BOM first, as a .NET writer sends
    time.sleep(0.2)
    os.write(w, b"el\nst")                                   # a line that arrives in two pieces is one line
    time.sleep(0.2)
    os.write(w, b"op")
    os.close(w)
    channel.thread.join(5)
    os.close(r)
    assert got == ["pause", "resume", "extend", "cancel", "stop", "eof"]
    assert "unknown control command 'bogus' ignored" in capsys.readouterr().err


@pytest.mark.parametrize("chunks, verbs", [
    ([b"pau", b"se\r\nstop\r\n", b""], ["pause", "stop", "eof"]),
    ([b"pau", b"se\r\nstop\r", b"\nexte", b"nd", b""], ["pause", "stop", "extend", "eof"])])   # \r | \n, no last line end
def test_the_reader_makes_lines_of_what_stdin_reader_returns_and_its_empty_read_is_the_end(monkeypatch, chunks, verbs):
    """Control._read only joins and splits what mt.stdin_reader(fd) returns (the Windows waiting is all in there): a line
    cut anywhere, even between its \\r and \\n, is one line; a last line without a line end still counts; b"" is the end,
    dispatched once as eof, and nothing is read after it. A faked reader, so the same on every platform."""
    got, asked, left = [], [], iter(chunks)

    def reader(fd):
        asked.append(fd)

        def read():
            data = next(left, None)
            assert data is not None, "read again after the end"
            return data
        return read
    monkeypatch.setattr(mt, "stdin_reader", reader)
    channel = mt.Control()
    channel.listen({verb: (lambda verb=verb: got.append(verb)) for verb in (*mt.CONTROL_VERBS, "eof")})
    channel._read(7)                                         # in this thread: it returns at the end
    assert got == verbs and asked == [7] and next(left, "spent") == "spent"


def within(seconds, call):
    """call() in a daemon thread, failing the test if it has not returned after `seconds`: a read that waits where it must
    not would otherwise hang until --timeout, which on Windows (no SIGALRM: pytest-timeout's thread method) ends the
    whole pytest run."""
    box = []

    def run():
        try:
            box.append((True, call()))
        except BaseException as e:                           # handed to the test's thread
            box.append((False, e))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    assert box, f"{call} did not return within {seconds} s"
    returned, value = box[0]
    if not returned:
        raise value
    return value


def test_stdin_reader_reads_a_real_pipe_as_it_comes_and_returns_b_empty_at_its_end():
    """A real os.pipe() read in this process: what was written comes back, and the end of the pipe (writer closed, the
    rest read) is b"" and stays b"". Outside Windows a plain os.read(fd, 4096): a 10000-byte write comes back in reads of
    4096, 4096 and 1808 bytes. os.pipe() is an anonymous pipe on Windows too, so there this is the PeekNamedPipe path: the
    bytes in the pipe are read, bytes written just before the writer closed are still read, then ERROR_BROKEN_PIPE is the
    end."""
    r, w = os.pipe()
    try:
        read = mt.stdin_reader(r)
        os.write(w, b"pause\n")
        assert within(10, read) == b"pause\n"
        if sys.platform != "win32":                          # (on Windows a write this big could wait for the reader)
            os.write(w, b"x" * 10000)
            assert [len(within(10, read)) for _ in range(3)] == [4096, 4096, 1808], "os.read(fd, 4096), as before"
        os.write(w, b"stop")
        os.close(w)
        w = None
        assert [within(10, read) for _ in range(3)] == [b"stop", b"", b""]
    finally:
        os.close(r)
        if w is not None:
            os.close(w)


class Through:
    """A module as mt.py sees it (mt.os, mt.time), with some of its functions replaced: nothing else in pytest's process
    sees the fakes."""
    def __init__(self, module, **replaced):
        self.__dict__.update(replaced)
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)


def failed(code):
    """In `peeks`: a PeekNamedPipe that fails with Windows error `code`."""
    return SimpleNamespace(error=code)


def keys(text):
    """Console input records (EventType, key down, virtual key, character) as typed: a key-down and a key-up for each
    character; \\r is Enter (VK_RETURN)."""
    return [(1, down, 0x0D if char == "\r" else 0x5A if char == "\x1a" else ord(char.upper()), char)
            for char in text for down in (True, False)]


CTRL_ENTER = [(1, True, 0x0D, "\n"), (1, False, 0x0D, "\n")]        # VK_RETURN, but the character \n
FOCUS = (0x10, False, 0, "\0")                                       # FOCUS_EVENT


@pytest.fixture
def windows(monkeypatch):
    """stdin_reader's Windows side faked, so that its rules run on every platform. kernel32, msvcrt and the last error as
    ctypes gives them; stdin a pipe (kind 3: PeekNamedPipe answers from `peeks`, a number of bytes there or failed(code);
    os.read takes from `piped`) or a console (kind 2: its input buffer `typed`, from which os.read takes a line as
    conhost's cooked read does, \\n skipped; one with no line end in it would wait, so it fails the test). time.sleep is
    recorded in `slept`; `at_sleep` {n: callable} runs a callable at the n-th sleep (what arrives while the reader waits).
    Through mt.os and mt.time only."""
    import ctypes
    w = SimpleNamespace(kind=3, peeks=[], peeked=0, piped=bytearray(), read_sizes=[], typed=[], dropped=0, broken=False,
                        slept=[], at_sleep={}, error=[0])

    def get_file_type(handle):
        return w.kind

    def get_console_mode(handle, mode):
        return int(w.kind == 2)

    def peek_named_pipe(handle, buffer, size, read, available, left):
        assert w.peeks, "PeekNamedPipe called once more than the test expects"
        w.peeked += 1
        answer = w.peeks.pop(0)
        if isinstance(answer, SimpleNamespace):
            w.error[0] = answer.error
            return 0
        assert len(w.piped) >= answer, "the fake pipe holds fewer bytes than PeekNamedPipe says"
        available._obj.value = answer
        return 1

    def number_of_events(handle, count):
        if w.broken:
            w.error[0] = 6
            return 0
        count._obj.value = len(w.typed)
        return 1

    def peek_console(handle, records, length, got):
        if w.broken:
            w.error[0] = 6
            return 0
        for record, (kind, down, vk, char) in zip(records[:length], w.typed):
            record.type, record.event.down, record.event.key, record.event.char = kind, down, vk, char
        got._obj.value = min(length, len(w.typed))
        return 1

    def read_console(handle, records, length, got):
        del w.typed[:length]
        w.dropped += length
        got._obj.value = length
        return 1

    def read(fd, size):
        w.read_sizes.append(size)
        if w.kind == 3:
            data = bytes(w.piped[:size])
            del w.piped[:size]
            return data
        end = next((k for k, (kind, down, _, char) in enumerate(w.typed) if kind == 1 and down and char == "\r"), None)
        assert end is not None, f"a console read started with no line end in the buffer would wait: {w.typed}"
        line = "".join(char for kind, down, _, char in w.typed[:end] if kind == 1 and down and char not in "\0\n")
        del w.typed[:end + 1]
        return b"" if line.startswith("\x1a") else (line + "\r\n").encode()[:size]

    def sleep(seconds):
        w.slept.append(seconds)
        assert len(w.slept) < 5000, "the reader waits for ever"
        w.at_sleep.pop(len(w.slept), lambda: None)()
    k32 = SimpleNamespace(GetFileType=get_file_type, GetConsoleMode=get_console_mode, PeekNamedPipe=peek_named_pipe,
                          GetNumberOfConsoleInputEvents=number_of_events, PeekConsoleInputW=peek_console,
                          ReadConsoleInputW=read_console)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: 4242))
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: k32, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: w.error[0], raising=False)
    monkeypatch.setattr(mt, "os", Through(os, read=read))
    monkeypatch.setattr(mt, "time", Through(time, sleep=sleep))
    return w


def test_on_windows_a_stdin_pipe_is_read_only_for_what_is_in_it_and_ends_on_the_errors_that_mean_its_end(windows):
    windows.peeks, windows.piped = [0, 0, 6, failed(109)], bytearray(b"pause\n")
    read = mt.stdin_reader(0)
    assert read() == b"pause\n" and windows.read_sizes == [6] and windows.slept == [0.05, 0.05]
    assert read() == b"" and windows.peeks == [], "ERROR_BROKEN_PIPE: the writer closed it, all was read"
    for code in (109, 230, 232, 233, 6):         # broken, bad pipe (disconnected), closing, not connected, bad handle
        windows.peeks = [0, failed(code)]
        assert mt.stdin_reader(0)() == b"" and windows.peeks == [], code


def test_on_windows_other_pipe_errors_are_not_the_end_which_would_cancel_the_job(windows, capsys):
    """An error that is not one of the pipe's ends (here ERROR_NOT_ENOUGH_MEMORY, which ReactOS's PeekNamedPipe gives
    when its allocation fails) is looked at again ever more slowly, up to 1 s, said once, and never ends the channel:
    the end of stdin cancels a transcription."""
    windows.peeks, windows.piped = [0, *[failed(8)] * 40, 5], bytearray(b"stop\n")
    read = mt.stdin_reader(0)
    assert read() == b"stop\n"
    assert windows.slept == [0.05, *(min(0.05 * k, 1.0) for k in range(1, 41))]
    windows.peeks, windows.piped, windows.slept = [failed(8), failed(8), 7, failed(109)], bytearray(b"cancel\n"), []
    assert read() == b"cancel\n" and windows.slept == [0.05, 0.1], "a peek that works starts the count afresh"
    assert read() == b""
    err = capsys.readouterr().err
    assert err.count("warning: MT_CONTROL=stdin: Windows error 8 looking at the stdin pipe; still listening") == 1, err


def test_on_windows_a_pipe_that_cannot_be_peeked_at_is_read_plainly_and_an_empty_one_at_its_end_is_the_end(windows):
    """GetFileType calls a socket a pipe as well; PeekNamedPipe fails on it (here ERROR_INVALID_FUNCTION). When not one
    peek worked, stdin is read as before: a plain os.read(fd, 4096), with no peek after it. A first peek that fails with
    one of the pipe's ends is the end, not a plain read."""
    windows.peeks, windows.piped = [failed(1)], bytearray(b"stop\n")
    read = mt.stdin_reader(0)
    assert read() == b"stop\n" and read() == b"" and windows.read_sizes == [4096, 4096] and windows.peeked == 1
    windows.peeks, windows.read_sizes = [failed(109)], []
    assert mt.stdin_reader(0)() == b"" and windows.read_sizes == []


def test_on_windows_a_console_is_read_only_once_a_line_end_is_in_its_input_buffer(windows):
    """A focus event alone is dropped; a Ctrl+Enter (VK_RETURN with \\n, which conhost's cooked read skips) starts no read;
    the \\r of Enter does, at the next look (0.1 s); a line of Ctrl+Z comes back empty and is not the end."""
    windows.kind, windows.typed = 2, [FOCUS]
    windows.at_sleep = {1: lambda: windows.typed.extend(CTRL_ENTER + keys("st")),
                        2: lambda: windows.typed.extend(keys("op\r"))}
    read = mt.stdin_reader(0)
    assert read() == b"stop\r\n" and windows.slept == [0.1, 0.1] and windows.dropped == 1
    assert windows.typed == keys("\r")[1:], "the read ends at Enter's key-down: its key-up stays, for the next read"
    windows.typed, windows.slept = keys("\x1a\r") + keys("pause\r"), []
    assert read() == b"pause\r\n" and windows.slept == [0.1] and windows.read_sizes == [4096] * 3


def test_on_windows_a_console_with_4096_keys_and_no_line_end_among_them_is_read_all_the_same(windows):
    """Only the first 4096 input records are looked at: a line end beyond them (2100 characters typed or pasted before
    Enter) must still be read, at once, not waited for while the buffer only grows."""
    windows.kind, windows.typed = 2, keys("x" * 2100 + "\r")
    assert mt.stdin_reader(0)() == b"x" * 2100 + b"\r\n" and windows.slept == []


def test_on_windows_a_console_that_cannot_be_looked_at_for_100_s_is_the_end_and_a_shorter_spell_is_not(windows):
    windows.kind, windows.broken = 2, True
    windows.at_sleep = {500: lambda: (setattr(windows, "broken", False), windows.typed.extend(keys("stop\r")))}
    read = mt.stdin_reader(0)
    assert read() == b"stop\r\n" and len(windows.slept) == 500
    windows.broken, windows.slept = True, []
    assert read() == b"" and len(windows.slept) == 999, "1000 looks 0.1 s apart, as the reader before gave up"


def test_with_nothing_listening_stop_is_sigints_effect_and_cancel_or_the_end_is_sigterms(monkeypatch, capsys):
    import _thread
    calls = []
    monkeypatch.setattr(mt.Control, "terminate", staticmethod(lambda: calls.append("terminate")))
    monkeypatch.setattr(_thread, "interrupt_main", lambda *a: calls.append("KeyboardInterrupt"))
    channel = mt.Control()
    for verb in ("stop", "cancel", "eof", "pause", "extend"):
        channel.dispatch(verb)
    assert calls == ["KeyboardInterrupt", "terminate", "terminate"]
    err = capsys.readouterr().err
    assert "'pause' does nothing now" in err and "'extend' does nothing now" in err, "as SIGUSR1/SIGUSR2 are ignored then"
    assert ("cancel on stdin (MT_CONTROL=stdin): mt.py ends at once, as on SIGTERM\n" in err
            and "stdin ended (MT_CONTROL=stdin): mt.py ends at once" in err), "the cause is said before the end"


def test_a_real_stdin_pipe_is_read_by_a_separate_process(tmp_path):
    script = ("import sys, time; sys.path.insert(0, sys.argv[1]); import mt; done = []\n"
              "mt.CONTROL.listen({'stop': lambda: print('STOP', flush=True), "
              "'eof': lambda: (print('EOF', flush=True), done.append(1))})\n"
              "mt.CONTROL.start()\n"
              "for _ in range(400):\n"
              "    if done: break\n"
              "    time.sleep(0.05)\n")
    done = subprocess.run([sys.executable, "-c", script, str(MT.parent)], input=b"Stop\nwhat\n", capture_output=True,
                          timeout=60)
    assert done.returncode == 0 and done.stdout.decode().split() == ["STOP", "EOF"], done
    assert b"unknown control command 'what' ignored" in done.stderr


def test_cancel_with_nothing_listening_ends_the_process_as_sigterm_does():
    """transcribe, call and translate have no SIGTERM handler: the signal ends them at once (exit status -15 on POSIX;
    on Windows os.kill(pid, SIGTERM) is TerminateProcess with exit code 15). The cancel line does the same."""
    script = ("import sys, time; sys.path.insert(0, sys.argv[1]); import mt; mt.CONTROL.start()\n"
              "time.sleep(30); print('NOT ENDED')\n")
    for given, cause in ((b"cancel\n", b"cancel on stdin"), (b"", b"stdin ended")):   # the line, and the end of stdin
        done = subprocess.run([sys.executable, "-c", script, str(MT.parent)], input=given, capture_output=True, timeout=60)
        assert done.returncode == (15 if sys.platform == "win32" else -signal.SIGTERM), done
        assert b"NOT ENDED" not in done.stdout and cause + b" (MT_CONTROL=stdin): mt.py ends at once" in done.stderr


# ---------- the control channel: no read left pending on stdin (the windows-latest hang) ----------
# Run 45796780693, job "Engine on Windows": mt.py transcribe with MT_CONTROL=stdin and a stdin pipe kept open printed
# @@FILE and @@FOLDER and never decoded (killed after 150 s); without MT_CONTROL the same command took 3.7 s. The reader
# thread sat in os.read(0, 4096), a synchronous ReadFile pending on the pipe. The tests below put the main thread's calls
# next to a waiting reader in a real child process, each wait with a deadline, so that a hang fails one test (or is
# reported) instead of running into --timeout, which on Windows ends the whole pytest run.
PROBE_CALLS = ("subprocess.run(child_options)", "os.fstat(0)", "os.isatty(0)", "sys.stdin.isatty()",
               "subprocess.run(inherited_stdin)")

PIPE_CHILD = r'''
import os, subprocess, sys, threading, time
sys.path.insert(0, sys.argv[1])
import mt
reader, calls = sys.argv[2], sys.argv[3].split(",")
ended = threading.Event()
if reader == "mt":                              # mt.py's control channel, as main() starts it with MT_CONTROL=stdin
    def handler(verb):
        def say():
            print("got", verb, flush=True)
            if verb == "eof":
                ended.set()
        return say
    mt.CONTROL.listen({verb: handler(verb) for verb in (*mt.CONTROL_VERBS, "eof")})
    mt.CONTROL.start()
else:                                           # the reader before the fix: os.read left pending on stdin
    def read():
        while os.read(0, 4096):
            pass
        ended.set()
    threading.Thread(target=read, daemon=True).start()
time.sleep(0.5)                                 # the reader waits on the pipe now; nothing has been written to it
print("ready", flush=True)
CALLS = {
    # as mt.py starts ffprobe and ffmpeg: stdin NUL, its own pipes for stdout and stderr, on Windows no console window
    "subprocess.run(child_options)": lambda: subprocess.run([sys.executable, "-c", "pass"], capture_output=True,
                                                            timeout=30, **mt.child_options()),
    "os.fstat(0)": lambda: os.fstat(0),         # on Windows GetFileType of the pipe
    "os.isatty(0)": lambda: os.isatty(0),
    "sys.stdin.isatty()": lambda: sys.stdin.isatty(),
    # a child that inherits the stdin pipe (mt.py starts none; a library might): its C runtime asks GetFileType of it
    "subprocess.run(inherited_stdin)": lambda: subprocess.run([sys.executable, "-c", "pass"], capture_output=True,
                                                              timeout=30),
}
for name in calls:
    CALLS[name]()
    print("ok", name, flush=True)
print("main done", flush=True)
sys.exit(0 if ended.wait(60) else 3)
'''


class Child:
    """A Python child (`script`, given mt.py's folder and `args`) whose stdout lines a test waits for, every wait with
    its own deadline. stdout is read raw (os.read) by a daemon thread, so no buffered reader's lock is left held should
    the child never end; stderr goes to a file, quoted in failures; UTF-8 both ways (PYTHONIOENCODING=utf-8 there)."""

    def __init__(self, script, args, tmp_path, name, stdin=subprocess.PIPE, **popen):
        path = tmp_path / f"{name}.py"
        path.write_text(script, encoding="utf-8")
        self.err_path = tmp_path / f"{name}.stderr.txt"
        self.err = open(self.err_path, "wb")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", MT_HOME=str(tmp_path / "data"))
        env.pop("MT_CONTROL", None)
        try:
            self.proc = subprocess.Popen([sys.executable, str(path), str(MT.parent), *args], stdin=stdin,
                                         stdout=subprocess.PIPE, stderr=self.err, env=env, bufsize=0, **popen)
        except BaseException:
            self.err.close()
            raise
        self.lines, self.seen, self.ended = queue.Queue(), [], False
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()

    def _pump(self):
        pending, fd = b"", self.proc.stdout.fileno()
        while True:
            try:
                data = os.read(fd, 4096)
            except OSError:
                data = b""
            if not data:
                break
            *done, pending = (pending + data).split(b"\n")
            for raw in done:
                self.lines.put(raw.decode("utf-8", "replace").rstrip("\r"))
        if pending:
            self.lines.put(pending.decode("utf-8", "replace"))
        self.lines.put(None)                                 # the end of the child's stdout
        self.proc.stdout.close()

    def until(self, wanted, seconds):
        """Reads lines into `seen` until the line `wanted` (True), the end of stdout or `seconds` from now (False)."""
        deadline = time.monotonic() + seconds
        while not self.ended:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return False
            if line is None:
                self.ended = True
            else:
                self.seen.append(line)
                if line == wanted:
                    return True
        return False

    def send(self, text):
        self.proc.stdin.write(text.encode("utf-8"))          # unbuffered (bufsize=0): written at once

    def close_stdin(self):
        if self.proc.stdin is not None and not self.proc.stdin.closed:
            self.proc.stdin.close()

    def stderr(self):
        return self.err_path.read_text(encoding="utf-8", errors="replace")[-1500:]

    def end(self):
        """Kills the child if it still runs and closes its stdin (which also ends a read pending on it), then waits."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.close_stdin()
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            pass
        self.pump.join(10)
        self.err.close()


def test_with_the_reader_waiting_on_an_open_stdin_pipe_the_main_thread_starts_children_and_asks_about_stdin(tmp_path):
    """The hang's situation, in a real process on every platform (on Windows it is the proof): mt.CONTROL waits on a stdin
    pipe that stays open with nothing in it, and meanwhile the main thread starts a child as mt.py starts ffmpeg
    (child_options), asks os.fstat, os.isatty and sys.stdin.isatty about stdin, and starts a child that inherits stdin.
    Each comes back within the deadline; after that the channel still reads a line (pause) and the end of the pipe
    (eof), and the process exits 0."""
    child = Child(PIPE_CHILD, ["mt", ",".join(PROBE_CALLS)], tmp_path, "pipe-mt")
    try:
        if not child.until("main done", 90):
            done = [line for line in child.seen if line.startswith("ok ")]
            hung = PROBE_CALLS[len(done)] if "ready" in child.seen and len(done) < len(PROBE_CALLS) else "the start"
            pytest.fail(f"the main thread did not get past {hung} with the reader waiting on stdin (the last ok line: "
                        f"{done[-1] if done else 'none'}; stdout {child.seen}; stderr: {child.stderr()})")
        assert child.seen == ["ready", *(f"ok {name}" for name in PROBE_CALLS), "main done"], child.seen
        child.send("pause\n")
        assert child.until("got pause", 30), (child.seen, child.stderr())
        child.close_stdin()
        assert child.until("got eof", 30), (child.seen, child.stderr())
        assert child.proc.wait(30) == 0, child.stderr()
    finally:
        child.end()


@pytest.mark.skipif(sys.platform != "win32", reason="a root cause on Windows: elsewhere a pending read blocks nothing else")
def test_windows_report_which_main_thread_calls_wait_behind_a_read_left_pending_on_stdin(tmp_path):
    """A report, never a failure: the reader before the fix (a daemon thread in os.read(0, 4096), a synchronous ReadFile
    pending on the stdin pipe) and, in a fresh child each, one call of the test above, given 10 s. A call that has not
    come back by then is given 5 s more after stdin is closed (the pending read then ends with ERROR_BROKEN_PIPE): one that
    comes back at once was waiting for that read. The result is a warning, so it is in pytest's warnings summary in the CI
    log: whether the hypothesis holds (a pending synchronous read holds the file object's I/O lock) and which call hangs."""
    report, begun = [], time.monotonic()
    for k, name in enumerate(PROBE_CALLS):
        if time.monotonic() - begun > 150:                   # (a child that cannot be killed takes up to 70 s)
            report.append(f"{name}: not tried, the report's 150 s were used up")
            continue
        try:
            child = Child(PIPE_CHILD, ["plain", name], tmp_path, f"pipe-plain-{k}")
        except OSError as e:
            report.append(f"{name}: the child could not be started ({e})")
            continue
        try:
            if not child.until("ready", 30):
                report.append(f"{name}: the child did not get ready (exit {child.proc.poll()}, stderr {child.stderr()!r})")
                continue
            started = time.monotonic()
            if child.until(f"ok {name}", 10):
                report.append(f"{name}: came back in {time.monotonic() - started:.2f} s")
            elif child.ended:
                report.append(f"{name}: the child ended without it (exit {child.proc.poll()}, stderr {child.stderr()!r})")
            else:
                child.close_stdin()
                released = time.monotonic()
                if child.until(f"ok {name}", 5):
                    report.append(f"{name}: HUNG 10 s while the read was pending, came back "
                                  f"{time.monotonic() - released:.2f} s after stdin was closed")
                else:
                    report.append(f"{name}: HUNG 10 s while the read was pending, and 5 s more after stdin was closed")
        finally:
            child.end()
    warnings.warn(f"Windows report, a plain read pending on a stdin pipe in another thread (Python "
                  f"{sys.version.split()[0]}): " + "; ".join(report))


CONSOLE_CHILD = r'''
import ctypes, msvcrt, os, subprocess, sys, threading, time
from ctypes import wintypes
sys.path.insert(0, sys.argv[1])
import mt
mode = sys.argv[2]                              # "mt": mt.Control reads the console; "plain": the reader before the fix


def say(line):
    print(line, flush=True)


def skip(why):
    say("skip " + why)
    say("main done")
    sys.exit(0)


def until(check, seconds):
    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


class KeyEventRecord(ctypes.Structure):         # KEY_EVENT_RECORD
    _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD), ("wVirtualKeyCode", wintypes.WORD),
                ("wVirtualScanCode", wintypes.WORD), ("UnicodeChar", wintypes.WCHAR),
                ("dwControlKeyState", wintypes.DWORD)]


class FocusEventRecord(ctypes.Structure):       # FOCUS_EVENT_RECORD
    _fields_ = [("bSetFocus", wintypes.BOOL)]


class Event(ctypes.Union):                      # (MOUSE_EVENT_RECORD, the other 16-byte member, is not needed here)
    _fields_ = [("KeyEvent", KeyEventRecord), ("FocusEvent", FocusEventRecord)]


class InputRecord(ctypes.Structure):            # INPUT_RECORD: a WORD, two bytes of padding, the union at offset 4
    _fields_ = [("EventType", wintypes.WORD), ("Event", Event)]


layout = (ctypes.sizeof(KeyEventRecord), ctypes.sizeof(Event), ctypes.sizeof(InputRecord), InputRecord.Event.offset)
assert layout == (16, 16, 20, 4), layout
say("ok layout")

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                            wintypes.DWORD, wintypes.HANDLE]
k32.CreateFileW.restype = ctypes.c_void_p
k32.GetConsoleMode.argtypes, k32.GetConsoleMode.restype = [wintypes.HANDLE, wintypes.LPDWORD], wintypes.BOOL
k32.GetNumberOfConsoleInputEvents.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
k32.GetNumberOfConsoleInputEvents.restype = wintypes.BOOL
for name in ("WriteConsoleInputW", "PeekConsoleInputW"):
    getattr(k32, name).argtypes = [wintypes.HANDLE, ctypes.POINTER(InputRecord), wintypes.DWORD, wintypes.LPDWORD]
    getattr(k32, name).restype = wintypes.BOOL
k32.FlushConsoleInputBuffer.argtypes, k32.FlushConsoleInputBuffer.restype = [wintypes.HANDLE], wintypes.BOOL
user32.MapVirtualKeyW.argtypes, user32.MapVirtualKeyW.restype = [wintypes.UINT, wintypes.UINT], wintypes.UINT


def open_console_input():
    """A new handle (its own file object) of this process's console input, CONIN$: (fd, handle)."""
    try:
        fd = os.open("CONIN$", os.O_RDWR | os.O_BINARY)
    except OSError as e:
        h = k32.CreateFileW("CONIN$", 0x80000000 | 0x40000000, 1 | 2, None, 3, 0, None)   # read+write, shared, existing
        if h is None or h == ctypes.c_void_p(-1).value:
            skip(f"this console's input cannot be opened (os.open: {e}; CreateFileW: error {ctypes.get_last_error()})")
        fd = msvcrt.open_osfhandle(h, 0)
    return fd, wintypes.HANDLE(msvcrt.get_osfhandle(fd))


fd, handle = open_console_input()               # this process's own console (CREATE_NEW_CONSOLE); stdin is NUL
if not k32.GetConsoleMode(handle, ctypes.byref(wintypes.DWORD())):
    skip(f"CONIN$ is no console input here (GetConsoleMode: error {ctypes.get_last_error()})")
say("ok console")


def key(char, vk, down, state):
    record = InputRecord()
    record.EventType = 1                        # KEY_EVENT
    event = record.Event.KeyEvent
    event.bKeyDown, event.wRepeatCount, event.wVirtualKeyCode = down, 1, vk
    event.wVirtualScanCode = user32.MapVirtualKeyW(vk, 0)                  # MAPVK_VK_TO_VSC
    event.UnicodeChar, event.dwControlKeyState = char, state
    return record


def typed(text):                                # each character pressed and let go
    records = []
    for char in text:
        if char == "\r":
            vk, state = 0x0D, 0                 # VK_RETURN
        elif char == "\x1a":
            vk, state = 0x5A, 0x0008            # Z with LEFT_CTRL_PRESSED
        else:
            vk, state = ord(char.upper()), 0
        records += [key(char, vk, True, state), key(char, vk, False, state)]
    return records


def write(records, into=handle):
    array, written = (InputRecord * len(records))(*records), wintypes.DWORD()
    if not k32.WriteConsoleInputW(into, array, len(records), ctypes.byref(written)) or written.value != len(records):
        say(f"FAIL WriteConsoleInputW: error {ctypes.get_last_error()}, {written.value} of {len(records)} written")
        say("main done")
        sys.exit(5)


def waiting():
    """The input buffer's events, (EventType, key down, virtual key) each."""
    count, seen = wintypes.DWORD(), wintypes.DWORD()
    if not k32.GetNumberOfConsoleInputEvents(handle, ctypes.byref(count)) or not count.value:
        return []
    records = (InputRecord * count.value)()
    k32.PeekConsoleInputW(handle, records, count.value, ctypes.byref(seen))
    return [(r.EventType, bool(r.Event.KeyEvent.bKeyDown), r.Event.KeyEvent.wVirtualKeyCode) for r in records[:seen.value]]


def enter_waiting():
    return any(kind == 1 and down and vk == 0x0D for kind, down, vk in waiting())


CALLS = {                                       # the main thread's calls while a reader waits on the console
    "os.fstat": lambda: os.fstat(fd),
    "os.isatty": lambda: os.isatty(fd),
    # as mt.py starts ffprobe and ffmpeg: stdin NUL, its own pipes for stdout and stderr, no console window
    "subprocess.run(child_options)": lambda: subprocess.run([sys.executable, "-c", "pass"], capture_output=True,
                                                            timeout=30, **mt.child_options()),
}

if mode == "plain":                             # the report: os.read(fd, 4096) left pending, a cooked read on CONIN$
    other = open_console_input()[1]             # types the Enter that ends it, through a file object of its own

    def plain_reader():
        while True:
            try:
                os.read(fd, 4096)
            except OSError:
                time.sleep(0.1)
    threading.Thread(target=plain_reader, daemon=True).start()
    time.sleep(0.5)                             # the read is pending now; nothing has been typed
    for name, call in CALLS.items():
        done, raised = threading.Event(), []

        def run(call=call, done=done, raised=raised):
            try:
                call()
            except Exception as e:
                raised.append(e)
            done.set()
        threading.Thread(target=run, daemon=True).start()
        started = time.monotonic()
        if done.wait(10):
            outcome = f"came back in {time.monotonic() - started:.2f} s"
        else:
            write(typed("\r"), into=other)
            released = time.monotonic()
            outcome = "HUNG 10 s while the read was pending, " + (
                f"came back {time.monotonic() - released:.2f} s after Enter was typed" if done.wait(5)
                else "and 5 s more after Enter was typed")
        say(f"report {name}: {outcome}" + (f" (it raised {raised[0]!r})" if raised else ""))
    say("main done")
    os._exit(0)                                 # a call still hung must not hold up the exit

got, real_reader = [], mt.stdin_reader


def logged_reader(fd):                          # what Control._read gets from stdin_reader, shown to the parent
    read = real_reader(fd)

    def logged():
        data = read()
        say("read " + repr(data))
        return data
    return logged


mt.stdin_reader = logged_reader
channel = mt.Control()
channel.listen({verb: (lambda verb=verb: (got.append(verb), say("got " + verb))) for verb in (*mt.CONTROL_VERBS, "eof")})
channel.start(fd=fd)
time.sleep(0.5)                                 # the reader waits on the console now
CALLS["os.fstat"]()
say("ok os.fstat")
say("ok os.isatty" if CALLS["os.isatty"]() else "FAIL os.isatty is False for a console")
CALLS["subprocess.run(child_options)"]()
say("ok subprocess.run(child_options)")
focus = InputRecord()
focus.EventType = 0x10                          # FOCUS_EVENT, which a line read skips
focus.Event.FocusEvent.bSetFocus = True
write([focus])
say("ok non-key events dropped" if until(lambda: not waiting(), 5) else f"FAIL the input buffer keeps {waiting()}")
write([key("\n", 0x0D, True, 0x0008), key("\n", 0x0D, False, 0x0008)])  # Ctrl+Enter: VK_RETURN, but the character \n
time.sleep(1)                                   # 10 looks of the reader; a cooked read started would take the records
say("ok ctrl+enter starts no read" if enter_waiting() else f"FAIL the Ctrl+Enter key-down was taken in: {waiting()}")
k32.FlushConsoleInputBuffer(handle)
write(typed("\x1a\r"))                          # Ctrl+Z, Enter: a line that conhost returns as 0 bytes (ProcessControlZ)
say("ok ctrl+z line read" if until(lambda: not enter_waiting(), 5) else f"FAIL the Ctrl+Z line stays: {waiting()}")
time.sleep(0.5)                                 # were that line the end, eof would be dispatched by now
write(typed("stop\r"))
say("ok stop handled" if until(lambda: "stop" in got, 20) else f"FAIL no stop within 20 s, got {got}")
say("main done")
sys.exit(0 if "stop" in got and "eof" not in got else 6)
'''
CONSOLE_STEPS = ["ok layout", "ok console", "ok os.fstat", "ok os.isatty", "ok subprocess.run(child_options)",
                 "ok non-key events dropped", "ok ctrl+enter starts no read", "ok ctrl+z line read", "ok stop handled"]


def console_child(mode, tmp_path, name):
    """CONSOLE_CHILD in a console of its own (CREATE_NEW_CONSOLE, its window hidden); OSError if none can be made."""
    startup = subprocess.STARTUPINFO()
    startup.dwFlags, startup.wShowWindow = subprocess.STARTF_USESHOWWINDOW, subprocess.SW_HIDE
    return Child(CONSOLE_CHILD, [mode], tmp_path, name, stdin=subprocess.DEVNULL, startupinfo=startup,
                 creationflags=subprocess.CREATE_NEW_CONSOLE)


@pytest.mark.skipif(sys.platform != "win32", reason="a real Windows console")
def test_in_a_real_windows_console_a_typed_stop_is_read_and_nothing_waits_behind_the_reader(tmp_path):
    """mt.Control on the input of a console of the child's own (CREATE_NEW_CONSOLE, hidden; CONIN$), typed into with
    WriteConsoleInputW (key-down and key-up records). While the reader waits, the main thread's os.fstat and os.isatty of
    it and a child started as mt.py starts ffmpeg all come back. A focus event alone in the input buffer is dropped (the
    buffer empties); a Ctrl+Enter (VK_RETURN with the character \\n, which ends no cooked read) is not taken in by a read
    within 1 s; a line of Ctrl+Z is read and is not the end; then "stop" and Enter reach the stop handler, with no eof
    before it. The ctypes INPUT_RECORD is checked against the Windows layout (20 bytes, the union at offset 4)."""
    try:
        child = console_child("mt", tmp_path, "console")
    except OSError as e:
        pytest.skip(f"no console of its own could be made for a child (CreateProcess, CREATE_NEW_CONSOLE: {e})")
    try:
        finished = child.until("main done", 90)
        skipped = [line[len("skip "):] for line in child.seen if line.startswith("skip ")]
        if skipped:
            pytest.skip(skipped[0])
        steps = [line for line in child.seen if line.startswith(("ok ", "FAIL "))]
        assert finished and steps == CONSOLE_STEPS, (child.seen, child.stderr())
        reads = [line for line in child.seen if line.startswith("read ")]
        assert "got stop" in child.seen and "got eof" not in child.seen, child.seen
        assert reads[-1:] == ["read b'stop\\r\\n'"], (reads, child.stderr())
        assert child.proc.wait(30) == 0, child.stderr()
    finally:
        child.end()


@pytest.mark.skipif(sys.platform != "win32", reason="a real Windows console")
def test_windows_report_which_main_thread_calls_wait_behind_a_read_left_pending_on_a_console(tmp_path):
    """A report, never a failure, as the one above for a pipe: whether stdin_reader needs its console branch at all (it
    costs the echo while typing). The reader before the fix (a daemon thread in os.read(fd, 4096), a cooked read pending
    on CONIN$ of a console of the child's own), then the main thread's os.fstat and os.isatty of that fd and a child
    started as mt.py starts ffmpeg, each given 10 s. One that has not come back is given 5 s more after an Enter is typed
    (through a second CONIN$ handle), which ends the read: coming back then means it waited for that read."""
    report = []
    try:
        child = console_child("plain", tmp_path, "console-plain")
    except OSError as e:
        report.append(f"not run: no console of its own could be made for a child ({e})")
    else:
        try:
            finished = child.until("main done", 90)
            for line in child.seen:
                if line.startswith("report "):
                    report.append(line[len("report "):])
                elif line.startswith("skip "):
                    report.append("not run: " + line[len("skip "):])
                elif line.startswith("FAIL "):
                    report.append(line)
            if not finished:
                report.append(f"the child ended or did not finish within 90 s (stdout {child.seen}, "
                              f"stderr {child.stderr()!r})")
        finally:
            child.end()
    warnings.warn(f"Windows report, a plain read pending on a console in another thread (Python "
                  f"{sys.version.split()[0]}): " + "; ".join(report))


# ---------- the control channel: record() ----------
def test_record_pauses_and_resumes_on_the_lines_and_a_second_pause_or_resume_changes_nothing(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    notes = []
    threading.Timer(0.3, lambda: control.send("pause\n")).start()           # after about 3 s of (fast) audio
    threading.Timer(0.6, lambda: control.send("  PAUSE \n")).start()        # paused already: nothing changes
    threading.Timer(1.4, lambda: control.send("Resume\n")).start()          # 1.1 s after the pause
    threading.Timer(1.6, lambda: control.send("resume\n")).start()          # recording already: nothing changes
    folder, wav, _ = mt.record("x", 12 / 3600, None, notes=notes)
    assert 12 <= _seconds(wav) <= 20, "the pause is not in the file and does not use up the limit"
    pause = [n for n in notes if "Paused here" in n]
    assert len(pause) == 1 and re.fullmatch(r"\[00:00:0\d\] Paused here for 1 s", pause[0]), pause
    out = capsys.readouterr()
    assert out.err.count("paused\n") == 1 and out.err.count("recording\n") == 1 and f"@@NOTE\t{pause[0]}" in out.out
    assert any(n.startswith("Recording stopped at the") for n in notes)
    assert not control.terminated and not control.interrupted


def test_record_is_extended_by_the_extend_line(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EXTEND_S", 10)
    monkeypatch.setattr(mt, "NOTICE_S", 1)
    threading.Timer(0.3, lambda: control.send("extend\n")).start()
    folder, wav, _ = mt.record(None, 5 / 3600, None)                        # 5 s, extended by 10 s
    assert _seconds(wav) >= 15
    assert "Recording extended: it now stops after 0.25 min" in capsys.readouterr().err


def test_the_stop_line_ends_a_recording_as_ctrl_c_does_and_a_pause_before_it_is_still_told(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    notes = []
    threading.Timer(0.3, lambda: control.send("pause\n")).start()
    threading.Timer(0.8, lambda: control.send("stop\n")).start()
    folder, wav, _ = mt.record("x", 60 / 3600, None, notes=notes)          # a 1-min limit: 6 s at 10x real time
    assert 1 <= _seconds(wav) <= 10 and [n for n in notes if "Paused here" in n]
    assert not any(n.startswith("Recording stopped at the") for n in notes), "stopped by the line, not at the limit"
    err = capsys.readouterr().err
    assert "paused" in err and "Saved" in err and "type pause, resume, extend" in err
    assert mt.CONTROL.handlers == {}, "record() gave the channel back"


def test_nothing_captured_while_paused_by_the_lines_reaches_the_file(monkeypatch, tmp_path, control):
    _fake_recording(monkeypatch, tmp_path)                  # skips without soundfile, as the other recording tests do
    import soundfile as sf
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=StampedStream))
    StampedStream.deliveries = []
    marks = {}

    def say(name, line):
        marks[name] = time.monotonic()
        control.send(line)
    threading.Timer(0.3, lambda: say("pause", "pause\n")).start()
    threading.Timer(0.9, lambda: say("resume", "resume\n")).start()
    threading.Timer(1.2, lambda: say("pause2", "pause\n")).start()
    threading.Timer(1.3, lambda: control.send("stop\n")).start()            # stopped while paused, with blocks pending
    folder, wav, _ = mt.record("x", 60 / 3600, None)
    written = set(sf.read(str(wav), dtype="int16")[0].reshape(-1).tolist())
    # How long after a pause line its gate can still be open: 0.05 s to hand the line over and act on it, and on Windows
    # one poll of stdin_reader's pipe reader (it looks every 50 ms, so no read is left pending) plus Windows' timer ticks
    # (a sleep can run about 15.6 ms long). With 0.05 s alone run c0d0d04 counted block 124, written 0.05-0.07 s after
    # the line, as captured while paused.
    lag = 0.05 + (0.1 if sys.platform == "win32" else 0.0)
    paused = {n for n, t in StampedStream.deliveries if marks["pause"] + lag < t < marks["resume"] - 0.02
              or t > marks["pause2"] + lag}
    assert paused and not (paused & written), sorted(paused & written)[:5]
    before = {n for n, t in StampedStream.deliveries if t < marks["pause"] - 0.02}
    assert before <= written, "what came before the pause is all there"


def test_the_cancel_line_keeps_the_recording_and_transcribes_nothing(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not transcribe")))
    threading.Timer(0.3, lambda: control.send("cancel\n")).start()
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(60 / 3600)])
    mt.main()
    folder = next(tmp_path.iterdir())
    assert 1 <= _seconds(folder / "audio.wav") < 20                                       # closed properly, header right
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"@@FOLDER\t{folder}"
    assert out[1] == f"@@NOTE\tRecording cancelled (cancel on stdin); the audio is kept in {folder}, nothing is transcribed"
    assert out[2] == f"@@OUT\t{folder}" and out[3] == "@@DONE\t0" and not control.terminated


@pytest.mark.parametrize("line, ends", [("cancel\n", True), ("stop\n", False), (None, False)])
def test_a_cancel_while_the_live_tail_is_finished_ends_the_process_as_sigterm_does_then(monkeypatch, tmp_path, control,
                                                                                        line, ends):
    """record --live: after the stop the loop reads no more lines, and SIGTERM is back to its default; a cancel while the
    live tail is finished ends the process as that SIGTERM would on the Mac. A second stop or the end of stdin (None)
    changes nothing."""
    _fake_recording(monkeypatch, tmp_path)

    class Track:
        def __init__(self, *a, **k):
            pass

        def feed(self, block):
            pass

    class Worker:
        def __init__(self, tracks, **k):
            self.tracks = tracks

        def start(self):
            pass

        def stop(self):                                     # the tail takes a while; the line comes meanwhile
            control.close() if line is None else control.send(line)
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline and not control.terminated:
                time.sleep(0.05)
    monkeypatch.setattr(mt, "LiveTrack", Track)
    monkeypatch.setattr(mt, "LiveWorker", Worker)
    threading.Timer(0.3, lambda: control.send("stop\n")).start()
    mt.record("x", 60 / 3600, None, live={})
    assert control.terminated == ([True] if ends else []) and not control.interrupted


def test_a_cancel_while_the_recorder_starts_records_nothing(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    real = mt.CONTROL.listen

    def listen(handlers):
        before = real(handlers)
        if "pause" in handlers:
            handlers["cancel"]()                           # the line comes as soon as record() listens, before its folder
        return before
    monkeypatch.setattr(mt.CONTROL, "listen", listen)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", "1"])
    mt.main()
    assert not list(tmp_path.iterdir()), "no folder was made"
    assert "@@NOTE\tRecording cancelled (cancel on stdin) before it began; nothing was recorded" in capsys.readouterr().out


def test_the_end_of_stdin_stops_a_recording_which_is_then_saved_and_transcribed(monkeypatch, tmp_path, capsys, control):
    _fake_recording(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(mt, "transcribe", lambda src, d, *a, **k: seen.append(src) or {"lines": [], "translations": {}})
    monkeypatch.setattr(mt, "write_outputs", lambda *a, **k: [])
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    threading.Timer(0.3, control.close).start()                             # the front end went away
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(60 / 3600)])
    mt.main()
    assert len(seen) == 1 and 1 <= _seconds(seen[0]) < 20, "stopped by the end of stdin, saved, then transcribed"
    assert not control.terminated and not control.interrupted


def test_without_mt_control_stdin_is_never_read_and_its_end_stops_nothing(monkeypatch, tmp_path, capsys):
    """The Mac app runs mt.py with /dev/null as stdin and without MT_CONTROL: that must mean nothing."""
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.delenv("MT_CONTROL", raising=False)
    channel = mt.Control()
    monkeypatch.setattr(mt, "CONTROL", channel)
    asked = []
    monkeypatch.setattr(mt.Control, "terminate", staticmethod(lambda: asked.append("terminate")))   # were it read: a
    # failure here, not pytest ended by SIGTERM
    devnull = open(os.devnull, "rb")

    class Ended:
        """/dev/null as the Mac app gives it (at its end at once), telling the test when anything touches it."""
        def __getattr__(self, name):
            asked.append(name)
            return getattr(devnull, name)
    monkeypatch.setattr(sys, "stdin", Ended())
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(2 / 3600), "--no-transcribe"])
    try:
        mt.main()
    finally:
        devnull.close()
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "@@NOTE\tRecording stopped at the 0.0333333 min limit", "it ran to its limit"
    assert asked == [] and channel.thread is None, "stdin was not touched"


def test_main_starts_the_reader_with_mt_control_and_an_end_of_stdin_after_the_work_changes_nothing(monkeypatch, tmp_path,
                                                                                                    capsys):
    """main()'s own wiring: it starts the reader (MT_CONTROL=stdin, a long command), and once the work is done a front end
    closing the pipe on @@DONE must not turn exit status 0 into SIGTERM's."""
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    channel = mt.Control()
    monkeypatch.setattr(mt, "CONTROL", channel)
    monkeypatch.setenv("MT_CONTROL", "stdin")
    r, w = os.pipe()
    real_start = channel.start
    monkeypatch.setattr(channel, "start", lambda fd=None: real_start(fd=r))     # main() starts it, on a pipe
    ended = []
    monkeypatch.setattr(mt.Control, "terminate", staticmethod(lambda: ended.append("terminate")))
    threading.Timer(0.3, lambda: os.write(w, b"stop\n")).start()
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(60 / 3600), "--no-transcribe"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    assert channel.thread is not None, "main() started the reader"
    assert out[-1] == "@@DONE\t0" and not any("limit" in o for o in out), out    # stopped by the line, not the limit
    os.close(w)                                            # the front end closes the pipe on @@DONE
    channel.thread.join(5)
    os.close(r)
    assert ended == [], "an end of stdin after the work must not turn exit status 0 into SIGTERM's"


def test_mt_py_ends_while_its_stdin_pipe_stays_open_and_the_reader_waits_on_it(tmp_path):
    """A front end keeps the pipe open until mt.py has ended, so the reader thread is still blocked in a read of it when
    the process exits: that must neither hang nor change the exit status (run for real on the Windows CI job)."""
    script = ("import sys; sys.path.insert(0, sys.argv[1]); import mt; mt.CONTROL.start(); "
              "mt.CONTROL.listen({v: (lambda: None) for v in (*mt.CONTROL_VERBS, 'eof')}); print('DONE', flush=True)")
    env = dict(os.environ, MT_CONTROL="stdin", MT_HOME=str(tmp_path / "data"))
    env.pop("ANTHROPIC_API_KEY", None)
    for command, status in (([sys.executable, "-c", script, str(MT.parent)], 0),
                            ([sys.executable, str(MT), "translate", str(tmp_path / "x.txt"), "--to", "English"], 1)):
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            code = p.wait(timeout=30)                      # stdin is still open here
            out, err = p.stdout.read(), p.stderr.read()
            assert code == status and not p.stdin.closed, (command[-1], code, err[-300:])
            assert out.split() == [b"DONE"] if status == 0 else b"needs ANTHROPIC_API_KEY" in err
        finally:
            p.stdin.close()
            if p.poll() is None:
                p.kill()
            p.stdout.close(), p.stderr.close()


def test_an_unknown_mt_device_stops_before_anything_starts(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MT_DEVICE", "gpu")
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the job started")))
    src = tmp_path / "x.wav"
    src.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(src)])
    with pytest.raises(SystemExit, match="use auto, cuda or cpu"):
        mt.main()
    assert "@@FILE" not in capsys.readouterr().out


# ---------- the control channel: follow ----------
@pytest.mark.parametrize("end", ["stop", "cancel", "eof"])
def test_follow_ends_on_stop_on_cancel_and_on_the_end_of_stdin(monkeypatch, tmp_path, capsys, control, end):
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    mic = tmp_path / "mic.wav"
    writer = mt.WavFile(mic)
    writer.write(_speech(20, gaps={5, 6, 12, 13, 19}))
    writer.close()
    threading.Timer(2.0, lambda: control.close() if end == "eof" else control.send(end + "\n")).start()
    mt.follow([mic], tmp_path, "en", "normal")
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "@@DONE\t0" and any(o.startswith("@@LINE\t") and '"live": true' in o for o in out), out[-3:]
    assert (tmp_path / "partial.json").exists() and not control.terminated and not control.interrupted


# ---------- UTF-8 output ----------
def test_events_and_text_are_utf8_even_where_the_stream_was_cp1252(monkeypatch):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", newline="\r\n"))
    monkeypatch.setattr(sys, "platform", "win32")
    mt.utf8_streams()
    mt.ev("LINE", '{"text": "Привет"}')
    print("Grüße, 你好")
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8") == '@@LINE\t{"text": "Привет"}\nGrüße, 你好\n', "UTF-8, and \\n stays \\n"


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_on_macos_and_linux_the_streams_stay_as_python_set_them(monkeypatch, platform):
    """A Latin-1 terminal keeps getting Latin-1, an undecodable file name its own bytes: as before 1.11.0."""
    raw = io.BytesIO()
    # newline="\n": the stream's own line end, so that the test reads the same bytes when it runs on Windows (whose
    # default would be "\r\n"); utf8_streams must leave the stream exactly as it is.
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="latin-1", errors="surrogateescape", newline="\n"))
    monkeypatch.setattr(sys, "platform", platform)
    mt.utf8_streams()
    print("Schöne Grüße", "Pr\udce4s")
    sys.stdout.flush()
    assert sys.stdout.encoding == "latin-1" and raw.getvalue() == b"Sch\xf6ne Gr\xfc\xdfe Pr\xe4s\n"


@pytest.mark.skipif(sys.platform != "win32", reason="mt.py leaves the streams as they are outside Windows")
def test_a_process_whose_streams_would_be_cp1252_prints_utf8(tmp_path):
    folder = tmp_path / "data" / "20260101-000000-x"
    folder.mkdir(parents=True)
    (folder / "transcript.json").write_text(json.dumps({"source": "s", "whisper": "w", "translations": {}, "lines": [
        {"t": 1.0, "lang": "ru", "conf": 0.9, "text": "Привет, мир"}]}, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="cp1252", MT_HOME=str(tmp_path / "data"))
    env.pop("PYTHONUTF8", None)
    done = subprocess.run([sys.executable, str(MT), "show"], capture_output=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.decode("utf-8") == "[00:00:01] [ru] Привет, мир\n"


@pytest.mark.skipif(sys.platform != "win32", reason="mt.py leaves the streams as they are outside Windows")
@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is not None and not shutil.which("ffmpeg"),
                    reason="would fetch ffmpeg: the job must fail early, at the missing file")
def test_at_events_with_cyrillic_survive_a_cp1252_pipe(tmp_path):
    """Before 1.11.0 the first @@FILE with a Cyrillic path raised UnicodeEncodeError on such a pipe (mtpy-windows #3)."""
    src = tmp_path / "встреча.wav"
    src.write_bytes(b"")                                   # unreadable: the job fails at once, after its events
    env = dict(os.environ, PYTHONIOENCODING="cp1252", MT_HOME=str(tmp_path / "data"), MT_EVENTS="1")
    env.pop("PYTHONUTF8", None)
    done = subprocess.run([sys.executable, str(MT), "transcribe", str(src)], capture_output=True, env=env, timeout=120)
    out = done.stdout.decode("utf-8")
    assert f"@@FILE\t1\t1\t{src}\n" in out and out.endswith("@@DONE\t1\n") and "UnicodeEncodeError" not in done.stderr.decode(
        "utf-8", "replace"), (out, done.stderr[-500:])
    assert b"\r\n" not in done.stdout


# ---------- names ----------
@pytest.mark.parametrize("bad", ["a<b", "a>b", 'say "hi"', "a|b", "why?", "a*", "tab\there", "bell\x07", "name.", "name ",
                                 "CON", "con", "nul.txt", "Com1", "LPT9.tar.gz", "COM¹", "aux .md", "x" * 65, "😀" * 33])
def test_on_windows_a_name_windows_cannot_use_stops_before_anything_starts(monkeypatch, bad):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(SystemExit, match="Nothing was started"):
        mt.check_name(bad)


def test_on_windows_ordinary_names_pass_and_elsewhere_the_rules_stay_as_they_were(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    for good in ("weekly", "Встреча 1", "a.b", "console", "COM10", "LPT", "nul-points", ".hidden", "x" * 64, None, ""):
        assert mt.check_name(good) == good
    with pytest.raises(SystemExit, match="cannot contain / "):
        mt.check_name("a:b")
    monkeypatch.setattr(sys, "platform", "linux")
    for name in ("a<b", "CON", "name.", "why?", "x" * 200):
        assert mt.check_name(name) == name, "outside Windows only / \\ : and NUL are refused, as in the Mac app"
    with pytest.raises(SystemExit, match="cannot contain / "):
        mt.check_name("a/b")


def test_on_windows_a_long_input_name_is_cut_for_its_data_folder(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert mt.windows_folder_name("a" * 63 + ". b" * 5) == "a" * 63
    assert mt.windows_folder_name("short") == "short" and mt.windows_folder_name(None) is None
    assert mt.utf16_len(mt.windows_folder_name("😀" * 40)) == 64
    monkeypatch.setattr(sys, "platform", "linux")
    assert mt.windows_folder_name("a" * 200) == "a" * 200


def test_the_longest_data_paths_with_a_64_character_name_fit_windows_max_path_in_the_apps_data_folder(monkeypatch):
    """Built with mt.py's own path code: an offline translation's log, and a recording's translation file, which stays
    in its data folder (the name twice, and the --to text)."""
    monkeypatch.setattr(mt, "DATA", Path("C:\\Users\\" + "u" * 20 + "\\AppData\\Local\\Meeting Transcriber\\data"))
    name = "n" * mt.WIN_NAME_MAX
    folder = mt._data_path(None, name)
    log = folder / "translations" / mt.translation_log_name("zho_Hant", mt.NLLB, "0" * 16)
    text = folder / mt.translation_file_name(name, "Brazilian Portuguese")
    assert [mt.utf16_len(str(p)) for p in (log, text)] == [235, 239]                    # both within 259


def test_on_windows_a_job_whose_longest_data_path_would_not_fit_stops_before_anything_starts(monkeypatch, tmp_path, capsys):
    data = ("C:\\Users\\" + "u" * 20 + "\\OneDrive - Contoso Corporation Ltd\\Documents\\GitHub\\mini-projs\\"
            "meeting-transcriber\\data")                                             # 117 characters
    monkeypatch.setattr(mt, "DATA", Path(data if os.name == "nt" else "/" + data[1:]))      # absolute here too, as long
    monkeypatch.setattr(mt, "windows_long_paths", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")
    folder = mt._data_path(None, "n" * 64)
    assert re.match(r".*translations.*\.jsonl would be 284 characters long, and without long paths Windows allows 259",
                    mt.windows_path_problem(folder, ["English"], True))
    assert "n.Brazilian_Portuguese.txt would be 288" in mt.windows_path_problem(folder, ["Brazilian Portuguese"], False,
                                                                               stem="n" * 64)
    assert mt.windows_path_problem(mt._data_path(None, "weekly"), ["English"], True) is None
    assert mt.windows_path_problem(folder, [], True) is None and mt.windows_path_problem(folder, ["English"], False) is None
    monkeypatch.setattr(mt, "utf8_streams", lambda: None)
    monkeypatch.setattr(mt, "keep_awake", lambda cmd: None)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--name", "n" * 64, "--to", "English", "--offline"])
    with pytest.raises(SystemExit, match=r"would be 284 characters long.*Nothing was started"):
        mt.main()
    monkeypatch.setattr(mt, "windows_long_paths", lambda: True)
    assert mt.windows_path_problem(folder, ["English"], True) is None
    monkeypatch.setattr(mt, "windows_long_paths", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert mt.windows_path_problem(folder, ["English"], True) is None, "never outside Windows"


def test_on_windows_a_long_input_name_is_cut_for_the_data_folder_it_makes(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    assert re.fullmatch(r"\d{8}-\d{6}-a{64}", mt._new_dir("a" * 100).name)


def test_on_windows_two_long_names_cut_to_the_same_64_characters_get_their_own_folders_within_the_cap(monkeypatch, tmp_path):
    """Codex on #99: two inputs whose names agree in their first 64 characters, begun in the same second, made one
    folder name, and the second failed with FileExistsError. The number now fits inside WIN_NAME_MAX (the name cut as far
    as it needs), and a short name is not cut at all."""
    from test_mt import FrozenClock
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "datetime", FrozenClock)
    monkeypatch.setattr(sys, "platform", "win32")
    common = "Quarterly review with the regional sales teams and the board, part"
    first, second = mt._new_dir(common + " one"), mt._new_dir(common + " two")
    assert first != second and first.is_dir() and second.is_dir()
    assert first.name == "20261126-152011-" + common[:64]
    assert second.name == "20261126-152011-" + common[:62] + "-2"
    for n in range(3, 11):
        last = mt._new_dir(common + f" {n}")
    assert last.name == "20261126-152011-" + common[:61] + "-10"
    assert all(mt.utf16_len(d.name[16:]) <= mt.WIN_NAME_MAX for d in tmp_path.iterdir())
    assert [mt._new_dir("short").name, mt._new_dir("short").name] == ["20261126-152011-short", "20261126-152011-short-2"]
    emoji = "\U0001f600" * 32                                        # 64 UTF-16 units in 32 characters
    assert mt._new_dir(emoji).name == "20261126-152011-" + emoji
    assert mt._new_dir(emoji).name == "20261126-152011-" + "\U0001f600" * 31 + "-2"   # a whole character goes, never half
    assert all(mt.utf16_len(d.name[16:]) <= mt.WIN_NAME_MAX for d in tmp_path.iterdir()), "counted in UTF-16 units"


def test_on_windows_a_numbered_data_folder_too_long_for_the_path_limit_is_refused_before_anything_is_recorded(
        monkeypatch, tmp_path, capsys):
    """record checks its data folder's longest path before it begins, for the first folder of that second. A second
    recording of that name in the same second gets `-2`, 2 characters longer: with the data folder at the limit it
    passed that check and would have failed only when it wrote its text files, after the whole recording and
    transcription. The numbered folder is checked again, removed again, and nothing is recorded."""
    import signal
    from test_mt import FrozenClock
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mt, "windows_long_paths", lambda: False)
    monkeypatch.setattr(mt, "datetime", FrozenClock)
    monkeypatch.setattr(mt, "OUT", None)                       # the text files stay in the data folder: weekly.English.txt
    monkeypatch.setattr(mt, "utf8_streams", lambda: None)
    monkeypatch.setattr(mt, "keep_awake", lambda cmd: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-used")  # refused before any translation
    monkeypatch.setattr(mt, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not transcribe")))

    def problem(data, n):
        monkeypatch.setattr(mt, "DATA", data)
        return mt.windows_path_problem(mt._data_path(None, "weekly", n=n), ["English"], False, "weekly")
    pad = next(k for k in range(1, 255) if problem(tmp_path / ("d" * (k + 1)), 1))   # the longest DATA the check passes
    data = tmp_path / ("d" * pad)
    assert problem(data, 1) is None and "would be 261 characters long" in problem(data, 2)
    (data / "20261126-152011-weekly").mkdir(parents=True)     # the first recording of that name in this second
    term = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--name", "weekly", "--to", "English", "--hours", str(2 / 3600)])
    with pytest.raises(SystemExit, match=r"weekly-2.weekly\.\w+\.txt would be 261 characters long.*Nothing was started"):
        mt.main()
    assert [d.name for d in data.iterdir()] == ["20261126-152011-weekly"], "the numbered folder is gone again"
    assert "Recording to" not in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) == term, "record() gave its SIGTERM handler back"


# ---------- the PC, child processes, memory, files ----------
def test_windows_is_kept_awake_while_a_long_command_runs_and_let_sleep_at_exit(monkeypatch):
    import atexit
    import ctypes
    calls, registered = [], []

    class State:
        def __call__(self, flags):
            calls.append(flags)
            return 0x80000000
    state = State()
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(SetThreadExecutionState=state)), raising=False)
    monkeypatch.setattr(atexit, "register", lambda f, *args: registered.append((f, args)))
    monkeypatch.setattr(sys, "platform", "win32")
    mt.keep_awake("list")
    assert calls == [] and registered == []
    mt.keep_awake("record")
    assert calls == [0x80000001] and state.argtypes == [ctypes.c_uint32]       # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    f, args = registered[0]
    f(*args)
    assert calls[-1] == 0x80000000                                            # ES_CONTINUOUS alone: sleep allowed again
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    mt.keep_awake("transcribe")                                               # a failure changes nothing


def test_memory_on_windows_comes_from_globalmemorystatusex(monkeypatch, tmp_path, capsys):
    import ctypes

    def status(ref):
        assert ref._obj.dwLength == 64                      # sizeof(MEMORYSTATUSEX)
        ref._obj.ullTotalPhys = 8 * 2**30
        return 1
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(GlobalMemoryStatusEx=status)), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert mt.total_memory() == 8 * 2**30
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=2**40))
    mt.preflight(2, 1)
    assert "This computer has 8 GB of memory: a recording this long needs about 3 GB" in capsys.readouterr().err
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    assert mt.total_memory() is None


def test_child_processes_get_no_stdin_and_on_windows_no_console_window(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: seen.append((cmd[0], kw)) or SimpleNamespace(
        returncode=1, stdout="", stderr=b"no"))
    monkeypatch.setattr(mt, "ensure_ffmpeg", lambda: True)
    for platform, flags in (("linux", None), ("win32", 0x08000000)):
        monkeypatch.setattr(sys, "platform", platform)
        seen.clear()
        mt.audio_tracks(tmp_path / "x.wav")
        with pytest.raises(RuntimeError, match="ffmpeg could not read"):
            mt.load_audio(tmp_path / "x.wav")
        assert [c for c, _ in seen] == ["ffprobe", "ffprobe", "ffmpeg"]
        assert all(kw["stdin"] is subprocess.DEVNULL and kw.get("creationflags") == flags for _, kw in seen), seen
        assert (seen[0][1]["encoding"], seen[0][1]["errors"]) == ("utf-8", "replace"), "not the ANSI code page"


def test_replace_is_retried_on_windows_while_the_target_is_held_and_a_lasting_refusal_is_said(monkeypatch, tmp_path, capsys):
    calls, slept = [], []
    real = os.replace

    def held(times):
        def replace(src, dst):
            calls.append(len(slept))
            if len(calls) <= times:
                raise PermissionError(13, "The process cannot access the file because it is being used by another process")
            return real(src, dst)
        return replace
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("new")
    b.write_text("old")
    # The waits are counted, not timed: before Python 3.13 time.monotonic() on Windows ticks in 15.6 ms steps.
    monkeypatch.setattr(mt, "time", SimpleNamespace(sleep=slept.append))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "replace", held(2))
    mt.replace_file(a, b)
    assert b.read_text() == "new" and calls == [0, 1, 2] and slept == [0.05, 0.1]      # 50 + 100 ms apart
    a.write_text("newer")
    calls.clear(), slept.clear()
    monkeypatch.setattr(os, "replace", held(99))
    with pytest.raises(PermissionError):
        mt.replace_file(a, b)
    assert len(calls) == 5 and slept == [0.05, 0.1, 0.2, 0.4] and b.read_text() == "new"   # 0.75 s in all
    assert "Controlled folder access" in capsys.readouterr().err
    monkeypatch.setattr(sys, "platform", "linux")
    calls.clear()
    with pytest.raises(PermissionError):
        mt.replace_file(a, b)
    assert len(calls) == 1, "elsewhere os.replace as it was, no retry"


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_an_output_file_windows_refuses_names_controlled_folder_access(monkeypatch, tmp_path, capsys, platform):
    """Controlled folder access protects Documents, where the output folder usually is; the transcript is written there
    with a plain write, not through replace_file."""
    real = Path.write_text

    def refused(self, *args, **kw):
        if self.name.endswith(".transcript.txt"):
            raise PermissionError(13, "Access is denied")
        return real(self, *args, **kw)
    monkeypatch.setattr(Path, "write_text", refused)
    monkeypatch.setattr(sys, "platform", platform)
    doc = {"lines": [{"t": 1.0, "lang": "en", "conf": 0.9, "text": "Hello."}]}
    with pytest.raises(PermissionError):
        mt.write_outputs(doc, tmp_path / "Documents", "m", [])
    err = capsys.readouterr().err
    assert ("Controlled folder access" in err) == (platform == "win32") and (platform != "win32" or "m.transcript.txt" in err)


def test_saves_go_through_the_retry(monkeypatch, tmp_path):
    used = []
    monkeypatch.setattr(mt, "replace_file", lambda src, dst: used.append(Path(dst).name) or os.replace(src, dst))
    mt._save(tmp_path, {"lines": []})
    mt.Checkpoint(tmp_path).start({"x": 1}, [(0, 1)])
    mt.claim_stem(tmp_path, "m", {}, owner="o")
    assert used == ["transcript.json", "partial.json", mt.OWNERS]


# ---------- devices ----------
def test_on_windows_devices_are_matched_among_the_wasapi_ones_and_opened_with_auto_convert(monkeypatch):
    devices = [{"name": "Microphone (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 0},          # MME
               {"name": "Microphone (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 1},          # DirectSound
               {"name": "Microphone (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 2},          # WASAPI
               {"name": "Speakers (Realtek(R) Audio)", "max_input_channels": 0, "hostapi": 2},
               {"name": "Microphone (Realtek HD Audio Mic input)", "max_input_channels": 2, "hostapi": 3}]  # WDM-KS
    hostapis = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]
    sd = SimpleNamespace(query_devices=lambda i=None: devices if i is None else devices[i], query_hostapis=lambda: hostapis,
                         WasapiSettings=lambda **kw: ("wasapi", kw))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(SystemExit, match="3 input devices match"):
        mt.resolve_device("Microphone (Realtek(R) Audio)")
    assert mt.stream_options(sd, 2) == {}, "elsewhere nothing changes"
    monkeypatch.setattr(sys, "platform", "win32")
    assert mt.resolve_device("Microphone (Realtek(R) Audio)") == 2 and mt.resolve_device("realtek") == 2
    assert mt.resolve_device("2") == 2
    with pytest.raises(SystemExit, match="0 is not an input device"):
        mt.resolve_device("0")
    assert mt.stream_options(sd, 2) == {"extra_settings": ("wasapi", {"auto_convert": True})}
    assert mt.stream_options(sd, 0) == {} and mt.stream_options(sd, None) == {}
    hostapis[2]["name"] = "ASIO"                                              # no WASAPI: every host API, as before
    with pytest.raises(SystemExit, match="3 input devices match"):
        mt.resolve_device("Microphone (Realtek(R) Audio)")


# ---------- texts ----------
def test_the_recording_line_names_the_signals_elsewhere_and_the_lines_on_windows(monkeypatch):
    pid = os.getpid()
    monkeypatch.delenv("MT_CONTROL", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert mt.recording_controls(4 * 3600) == (f"Ctrl+C to stop, stops by itself after 4 h; kill -USR1 {pid} adds an hour, "
                                               f"kill -USR2 {pid} pauses and resumes"), "as before 1.11.0"
    monkeypatch.setattr(sys, "platform", "win32")
    assert "MT_CONTROL=stdin" in mt.recording_controls(3600) and "kill" not in mt.recording_controls(3600)
    assert "winget" in mt.ffmpeg_hint()
    monkeypatch.setenv("MT_CONTROL", "stdin")
    assert mt.recording_controls(3600).startswith("stops by itself after 1 h; type pause, resume, extend")


# ---------- the GPU ----------
class FakeWhisper:
    """faster_whisper.WhisperModel: CUDA fails to load or at the first decode when told to; `missing` = not downloaded
    (huggingface_hub's LocalEntryNotFoundError, a FileNotFoundError); `decoded` counts the decodes (the warm-up)."""
    made, fail, missing, decoded = [], set(), False, []

    def __init__(self, name, device="auto", compute_type="default", cpu_threads=0, local_files_only=False):
        FakeWhisper.made.append((device, compute_type, local_files_only))
        if local_files_only and FakeWhisper.missing:
            raise FileNotFoundError("not in the cache")
        if device == "cuda" and "load" in FakeWhisper.fail:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        self.device = device

    def transcribe(self, audio, **kw):
        FakeWhisper.decoded.append(self.device)
        if self.device == "cuda" and "decode" in FakeWhisper.fail:
            raise RuntimeError("CUDA failed with error no kernel image is available for execution on the device")
        return iter([]), None


class FakeTranslator:
    made, translated = [], []

    def __init__(self, path, device="cpu", compute_type="default", intra_threads=0):
        FakeTranslator.made.append((device, compute_type))
        self.device = device

    def translate_batch(self, tokens, target_prefix=None, **kw):
        FakeTranslator.translated.append(self.device)
        if self.device == "cuda":
            raise RuntimeError("cuBLAS failed with status CUBLAS_STATUS_NOT_INITIALIZED")
        return []


@pytest.fixture
def gpu(monkeypatch, tmp_path):
    """Fake ctranslate2 / faster_whisper / huggingface_hub / tokenizers; `counted` GPUs. The cached models are let go."""
    counted = SimpleNamespace(n=0, asked=0)

    def count():
        counted.asked += 1
        return counted.n
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=count, Translator=FakeTranslator))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda repo, **kw: str(tmp_path)))
    (tmp_path / "model.bin").write_bytes(b"")          # a complete NLLB snapshot (one without its weights is fetched again)
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: "tok")))
    monkeypatch.delenv("MT_DEVICE", raising=False)
    FakeWhisper.made, FakeWhisper.fail, FakeWhisper.missing, FakeWhisper.decoded = [], set(), False, []
    FakeTranslator.made, FakeTranslator.translated = [], []
    mt.get_model.cache_clear()
    mt._load_nllb.cache_clear()
    yield counted
    mt.get_model.cache_clear()
    mt._load_nllb.cache_clear()


@pytest.fixture(params=["windows", "MT_DEVICE=auto"])
def checked(request, gpu, monkeypatch):
    """One counted GPU where a CUDA model is checked at load and left to the CPU when it fails (mt.gpu_checked): on
    Windows (MT_DEVICE unset; the DLL preload recorded, not done), and on Linux with MT_DEVICE set."""
    preloads = []
    if request.param == "windows":
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(mt, "nvidia_libraries", lambda: preloads.append(True) or [])
    else:
        monkeypatch.setenv("MT_DEVICE", "auto")
    gpu.n = 1
    return SimpleNamespace(gpu=gpu, preloads=preloads, where=request.param)


def test_without_a_gpu_the_speech_model_is_loaded_exactly_as_before(gpu, capsys):
    mt.get_model()
    assert FakeWhisper.made == [("auto", "int8", True)] and capsys.readouterr().err == ""
    mt.get_model.cache_clear()
    FakeWhisper.made, FakeWhisper.missing = [], True
    mt.get_model()
    assert FakeWhisper.made == [("auto", "int8", True), ("auto", "int8", False)], "not downloaded yet: fetched"
    assert FakeWhisper.decoded == []


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_outside_windows_without_mt_device_a_gpu_loads_both_models_exactly_as_before(gpu, monkeypatch, capsys, platform):
    """Before 1.11.0: WhisperModel(compute_type="float16") with CTranslate2's own device choice and Translator(device=
    "cuda"); no warm-up, no fallback, nothing said."""
    monkeypatch.setattr(sys, "platform", platform)
    gpu.n = 1
    mt.get_model()
    mt._load_nllb()
    assert FakeWhisper.made == [("auto", "float16", True)] and FakeWhisper.decoded == []
    assert FakeTranslator.made == [("cuda", "default")] and FakeTranslator.translated == []
    assert capsys.readouterr().err == ""
    mt.get_model.cache_clear()
    FakeWhisper.made, FakeWhisper.missing = [], True
    mt.get_model()
    assert FakeWhisper.made == [("auto", "float16", True), ("auto", "float16", False)], "not downloaded yet: fetched"


def test_a_working_gpu_takes_the_speech_model_in_float16(checked, capsys):
    assert mt.get_model().device == "cuda" and FakeWhisper.made == [("cuda", "float16", True)]
    assert FakeWhisper.decoded == ["cuda"], "the warm-up decode"
    assert capsys.readouterr().err == ""
    assert checked.preloads == ([True] if checked.where == "windows" else [])


@pytest.mark.parametrize("failure", ["load", "decode"])
def test_a_gpu_that_fails_to_load_or_to_decode_leaves_the_speech_model_to_the_cpu_with_a_note(checked, capsys, monkeypatch,
                                                                                             failure):
    FakeWhisper.fail = {failure}
    monkeypatch.setattr(mt, "EVENTS", True)
    assert mt.get_model().device == "cpu"
    assert FakeWhisper.made == [("cuda", "float16", True), ("cpu", "int8", True)], "no second CUDA try that asks the Hub"
    out = capsys.readouterr().out
    assert re.search(r"@@NOTE\tThe GPU could not be used for the speech model \(RuntimeError: .*(cublas64_12|no kernel).*\); "
                     r"it runs on the CPU instead", out), out


def test_a_gpu_pc_without_the_model_downloads_it_for_cuda(checked):
    FakeWhisper.missing = True
    FakeWhisper.made = []
    mt.get_model()
    assert FakeWhisper.made == [("cuda", "float16", True), ("cuda", "float16", False)]


def test_mt_device_forces_the_cpu_or_insists_on_cuda(gpu, monkeypatch):
    gpu.n = 1
    monkeypatch.setenv("MT_DEVICE", "cpu")
    assert mt.get_model().device == "cpu" and FakeWhisper.made == [("cpu", "int8", True)] and gpu.asked == 0
    mt.get_model.cache_clear()
    monkeypatch.setenv("MT_DEVICE", "cuda")
    gpu.n, FakeWhisper.fail, FakeWhisper.made = 0, {"load"}, []
    with pytest.raises(RuntimeError, match="cublas64_12"):
        mt.get_model()                                       # an error, not the fallback: for testing the GPU
    assert FakeWhisper.made == [("cuda", "float16", True)], "the first CUDA error, without asking the Hub"
    monkeypatch.setenv("MT_DEVICE", "gpu")
    with pytest.raises(SystemExit, match="use auto, cuda or cpu"):
        mt.compute_device()


def test_the_translation_model_is_unchanged_without_a_gpu(gpu):
    translator, tok = mt._load_nllb()
    assert FakeTranslator.made == [("cpu", "default")] and tok == "tok", "no GPU: as before"


def test_the_translation_model_falls_back_to_the_cpu_when_cuda_fails(checked, capsys):
    translator, _ = mt._load_nllb()
    assert translator.device == "cpu" and FakeTranslator.made == [("cuda", "default"), ("cpu", "int8")]
    assert "The GPU could not be used for the translation model (RuntimeError: cuBLAS failed" in capsys.readouterr().err


def test_the_dll_preload_runs_before_ctranslate2_loads_on_windows_only_and_not_with_mt_device_cpu(gpu, monkeypatch):
    preloads = []
    monkeypatch.setattr(mt, "nvidia_libraries", lambda: preloads.append(True) or [])
    monkeypatch.setattr(sys, "platform", "win32")
    assert mt.import_ctranslate2() is sys.modules["ctranslate2"] and preloads == [True]
    monkeypatch.setenv("MT_DEVICE", "cpu")
    mt.import_ctranslate2()
    monkeypatch.delenv("MT_DEVICE")
    monkeypatch.setattr(sys, "platform", "linux")
    mt.import_ctranslate2()
    assert preloads == [True]


def test_the_nvidia_wheels_dlls_are_made_findable_and_loaded_on_windows(monkeypatch, tmp_path):
    import ctypes
    import site as site_module
    import sysconfig
    site, planted, elsewhere = tmp_path / "site-packages", tmp_path / "Downloads", tmp_path / "on-sys-path"
    for root, tree in ((site, {"cublas/bin": ["cublas64_12.dll", "cublasLt64_12.dll", "nvblas64_12.dll"],
                               "cudnn/bin": ["cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll"],   # an older install
                               "cuda_nvrtc/bin": ["nvrtc64_120_0.dll", "nvrtc64_120_0.alt.dll", "nvrtc-builtins64_129.dll"]}),
                       (planted, {"cublas/bin": ["cublas64_12.dll"]}), (elsewhere, {"evil/bin": ["cublasLt64_12.dll"]})):
        for folder, names in tree.items():
            (root / "nvidia" / folder).mkdir(parents=True)
            for name in names:
                (root / "nvidia" / folder / name).write_bytes(b"")
    added, loaded = [], []
    monkeypatch.setattr(os, "add_dll_directory", lambda p: added.append(str(p)) or p, raising=False)
    monkeypatch.setattr(ctypes, "WinDLL", lambda p: loaded.append(str(p)) or p, raising=False)
    monkeypatch.setattr(site_module, "getsitepackages", lambda: [str(site)])
    monkeypatch.setattr(site_module, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(site), "platlib": str(site)})
    monkeypatch.chdir(planted)                               # a planted DLL in the current folder, '' on sys.path
    monkeypatch.setattr(sys, "path", ["", str(elsewhere)] + sys.path)
    monkeypatch.setenv("PATH", "first-before")
    monkeypatch.setattr(sys, "platform", "win32")
    mt.nvidia_libraries.cache_clear()
    try:
        assert mt.nvidia_libraries() == [Path(p).name for p in loaded] == [
            "cublasLt64_12.dll", "cublas64_12.dll", "nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll"], \
            "cuBLASLt before cuBLAS; no .alt twin of NVRTC; no cuDNN (CTranslate2 calls none)"
        assert all(Path(p).is_relative_to(site) for p in loaded + added), "only this Python's own site-packages"
        assert [Path(p).parent.name for p in added] == ["cublas", "cuda_nvrtc", "cudnn"]
        path = os.environ["PATH"].split(os.pathsep)
        assert path[:3] == added and path[3] == "first-before"
        monkeypatch.setattr(sys, "platform", "linux")
        mt.nvidia_libraries.cache_clear()
        assert mt.nvidia_libraries() == []
    finally:
        mt.nvidia_libraries.cache_clear()
        mt._NVIDIA_HANDLES.clear()
