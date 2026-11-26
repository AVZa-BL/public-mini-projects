"""tools/win_probe.py: the Windows probe. On Linux the Windows checks are reported as skipped; what is tested here is
that the probe imports with the standard library only, runs to the end in --ci, writes valid JSON with every section,
keeps the home folder out of the report, and the pure helpers that turn Windows answers into findings."""
import importlib.util
import json
import os
import py_compile
import struct
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

PROBE = Path(__file__).parent.parent / "tools" / "win_probe.py"
SECTIONS = ["os", "python", "security", "audio", "loopback", "output", "gpu", "ffmpeg"]
OPTIONAL = ["sounddevice", "numpy", "soundcard", "pyaudiowpatch", "pycaw", "comtypes", "ctranslate2", "faster_whisper",
            "static_ffmpeg"]


def load():
    spec = importlib.util.spec_from_file_location("win_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wp = load()


@pytest.fixture(scope="module")
def ci_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("probe") / "probe.json"
    done = subprocess.run([sys.executable, str(PROBE), "--ci", "--strict", "--out", str(out)], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL)
    return done, out


def test_the_file_compiles():
    py_compile.compile(str(PROBE), doraise=True)


def test_importing_loads_no_optional_package():
    code = (f"import importlib.util, json, sys; spec = importlib.util.spec_from_file_location('p', {str(PROBE)!r}); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            f"print(json.dumps(sorted(set(sys.modules) & set({OPTIONAL!r}))))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_ci_run_ends_with_exit_0_and_every_section(ci_run):
    done, out = ci_run
    assert done.returncode == 0, done.stderr[-2000:]
    text = done.stdout
    for n, title in enumerate(["Operating system", "Python", "Security features", "Microphone", "The computer's sound",
                               "Output device kind", "NVIDIA and CUDA", "ffmpeg"], 1):
        assert f"== {n}. {title}" in text
    assert "== What this means" in text
    block = text.split("-----BEGIN MT PROBE-----\n", 1)[1].split("\n-----END MT PROBE-----", 1)[0]
    report = json.loads(block)
    assert json.loads(out.read_text(encoding="utf-8")) == report
    assert [s["id"] for s in report["sections"]] == SECTIONS
    assert report["summary"] and report["mode"] == "ci"
    for section in report["sections"]:
        assert section["checks"], section["id"]
        for check in section["checks"]:
            assert check["status"] in ("ok", "failed", "skipped"), check     # "error" is a bug of the probe


def test_an_out_file_that_cannot_be_written_still_leaves_the_report_on_screen(tmp_path):
    """--out in a folder that does not exist: the report block is printed before the write, the write failure is a
    warning, and the exit status stays 0 (a user's whole run is not lost to a mistyped path)."""
    out = tmp_path / "no such folder" / "probe.json"
    done = subprocess.run([sys.executable, str(PROBE), "--ci", "--out", str(out)], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=300, stdin=subprocess.DEVNULL)
    assert done.returncode == 0, done.stderr[-2000:]
    block = done.stdout.split("-----BEGIN MT PROBE-----\n", 1)[1].split("\n-----END MT PROBE-----", 1)[0]
    assert json.loads(block)["mode"] == "ci"
    assert "could not write --out" in done.stderr and not out.exists()


def test_the_powershell_child_gets_no_module_path_whatever_the_case_of_its_name():
    """Windows' os.environ has upper-case names (PSMODULEPATH): the variable must go in any spelling, the rest stays."""
    env = {"PSMODULEPATH": r"C:\Program Files\PowerShell\7\Modules", "PSModulePath": "x", "psmodulepath": "y",
           "PATH": r"C:\Windows", "MTPROBE_FILES": "a|b"}
    assert wp.without_ps_module_path(env) == {"PATH": r"C:\Windows", "MTPROBE_FILES": "a|b"}


def test_signatures_fail_when_every_file_errored(monkeypatch):
    """A PowerShell that cannot load Get-AuthenticodeSignature (PowerShell 7's module path in 5.1) is a failed check,
    not an ok with one error line per file."""
    monkeypatch.setattr(wp, "need_windows", lambda: None)
    probe = Namespace(powershell=lambda: {"signatures": [{"file": "a.dll", "status": "error", "signer": "not loaded"},
                                                         {"file": "b.dll", "status": "error", "signer": "not loaded"}]})
    with pytest.raises(wp.Fail, match="every file"):
        wp.check_signatures(probe)
    probe = Namespace(powershell=lambda: {"signatures": [{"file": "a.dll", "status": "error", "signer": "x"},
                                                         {"file": "b.dll", "status": "NotSigned", "signer": ""}]})
    assert "b.dll: NotSigned" in wp.check_signatures(probe)[0]


@pytest.mark.skipif(sys.platform == "win32", reason="on Windows these checks run")
def test_ci_run_skips_the_windows_checks_elsewhere(ci_run):
    report = json.loads(ci_run[1].read_text(encoding="utf-8"))
    checks = {c["id"]: c for s in report["sections"] for c in s["checks"]}
    for cid in ("os.paths", "os.powershell", "py.vcruntime", "py.signatures", "sec.sac", "sec.cfa", "sec.documents",
                "sec.microphone", "audio.default", "audio.wasapi_auto", "audio.wasapi_plain", "loop.soundcard.once",
                "loop.pyaudiowpatch.once", "out.formfactor", "gpu.nvcuda", "gpu.wheels", "gpu.dlls", "gpu.whisper"):
        assert checks[cid]["status"] == "skipped" and "not Windows" in checks[cid]["detail"], checks[cid]
    # The encoding check runs everywhere, in two child processes.
    assert checks["py.encoding"]["status"] == "ok"
    assert checks["py.encoding"]["data"]["pipe_utf8"]["writable"] is True


def test_the_report_hides_the_home_folder(ci_run):
    home = os.path.expanduser("~").rstrip("/\\")
    if len(home) > 3:
        assert home not in ci_run[0].stdout


def test_levels():
    r = wp.levels([0, 16384, -32768, 0], 32768)
    assert r["samples"] == 4 and r["peak"] == 1.0 and r["peak_dbfs"] == 0.0
    assert abs(r["rms"] - (1.25 / 4) ** 0.5) < 1e-4
    assert wp.levels([], 1.0) == {"samples": 0, "rms": 0.0, "peak": 0.0, "rms_dbfs": None, "peak_dbfs": None}
    assert wp.levels([0.0, 0.0], 1.0)["peak_dbfs"] is None


def test_output_kind_follows_the_mac_rules():
    assert [wp.output_kind(f) for f in ("Speakers", "DigitalAudioDisplayDevice", "SPDIF")] == ["speaker"] * 3
    assert [wp.output_kind(f) for f in ("Headphones", "Headset", "Handset")] == ["headphones"] * 3
    assert wp.output_kind("LineLevel") == wp.output_kind(None) == "unknown"
    assert wp.FORM_FACTORS[1] == "Speakers" and wp.FORM_FACTORS[3] == "Headphones" and len(wp.FORM_FACTORS) == 11


def test_python_build_kind():
    exe = r"C:\Program Files\Python312\python.exe"
    assert wp.python_build_kind("3.12.7 (tags/v3.12.7:0b05ead, Oct  1 2024, 03:06:41) [MSC v.1941 64 bit (AMD64)]",
                                exe) == "python.org"
    assert wp.python_build_kind("3.12.16 (main, Oct 26 2026, 00:00:00) [MSC v.1944 64 bit (AMD64)]",
                                exe) == "python-build-standalone"
    assert wp.python_build_kind("3.12.3 (main, Nov  6 2025) [GCC 13.3.0]", "/usr/bin/python3") == "other"
    assert wp.python_build_kind("3.12.1 | packaged by conda-forge | (main) [MSC v.1937]", exe) == "conda"
    store = r"C:\Users\x\AppData\Local\Microsoft\WindowsApps\python.exe"
    assert wp.python_build_kind("3.12.7 (tags/v3.12.7:0b05ead) [MSC]", store) == "Microsoft Store"


def test_parse_core_info_counts_cores_smt_and_efficiency_classes():
    def record(relationship, flags=0, efficiency=0, size=48):
        return struct.pack("<IIBB", relationship, size, flags, efficiency) + b"\0" * (size - 10)
    raw = record(0, 1, 1) + record(0, 1, 1) + record(2, size=80) + record(0, 0, 0) + record(0, 0, 0) + record(0, 0, 0)
    assert wp.parse_core_info(raw) == (5, 2, {1: 2, 0: 3})
    assert wp.parse_core_info(b"") == (0, 0, {})


def test_citool_and_smart_app_control():
    text = ("Policy ID: {a}\r\nFriendly Name: VerifiedAndReputableDesktopEvaluation\r\nIs Currently Enforced: true\r\n\r\n"
            "Policy ID: {b}\r\nFriendly Name: Microsoft Windows Driver Policy\r\nIs Currently Enforced: true\r\n\r\n"
            "Policy ID: {c}\r\nFriendly Name: VerifiedAndReputableDesktop\r\nIs Currently Enforced: false\r\n")
    policies = wp.parse_citool(text)
    assert len(policies) == 3 and policies[2] == ("VerifiedAndReputableDesktop", False)
    assert wp.sac_from_citool(policies) == "Evaluation"
    assert wp.sac_from_citool([("VerifiedAndReputableDesktop", True)]) == "On (enforcement)"
    assert wp.sac_from_citool([]) is None
    assert wp.SAC_STATES == {0: "Off", 1: "On (enforcement)", 2: "Evaluation"}
    assert wp.decode_output("a: b".encode("utf-16-le")) == "a: b"


def test_parse_nvidia_smi():
    gpus, cuda = wp.parse_nvidia_smi("NVIDIA GeForce RTX 4070, 12282 MiB, 581.57\n",
                                     "| NVIDIA-SMI 581.57   Driver Version: 581.57   CUDA Version: 13.0 |")
    assert gpus == [{"name": "NVIDIA GeForce RTX 4070", "memory": "12282 MiB", "driver": "581.57"}] and cuda == "13.0"
    assert wp.parse_nvidia_smi("", "") == ([], None)


def test_duplicate_names_and_loopback_inputs():
    devices = [{"index": 1, "name": "Microphone (Realtek(R) Audio)", "hostapi": "MME", "in": 2},
               {"index": 5, "name": "Microphone (Realtek(R) Audio)", "hostapi": "Windows WASAPI", "in": 2},
               {"index": 6, "name": "Stereo Mix (Realtek(R) Audio)", "hostapi": "Windows WASAPI", "in": 2},
               {"index": 7, "name": "Speakers (Realtek(R) Audio)", "hostapi": "Windows WASAPI", "in": 0, "out": 2}]
    assert wp.same_name_groups(devices) == {"microphone (realtek(r) audio)": ["MME", "Windows WASAPI"]}
    assert [d["index"] for d in wp.loopback_inputs(devices)] == [6]


def test_scrubber_replaces_folders_and_names():
    env = {"USERPROFILE": r"C:\Users\Anna", "LOCALAPPDATA": r"C:\Users\Anna\AppData\Local", "USERNAME": "Anna",
           "COMPUTERNAME": "ANNA-PC", "OneDrive": r"C:\Users\Anna\OneDrive - Contoso"}
    scrub = wp.make_scrubber(env=env, home=r"C:\Users\Anna", names=["Petrova"])
    assert scrub(r"C:\Users\Anna\AppData\Local\Meeting Transcriber") == r"%LOCALAPPDATA%\Meeting Transcriber"
    assert scrub(r"c:\users\anna\OneDrive - Contoso\Documents") == r"%OneDrive%\Documents"
    assert scrub("C:/Users/Anna/x.wav") == "%USERPROFILE%/x.wav"
    assert scrub("Anna's AirPods, Petrova, ANNA-PC") == "%USERNAME%'s AirPods, %USERNAME%, %COMPUTERNAME%"
    assert scrub({"k": ["Annabel", 3]}) == {"k": ["Annabel", 3]}               # whole words only
    unix = wp.make_scrubber(env={"USER": "ro"}, home="/root")
    # The home folder's label follows the machine the probe runs on (%USERPROFILE% on Windows); what counts here is that
    # only the whole folder is replaced, not the start of /rootfs.
    assert unix("/root/x /rootfs/y") == ("%USERPROFILE%" if wp.WIN else "~") + "/x /rootfs/y"


def test_child_reports_a_missing_package_and_an_error(monkeypatch, capsys):
    def missing():
        raise ModuleNotFoundError("No module named 'sounddevice'")

    def broken():
        raise RuntimeError("no device")
    monkeypatch.setitem(wp.CHILD_TASKS, "missing", missing)
    monkeypatch.setitem(wp.CHILD_TASKS, "broken", broken)
    for task, expected in (("missing", {"ok": False, "missing": True}), ("broken", {"ok": False})):
        wp.child_main(task, "{}")
        line = capsys.readouterr().out.strip().splitlines()[-1]
        assert line.startswith(wp.CHILD_TAG)
        result = json.loads(line[len(wp.CHILD_TAG):])
        assert {k: result[k] for k in expected} == expected and "error" in result
        assert ("missing" in result) == (task == "missing")


def test_child_process_encoding_task():
    r = wp.child("encoding", {"text": "@@LINE\tПривет"}, 60, env={"PYTHONIOENCODING": "utf-8"})
    assert r["ok"] and r["writable"] is True and r["encoding"].lower().replace("-", "") == "utf8"


def test_run_turns_outcomes_into_statuses():
    probe = wp.Probe(Namespace(ci=True, interactive=False, cuda_load=False, fetch_ffmpeg=False), lambda v: v)
    probe.section("t", "test")

    def raise_(e):
        raise e
    assert probe.run("a", "ok", lambda: ("fine", {"x": 1}))["status"] == "ok"
    assert probe.run("b", "skip", lambda: raise_(wp.Skip("n/a")))["status"] == "skipped"
    assert probe.run("c", "fail", lambda: raise_(wp.Fail("bad")))["status"] == "failed"
    assert probe.run("d", "os", lambda: raise_(PermissionError("denied")))["status"] == "failed"
    assert probe.run("e", "bug", lambda: raise_(KeyError("x")))["status"] == "error"
    began = time.monotonic()
    slow = probe.run("f", "slow", lambda: time.sleep(5), timeout=0.5)
    assert slow["status"] == "failed" and "timed out" in slow["detail"] and time.monotonic() - began < 3
    assert probe.data("a") == {"x": 1} and probe.data("c") == {}


def entry(data, status="ok", detail=""):
    return {"status": status, "data": data, "detail": detail}


def test_summary_of_a_windows_11_arm_pc_with_smart_app_control_on():
    r = {"os.version": entry({"windows": True, "windows11": True, "build": 26100, "process_loopback": True}),
         "os.arch": entry({"emulated": True, "native": "Arm64", "python_arch": "x64"}),
         "os.memory": entry({"total_gb": 8.0}),
         "os.disk": entry({"LOCALAPPDATA": {"drive": "C:", "free_gb": 5.0}}),
         "py.encoding": entry({"pipe_as_is": {"encoding": "cp1252", "writable": "UnicodeEncodeError: ..."}}),
         "py.vcruntime": entry({}, "failed", "msvcp140.dll MISSING"),
         "sec.sac": entry({"state": 1}), "sec.cfa": entry({"state": 1}), "sec.documents": entry({"onedrive": True}),
         "sec.microphone": entry({}, "failed", "blocked"),
         "audio.default": entry({"frames": 48000, "peak": 0.3}),
         "audio.wasapi_auto": entry({}), "audio.wasapi_plain": entry({}, "failed", "Invalid sample rate"),
         "audio.inputs": entry({"duplicates": {"mic": ["MME", "Windows WASAPI"]}, "loopback_inputs": []}),
         "loop.pyaudiowpatch.silent": entry({"frames": 0, "expected_frames": 144000, "peak": 0.0}),
         "loop.soundcard.playing": entry({"peak": 0.2}),
         "out.formfactor": entry({"defaults": {"render/console": {"name": "Speakers", "form_factor": "Speakers",
                                                                   "kind": "speaker"}}}),
         "gpu.smi": entry({"gpus": [{"name": "RTX"}]}),
         "gpu.whisper": entry({"runs": {"cpu": {"ok": True}, "cuda as is": {"ok": False},
                                        "cuda + preload by full path": {"ok": True}}}),
         "ffmpeg": entry({}, "failed", "not found")}
    text = "\n".join(wp.summarize(r))
    for expected in ("Windows 11 (build 26100): process loopback is available", "under emulation on an Arm64 PC",
                     "8.0 GB of memory", "Only 5.0 GB free", "PYTHONIOENCODING=utf-8", "Visual C++ runtime is missing",
                     "Smart App Control is On", "Controlled folder access is On", "synced by OneDrive",
                     "microphone is blocked", "works as mt.py opens it", "only with auto_convert=True",
                     "several host APIs", "0 % of the frames", "captured through soundcard", "would untick",
                     "GPU usable: Whisper ran on CUDA float16 (cuda + preload by full path)", "ffmpeg not found"):
        assert expected in text, expected


def test_summary_when_the_gpu_fails_and_windows_10():
    r = {"os.version": entry({"windows": True, "windows11": False, "build": 19045, "process_loopback": False}),
         "gpu.smi": entry({"gpus": [{"name": "RTX"}]}),
         "gpu.whisper": entry({"runs": {"cpu": {"ok": True}, "cuda as is": {"ok": False}}}),
         "sec.sac": entry({"state": 0}), "x": entry({}, "error", "KeyError")}
    text = "\n".join(wp.summarize(r))
    assert "no process loopback" in text and "Smart App Control is Off" in text
    assert "falls back to the CPU (int8) with a note" in text and "CUDA_VISIBLE_DEVICES" not in text, "mt.py's own fallback"
    assert "bug in this probe" in text and "Loopback was not tested" in text
