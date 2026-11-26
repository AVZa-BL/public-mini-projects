#!/usr/bin/env python3
"""Windows probe for meeting-transcriber: checks once what a Windows version of the app depends on, on this PC, and
prints a report to send back. The Windows counterpart of app/diagnose_tap.swift (PR W1 of the Windows plan): the
operating system and CPU, this Python, the security features that stop an unsigned app, the microphone through
sounddevice, the capture of the computer's sound, the kind of output device, the NVIDIA GPU and ffmpeg.

    python tools\\win_probe.py                    every check, no questions (recordings take whatever is heard)
    python tools\\win_probe.py --interactive      asks you to speak or play sound at the right moments, and loads
                                                 Whisper "tiny" on the GPU (downloads it once, about 75 MB)
    python tools\\win_probe.py --out probe.json   also writes the report as a JSON file (send this file)
    python tools\\win_probe.py --cuda-load        loads Whisper "tiny" on the GPU without the questions
    python tools\\win_probe.py --ci               for the windows-latest runner: no questions, 1-s recordings, short
                                                 timeouts; exits 0 without sound devices or GPU (--strict: 1 if a check
                                                 hit a bug in this script)

Python 3.9 or newer, standard library only at import time. Every optional package (sounddevice, numpy, soundcard,
pyaudiowpatch, pycaw/comtypes, ctranslate2, faster_whisper) is imported inside a check, most of them in a child process
of this script: a missing package is a "skipped" line, a driver that hangs is ended at the check's timeout, and a DLL
that crashes the interpreter ends only that child. Nothing is uploaded. The only downloads are the optional ones above
and ffmpeg with --fetch-ffmpeg. In the report the home folder reads %USERPROFILE%, and the account, computer and display
names are replaced wherever they appear (a Bluetooth headset named after its owner, for example). Read it before sending.
"""
import argparse
import json
import locale
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

PROBE_VERSION = "1"
WIN = sys.platform == "win32"
HERE = Path(__file__).resolve().parent
MT_DIR = HERE.parent
RATE = 16000            # mt.py:21, the rate every track is recorded at
SILENT_PEAK = 0.002     # mt.py:74: a track whose loudest sample stays below this was silent
CHILD_TAG = "@@PROBE\t"
CREATE_NO_WINDOW = 0x08000000

# Windows builds that decide what the app can do. Windows 11 starts at 22000. Process loopback (capturing every process
# but the app's own tree, the counterpart of the Mac's process tap) needs "Windows 10 Build 20348" per
# https://learn.microsoft.com/en-us/windows/win32/api/audioclientactivationparams/ns-audioclientactivationparams-audioclient_process_loopback_params
# (the ActivateAudioInterfaceAsync page says 20438; an OBS plug-in reports it working on some Windows 10 2004+ builds).
WIN11_BUILD = 22000
PROCESS_LOOPBACK_BUILD = 20348

# EndpointFormFactor, in the order of the enum (RemoteNetworkDevice = 0 ...):
# https://learn.microsoft.com/en-us/windows/win32/api/mmdeviceapi/ne-mmdeviceapi-endpointformfactor
FORM_FACTORS = ["RemoteNetworkDevice", "Speakers", "LineLevel", "Headphones", "Microphone", "Headset", "Handset",
                "UnknownDigitalPassthrough", "SPDIF", "DigitalAudioDisplayDevice", "UnknownFormFactor"]
# Property keys as pycaw names them (str(PROPERTYKEY) = "{FMTID} pid", upper case). FormFactor is pid 0 and JackSubType
# pid 8 of {1da5d803-...} (mmdeviceapi.idl; van Eerde, https://matthewvaneerde.wordpress.com/2019/04/17/how-can-my-
# application-tell-what-kind-of-audio-device-its-using). FriendlyName 14 and DeviceDesc 2 are pycaw's own
# (pycaw/utils.py); EnumeratorName 24 (the bus: USB, BTHENUM, HDAUDIO) and the interface name are from
# functiondiscoverykeys_devpkey.h, not re-read [I]: a key that is absent is reported as missing, nothing else.
PKEY_FORM_FACTOR = "{1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E} 0"
PKEY_JACK_SUBTYPE = "{1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E} 8"
PKEY_DEVICE_DESC = "{A45C254E-DF1C-4EFD-8020-67D146A850E0} 2"
PKEY_ENUMERATOR = "{A45C254E-DF1C-4EFD-8020-67D146A850E0} 24"
PKEY_INTERFACE_NAME = "{026E516E-B814-414B-83CD-856D6FEF4822} 2"

# Smart App Control: HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy, VerifiedAndReputablePolicyState. Microsoft's
# "Test App Signatures with Smart App Control" (updated 2026-09-21) gives the values: "0 Off, 1 On (Enforcement),
# 2 Evaluation", and `citool.exe -lp` as the check (Friendly Name VerifiedAndReputableDesktop = enforcement,
# VerifiedAndReputableDesktopEvaluation = evaluation):
# https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/test-your-app-with-smart-app-control
SAC_STATES = {0: "Off", 1: "On (enforcement)", 2: "Evaluation"}
# Controlled folder access, (Get-MpPreference).EnableControlledFolderAccess: "0 Disabled, 1 Enabled (block), 2 Audit
# Mode, 3 Block disk modification only, 4 Audit disk modification only"
# (https://learn.microsoft.com/en-us/defender-endpoint/controlled-folder-access-overview).
CFA_STATES = {0: "Disabled", 1: "Enabled (block)", 2: "Audit mode", 3: "Block disk modification only",
              4: "Audit disk modification only"}
# Microphone privacy for unpackaged (desktop) apps. Microsoft documents only the settings path, "Settings > Privacy &
# security > Microphone > Let desktop apps access your microphone"
# (https://support.microsoft.com/en-us/windows/privacy/turn-on-app-permissions-for-your-microphone-in-windows), not the
# registry. The keys below are the ones community guides write the switches to, a REG_SZ "Value" of Allow or Deny
# (https://www.elevenforum.com/t/enable-or-disable-apps-access-to-microphone-in-windows-11.17117/,
# https://geekrewind.com/how-to-turn-microphone-access-for-apps-on-or-off-in-windows-11/) [secondary sources]:
CONSENT = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
MIC_SWITCHES = [("HKEY_LOCAL_MACHINE", CONSENT, "Microphone access (for the whole PC)"),
                ("HKEY_CURRENT_USER", CONSENT, "Let apps access your microphone"),
                ("HKEY_CURRENT_USER", CONSENT + r"\NonPackaged", "Let desktop apps access your microphone")]
# The policy that overrides them: LetAppsAccessMicrophone, "0 User in control, 1 Force allow, 2 Force deny", Group Policy
# key Software\Policies\Microsoft\Windows\AppPrivacy (https://learn.microsoft.com/en-us/windows/client-management/mdm/
# policy-csp-privacy). The value names under that key are listed as found, not guessed.
APP_PRIVACY = r"SOFTWARE\Policies\Microsoft\Windows\AppPrivacy"
POLICY_STATES = {0: "user in control", 1: "force allow", 2: "force deny"}
FOLDERID_DOCUMENTS = "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}"
# IsProcessorFeaturePresent numbers (https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/
# nf-processthreadsapi-isprocessorfeaturepresent). The AVX ones are recognised from Windows 10 2004 on; before that the
# call returns 0 whatever the CPU has, as it does for any feature "the HAL does not support detection" of.
CPU_FEATURES = {"AVX": 39, "AVX2": 40, "AVX512F": 41, "ARM_V8": 29, "ARM_V82_DP": 43}
MACHINES = {0x014C: "x86", 0x8664: "x64", 0xAA64: "Arm64", 0x01C4: "Arm (32-bit)", 0: "unknown"}
CUDA_DLLS = ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"]


class Skip(Exception):
    """The check does not apply here (not Windows, a package not installed): reported as skipped, with the reason."""


class Fail(Exception):
    """The check ran and found a problem: reported as failed, with the reason."""


# ---------- small helpers (pure, tested on Linux) ----------

def levels(samples, full_scale):
    """RMS and peak of a sequence of samples as a fraction of full scale (0..1), and both in dBFS."""
    n, squares, peak = 0, 0.0, 0.0
    for s in samples:
        n += 1
        squares += s * s
        peak = max(peak, abs(s))
    if not n:
        return {"samples": 0, "rms": 0.0, "peak": 0.0, "rms_dbfs": None, "peak_dbfs": None}
    rms, peak = math.sqrt(squares / n) / full_scale, peak / full_scale
    db = lambda x: round(20 * math.log10(x), 1) if x > 0 else None
    return {"samples": n, "rms": round(rms, 5), "peak": round(peak, 5), "rms_dbfs": db(rms), "peak_dbfs": db(peak)}


def output_kind(form_factor):
    """The Mac app's OutputKind for a Windows form factor (windows-platform research, section 3): a speaker means the
    microphone hears both sides, so the app unticks the computer's-sound box; headphones keep it ticked."""
    if form_factor in ("Speakers", "DigitalAudioDisplayDevice", "SPDIF"):
        return "speaker"
    if form_factor in ("Headphones", "Headset", "Handset"):
        return "headphones"
    return "unknown"


def python_build_kind(version, executable, has_conda_meta=False):
    """Where this Python comes from, from sys.version. python.org's Windows builds (and the Store's, built from the same
    release) name the release tag, "(tags/v3.12.x:<hash>, ...)" [I: from python.org builds, not re-read]; the
    python-build-standalone runtime the Mac app bundles is built from a branch called main ("main" and "MSC v.1944" are
    the build strings in python312.dll of cpython-3.12.16+20261026-x86_64-pc-windows-msvc, read with strings) [C]."""
    low = version.lower()
    if has_conda_meta or "conda" in low or "anaconda" in low:
        return "conda"
    if "\\windowsapps\\" in executable.lower():
        return "Microsoft Store"
    if "(tags/v" in version:
        return "python.org"
    if "(main," in version and "MSC" in version:
        return "python-build-standalone"
    return "other"


def parse_core_info(raw):
    """Physical cores from GetLogicalProcessorInformationEx(RelationProcessorCore): a run of variable-size records, each
    Relationship (DWORD), Size (DWORD), then PROCESSOR_RELATIONSHIP's Flags (LTP_PC_SMT = 1) and EfficiencyClass (higher
    = faster core). Returns (cores, cores with SMT, {efficiency class: cores})."""
    cores, smt, classes, offset = 0, 0, {}, 0
    while offset + 10 <= len(raw):
        relationship, size = struct.unpack_from("<II", raw, offset)
        if relationship == 0:
            flags, efficiency = struct.unpack_from("<BB", raw, offset + 8)
            cores += 1
            smt += bool(flags & 1)
            classes[efficiency] = classes.get(efficiency, 0) + 1
        if size <= 0:
            break
        offset += size
    return cores, smt, classes


def parse_citool(text):
    """`citool.exe -lp` prints one block of "Key: value" lines per code-integrity policy [I: format from Microsoft's
    page, not seen here]. Returns [(friendly name, currently enforced)] for every policy that names a friendly name."""
    policies = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "")):
        fields = {}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip().lower()] = value.strip()
        if fields.get("friendly name"):
            policies.append((fields["friendly name"], fields.get("is currently enforced", "").lower() == "true"))
    return policies


def sac_from_citool(policies):
    """Smart App Control's mode from the enforced code-integrity policies, by Microsoft's two friendly names."""
    names = {name for name, enforced in policies if enforced}
    if any(n.startswith("VerifiedAndReputableDesktopEvaluation") for n in names):
        return "Evaluation"
    if "VerifiedAndReputableDesktop" in names:
        return "On (enforcement)"
    return None


def parse_nvidia_smi(csv_text, header_text=""):
    """`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` lines, and the CUDA version the
    driver supports from the plain `nvidia-smi` header ("CUDA Version: 12.8")."""
    gpus = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", header_text)
    return gpus, cuda.group(1) if cuda else None


def decode_output(raw):
    """A Windows tool's output: some write UTF-16 to a pipe (NUL bytes between the letters), most the console's code
    page; the citool output format is not known here, so both are handled."""
    if not raw:
        return ""
    if b"\x00" in raw[:200]:
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("mbcs" if WIN else "latin-1", errors="replace")


def same_name_groups(devices):
    """Input devices whose name (cut to MME's 31 characters, as MME truncates them) appears under several host APIs:
    the reason a name given to `mt.py record --device` matches several devices on Windows (mtpy-windows #6)."""
    groups = {}
    for d in devices:
        if d.get("in", 0) > 0:
            groups.setdefault(d["name"][:31].strip().lower(), set()).add(d["hostapi"])
    return {name: sorted(apis) for name, apis in groups.items() if len(apis) > 1}


LOOPBACK_WORDS = ("loopback", "stereo mix", "what u hear", "wave out mix", "cable output", "voicemeeter out")


def loopback_inputs(devices):
    """Input devices that carry the computer's sound: a vendor "Stereo Mix", a virtual cable, a PortAudio built with
    WASAPI loopback ("[Loopback]"). With one of these, `record --system-device N` works on Windows as it is."""
    return [d for d in devices if d.get("in", 0) > 0 and any(w in d["name"].lower() for w in LOOPBACK_WORDS)]


def without_ps_module_path(env):
    """The environment for the Windows PowerShell (5.1) child without PSModulePath, whatever its case. Started from
    PowerShell 7 (as on the CI runner, whose steps run in pwsh), the child would inherit PowerShell 7's module path and
    fail to load Microsoft.PowerShell.Security ("the module could not be loaded" for Get-AuthenticodeSignature); without
    the variable it builds its own default one. dict(os.environ) on Windows has its names in upper case (PSMODULEPATH),
    so a pop of "PSModulePath" alone removed nothing: run 45805995625 still showed the failure."""
    return {k: v for k, v in env.items() if k.upper() != "PSMODULEPATH"}


def make_scrubber(env=None, home=None, names=()):
    """Replaces the user's folders and names in every string of the report. Folders first, longest first (they hold the
    names); then the account name, the profile folder's name, the computer name and the display name's words, as whole
    words, case-insensitive, three characters or more (a shorter one would hit ordinary words)."""
    env = os.environ if env is None else env
    home = home if home is not None else os.path.expanduser("~")
    folders = []
    for var, label in (("LOCALAPPDATA", "%LOCALAPPDATA%"), ("APPDATA", "%APPDATA%"), ("OneDriveCommercial", "%OneDrive%"),
                       ("OneDriveConsumer", "%OneDrive%"), ("OneDrive", "%OneDrive%"), ("USERPROFILE", "%USERPROFILE%")):
        if len(env.get(var, "")) > 3:
            folders.append((env[var].rstrip("\\/"), label))
    if len(home) > 3:
        folders.append((home.rstrip("\\/"), "%USERPROFILE%" if WIN or env.get("USERPROFILE") else "~"))
    patterns = []
    for folder, label in sorted(folders, key=lambda f: -len(f[0])):
        for variant in {folder, folder.replace("\\", "/")}:
            # Only the whole folder: /root, not the start of /rootfs.
            patterns.append((re.compile(re.escape(variant) + r"(?![\w.-])", re.IGNORECASE), label))
    words = set(names)
    for var in ("USERNAME", "USER", "COMPUTERNAME"):
        words.add(env.get(var, ""))
    words.add(os.path.basename(home.rstrip("\\/")))
    for word in sorted((w for w in words if len(w) >= 3), key=len, reverse=True):
        label = "%COMPUTERNAME%" if word == env.get("COMPUTERNAME") else "%USERNAME%"
        patterns.append((re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE), label))

    def scrub(value):
        if isinstance(value, str):
            for pattern, label in patterns:
                value = pattern.sub(label.replace("\\", "\\\\"), value)
            return value
        if isinstance(value, dict):
            return {scrub(k): scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        return value
    return scrub


# ---------- Windows helpers (ctypes, winreg), imported only when called ----------

def kernel32():
    import ctypes
    return ctypes.WinDLL("kernel32", use_last_error=True)


def reg(root, path, name):
    """One registry value, or None when the key or value is absent (PermissionError is passed on)."""
    import winreg
    try:
        with winreg.OpenKey(getattr(winreg, root), path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            return winreg.QueryValueEx(key, name)[0]
    except FileNotFoundError:
        return None


def reg_values(root, path):
    import winreg
    try:
        with winreg.OpenKey(getattr(winreg, root), path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            out, i = {}, 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                except OSError:
                    return out
                out[name], i = value, i + 1
    except FileNotFoundError:
        return None


def reg_subkeys(root, path):
    import winreg
    try:
        with winreg.OpenKey(getattr(winreg, root), path, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            out, i = [], 0
            while True:
                try:
                    out.append(winreg.EnumKey(key, i))
                except OSError:
                    return out
                i += 1
    except FileNotFoundError:
        return []


def module_path(handle):
    """The file a loaded module came from (GetModuleFileNameW): which copy of a DLL Windows actually picked."""
    import ctypes
    from ctypes import wintypes
    k32 = kernel32()
    k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    buf = ctypes.create_unicode_buffer(32768)
    return buf.value if k32.GetModuleFileNameW(handle, buf, 32768) else "?"


def file_version(path):
    """The FILEVERSION of a DLL or EXE (version.dll), e.g. 14.44.35211.0 for a current msvcp140.dll."""
    import ctypes
    from ctypes import wintypes
    ver = ctypes.WinDLL("version")
    ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(wintypes.UINT)]
    size = ver.GetFileVersionInfoSizeW(path, None)
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if not ver.GetFileVersionInfoW(path, 0, size, buf) or not ver.VerQueryValueW(buf, "\\", ctypes.byref(pointer),
                                                                                 ctypes.byref(length)):
        return None
    signature, _, ms, ls = struct.unpack("<IIII", ctypes.string_at(pointer.value, 16))
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}" if signature == 0xFEEF04BD else None


def known_folder(guid_text):
    """SHGetKnownFolderPath: where Windows really keeps a known folder (OneDrive's Known Folder Move redirects
    Documents into the OneDrive folder; %USERPROFILE%\\Documents then is not the folder Explorer shows)."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_ubyte * 8)]
    u = uuid.UUID(guid_text)
    g = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    shell32, ole32 = ctypes.WinDLL("shell32"), ctypes.WinDLL("ole32")
    shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(GUID), wintypes.DWORD, wintypes.HANDLE,
                                             ctypes.POINTER(ctypes.c_void_p)]
    shell32.SHGetKnownFolderPath.restype = ctypes.c_long
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    pointer = ctypes.c_void_p()
    hr = shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(pointer))
    try:
        if hr != 0:
            raise Fail(f"SHGetKnownFolderPath returned 0x{hr & 0xFFFFFFFF:08X}")
        return ctypes.wstring_at(pointer.value)
    finally:
        if pointer.value:
            ole32.CoTaskMemFree(pointer)


def display_name_words():
    """The words of the account's display name (GetUserNameExW, NameDisplay), only so that they can be replaced in the
    report; never printed. Local accounts often have none."""
    if not WIN:
        return []
    try:
        import ctypes
        from ctypes import wintypes
        secur32 = ctypes.WinDLL("secur32")
        size = wintypes.ULONG(256)
        buf = ctypes.create_unicode_buffer(256)
        if secur32.GetUserNameExW(3, buf, ctypes.byref(size)):
            return [w for w in re.split(r"[\s,.()]+", buf.value) if len(w) >= 3]
    except Exception:
        pass
    return []


def site_package_dirs():
    import site
    dirs = []
    for get in (lambda: site.getsitepackages(), lambda: [site.getusersitepackages()],
                lambda: [sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]]):
        try:
            dirs += [d for d in get() if d and os.path.isdir(d) and d not in dirs]
        except Exception:
            pass
    return dirs


def package_dir(name):
    """The folder of an installed package, without importing it (find_spec of a top-level name imports nothing)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    return os.path.dirname(spec.origin) if spec.origin else None


def dist_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def nvidia_bin_dirs():
    """site-packages\\nvidia\\<library>\\bin folders of the nvidia-*-cu12 wheels, and the DLLs each holds."""
    found = {}
    for sp in site_package_dirs():
        root = Path(sp) / "nvidia"
        if root.is_dir():
            for bin_dir in sorted(root.glob("*/bin")):
                found[str(bin_dir)] = sorted(p.name for p in bin_dir.glob("*.dll"))
    return found


# ---------- child processes ----------

def quiet():
    """subprocess.run arguments for a console program started from a process without a console (pythonw): without
    CREATE_NO_WINDOW each one opens a console window of its own. With a console the children share it, so Ctrl+C
    reaches them too."""
    if WIN:
        try:
            import ctypes
            if not ctypes.WinDLL("kernel32").GetConsoleWindow():
                return {"creationflags": CREATE_NO_WINDOW}
        except Exception:
            pass
    return {}


def child(task, args=None, timeout=30, env=None):
    """Runs one measurement in a fresh Python process (this file with --_child): a driver that hangs is ended at the
    timeout, a DLL that crashes the interpreter ends only the child, and a DLL loaded by one test cannot make the next
    one pass (Windows hands out a module already loaded under the same name). env: variables to set, None to remove."""
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    for key, value in (env or {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_child", task, json.dumps(args or {})]
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=environment, **quiet())
    except subprocess.TimeoutExpired:
        raise Fail(f"no answer within {timeout:g} s (the test process was ended)")
    for line in reversed(done.stdout.splitlines()):
        if line.startswith(CHILD_TAG):
            return json.loads(line[len(CHILD_TAG):])
    code = done.returncode
    code_text = f"{code} (0x{code & 0xFFFFFFFF:08X})" if code < 0 or code > 255 else str(code)
    tail = [ln for ln in (done.stderr or "").strip().splitlines() if ln.strip()][-3:]
    raise Fail(f"the test process ended with code {code_text} and no result" + (": " + " | ".join(tail) if tail else ""))


def child_result(task, args=None, timeout=30, package=None, env=None):
    """child(), with a package that is not installed as Skip and any exception in the child as Fail."""
    result = child(task, args, timeout, env)
    if result.get("missing"):
        hint = f" (pip install {package})" if package else ""
        raise Skip(f"not installed: {result['error']}{hint}")
    if not result.get("ok"):
        raise Fail(result.get("error", "failed"))
    return result


def task_encoding(text):
    """What mt.py's @@ events meet on a pipe: the stream's encoding, and whether a line of Cyrillic and Chinese text
    could be written with it (mtpy-windows #3: on a cp1252 pipe, print raises UnicodeEncodeError)."""
    encoding = sys.__stdout__.encoding
    try:
        text.encode(encoding)
        writable = True
    except UnicodeEncodeError as e:
        writable = f"UnicodeEncodeError: {e}"
    return {"encoding": encoding, "utf8_mode": sys.flags.utf8_mode, "writable": writable}


def task_sd_info():
    import sounddevice as sd
    apis = [{"index": i, "name": a["name"], "devices": len(a["devices"]), "default_input": a["default_input_device"],
             "default_output": a["default_output_device"]} for i, a in enumerate(sd.query_hostapis())]
    devices = [{"index": d["index"], "name": d["name"], "hostapi": apis[d["hostapi"]]["name"],
                "in": d["max_input_channels"], "out": d["max_output_channels"], "rate": d["default_samplerate"]}
               for d in sd.query_devices()]
    return {"version": sd.__version__, "portaudio": sd.get_portaudio_version()[1],
            "library": str(getattr(sd, "_libname", "?")), "hostapis": apis, "devices": devices,
            "default": list(sd.default.device)}


def task_sd_record(seconds, wasapi=False, auto_convert=None):
    """A microphone recording at 16 kHz mono int16, as mt.py opens it (mt.py:288), through a callback so that a device
    that delivers nothing ends as "0 frames" instead of a hang. wasapi: the WASAPI default input instead of PortAudio's
    default device (MME on Windows); auto_convert: WasapiSettings(auto_convert=...), which asks Windows to convert rate
    and channels (https://python-sounddevice.readthedocs.io/en/latest/api/platform-specific-settings.html)."""
    import sounddevice as sd
    from array import array
    device = None
    if wasapi:
        apis = [a for a in sd.query_hostapis() if "WASAPI" in a["name"]]
        if not apis:
            raise RuntimeError("no WASAPI host API in this PortAudio")
        device = apis[0]["default_input_device"]
        if device is None or device < 0:
            raise RuntimeError("WASAPI has no default input device")
    info = sd.query_devices(device, "input")
    extra = sd.WasapiSettings(auto_convert=True) if wasapi and auto_convert else None
    try:
        sd.check_input_settings(device=info["index"], channels=1, dtype="int16", extra_settings=extra, samplerate=RATE)
        supported = "yes"
    except Exception as e:
        supported = f"{type(e).__name__}: {e}"
    chunks, statuses = [], []

    def callback(indata, frames, at, status):
        if status:
            statuses.append(str(status))
        chunks.append(bytes(indata))
    began = time.monotonic()
    with sd.RawInputStream(samplerate=RATE, channels=1, dtype="int16", device=info["index"], extra_settings=extra,
                           callback=callback) as stream:
        opened = time.monotonic() - began
        rate, latency = stream.samplerate, stream.latency
        time.sleep(seconds)
    samples = array("h")
    for c in chunks:
        samples.frombytes(c)
    return {"device": info["name"], "index": info["index"], "hostapi": sd.query_hostapis(info["hostapi"])["name"],
            "device_rate": info["default_samplerate"], "format_supported": supported, "open_s": round(opened, 2),
            "rate": rate, "latency": latency, "frames": len(samples), "expected_frames": int(RATE * seconds),
            "status_flags": sorted(set(statuses))[:5], **levels(samples, 32768)}


def task_sc_loopback(seconds):
    """The computer's sound through SoundCard's WASAPI loopback: the default speaker opened as a microphone, 2 channels
    (SoundCard's known issue: "records garbage if you record only a single channel"). SoundCard pads with zeros when
    WASAPI delivers nothing, so the frame count is always full; the peak tells sound from silence."""
    import warnings
    import soundcard as sc
    speaker = sc.default_speaker()
    microphone = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        began = time.monotonic()
        with microphone.recorder(samplerate=48000, channels=2) as recorder:
            data = recorder.record(numframes=int(48000 * seconds))
        wall = time.monotonic() - began
    return {"version": dist_version("soundcard"), "speaker": speaker.name, "loopback": microphone.name,
            "frames": len(data), "wall_s": round(wall, 2), "warnings": sorted({str(w.message) for w in caught})[:3],
            **levels(data.ravel().tolist(), 1.0)}


def task_paw_loopback(seconds):
    """The computer's sound through PyAudioWPatch's loopback device of the default WASAPI output, in callback mode: the
    frames received against the frames expected show whether WASAPI loopback delivers data while nothing plays."""
    import pyaudiowpatch as pyaudio
    from array import array
    audio = pyaudio.PyAudio()
    try:
        loopback = audio.get_default_wasapi_loopback()
        count = sum(1 for _ in audio.get_loopback_device_info_generator())
        channels, rate = int(loopback["maxInputChannels"]) or 2, int(loopback["defaultSampleRate"])
        chunks = []

        def callback(data, frames, at, status):
            chunks.append(data)
            return (None, pyaudio.paContinue)
        stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                            input_device_index=loopback["index"], frames_per_buffer=1024, stream_callback=callback)
        stream.start_stream()
        time.sleep(seconds)
        stream.stop_stream()
        stream.close()
    finally:
        audio.terminate()
    samples = array("h")
    for c in chunks:
        samples.frombytes(c)
    return {"version": dist_version("PyAudioWPatch"), "loopback": loopback["name"], "loopback_devices": count,
            "channels": channels, "rate": rate, "frames": len(samples) // channels,
            "expected_frames": int(rate * seconds), **levels(samples, 32768)}


def task_formfactor():
    """The kind of the default output (and the other active outputs) through pycaw: PKEY_AudioEndpoint_FormFactor and
    the properties that help when it says Speakers for a headphone jack (combo jacks) or Unknown for Bluetooth."""
    from pycaw.utils import AudioUtilities
    from pycaw.constants import DEVICE_STATE, EDataFlow, ERole

    def describe(device):
        props = {str(k).upper(): v for k, v in device.properties.items()}
        factor = props.get(PKEY_FORM_FACTOR)
        name = FORM_FACTORS[factor] if isinstance(factor, int) and 0 <= factor < len(FORM_FACTORS) else factor
        return {"name": device.FriendlyName, "form_factor": name, "kind": output_kind(name),
                "description": props.get(PKEY_DEVICE_DESC), "adapter": props.get(PKEY_INTERFACE_NAME),
                "bus": props.get(PKEY_ENUMERATOR), "jack_subtype": props.get(PKEY_JACK_SUBTYPE)}
    enumerator = AudioUtilities.GetDeviceEnumerator()
    defaults = {}
    for flow, role in (("eRender", "eConsole"), ("eRender", "eMultimedia"), ("eRender", "eCommunications"),
                       ("eCapture", "eConsole")):
        try:
            endpoint = enumerator.GetDefaultAudioEndpoint(getattr(EDataFlow, flow).value, getattr(ERole, role).value)
            defaults[f"{flow[1:].lower()}/{role[1:].lower()}"] = describe(AudioUtilities.CreateDevice(endpoint))
        except Exception as e:
            defaults[f"{flow[1:].lower()}/{role[1:].lower()}"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        outputs = AudioUtilities.GetAllDevices(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)
    except TypeError:                                  # older pycaw: no filter arguments
        outputs = [d for d in AudioUtilities.GetAllDevices() if "ACTIVE" in str(d.state).upper()]
    return {"version": dist_version("pycaw"), "defaults": defaults, "outputs": [describe(d) for d in outputs if d]}


def task_ct2_info():
    import ctranslate2
    types = {}
    for device in ("cpu", "cuda"):
        try:
            types[device] = sorted(ctranslate2.get_supported_compute_types(device))
        except Exception as e:
            types[device] = f"{type(e).__name__}: {e}"
    folder = os.path.dirname(ctranslate2.__file__)
    return {"version": ctranslate2.__version__, "cuda_devices": ctranslate2.get_cuda_device_count(),
            "compute_types": types, "dlls": sorted(p.name for p in Path(folder).glob("*.dll"))}


def task_nvcuda():
    """The NVIDIA driver's own CUDA entry point, without any toolkit: nvcuda.dll in System32 (installed by the driver)."""
    import ctypes
    lib = ctypes.WinDLL("nvcuda.dll")
    version, count = ctypes.c_int(), ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(version))
    status = lib.cuInit(0)
    if status == 0:
        lib.cuDeviceGetCount(ctypes.byref(count))
    return {"path": module_path(lib._handle), "cuda": f"{version.value // 1000}.{version.value % 1000 // 10}",
            "init": status, "devices": count.value}


ADDED_DLL_DIRS = []      # the handles os.add_dll_directory returns: a folder stays added until close() (no __del__)


def prepare_dlls(strategy, dirs):
    """Makes the nvidia wheels' DLL folders reachable in one of the ways mt.py could (W3), before anything loads them."""
    import ctypes
    if strategy.startswith("add_dll_directory"):
        for d in dirs:
            ADDED_DLL_DIRS.append(os.add_dll_directory(d))
    elif strategy.startswith("path"):
        os.environ["PATH"] = os.pathsep.join(list(dirs) + [os.environ.get("PATH", "")])
    elif strategy == "preload":
        # Every DLL of the cuBLAS and cuDNN folders by full path, cuBLASLt before cuBLAS: once loaded, a later
        # LoadLibrary("cublas64_12.dll") inside ctranslate2 gets the loaded module instead of searching.
        for d in dirs:
            for p in sorted(Path(d).glob("*.dll"), key=lambda p: (not p.name.startswith("cublasLt"), p.name)):
                try:
                    ctypes.WinDLL(str(p))
                except OSError:
                    pass


def task_dll_load(strategy, dirs, names):
    """Loads the CUDA libraries ctranslate2 needs by name, after making their folders reachable one way. "python" is
    ctypes' default search (application folder, System32, os.add_dll_directory folders); "standard" (winmode=0) is the
    classic order with PATH, the one a plain LoadLibrary in native code such as ctranslate2.dll uses [I]."""
    import ctypes
    prepare_dlls(strategy, dirs)
    results = {}
    for name in names:
        target = name
        if strategy == "full_path":
            target = next((str(Path(d) / name) for d in dirs if (Path(d) / name).exists()), None)
            if target is None:
                results[name] = "not in the nvidia folders"
                continue
        try:
            lib = ctypes.WinDLL(target, winmode=0) if strategy.endswith("standard") else ctypes.WinDLL(target)
            results[name] = "loaded from " + module_path(lib._handle)
        except OSError as e:
            results[name] = f"failed: {e}"
    return {"results": results}


def task_whisper(strategy, dirs, device, compute_type):
    """Whisper "tiny" through faster-whisper, as mt.py loads its model (get_model), decoding 1 s of silence: CTranslate2
    loads cuBLAS and runs its first CUDA kernels only when it first computes, so a missing cuBLAS shows here rather than
    in the load. (Whether the Windows build calls cuDNN at all is [U]: its ctranslate2.dll 4.9.1 names no cuDNN library.)"""
    prepare_dlls(strategy, dirs)
    import numpy as np
    from faster_whisper import WhisperModel
    began = time.monotonic()
    model = WhisperModel("tiny", device=device, compute_type=compute_type)
    loaded = time.monotonic()
    segments, info = model.transcribe(np.zeros(RATE, dtype=np.float32), beam_size=1)
    text = " ".join(s.text.strip() for s in segments)
    return {"load_s": round(loaded - began, 1), "decode_s": round(time.monotonic() - loaded, 2),
            "language": info.language, "text": text[:60]}


def task_ffmpeg_fetch(download_dir):
    from static_ffmpeg import run
    ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise(download_dir=download_dir)
    return {"ffmpeg": ffmpeg, "ffprobe": ffprobe}


CHILD_TASKS = {"encoding": task_encoding, "sd_info": task_sd_info, "sd_record": task_sd_record,
               "sc_loopback": task_sc_loopback, "paw_loopback": task_paw_loopback, "formfactor": task_formfactor,
               "ct2_info": task_ct2_info, "nvcuda": task_nvcuda, "dll_load": task_dll_load, "whisper": task_whisper,
               "ffmpeg_fetch": task_ffmpeg_fetch}


def child_main(task, args_json):
    """The child's side: one task, one tagged JSON line on stdout. A module that is not installed is reported as
    missing (ModuleNotFoundError), any other exception (a DLL that does not load raises plain ImportError) as an error."""
    try:
        result = {"ok": True, **CHILD_TASKS[task](**json.loads(args_json))}
    except ModuleNotFoundError as e:
        result = {"ok": False, "missing": True, "error": f"{type(e).__name__}: {e}"}
    except BaseException as e:                         # SystemExit or KeyboardInterrupt from a library as well
        result = {"ok": False, "error": f"{type(e).__name__}: {e}".strip()}
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(CHILD_TAG + json.dumps(result, default=str), flush=True)
    return 0


# ---------- the report ----------

class Probe:
    def __init__(self, args, scrub):
        self.args = args
        self.scrub = scrub
        self.sections = []
        self.results = {}        # check id -> entry as reported (scrubbed)
        self.raw = {}            # check id -> entry as measured: later checks use these paths and names
        self.ps_cache = None
        self.ps_lock = threading.Lock()
        self.interrupted = False
        ci = args.ci
        self.seconds = 1.0 if ci else 3.0
        self.t_fast = 5 if ci else 10              # registry, ctypes, file system
        self.t_child = 20 if ci else 30            # an import in a child process
        self.t_ps = 40 if ci else 60               # one PowerShell start with the Defender and CIM modules

    def section(self, sid, title):
        self.sections.append({"id": sid, "title": title, "checks": []})
        print(f"\n== {title}", flush=True)

    def run(self, cid, title, fn, timeout=None):
        """One check: fn returns a detail line, or (detail, data); raises Skip or Fail. An OSError (a Windows call that
        was refused, a program that is missing) or a subprocess timeout is a failure too; any other exception is a bug
        of this script: status "error". It runs in a thread so that a call that never returns becomes "timed out"."""
        timeout = timeout or self.t_fast
        entry = {"id": cid, "title": title, "status": "error", "detail": "", "data": {}, "seconds": None}
        if self.interrupted:
            entry.update(status="skipped", detail="interrupted (Ctrl+C)")
        else:
            box = {}

            def target():
                try:
                    out = fn()
                    detail, data = out if isinstance(out, tuple) else (out, {})
                    box.update(status="ok", detail=detail or "", data=data or {})
                except Skip as e:
                    box.update(status="skipped", detail=str(e))
                except Fail as e:
                    box.update(status="failed", detail=str(e))
                except (OSError, subprocess.SubprocessError) as e:
                    box.update(status="failed", detail=f"{type(e).__name__}: {e}")
                except Exception as e:
                    box.update(status="error", detail=f"{type(e).__name__}: {e}",
                               data={"traceback": traceback.format_exc().strip().splitlines()[-4:]})
            began = time.monotonic()
            worker = threading.Thread(target=target, daemon=True)
            worker.start()
            try:
                while worker.is_alive() and time.monotonic() - began < timeout:
                    worker.join(0.2)       # short waits: Ctrl+C stays responsive on Windows
            except KeyboardInterrupt:
                self.interrupted = True
                box = {"status": "skipped", "detail": "interrupted (Ctrl+C)"}
            if worker.is_alive() and not box:
                box = {"status": "failed", "detail": f"timed out after {timeout:g} s"}
            entry.update(box)
            entry["seconds"] = round(time.monotonic() - began, 2)
        self.raw[cid] = entry
        entry = self.scrub(entry)
        self.sections[-1]["checks"].append(entry)
        self.results[cid] = entry
        lines = str(entry["detail"]).splitlines() or [""]
        print(f"  {entry['status']:<8} {title}: {lines[0]}", flush=True)
        for line in lines[1:]:
            print(f"  {'':<8}   {line}", flush=True)
        return entry

    def ask(self, text):
        """Waits for Enter, only with --interactive at a terminal (never in --ci)."""
        if self.args.interactive and not self.args.ci and not self.interrupted and sys.stdin and sys.stdin.isatty():
            try:
                input(f"\n>> {text} ")
            except EOFError:
                pass
            except KeyboardInterrupt:          # the rest is reported as interrupted; the report is still printed
                self.interrupted = True

    def data(self, cid):
        """An earlier check's data as measured (real paths), {} unless it passed."""
        entry = self.raw.get(cid)
        return entry["data"] if entry and entry["status"] == "ok" else {}

    def powershell(self):
        """One PowerShell run for everything that needs it (Defender, CIM, services, Authenticode): PowerShell takes
        seconds to start. Statements one per line, read from stdin (no -EncodedCommand, which antivirus treats as
        suspicious); the files to check come in through an environment variable, so no path is quoted."""
        with self.ps_lock:
            if self.ps_cache is not None:
                return self.ps_cache
            exe = shutil.which("powershell") or os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                                              r"System32\WindowsPowerShell\v1.0\powershell.exe")
            error = '"$($_.Exception.Message)"'
            script = "\n".join([            # one PowerShell statement per line (adjacent strings join into one)
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
                "$ErrorActionPreference = 'Stop'",
                "$r = [ordered]@{}",
                "try { $r.cfa = [int](Get-MpPreference).EnableControlledFolderAccess } "
                f"catch {{ $r.cfa_error = {error} }}",
                "try { $s = Get-MpComputerStatus; $r.defender_mode = \"$($s.AMRunningMode)\"; "
                "$r.defender_realtime = [bool]$s.RealTimeProtectionEnabled } "
                f"catch {{ $r.defender_error = {error} }}",
                "try { $p = @(Get-CimInstance Win32_Processor)[0]; $r.cpu = [ordered]@{ name = \"$($p.Name)\".Trim(); "
                "cores = [int]$p.NumberOfCores; logical = [int]$p.NumberOfLogicalProcessors; "
                f"architecture = [int]$p.Architecture }} }} catch {{ $r.cpu_error = {error} }}",
                f"try {{ $r.audiosrv = \"$((Get-Service Audiosrv).Status)\" }} catch {{ $r.audiosrv_error = {error} }}",
                "$r.signatures = @()",
                "foreach ($f in ($env:MTPROBE_FILES -split '\\|')) { if ($f) { try { "
                "$g = Get-AuthenticodeSignature -LiteralPath $f; $r.signatures += [ordered]@{ file = $f; "
                "status = \"$($g.Status)\"; signer = \"$($g.SignerCertificate.Subject)\" } } "
                f"catch {{ $r.signatures += [ordered]@{{ file = $f; status = 'error'; signer = {error} }} }} }} }}",
                "$r | ConvertTo-Json -Depth 4 -Compress",
                ""])
            env = without_ps_module_path(dict(os.environ, MTPROBE_FILES="|".join(self.signature_files())))
            try:
                done = subprocess.run([exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", "-"], input=script,
                                      capture_output=True, encoding="utf-8", errors="replace", timeout=self.t_ps - 5,
                                      env=env, **quiet())
            except (OSError, subprocess.TimeoutExpired) as e:
                self.ps_cache = {"error": f"{type(e).__name__}: {e}"}
                return self.ps_cache
            found = [ln for ln in done.stdout.splitlines() if ln.strip().startswith("{")]
            try:
                self.ps_cache = json.loads(found[-1]) if found else {"error": (done.stderr or done.stdout).strip()[-300:]}
            except ValueError as e:
                self.ps_cache = {"error": f"unreadable PowerShell output ({e})"}
            return self.ps_cache

    def signature_files(self):
        """The executables Smart App Control would judge first: this Python, its DLL, and the native parts of the
        packages the engine loads (found without importing them)."""
        files = [getattr(sys, "_base_executable", sys.executable), sys.executable]
        try:
            files.append(module_path(sys.dllhandle))
        except Exception:
            pass
        # The 64-bit PortAudio that sounddevice loads (the folder also holds 32-bit and ASIO builds, first by name).
        for package, pattern in (("ctranslate2", "ctranslate2*.dll"),
                                 ("_sounddevice_data", "portaudio-binaries/libportaudio64bit.dll"),
                                 ("onnxruntime", "capi/*.pyd")):
            folder = package_dir(package)
            if folder:
                files += [str(p) for p in sorted(Path(folder).glob(pattern))[:1]]
        if shutil.which("ffmpeg"):
            files.append(shutil.which("ffmpeg"))
        unique = []
        for f in files:
            if f and os.path.exists(f) and os.path.normcase(f) not in [os.path.normcase(u) for u in unique]:
                unique.append(f)
        return unique


def need_windows():
    if not WIN:
        raise Skip(f"not Windows ({sys.platform})")


# ---------- 1. operating system and hardware ----------

def check_os_version():
    if not WIN:
        return f"{platform.system()} {platform.release()} (not Windows: the Windows checks below are skipped)", {
            "system": platform.system(), "release": platform.release(), "windows": False}
    v = sys.getwindowsversion()
    nt = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    product, display = reg("HKEY_LOCAL_MACHINE", nt, "ProductName"), reg("HKEY_LOCAL_MACHINE", nt, "DisplayVersion")
    ubr, edition = reg("HKEY_LOCAL_MACHINE", nt, "UBR"), reg("HKEY_LOCAL_MACHINE", nt, "EditionID")
    install = reg("HKEY_LOCAL_MACHINE", nt, "InstallationType")
    build = v.build
    server = (install or "").lower().startswith("server")          # the windows-latest runner is Server 2025
    windows11 = build >= WIN11_BUILD and not server
    # ProductName still says "Windows 10" on Windows 11 (the key was never renamed); the build decides.
    name = (product or "Windows").replace("Windows 10", "Windows 11") if windows11 else (product or "Windows")
    detail = (f"{name} {display or ''}, build {build}{'.' + str(ubr) if ubr is not None else ''} "
              f"(platform.version {platform.version()}); Windows 11: {'yes' if windows11 else 'no'} (build >= "
              f"{WIN11_BUILD}); process loopback: {'should be available' if build >= PROCESS_LOOPBACK_BUILD else 'no'}"
              f" (build >= {PROCESS_LOOPBACK_BUILD})")
    return detail, {"windows": True, "server": server, "build": build, "ubr": ubr, "major": v.major, "minor": v.minor,
                    "product": product, "display_version": display, "edition": edition, "installation": install,
                    "windows11": windows11, "process_loopback": build >= PROCESS_LOOPBACK_BUILD,
                    "platform_version": platform.version(), "win32_edition": getattr(platform, "win32_edition",
                                                                                     lambda: None)()}


def check_arch():
    python_platform = sysconfig.get_platform()           # win-amd64, win-arm64, win32: what this Python was built for
    data = {"python_platform": python_platform, "python_bits": struct.calcsize("P") * 8,
            "platform_machine": platform.machine(), "PROCESSOR_ARCHITECTURE": os.environ.get("PROCESSOR_ARCHITECTURE"),
            "PROCESSOR_ARCHITEW6432": os.environ.get("PROCESSOR_ARCHITEW6432")}
    if not WIN:
        return f"{platform.machine()}, Python {python_platform}", data
    import ctypes
    k32 = kernel32()
    native = None
    try:
        # IsWow64Process2 (Windows 10 1709+): the PC's own machine. An x64 process under emulation on Arm64 is not
        # WOW64, so its process machine reads "unknown" and only the native machine tells the emulation apart.
        process_machine, native_machine = ctypes.c_ushort(), ctypes.c_ushort()
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.IsWow64Process2.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
        if k32.IsWow64Process2(k32.GetCurrentProcess(), ctypes.byref(process_machine), ctypes.byref(native_machine)):
            native = MACHINES.get(native_machine.value, hex(native_machine.value))
            data["process_machine"] = MACHINES.get(process_machine.value, hex(process_machine.value))
    except (AttributeError, OSError) as e:
        data["IsWow64Process2"] = f"{type(e).__name__}: {e}"
    system_arch = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
                      "PROCESSOR_ARCHITECTURE")
    data["system_PROCESSOR_ARCHITECTURE"] = system_arch
    native = native or {"AMD64": "x64", "ARM64": "Arm64", "x86": "x86"}.get(system_arch or "", system_arch)
    own = {"win-amd64": "x64", "win-arm64": "Arm64", "win32": "x86"}.get(python_platform, python_platform)
    emulated = bool(native and own and native != own)
    data.update(native=native, python_arch=own, emulated=emulated)
    how = ""
    if emulated:
        how = " (x64 emulation, Windows 11 on Arm)" if native == "Arm64" and own == "x64" else " (WOW64)"
    return (f"PC {native or '?'}, this Python {own} {data['python_bits']}-bit, "
            f"emulated: {'yes' + how if emulated else 'no'}"), data


def check_cpu():
    data = {"logical": os.cpu_count()}
    if not WIN:
        return f"{os.cpu_count()} logical processors", data
    import ctypes
    k32 = kernel32()
    data["name"] = (reg("HKEY_LOCAL_MACHINE", r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                        "ProcessorNameString") or "?").strip()
    try:
        k32.GetLogicalProcessorInformationEx.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        size = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
        buf = ctypes.create_string_buffer(size.value)
        if size.value and k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            cores, smt, classes = parse_core_info(buf.raw[:size.value])
            data.update(cores=cores, cores_with_smt=smt, efficiency_classes={str(k): v for k, v in classes.items()})
    except (AttributeError, OSError) as e:
        data["cores_error"] = f"{type(e).__name__}: {e}"
    k32.IsProcessorFeaturePresent.argtypes = [ctypes.c_ulong]
    features = {name: bool(k32.IsProcessorFeaturePresent(number)) for name, number in CPU_FEATURES.items()}
    data["features"] = features
    classes = data.get("efficiency_classes", {})
    hybrid = ""
    if len(classes) > 1:
        top = max(classes, key=int)
        hybrid = f", hybrid: {classes[top]} performance + {sum(classes.values()) - classes[top]} efficiency cores"
    shown = [n for n in ("AVX", "AVX2", "AVX512F") if features.get(n)] or ["no AVX"]
    if features.get("ARM_V8"):
        shown = ["Arm v8"] + (["v8.2 dot product"] if features.get("ARM_V82_DP") else [])
    return (f"{data['name']}; {data.get('cores', '?')} cores, {data['logical']} logical processors{hybrid}; "
            f"{', '.join(shown)}"), data


def check_memory():
    if not WIN:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return f"{total / 2**30:.1f} GB", {"total_gb": round(total / 2**30, 1)}
        except (ValueError, OSError, AttributeError):
            raise Skip("not Windows, and the memory size is not available")
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not kernel32().GlobalMemoryStatusEx(ctypes.byref(status)):
        raise Fail(f"GlobalMemoryStatusEx failed: {ctypes.get_last_error()}")
    total, free = status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
    return f"{total:.1f} GB, {free:.1f} GB free ({status.dwMemoryLoad} % in use)", {
        "total_gb": round(total, 1), "free_gb": round(free, 1), "load_percent": status.dwMemoryLoad}


def check_disk():
    """Free space where the engine would live: %LOCALAPPDATA% (the plan puts the runtime, venv, models and MT_HOME
    there), and MT_HOME / HF_HOME when they are set elsewhere."""
    places = {"LOCALAPPDATA": os.environ["LOCALAPPDATA"]} if os.environ.get("LOCALAPPDATA") else {
        "home": os.path.expanduser("~")}
    for var in ("MT_HOME", "HF_HOME"):
        if os.environ.get(var):
            places[var] = os.environ[var]
    data, parts = {}, []
    for var, path in places.items():
        probe = Path(path)
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(str(probe))
        drive = os.path.splitdrive(str(probe))[0] or str(probe)
        data[var] = {"drive": drive, "free_gb": round(usage.free / 2**30, 1), "total_gb": round(usage.total / 2**30, 1)}
        parts.append(f"{var} on {drive}: {usage.free / 2**30:.1f} GB free of {usage.total / 2**30:.0f} GB")
    return "; ".join(parts), data


def check_language_and_paths():
    """The display language (the UI the user reads), long paths (MAX_PATH, mtpy-windows #16), Developer Mode (Hugging
    Face cache symlinks), and whether the profile folder's name is ASCII (open question: model paths with non-ASCII
    characters); the name itself is not printed."""
    need_windows()
    import ctypes
    k32 = kernel32()
    buf = ctypes.create_unicode_buffer(85)
    language = buf.value if k32.LCIDToLocaleName(k32.GetUserDefaultUILanguage(), buf, 85, 0) else "?"
    long_paths = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\FileSystem", "LongPathsEnabled")
    developer = reg("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock",
                    "AllowDevelopmentWithoutDevLicense")
    profile = os.environ.get("USERPROFILE", "")
    ascii_profile = all(ord(c) < 128 for c in profile)
    return (f"display language {language}; long paths {'on' if long_paths == 1 else 'off'}; Developer Mode "
            f"{'on' if developer == 1 else 'off'}; profile folder name ASCII: {'yes' if ascii_profile else 'no'}"), {
        "ui_language": language, "long_paths": long_paths == 1, "developer_mode": developer == 1,
        "profile_ascii": ascii_profile, "ansi_codepage": k32.GetACP(), "console_codepage": k32.GetConsoleOutputCP()}


def check_powershell_facts(probe):
    """CPU, Windows Audio service and Defender facts from the one PowerShell run (used again in later sections)."""
    need_windows()
    ps = probe.powershell()
    if "error" in ps and len(ps) == 1:
        raise Fail(f"PowerShell did not answer: {ps['error']}")
    cpu = ps.get("cpu") or {}
    arch = {0: "x86", 5: "Arm", 9: "x64", 12: "Arm64"}.get(cpu.get("architecture"), cpu.get("architecture"))
    detail = (f"Win32_Processor: {cpu.get('cores', '?')} cores, {cpu.get('logical', '?')} logical, architecture {arch} "
              f"(12 = Arm64); Windows Audio service: {ps.get('audiosrv', ps.get('audiosrv_error', '?'))}; Defender: "
              f"{ps.get('defender_mode', ps.get('defender_error', '?'))}")
    return detail, {k: v for k, v in ps.items() if k != "signatures"}


# ---------- 2. Python ----------

def check_python():
    base = getattr(sys, "_base_executable", sys.executable)
    kind = python_build_kind(sys.version, base, os.path.isdir(os.path.join(sys.base_prefix, "conda-meta")))
    venv = sys.prefix != sys.base_prefix
    return (f"{platform.python_version()} {struct.calcsize('P') * 8}-bit, {kind}; {sys.executable}"
            f"{' (venv of ' + base + ')' if venv else ''}"), {
        "version": sys.version, "executable": sys.executable, "base_executable": base, "prefix": sys.prefix,
        "base_prefix": sys.base_prefix, "venv": venv, "build": kind, "implementation": platform.python_implementation()}


def check_encoding(probe, original):
    """The encodings mt.py's output meets: this console as the probe was started, and a pipe (what the app reads) with
    the environment as it is and with PYTHONIOENCODING=utf-8 (the Mac app's JobSettings.swift:368 sets it for jobs)."""
    text = "@@LINE\t00:00:01\tru\tПривет, 你好"
    as_is = child("encoding", {"text": text}, probe.t_child, env={"PYTHONIOENCODING": None})
    fixed = child("encoding", {"text": text}, probe.t_child, env={"PYTHONIOENCODING": "utf-8"})
    if not (as_is.get("ok") and fixed.get("ok")):
        raise Fail(as_is.get("error") or fixed.get("error"))
    pipe_ok = as_is["writable"] is True
    data = {"console": original, "utf8_mode": sys.flags.utf8_mode, "preferred": locale.getpreferredencoding(False),
            "filesystem": sys.getfilesystemencoding(), "PYTHONIOENCODING": os.environ.get("PYTHONIOENCODING"),
            "PYTHONUTF8": os.environ.get("PYTHONUTF8"), "pipe_as_is": as_is, "pipe_utf8": fixed}
    return (f"console {original['stdout']} ({'terminal' if original['tty'] else 'redirected'}), UTF-8 mode "
            f"{sys.flags.utf8_mode}; a pipe as is: {as_is['encoding']}, Cyrillic {'ok' if pipe_ok else 'FAILS'}; "
            f"with PYTHONIOENCODING=utf-8: {fixed['encoding']}, {'ok' if fixed['writable'] is True else 'FAILS'}"), data


def check_vcruntime():
    """The Visual C++ runtime: ctranslate2.dll imports MSVCP140 and onnxruntime MSVCP140 and MSVCP140_1, and the
    python-build-standalone archive ships only vcruntime140*.dll (mtpy-windows #13). Which copy Windows loads, and its
    version."""
    need_windows()
    import ctypes
    results, missing = {}, []
    for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        try:
            path = module_path(ctypes.WinDLL(name)._handle)
            results[name] = {"path": path, "version": file_version(path)}
        except OSError as e:
            results[name] = {"error": str(e)}
            missing.append(name)
    shown = [f"{n} {r['version'] or '?'} from {os.path.dirname(r['path'])}" if "path" in r else f"{n} MISSING"
             for n, r in results.items()]
    if missing:
        raise Fail("; ".join(shown) + " — install the Visual C++ runtime (winget install Microsoft.VCRedist.2015+.x64)")
    return "; ".join(shown), results


def check_signatures(probe):
    """Authenticode signatures of this Python and of the engine's native files (Get-AuthenticodeSignature). Smart App
    Control in enforcement blocks unsigned files without cloud reputation; python-build-standalone's python.exe and the
    ctranslate2 and PortAudio DLLs were found unsigned (critic 1.7)."""
    need_windows()
    ps = probe.powershell()
    signatures = ps.get("signatures")
    if signatures is None:
        raise Fail(f"PowerShell did not answer: {ps.get('error', '?')}")
    if isinstance(signatures, dict):
        signatures = [signatures]
    if signatures and all(s.get("status") == "error" for s in signatures):
        raise Fail("Get-AuthenticodeSignature failed for every file: " + (signatures[0].get("signer") or "?"))
    lines = [f"{os.path.basename(s.get('file', '?'))}: {s.get('status')}"
             f"{' by ' + re.sub(r',.*', '', s.get('signer') or '').replace('CN=', '') if s.get('signer') else ''}"
             for s in signatures]
    return "\n".join(lines) or "no files", {"files": signatures}


# ---------- 3. security features that stop an unsigned app ----------

def check_smart_app_control():
    need_windows()
    state = reg("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\CI\Policy", "VerifiedAndReputablePolicyState")
    if state is None:
        return "not present (Windows 10, Windows Server, or a Windows 11 without it)", {"state": None, "meaning": None}
    meaning = SAC_STATES.get(state, f"unknown value {state}")
    return f"{meaning} (VerifiedAndReputablePolicyState = {state})", {"state": state, "meaning": meaning}


def check_citool():
    need_windows()
    exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "citool.exe")
    if not os.path.exists(exe):
        raise Skip("citool.exe not found (Windows 11 22H2 and later have it)")
    done = subprocess.run([exe, "-lp"], capture_output=True, timeout=15, stdin=subprocess.DEVNULL, **quiet())
    text = decode_output(done.stdout)
    policies = parse_citool(text)
    if not policies:
        raise Fail(f"no policy list (exit code {done.returncode}): {(decode_output(done.stderr) or text).strip()[:200]}")
    sac = [(n, e) for n, e in policies if n.startswith("VerifiedAndReputable")]
    mode = sac_from_citool(policies)
    return (f"Smart App Control by citool: {mode or 'no Smart App Control policy enforced'}; "
            f"{sum(e for _, e in policies)} of {len(policies)} policies enforced"), {
        "mode": mode, "sac_policies": sac, "policies": len(policies)}


def check_controlled_folders(probe):
    need_windows()
    ps = probe.powershell()
    state, source = ps.get("cfa"), "Get-MpPreference"
    if state is None:
        try:
            state = reg("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\Windows Defender\Windows Defender Exploit Guard"
                        r"\Controlled Folder Access", "EnableControlledFolderAccess")
            source = "registry"
        except PermissionError:
            state = None
    if state is None:
        raise Fail(f"not readable: {ps.get('cfa_error') or ps.get('error') or 'no value'} (a third-party antivirus "
                   f"puts Defender in passive mode: {ps.get('defender_mode', '?')})")
    meaning = CFA_STATES.get(int(state), f"unknown value {state}")
    return f"{meaning} ({source}: {state})", {"state": int(state), "meaning": meaning, "source": source}


def check_documents():
    need_windows()
    documents = known_folder(FOLDERID_DOCUMENTS)
    onedrive = [os.environ[v] for v in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial") if os.environ.get(v)]
    synced = any(os.path.normcase(documents).startswith(os.path.normcase(o.rstrip("\\")) + "\\") for o in onedrive)
    usage = shutil.disk_usage(documents)
    return (f"{documents}; inside OneDrive: {'yes (Known Folder Move: recordings there are synced)' if synced else 'no'}"
            f"; {usage.free / 2**30:.1f} GB free"), {"documents": documents, "onedrive": synced,
                                                       "onedrive_folders": onedrive}


def check_documents_write(probe):
    """Writes and deletes one small file in Documents, but only when Controlled folder access is on, in audit mode or
    unreadable: that is what decides whether a recording can be saved there (an unknown app is blocked, and Windows
    shows "Unauthorized changes blocked")."""
    need_windows()
    if probe.args.ci:
        raise Skip("not in --ci")
    state = probe.data("sec.cfa").get("state")
    if state == 0:
        raise Skip("not needed: Controlled folder access is off")
    folder = probe.data("sec.documents").get("documents") or known_folder(FOLDERID_DOCUMENTS)
    target = Path(folder) / "meeting-transcriber-probe.tmp"
    try:
        target.write_bytes(b"probe")
        target.unlink()
    except OSError as e:
        raise Fail(f"this Python cannot write into Documents: {type(e).__name__}: {e}")
    return "this Python could write a file into Documents (and deleted it)", {"writable": True}


def check_microphone_privacy():
    need_windows()
    switches, problems = {}, []
    for root, path, label in MIC_SWITCHES:
        try:
            value = reg(root, path, "Value")
        except PermissionError:
            value = "not readable"
        switches[label] = value
        if value == "Deny":
            problems.append(label)
    try:
        policy = reg_values("HKEY_LOCAL_MACHINE", APP_PRIVACY) or {}
    except PermissionError:
        policy = {"error": "not readable"}
    mic_policy = {k: v for k, v in policy.items() if k.lower().startswith("letappsaccessmicrophone")}
    if mic_policy.get("LetAppsAccessMicrophone") == 2:
        problems.append("policy LetAppsAccessMicrophone = force deny")
    # Desktop apps Windows has seen using the microphone: only their number and whether a python*.exe is among them
    # (the key names are full program paths, with the account name, and which other programs are there is private).
    used = [k.rsplit("#", 1)[-1].lower() for k in reg_subkeys("HKEY_CURRENT_USER", CONSENT + r"\NonPackaged")]
    python_seen = any(u.startswith("python") for u in used)
    shown = "; ".join(f"{label}: {value if value is not None else 'not set (default Allow)'}"
                      for label, value in switches.items())
    if mic_policy:
        shown += "; policy: " + ", ".join(f"{k} = {POLICY_STATES.get(v, v)}" for k, v in mic_policy.items())
    shown += f"; Python seen using the microphone before: {'yes' if python_seen else 'no'}"
    data = {"switches": switches, "policy": mic_policy, "desktop_apps_seen": len(used), "python_seen": python_seen}
    if problems:
        raise Fail(shown + " — blocked by: " + ", ".join(problems))
    return shown, data


# ---------- 4. microphone through sounddevice ----------

def check_sounddevice(probe):
    info = child_result("sd_info", {}, probe.t_child, package="sounddevice")
    devices = info["devices"]
    data = {k: info[k] for k in ("version", "portaudio", "library", "hostapis", "devices", "default")}
    default_in, default_out = info["default"]
    name = lambda i: next((d["name"] for d in devices if d["index"] == i), "none")
    lines = [f"sounddevice {info['version']}, {info['portaudio']}, {info['library']}",
             "host APIs: " + ", ".join(f"{a['name']} ({a['devices']})" for a in info["hostapis"]),
             f"default input: {name(default_in)}; default output: {name(default_out)}"]
    lines += [f"  [{d['index']}] {d['name']} ({d['hostapi']}; in {d['in']}, out {d['out']}; {d['rate']:g} Hz)"
              for d in devices]
    if not devices:
        lines.append("no audio devices at all (a CI runner has none; on a PC: is the Windows Audio service running?)")
    return "\n".join(lines), data


def describe_recording(r):
    level = f"peak {r['peak']:.4f} ({r['peak_dbfs']} dBFS), RMS {r['rms']:.4f}" if r.get("samples") else "no samples"
    return (f"{r['device']} ({r['hostapi']}, native {r['device_rate']:g} Hz): {r['frames']} of {r['expected_frames']} "
            f"frames, {level}; opened in {r['open_s']} s; format check: {r['format_supported']}")


def check_recording(probe, wasapi, auto_convert):
    need_windows()
    if not probe.data("audio.sounddevice"):
        raise Skip("sounddevice is not usable (see above)")
    r = child_result("sd_record", {"seconds": probe.seconds, "wasapi": wasapi, "auto_convert": auto_convert},
                     probe.seconds + probe.t_child, package="sounddevice")
    return describe_recording(r), r


def check_loopback_inputs(probe):
    info = probe.data("audio.sounddevice")
    if not info:
        raise Skip("sounddevice is not usable")
    found = loopback_inputs(info["devices"])
    duplicates = same_name_groups(info["devices"])
    lines = [("loopback-capable inputs: " + ", ".join(f"[{d['index']}] {d['name']} ({d['hostapi']})" for d in found))
             if found else "no input carries the computer's sound (no Stereo Mix, virtual cable or [Loopback] device)"]
    lines.append(f"{len(duplicates)} input name(s) appear under several host APIs" +
                 (": " + "; ".join(f"{n} ({', '.join(a)})" for n, a in list(duplicates.items())[:6]) if duplicates else ""))
    return "\n".join(lines), {"loopback_inputs": found, "duplicates": duplicates}


# ---------- 5. the computer's sound (loopback) ----------

def describe_loopback(r):
    level = f"peak {r['peak']:.4f} ({r['peak_dbfs']} dBFS), RMS {r['rms']:.4f}" if r.get("samples") else "no samples"
    frames = f"{r['frames']} of {r['expected_frames']} frames" if "expected_frames" in r else f"{r['frames']} frames"
    wall = f" in {r['wall_s']} s" if "wall_s" in r else ""
    warn = f"; warnings: {'; '.join(r['warnings'])}" if r.get("warnings") else ""
    return f"{r['loopback']}: {frames}{wall}, {level}{warn}"


def check_loopback(probe, task, package):
    need_windows()
    r = child_result(task, {"seconds": probe.seconds}, probe.seconds + probe.t_child + 5, package=package)
    return describe_loopback(r), r


# ---------- 6. the kind of output device ----------

def check_formfactor(probe):
    need_windows()
    r = child_result("formfactor", {}, probe.t_child, package="pycaw (it brings comtypes and psutil; needs Python 3.10+)")
    lines = []
    for role, d in r["defaults"].items():
        if "error" in d:
            lines.append(f"default {role}: {d['error']}")
        else:
            lines.append(f"default {role}: {d['name']}: {d['form_factor']} -> {d['kind']} (bus {d['bus'] or '?'})")
    lines += [f"  output {d['name']}: {d['form_factor']} -> {d['kind']}" for d in r["outputs"]]
    return "\n".join(lines), r


# ---------- 7. NVIDIA and CUDA ----------

def check_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe and WIN:
        for candidate in (os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"),
                          os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation",
                                       "NVSMI", "nvidia-smi.exe")):
            if os.path.exists(candidate):
                exe = candidate
    if not exe:
        raise Skip("nvidia-smi not found: no NVIDIA driver (or no NVIDIA GPU)")
    query = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, encoding="utf-8", errors="replace", timeout=10, stdin=subprocess.DEVNULL,
                           **quiet())
    if query.returncode != 0:
        raise Fail(f"nvidia-smi exit code {query.returncode}: {(query.stderr or query.stdout).strip()[:200]}")
    header = subprocess.run([exe], capture_output=True, encoding="utf-8", errors="replace", timeout=10,
                            stdin=subprocess.DEVNULL, **quiet()).stdout
    gpus, cuda = parse_nvidia_smi(query.stdout, header)
    if not gpus:
        raise Fail(f"no GPU listed: {query.stdout.strip()[:200]}")
    return "; ".join(f"{g['name']}, {g['memory']}, driver {g['driver']}" for g in gpus) + \
        f"; the driver supports CUDA {cuda or '?'}", {"gpus": gpus, "driver_cuda": cuda}


def check_nvcuda(probe):
    need_windows()
    r = child("nvcuda", {}, probe.t_child)
    if not r.get("ok"):
        raise Skip(f"nvcuda.dll does not load: no NVIDIA driver ({r.get('error')})")
    state = "ok" if r["init"] == 0 else f"cuInit returned {r['init']} (100 = no CUDA device)"
    return f"{r['path']}: CUDA {r['cuda']} driver API, {state}, {r['devices']} device(s)", r


def check_ctranslate2(probe):
    r = child_result("ct2_info", {}, probe.t_child, package="ctranslate2")
    return (f"ctranslate2 {r['version']}: {r['cuda_devices']} CUDA device(s); compute types on the CPU: "
            f"{r['compute_types']['cpu']}; on CUDA: {r['compute_types']['cuda']}; "
            f"its DLLs: {', '.join(r['dlls']) or '-'}"), r


def check_nvidia_wheels():
    need_windows()
    found = nvidia_bin_dirs()
    if not found:
        raise Skip("no site-packages\\nvidia\\*\\bin folder (for the GPU: pip install -r requirements-cuda-windows.txt, "
                   "which pins the last cuBLAS with a Windows wheel)")
    lines = [f"{d}: {', '.join(dlls) or 'no DLL'}" for d, dlls in found.items()]
    wanted = {n: any(n in dlls for dlls in found.values()) for n in CUDA_DLLS}
    return "\n".join(lines), {"folders": found, "present": wanted}


DLL_STRATEGIES = ["python", "standard", "add_dll_directory+python", "add_dll_directory+standard", "path+standard",
                  "full_path"]


def check_dll_loading(probe):
    """cuBLAS and cuDNN loaded by name in six ways, each in a fresh process: which of them W3 can rely on. A load that
    works with "python" or "standard" alone found the DLL in System32 or on PATH (a CUDA toolkit, another program)."""
    need_windows()
    dirs = list(nvidia_bin_dirs())
    results, lines = {}, []
    for strategy in DLL_STRATEGIES:
        if strategy != "python" and strategy != "standard" and not dirs:
            continue
        r = child("dll_load", {"strategy": strategy, "dirs": dirs, "names": CUDA_DLLS}, probe.t_child)
        results[strategy] = r.get("results") if r.get("ok") else {"error": r.get("error")}
        ok = [n for n, v in (results[strategy] or {}).items() if str(v).startswith("loaded")]
        lines.append(f"{strategy}: {len(ok)} of {len(CUDA_DLLS)} load" + (f" ({', '.join(ok)})" if ok else ""))
    if not dirs:
        lines.append("(no nvidia wheel folders: only the system search was tried)")
    return "\n".join(lines), {"strategies": results, "dirs": dirs}


WHISPER_RUNS = [("cpu", "as_is", "cpu", "int8"), ("cuda as is", "as_is", "cuda", "float16"),
                ("cuda + add_dll_directory", "add_dll_directory", "cuda", "float16"),
                ("cuda + PATH", "path", "cuda", "float16"), ("cuda + preload by full path", "preload", "cuda", "float16")]


def check_whisper(probe):
    need_windows()
    if not (probe.args.cuda_load or (probe.args.interactive and not probe.args.ci)):
        raise Skip("pass --cuda-load (or --interactive) to load Whisper tiny on the GPU")
    if not package_dir("faster_whisper"):
        raise Skip("faster_whisper is not installed (pip install faster-whisper)")
    dirs = list(nvidia_bin_dirs())
    results, lines = {}, []
    for label, strategy, device, compute in WHISPER_RUNS:
        if strategy in ("add_dll_directory", "path", "preload") and not dirs:
            continue
        timeout = 600 if not results else 180         # the first run downloads the model (about 75 MB)
        try:
            r = child("whisper", {"strategy": strategy, "dirs": dirs, "device": device, "compute_type": compute}, timeout)
        except Fail as e:
            r = {"ok": False, "error": str(e)}
        results[label] = r
        lines.append(f"{label} ({compute}): " + (f"ok, loaded in {r['load_s']} s, 1 s decoded in {r['decode_s']} s"
                                                 if r.get("ok") else f"FAILED: {r.get('error')}"))
    return "\n".join(lines), {"runs": results}


# ---------- 8. ffmpeg ----------

def check_ffmpeg(probe):
    """ffmpeg as mt.py finds it (mt.py:522-547): on PATH, else the copy mt.py fetched through static-ffmpeg into
    <MT_HOME's parent>\\ffmpeg\\<platform>, else static-ffmpeg's own folder."""
    data_dir = Path(os.environ.get("MT_HOME", MT_DIR / "data"))
    platform_key = "win32" if WIN else ("darwin" if sys.platform == "darwin" else "linux")
    candidates = [("PATH", shutil.which("ffmpeg"))]
    engine_dir = data_dir.parent / "ffmpeg" / platform_key
    candidates.append(("mt.py's folder", str(engine_dir / ("ffmpeg.exe" if WIN else "ffmpeg"))))
    static = package_dir("static_ffmpeg")
    if static:
        candidates.append(("static-ffmpeg's folder", str(Path(static) / "bin" / platform_key /
                                                          ("ffmpeg.exe" if WIN else "ffmpeg"))))
    found = next(((where, p) for where, p in candidates if p and os.path.exists(p)), None)
    if not found and probe.args.fetch_ffmpeg:
        r = child_result("ffmpeg_fetch", {"download_dir": str(engine_dir)}, 600, package="static-ffmpeg")
        found = ("fetched now into mt.py's folder", r["ffmpeg"])
    if not found:
        raise Fail("not found (mt.py fetches it at `python mt.py prepare`, about 80 MB; or pass --fetch-ffmpeg)"
                   + ("" if static else "; static-ffmpeg is not installed either"))
    where, exe = found
    out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, encoding="utf-8", errors="replace",
                         timeout=10, stdin=subprocess.DEVNULL, **quiet())
    first = (out.stdout.strip().splitlines() or ["?"])[0]
    return f"{first} ({where}: {exe})", {"where": where, "path": exe, "version": first,
                                          "ffprobe": bool(shutil.which("ffprobe")) if where == "PATH" else None}


# ---------- what it means ----------

def summarize(r):
    """Plain conclusions from the results (r: check id -> entry), most decisive first. Each line says what the finding
    means for the Windows app; an item that could not be checked says so instead of guessing."""
    ok = lambda cid: r.get(cid, {}).get("status") == "ok"
    data = lambda cid: r.get(cid, {}).get("data", {}) if ok(cid) else {}
    out = []
    osd = data("os.version")
    if osd and not osd.get("windows"):
        out.append("Not Windows: the Windows checks were skipped; run this on the Windows PC.")
    elif osd:
        if osd.get("windows11"):
            out.append(f"Windows 11 (build {osd['build']}): process loopback is available, so a capture helper could "
                       "leave the app's own sounds out of the computer's-sound track.")
        elif osd.get("process_loopback"):
            out.append(f"{'Windows Server' if osd.get('server') else 'Windows'} build {osd['build']}: process loopback "
                       f"should be available (build {PROCESS_LOOPBACK_BUILD}+).")
        else:
            out.append(f"Windows 10 build {osd['build']}: no process loopback (it needs build {PROCESS_LOOPBACK_BUILD}); "
                       "only endpoint loopback, which also records the app's own sounds. Windows 10 support ended "
                       "on 2025-10-14.")
    arch = data("os.arch")
    if arch.get("emulated") and arch.get("native") == "Arm64":
        out.append("x64 Python under emulation on an Arm64 PC: the only way ctranslate2 runs here (it has no win_arm64 "
                   "wheel); its speed under emulation is unmeasured.")
    elif arch.get("python_arch") == "Arm64":
        out.append("Arm64 Python: ctranslate2 has no win_arm64 wheel, so faster-whisper cannot be installed; use the x64 "
                   "Python (Windows 11 on Arm emulates it).")
    elif arch.get("python_arch") == "x86":
        out.append("32-bit Python: ctranslate2 has no 32-bit wheel; install the 64-bit Python.")
    cpu = data("os.cpu")
    features = cpu.get("features", {})
    if features and not features.get("ARM_V8") and not features.get("AVX2"):
        out.append("No AVX2 reported (or a Windows older than 2004, which cannot report it): expect a slower CPU path "
                   "in CTranslate2 [I].")
    if features.get("AVX512F"):
        out.append("AVX-512 present: compare the CPU speed with and without MKL_ENABLE_INSTRUCTIONS=AVX512.")
    classes = cpu.get("efficiency_classes", {})
    if len(classes) > 1:
        out.append("Hybrid CPU (performance and efficiency cores): mt.py uses every logical processor; measure whether "
                   "the performance cores alone are faster, as on Apple silicon (mtpy-windows #15).")
    memory = data("os.memory").get("total_gb")
    if memory and memory < 12:
        out.append(f"{memory} GB of memory: a long recording needs about 3 GB to transcribe; close other big programs "
                   "(mt.py advises the same below 12 GB).")
    for var, d in data("os.disk").items():
        if d.get("free_gb", 99) < 8:
            out.append(f"Only {d['free_gb']} GB free on {d['drive']} ({var}): the engine and models take about 4-5 GB "
                       "(Whisper 1.6 GB, NLLB 1.4 GB, NVIDIA libraries 1.2 GB or more) and a call writes about 115 MB "
                       "per hour per track.")
    paths = data("os.paths")
    if paths and not paths.get("profile_ascii"):
        out.append("The profile folder's name is not ASCII: check that models load from it (open question 5).")
    py = data("py.version")
    if py.get("build") == "python-build-standalone":
        out.append("This Python is python-build-standalone: its python.exe is not Authenticode-signed.")
    enc = data("py.encoding")
    if enc:
        if enc["pipe_as_is"].get("writable") is not True:
            out.append(f"On a pipe this Python writes {enc['pipe_as_is'].get('encoding')}: mt.py's @@ events crash on "
                       "Cyrillic or Chinese text unless the app sets PYTHONIOENCODING=utf-8 or PYTHONUTF8=1 "
                       "(mtpy-windows #3, reproduced here).")
        else:
            out.append("A pipe takes Cyrillic text here as is; the app should still set PYTHONIOENCODING=utf-8.")
    if r.get("py.vcruntime", {}).get("status") == "failed":
        out.append("The Visual C++ runtime is missing: ctranslate2 and onnxruntime will not load; install it "
                   "(winget install Microsoft.VCRedist.2015+.x64) or ship msvcp140*.dll next to python.exe.")
    sac = data("sec.sac").get("state")
    citool = data("sec.citool").get("mode")
    if sac == 1 or citool == "On (enforcement)":
        out.append("Smart App Control is On: the unsigned app, and the unsigned python.exe and DLLs it runs, would be "
                   "blocked. Signing every file, python.org's signed Python, or turning Smart App Control off "
                   "(it cannot be turned back on without reinstalling Windows) are the ways out.")
    elif sac == 2 or citool == "Evaluation":
        out.append("Smart App Control is in evaluation: Windows may turn it On by itself later, and then the unsigned "
                   "app would be blocked.")
    elif sac == 0:
        out.append("Smart App Control is Off: unsigned files run (SmartScreen still warns about a downloaded zip).")
    cfa = data("sec.cfa").get("state")
    if cfa in (1, 2):
        out.append("Controlled folder access is " + ("On: an unknown app cannot write into Documents (allow python.exe "
                                                     "in Ransomware protection, or use another output folder)."
                                                     if cfa == 1 else "in audit mode: writes are logged, not blocked."))
    if r.get("sec.docwrite", {}).get("status") == "failed":
        out.append("This Python could not write into Documents: the output folder must be elsewhere or allowed.")
    if data("sec.documents").get("onedrive"):
        out.append("Documents is synced by OneDrive: recordings saved there are uploaded (about 115 MB per hour per "
                   "track); default the output folder to one outside OneDrive.")
    mic = r.get("sec.microphone", {})
    if mic.get("status") == "failed":
        out.append("The microphone is blocked for desktop apps: Settings > Privacy & security > Microphone > "
                   "Microphone access and Let desktop apps access your microphone.")
    default = data("audio.default")
    if default:
        if default.get("frames", 0) == 0:
            out.append("The microphone opened as mt.py opens it but delivered no audio.")
        elif default.get("peak", 0) < SILENT_PEAK:
            out.append(f"The microphone delivered only silence (peak below {SILENT_PEAK}): muted, the wrong default "
                       "device, or the privacy switch.")
        else:
            out.append("The microphone works as mt.py opens it today (default device, 16 kHz mono).")
    elif r.get("audio.default", {}).get("status") == "failed":
        out.append("The microphone could not be opened as mt.py opens it: " + r["audio.default"]["detail"][:160])
    auto, plain = r.get("audio.wasapi_auto", {}), r.get("audio.wasapi_plain", {})
    if auto.get("status") == "ok" and plain.get("status") == "failed":
        out.append("WASAPI opens at 16 kHz only with auto_convert=True (mtpy-windows #7 confirmed).")
    elif auto.get("status") == "ok" and plain.get("status") == "ok":
        out.append("WASAPI opened at 16 kHz even without auto_convert; keep auto_convert anyway.")
    elif auto.get("status") == "failed":
        out.append("WASAPI did not open at 16 kHz even with auto_convert: " + auto["detail"][:160])
    inputs = data("audio.inputs")
    if inputs.get("duplicates"):
        out.append("Each input appears under several host APIs: mt.py must keep one host API before matching a name "
                   "(mtpy-windows #6).")
    if inputs.get("loopback_inputs"):
        out.append("An input carries the computer's sound (" + inputs["loopback_inputs"][0]["name"] +
                   "): `record --system-device` could use it today.")
    loop_ok = []
    for lib in ("soundcard", "pyaudiowpatch"):
        for when in ("playing", "silent", "once"):
            cid = f"loop.{lib}.{when}"
            d = data(cid)
            if not d:
                continue
            if when != "silent" and d.get("peak", 0) >= SILENT_PEAK:
                loop_ok.append(lib)
            if lib == "pyaudiowpatch" and when == "silent" and d.get("expected_frames"):
                share = 100 * d["frames"] // max(1, d["expected_frames"])
                out.append(f"PyAudioWPatch got {share} % of the frames while nothing played: the recorder must pad with "
                           "silence by the clock.")
    if loop_ok:
        out.append("The computer's sound was captured through " + " and ".join(sorted(set(loop_ok))) + ".")
    elif any(ok(f"loop.{lib}.{w}") for lib in ("soundcard", "pyaudiowpatch") for w in ("playing", "once")):
        out.append("Loopback ran but recorded silence: was something playing? Run again with --interactive.")
    elif osd.get("windows"):
        out.append("Loopback was not tested: pip install soundcard pyaudiowpatch, then run again.")
    ff = data("out.formfactor").get("defaults", {}).get("render/console", {})
    if ff.get("form_factor"):
        what = {"speaker": "the app would untick \"Also record the sound this computer plays\" (the microphone hears "
                           "both sides)", "headphones": "the box stays ticked",
                "unknown": "the app would learn it from the user's own untick"}[ff["kind"]]
        out.append(f"Default output {ff['name']}: {ff['form_factor']}, so {what}.")
    gpu = data("gpu.smi")
    ct2 = data("gpu.ct2")
    whisper = data("gpu.whisper").get("runs", {})
    cuda_ok = [label for label, w in whisper.items() if label.startswith("cuda") and w.get("ok")]
    if not gpu and not data("gpu.nvcuda"):
        out.append("No NVIDIA GPU or driver: Whisper runs on the CPU (int8).")
    elif cuda_ok:
        out.append(f"GPU usable: Whisper ran on CUDA float16 ({', '.join(cuda_ok)}).")
    elif whisper:
        failed = [label for label in whisper if label.startswith("cuda")]
        out.append("GPU present, but Whisper failed on CUDA in every way tried (" + ", ".join(failed) + "): mt.py falls "
                   "back to the CPU (int8) with a note (MT_DEVICE=cpu skips the attempt).")
    else:
        count = ct2.get("cuda_devices")
        out.append(f"GPU present ({gpu.get('gpus', [{}])[0].get('name', '?') if gpu else 'nvcuda.dll loads'}); "
                   f"ctranslate2 sees {count if count is not None else '?'} CUDA device(s); Whisper on the GPU not "
                   "tested (run with --cuda-load).")
    if r.get("ffmpeg", {}).get("status") == "failed":
        out.append("ffmpeg not found: `python mt.py prepare` fetches it (about 80 MB).")
    errors = [cid for cid, e in r.items() if e.get("status") == "error"]
    if errors:
        out.append("Checks that hit a bug in this probe (please send the report anyway): " + ", ".join(errors))
    return out


# ---------- main ----------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Windows probe for meeting-transcriber (see the top of this file).")
    parser.add_argument("--interactive", action="store_true", help="ask you to speak / play sound; load Whisper on CUDA")
    parser.add_argument("--ci", action="store_true", help="no questions, 1-s recordings, short timeouts, exit 0")
    parser.add_argument("--out", metavar="FILE", help="also write the report as JSON to FILE")
    parser.add_argument("--cuda-load", action="store_true", help="load Whisper tiny on CUDA (downloads it once)")
    parser.add_argument("--fetch-ffmpeg", action="store_true", help="fetch ffmpeg through static-ffmpeg if missing")
    parser.add_argument("--strict", action="store_true", help="exit 1 when a check hit a bug in this script")
    args = parser.parse_args(argv)

    original = {"stdout": getattr(sys.stdout, "encoding", None), "stderr": getattr(sys.stderr, "encoding", None),
                "tty": bool(sys.stdout and sys.stdout.isatty())}
    try:
        # The report holds device names in any script; a redirected stdout in the ANSI code page must not crash on them.
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    probe = Probe(args, make_scrubber(names=display_name_words()))
    try:
        mt_version = (MT_DIR / "VERSION").read_text().strip()
    except OSError:
        mt_version = "?"
    mode = "ci" if args.ci else "interactive" if args.interactive else "full"
    started = datetime.now()
    print(f"win_probe {PROBE_VERSION} for meeting-transcriber {mt_version}, {started:%Y-%m-%d %H:%M}, mode {mode}, "
          f"pid {os.getpid()}", flush=True)
    if not args.ci:
        print("Nothing is uploaded. The recordings below last a few seconds each and are not saved.", flush=True)

    probe.section("os", "1. Operating system and hardware")
    probe.run("os.version", "Windows version", check_os_version)
    probe.run("os.arch", "Architecture", check_arch)
    probe.run("os.cpu", "CPU", check_cpu)
    probe.run("os.memory", "Memory", check_memory)
    probe.run("os.disk", "Free disk", check_disk)
    probe.run("os.paths", "Language and paths", check_language_and_paths)
    probe.run("os.powershell", "PowerShell facts", lambda: check_powershell_facts(probe), probe.t_ps)

    probe.section("python", "2. Python")
    probe.run("py.version", "This Python", check_python)
    probe.run("py.encoding", "Text encoding", lambda: check_encoding(probe, original), 2 * probe.t_child + 5)
    probe.run("py.vcruntime", "Visual C++ runtime", check_vcruntime)
    probe.run("py.signatures", "Signatures", lambda: check_signatures(probe), probe.t_ps)

    probe.section("security", "3. Security features that affect an unsigned app")
    probe.run("sec.sac", "Smart App Control", check_smart_app_control)
    probe.run("sec.citool", "Code integrity policies", check_citool, 20)
    probe.run("sec.cfa", "Controlled folder access", lambda: check_controlled_folders(probe), probe.t_ps)
    probe.run("sec.documents", "Documents folder", check_documents)
    probe.run("sec.docwrite", "Writing into Documents", lambda: check_documents_write(probe))
    probe.run("sec.microphone", "Microphone privacy", check_microphone_privacy)

    probe.section("audio", "4. Microphone (sounddevice)")
    probe.run("audio.sounddevice", "PortAudio and devices", lambda: check_sounddevice(probe), probe.t_child + 5)
    if WIN and probe.data("audio.sounddevice"):
        probe.ask(f"Microphone tests: press Enter, then speak (count aloud) until the three of them finish, about "
                  f"{int(3 * probe.seconds + 6)} s.")
    record_timeout = probe.seconds + probe.t_child + 10
    probe.run("audio.default", "Default input, as mt.py opens it", lambda: check_recording(probe, False, None),
              record_timeout)
    probe.run("audio.wasapi_auto", "WASAPI default input, auto_convert", lambda: check_recording(probe, True, True),
              record_timeout)
    probe.run("audio.wasapi_plain", "WASAPI default input, no auto_convert", lambda: check_recording(probe, True, False),
              record_timeout)
    probe.run("audio.inputs", "Loopback inputs and duplicates", lambda: check_loopback_inputs(probe))

    probe.section("loopback", "5. The computer's sound (loopback)")
    interactive = args.interactive and not args.ci
    rounds = [("playing", "Play something through your usual output now (a video, music), then press Enter."),
              ("silent", "Now stop all sound (pause the video), then press Enter.")] if interactive else [("once", None)]
    loop_timeout = probe.seconds + probe.t_child + 15
    for when, prompt in rounds:
        if WIN and prompt:
            probe.ask(prompt)
        suffix = {"playing": ", sound playing", "silent": ", nothing playing", "once": ""}[when]
        probe.run(f"loop.soundcard.{when}", "SoundCard loopback" + suffix,
                  lambda: check_loopback(probe, "sc_loopback", "soundcard"), loop_timeout)
        probe.run(f"loop.pyaudiowpatch.{when}", "PyAudioWPatch loopback" + suffix,
                  lambda: check_loopback(probe, "paw_loopback", "pyaudiowpatch"), loop_timeout)

    probe.section("output", "6. Output device kind")
    probe.run("out.formfactor", "Form factor", lambda: check_formfactor(probe), probe.t_child + 5)

    probe.section("gpu", "7. NVIDIA and CUDA")
    probe.run("gpu.smi", "nvidia-smi", check_nvidia_smi, 25)
    probe.run("gpu.nvcuda", "CUDA driver (nvcuda.dll)", lambda: check_nvcuda(probe), probe.t_child + 5)
    probe.run("gpu.ct2", "ctranslate2", lambda: check_ctranslate2(probe), probe.t_child + 5)
    probe.run("gpu.wheels", "nvidia wheel folders", check_nvidia_wheels)
    probe.run("gpu.dlls", "Loading cuBLAS and cuDNN", lambda: check_dll_loading(probe), 7 * probe.t_child)
    probe.run("gpu.whisper", "Whisper tiny on the GPU", lambda: check_whisper(probe), 600 + 4 * 180 + 30)

    probe.section("ffmpeg", "8. ffmpeg")
    probe.run("ffmpeg", "ffmpeg", lambda: check_ffmpeg(probe), 620 if args.fetch_ffmpeg else 20)

    summary = probe.scrub(summarize(probe.raw))
    print("\n== What this means", flush=True)
    for line in summary:
        print(f"  - {line}", flush=True)
    report = {"probe": "win_probe", "probe_version": PROBE_VERSION, "mt_version": mt_version,
              "started": started.isoformat(timespec="seconds"), "mode": mode,
              "seconds": round((datetime.now() - started).total_seconds(), 1), "sections": probe.sections,
              "summary": summary}
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    print("\n== JSON report (send this block, or the file written by --out)", flush=True)
    print("-----BEGIN MT PROBE-----", flush=True)
    print(text, flush=True)
    print("-----END MT PROBE-----", flush=True)
    if args.out:
        # After the block is printed: a folder that does not exist (or a refused write) must not lose the report.
        try:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"\nReport written to {probe.scrub(str(Path(args.out).resolve()))}", flush=True)
        except OSError as e:
            print(f"\nwarning: could not write --out {probe.scrub(str(args.out))} ({type(e).__name__}: "
                  f"{probe.scrub(str(e))}); send the block above instead", file=sys.stderr, flush=True)
    if not args.ci:
        print("\nDone. Run it once with headphones, once with speakers and once with the microphone switch off "
              "(Settings > Privacy & security > Microphone), and send the three reports.", flush=True)
    errors = [cid for cid, e in probe.results.items() if e["status"] == "error"]
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--_child":
        sys.exit(child_main(sys.argv[2], sys.argv[3]))
    sys.exit(main())
