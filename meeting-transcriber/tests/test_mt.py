import collections
import functools
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
import mt


def test_group_spans_closes_chunk_at_a_pause_once_long_enough_and_caps_at_30s():
    s = lambda a, b: {"start": a * mt.RATE, "end": b * mt.RATE}
    # 0-2 and 3-4 are too short alone and merge (0-4 < 5 s, then 6-9 joins: 0-9 >= 5 s closes)
    assert mt.group_spans([s(0, 2), s(3, 4), s(6, 9), s(10, 12)]) == [
        [0, 9 * mt.RATE], [10 * mt.RATE, 12 * mt.RATE]]
    # never above 30 s
    assert mt.group_spans([s(0, 3), s(4, 6), s(7, 40)]) == [[0, 6 * mt.RATE], [7 * mt.RATE, 40 * mt.RATE]]


def test_fmt_renames_whole_speaker_names_in_an_overlap_note():
    note = {"t": 5, "note": "overlap 00:00:05–00:00:07 Speaker 1 / Speaker 10"}
    assert mt.fmt(note, None, {"Speaker 1": "Anna", "Speaker 10": "Boris"}) == "# [00:00:05] overlap 00:00:05–00:00:07 Anna / Boris"
    assert mt.fmt({"t": 1, "note": "Speaker 1 said nothing"}, None, {"Speaker 1": "Anna"}) == "# [00:00:01] Speaker 1 said nothing"


def test_fmt_marks_low_confidence_as_guess():
    assert mt.fmt({"t": 3725.4, "lang": "ru", "conf": 0.9, "text": "привет"}) == "[01:02:05] [ru] привет"
    assert mt.fmt({"t": 0, "lang": "de", "conf": 0.3, "text": "x"}, "y") == "[00:00:00] [de?] y"


def test_translate_batches_caches_and_keeps_original_when_a_line_is_missing(monkeypatch):
    calls = []

    class Fake:
        def __init__(self): self.messages = self
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="0|Hello\n1|Bye")])  # line 2 missing

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Fake))
    doc = {"lines": [{"text": "Привет"}, {"text": "Пока"}, {"text": "Ещё"}], "translations": {}}
    assert mt.translate(doc, "English") == ["Hello", "Bye", "Ещё"]
    assert doc["translations"] == {}          # incomplete: not cached, so the next run retries
    mt.translate(doc, "English")
    assert len(calls) == 2


def test_complete_translation_is_cached(monkeypatch):
    calls = []

    class Fake:
        def __init__(self): self.messages = self
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="0|Hello")])

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Fake))
    doc = {"lines": [{"text": "Привет"}], "translations": {}}
    mt.translate(doc, "English"); mt.translate(doc, "English")
    assert len(calls) == 1 and doc["translations"]["English"] == ["Hello"]


# ---------- loading audio and writing the text files ----------
import shutil
import struct
import subprocess

import pytest

# Tests that send POSIX signals to pytest's own process. SIGUSR1/SIGUSR2 do not exist on Windows, and there
# os.kill(pid, SIGINT or SIGTERM) is TerminateProcess, which would end pytest itself. On Windows the same behaviour is
# covered through the control channel (MT_CONTROL=stdin) in test_windows.py.
posix_signals = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows: see test_windows.py")

SAMPLE = "[00:00:04] [en] Hello.\n[00:01:05] [de?] Guten Tag.\nplain line without a time\n"


def test_parse_transcript_roundtrips_our_format_and_keeps_plain_lines():
    doc = mt.parse_transcript("﻿".lstrip("﻿") + SAMPLE + "\n\n")
    assert [l["text"] for l in doc["lines"]] == ["Hello.", "Guten Tag.", "plain line without a time"]
    assert doc["lines"][1]["conf"] < mt.LOW_CONF and doc["lines"][1]["t"] == 65
    assert doc["lines"][2]["t"] is None
    assert mt.render(doc) == SAMPLE                      # what we read is what we write back


class FakeAnthropic:
    def __init__(self):
        self.messages = self

    def create(self, **kw):
        n = kw["messages"][0]["content"].split("\n\n", 1)[1].count("\n") + 1
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="\n".join(f"{i}|T{i}" for i in range(n)))])


def test_write_outputs_makes_separate_transcript_and_translation_files(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    doc = mt.parse_transcript(SAMPLE)
    paths = mt.write_outputs(doc, tmp_path / "out", "call", ["English", "Brazilian Portuguese"])
    assert [p.name for p in paths] == ["call.transcript.txt", "call.English.txt", "call.Brazilian_Portuguese.txt"]
    assert (tmp_path / "out" / "call.transcript.txt").read_text(encoding="utf-8") == SAMPLE
    assert (tmp_path / "out" / "call.English.txt").read_text(encoding="utf-8") == (
        "[00:00:04] [en] T0\n[00:01:05] [de?] T1\nT2\n")


def test_translate_never_overwrites_its_own_input(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    src = tmp_path / "call.English.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    assert mt.write_outputs(mt.parse_transcript(SAMPLE), tmp_path, "call", ["English"], transcript=False, protect=src) == []
    assert src.read_text(encoding="utf-8") == SAMPLE


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("ext", ["wav", "mp3", "m4a", "ogg", "opus", "flac", "aac", "amr", "wma", "aiff", "webm", "mp4"])
def test_load_audio_reads_any_format_ffmpeg_can_decode(tmp_path, ext):
    src = tmp_path / f"tone.{ext}"
    extra = ["-ar", "8000"] if ext == "amr" else []
    done = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-ac", "1",
                           *extra, str(src)], capture_output=True)
    if done.returncode:
        pytest.skip(f"this ffmpeg cannot encode {ext}")
    audio = mt.load_audio(src)
    assert 1.8 * mt.RATE < len(audio) < 2.3 * mt.RATE and audio.dtype.name == "float32"


def _wav_counting(path, seconds_held, seconds_counted, chunk_after=b""):
    """A WAV in the layout the app's WavWriter and mt.py's WavFile write, holding `seconds_held` of a tone while its header
    counts only `seconds_counted` (a recording cut short before the header was patched), optionally with a chunk after."""
    import numpy as np
    t = np.arange(int(seconds_held * mt.RATE)) / mt.RATE
    data = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes()
    counted = int(seconds_counted * mt.RATE) * 2
    header = (b"RIFF" + struct.pack("<I", 36 + counted) + b"WAVE" + b"fmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, mt.RATE, mt.RATE * 2, 2, 16) + b"data" + struct.pack("<I", counted))
    path.write_bytes(header + data + chunk_after)
    return path


def test_a_recording_cut_short_is_read_to_its_end_not_to_its_stale_header(tmp_path):
    # The app's WAV headers are patched every 30 s and at Stop; a Stop that never finished (1.10.0) left a header that counts
    # less than the file holds, and ffmpeg stops at the header's count.
    src = _wav_counting(tmp_path / "mic.wav", 3.0, 1.0)
    assert mt.cut_short_wav(src)
    if shutil.which("ffmpeg"):
        assert 2.9 * mt.RATE < len(mt.load_audio(src)) < 3.1 * mt.RATE


def test_a_whole_wav_or_one_with_a_chunk_after_its_data_is_read_as_its_header_says(tmp_path):
    assert not mt.cut_short_wav(_wav_counting(tmp_path / "whole.wav", 2.0, 2.0))
    listed = _wav_counting(tmp_path / "listed.wav", 2.0, 2.0, chunk_after=b"LIST" + struct.pack("<I", 12) + b"INFOISFT\x00\x00\x00\x00")
    assert not mt.cut_short_wav(listed), "an imported WAV's metadata chunk is not sound"
    other = tmp_path / "tone.mp3"
    other.write_bytes(b"ID3" + bytes(100))
    assert not mt.cut_short_wav(other) and not mt.cut_short_wav(tmp_path / "missing.wav")
    if shutil.which("ffmpeg"):
        assert 1.9 * mt.RATE < len(mt.load_audio(listed)) < 2.1 * mt.RATE


def test_load_audio_reports_an_unreadable_file(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="could not read"):
        mt.load_audio(bad)


# ---------- video files: the sound is taken out of the video ----------
def _make(path, *audio, video=True):
    """Make a test file with a picture and/or sine-wave audio tracks (frequency, seconds)."""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if video:
        cmd += ["-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=3"]
    for freq, sec in audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={sec}"]
    done = subprocess.run(cmd + ["-map", "0" if len(audio) + video == 1 else "0"] + [
        a for i in range(1, len(audio) + video) for a in ("-map", str(i))] + [str(path)], capture_output=True)
    if done.returncode:
        pytest.skip(f"this ffmpeg cannot make {path.suffix}: {done.stderr.decode()[:100]}")


needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


@needs_ffmpeg
@pytest.mark.parametrize("ext", ["mp4", "mov", "m4v", "mkv", "webm", "avi", "3gp", "mpg", "flv", "ts"])
def test_load_audio_takes_the_sound_out_of_common_video_formats(tmp_path, ext):
    video = tmp_path / f"clip.{ext}"
    _make(video, (440, 2))
    audio = mt.load_audio(video)
    assert 1.7 * mt.RATE < len(audio) < 2.4 * mt.RATE and abs(audio).max() > 0.05


@needs_ffmpeg
def test_video_with_several_audio_tracks_uses_the_first_or_the_chosen_one(tmp_path):
    video = tmp_path / "two.mp4"
    _make(video, (440, 2), (880, 3))
    assert mt.audio_tracks(video) == 2
    first, second = mt.load_audio(video), mt.load_audio(video, track=1)
    assert len(first) < 2.4 * mt.RATE and len(second) > 2.7 * mt.RATE
    with pytest.raises(RuntimeError, match="has 2 audio track"):
        mt.load_audio(video, track=2)


@needs_ffmpeg
def test_silent_video_is_reported_not_crashed(tmp_path):
    video = tmp_path / "silent.mp4"
    _make(video)
    assert mt.audio_tracks(video) == 0
    with pytest.raises(RuntimeError, match="no audio track"):
        mt.load_audio(video)


# ---------- the Mac app talks to mt.py through "@@KIND<TAB>fields" lines ----------
def test_events_mode_reports_files_outputs_errors_and_done(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    good, bad = tmp_path / "a.txt", tmp_path / "b.txt"

    def work(src):
        if src == bad:
            raise RuntimeError("no audio\ttrack\nhere")
        mt.announce(src.with_suffix(".out"))                              # as write_outputs does for each file it writes
        return [src.with_suffix(".out")]

    with pytest.raises(SystemExit) as exit_info:
        mt._each([good, bad], work)
    lines = capsys.readouterr().out.splitlines()
    assert exit_info.value.code == 1
    assert lines == [f"@@FILE\t1\t2\t{good}", f"@@OUT\t{good.with_suffix('.out')}",
                     f"@@FILE\t2\t2\t{bad}", f"@@ERR\t{bad}\tno audio track here", "@@DONE\t1"]


def test_event_fields_keep_unicode_spaces_in_paths(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    name = "Screen Recording 2026-11-03 at 10.00.00\u202fAM.transcript.txt"      # macOS puts U+202F before AM/PM
    mt.ev("OUT", "/Users/me/" + name)
    assert capsys.readouterr().out == "@@OUT\t/Users/me/" + name + "\n"


def test_plain_mode_prints_only_the_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", False)
    mt._each([tmp_path / "a.txt"], lambda src: mt.announce(src.with_suffix(".out")))
    assert capsys.readouterr().out == f"{tmp_path / 'a.out'}\n"


# ---------- offline translation (NLLB on this computer) ----------
def test_nllb_code_accepts_names_whisper_codes_and_raw_nllb_codes():
    assert [mt.nllb_code(x) for x in ("English", "en", " RUSSIAN ", "Brazilian Portuguese", "zh", "Traditional Chinese",
                                      "ces_Latn", "Farsi", "jw")] == [
        "eng_Latn", "eng_Latn", "rus_Cyrl", "por_Latn", "zho_Hans", "zho_Hant", "ces_Latn", "pes_Arab", "jav_Latn"]
    assert mt.nllb_code("Klingon") is None and mt.nllb_code("") is None and mt.nllb_code(None) is None


def test_every_language_code_of_the_table_is_in_the_models_vocabulary():
    """Needs the model's tokenizer in the local cache (it is not downloaded in CI), so it is skipped without it."""
    try:
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(snapshot_download(mt.NLLB, local_files_only=True) + "/tokenizer.json")
    except Exception:
        pytest.skip("the translation model is not in the local cache")
    assert [c for c in mt.NLLB_LANGS if tok.token_to_id(c) is None] == []


def test_sentences_split_at_sentence_ends_only():
    assert mt.sentences("Good day. How are you? Fine!") == ["Good day.", "How are you?", "Fine!"]
    assert mt.sentences("Version 3.5 is out") == ["Version 3.5 is out"]          # a dot inside a number is not a sentence end
    assert mt.sentences("你好。今天好吗？很好") == ["你好。", "今天好吗？", "很好"]
    assert mt.sentences("no end") == ["no end"] and mt.sentences("...") == ["..."]


class FakeTokenizer:
    vocab = ["<pad>"]

    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(tokens=text.split())

    def token_to_id(self, token):
        if token not in self.vocab:
            self.vocab.append(token)
        return self.vocab.index(token)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.vocab[i] for i in ids)


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate_batch(self, tokens, target_prefix, **kw):
        self.calls.append((tokens, target_prefix))
        return [SimpleNamespace(hypotheses=[[prefix[0]] + ["T" + t for t in sent[1:-1]]])
                for sent, prefix in zip(tokens, target_prefix)]


def _fake_model(monkeypatch):
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda: (translator, FakeTokenizer()))
    return translator


def test_translate_offline_groups_by_source_language_and_keeps_same_language_and_unknown_lines(monkeypatch, capsys):
    translator = _fake_model(monkeypatch)
    doc = mt.parse_transcript("[00:00:01] [ru] privet mir. kak dela\n[00:00:02] [en] hello\n[00:00:03] [xx] ???\nplain\n")
    out = mt.translate(doc, "English", offline=True)
    assert out == ["Tprivet Tmir. Tkak Tdela", "hello", "???", "plain"]    # two sentences translated apart, the rest kept
    sources = {sentence[0] for tokens, _ in translator.calls for sentence in tokens}
    assert sources == {"rus_Cyrl"}                                        # only the Russian line was sent to the model
    assert all(prefix == [["eng_Latn"]] * len(prefix) for _, prefix in translator.calls)
    assert "2 of unknown language kept as they are (give the language with --from)" in capsys.readouterr().err
    assert f"eng_Latn [{mt.NLLB}]" in doc["translations"] and "English" not in doc["translations"]     # separate cache key


def test_translate_offline_uses_the_source_option_for_plain_lines_and_no_spaces_for_chinese(monkeypatch):
    _fake_model(monkeypatch)
    doc = mt.parse_transcript("Hola amigo. Adios\n")
    assert mt.translate(doc, "Spanish", offline=True, source="es") == ["Hola amigo. Adios"]     # already Spanish
    assert mt.translate(doc, "Chinese", offline=True, source="es") == ["THola Tamigo.TAdios"]    # 2 sentences, joined without a space


def test_offline_targets_are_checked_before_anything_starts(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mt._need_key(["English", "ru"], offline=True)               # no key needed
    with pytest.raises(SystemExit, match="does not know 'Klingon'"):
        mt._need_key(["English", "Klingon"], offline=True)
    with pytest.raises(SystemExit, match="or use --offline"):
        mt._need_key(["English"], offline=False)


def test_sentences_do_not_split_after_abbreviations_initials_numbers_or_before_a_lowercase_word():
    assert mt.sentences("Dr. Smith met Mr. Lee at 3.30 p.m. in Washington D.C. yesterday. Then he left.") == \
        ["Dr. Smith met Mr. Lee at 3.30 p.m. in Washington D.C. yesterday.", "Then he left."]
    assert mt.sentences("Das Treffen ist am 3. Mai. Wir sehen uns z.B. dort. Gut.") == ["Das Treffen ist am 3. Mai.", "Wir sehen uns z.B. dort.", "Gut."]
    assert mt.sentences("Он пришёл, т.е. вовремя. Хорошо.") == ["Он пришёл, т.е. вовремя.", "Хорошо."]
    assert mt.sentences("Hello. World.") == ["Hello.", "World."]


def test_long_sentences_are_cut_at_clause_punctuation_then_by_length():
    tok = FakeTokenizer()
    short = "a few words only"
    assert mt.split_long(short, tok, limit=200) == [short]
    long = ", ".join(f"clause number {i} with some words" for i in range(60))      # 360 words, split at commas
    parts = mt.split_long(long, tok, limit=50)
    assert all(len(p.split()) <= 50 for p in parts) and " ".join(parts).replace("  ", " ").split() == long.split()
    run_on = " ".join(f"w{i}" for i in range(120))                                 # no punctuation: halves
    parts = mt.split_long(run_on, tok, limit=50)
    assert all(len(p.split()) <= 50 for p in parts) and " ".join(parts).split() == run_on.split()


def test_the_whisper_languages_the_model_knows_are_all_translatable():
    for code in "as ba bo fo ht ln mg mi oc ps sa sd sn su tk tt yi".split():
        assert mt.nllb_code(code), code
    assert mt.nllb_code("tt") == "tat_Cyrl" and mt.nllb_code("yiddish") == "ydd_Hebr"
    assert mt.nllb_code("br") is None and mt.nllb_code("haw") is None and mt.nllb_code("la") is None   # not in NLLB-200


def test_windowed_vad_never_splits_a_stretch_of_speech(monkeypatch):
    import numpy as np
    _content_vad(monkeypatch)
    audio = _speech(40, gaps=set(range(0, 40)) - {3, 4, 5, 9, 10, 11, 12, 20, 21, 30, 31, 32}).reshape(-1).astype("float32")
    spans = mt.speech_spans(audio, 0, 700, window_s=10)                             # windows of 10 s; speech 9-13 s crosses one
    assert [(s["start"] // 16000, s["end"] // 16000) for s in spans] == [(3, 6), (9, 13), (20, 22), (30, 33)]
    assert spans == mt.speech_spans(audio, 0, 700, window_s=3600)                   # the same as one big window


def test_translate_takes_a_stored_meeting_folder_and_writes_beside_it(monkeypatch, tmp_path, capsys):
    _fake_model(monkeypatch)
    folder = tmp_path / "20261110-103728-prm"; folder.mkdir()          # not prn: PRN.txt is the printer on older Windows
    mt._save(folder, {"source": "s", "whisper": "w", "lines": [{"t": 1.0, "lang": "ru", "conf": 1.0, "text": "Privet"}], "translations": {}})
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", str(folder), "--offline", "--to", "English"])
    monkeypatch.setattr(mt, "EVENTS", False)
    mt.main()
    assert (folder / "prm.English.txt").read_text().strip() == "[00:00:01] [ru] TPrivet"
    assert json.loads((folder / "transcript.json").read_text())["translations"]           # cached in the folder


def test_preflight_refuses_a_recording_the_disk_cannot_hold(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=100 * 2**20))
    with pytest.raises(SystemExit, match="Not enough free disk"):
        mt.preflight(hours=4, tracks=1)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=10 * 2**30))
    mt.preflight(hours=4, tracks=2)                                                    # 1.8 GB needed: fine


def test_tidy_closes_gaps_before_punctuation_except_in_french_and_cjk():
    assert mt.tidy("Hello , my friends .", "eng_Latn") == "Hello, my friends."
    assert mt.tidy("Are you ready ?", "fra_Latn") == "Are you ready ?"
    assert mt.tidy("你好 , 朋友 .", "zho_Hans") == "你好 , 朋友 ."


# ---------- several spoken languages: detection restricted to the chosen set ----------
def test_whisper_code_accepts_codes_names_and_nllb_codes():
    assert [mt.whisper_code(x) for x in ("ru", " EN ", "Russian", "Brazilian Portuguese", "Traditional Chinese", "Norwegian",
                                         "Javanese", "Cantonese", "rus_Cyrl", "yue")] == [
        "ru", "en", "ru", "pt", "zh", "no", "jw", "yue", "ru", "yue"]
    assert mt.whisper_code("Klingon") is None and mt.whisper_code("") is None and mt.whisper_code(None) is None


def test_every_whisper_language_can_be_chosen_by_name():
    assert set(mt.WHISPER_NAMES) == mt.WHISPER_LANGS                        # a name for every one of Whisper's languages
    for code, names in mt.WHISPER_NAMES.items():
        for name in names.split("|"):
            assert mt.whisper_code(name) == code and mt.whisper_code(name.upper()) == code, (code, name)
    assert [mt.whisper_code(x) for x in ("Latin", "Hawaiian", "Tibetan", "Haitian Creole", "Maori", "Castilian")] == [
        "la", "haw", "bo", "ht", "mi", "es"]
    assert mt.parse_languages("Latin, Hawaiian, Tibetan") == ["la", "haw", "bo"]


def _app_languages(name):
    """The names of one of the app's language lists (Languages.spokenAll / targetAll in JobSettings.swift)."""
    swift = (Path(__file__).parent.parent / "app" / "Sources" / "Core" / "JobSettings.swift").read_text(encoding="utf-8")
    quoted = lambda text: set(re.findall(r'"([^"]+)"', text))
    if name == "spokenAll":
        return quoted(re.search(r"static let spokenAll = menuOrder\(\[(.*?)\]\)", swift, re.S).group(1))
    # targetAll = menuOrder(spokenAll.filter { ![excluded].contains($0) } + [added])
    excluded, added = re.search(r"static let targetAll = menuOrder\(spokenAll\.filter \{ !\[(.*?)\]\.contains\(\$0\) \}"
                                r"\s*\+ \[(.*?)\]\)", swift, re.S).groups()
    return (_app_languages("spokenAll") - quoted(excluded)) | quoted(added)


def test_every_language_the_app_offers_is_one_mt_py_accepts():
    spoken = _app_languages("spokenAll")
    assert len(spoken) == len(mt.WHISPER_LANGS) == 100
    assert {mt.whisper_code(name) for name in spoken} == mt.WHISPER_LANGS, "one name for each of Whisper's languages"
    assert mt.parse_languages(", ".join(sorted(spoken))) is not None
    targets = _app_languages("targetAll")
    unknown = [name for name in targets if not mt.nllb_code(name)]
    assert not unknown, f"the offline model does not know {unknown}"
    assert len({(mt.nllb_code(name), mt.gemma_code(name), mt.hy_code(name)) for name in targets}) == len(targets), \
        "two names for one language"
    assert sum(1 for name in targets if mt.hy_code(name)) == 92, "Hy-MT3 takes 92 of the app's targets, NLLB the others"
    assert {"pt-BR", "pt-PT", "zh-TW"} <= {mt.gemma_code(name) for name in targets}, "TranslateGemma's regional targets"
    assert all(mt.nllb_code(name) is None for name in ("Breton", "Hawaiian", "Latin")), "the three left out of the targets"


def test_whisper_names_and_translation_aliases_agree_where_they_overlap():
    for nllb, names in mt.NLLB_LANGS.items():
        for alias in names.split("|"):
            by_whisper = mt._WHISPER_BY_NAME.get(re.sub("[^a-z]", "", alias))
            if by_whisper and nllb != "zho_Hant":
                assert by_whisper == mt.whisper_code(nllb), (alias, nllb)       # one name never points at two languages


def test_parse_languages_lists_dedupes_and_rejects_unknown_names():
    assert mt.parse_languages("") == [] and mt.parse_languages(None) == []
    assert mt.parse_languages("ru, English;de\nrussian") == ["ru", "en", "de"]
    with pytest.raises(ValueError, match="does not know 'Klingon', 'Elvish'"):
        mt.parse_languages("ru, Klingon, Elvish")


def test_whisper_language_table_matches_faster_whisper():
    tokenizer = pytest.importorskip("faster_whisper.tokenizer")
    assert set(tokenizer._LANGUAGE_CODES) == mt.WHISPER_LANGS


def test_every_translation_language_can_be_chosen_as_spoken_except_those_whisper_cannot_transcribe():
    missing = [c for c in mt.NLLB_LANGS if mt.whisper_code(c) is None]
    assert missing == ["zul_Latn"]    # Zulu can be translated to and from, but Whisper has no Zulu


def test_best_allowed_takes_the_likeliest_chosen_language_with_its_own_probability():
    probs = [("de", 0.8), ("en", 0.1), ("ru", 0.05)]
    assert mt.best_allowed(probs, ["ru", "en"]) == ("en", 0.1)       # German is not chosen: the guess is weak
    assert mt.best_allowed(probs, ["de", "en"]) == ("de", 0.8)
    assert mt.best_allowed(None, ["fr", "es"]) == ("fr", 0.0)        # no detection data: the first, tagged as a guess


def _spans_for(audio, spans):
    """The VAD stand-in: the scripted spans for the whole audio; a chunk handed back for an adaptive re-cut is one span
    (no pause inside it), so the tests that script chunk-level probabilities keep one language per chunk."""
    return spans if len(audio) > 60 * 16000 else [{"start": 0, "end": len(audio)}]


class FakeWhisper:
    """Chunk 1 is German, chunk 2 is Russian, chunk 3 is English (the test audio carries each chunk's start second in
    its samples); it records how each chunk was decoded: (chunk, language asked for, language decoded)."""
    DETECT = [("de", {"de": .9, "en": .06, "ru": .02}), ("ru", {"ru": .8, "en": .1, "de": .05}),
              ("en", {"en": .95, "de": .02, "ru": .01})]

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, language=None, **kw):
        chunk = int(audio[0]) // 20
        detected, probs = self.DETECT[chunk]
        decoded = language or detected
        self.calls.append((chunk + 1, language, decoded))
        info = SimpleNamespace(language=decoded, language_probability=probs[decoded], all_language_probs=list(probs.items()))
        return iter([SimpleNamespace(start=0.0, text=f" chunk{chunk + 1}-{decoded}")]), info


def _run_transcribe(monkeypatch, tmp_path, language):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    audio = np.repeat(np.arange(70, dtype="float32"), 16000)         # sample value = its second, so a chunk shows where it starts
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio)
    model = FakeWhisper()
    monkeypatch.setattr(mt, "get_model", lambda: model)
    mt.get_engine.cache_clear()                                       # the engine wraps the model of the previous test otherwise
    try:
        doc = mt.transcribe(tmp_path / "x.wav", tmp_path, language)
    finally:
        mt.get_engine.cache_clear()
    return doc, model


def test_transcribe_detects_among_all_languages_by_default(monkeypatch, tmp_path):
    doc, model = _run_transcribe(monkeypatch, tmp_path, None)
    assert [(l["lang"], l["conf"]) for l in doc["lines"]] == [("de", 0.9), ("ru", 0.8), ("en", 0.95)]
    assert [c[1] for c in model.calls] == [None, None, None]


def test_transcribe_with_several_languages_keeps_a_sure_unlisted_language_as_a_guess(monkeypatch, tmp_path):
    doc, model = _run_transcribe(monkeypatch, tmp_path, "ru, en")
    assert [(l["lang"], l["conf"]) for l in doc["lines"]] == [("de", 0.9), ("ru", 0.8), ("en", 0.95)]
    assert doc["lines"][0]["text"].endswith("-de") and doc["lines"][0]["guess"]   # German was not listed, but Whisper is sure: kept
    assert model.calls == [(1, None, "de"), (2, None, "ru"), (3, None, "en")]     # no second pass
    assert mt.fmt(doc["lines"][0]) == "[00:00:00] [de?] chunk1-de"    # shown as a guess, in its own language
    assert "guess" not in doc["lines"][1] and doc["lines"][1]["end"] == 20.0


def test_transcribe_with_several_languages_falls_back_to_the_likeliest_listed_one_when_unsure(monkeypatch, tmp_path):
    class Unsure(FakeWhisper):
        DETECT = [("fr", {"fr": .5, "en": .3, "ru": .1}), *FakeWhisper.DETECT[1:]]
    monkeypatch.setattr(mt, "FakeWhisper", Unsure, raising=False)
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(70, dtype="float32"), 16000))
    model = Unsure()
    monkeypatch.setattr(mt, "get_model", lambda: model)
    mt.get_engine.cache_clear()
    try:
        doc = mt.transcribe(tmp_path / "x.wav", tmp_path, "ru, en")
    finally:
        mt.get_engine.cache_clear()
    assert (doc["lines"][0]["lang"], doc["lines"][0]["conf"]) == ("en", 0.3) and "guess" not in doc["lines"][0]
    assert model.calls[:3] == [(1, None, "fr"), (1, "en", "en"), (1, "ru", "ru")]   # unsure (< 0.5): the runner-up is tried too
    assert mt.fmt(doc["lines"][0]).startswith("[00:00:00] [en?] ")


def test_choose_language_rules():
    assert mt.choose_language([("de", .9), ("en", .05)], ["ru", "en"]) == ("de", 0.9, True)      # sure of an unlisted language
    assert mt.choose_language([("de", .6), ("en", .3)], ["ru", "en"]) == ("en", 0.3, False)      # not sure: the best listed one
    assert mt.choose_language([("en", .6), ("de", .3)], ["ru", "en"]) == ("en", 0.6, False)      # listed: taken as is
    assert mt.choose_language([("de", .9)], []) == ("de", 0.9, False)                           # no list: anything goes
    assert mt.choose_language(None, ["ru"]) == ("ru", 0.0, True)


def test_notes_are_written_as_comment_lines_kept_in_every_file_and_never_translated(monkeypatch):
    doc = {"lines": [{"t": 1.0, "lang": "en", "conf": 1.0, "text": "Hello."},
                     {"t": 4.0, "note": "speech not recognised for 6 s"},
                     {"t": 12.5, "end": 14.0, "lang": "de", "conf": 0.9, "guess": True, "text": "Danke."},
                     {"t": 20.0, "note": "decoded 80 % of the speech found (6 s not recognised)"}], "translations": {}}
    text = mt.render(doc)
    assert text.splitlines() == ["[00:00:01] [en] Hello.", "# [00:00:04] speech not recognised for 6 s",
                                 "[00:00:12] [de?] Danke.", "# [00:00:20] decoded 80 % of the speech found (6 s not recognised)"]
    again = mt.parse_transcript(text)["lines"]
    assert again[1] == {"t": 4, "note": "speech not recognised for 6 s"} and again[2]["conf"] == 0.0
    monkeypatch.setattr(mt, "translate_offline", lambda d, target, source=None, checkpoint=None: [l.get("text") and "X" for l in d["lines"]])
    assert mt.render(doc, mt.translate(doc, "Russian", offline=True)).splitlines()[1] == "# [00:00:04] speech not recognised for 6 s"
    merged = mt.merge_call([{"t": 1.0, "note": "n"}], [{"t": 2.0, "lang": "en", "conf": 1.0, "text": "Yes, that works for me."}])
    assert [l.get("note") or l["text"] for l in merged] == ["n", "Yes, that works for me."]  # a note is never an echo


class FakeEngine:
    """Scripted probabilities and segments per chunk (chunk index from the audio's first sample, as FakeWhisper)."""
    name = "fake"

    def __init__(self, script):
        self.script, self.calls = script, []

    def analyse(self, chunk, detect=True):
        probs, _ = self.script[int(chunk[0]) // 20]
        return SimpleNamespace(chunk=chunk, probs=probs if detect else None)

    def decode(self, state, language):
        _, segs = self.script[int(state.chunk[0]) // 20]
        self.calls.append(language)
        return [dict(s) for s in segs]


def test_unrecognised_chunks_become_notes_with_a_coverage_note_and_progress_events(monkeypatch, tmp_path, capsys):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(70, dtype="float32"), 16000))
    seg = lambda text: {"start": 0.5, "end": 2.0, "text": text, "logprob": -0.2}
    engine = FakeEngine([([("en", .9)], [seg(" one")]), ([("en", .8)], []), ([("ru", .9)], [seg(" три")])])
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "ETA_AFTER", 2)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")                  # the remembered speed must not land in the repo
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en,ru")
    assert [l.get("text") or l["note"] for l in lines] == ["one", "speech not recognised for 6 s", "три",
                                                            "decoded 67 % of the speech found (6 s not recognised)"]
    assert lines[0] == {"t": 0.5, "end": 2.0, "lang": "en", "conf": 0.9, "text": "one"}
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "@@PROG\t1\t3" and out[1] == "@@POS\t6.0\t70.0" and out[2].startswith("@@LINE\t{")
    assert any(o.startswith("@@ETA\t") for o in out) and out[-1].startswith("@@NOTE\tdecoded 67 %")
    assert any(o.startswith("@@LINE\t") and "speech not recognised" in o for o in out)


# ---------- what Whisper makes up on noise (subtitle credits, outros) ----------
def _said(text, t=0.0, end=None):
    return {"t": t, "end": t + 2.0 if end is None else end, "lang": "ru", "conf": 0.9, "text": text}


@pytest.mark.parametrize("text", ["Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok.", "Редактор субтитров А.Синецкая Корректор А.Егорова",
                                  "Редактор субтитров Е.Воинова Корректор А.Кулакова", "Корректор А.Егорова", "Спасибо за субтитры Алексею Дубровскому!",
                                  "Subtitles by the Amara.org community", "Untertitel im Auftrag des ZDF, 2020", "Sottotitoli creati dalla comunità Amara.org",
                                  "Napisy stworzone przez społeczność Amara.org", "ДИНАМИЧНАЯ МУЗЫКА", "Продолжение следует... Субтитры сделал DimaTorzok"])
def test_subtitle_credits_are_made_up_whatever_their_length(text):
    assert mt.made_up(_said(text), alone=False) == "a subtitle credit"


@pytest.mark.parametrize("text", ["Thank you.", "Спасибо.", "Danke.", "Да.", "Субтитры к фильму мы сделаем сами.", "Продолжение следует за вступлением.",
                                  "Thank you for watching the demo, any questions?", "We use Amara.org for our subtitles at work.",
                                  "Редактор субтитров у нас Аня.", "Корректор нашёл три ошибки.", "Vielen Dank fürs Zuhören."])
def test_real_speech_is_never_made_up_even_stretched_over_silence(text):
    assert mt.made_up(_said(text, 0.0, 25.0), alone=True) is None


def test_an_outro_is_made_up_only_when_it_stands_alone_stretched_over_near_silence():
    stretched, spoken = _said("Продолжение следует...", 0.0, 29.9), _said("Продолжение следует...", 0.0, 1.4)
    assert mt.made_up(stretched, alone=True) == "an outro over near-silence"
    assert mt.made_up(spoken, alone=True) is None, "said at speaking speed: a real video's outro"
    assert mt.made_up(stretched, alone=False) is None, "beside other lines of its chunk: real"
    assert mt.made_up(_said("Thanks for watching!", 0.0, 12.0), alone=True) == "an outro over near-silence"


def test_a_credit_at_the_end_of_a_real_line_is_trimmed_off():
    lines = mt.screen_made_up([_said("Это был отличный доклад. Субтитры сделал DimaTorzok")])
    assert lines[0]["text"] == "Это был отличный доклад." and "made_up" not in lines[0]
    marked = mt.screen_made_up([_said("Субтитры сделал DimaTorzok", 0.0, 29.0), {"t": 30.0, "note": "n"}])
    assert marked[0]["made_up"] == "a subtitle credit" and marked[1] == {"t": 30.0, "note": "n"}


def test_made_up_lines_are_left_out_of_the_track_with_one_note_and_no_notes_of_their_own(monkeypatch, tmp_path, capsys):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40, 60, 80)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(90, dtype="float32"), 16000))
    seg = lambda text, end=2.0: {"start": 0.5, "end": end, "text": text, "logprob": -0.3}
    engine = FakeEngine([([("ru", .9)], [seg(" Привет, начнём.")]), ([("ru", .9)], [seg(" Субтитры сделал DimaTorzok", 6.0)]),
                         ([("ru", .9)], [seg(" Продолжение следует...", 1.5)]), ([("ru", .9)], [seg(" Продолжение следует...", 1.5)]),
                         ([("ru", .9)], [seg(" Продолжение следует...", 1.5)])])
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    lines = mt.transcribe_lines(tmp_path / "x.wav", "ru")
    assert [l.get("text") or l["note"] for l in lines] == [
        "Привет, начнём.", "4 lines that Whisper makes up on noise (subtitle credits, outros said again and again) were left out"]
    out = capsys.readouterr().out
    assert "DimaTorzok" not in out and "speech not recognised" not in out, "never shown, and no note per chunk"
    shown = out.split("@@CLEAR\n")
    assert len(shown) == 2, "the repeated outros were shown as they were decoded: the window is told to start again"
    assert "Продолжение" not in shown[1] and "Привет, начнём." in shown[1]


def test_an_outro_left_out_as_stretched_counts_toward_its_repetition(monkeypatch, tmp_path, capsys):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 12)} for s in (0, 20, 40, 60)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(80, dtype="float32"), 16000))
    seg = lambda text, end: {"start": 0.0, "end": end, "text": text, "logprob": -0.3}
    engine = FakeEngine([([("ru", .9)], [seg(" Привет, начнём.", 2.0)]), ([("ru", .9)], [seg(" Продолжение следует...", 10.0)]),
                         ([("ru", .9)], [seg(" Продолжение следует...", 1.4)]), ([("ru", .9)], [seg(" Продолжение следует...", 1.4)])])
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    lines = mt.transcribe_lines(tmp_path / "x.wav", "ru")
    assert [l.get("text") or l["note"] for l in lines] == [
        "Привет, начнём.", "3 lines that Whisper makes up on noise (subtitle credits, outros said again and again) were left out"]
    kept, again = mt.drop_repeated_outros([_said("Продолжение следует...", 40.0, 41.4)], left_out=[_said("Продолжение следует...", 0.0, 10.0)])
    assert (kept, again) == ([_said("Продолжение следует...", 40.0, 41.4)], 0), "twice in all is not again and again"


def test_a_live_track_neither_shows_nor_translates_a_made_up_line_but_keeps_it_for_the_count(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    seen = []
    track = SimpleNamespace(pending=[(0, 16000 * 6)], chunks=[(0, 16000 * 6)], base=0, audio=list(range(16000 * 6)), allowed=["ru"],
                            forced="ru", speaker=None, granularity="normal", covered=16000 * 6, fed=16000 * 6, _settle=lambda: None,
                            cp=SimpleNamespace(record=lambda i, lines: seen.append(lines)))
    monkeypatch.setattr(mt, "decode_chunk", lambda *a, **k: [dict(_said("Субтитры сделал DimaTorzok"), made_up="a subtitle credit"),
                                                              _said("Добрый день.", 3.0)])
    translated = []
    shown = mt.LiveTrack.decode(track, None, translate=lambda lines: translated.extend(l["text"] for l in lines))
    assert [l["text"] for l in shown] == ["Добрый день."] and translated == ["Добрый день."]
    assert "DimaTorzok" not in capsys.readouterr().out
    assert [l.get("made_up") for l in seen[0]] == ["a subtitle credit", None], "the checkpoint keeps it, for the final count"


VIDEO_PHRASES = ["I can see you in the next video.", "I'll see you in the next episode.", "Danke fürs Zuschauen.", "Продолжение следует...",
                 "Thanks for watching!", "Thank you for watching.", "Подпишитесь на мой канал.", "Подписывайтесь на мой канал!",
                 "Ставьте лайки и подписывайтесь на канал.", "Субтитры сделал Иван Петров", "Субтитры подготовил Симон", "Fortsetzung folgt.",
                 "Продовження буде...", "Спасибо за просмотр!", "Wenn ja, dann lasst mir einen Daumen da und abonniert meinen Kanal.",
                 "Music playing", "Субтитры подготовлены Иваном Петровым", "SWR 2021", "Hello everyone, welcome to my channel.",
                 "Всем привет и добро пожаловать на мой канал."]


@pytest.mark.parametrize("text", VIDEO_PHRASES)
def test_in_a_recording_a_videos_outro_or_channel_plug_is_dropped_by_its_text(text):
    """A meeting does not say "Thanks for watching" or "Подпишитесь на канал": recorded, such a line is one Whisper made
    up, even at speaking speed beside other lines (1.11.4: they showed live on short chunks, never stretched)."""
    assert mt.made_up(_said(text, 0, 1.4), alone=False, audio_end=1.5, context="recording") == "a video's outro or channel plug"


@pytest.mark.parametrize("text", VIDEO_PHRASES)
def test_in_an_imported_file_a_videos_outro_is_dropped_only_with_a_sign_that_nobody_said_it(text):
    """An imported file may be that video: its outro is said. Only a decode Whisper itself failed hides it."""
    assert mt.made_up(_said(text, 0, 1.4), alone=True, audio_end=1.5) is None
    assert mt.made_up(_said(text, 0, 1.4), alone=True, audio_end=1.5, context="media") is None
    failed = dict(_said(text, 0, 1.4), _lp=-1.2, _temp=1.0)
    assert mt.made_up(failed, alone=False, audio_end=1.5, context="media") == "a video's phrase Whisper itself rejected"
    assert mt.made_up(dict(failed, end=30.0), alone=False, audio_end=1.5, context="media") == "an outro over near-silence", \
        "timed into the padding: that sign is named first"


NEW_CREDITS = ["Субтитры А.Синецкая", "Корректор .Кулакова", "Субтитры и корректура Оксаны Каменской", "www.northstarit.co.uk",
               "This video is a derivative work of the Touhou Project.", "[Music]", "[APPLAUSE]", "* Musik *", "*Klopfen*", "♪♪", "(Musik)",
               "♪ Musik ♪", "Корректор.Кулакова"]
OLD_CREDITS = ["Субтитры сделал DimaTorzok", "Субтитры создавал DimaTorzok.", "Редактор субтитров А.Синецкая Корректор А.Егорова",
               "Редактор субтитров Е.Воинова Корректор А.Кулакова", "Корректор А.Егорова", "Спасибо за субтитры Алексею Дубровскому!",
               "Subtitles by the Amara.org community", "Untertitel im Auftrag des ZDF, 2020", "Sottotitoli creati dalla comunità Amara.org",
               "Napisy stworzone przez społeczność Amara.org", "ДИНАМИЧНАЯ МУЗЫКА", "Продолжение следует... Субтитры сделал DimaTorzok"]


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", NEW_CREDITS + OLD_CREDITS)
def test_subtitle_credits_and_bracketed_sound_captions_are_made_up_in_every_context(text, context):
    assert mt.made_up(_said(text), alone=False, context=context) == "a subtitle credit"


# The REAL sentences of the 1.11.4 phrase study (scratchpad research/phrases.py): what meetings say that looks like an outro
STUDY_REAL = ["Спасибо.", "Спасибо большое.", "Спасибо за внимание.", "Всем спасибо, до свидания.",
              "Продолжение следует на следующей неделе, когда придут цифры.", "Мы решили, что продолжение следует завтра.",
              "Субтитры к фильму мы сделаем сами.", "Корректор нашёл три ошибки в договоре.", "Подпишите, пожалуйста, договор.",
              "Музыка в рекламе слишком громкая.", "Редактор субтитров у нас один.", "Thank you.", "Thanks.", "Okay, thank you, bye.",
              "I'm sorry, could you repeat that?", "Thanks for watching the demo, any questions?", "Please subscribe to the mailing list.",
              "Subtitles by Friday would be great.", "We'll be right back after the break.", "Vielen Dank.", "Danke schön.",
              "Bis zum nächsten Mal, dann mit den Zahlen.", "Das war's für heute.", "Untertitel brauchen wir für das Video auch.",
              "Musik machen wir später.", "Барсук, живущий в Киевском зоопарке, совершил побег из своего вольера.",
              "Ask not what your country can do for you. Ask what you can do for your country.",
              "Alles hat ein Ende, nur die Wurst hat zwei.",
              "Raptorium-Bergbau scheint profitierter als Monroe als Reaktion auf die wirtschaftlichen Ausfälle zu sein."]
REAL_REPLIES = ["Thank you.", "Спасибо.", "Danke.", "Да.", "Vielen Dank.", "Большое спасибо.", "Gracias.", "Bye.", "Музыка.",
                "Спасибо за внимание.", "Спасибо за внимание и до новых встреч.", "Bis zum nächsten Mal.", "Bis zum nächsten Mal. Tschüss.",
                "Das war's für heute.", "I'll see you next time.", "See you next week.", "Okay, see you tomorrow.",
                "Thank you for watching the demo, any questions?", "Please subscribe to the mailing list.", "Musik machen wir später.",
                "Продолжение следует на следующей неделе, когда придут цифры.", "Продолжение следует за вступлением.",
                "Vielen Dank fürs Zuhören.", "Субтитры к фильму мы сделаем сами.", "We use Amara.org for our subtitles at work.",
                "Редактор субтитров у нас Аня.", "Корректор нашёл три ошибки."]


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", sorted(set(REAL_REPLIES + STUDY_REAL)))
def test_real_replies_and_meeting_outros_are_kept_in_every_context_even_when_mumbled(text, context):
    """Short thanks and the outros people say in meetings are never dropped by their text, alone or not, and a failed
    decode (speech under mumble) is no sign against them, not even for a whole chunk."""
    for alone in (True, False):
        line = dict(_said(text, 0, 1.8), _lp=-0.3, _temp=0.0)
        assert mt.made_up(line, alone=alone, audio_end=2.0, context=context) is None
        assert mt.screen_made_up([line, *([] if alone else [_said("Und weiter.", 2.0)])], 2.0, context)[0]["text"] == text
        mumbled = dict(_said(text, 0, 1.8), _lp=-1.3, _temp=1.0)
        assert mt.made_up(mumbled, alone=alone, audio_end=2.0, context=context) is None
        assert mt.screen_made_up([dict(mumbled)], 2.0, context)[0] == _said(text, 0, 1.8), "a chunk of failed decodes is kept"


def test_a_short_thanks_is_dropped_only_alone_and_timed_past_the_end_of_its_audio():
    """Whisper put 'Thank you.' at 0-30 s on a 1.5 s chunk: the words are in the zero padding, nobody said them. Real
    lines ended at most 0.51 s past their audio in the harvest, 1.62 s on 1-1.4 s chunks (the threshold is 5 s)."""
    long = _said("Thank you.", 0, 30.0)
    assert mt.made_up(long, alone=True, audio_end=1.5) == "thanks over near-silence"
    assert mt.made_up(long, alone=False, audio_end=1.5) is None, "beside another line of its chunk: kept"
    assert mt.made_up(long, alone=True, audio_end=29.5) is None, "0.5 s past the audio is how real lines end"
    assert mt.made_up(long, alone=True, audio_end=25.5) is None, "4.5 s past it is still not the padding's end"
    assert mt.made_up(dict(_said("Thank you.", 0, 1.4), _lp=-1.3, _temp=1.0), alone=True, audio_end=1.5) is None, "a failed decode alone is no sign"
    assert mt.made_up(long, alone=True) is None, "without the audio's end (a checkpoint's line) nothing changes"
    assert mt.made_up(_said("Спасибо за внимание.", 0, 30.0), alone=False, audio_end=2.0) == "an outro over near-silence"
    assert mt.made_up(_said("Спасибо за внимание.", 0, 2.0), alone=True, audio_end=2.0) is None
    assert mt.made_up(_said("Bis zum nächsten Mal.", 0, 25.0), alone=True) == "an outro over near-silence", "the stretch rule stays"


def test_an_outro_glued_to_a_real_line_comes_off_only_when_the_line_runs_past_its_audio():
    glued = "Хорошо, договорились. Продолжение следует..."
    assert mt.screen_made_up([_said(glued, 0, 30.0)], 3.0)[0]["text"] == "Хорошо, договорились."
    assert mt.screen_made_up([_said(glued, 0, 3.0)], 3.0, "media")[0]["text"] == glued, "an imported file: it may be said"
    # In a recording a video's phrase at the end comes off by its text, as a whole line of one is dropped (1.11.4
    # review); a meeting's outro, a goodbye or a short thanks there still needs the line to run past its audio
    assert mt.screen_made_up([_said(glued, 0, 3.0)], 3.0, "recording")[0]["text"] == "Хорошо, договорились."
    for outro in ("Спасибо за внимание.", "До новых встреч.", "Спасибо.", "Подписывайтесь на наш канал."):
        said = "Хорошо, договорились. " + outro
        assert mt.screen_made_up([_said(said, 0, 3.0)], 3.0, "recording")[0]["text"] == said
        if outro != "Спасибо.":                                    # a short thanks at the end of a line is never cut off
            assert mt.screen_made_up([_said(said, 0, 30.0)], 3.0, "recording")[0]["text"] == "Хорошо, договорились."
    assert mt.screen_made_up([_said("Продолжение следует... Хорошо, договорились.", 0, 3.0)], 3.0, "recording")[0]["text"] == \
        "Продолжение следует... Хорошо, договорились.", "only at the end, where Whisper appends its outros"
    assert mt.screen_made_up([_said("Это был отличный доклад. Субтитры сделал DimaTorzok")])[0]["text"] == "Это был отличный доклад."
    # a credit that _sentences cuts in two ('Untertitelung.' + 'BR 2018') is still one credit
    assert mt.screen_made_up([_said("Доклад окончен. Untertitelung. BR 2018")])[0]["text"] == "Доклад окончен."
    for context, end in (("recording", 2.0), ("media", 30.0)):      # 'BR 2018' alone is a video's tag: the credit goes whole
        assert mt.screen_made_up([_said("Gut. Untertitelung. BR 2018", 0, end)], 3.0, context)[0]["text"] == "Gut."
    assert mt.made_up(_said("Untertitelung. BR 2018"), alone=False) == "a subtitle credit"
    for credits in ("Продолжение следует... Untertitelung. BR 2018", "Субтитры сделал DimaTorzok. Untertitelung. BR 2018"):
        assert mt.made_up(_said(credits), alone=False) == "a subtitle credit", "nothing but credits: not a line to trim"


# What a meeting says that looks like a video's phrase or a credit (1.11.4 review): never dropped by its text alone
SAID_AT_WORK = ["Subtitles by Friday.", "Captions by Monday.", "Captions by Zoom.", "Transcription by Whisper.", "Captions by Rev.",
                "Transcribed by Otter.", "Субтитры подготовлены.", "Субтитры сделаны.", "Субтитры подготовлены вчера.",
                "Субтитры сделаны для второго ролика.", "Субтитры созданы автоматически.", "Субтитры созданы, но не проверены.",
                "Subscribe.", "Please subscribe.", "Subscribe to the channel.", "Subscribe to our channel.", "Don't forget to subscribe.",
                "Подпишитесь на канал.", "Подпишись на канал.", "Подписывайтесь на наш канал.", "Подписаться на канал.",
                "Не забудьте подписаться.", "Abonniert den Kanal.", "Bitte abonnieren.", "Продолжение следует?", "Fortsetzung folgt?",
                "See you in the next one.", "I'll see you in the next one.", "Okay, see you in the next one.", "See you tomorrow.",
                "See you later.", "Until next time.", "До новых встреч.", "Корректор Егорова.", "Корректор — Кулакова.",
                "Корректор. Егорова.", "♪ Happy birthday to you ♪"]


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", SAID_AT_WORK)
def test_what_a_meeting_says_like_a_videos_phrase_is_kept_without_a_sign_that_nobody_said_it(text, context):
    for alone in (True, False):
        assert mt.made_up(_said(text, 0, 1.4), alone=alone, audio_end=1.5, context=context) is None
        lines = mt.screen_made_up([_said(text, 0, 1.4), *([] if alone else [_said("Und weiter.", 2.0)])], 1.5, context)
        assert lines[0]["text"] == text and "made_up" not in lines[0]
    glued = "Хорошо, договорились. " + text
    assert mt.screen_made_up([_said(glued, 0, 2.4)], 2.5, context)[0]["text"] == glued


@pytest.mark.parametrize("text", ["Спасибо.", "Thank you.", "Danke.", "Vielen Dank.", "Bye.", "Музыка.", "See you tomorrow.",
                                  "До новых встреч.", "Спасибо за внимание.", "Das war's für heute.", "Bis zum nächsten Mal.",
                                  "Subscribe.", "I'm not sure what I'm doing here."])
def test_a_short_reply_whisper_timed_to_a_round_second_past_its_short_chunk_is_kept(text):
    """On a 1-1.4 s chunk Whisper ends real words at a round 2 or 3 s after their start: 'барсук' on the chunk
    1.14-2.51 s came out as 1.1-4.1 s, 1.59 s past the audio (1.11.4 measurement with large-v3-turbo)."""
    for context in ("media", "recording"):
        for alone in (True, False):
            assert mt.made_up(_said(text, 1.1, 4.1), alone=alone, audio_end=2.51, context=context) is None
        assert mt.screen_made_up([_said(text, 1.1, 4.1)], 2.51, context)[0] == _said(text, 1.1, 4.1)
        glued = "Хорошо, договорились. " + text
        assert mt.screen_made_up([_said(glued, 1.1, 4.1)], 2.51, context)[0]["text"] == glued
    assert mt.made_up(_said("Спасибо.", 0.0, 3.0), alone=True, audio_end=1.4) is None, "1.6 s past a 1.4 s chunk: kept"


def test_a_thanks_left_when_a_credit_is_trimmed_off_is_screened_again():
    """'Спасибо. Субтитры сделал DimaTorzok' timed 28.5 s past its audio: the credit comes off and what is left is the
    short thanks Whisper put in the padding; at speaking speed the thanks stays."""
    for text in ("Спасибо. Субтитры сделал DimaTorzok", "Thank you. Subtitles by the Amara.org community"):
        hidden = mt.screen_made_up([_said(text, 0, 30.0)], 1.5, "recording")[0]
        assert hidden["made_up"] == "thanks over near-silence"
        kept = mt.screen_made_up([_said(text, 0, 1.4)], 1.5, "recording")[0]
        assert kept["text"] == text.split(". ")[0] + "." and "made_up" not in kept
    assert mt.screen_made_up([_said("Продолжение следует... Спасибо.", 0, 30.0)], 1.5, "recording")[0]["made_up"]


@pytest.mark.parametrize("outro", ["See you tomorrow.", "See you later!", "I'll see you tomorrow.", "Until next time.", "До новых встреч.",
                                   "Спасибо за внимание и до новых встреч.", "Das war's für heute.", "See you in the next one.",
                                   "Bis zum nächsten Mal.", "Bis zum nächsten Mal!", "До зустрічі.", "Дякую за увагу."])
def test_a_goodbye_three_people_say_on_one_track_is_neither_withdrawn_nor_left_out(outro):
    """At the end of a call the other side's track carries everyone: three "See you tomorrow." are three people."""
    track, lines = SimpleNamespace(), []
    for k in range(3):
        chunk = mt.screen_made_up([_said(outro, 10.0 * k, 10.0 * k + 1.4), _said("Ja, gut.", 10.0 * k + 2.0)], 10.0 * k + 4.0, "recording")
        assert mt.mark_repeated_outros(track, chunk) is False
        lines += chunk
    assert [l["text"] for l in mt.without_made_up(lines)].count(outro) == 3
    assert mt.drop_repeated_outros(lines) == (lines, 0)


def test_copies_of_an_outro_never_shown_are_marked_without_withdrawing_anything():
    """Three copies in one chunk: all are left out, but none was on screen, so the window is not sent again (1.11.4
    review: a needless @@CLEAR and resend); a copy shown in an earlier chunk is withdrawn."""
    track, chunk = SimpleNamespace(), [_said("Bitte abonnieren.", 2.0 * k, 2.0 * k + 1.4) for k in range(3)]
    assert mt.mark_repeated_outros(track, chunk) is False
    assert [l.get("made_up") for l in chunk] == [mt.REPEATED] * 3
    track, shown = SimpleNamespace(), [_said("Bitte abonnieren.", 0.0, 1.4)]
    assert mt.mark_repeated_outros(track, shown) is False and "made_up" not in shown[0]
    later = [_said("Bitte abonnieren.", 10.0, 11.4), _said("Bitte abonnieren.", 12.0, 13.4)]
    assert mt.mark_repeated_outros(track, later) is True
    assert [l.get("made_up") for l in shown + later] == [mt.REPEATED] * 3


# Channel plugs (1.11.4 review): only the 'my channel' forms are a video's; the bare, 'our' and 'the' forms are a work
# channel or a button too, as in English, Russian and German
MY_CHANNEL = ["Підписуйтесь на мій канал!", "Abonnez-vous à ma chaîne !", "N'oubliez pas de vous abonner à ma chaîne.",
              "¡Suscríbete a mi canal!", "Iscriviti al mio canale!", "Inscreva-se no meu canal!", "Subskrybuj mój kanał!"]
WORK_CHANNEL = ["Підписуйтесь на канал.", "Підписуйтесь на наш канал.", "Abonnez-vous.", "Abonnez-vous à notre chaîne.",
                "Abonnez-vous à la chaîne.", "N'oubliez pas de vous abonner.", "Suscríbete.", "Suscríbete al canal.", "Iscriviti.",
                "Iscriviti al canale.", "Inscreva-se no canal.", "Subskrybuj.", "Subskrybuj nasz kanał.", "Zasubskrybuj."]


@pytest.mark.parametrize("text", MY_CHANNEL)
def test_a_plug_for_my_channel_is_a_videos_phrase(text):
    assert mt._made_up_kind(text) == "media"
    assert mt.made_up(_said(text, 0, 1.4), alone=False, audio_end=1.5, context="recording") == "a video's outro or channel plug"


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", WORK_CHANNEL)
def test_a_plug_for_the_or_our_channel_is_kept_without_a_sign(text, context):
    assert mt._made_up_kind(text) == "doubtful"
    assert mt.screen_made_up([_said(text, 0, 1.4)], 1.5, context)[0] == _said(text, 0, 1.4)
    assert mt.made_up(_said(text, 0, 30.0), alone=True, audio_end=1.5, context=context) == "an outro over near-silence"


def test_a_meeting_outro_with_a_credit_after_it_keeps_the_outro():
    """'Спасибо за внимание. Субтитры сделал DimaTorzok' at speaking speed: the credit comes off and the outro is
    screened again, as a short thanks is; timed into the padding it is hidden (1.11.4 review: it was hidden whole)."""
    for context in ("recording", "media"):
        for text in ("Спасибо за внимание. Субтитры сделал DimaTorzok", "Dziękuję za uwagę. Napisy stworzone przez społeczność Amara.org"):
            kept = mt.screen_made_up([_said(text, 0, 2.0)], 2.0, context)[0]
            assert kept["text"] == text.split(". ")[0] + "." and "made_up" not in kept
            assert mt.screen_made_up([_said(text, 0, 30.0)], 2.0, context)[0]["made_up"] == "an outro over near-silence"
        assert mt.made_up(_said("Продолжение следует... Субтитры сделал DimaTorzok"), alone=False, context=context) == "a subtitle credit"


GERMAN_CREDITS = ["Untertitel im Auftrag des ZDF, 2020", "Untertitel im Auftrag des ZDF, 2017", "Untertitel im Auftrag des ZDF für funk, 2017",
                  "Untertitel im Auftrag des ZDF für funk, 2017-2021", "Untertitelung im Auftrag des ZDF, 2021",
                  "Untertitelung des ZDF, 2020", "Untertitelung des ZDF für funk, 2017", "Untertitelung des ZDF.", "Untertitelung. BR 2018"]


@pytest.mark.parametrize("context", ["media", "recording"])
def test_the_german_broadcaster_credits_are_hidden_and_a_plain_mention_of_the_broadcaster_is_kept(context):
    for text in GERMAN_CREDITS:
        assert mt.made_up(_said(text), alone=False, context=context) == "a subtitle credit", text
    for text in ("Untertitel für ARD und ZDF.", "Untertitel vom ZDF.", "Untertitel brauchen wir bis Freitag.",
                 "Welcher Sender war das? Untertitel des ZDF?", "Untertitel des ZDF? Ja."):
        assert mt.screen_made_up([_said(text, 0, 1.4)], 1.5, context)[0] == _said(text, 0, 1.4), text


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", ["Music playing?", "Music continues?", "Phone rings?", "Bell rings?", "Birds chirping?", "SWR 2021?",
                                  "ZDF, 2021?", "This is the end of the video.", "This is the end of the video?",
                                  "♪ Happy birthday ♪", "♪ Zum Geburtstag ♪", "♪ Happy birthday to you ♪",
                                  # every video's phrase asked, and a German broadcaster credit asked (1.11.4 review)
                                  "Thanks for watching?", "Спасибо за просмотр?", "Субтитры сделал Иван?", "Ставьте лайки?",
                                  "Ставь лайк?", "Subscribe to my channel?", "Welcome to my channel?", "Danke fürs Zuschauen?",
                                  "Bis zum nächsten Video?", "Dziękuję za obejrzenie?", "Suscríbete a mi canal?",
                                  "Untertitel des ZDF?", "Untertitelung der ARD?", "Untertitel ZDF 2024?"])
def test_a_question_a_clip_ending_or_a_sung_line_is_never_a_videos_phrase_or_a_caption(text, context):
    for alone in (True, False):
        lines = mt.screen_made_up([_said(text, 0, 1.4), *([] if alone else [_said("Und weiter.", 2.0)])], 1.5, context)
        assert lines[0] == _said(text, 0, 1.4)


@pytest.mark.parametrize("text", ["Music playing", "Phone rings.", "SWR 2021", "This is the end of the video, remember to like and subscribe!",
                                  "♪ Musik ♪", "♪♪", "♪ Sanfte Musik ♪", "♪ música ♪", "♪ Music ♪ ♪ Music ♪"])
def test_the_statements_of_those_captions_and_phrases_stay_hidden_in_a_recording(text):
    assert mt.made_up(_said(text, 0, 1.4), alone=False, audio_end=1.5, context="recording") is not None


# The variants Whisper writes of the phrases already listed (1.11.4 review, fp-lens hallucinations.py)
VARIANTS = ["Ставьте лайки, подписывайтесь на канал!", "Подписывайтесь на канал, ставьте лайки!", "Подписывайтесь на канал и жмите на колокольчик!",
            "Подписывайтесь на канал, ставьте лайки и жмите на колокольчик!", "Ставь лайк!", "Субтитры сделал DimaTorzok Субтитры сделал DimaTorzok",
            "Продолжение следует Продолжение следует", "Продолжение следует... Продолжение следует...", "Спасибо за просмотр видео!",
            "Спасибо за просмотр этого видео!", "Спасибо, что посмотрели это видео!", "Спасибо за просмотр, до новых встреч!",
            "Дякую за перегляд і до зустрічі!", "Dziękuję za obejrzenie!", "Wir sehen uns im nächsten Video!", "Bis zum nächsten Video!",
            "¡Suscríbete y dale like!", "Thanks for watching, everyone!", "Thanks for watching, guys!", "Thanks for watching, bye!",
            "Thank you for watching, and don't forget to like and subscribe!", "Welcome to my channel!", "Welcome back to my channel."]


@pytest.mark.parametrize("text", VARIANTS)
def test_the_variants_of_a_videos_phrases_are_dropped_in_a_recording_and_kept_in_an_imported_file(text):
    assert mt.made_up(_said(text, 0, 1.4), alone=False, audio_end=1.5, context="recording") is not None
    credit = "DimaTorzok" in text                                  # a named subtitle author: a credit everywhere
    assert mt.made_up(_said(text, 0, 1.4), alone=True, audio_end=1.5, context="media") == ("a subtitle credit" if credit else None)


@pytest.mark.parametrize("text", ["Дякую.", "Дякую вам.", "Щиро дякую.", "Дуже дякую.", "Всем пока!", "До побачення.", "Бувайте!",
                                  "Дякую за увагу.", "До зустрічі!"])
def test_the_ukrainian_thanks_and_goodbyes_need_a_sign_that_nobody_said_them(text):
    assert mt._made_up_kind(text) in ("short", "meeting")
    for context in ("media", "recording"):
        for alone in (True, False):
            assert mt.made_up(_said(text, 0, 1.4), alone=alone, audio_end=1.5, context=context) is None
        assert mt.made_up(_said(text, 0, 30.0), alone=True, audio_end=1.5, context=context) is not None


@pytest.mark.parametrize("context", ["media", "recording"])
@pytest.mark.parametrize("text", ["Субтитры сделал DimaTorzok Продолжение следует", "Субтитры сделал DimaTorzok, Продолжение следует...",
                                  "Редактор субтитров А.Синецкая Корректор А.Егорова Продолжение следует..."])
def test_a_credit_run_into_an_outro_is_a_credit(text, context):
    assert mt.made_up(_said(text, 0, 1.4), alone=False, audio_end=1.5, context=context) == "a subtitle credit"
    assert mt.screen_made_up([_said("Мы решили, что продолжение следует завтра.", 0, 1.4)], 1.5, context)[0]["text"] == \
        "Мы решили, что продолжение следует завтра."


def _replay_screen_cases(context):
    """The harvested lines (tests/data/screen_cases.jsonl: 440 lines that large-v3-turbo int8 gave through mt.py's own
    live and batch paths on 35 synthetic cases of noise, wordless voice and real speech) screened again as decode_chunk
    screens them, one call per chunk or piece, with live repetition per run. -> {row index: (what, text)}."""
    rows = [json.loads(x) for x in open(Path(__file__).parent / "data" / "screen_cases.jsonl", encoding="utf-8")]
    result, runs = {}, {}
    for i, r in enumerate(rows):
        runs.setdefault(r["run"], {}).setdefault(r["piece"], []).append((i, r))
    for run, pieces in runs.items():
        track = SimpleNamespace()
        for piece in pieces.values():
            lines = [{"t": r["t"], "end": r["end"], "lang": "x", "conf": 1.0, "text": r["text"], "_lp": r["lp"], "_temp": r["temp"],
                      "_cr": r["cr"]} for _, r in piece]
            ids = {id(l): i for l, (i, _) in zip(lines, piece)}
            out = mt.screen_made_up(lines, piece[0][1]["audio_end"], context)
            if piece[0][1]["mode"] == "live":
                mt.mark_repeated_outros(track, out)
            for line in out:
                i = ids[id(line)]
                assert not any(k in line for k in ("_lp", "_temp", "_cr"))
                result[i] = ("hidden" if line.get("made_up") else "trimmed" if line["text"] != rows[i]["text"] else "shown", line["text"])
    return rows, result


@pytest.mark.parametrize("context", ["recording", "media"])
def test_the_harvested_lines_lose_their_made_up_outros_and_no_real_speech(context):
    rows, result = _replay_screen_cases(context)
    real = [i for i, r in enumerate(rows) if r["label"] in ("real", "unmatched", "real+appended")]
    assert len(real) == 219 and [rows[i]["text"] for i in real if result[i][0] != "shown"] == []
    live = [i for i, r in enumerate(rows) if r["mode"] == "live" and r["label"] == "made_up" and r["shown_before"]]
    assert len(live) == 97
    gone = [i for i in live if result[i][0] == "hidden"]
    assert len(gone) >= 4, "the measured floor (1.11.4): later pattern work may only raise it"
    for i in live:
        if rows[i]["text"] in ("Danke fürs Zuschauen.", "I can see you in the next video.", "Vielen Dank."):
            assert result[i][0] == "hidden", rows[i]["text"]
    # The limits, documented: a short thanks or a meeting's goodbye with no sign that nobody said it stays (a real
    # 'Gracias.' or 'I'll see you next time.' must), and so does fluent text on wordless voice
    assert any(result[i][0] == "shown" for i in live if rows[i]["text"] == "Gracias.")
    assert all(result[i][0] == "shown" for i in live if rows[i]["text"] == "I'll see you next time.")
    assert any(result[i][0] == "shown" for i, r in enumerate(rows) if r["mode"] == "batch" and r["text"] == "Thank you."
               and r["label"] == "made_up")


def test_a_chunk_whose_every_line_failed_its_decode_is_kept_as_speech(monkeypatch, tmp_path):
    """A decode Whisper rejected at every temperature is still Whisper's best text: the real 'als' of a 1.3 s chunk
    came out as 'So als Riech.' at -1.02 (1.11.4 measurement), so a failed chunk is never turned into a note."""
    import numpy as np
    bad = {"start": 0.0, "end": 1.0, "text": " So als Riech.", "logprob": -1.021, "temp": 1.0, "cr": 0.62}
    chunk = np.zeros(int(16000 * 1.34), dtype="float32")
    for context in ("recording", "media"):
        lines = mt.decode_chunk(FakeEngine([([("de", .9)], [bad])]), chunk, 0, [], None, None, context=context)
        assert lines == [{"t": 0.0, "end": 1.0, "lang": "de", "conf": 0.9, "text": "So als Riech."}]


def test_the_values_of_a_failed_decode_never_leave_the_screening(monkeypatch, tmp_path, capsys):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(70, dtype="float32"), 16000))
    engine = FakeEngine([([("en", .9)], [{"start": 0.5, "end": 2.0, "text": " one", "logprob": -0.2, "temp": 0.0, "cr": 0.9}]),
                         ([("en", .9)], [{"start": 0.5, "end": 5.5, "text": " I don't know.", "logprob": -1.4, "temp": 1.0, "cr": 1.1}])])
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", checkpoint=tmp_path)
    assert [l.get("text") or l["note"] for l in lines] == ["one", "I don't know."]
    log = (tmp_path / "partial.jsonl").read_text(encoding="utf-8")
    out = capsys.readouterr().out
    for where in (log, out, json.dumps(lines)):
        assert not any(k in where for k in ('"_lp"', '"_temp"', '"_cr"'))
    assert "rejected" not in log


def test_a_recording_is_screened_as_one_and_an_imported_file_as_media(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(mt, "transcribe_lines", lambda *a, **k: seen.append(k.get("context")) or [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "x"}])
    d = tmp_path / "20261127-100000-call"
    d.mkdir()
    mt.transcribe(d / "audio.wav", d)
    mt.transcribe(tmp_path / "talk.mp4", d)
    mt.transcribe(tmp_path / "audio.wav", d)                       # an audio.wav from somewhere else is a file like any
    mt.transcribe_call(d / "audio.wav", d / "system.wav", d)
    assert seen == ["recording", "media", "media", "recording", "recording"]
    assert mt.screening_context(Path("audio.wav"), Path(".")) == "recording"


def _live_track(tmp_path, tag, src, allowed=("de",)):
    return mt.LiveTrack(str(src), tmp_path, tag, src, list(allowed), "fine", None, mt.engine_name())


class ScriptEngine:
    """Decodes a chunk into the scripted line of the second it starts at (speech samples carry their second + 1)."""
    name = "fake"

    def __init__(self, script):
        self.script = script

    def analyse(self, chunk, detect=True):
        return SimpleNamespace(chunk=chunk, probs=[("de", 0.99)] if detect else None)

    def decode(self, state, language):
        import numpy as np
        first = int(np.asarray(state.chunk)[np.flatnonzero(state.chunk)[0]] * 32768) - 1
        return [{"start": 0.0, "end": 1.4, "text": " " + self.script[first], "logprob": -0.3, "temp": 0.0, "cr": 0.8}]


def test_an_outro_said_again_and_again_is_withdrawn_from_the_live_window_and_left_out_of_the_file(monkeypatch, tmp_path, capsys):
    """A plug said in meetings too ("Bitte abonnieren.") at speaking speed is shown; the third copy on one track shows it
    is Whisper's: it is never shown, the window is cleared and sent every other line of both tracks again, the chunks of
    the earlier copies are written to the checkpoint again with the mark, and the final pass leaves all three out with
    the note the window agrees with, without showing them again ('Bis zum nächsten Mal.' is a goodbye several people
    say: never counted, test_a_goodbye_three_people_say_on_one_track_is_neither_withdrawn_nor_left_out)."""
    _content_vad(monkeypatch)
    outro = "Bitte abonnieren."
    engine = ScriptEngine({0: outro, 1: "Hello.", 8: "Добрый день.", 16: outro, 24: outro})
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    mine = _speech(34, gaps=set(range(34)) - {0, 1, 8, 9, 16, 17, 24, 25})
    theirs = _speech(34, gaps=set(range(34)) - {1, 2})
    one, two = _live_track(tmp_path, "", tmp_path / "mine.wav"), _live_track(tmp_path, ".2", tmp_path / "theirs.wav")
    one.feed(mine)
    two.feed(theirs)
    worker = mt.LiveWorker([one, two])
    worker.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (len(one.chunks) < 4 or one.pending or two.pending):
        time.sleep(0.05)
    worker.stop(wait=10)
    out = capsys.readouterr().out.splitlines()
    shown = [json.loads(o.split("\t", 1)[1]) for o in out if o.startswith("@@LINE\t")]
    clear = out.index("@@CLEAR")
    before = [json.loads(o.split("\t", 1)[1])["text"] for o in out[:clear] if o.startswith("@@LINE\t")]
    after = [json.loads(o.split("\t", 1)[1])["text"] for o in out[clear:] if o.startswith("@@LINE\t")]
    assert before.count(outro) == 2 and "Hello." in before and "Добрый день." in before
    assert after == ["Hello.", "Добрый день."] and out.count("@@CLEAR") == 1
    assert all(l["live"] for l in shown) and sum(1 for l in shown if l["text"] == outro) == 2, "the third is never shown"
    records = [json.loads(x) for x in (tmp_path / "partial.jsonl").read_text(encoding="utf-8").splitlines()]
    repeated = "an outro said again and again"
    assert [(r["i"], l.get("made_up")) for r in records for l in r["lines"] if l["text"] == outro] == [
        (1, None), (3, None), (1, repeated), (3, repeated), (4, repeated)], "the earlier chunks written again, marked"
    last = {r["i"]: r["lines"] for r in records}                   # what Checkpoint.resume keeps: the last record of a chunk
    assert [l.get("made_up") for i in sorted(last) for l in last[i] if l["text"] == outro] == [repeated] * 3
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: mine.reshape(-1).astype("float32") / 32768)
    lines = mt.transcribe_lines(tmp_path / "mine.wav", "de", granularity="fine", checkpoint=tmp_path, context="recording")
    assert [l.get("text") or l["note"] for l in lines] == [
        "Добрый день.", "3 lines that Whisper makes up on noise (subtitle credits, outros said again and again) were left out"]
    final = capsys.readouterr().out.splitlines()
    assert not any(o.startswith("@@LINE\t") and outro in o for o in final), "the final pass never shows a withdrawn copy again"
    assert "@@CLEAR" not in final, "nothing it shows is left out at its end: no need to start the window over"


def test_a_withdrawn_outro_is_not_sent_again_by_its_late_translation(monkeypatch, tmp_path, capsys):
    """The outro test above with a translation into English whose Claude requests answer only after the CLEAR: the
    translation (or the tr_skip at the stop) of a copy withdrawn since it was queued sends no @@LINE for it again."""
    _content_vad(monkeypatch)
    outro = "Bitte abonnieren."
    engine = ScriptEngine({0: outro, 1: "Hello.", 8: "Добрый день.", 16: outro, 24: outro})
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    release = threading.Event()

    class SlowAnthropic:                       # a Claude request that answers only once released
        def __init__(self):
            self.messages = self

        def create(self, **kw):
            release.wait(20)
            n = kw["messages"][0]["content"].split("\n\n", 1)[1].count("\n") + 1
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="\n".join(f"{i}|T{i}" for i in range(n)))])
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=SlowAnthropic, APIConnectionError=OSError))
    mine = _speech(34, gaps=set(range(34)) - {0, 1, 8, 9, 16, 17, 24, 25})
    theirs = _speech(34, gaps=set(range(34)) - {1, 2})
    one, two = _live_track(tmp_path, "", tmp_path / "mine.wav"), _live_track(tmp_path, ".2", tmp_path / "theirs.wav")
    one.feed(mine)
    two.feed(theirs)
    worker = mt.LiveWorker([one, two], targets=["English"], offline=False)
    worker.start()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and (len(one.chunks) < 4 or one.pending or two.pending):
        time.sleep(0.05)
    time.sleep(0.5)
    release.set()                              # the first request answers after the CLEAR
    time.sleep(1.5)
    worker.stop(wait=10)
    out = capsys.readouterr().out.splitlines()
    clear = out.index("@@CLEAR")
    after = [json.loads(o.split("\t", 1)[1]) for o in out[clear:] if o.startswith("@@LINE\t")]
    assert after and not [l for l in after if l["text"] == outro], "a withdrawn copy is never shown again"
    assert any(l.get("tr") or l.get("tr_skip") for l in after), "the other lines' translations (or skips) still follow"


def test_a_live_track_neither_shows_nor_translates_a_videos_outro_and_screens_as_a_recording(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    seen, contexts = [], []
    track = SimpleNamespace(pending=[(0, 16000 * 6)], chunks=[(0, 16000 * 6)], base=0, audio=list(range(16000 * 6)), allowed=["ru"],
                            forced="ru", speaker=None, granularity="normal", covered=16000 * 6, fed=16000 * 6, _settle=lambda: None,
                            cp=SimpleNamespace(record=lambda i, lines: seen.append(lines)))

    def decode_chunk(*a, context="media", **k):
        contexts.append(context)
        return mt.screen_made_up([_said("Спасибо за просмотр!", 0.0, 1.4), _said("Добрый день.", 3.0)], 6.0, context)
    monkeypatch.setattr(mt, "decode_chunk", decode_chunk)
    translated = []
    shown = mt.LiveTrack.decode(track, None, translate=lambda lines: translated.extend(l["text"] for l in lines))
    assert contexts == ["recording"]
    assert [l["text"] for l in shown] == ["Добрый день."] and translated == ["Добрый день."]
    assert "Спасибо за просмотр" not in capsys.readouterr().out
    assert [l.get("made_up") for l in seen[0]] == ["a video's outro or channel plug", None]


def test_a_final_pass_that_cannot_reuse_the_live_run_starts_the_window_empty(monkeypatch, tmp_path, capsys):
    """The languages were changed after a live recording: its chunks are decoded again, so the window, which still shows
    the live lines, is cleared first; with the live run reused it is not."""
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    audio = _speech(23, gaps={6, 7, 14, 15, 22})
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio.reshape(-1).astype("float32") / 32768)
    for languages, cleared in (("ru,en", True), ("ru", False)):
        folder = tmp_path / languages
        folder.mkdir()
        track = mt.LiveTrack("x.wav", folder, "", tmp_path / "x.wav", ["ru"], "normal", None, mt.engine_name())
        track.feed(audio[:16000 * 10])
        track.cut()
        track.decode(engine)
        track.close()
        capsys.readouterr()
        mt.transcribe_lines(tmp_path / "x.wav", languages, granularity="normal", checkpoint=folder)
        out = capsys.readouterr().out.splitlines()
        first_line = next(i for i, o in enumerate(out) if o.startswith("@@LINE\t"))
        assert ("@@CLEAR" in out[:first_line]) is cleared, languages
        assert out.count("@@CLEAR") == (1 if cleared else 0)


def test_both_engines_hand_on_the_temperature_and_compression_ratio_of_a_decode(monkeypatch):
    seg = mt._segment(SimpleNamespace(start=0, end=1, text="x", avg_logprob=-1.2, temperature=1.0, compression_ratio=1.1))
    assert seg == {"start": 0.0, "end": 1.0, "text": "x", "logprob": -1.2, "temp": 1.0, "cr": 1.1}
    assert mt._segment(SimpleNamespace(start=0, end=1, text="x", avg_logprob=-0.2)) == {"start": 0.0, "end": 1.0, "text": "x", "logprob": -0.2}
    engine = object.__new__(mt.MLXEngine)
    engine.repo = "r"
    engine.lib = SimpleNamespace(transcribe=lambda chunk, **kw: {"segments": [{"start": 0, "end": 30.0, "text": " Thank you.",
                                                                               "avg_logprob": -0.5, "temperature": 0.0,
                                                                               "compression_ratio": 0.6}]})
    assert engine.decode(SimpleNamespace(chunk=[0.0]), "en") == [{"start": 0.0, "end": 30.0, "text": " Thank you.", "logprob": -0.5,
                                                                   "temp": 0.0, "cr": 0.6}]


def test_transcribe_with_one_language_forces_it(monkeypatch, tmp_path):
    doc, model = _run_transcribe(monkeypatch, tmp_path, "Russian")
    assert [l["lang"] for l in doc["lines"]] == ["ru", "ru", "ru"] and all(l["conf"] == 1.0 for l in doc["lines"])
    assert [c[1] for c in model.calls] == ["ru", "ru", "ru"]


# ---------- granularity: how finely speech is cut before each language decision ----------
def test_granularity_presets_get_finer_and_auto_follows_the_number_of_languages():
    pauses = [mt.GRANULARITY[g][1] for g in ("coarse", "normal", "fine", "finest")]
    mins = [mt.GRANULARITY[g][0] for g in ("coarse", "normal", "fine", "finest")]
    assert pauses == sorted(pauses, reverse=True) and mins == sorted(mins, reverse=True)
    assert mt.GRANULARITY["normal"] == (mt.MIN_S, 700)                       # the behaviour before presets existed
    assert [mt.resolve_granularity("auto", l) for l in ([], ["ru"], ["ru", "en"])] == ["adaptive", "normal", "finest"]
    assert mt.GRANULARITY["adaptive"] == mt.GRANULARITY["normal"]             # adaptive cuts like normal, then re-cuts mixed chunks
    assert mt.resolve_granularity("adaptive", ["ru", "en"]) == "adaptive"
    assert mt.resolve_granularity(None, ["ru", "en", "de"]) == "finest"
    assert mt.resolve_granularity("finest", []) == "finest"                  # an explicit choice is kept
    with pytest.raises(ValueError, match="unknown granularity"):
        mt.resolve_granularity("tiny", [])


def test_group_spans_with_the_presets_cuts_the_same_speech_into_more_chunks_when_finer():
    s = lambda a, b: {"start": int(a * mt.RATE), "end": int(b * mt.RATE)}
    spans = [s(0, 1.5), s(2, 3.5), s(4, 5.5), s(6, 7.5), s(8, 9.5)]          # five 1.5 s sentences, 0.5 s apart
    counts = {g: len(mt.group_spans(spans, min_s=mt.GRANULARITY[g][0])) for g in ("coarse", "normal", "fine", "finest")}
    assert counts == {"coarse": 1, "normal": 2, "fine": 3, "finest": 5}


def test_transcribe_resolves_auto_granularity_from_the_languages(monkeypatch, tmp_path):
    seen = []
    real = mt.group_spans
    monkeypatch.setattr(mt, "group_spans", lambda spans, min_s=mt.MIN_S, **kw: seen.append(min_s) or real(spans, min_s=min_s, **kw))
    _run_transcribe(monkeypatch, tmp_path, "ru, en")                 # two languages: auto = finest
    _run_transcribe(monkeypatch, tmp_path, None)                     # none: auto = adaptive, which cuts like normal ...
    assert seen[:2] == [mt.GRANULARITY["finest"][0], mt.GRANULARITY["normal"][0]]
    assert seen[2:] and set(seen[2:]) == {mt.GRANULARITY["finest"][0]}   # ... and re-cuts the chunks that sound mixed at finest


class PieceEngine:
    """Scripted per (start second, length in seconds): a 6 s chunk at 0 sounds mixed and splits into an English and a
    Russian piece; the chunk at 20 is plain German; the chunk at 40 sounds mixed but has no pause to split at."""
    name = "fake"
    SCRIPT = {(0, 6): ([("en", .5), ("ru", .45)], None),
              (0, 3): ([("en", .9), ("ru", .05)], [{"start": 0.2, "end": 2.8, "text": " hello", "logprob": -0.2}]),
              (3, 2): ([("ru", .9), ("en", .05)], [{"start": 0.1, "end": 1.9, "text": " привет", "logprob": -0.2}]),
              (20, 6): ([("de", .9), ("en", .05)], [{"start": 0.5, "end": 5.0, "text": " hallo", "logprob": -0.2}]),
              (40, 6): ([("en", .4), ("ru", .35)], {"en": [{"start": 0.5, "end": 5.0, "text": " guess", "logprob": -1.0}],
                                                    "ru": [{"start": 0.5, "end": 5.0, "text": " угадал", "logprob": -0.3}]})}

    def __init__(self):
        self.analysed, self.decoded = [], []

    def analyse(self, chunk, detect=True):
        key = (int(chunk[0]), round(len(chunk) / mt.RATE))
        self.analysed.append(key)
        return SimpleNamespace(key=key, probs=self.SCRIPT[key][0] if detect else None)

    def decode(self, state, language):
        self.decoded.append((state.key, language))
        segs = self.SCRIPT[state.key][1]
        return [dict(x) for x in ((segs.get(language) if isinstance(segs, dict) else segs) or [])]


def _piece_spans(audio, start, pause_ms, window_s=None):
    """The VAD stand-in: three 6 s chunks in the whole audio; inside a 6 s chunk, a pause after 3 s only at the finest setting."""
    if len(audio) > 20 * mt.RATE:
        return [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40)]
    if pause_ms <= 200 and int(audio[0]) == 0:
        return [{"start": 0, "end": 3 * mt.RATE}, {"start": 3 * mt.RATE + 8000, "end": 6 * mt.RATE}]
    return [{"start": 0, "end": len(audio)}]


def _adaptive_run(monkeypatch, tmp_path, language=None, granularity="auto"):
    import numpy as np
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: [])   # CI has no faster_whisper
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "speech_spans", _piece_spans)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(70, dtype="float32"), 16000))
    engine = PieceEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    return mt.transcribe_lines(tmp_path / "x.wav", language, granularity=granularity), engine


def test_adaptive_chunking_re_cuts_a_mixed_chunk_and_tags_each_part_on_its_own(monkeypatch, tmp_path):
    lines, engine = _adaptive_run(monkeypatch, tmp_path)
    assert [(l["t"], l["lang"], l["conf"], l["text"]) for l in lines if "text" in l] == [
        (0.2, "en", 0.9, "hello"), (3.6, "ru", 0.9, "привет"),          # the mixed chunk: two pieces, two tags
        (20.5, "de", 0.9, "hallo"),                                      # a plain chunk is decoded as it is
        (40.5, "ru", 0.4, "угадал")]                                     # mixed but unsplittable: Whisper's second choice won
    assert engine.analysed == [(0, 6), (0, 3), (3, 2), (20, 6), (40, 6)]
    assert engine.decoded == [((0, 3), "en"), ((3, 2), "ru"), ((20, 6), "de"), ((40, 6), "en"), ((40, 6), "ru")]
    assert mt.fmt(lines[3]) == "[00:00:40] [ru?] угадал"                 # below LOW_CONF: still a guess


def test_a_piece_of_a_re_cut_chunk_that_gave_nothing_is_noted_and_counted(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(PieceEngine.SCRIPT, (3, 2), ([("ru", .9), ("en", .05)], []))     # the Russian piece: no text
    monkeypatch.setattr(mt, "EVENTS", True)
    lines, engine = _adaptive_run(monkeypatch, tmp_path)
    assert [l.get("text") or l["note"] for l in lines][:2] == ["hello", "speech not recognised for 2 s"]
    assert lines[1]["t"] == 3.5
    assert lines[-1]["note"].startswith("decoded 89 % of the speech found (2 s not recognised)")   # 2 of 18 s
    assert any(o.startswith("@@NOTE\tdecoded 89 %") for o in capsys.readouterr().out.splitlines())


def test_without_adaptive_a_mixed_chunk_keeps_one_tag(monkeypatch, tmp_path):
    lines, engine = _adaptive_run(monkeypatch, tmp_path, granularity="normal")
    assert engine.analysed == [(0, 6), (20, 6), (40, 6)]
    assert [l.get("text", l.get("note")) for l in lines][:1] == ["speech not recognised for 6 s"]   # the script has no chunk-level decode
    forced, engine = _adaptive_run(monkeypatch, tmp_path, language="de")
    assert engine.analysed == [(0, 6), (20, 6), (40, 6)] and {l["lang"] for l in forced if "text" in l} == {"de"}


def test_looks_mixed_rules():
    assert mt.looks_mixed([("en", .5), ("ru", .45)]) and mt.looks_mixed([("en", .985), ("ru", .012)])     # best below 99 %
    assert mt.looks_mixed([("en", .992), ("ru", .031)])                                                   # second above 3 %
    assert not mt.looks_mixed([("en", .995), ("ru", .004)]) and not mt.looks_mixed([("en", .99), ("ru", .03)]) and not mt.looks_mixed(None)
    assert (mt.SPLIT_CONF, mt.SPLIT_SECOND) == (0.99, 0.03), "the measured thresholds (step 0 of 1.9.2)"


def test_a_checkpoint_of_another_run_is_not_reused_and_says_what_changed(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    cp = mt.Checkpoint(tmp_path, "")
    cp.start({"source": "a.wav", "languages": [], "granularity": "normal", "model": "turbo"}, [(0, 16000)])
    cp.record(1, [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "hi"}])
    again = mt.Checkpoint(tmp_path, "")
    assert again.resume({"source": "a.wav", "languages": [], "granularity": "adaptive", "model": "turbo"}) == (None, {}, 0)
    assert "@@NOTE\tStarting over: the earlier run in this folder used another granularity, so its chunks cannot be reused" in capsys.readouterr().out
    same = mt.Checkpoint(tmp_path, "").resume({"source": "a.wav", "languages": [], "granularity": "normal", "model": "turbo"})
    assert same[0] == [(0, 16000)] and 1 in same[1]


# ---------- a call: your microphone and the other side as two tracks ----------
def test_speaker_labels_survive_writing_reading_and_translation_output():
    line = {"t": 12.4, "lang": "ru", "conf": 0.9, "text": "Привет", "speaker": "Them"}
    assert mt.fmt(line) == "[00:00:12] [ru] <Them> Привет"
    assert mt.fmt(line, "Hello") == "[00:00:12] [ru] <Them> Hello"                    # a translation keeps who spoke
    doc = mt.parse_transcript("[00:00:12] [ru] <Them> Привет\n[00:00:20] [en?] <You> Hi there\n[00:00:30] [en] [Music] no speaker\n")
    assert [(l.get("speaker"), l["text"]) for l in doc["lines"]] == [("Them", "Привет"), ("You", "Hi there"), (None, "[Music] no speaker")]
    assert mt.render(doc) == "[00:00:12] [ru] <Them> Привет\n[00:00:20] [en?] <You> Hi there\n[00:00:30] [en] [Music] no speaker\n"


def test_is_echo_needs_a_similar_text_at_the_same_moment_and_never_calls_a_later_repetition_or_a_short_line_echo():
    them = [{"t": 10.0, "text": "Can we move the meeting to Friday?"}]
    assert mt.is_echo({"t": 10.6, "text": "can we move the meeting to friday"}, them)                  # the same words, a moment later
    assert mt.is_echo({"t": 10.6, "text": "move the meeting to Friday"}, them)                       # only part of it was heard
    assert not mt.is_echo({"t": 13.0, "text": "Can we move the meeting to Friday?"}, them)            # said back 3 s later: an answer, not an echo
    assert not mt.is_echo({"t": 10.6, "text": "What time works for you?"}, them)                      # a different sentence
    assert not mt.is_echo({"t": 10.0, "text": "Yes"}, [{"t": 10.0, "text": "Yes"}])                   # too short to be sure
    assert not mt.is_echo({"t": 10.2, "text": "Thank you very much indeed"}, [])                      # the other side said nothing


def test_a_failed_connection_to_claude_names_its_real_cause(monkeypatch):
    class Boom(Exception):
        pass

    class Conn(Exception):                                   # stands for anthropic.APIConnectionError
        pass

    class Messages:
        def create(self, **kw):
            try:
                raise Boom("certificate verify failed")
            except Boom as inner:
                raise Conn("Connection error.") from inner

    class Client:
        messages = Messages()

    fake = SimpleNamespace(APIConnectionError=Conn, Anthropic=lambda: Client())
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    doc = {"lines": [{"t": 1.0, "lang": "ru", "conf": 1.0, "text": "привет"}], "translations": {}}
    with pytest.raises(RuntimeError) as e:
        mt.translate(doc, "English")
    assert "Boom: certificate verify failed" in str(e.value) and "--offline" in str(e.value)
    assert doc["translations"] == {}                         # nothing half-done is cached


def test_trim_echo_removes_the_echoed_words_and_keeps_the_users_own(monkeypatch):
    them = [{"t": 10.0, "end": 14.0, "text": "Can we move the meeting to Friday, because Thursday is a holiday here?"}]
    mine = {"t": 10.5, "end": 15.0, "text": "Can we move the meeting to Friday, because Thursday is a holiday here? Yes, that works for me."}
    assert mt.trim_echo(mine, them)["text"] == "Yes, that works for me."                   # 78 + 21 characters: the answer stays
    assert mt.trim_echo({"t": 10.6, "end": 13.0, "text": "can we move the meeting to friday"}, them) is None   # only echo
    split = [{"t": 10.0, "end": 12.0, "text": "Can we move the meeting"}, {"t": 12.0, "end": 14.0, "text": "to Friday, because Thursday is a holiday here?"}]
    assert mt.trim_echo({"t": 10.5, "end": 14.0, "text": "Can we move the meeting to Friday, because Thursday is a holiday here?"}, split) is None
    assert mt.trim_echo({"t": 12.1, "end": 14.0, "text": "Can we move the meeting to Friday, because Thursday is a holiday here?"}, them) is None   # 2.1 s later but overlapping
    assert mt.trim_echo({"t": 20.0, "end": 22.0, "text": "Can we move the meeting to Friday, because Thursday is a holiday here?"}, them)["text"].startswith("Can we")  # later: an answer
    assert mt.trim_echo({"t": 10.0, "end": 10.5, "text": "Yes"}, [{"t": 10.0, "end": 10.5, "text": "Yes"}]) is None          # identical at the same moment
    assert mt.trim_echo({"t": 10.0, "end": 11.0, "text": "Yes, I think so"}, [{"t": 10.0, "end": 10.5, "text": "Yes"}])["text"] == "Yes, I think so"
    merged = mt.merge_call([mine], them, "Anna", "Boris")
    assert [(l["speaker"], l["text"]) for l in merged] == [("Boris", them[0]["text"]), ("Anna", "Yes, that works for me.")]


def test_notes_given_on_the_command_line_head_the_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "transcribe_lines", lambda *a, **k: [{"t": 1.0, "lang": "en", "conf": 1.0, "text": "hi"}])
    doc = mt.transcribe(tmp_path / "x.wav", tmp_path, "en", notes=["The Mac slept from 10:01:00 to 10:05:30", " "])
    assert doc["lines"][0] == {"t": 0, "note": "The Mac slept from 10:01:00 to 10:05:30"} and len(doc["lines"]) == 2
    assert mt.render(doc).splitlines()[0] == "# [00:00:00] The Mac slept from 10:01:00 to 10:05:30"


def test_record_reports_the_level_and_a_silent_recording(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    mt.record("x", 12 / 3600, None)
    out, err = capsys.readouterr()
    out = out.splitlines()
    assert any(o.startswith("@@LEVEL\t0.0") for o in out), out
    assert any(o.startswith("@@NOTE\tThe recording is silent") for o in out)
    # The minute counter ("\r   0 min", never ended by a line break) is for the terminal only: the app reads stdout and stderr
    # as one stream, and an event glued behind the fragment is not an event any more (every live line and level was lost, 1.9.11).
    assert not re.search(r"\r\s*\d+ min", err), err


def test_record_warns_about_a_silent_recording_on_the_command_line_too(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", False)
    mt.record("x", 12 / 3600, None)
    assert "The recording is silent" in capsys.readouterr().err


def test_record_reports_a_silent_second_track_too(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    folder, mic, system = mt.record("x", 12 / 3600, None, system_device=7)
    assert system is not None and _seconds(system) >= 10
    out = capsys.readouterr().out.splitlines()
    assert any(o.startswith("@@NOTE\tThe recording is silent") for o in out)
    assert any(o.startswith("@@NOTE\tThe computer's sound was silent for the whole recording") for o in out)


@posix_signals
def test_record_pauses_and_resumes_on_sigusr2_without_padding_and_says_where(monkeypatch, tmp_path, capsys):
    import signal
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    notes = []
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGUSR2)).start()      # pause after about 3 s of (fast) audio
    threading.Timer(1.4, lambda: os.kill(os.getpid(), signal.SIGUSR2)).start()      # resume 1.1 s later
    folder, wav, _ = mt.record("x", 12 / 3600, None, notes=notes)
    assert 12 <= _seconds(wav) <= 20, "the pause is not in the file and does not use up the limit"
    out = capsys.readouterr()
    assert not any("filled with silence" in n for n in notes) and "reopen" not in out.out, "a pause is not a gap or a stall"
    pause = [n for n in notes if "Paused here" in n]
    assert len(pause) == 1 and re.fullmatch(r"\[00:00:0\d\] Paused here for 1 s", pause[0]), pause
    assert f"@@NOTE\t{pause[0]}" in out.out and "paused\n" in out.err and "recording\n" in out.err
    assert any(n.startswith("Recording stopped at the") for n in notes), "every note of the recording is collected"


@posix_signals
def test_a_recording_stopped_while_paused_still_tells_the_pause(monkeypatch, tmp_path, capsys):
    import signal
    _fake_recording(monkeypatch, tmp_path)
    notes = []
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGUSR2)).start()
    threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    folder, wav, _ = mt.record("x", 1, None, notes=notes)
    assert 1 <= _seconds(wav) <= 10 and [n for n in notes if "Paused here" in n] and "paused" in capsys.readouterr().err


def test_timed_notes_are_placed_at_their_time_in_the_transcript(monkeypatch, tmp_path):
    assert mt.header_notes(["[00:12:30] Paused here for 3 min 5 s", "plain", "", "[1:02:03] late"]) == [
        {"t": 750, "note": "Paused here for 3 min 5 s"}, {"t": 0, "note": "plain"}, {"t": 3723, "note": "late"}]
    monkeypatch.setattr(mt, "transcribe_lines", lambda *a, **k: [{"t": 5.0, "lang": "en", "conf": 1.0, "text": "before"},
                                                                  {"t": 20.0, "lang": "en", "conf": 1.0, "text": "after"}])
    doc = mt.transcribe(tmp_path / "x.wav", tmp_path, notes=["[00:00:12] Paused here for 1 min 0 s", "top"])
    assert [l.get("note") or l["text"] for l in doc["lines"]] == ["top", "before", "Paused here for 1 min 0 s", "after"]
    assert mt.render(doc).splitlines()[2] == "# [00:00:12] Paused here for 1 min 0 s"
    assert mt.pause_note(750.4, 185.2) == "[00:12:30] Paused here for 3 min 5 s" and mt.pause_note(3, 7.6) == "[00:00:03] Paused here for 8 s"


def test_the_record_command_puts_its_own_notes_into_the_transcription(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(mt, "transcribe", lambda src, d, *a, **k: seen.update(k) or {"lines": [], "translations": {}})
    monkeypatch.setattr(mt, "write_outputs", lambda *a, **k: [])
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(5 / 3600), "--note", "given"])
    mt.main()
    assert seen["notes"][0] == "given" and any(n.startswith("Recording stopped at the") for n in seen["notes"][1:])


def test_record_notices_a_microphone_that_stops_and_reopens_it(monkeypatch, tmp_path, capsys):
    import numpy as np
    pytest.importorskip("soundfile")
    opened = []

    class DyingStream(FakeStream):                                 # delivers 3 s of audio, then nothing (the device went away)
        def __init__(self, samplerate, channels, dtype, device, callback):
            super().__init__(samplerate, channels, dtype, device, callback)
            opened.append(self)
            self.delivered = 0

        def _run(self):
            while not self.stop_flag.is_set():
                if self.delivered < 30 or len(opened) > 1:
                    self.callback(self.block, len(self.block), None, None)
                    self.delivered += 1
                time.sleep(0.01)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=DyingStream))
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    folder, wav, _ = mt.record("x", 40 / 3600, None)
    out = capsys.readouterr().out
    assert "No sound has come from the microphone for" in out and "trying to reopen it" in out and len(opened) >= 2
    assert _seconds(wav) >= 40


def test_a_microphone_that_cannot_be_reopened_at_once_is_tried_again(monkeypatch, tmp_path, capsys):
    import numpy as np
    pytest.importorskip("soundfile")
    opened = []

    class DyingStream(FakeStream):                                 # 3 s of audio, then nothing; the 2nd open fails, the 3rd works
        def __init__(self, samplerate, channels, dtype, device, callback):
            if len(opened) == 1:
                opened.append(None)
                raise RuntimeError("device busy")
            super().__init__(samplerate, channels, dtype, device, callback)
            opened.append(self)
            self.delivered = 0

        def _run(self):
            while not self.stop_flag.is_set():
                if self.delivered < 30 or len(opened) > 2:
                    self.callback(self.block, len(self.block), None, None)
                    self.delivered += 1
                time.sleep(0.01)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=DyingStream))
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "REOPEN_S", 0.5)
    folder, wav, _ = mt.record("x", 40 / 3600, None)
    out = capsys.readouterr().out
    assert "The microphone could not be reopened: device busy; trying again in 0.5 s" in out
    assert len(opened) == 3 and opened[2].delivered > 0, "the third open worked without any sound having arrived in between"
    assert _seconds(wav) >= 40


def test_a_gap_on_one_track_is_filled_on_that_track_only(monkeypatch, tmp_path, capsys):
    import numpy as np
    import soundfile as sf
    pytest.importorskip("soundfile")

    class TwoTracks(FakeStream):                                   # the microphone (device None) dies after 3 s; the system goes on
        def __init__(self, samplerate, channels, dtype, device, callback):
            super().__init__(samplerate, channels, dtype, device, callback)
            self.device, self.delivered = device, 0
            self.block = np.full((samplerate // 10, channels), 7, dtype=dtype)

        def _run(self):
            while not self.stop_flag.is_set():
                if self.device == 3 or self.delivered < 30:
                    self.callback(self.block, len(self.block), None, None)
                    self.delivered += 1
                time.sleep(0.01)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=TwoTracks))
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "REOPEN_S", 100)
    folder, mic, system = mt.record("x", 70 / 3600, None, 3)
    out = capsys.readouterr().out
    assert "with no sound on the microphone" in out and "on the computer's sound" not in out
    system_data = sf.read(str(system), dtype="int16")[0]
    assert len(system_data) >= 70 * 16000 and (system_data != 0).all(), "the healthy track got no silence"
    mic_data = sf.read(str(mic), dtype="int16")[0]
    assert (mic_data[:16000 * 3] == 7).all() and (mic_data[16000 * 4:] == 0).any(), "the stalled track was padded"


def test_a_microphone_line_that_goes_on_after_the_echoed_words_is_kept():
    them = [{"t": 10.0, "text": "Thank you everyone"}]
    mine = {"t": 10.5, "text": "Thank you everyone. I disagree with the proposal because the budget is too small."}
    assert not mt.is_echo(mine, them)                                                                 # the answer is not lost
    assert mt.is_echo({"t": 10.5, "text": "thank you everyone"}, them)                               # the bare echo still goes


def test_speaker_names_cannot_break_the_format_and_lines_starting_with_an_angle_bracket_are_escaped():
    assert mt.speaker_name(" <Anna> \n", "You") == "Anna" and mt.speaker_name("<>", "You") == "You" and mt.speaker_name(None, "Them") == "Them"
    plain = {"t": 5.0, "lang": "en", "conf": 1.0, "text": "<laughs> that was good"}                  # Whisper's own tag, no speaker
    written = mt.fmt(plain)
    assert written == "[00:00:05] [en] \u200b<laughs> that was good"
    again = mt.parse_transcript(written)["lines"][0]
    assert again["text"] == "<laughs> that was good" and "speaker" not in again                       # not read as a speaker
    said = dict(plain, speaker="You")
    assert mt.parse_transcript(mt.fmt(said))["lines"][0]["text"] == "<laughs> that was good"          # with a speaker nothing is escaped
    assert mt.parse_transcript(mt.fmt(said))["lines"][0]["speaker"] == "You"


def test_merge_call_orders_by_time_labels_speakers_and_drops_only_echo():
    mine = [{"t": 7.0, "lang": "en", "conf": 1.0, "text": "Thank you, I have a question."},
            {"t": 14.2, "lang": "de", "conf": 1.0, "text": "Ich habe die Zahlen vorbereitet."}]    # the speakers, heard again
    theirs = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Добрый день, коллеги."},
              {"t": 14.0, "lang": "de", "conf": 1.0, "text": "Ich habe die Zahlen vorbereitet."}]
    merged = mt.merge_call(mine, theirs, me="Anna", them="Client")
    assert [(l["t"], l["speaker"]) for l in merged] == [(0.0, "Client"), (7.0, "Anna"), (14.0, "Client")]
    kept = mt.merge_call(mine, theirs, echo=False)
    assert [(l["t"], l["speaker"]) for l in kept] == [(0.0, "Them"), (7.0, "You"), (14.0, "Them"), (14.2, "You")]
    assert "speaker" not in mine[0] and "speaker" not in theirs[0]          # the inputs are not changed


def _fake_mlx(monkeypatch, arm=True):
    import types
    import numpy as np
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64" if arm else "x86_64")
    calls = []
    model = SimpleNamespace(dims=SimpleNamespace(n_mels=128), detect_language=lambda mel: (None, {"en": 0.1, "de": 0.85, "ru": 0.05}))
    audio = SimpleNamespace(N_FRAMES=3000, log_mel_spectrogram=lambda chunk, n_mels: SimpleNamespace(astype=lambda t: "mel"),
                            pad_or_trim=lambda mel, n, axis=-1: mel)
    mx = types.ModuleType("mlx.core"); mx.float16 = "f16"
    mlx = types.ModuleType("mlx"); mlx.core = mx
    lib = types.ModuleType("mlx_whisper")
    lib.audio = audio
    lib.transcribe = lambda chunk, path_or_hf_repo, language, **kw: (calls.append((path_or_hf_repo, language)) or
                                                                    {"segments": [{"start": 0.0, "end": 2.0, "text": f" {language}-text", "avg_logprob": -0.3}]})
    tr = types.ModuleType("mlx_whisper.transcribe"); tr.ModelHolder = SimpleNamespace(get_model=lambda repo, dtype: model)
    lib.transcribe_module = tr
    for name, mod in (("mlx", mlx), ("mlx.core", mx), ("mlx_whisper", lib), ("mlx_whisper.transcribe", tr)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(lib, "transcribe", lib.transcribe)
    return calls


def test_the_apple_gpu_engine_is_chosen_on_apple_silicon_when_installed(monkeypatch):
    monkeypatch.delenv("MT_ENGINE", raising=False)
    monkeypatch.delenv("MT_MLX_MODEL", raising=False)
    _fake_mlx(monkeypatch, arm=True)
    assert mt.engine_name() == "mlx"
    _fake_mlx(monkeypatch, arm=False)
    assert mt.engine_name() == "ct2"                                              # an Intel Mac: the CPU engine
    monkeypatch.setenv("MT_ENGINE", "mlx")
    with pytest.raises(SystemExit, match="needs Apple silicon"):
        mt.engine_name()
    monkeypatch.setenv("MT_ENGINE", "ct2")
    assert mt.engine_name() == "ct2"
    monkeypatch.setattr(mt, "WHISPER", "small")
    assert mt.mlx_repo() == "mlx-community/whisper-small-mlx"
    monkeypatch.setenv("MT_MLX_MODEL", "me/my-whisper")
    assert mt.mlx_repo() == "me/my-whisper"


def test_the_apple_gpu_engine_detects_from_the_mel_and_decodes_with_mlx_whisper(monkeypatch):
    calls = _fake_mlx(monkeypatch, arm=True)
    engine = mt.MLXEngine("mlx-community/whisper-large-v3-turbo")
    state = engine.analyse([0.0] * 16000)
    assert state.probs == [("de", 0.85), ("en", 0.1), ("ru", 0.05)]
    assert engine.decode(state, "de") == [{"start": 0.0, "end": 2.0, "text": " de-text", "logprob": -0.3}]
    assert calls == [("mlx-community/whisper-large-v3-turbo", "de")]
    assert engine.analyse([0.0] * 16000, detect=False).probs is None


def test_ffmpeg_comes_from_the_system_or_is_fetched_once_by_the_engine(monkeypatch, tmp_path):
    import shutil
    mt.ensure_ffmpeg.cache_clear()
    monkeypatch.setattr(mt, "FFMPEG_DIR", tmp_path / "ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: None)                          # no system ffmpeg
    calls = []

    def fetch(download_dir=None, **kw):
        """What static_ffmpeg.run does: the zip's platform folder is unpacked beside download_dir, the binaries are then
        expected inside download_dir (a chmod on a missing file fails)."""
        calls.append(download_dir)
        exe_dir = Path(download_dir)
        unpacked = exe_dir.parent / "darwin_arm64"
        unpacked.mkdir(parents=True, exist_ok=True)
        for tool in ("ffmpeg", "ffprobe"):
            (unpacked / tool).write_text("#!/bin/sh\n"); (unpacked / tool).chmod(0o755)
        for tool in ("ffmpeg", "ffprobe"):
            if not (exe_dir / tool).exists():
                raise FileNotFoundError(exe_dir / tool)
        return str(exe_dir / "ffmpeg"), str(exe_dir / "ffprobe")
    run = SimpleNamespace(get_or_fetch_platform_executables_else_raise=fetch, get_platform_key=lambda: "darwin_arm64")
    monkeypatch.setitem(sys.modules, "static_ffmpeg", SimpleNamespace(run=run))
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", run)
    bin_dir = tmp_path / "ffmpeg" / "darwin_arm64"
    monkeypatch.setattr(shutil, "which", lambda name: str(bin_dir / name) if (bin_dir / name).exists() else None)
    assert mt.ensure_ffmpeg() is True and calls == [str(bin_dir)]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)
    mt.ensure_ffmpeg.cache_clear()
    monkeypatch.setitem(sys.modules, "static_ffmpeg", None)                           # not installed, no system ffmpeg
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert mt.ensure_ffmpeg() is False


# ---------- recording (a fake sound device that delivers silence faster than real time) ----------
class FakeStream:
    """Stands in for sounddevice.InputStream: delivers 0.1 s of silence every 10 ms (10x real time)."""
    def __init__(self, samplerate, channels, dtype, device, callback):
        import numpy as np
        self.callback, self.block = callback, np.zeros((samplerate // 10, channels), dtype=dtype)
        self.stop_flag = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_flag.is_set():
            self.callback(self.block, len(self.block), None, None)
            time.sleep(0.01)

    def stop(self):
        self.stop_flag.set()
        self.thread.join(1)

    def close(self):
        pass


def _fake_recording(monkeypatch, tmp_path):
    pytest.importorskip("soundfile")
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeStream))
    monkeypatch.setattr(mt, "DATA", tmp_path)


def _seconds(wav):
    import soundfile as sf
    with sf.SoundFile(str(wav)) as f:
        return f.frames / f.samplerate


class StampedStream(FakeStream):
    """Each block carries a running number, and the time it was delivered is kept, so a test can tell which blocks
    were captured while the recording was paused."""
    deliveries = []

    def _run(self):
        n = 0
        while not self.stop_flag.is_set():
            n = (n + 1) % 30000
            self.block[:] = n
            StampedStream.deliveries.append((n, time.monotonic()))
            self.callback(self.block, len(self.block), None, None)
            time.sleep(0.01)


@posix_signals
def test_nothing_captured_while_paused_reaches_the_file_even_if_resumed_or_stopped_at_once(monkeypatch, tmp_path):
    import signal
    import soundfile as sf
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=StampedStream))
    StampedStream.deliveries = []
    marks = {}

    def toggle(name):
        marks[name] = time.monotonic()
        os.kill(os.getpid(), signal.SIGUSR2)
    threading.Timer(0.3, lambda: toggle("pause")).start()
    threading.Timer(0.9, lambda: toggle("resume")).start()
    threading.Timer(1.2, lambda: toggle("pause2")).start()
    threading.Timer(1.3, lambda: os.kill(os.getpid(), signal.SIGINT)).start()     # stopped while paused, with blocks pending
    folder, wav, _ = mt.record("x", 1, None)
    written = set(sf.read(str(wav), dtype="int16")[0].reshape(-1).tolist())
    paused = {n for n, t in StampedStream.deliveries if marks["pause"] + 0.02 < t < marks["resume"] - 0.02
              or t > marks["pause2"] + 0.02}
    assert paused and not (paused & written), sorted(paused & written)[:5]
    before = {n for n, t in StampedStream.deliveries if t < marks["pause"] - 0.02}
    assert before <= written, "what came before the pause is all there"


def test_record_stops_at_its_limit_of_recorded_audio_and_warns_before(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "NOTICE_S", 5)
    folder, wav, system = mt.record("x", 10 / 3600, None)                  # a 10 s limit
    assert system is None and wav.name == "audio.wav" and folder.name.endswith("-x")
    assert 10 <= _seconds(wav) <= 20                                       # the cap is checked every 0.5 s (5 s of fake audio)
    err = capsys.readouterr().err
    assert "The recording stops in 5 seconds (0.166667 min limit)" in err and "stopped at the 0.166667 min limit" in err
    assert "Saved" in err


@posix_signals
def test_record_is_extended_by_sigusr1(monkeypatch, tmp_path, capsys):
    import signal
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EXTEND_S", 10)
    monkeypatch.setattr(mt, "NOTICE_S", 1)
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGUSR1)).start()
    folder, wav, _ = mt.record(None, 5 / 3600, None)                        # 5 s, extended by 10 s
    assert _seconds(wav) >= 15
    assert "Recording extended: it now stops after 0.25 min" in capsys.readouterr().err


@posix_signals
def test_an_extend_signal_before_the_recorder_has_its_handler_is_ignored(monkeypatch, tmp_path):
    import signal
    monkeypatch.setattr(mt, "DATA", tmp_path)
    seen = []

    def fake_record(name, hours, device, system_device=None, live=None, notes=None, check=None):
        os.kill(os.getpid(), signal.SIGUSR1)        # as early as the app's Extend can come: record() has no handler yet
        os.kill(os.getpid(), signal.SIGUSR2)        # the app's Pause likewise
        time.sleep(0.2)                              # the default action would have ended the process by now
        seen.append(signal.getsignal(signal.SIGUSR1))
        d = tmp_path / "x"
        d.mkdir()
        return d, d / "audio.wav", None
    monkeypatch.setattr(mt, "record", fake_record)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--no-transcribe"])
    mt.main()
    assert seen == [signal.SIG_IGN]


def test_record_only_reports_the_folder_as_output(monkeypatch, tmp_path, capsys):
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(2 / 3600), "--no-transcribe"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    assert out[-2].startswith("@@OUT\t" + str(tmp_path)) and out[-1] == "@@DONE\t0"
    assert out[0].startswith("@@FOLDER\t" + str(tmp_path)) and len(out) == 4                       # the folder first, as soon as it exists
    assert out[1] == "@@NOTE\tRecording stopped at the 0.0333333 min limit"                        # no 10-minute notice for a 2 s limit


def test_the_help_of_record_only_says_to_transcribe_into_the_recordings_folder(monkeypatch, tmp_path, capsys):
    """Transcribed into its own folder, the recording is screened as one (a video's phrases go by their text); without
    --into, transcribe makes a new folder and screens it as an imported file (1.11.4 review)."""
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--help"])
    with pytest.raises(SystemExit):
        mt.main()
    text = " ".join(capsys.readouterr().out.split())
    assert "transcribe <folder>/audio.wav --into <folder>" in text
    assert "call <folder>/mic.wav <folder>/system.wav --into <folder>" in text
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    d = tmp_path / "20261127-100000-plain"
    d.mkdir()
    assert mt.screening_context(d / "audio.wav", mt._data_dir(str(d), "audio")) == "recording"


def test_events_of_two_threads_never_glue_into_one_line():
    """The live worker's resend (@@CLEAR, then many @@LINE) against the main thread's @@LEVEL and @@NOTE, on a pipe as
    the app reads them: each event is one whole line (1.11.4 review: print() wrote a line and its newline apart,
    '@@CLEAR@@LEVEL\t0.25'; about 1 line in 4 was glued this way in 0.2 s before the fix)."""
    line = json.dumps({"t": 1.0, "lang": "de", "conf": 0.9, "text": "x" * 300, "live": True})
    script = f"""
import sys, threading
sys.path.insert(0, {str(Path(mt.__file__).parent)!r})
import mt
mt.EVENTS = True
line = {line!r}
def worker():
    for k in range(3000):
        mt.ev("CLEAR") if k % 100 == 0 else mt.ev("LINE", line)
def main_thread():
    for k in range(3000):
        mt.note(f"note {{k}}") if k % 2 else mt.ev("LEVEL", 0.25)
threads = [threading.Thread(target=worker), threading.Thread(target=main_thread)]
for t in threads:
    t.start()
for t in threads:
    t.join()
with mt.EV_LOCK:                     # held, as when a signal handler sends an event while ev() runs: it still goes out
    mt.ev("NOTE", "inside")
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    got = done.stdout.splitlines()
    good = re.compile(r"^@@(?:CLEAR|LEVEL\t0\.25|NOTE\tnote \d+|NOTE\tinside|LINE\t" + re.escape(line) + ")$")
    assert len(got) == 6001 and [g for g in got if not good.match(g)] == []


def test_events_stay_whole_on_a_stdout_that_writes_in_pieces(monkeypatch):
    """The one write() per line keeps events whole on CPython's own stdout; EV_LOCK keeps them whole on a stdout that
    lets another thread in between its characters as well (a wrapper, a free-threaded build). Without the lock this
    test glues lines (1.11.4 review: the pipe test above passes without it)."""
    class Piecewise:
        def __init__(self):
            self.out = []

        def write(self, s):
            for c in s:                                            # one character at a time, letting the other thread in
                self.out.append(c)
                time.sleep(0)
            return len(s)

        def flush(self):
            pass

    stdout = Piecewise()
    monkeypatch.setattr(sys, "stdout", stdout)

    def send(kind, text):
        for k in range(200):
            mt.ev(kind, f"{text} {k}")

    threads = [threading.Thread(target=send, args=("LINE", "worker")), threading.Thread(target=send, args=("NOTE", "main"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = "".join(stdout.out).splitlines()
    good = re.compile(r"^@@(?:LINE\tworker|NOTE\tmain) \d+$")
    assert len(got) == 400 and [g for g in got if not good.match(g)] == []


def test_hours_limit_and_recording_name_are_checked_before_anything_starts(monkeypatch, tmp_path):
    _fake_recording(monkeypatch, tmp_path)
    for argv in (["mt.py", "record", "--hours", "9"], ["mt.py", "record", "--hours", "0"], ["mt.py", "record", "--name", "Q3/Q4"],
                 ["mt.py", "call", "a.wav", "b.wav", "--stem", "x:y"]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            mt.main()
        assert "Nothing was started" in str(e.value)
    assert not list(tmp_path.iterdir())


# ---------- a run cut short goes on where it stopped ----------
def _three_chunks(monkeypatch, texts):
    import numpy as np
    spans = [{"start": 16000 * s, "end": 16000 * (s + 6)} for s in (0, 20, 40)]
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.repeat(np.arange(70, dtype="float32"), 16000))
    seg = lambda text: {"start": 0.5, "end": 2.0, "text": text, "logprob": -0.2}
    engine = FakeEngine([([("en", .9)], [seg(" " + t)]) for t in texts])
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    return engine


class Killed(BaseException):
    """Stands for a kill or a crash: not an Exception, so no handler in mt.py swallows it."""


def test_the_eta_uses_the_median_chunk_time_and_is_remembered_for_the_next_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "ETA_AFTER", 2)
    _three_chunks(monkeypatch, ["one", "two", "three"])
    mt.transcribe_lines(tmp_path / "x.wav", "en")
    out = capsys.readouterr().out.splitlines()
    assert not out[0].startswith("@@ETA"), "nothing is known about this computer yet"
    assert sum(1 for o in out if o.startswith("@@ETA\t")) == 1 and (tmp_path / "data" / ".speed.json").exists()
    _three_chunks(monkeypatch, ["one", "two", "three"])
    mt.transcribe_lines(tmp_path / "x.wav", "en")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("@@ETA\t"), "the remembered speed gives an estimate before the first chunk"
    assert json.loads((tmp_path / "data" / ".speed.json").read_text()).keys() == {"fake:" + mt.WHISPER}


def test_a_transcription_cut_short_goes_on_from_its_checkpoint(monkeypatch, tmp_path, capsys):
    engine = _three_chunks(monkeypatch, ["one", "two", "three"])
    real = engine.decode
    killed = []

    def decode(state, language):
        if int(state.chunk[0]) // 20 == 2 and not killed:
            killed.append(True)
            raise Killed()
        return real(state, language)
    monkeypatch.setattr(engine, "decode", decode)
    with pytest.raises(Killed):
        mt.transcribe_lines(tmp_path / "x.wav", "en", checkpoint=tmp_path)
    assert (tmp_path / "partial.json").exists() and len((tmp_path / "partial.jsonl").read_text().splitlines()) == 2
    engine.calls.clear()
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", checkpoint=tmp_path)
    assert [l["text"] for l in lines] == ["one", "two", "three"] and engine.calls == ["en"]    # only the third chunk is decoded
    assert "Going on from an earlier run: 2 of 3 chunks were already done" in capsys.readouterr().err
    engine.calls.clear()
    mt.transcribe_lines(tmp_path / "x.wav", "en", granularity="finest", checkpoint=tmp_path)   # another setting: start over
    assert engine.calls == ["en", "en", "en"]
    doc = mt.transcribe(tmp_path / "x.wav", tmp_path, "en")                                    # the real flow cleans up
    assert (tmp_path / "transcript.json").exists() and not (tmp_path / "partial.json").exists()
    assert not (tmp_path / "transcript.json.tmp").exists() and len(doc["lines"]) == 3


def test_a_failing_chunk_becomes_a_note_and_the_rest_goes_on(monkeypatch, tmp_path, capsys):
    engine = _three_chunks(monkeypatch, ["one", "two", "three"])
    real = engine.decode

    def decode(state, language):
        if int(state.chunk[0]) // 20 == 1:
            raise RuntimeError("boom")
        return real(state, language)
    monkeypatch.setattr(engine, "decode", decode)
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en")
    assert [l.get("text") or l["note"] for l in lines] == ["one", "chunk failed (RuntimeError: boom)", "three"]
    assert "chunk 2 at 00:00:20 failed: RuntimeError: boom" in capsys.readouterr().err
    monkeypatch.setattr(engine, "decode", lambda state, language: (_ for _ in ()).throw(RuntimeError("all")))
    with pytest.raises(RuntimeError, match="every chunk failed"):
        mt.transcribe_lines(tmp_path / "x.wav", "en")


def test_a_chunk_that_failed_is_tried_again_on_the_next_run(monkeypatch, tmp_path, capsys):
    engine = _three_chunks(monkeypatch, ["one", "two", "three"])
    real = engine.decode
    broken = [True]

    def decode(state, language):
        if int(state.chunk[0]) // 20 == 1 and broken[0]:
            raise RuntimeError("boom")
        return real(state, language)
    monkeypatch.setattr(engine, "decode", decode)
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", checkpoint=tmp_path)
    assert [l.get("text") or l["note"] for l in lines] == ["one", "chunk failed (RuntimeError: boom)", "three"]
    broken[0] = False                                                      # the backend works again
    engine.calls.clear()
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", checkpoint=tmp_path)
    assert [l["text"] for l in lines] == ["one", "two", "three"] and engine.calls == ["en"]   # only the failed chunk
    broken[0] = True
    monkeypatch.setattr(engine, "decode", lambda state, language: (_ for _ in ()).throw(RuntimeError("all")))
    with pytest.raises(RuntimeError, match="every chunk failed"):
        mt.transcribe_lines(tmp_path / "y.wav", "en", checkpoint=tmp_path / "y")
    with pytest.raises(RuntimeError, match="every chunk failed"):                 # a rerun must not call the failures done
        mt.transcribe_lines(tmp_path / "y.wav", "en", checkpoint=tmp_path / "y")


def test_the_fingerprint_tells_audio_tracks_apart(tmp_path):
    a = mt.fingerprint(tmp_path / "v.mkv", ["en"], "normal", None, "ct2", track=0)
    b = mt.fingerprint(tmp_path / "v.mkv", ["en"], "normal", None, "ct2", track=1)
    assert a != b and a["track"] == 0 and mt.fingerprint(tmp_path / "v.mkv", ["en"], "normal", None, "ct2")["track"] is None


def test_the_translation_log_depends_on_how_sentences_are_cut(monkeypatch, tmp_path):
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Privet mir. Kak dela"}]
    before = mt._translation_log(tmp_path, "eng_Latn", lines)
    monkeypatch.setattr(mt, "sentences", lambda text: [text])                 # another version cuts differently
    assert mt._translation_log(tmp_path, "eng_Latn", lines) != before


def test_offline_translation_bookkeeping_stays_cheap_for_long_transcripts(monkeypatch):
    _fake_model(monkeypatch)
    doc = {"lines": [{"t": float(i), "lang": "ru", "conf": 1.0, "text": f"stroka {i}"} for i in range(3000)], "translations": {}}
    started = time.monotonic()
    out = mt.translate_offline(doc, "English")
    assert len(out) == 3000 and out[7] == "Tstroka T7"
    assert time.monotonic() - started < 5, "joining sentences must not rescan every sentence for every batch"


def test_the_translation_log_depends_on_the_lines_languages(tmp_path):
    en = [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "Gift"}]
    de = [{"t": 0.0, "lang": "de", "conf": 1.0, "text": "Gift"}]
    assert mt._translation_log(tmp_path, "eng_Latn", en) != mt._translation_log(tmp_path, "eng_Latn", de)
    assert mt._translation_log(tmp_path, "eng_Latn", en) == mt._translation_log(tmp_path, "eng_Latn", list(en))


def test_an_offline_translation_cut_short_reuses_its_finished_sentences(monkeypatch, tmp_path, capsys):
    translator = _fake_model(monkeypatch)
    doc = mt.parse_transcript("[00:00:01] [ru] Privet mir. Kak dela\n[00:00:02] [ru] Poka\n")
    mt.translate_offline(doc, "English", checkpoint=tmp_path)
    logs = list((tmp_path / "translations").glob("eng_Latn-*.jsonl"))
    assert len(logs) == 1 and len(logs[0].read_text().splitlines()) == 3                       # three sentences written down
    kept = logs[0].read_text().splitlines()[:2]
    logs[0].write_text("\n".join(kept) + "\n")                                                # as if the run died after two
    translator.calls.clear()
    out = mt.translate_offline(doc, "English", checkpoint=tmp_path)
    assert out == ["TPrivet Tmir. TKak Tdela", "TPoka"]
    assert [s[1:-1] for tokens, _ in translator.calls for s in tokens] == [["Poka"]]           # only the third sentence again
    assert "2 translated sentences reused" in capsys.readouterr().err


def test_write_outputs_announces_each_file_and_saves_the_doc_after_each_target(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "translate", lambda doc, lang, offline=False, source=None, checkpoint=None: ["X"] * len(doc["lines"]))
    doc = {"source": "s", "whisper": "w", "lines": [{"t": 1.0, "lang": "en", "conf": 1.0, "text": "hi"}], "translations": {}}
    paths = mt.write_outputs(doc, tmp_path, "m", ["Russian", "German"], offline=True, data=tmp_path)
    out = [l for l in capsys.readouterr().out.splitlines() if l.startswith("@@OUT")]
    assert out == [f"@@OUT\t{p}" for p in paths] and len(paths) == 3 and (tmp_path / "transcript.json").exists()


@posix_signals
def test_sigterm_during_a_recording_keeps_a_valid_file_and_transcribes_nothing(monkeypatch, tmp_path, capsys):
    import signal
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "transcribe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not transcribe")))
    threading.Timer(0.3, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", "1"])
    mt.main()
    folder = next(tmp_path.iterdir())
    assert 1 <= _seconds(folder / "audio.wav") < 20                                           # closed properly, header right
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"@@FOLDER\t{folder}" and out[1].startswith("@@NOTE\tRecording cancelled (SIGTERM)")
    assert out[2] == f"@@OUT\t{folder}" and out[3] == "@@DONE\t0"


@posix_signals
def test_a_cancelled_recording_names_its_own_folder_not_the_newest_entry_of_data(monkeypatch, tmp_path, capsys):
    """Another job's folder (or .speed.json) can be newer than the recording's: the folder comes from the recording itself."""
    import signal
    _fake_recording(monkeypatch, tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    decoy = tmp_path / "99991231-235959-other"

    def cancel():
        decoy.mkdir()
        (tmp_path / ".speed.json").write_text("{}")
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Timer(0.3, cancel).start()
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", "1", "--name", "mine"])
    mt.main()
    folder = next(p for p in tmp_path.iterdir() if p.name.endswith("-mine"))
    out = capsys.readouterr().out.splitlines()
    assert f"@@OUT\t{folder}" in out and any(o.startswith("@@NOTE\tRecording cancelled (SIGTERM); the audio is kept in " + str(folder)) for o in out)


def test_unique_stems_reserve_every_name_of_the_batch():
    assert mt.unique_stems(["x/a.wav", "y/b.wav"]) == ["a", "b"]
    assert mt.unique_stems(["a/audio.wav", "b/audio.wav"]) == ["audio", "b-audio"]
    # A made name must not take an input's own name, nor an earlier made one: the Codex review's case.
    assert mt.unique_stems(["a/audio.wav", "b/audio.wav", "c/b-audio.wav"]) == ["audio", "b-audio-2", "b-audio"]
    assert mt.unique_stems(["a/x.wav", "b/x.wav", "b/x.mp3"]) == ["x", "b-x", "b-x-2"]
    assert mt.unique_stems(["x.wav", "x.mp3"]) == ["x", "x-2"]


def test_files_that_share_a_name_get_distinct_output_files_in_one_output_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "audio_only.m4a").write_bytes(b"")
    monkeypatch.setattr(mt, "transcribe", lambda src, d, *a, **k: {"source": str(src), "whisper": "w", "translations": {},
                                                                     "lines": [{"t": 0, "lang": "en", "conf": 1.0, "text": src.parent.name}]})
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(tmp_path / "a" / "audio_only.m4a"), str(tmp_path / "b" / "audio_only.m4a"),
                                      "--out", str(out_dir)])
    mt.main()
    written = sorted(p.name for p in out_dir.glob("*.transcript.txt"))
    assert written == ["audio_only.transcript.txt", "b-audio_only.transcript.txt"], written
    assert (out_dir / "audio_only.transcript.txt").read_text().endswith(" a\n") and (out_dir / "b-audio_only.transcript.txt").read_text().endswith(" b\n")


def test_the_wav_header_is_right_after_every_flush_even_without_a_close(tmp_path):
    import numpy as np
    import soundfile as sf
    w = mt.WavFile(tmp_path / "a.wav")
    w.write(np.zeros((16000, 1), dtype="int16"))
    w.flush()
    with sf.SoundFile(str(tmp_path / "a.wav")) as f:                                           # not closed by us: the header already says 1 s
        assert f.frames == 16000 and f.samplerate == 16000 and f.channels == 1
    w.write(np.ones((8000, 1), dtype="int16"))
    w.close()
    data, rate = sf.read(str(tmp_path / "a.wav"), dtype="int16")
    assert len(data) == 24000 and data[-1] == 1


def test_into_reuses_a_data_folder_by_path_or_id(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    assert mt._data_dir(str(tmp_path / "x"), "n") == tmp_path / "x" and (tmp_path / "x").is_dir()   # a path is created
    assert mt._data_dir("20261110-1200-meeting", "n") == tmp_path / "data" / "20261110-1200-meeting"  # an id lives in data/
    assert mt._data_dir(None, "n").parent == tmp_path / "data"                                       # none: a new folder


# ---------- live: transcribing while the recording goes on ----------
def _content_vad(monkeypatch):
    """A VAD that calls non-zero samples speech: runs of at least 0.25 s, split by silences of at least the pause."""
    import numpy as np

    def spans(audio, opts):
        pause = int(16000 * opts.pause / 1000)
        on = np.abs(np.asarray(audio)) > 0
        out, start, last = [], None, None
        for i in np.flatnonzero(np.diff(np.concatenate([[0], on.astype("int8"), [0]]))):
            if start is None:
                start = int(i)
            else:
                if last is not None and start - last < pause and out:
                    out[-1]["end"] = int(i)
                else:
                    out.append({"start": start, "end": int(i)})
                last = int(i)
                start = None
        return [s for s in out if s["end"] - s["start"] >= 4000]
    vad = SimpleNamespace(VadOptions=lambda **kw: SimpleNamespace(pause=kw.get("min_silence_duration_ms", 700)), get_speech_timestamps=spans)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)


class SecondEngine:
    """Decodes a chunk into one line naming the second it starts at (speech samples carry their second + 1)."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def analyse(self, chunk, detect=True):
        return SimpleNamespace(chunk=chunk, probs=[("en", 0.9)] if detect else None)

    def decode(self, state, language):
        import numpy as np
        first = int(np.asarray(state.chunk)[np.flatnonzero(state.chunk)[0]] * 32768) - 1
        self.calls.append(first)
        return [{"start": 0.0, "end": 1.0, "text": f"t{first}", "logprob": -0.1}]


def _speech(seconds, gaps):
    """int16 audio: second s carries the value s + 1, silent in the given gap seconds."""
    import numpy as np
    audio = np.concatenate([np.full(16000, 0 if s in gaps else s + 1, dtype="int16") for s in range(seconds)])
    return audio.reshape(-1, 1)


def test_live_track_closes_speech_followed_by_a_pause_and_the_final_pass_does_the_tail(monkeypatch, tmp_path, capsys):
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    audio = _speech(23, gaps={6, 7, 14, 15, 22})                      # speech 0-6, 8-14, 16-22
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["en"], "normal", None, mt.engine_name())
    track.feed(audio[:16000 * 10])
    assert track.cut() == 1 and track.chunks == [(0, 96000)]          # 0-6 s closed (a pause follows), 8-10 s still open
    track.feed(audio[16000 * 10:])
    assert track.cut() == 1 and track.chunks[1] == (128000, 224000)   # 8-14 s; 16-22 s ends too close to the end: open
    assert track.decode(engine) == [{"t": 0.0, "end": 1.0, "lang": "en", "conf": 1.0, "text": "t0"}]
    assert track.decode(engine)[0]["text"] == "t8" and track.decode(engine) is None
    track.close()
    out = capsys.readouterr().out.splitlines()
    assert any(o.startswith("@@LINE\t") and '"live": true' in o and '"t8"' in o for o in out) and "@@POS\t14.0\t23.0" in out
    engine.calls.clear()
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio.reshape(-1).astype("float32") / 32768)
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", granularity="normal", checkpoint=tmp_path)
    assert [l["text"] for l in lines] == ["t0", "t8", "t16"] and engine.calls == [16]   # only the tail was decoded
    assert "@@NOTE\tGoing on from an earlier run: 2 of 3 chunks were already done" in capsys.readouterr().out


def test_live_track_closes_a_short_stretch_once_enough_silence_has_followed_it(monkeypatch, tmp_path):
    """A stretch shorter than the chunk minimum is held back while the next span may still join it, but not for good:
    after LIVE_CLOSE_S of silence it is decoded, so a short answer on one side of a call appears without waiting for
    that side to speak again (1.9.11 held it until then, and a short test showed no live line at all)."""
    _content_vad(monkeypatch)
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["en"], "normal", None, mt.engine_name())
    audio = _speech(3 + mt.LIVE_CLOSE_S + 5, gaps=set(range(3, 3 + mt.LIVE_CLOSE_S + 5)))   # 3 s of speech, then silence
    track.feed(audio[:16000 * (3 + mt.LIVE_CLOSE_S - 1)])       # 2 s of silence so far: the next span may still join it
    assert track.cut() == 0 and track.chunks == []
    track.feed(audio[16000 * (3 + mt.LIVE_CLOSE_S - 1):16000 * (3 + mt.LIVE_CLOSE_S)])   # LIVE_CLOSE_S of silence: decoded
    assert track.cut() == 1 and track.chunks == [(0, 48000)]
    track.close()


def test_the_live_worker_is_really_gone_when_stop_returns(monkeypatch, tmp_path, capsys):
    """A slow decode outlives the join timeout: stop() keeps waiting (and says so) instead of handing a checkpoint
    that is still being written to the final pass."""
    import numpy as np
    monkeypatch.setattr(mt, "EVENTS", True)

    class SlowTrack:
        def __init__(self):
            self.pending, self.closed, self.decoded = [1], False, 0

        def cut(self, final=True):
            return 0

        def decode(self, engine):
            if not self.pending:
                return None
            time.sleep(0.5)                                   # longer than the wait below
            self.pending.pop()
            self.decoded += 1
            return [{"t": 0.0, "end": 0.5, "lang": "en", "conf": 1.0, "text": "x"}]

        def close(self):
            self.closed = True
    monkeypatch.setattr(mt, "get_engine", lambda: object())
    track = SlowTrack()
    worker = mt.LiveWorker([track])
    worker.start()
    time.sleep(0.05)
    worker.stop(wait=0.1)
    assert not worker.thread.is_alive() and track.closed and track.decoded == 1
    assert "Still finishing what was transcribed live" in capsys.readouterr().err   # a plain line: it is not a note for the transcript


def test_a_live_worker_whose_model_will_not_load_says_so_instead_of_dying_silently(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)

    def broken():
        raise RuntimeError("no such model")
    monkeypatch.setattr(mt, "get_engine", broken)
    worker = mt.LiveWorker([])
    worker.start()
    worker.thread.join(5)
    assert isinstance(worker.failed, RuntimeError)
    assert "@@NOTE\tLive transcription stopped (RuntimeError: no such model)" in capsys.readouterr().out


def test_stop_does_not_wait_for_a_speech_model_that_is_still_loading(monkeypatch, capsys):
    """Nothing was written to the checkpoint before the model loaded, so the final pass can start at once."""
    import threading
    monkeypatch.setattr(mt, "EVENTS", True)
    release = threading.Event()

    def slow():
        release.wait(10)
        return object()
    monkeypatch.setattr(mt, "get_engine", slow)

    class Track:
        closed = False

        def close(self):
            self.closed = True
    track = Track()
    worker = mt.LiveWorker([track])
    worker.start()
    t0 = time.monotonic()
    worker.stop(wait=0.2)
    assert time.monotonic() - t0 < 2 and track.closed
    assert "@@NOTE\tThe speech model was still loading when the recording stopped" in capsys.readouterr().out
    release.set()
    worker.thread.join(5)                      # its @@STAT model (the load done) is sent here, not into a later test
    assert not worker.thread.is_alive()


def test_live_track_keeps_only_the_audio_its_chunks_still_need(monkeypatch, tmp_path):
    """A recording of hours must not keep every sample in memory: once a stretch is cut and decoded, the audio before
    the next uncut stretch is dropped, while chunk offsets stay absolute in the track."""
    _content_vad(monkeypatch)
    engine = SecondEngine()
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["en"], "normal", None, mt.engine_name())
    audio = _speech(100, gaps={s for s in range(100) if s % 10 >= 8})    # 8 s of speech, 2 s of pause, ten times
    for s in range(0, 100, 5):
        track.feed(audio[16000 * s:16000 * (s + 5)])
        track.cut()
        while track.decode(engine) is not None:
            pass
    assert track.chunks[0] == (0, 128000) and track.chunks[-1][1] == 1568000, track.chunks
    assert track.base > 16000 * 60 and len(track.audio) < 16000 * 45, (track.base, len(track.audio))
    assert engine.calls == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], "every stretch decoded once, from the right offset"
    track.close()


def test_record_live_transcribes_while_recording_and_the_final_pass_reuses_it(monkeypatch, tmp_path, capsys):
    import numpy as np
    pytest.importorskip("soundfile")
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)

    class PatternStream(FakeStream):                                  # 3 s of speech, 1 s of silence, 10x real time
        def __init__(self, samplerate, channels, dtype, device, callback):
            super().__init__(samplerate, channels, dtype, device, callback)
            self.second = 0

        def _run(self):
            while not self.stop_flag.is_set():
                s = self.second
                self.block = np.full((1600, 1), 0 if s % 4 == 3 else s + 1, dtype="int16")
                for _ in range(10):
                    if self.stop_flag.is_set():
                        break
                    self.callback(self.block, 1600, None, None)
                    time.sleep(0.01)
                self.second += 1
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=PatternStream))
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: __import__("soundfile").read(str(src), dtype="float32")[0])
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(24 / 3600), "--live", "--language", "en", "--granularity", "coarse"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    live = [o for o in out if o.startswith("@@LINE\t") and '"live": true' in o]
    assert live, "lines were produced while recording"
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    doc = json.loads((folder / "transcript.json").read_text())
    texts = [l["text"] for l in doc["lines"] if "text" in l]
    assert texts and texts == sorted(texts, key=lambda t: int(t[1:])) and not (folder / "partial.json").exists()
    assert len(engine.calls) == len(set(engine.calls)), "no chunk was decoded twice"
    assert "@@STAGE\tFinishing what was transcribed live" in out


def test_the_app_records_only_and_a_queued_transcribe_into_continues_the_live_work(monkeypatch, tmp_path, capsys):
    """The app's two-process flow: `record --no-transcribe --live` leaves a checkpoint in the data folder; the queued
    `transcribe audio.wav --into <folder> --stem <name>` reuses it and names the files after the recording."""
    import numpy as np
    pytest.importorskip("soundfile")
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)

    class PatternStream(FakeStream):                                  # 3 s of speech, 1 s of silence, 10x real time
        def __init__(self, samplerate, channels, dtype, device, callback):
            super().__init__(samplerate, channels, dtype, device, callback)
            self.second = 0

        def _run(self):
            while not self.stop_flag.is_set():
                s = self.second
                self.block = np.full((1600, 1), 0 if s % 4 == 3 else s + 1, dtype="int16")
                for _ in range(10):
                    if self.stop_flag.is_set():
                        break
                    self.callback(self.block, 1600, None, None)
                    time.sleep(0.01)
                self.second += 1
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=PatternStream))
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: __import__("soundfile").read(str(src), dtype="float32")[0])
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(24 / 3600), "--no-transcribe", "--live", "--name", "standup",
                                      "--language", "en", "--granularity", "coarse"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert out[-2:] == [f"@@OUT\t{folder}", "@@DONE\t0"], "the record-only process names its folder and ends"
    assert f"@@FOLDER\t{folder}" in out[:3], "and names it as soon as it exists"
    live = [o for o in out if o.startswith("@@LINE\t") and '"live": true' in o]
    assert live and (folder / "audio.wav").exists() and not (folder / "transcript.json").exists()
    assert any(p.name.startswith("partial") for p in folder.iterdir()), "the live work is kept for the queued transcription"
    done = list(engine.calls)

    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(folder / "audio.wav"), "--into", str(folder), "--out", str(folder),
                                      "--stem", "standup", "--language", "en", "--granularity", "coarse", "--note", "The Mac slept 10:00:00-10:00:30"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    assert any(o.startswith("@@NOTE\tGoing on from an earlier run") for o in out), "the live chunks are reused"
    assert len(engine.calls) == len(set(engine.calls)) and len(engine.calls) >= len(done), "no chunk was decoded twice"
    doc = json.loads((folder / "transcript.json").read_text())
    texts = [l["text"] for l in doc["lines"] if "text" in l]
    assert texts and texts == sorted(texts, key=lambda t: int(t[1:])) and not (folder / "partial.json").exists()
    assert (folder / "standup.transcript.txt").exists(), "named after the recording, not audio.transcript.txt"
    assert (folder / "standup.transcript.txt").read_text().startswith("# [00:00:00] The Mac slept 10:00:00-10:00:30\n")


def test_transcribe_stem_names_the_files_and_works_for_one_file_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "Hi"}], "translations": {}}
    monkeypatch.setattr(mt, "transcribe", lambda src, d, *a, **k: doc)
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    for f in ("a.wav", "b.wav"):
        (tmp_path / f).write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(tmp_path / "a.wav"), "--stem", "weekly"])
    mt.main()
    assert (tmp_path / "weekly.transcript.txt").exists() and not (tmp_path / "a.transcript.txt").exists()
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(tmp_path / "a.wav"), str(tmp_path / "b.wav"), "--stem", "weekly"])
    with pytest.raises(SystemExit, match="one file at a time"):
        mt.main()
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(tmp_path / "a.wav"), "--stem", "a/b"])
    with pytest.raises(SystemExit):
        mt.main()


def test_mt_threads_overrides_the_core_count_for_a_job_beside_a_recording(monkeypatch):
    mt.cpu_threads.cache_clear()
    monkeypatch.setenv("MT_THREADS", "3")
    assert mt.cpu_threads() == 3
    mt.cpu_threads.cache_clear()
    monkeypatch.setenv("MT_THREADS", "0")
    assert mt.cpu_threads() >= 1                                     # nonsense is ignored
    mt.cpu_threads.cache_clear()
    monkeypatch.delenv("MT_THREADS")
    assert mt.cpu_threads() == (os.cpu_count() or 4) or sys.platform == "darwin"
    mt.cpu_threads.cache_clear()


@posix_signals
def test_follow_transcribes_growing_files_and_stops_on_sigterm(monkeypatch, tmp_path, capsys):
    import signal
    import numpy as np
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    audio = _speech(20, gaps={5, 6, 12, 13, 19})
    other = np.where(audio != 0, audio + 1000, 0).astype("int16")             # the other side says other things
    mic, system = tmp_path / "mic.wav", tmp_path / "system.wav"
    writers = [mt.WavFile(mic), mt.WavFile(system)]

    def grow():
        for s in range(20):
            for w, a in zip(writers, (audio, other)):
                w.write(a[16000 * s:16000 * (s + 1)])
                w.flush()
            time.sleep(0.15)
        for w in writers:
            w.close()
        time.sleep(1.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=grow, daemon=True).start()
    mt.follow([mic, system], tmp_path, "en", "normal", "Anna", "Boris")
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "@@DONE\t0" and any('"speaker": "Boris"' in o and '"live": true' in o for o in out)
    assert (tmp_path / "partial.1.json").exists() and (tmp_path / "partial.2.jsonl").exists()
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: (other if "system" in str(src) else audio).reshape(-1).astype("float32") / 32768)
    engine.calls.clear()
    doc = mt.transcribe_call(mic, system, tmp_path, "Anna", "Boris", "en", "normal")
    assert {l["speaker"] for l in doc["lines"] if "text" in l} == {"Anna", "Boris"}
    assert [l["text"] for l in doc["lines"] if "text" in l] == ["t0", "t1000", "t7", "t1007", "t14", "t1014"]
    assert engine.calls == [14, 1014] and not (tmp_path / "partial.1.json").exists()      # only the tails were decoded


def test_call_transcribes_both_tracks_and_writes_the_merged_transcript(monkeypatch, tmp_path, capsys):
    tracks = {"mic.wav": [{"t": 3.0, "lang": "en", "conf": 1.0, "text": "Hello, can you hear me?"}],
              "system.wav": [{"t": 0.0, "lang": "ru", "conf": 0.9, "text": "Здравствуйте."},
                             {"t": 6.0, "lang": "ru", "conf": 0.9, "text": "Да, слышу вас хорошо."}]}
    monkeypatch.setattr(mt, "transcribe_lines", lambda src, language=None, track=None, granularity="auto", speaker=None, **kw:
                        [dict(l) for l in tracks[Path(src).name]])
    monkeypatch.setattr(mt, "EVENTS", True)
    doc = mt.transcribe_call(tmp_path / "mic.wav", tmp_path / "system.wav", tmp_path, "Anna", "Boris")
    assert [(l["speaker"], l["text"]) for l in doc["lines"]] == [
        ("Boris", "Здравствуйте."), ("Anna", "Hello, can you hear me?"), ("Boris", "Да, слышу вас хорошо.")]
    assert doc["source"].endswith("system.wav") and (tmp_path / "transcript.json").exists()
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"@@FOLDER\t{tmp_path}" and out[1:4] == ["@@STAGE\tTranscribing your microphone (track 1 of 2)",
                                                                "@@STAGE\tTranscribing the other side (track 2 of 2)", "@@CLEAR"]
    merged = [json.loads(o.split("\t", 1)[1]) for o in out[4:]]           # the window gets the merged conversation
    assert [(l["speaker"], l["text"]) for l in merged] == [(l["speaker"], l["text"]) for l in doc["lines"]]


def test_a_track_without_speech_gets_a_note_in_the_transcript_and_the_alert(monkeypatch, tmp_path, capsys):
    import numpy as np
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: [])
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.zeros(16000 * 20, dtype="float32"))
    monkeypatch.setattr(mt, "get_engine", lambda: FakeEngine([]))
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    lines = mt.transcribe_lines(tmp_path / "system.wav", None, position=(2, 2))
    assert lines == [{"t": 20.0, "note": "No speech was found in the computer's sound: the other side is missing from the "
                                         "transcript (was anything playing?)"}]
    assert "@@NOTE\tNo speech was found in the computer's sound" in capsys.readouterr().out
    assert mt.transcribe_lines(tmp_path / "mic.wav", None, position=(1, 2))[0]["note"].startswith("No speech was found in your microphone")
    assert mt.transcribe_lines(tmp_path / "quiet.wav", None) == [{"t": 20.0, "note": "No speech was found in quiet.wav"}]
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.zeros(0, dtype="float32"))      # a header-only file
    assert mt.transcribe_lines(tmp_path / "empty.wav", None)[0]["note"] == "No speech was found in empty.wav"


def test_a_call_whose_other_side_cannot_be_transcribed_keeps_your_side_and_reports_the_problem(monkeypatch, tmp_path, capsys):
    def fake_lines(src, language=None, track=None, granularity="auto", speaker=None, **kw):
        if Path(src).name == "system.wav":
            raise RuntimeError("Invalid data found when processing input")
        return [{"t": 3.0, "lang": "en", "conf": 1.0, "text": "Hello?"}]
    monkeypatch.setattr(mt, "transcribe_lines", fake_lines)
    monkeypatch.setattr(mt, "EVENTS", True)
    problems = []
    doc = mt.transcribe_call(tmp_path / "mic.wav", tmp_path / "system.wav", tmp_path, problems=problems)
    text = "The computer's sound could not be transcribed (RuntimeError: Invalid data found when processing input); only your side is in the transcript"
    assert [l.get("note") or l["text"] for l in doc["lines"]] == [text, "Hello?"]
    assert doc["lines"][1]["speaker"] == "You" and problems == [text] and (tmp_path / "transcript.json").exists()
    out = capsys.readouterr().out.splitlines()
    assert any(o.startswith("@@NOTE\tThe computer's sound could not be transcribed") for o in out)
    assert any(o == f"@@ERR\t{tmp_path / 'system.wav'}\t{text}" for o in out)
    assert out[-1].startswith("@@LINE\t")                                     # the window still gets the merged lines


def test_the_call_command_writes_the_files_and_ends_with_exit_1_when_the_other_side_failed(monkeypatch, tmp_path, capsys):
    import soundfile as sf
    import numpy as np
    for name in ("mic.wav", "system.wav"):
        sf.write(str(tmp_path / name), np.zeros(16000, dtype="int16"), 16000)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(mt, "transcribe_lines", lambda src, *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
                        if Path(src).name == "system.wav" else [{"t": 1.0, "lang": "en", "conf": 1.0, "text": "Hi"}])
    monkeypatch.setattr(sys, "argv", ["mt.py", "call", str(tmp_path / "mic.wav"), str(tmp_path / "system.wav")])
    with pytest.raises(SystemExit) as e:
        mt.main()
    assert e.value.code == 1
    text = (tmp_path / "call.transcript.txt").read_text(encoding="utf-8")       # the default stem of `call`
    assert "<You> Hi" in text and "# [00:00:00] The computer's sound could not be transcribed (RuntimeError: boom)" in text
    out = capsys.readouterr().out.splitlines()
    assert any(o.startswith("@@ERR\t") for o in out) and f"@@OUT\t{tmp_path / 'call.transcript.txt'}" in out and out[-1] == "@@DONE\t0"


def test_release_engine_frees_the_cached_model(monkeypatch):
    import functools
    calls = []

    @functools.lru_cache(maxsize=1)
    def fake_model():
        calls.append(1)
        return object()
    monkeypatch.setattr(mt, "get_model", fake_model)
    monkeypatch.setattr(mt, "CT2Engine", lambda model: SimpleNamespace(name="ct2", model=model))
    mt.get_engine.cache_clear()
    first = mt.get_engine()
    assert mt.get_engine() is first and calls == [1]
    mt.release_engine()                                   # must not raise: both caches exist and are cleared
    assert mt.get_engine() is not first and calls == [1, 1]
    monkeypatch.undo()
    assert hasattr(mt.get_model, "cache_clear"), "the real speech-model loader is cached (release_engine clears it)"
    mt.get_engine.cache_clear()


def test_release_engine_lets_go_of_the_mlx_library_model_too(monkeypatch):
    _fake_mlx(monkeypatch, arm=True)
    holder = sys.modules["mlx_whisper.transcribe"].ModelHolder
    holder.model, holder.model_path = object(), "mlx-community/whisper-large-v3-turbo"
    mt.release_engine()
    assert holder.model is None and holder.model_path is None


def test_the_fingerprint_names_the_mlx_model_repository(monkeypatch):
    monkeypatch.setenv("MT_MLX_MODEL", "mlx-community/whisper-small-mlx")
    assert mt.fingerprint("x.wav", ["en"], "normal", None, "mlx")["model"] == "mlx-community/whisper-small-mlx"
    monkeypatch.setenv("MT_MLX_MODEL", "mlx-community/whisper-large-v3-turbo")
    assert mt.fingerprint("x.wav", ["en"], "normal", None, "mlx")["model"] == "mlx-community/whisper-large-v3-turbo"
    assert mt.fingerprint("x.wav", ["en"], "normal", None, "ct2")["model"] == mt.WHISPER


# ---------- who spoke: speaker separation ----------
def _fake_sherpa(monkeypatch, segments, found_threads=None):
    """A sherpa_onnx that answers with the given (start, end, speaker) segments and records how it was configured."""
    calls = []

    class Result:
        def __init__(self):
            self.items = [SimpleNamespace(start=s, end=e, speaker=k) for s, e, k in segments]

        def sort_by_start_time(self):
            return sorted(self.items, key=lambda r: r.start)

    class Diarization:
        def __init__(self, cfg):
            calls.append(cfg)
            self.sample_rate = 16000

        def process(self, samples, callback=None):
            if callback:
                callback(1, 2)
                callback(2, 2)
            return Result()
    ns = lambda **kw: SimpleNamespace(validate=lambda: True, **kw)
    lib = SimpleNamespace(OfflineSpeakerDiarizationConfig=ns, OfflineSpeakerSegmentationModelConfig=ns,
                          OfflineSpeakerSegmentationPyannoteModelConfig=ns, SpeakerEmbeddingExtractorConfig=ns,
                          FastClusteringConfig=ns, OfflineSpeakerDiarization=Diarization)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", lib)
    monkeypatch.setattr(mt, "ensure_diarization_models", lambda: (Path("/m/seg.onnx"), Path("/m/emb.onnx")))
    return calls


def test_diarize_numbers_speakers_in_order_of_appearance_and_passes_the_count(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    calls = _fake_sherpa(monkeypatch, [(5.0, 8.0, 7), (0.0, 4.0, 3), (9.0, 12.0, 3)])
    assert mt.diarize([0.0] * 16000, "auto") == [(0.0, 4.0, 0), (5.0, 8.0, 1), (9.0, 12.0, 0)]
    assert calls[-1].clustering.num_clusters == -1 and calls[-1].clustering.threshold == mt.CLUSTER_THRESHOLD
    mt.diarize([0.0] * 16000, 3)
    assert calls[-1].clustering.num_clusters == 3
    assert "@@PROG\t1\t2" in capsys.readouterr().out.splitlines()
    with pytest.raises(ValueError, match="unknown diarizer"):
        mt.diarize([0.0], "auto", "magic")


def test_turn_cuts_fall_between_speakers_and_ignore_short_turns_and_small_gaps():
    segs = [(0.0, 4.0, 0), (4.2, 6.0, 0), (6.5, 10.0, 1), (10.2, 10.6, 0), (10.8, 14.0, 1), (14.5, 18.0, 0)]
    assert mt.turn_cuts(segs) == [6.25, 14.25]                    # 4.0-4.2 is the same speaker; the 0.4 s turn makes no cut
    assert mt.split_at([(0, 16000 * 20)], [6.25, 14.25]) == [(0, 100000), (100000, 228000), (228000, 320000)]
    assert mt.split_at([(0, 16000 * 20)], [0.2, 19.9]) == [(0, 320000)]  # a cut at the very edge does not split


def test_lines_get_the_speaker_that_overlaps_them_most_and_overlaps_become_notes():
    segs = [(0.0, 5.0, 0), (4.0, 9.0, 1), (9.5, 12.0, 0)]
    lines = [{"t": 0.5, "end": 3.0, "lang": "en", "conf": 1.0, "text": "a"},
             {"t": 4.5, "end": 8.5, "lang": "en", "conf": 1.0, "text": "b"},
             {"t": 20.0, "end": 21.0, "lang": "en", "conf": 1.0, "text": "nobody"},
             {"t": 10.0, "note": "chunk failed"}, {"t": 9.6, "lang": "ru", "conf": 1.0, "text": "c (no end)"}]
    out = mt.label_speakers(lines, segs)
    assert [l.get("speaker") for l in out] == ["Speaker 1", "Speaker 2", None, None, "Speaker 1"]
    assert mt.label_speakers(lines[:1], segs, "Them")[0]["speaker"] == "Them 1"
    notes = mt.overlap_notes(segs)
    assert notes == [{"t": 4.0, "note": "overlap 00:00:04\u201300:00:05 Speaker 1 / Speaker 2"}]
    assert mt.overlap_notes([(0.0, 5.0, 0), (4.8, 9.0, 1)]) == []           # 0.2 s is not an overlap worth a note
    merged = mt.with_notes(out, notes)
    assert [l.get("text") or l["note"] for l in merged][:3] == ["a", "overlap 00:00:04\u201300:00:05 Speaker 1 / Speaker 2", "b"]


def test_transcribe_with_speakers_cuts_at_turns_labels_lines_and_announces_the_count(monkeypatch, tmp_path, capsys):
    engine = _three_chunks(monkeypatch, ["one", "two", "three"])
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    _fake_sherpa(monkeypatch, [(0.0, 7.0, 0), (19.0, 27.0, 1), (39.0, 47.0, 0)])
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", speakers="auto")
    assert [l.get("speaker") for l in lines if "text" in l] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    out = capsys.readouterr().out.splitlines()
    assert "@@STAGE\tSeparating speakers" in out and "@@NOTE\t2 speakers found" in out
    assert out.count("@@CLEAR") == 1 and sum(1 for o in out if o.startswith("@@LINE\t") and "Speaker 2" in o) == 1
    doc = mt.transcribe(tmp_path / "x.wav", tmp_path / "d", "en", speakers=2)
    assert doc["lines"][1]["speaker"] == "Speaker 2" and out[0].startswith("@@") and (tmp_path / "d" / "transcript.json").exists()


def test_speaker_turns_split_a_chunk_that_spans_two_people(monkeypatch, tmp_path):
    import numpy as np
    spans = [{"start": 0, "end": 16000 * 20}]                                 # one 20 s stretch without a pause
    vad = SimpleNamespace(VadOptions=lambda **kw: None, get_speech_timestamps=lambda audio, opts: _spans_for(audio, spans))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(vad=vad))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: np.zeros(16000 * 20, dtype="float32"))
    starts = []

    class Engine:
        name = "fake"

        def analyse(self, chunk, detect=True):
            starts.append(len(chunk))
            return SimpleNamespace(chunk=chunk, probs=[("en", 1.0)])

        def decode(self, state, language):
            return [{"start": 0.0, "end": 1.0, "text": " x", "logprob": -0.1}]
    monkeypatch.setattr(mt, "get_engine", lambda: Engine())
    _fake_sherpa(monkeypatch, [(0.0, 9.0, 0), (9.4, 20.0, 1)])
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", speakers=2)
    assert starts == [16000 * 9 + 3200, 16000 * 20 - 16000 * 9 - 3200], "the chunk was cut where the speaker changed"
    assert [l["speaker"] for l in lines] == ["Speaker 1", "Speaker 2"]


def test_a_live_chunk_that_spans_a_speaker_change_is_cut_and_decoded_again_by_the_final_pass(monkeypatch, tmp_path, capsys):
    """The live run knows no speakers, so its chunk may hold two people; the final pass with --speakers splits it at the
    turn and decodes the pieces (the other live chunks are reused), and the checkpoint is written anew with the new numbering."""
    _content_vad(monkeypatch)
    engine = SecondEngine()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "EVENTS", True)
    audio = _speech(30, gaps={10, 11, 22, 23})                 # speech 0-10 (two people: a turn at 5 s), 12-22, 24-30
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["en"], "normal", None, mt.engine_name())
    track.feed(audio[:16000 * 26])
    assert track.cut() == 2 and track.chunks == [(0, 160000), (192000, 352000)]
    while track.decode(engine) is not None:
        pass
    track.close()
    engine.calls.clear()
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio.reshape(-1).astype("float32") / 32768)
    _fake_sherpa(monkeypatch, [(0.0, 4.9, 0), (5.1, 10.0, 1), (12.0, 22.0, 0), (24.0, 30.0, 1)])
    lines = mt.transcribe_lines(tmp_path / "x.wav", "en", granularity="normal", checkpoint=tmp_path, speakers="auto")
    spoken = [(l["speaker"], l["text"]) for l in lines if "text" in l]
    assert spoken == [("Speaker 1", "t0"), ("Speaker 2", "t5"), ("Speaker 1", "t12"), ("Speaker 2", "t24")], spoken
    assert engine.calls == [0, 5, 24], "the split chunk's two pieces and the tail; the untouched live chunk was reused"
    out = capsys.readouterr().out
    assert "@@NOTE\t1 chunk transcribed live span a change of speaker and are decoded again" in out
    assert "@@NOTE\tGoing on from an earlier run: 1 of 4 chunks were already done" in out


def test_a_call_separates_the_other_side_into_numbered_speakers(monkeypatch, tmp_path, capsys):
    tracks = {"mic.wav": [{"t": 3.0, "end": 5.0, "lang": "en", "conf": 1.0, "text": "Hello?"}],
              "system.wav": [{"t": 0.0, "end": 2.0, "lang": "ru", "conf": 0.9, "text": "Привет", "speaker": "Them 1"},
                             {"t": 6.0, "end": 8.0, "lang": "ru", "conf": 0.9, "text": "Да", "speaker": "Them 2"}]}
    seen = {}

    def lines(src, language=None, track=None, granularity="auto", speaker=None, **kw):
        seen[Path(src).name] = kw
        return [dict(l) for l in tracks[Path(src).name]]
    monkeypatch.setattr(mt, "transcribe_lines", lines)
    doc = mt.transcribe_call(tmp_path / "mic.wav", tmp_path / "system.wav", tmp_path, "Anna", "Them", speakers="auto")
    assert {k: seen["system.wav"][k] for k in ("speakers", "diarizer", "label")} == {"speakers": "auto", "diarizer": "sherpa", "label": "Them"}
    assert {k: seen["mic.wav"][k] for k in ("speakers", "diarizer", "label")} == {"speakers": "auto", "diarizer": "sherpa", "label": "Anna"}
    assert [(l["speaker"], l["text"]) for l in doc["lines"]] == [("Them 1", "Привет"), ("Anna", "Hello?"), ("Them 2", "Да")], \
        "one voice found on the microphone: its lines keep the plain name"
    plain = mt.transcribe_call(tmp_path / "mic.wav", tmp_path / "system.wav", tmp_path, "Anna", "Boris")
    assert {l["speaker"] for l in plain["lines"]} == {"Anna", "Boris"}, "without --speakers the other side is one name"
    assert "speakers" not in seen["mic.wav"], "and the microphone is not separated either"


def test_a_call_separates_several_people_on_the_microphone_too(monkeypatch, tmp_path, capsys):
    # A room on your side of the call (several people at one microphone): they are numbered like the other side.
    tracks = {"mic.wav": [{"t": 1.0, "end": 2.0, "lang": "en", "conf": 1.0, "text": "Hello from the room", "speaker": "You 1"},
                          {"t": 4.0, "end": 5.0, "lang": "de", "conf": 1.0, "text": "Ich bin auch hier", "speaker": "You 2"},
                          {"t": 4.5, "note": "overlap 00:00:04–00:00:05 You 1 / You 2"}],
              "system.wav": [{"t": 2.5, "end": 3.5, "lang": "ru", "conf": 0.9, "text": "Привет", "speaker": "Them 1"}]}
    seen = {}

    def lines(src, language=None, track=None, granularity="auto", speaker=None, **kw):
        seen[Path(src).name] = kw
        return [dict(l) for l in tracks[Path(src).name]]
    monkeypatch.setattr(mt, "transcribe_lines", lines)
    doc = mt.transcribe_call(tmp_path / "mic.wav", tmp_path / "system.wav", tmp_path, speakers="3")
    assert seen["system.wav"]["speakers"] == "3" and seen["mic.wav"]["speakers"] == "auto", \
        "a number of speakers counts the other side; your side is found by itself"
    assert [(l.get("speaker"), l.get("text") or l.get("note")) for l in doc["lines"]] == [
        ("You 1", "Hello from the room"), ("Them 1", "Привет"), ("You 2", "Ich bin auch hier"),
        ("You", "overlap 00:00:04–00:00:05 You 1 / You 2")]


def test_rename_gives_speakers_real_names_in_the_store_and_the_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "DATA", tmp_path)
    d = tmp_path / "20261110-1200-talk"
    d.mkdir()
    doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "speaker": "Speaker 1", "text": "Hi"},
                     {"t": 1.0, "note": "overlap 00:00:01\u201300:00:02 Speaker 1 / Speaker 2"},
                     {"t": 2.0, "lang": "en", "conf": 1.0, "speaker": "Speaker 2", "text": "Hey"}], "translations": {}}
    mt._save(d, doc)
    (d / "talk.transcript.txt").write_text(mt.render(doc), encoding="utf-8")
    (d / "talk.Russian.txt").write_text(mt.render(doc, ["Привет", None, "Эй"]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["mt.py", "rename", "20261110-1200-talk", "Speaker 1=Anna", "Speaker 2=<Boris>"])
    mt.main()
    stored = mt._load(d)
    assert stored["speakers"] == {"Speaker 1": "Anna", "Speaker 2": "Boris"}
    assert mt.render(stored).splitlines() == ["[00:00:00] [en] <Anna> Hi", "# [00:00:01] overlap 00:00:01\u201300:00:02 Anna / Boris",
                                              "[00:00:02] [en] <Boris> Hey"]
    assert (d / "talk.Russian.txt").read_text().splitlines()[0] == "[00:00:00] [en] <Anna> Привет"
    assert "<Speaker" not in (d / "talk.transcript.txt").read_text()
    monkeypatch.setattr(sys, "argv", ["mt.py", "rename", str(d), "Anna=Anya"])          # a folder path works too; renaming again
    mt.main()
    assert mt._load(d)["speakers"]["Speaker 1"] == "Anya" and "<Anya> Hi" in (d / "talk.transcript.txt").read_text()
    with pytest.raises(SystemExit, match="LABEL=NAME"):
        monkeypatch.setattr(sys, "argv", ["mt.py", "rename", str(d), "nonsense"])
        mt.main()
    monkeypatch.setattr(sys, "argv", ["mt.py", "rename", str(d), "Anya=Boris", "Boris=Anya"])      # the two were mixed up: swap
    mt.main()
    assert mt._load(d)["speakers"] == {"Speaker 1": "Boris", "Speaker 2": "Anya"}
    assert (d / "talk.transcript.txt").read_text().splitlines() == ["[00:00:00] [en] <Boris> Hi", "# [00:00:01] overlap 00:00:01\u201300:00:02 Boris / Anya",
                                                                     "[00:00:02] [en] <Anya> Hey"]


def test_a_line_over_several_turns_goes_to_the_speaker_with_the_most_time_in_it():
    segs = [(0.0, 3.0, 0), (3.0, 7.0, 1), (7.0, 10.0, 0)]            # Speaker 1 has 6 s of the line, Speaker 2 one 4 s turn
    line = {"t": 0.0, "end": 10.0, "lang": "en", "conf": 1.0, "text": "a long live chunk"}
    assert mt.label_speakers([line], segs)[0]["speaker"] == "Speaker 1"


def test_the_pyannote_backend_is_optional_and_maps_its_annotation(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    with pytest.raises(RuntimeError, match="requirements-pyannote.txt"):
        mt.diarize([0.0] * 16000, "auto", "pyannote")

    class Turn:
        def __init__(self, s, e):
            self.start, self.end = s, e

    class Annotation:
        def itertracks(self, yield_label=False):
            return [(Turn(0.0, 2.0), None, "SPEAKER_01"), (Turn(2.5, 4.0), None, "SPEAKER_00"), (Turn(4.5, 6.0), None, "SPEAKER_01")]
    seen = {}

    class Pipeline:
        @staticmethod
        def from_pretrained(name, token=None):
            seen["model"], seen["token"] = name, token
            return Pipeline()

        def __call__(self, audio, **kw):
            seen["kw"] = kw
            return SimpleNamespace(speaker_diarization=Annotation())
    class Tensor:
        def __init__(self, a):
            self.a = a

        def __getitem__(self, i):
            return self.a
    torch = SimpleNamespace(from_numpy=Tensor)
    monkeypatch.setitem(sys.modules, "pyannote", SimpleNamespace(audio=SimpleNamespace(Pipeline=Pipeline)))
    monkeypatch.setitem(sys.modules, "pyannote.audio", sys.modules["pyannote"].audio)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert mt.diarize([0.0] * 16000, 2, "pyannote") == [(0.0, 2.0, 0), (2.5, 4.0, 1), (4.5, 6.0, 0)]
    assert seen == {"model": mt.PYANNOTE_PIPELINE, "token": "hf_x", "kw": {"num_speakers": 2}}


def test_the_speaker_models_are_fetched_once_from_the_archive_and_the_release(monkeypatch, tmp_path, capsys):
    import io, tarfile
    monkeypatch.setattr(mt, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(mt, "EVENTS", True)
    fetched = []

    def download(url, dest):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith(".tar.bz2"):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
                for name, data in ((mt.SEGMENTATION_MODEL[0] + "/model.onnx", b"SEG"), (mt.SEGMENTATION_MODEL[0] + "/model.int8.onnx", b"small")):
                    info = tarfile.TarInfo(name); info.size = len(data); tar.addfile(info, io.BytesIO(data))
            dest.write_bytes(buf.getvalue())
        else:
            dest.write_bytes(b"EMB")
    monkeypatch.setattr(mt, "_download", download)
    seg, emb = mt.ensure_diarization_models()
    assert seg.read_bytes() == b"SEG" and emb.read_bytes() == b"EMB" and len(fetched) == 2
    assert not list((tmp_path / "models").glob("*.tar.bz2")) and "@@STAGE\tFetching the speaker-separation models" in capsys.readouterr().out
    assert mt.ensure_diarization_models() == (seg, emb) and len(fetched) == 2, "fetched once"


def test_speakers_argument_is_checked_before_anything_starts(monkeypatch):
    assert mt._speakers(SimpleNamespace(speakers=None)) is None and mt._speakers(SimpleNamespace(speakers="auto")) == "auto"
    assert mt._speakers(SimpleNamespace(speakers="3")) == 3
    with pytest.raises(SystemExit, match="Nothing was started"):
        mt._speakers(SimpleNamespace(speakers="two"))


def test_resolve_device_takes_an_index_an_exact_name_or_a_unique_part_of_one(monkeypatch):
    devices = [{"name": "MacBook Pro Microphone", "max_input_channels": 1}, {"name": "MacBook Pro Speakers", "max_input_channels": 0},
               {"name": "Jabra Evolve2", "max_input_channels": 1}, {"name": "jabra evolve2", "max_input_channels": 1},
               {"name": "BlackHole 2ch", "max_input_channels": 2}]
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=lambda: devices))
    assert mt.resolve_device(None) is None, "no choice: the default input"
    assert mt.resolve_device("4") == 4 and mt.resolve_device(" 0 ") == 0
    assert mt.resolve_device("macbook pro microphone") == 0, "the exact name, whatever the case"
    assert mt.resolve_device("Jabra Evolve2") == 2, "an exact match wins over the case-insensitive twin"
    assert mt.resolve_device("black") == 4, "a part of a name that fits one input only"
    assert mt.resolve_device("macbook") == 0, "the speakers have no input, so they do not make 'macbook' ambiguous"
    for bad, why in (("1", "1 is not an input device"), ("7", "7 is not an input device"), ("Airpods", "No input device is called 'Airpods'")):
        with pytest.raises(SystemExit) as e:
            mt.resolve_device(bad)
        assert why in str(e.value) and "Nothing was started" in str(e.value) and "  4  BlackHole 2ch" in str(e.value), (bad, e.value)
    with pytest.raises(SystemExit) as e:
        mt.resolve_device("JABRA EVOLVE2", "computer's-sound")
    assert "2 input devices match 'JABRA EVOLVE2'" in str(e.value) and "  2  Jabra Evolve2\n" in str(e.value) \
        and "  3  jabra evolve2\n" in str(e.value), "two case-insensitive twins: the index is asked for"
    with pytest.raises(SystemExit) as e:
        mt.resolve_device("Sony", "computer's-sound")
    assert str(e.value).startswith("No computer's-sound device is called 'Sony'")
    # two identical devices (the app sends name#k for the k-th of a name, in the order the devices are listed)
    devices.append({"name": "Jabra Evolve2", "max_input_channels": 1})
    assert mt.resolve_device("Jabra Evolve2#1") == 2 and mt.resolve_device("Jabra Evolve2#2") == 5
    assert mt.resolve_device("BlackHole 2ch#1") == 4, "#1 of a name that has one input is that input"
    with pytest.raises(SystemExit) as e:
        mt.resolve_device("Jabra Evolve2#3")
    assert "asks for input number 3 of the name 'Jabra Evolve2', but 2 inputs of that name are connected" in str(e.value)
    with pytest.raises(SystemExit) as e:
        mt.resolve_device("Jabra Evolve2")
    assert "2 input devices match 'Jabra Evolve2'; give the index instead, or NAME#2" in str(e.value)


# ---------- the output folder: one subfolder per meeting hour, with everything about the meeting ----------
def test_result_folder_takes_out_then_the_output_folders_hour_subfolder_then_the_default(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "OUT", None)
    assert mt.result_folder("/x", tmp_path) == Path("/x") and mt.result_folder(None, tmp_path) == tmp_path
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    data = tmp_path / "20261122-140512-weekly"
    assert mt.result_folder(None, data, data) == tmp_path / "out" / "26-11-22 14"      # the hour the recording began
    assert mt.result_folder("/x", data, data) == Path("/x")                            # --out still wins
    assert re.fullmatch(r"\d\d-\d\d-\d\d \d\d", mt.result_folder(None, tmp_path).name)  # no recording: the current hour
    assert mt.hour_stamp(tmp_path / "not-a-stamp") == mt.hour_stamp()


def test_two_meetings_of_one_hour_keep_their_files_apart_and_a_rerun_and_rename_find_them(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    monkeypatch.setattr(mt, "DATA", tmp_path)
    out = tmp_path / "out" / "26-11-22 14"
    docs = []
    for k in (1, 2):
        d = tmp_path / f"20261122-14{k:02}00-"
        d.mkdir()
        doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "speaker": "Speaker 1", "text": f"meeting {k}"}], "translations": {}}
        mt._save(d, doc)
        docs.append((d, doc))
        mt.write_outputs(doc, mt.result_folder(None, d, d), "recording", [], data=d)
    assert (out / "recording.transcript.txt").read_text().endswith("meeting 1\n")
    assert (out / "recording-2.transcript.txt").read_text().endswith("meeting 2\n"), "the second meeting of the hour is not overwritten"
    assert "called recording-2.*" in capsys.readouterr().err
    d, doc = docs[1]
    assert mt._load(d)["outputs"][str(out)] == {"stem": "recording-2", "files": [str(out / "recording-2.transcript.txt")]}
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    paths = mt.write_outputs(mt._load(d), out, "recording", ["English"], data=d)          # a rerun keeps its own stem
    assert [p.name for p in paths] == ["recording-2.transcript.txt", "recording-2.English.txt"]
    mt.rename_speakers(d, {"Speaker 1": "Anna"})                                           # the files in the output folder too
    assert "<Anna> meeting 2" in (out / "recording-2.transcript.txt").read_text()
    assert "<Anna> T0" in (out / "recording-2.English.txt").read_text()
    assert "<Speaker 1> meeting 1" in (out / "recording.transcript.txt").read_text(), "the other meeting's files are untouched"


def test_the_recording_is_kept_beside_its_transcript_in_the_output_folder_but_not_a_file_of_the_user(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    monkeypatch.setattr(mt, "EVENTS", True)
    d = tmp_path / "20261122-140512-call"
    d.mkdir()
    (d / "mic.wav").write_bytes(b"MIC"), (d / "system.wav").write_bytes(b"SYS")
    doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "Hi"}], "translations": {}}
    mt._save(d, doc)
    out = mt.result_folder(None, d, d)
    paths = mt.write_outputs(doc, out, "call", [], data=d, audio=[(d / "mic.wav", ".mic.wav"), (d / "system.wav", ".system.wav")])
    assert [p.name for p in paths] == ["call.transcript.txt", "call.mic.wav", "call.system.wav"]
    assert (out / "call.mic.wav").read_bytes() == b"MIC" and (out / "call.system.wav").read_bytes() == b"SYS"
    assert [o for o in capsys.readouterr().out.splitlines() if o.startswith("@@OUT")] == [f"@@OUT\t{p}" for p in paths]
    assert mt.write_outputs(doc, out, "call", [], data=d, audio=[(d / "mic.wav", ".mic.wav")]) == [out / "call.transcript.txt"], "already there: not copied again"
    theirs = tmp_path / "interview.m4a"
    theirs.write_bytes(b"THEIRS")
    d2 = tmp_path / "20261122-143000-interview"                   # another meeting of the hour: a file of the user's
    d2.mkdir()
    doc2 = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "Hi"}], "translations": {}}
    mt._save(d2, doc2)
    assert mt.write_outputs(doc2, out, "interview", [], data=d2, audio=[(theirs, ".wav")]) == [out / "interview.transcript.txt"]
    assert not (out / "interview.wav").exists(), "a file the user gave stays where it is"
    monkeypatch.setattr(mt, "OUT", None)
    assert mt.write_outputs(doc, tmp_path / "plain", "call", [], data=d, audio=[(d / "mic.wav", ".mic.wav")]) == [tmp_path / "plain" / "call.transcript.txt"], "without MT_OUT nothing is copied"


# ---------- live translation: the translation beside each line while the recording goes on ----------
def test_record_live_translates_each_line_as_it_appears_and_the_final_pass_reuses_it(monkeypatch, tmp_path, capsys):
    import numpy as np
    pytest.importorskip("soundfile")
    _content_vad(monkeypatch)
    engine = SecondEngine()
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)

    class PatternStream(FakeStream):                                  # 3 s of speech, 1 s of silence, 10x real time
        def __init__(self, samplerate, channels, dtype, device, callback):
            super().__init__(samplerate, channels, dtype, device, callback)
            self.second = 0

        def _run(self):
            while not self.stop_flag.is_set():
                s = self.second
                self.block = np.full((1600, 1), 0 if s % 4 == 3 else s + 1, dtype="int16")
                for _ in range(10):
                    if self.stop_flag.is_set():
                        break
                    self.callback(self.block, 1600, None, None)
                    time.sleep(0.01)
                self.second += 1
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=PatternStream))
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: __import__("soundfile").read(str(src), dtype="float32")[0])
    # 40 s at 10x real time: the first chunk (0-15 s) is decoded and translated while recording. Since 1.13.0 a chunk
    # decoded after the stop is not translated live (the stop leaves the translation to the final pass): it is sent
    # once more with tr_skip instead.
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(40 / 3600), "--live", "--language", "en", "--granularity", "coarse",
                                      "--to", "Russian", "--offline"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    lines = [json.loads(o.split("\t", 1)[1]) for o in out if o.startswith("@@LINE\t")]
    live = [l for l in lines if l.get("live")]
    shown_first = [l for l in live if "tr" not in l and "tr_skip" not in l]
    translated = [l for l in live if "tr" in l]
    assert shown_first and translated, "a line is shown at once, then again with its translation"
    assert all(l["tr"] == {"Russian": "T" + l["text"]} and l["tr_offline"] for l in translated)
    assert all(any(t["t"] == s["t"] for t in translated) or any(k["t"] == s["t"] and k["tr_skip"] == ["Russian"] for k in live if "tr_skip" in k)
               for s in shown_first), "every live line got its translation, or is marked as left to the final pass"
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    doc = json.loads((folder / "transcript.json").read_text())
    stored = [l for l in doc["lines"] if "tr" in l]                  # the tail is translated at the end, not on its lines
    assert stored and all(l["tr"]["Russian"] == "T" + l["text"] for l in stored), "the live translations are in the stored meeting"
    assert (folder / "recording.Russian.txt").exists()
    reused = [o for o in out if o.startswith("@@NOTE\tTranslated") and "translated live" in o]
    assert reused, "the final translation says what it took from the live one"
    final = [l for l in lines if not l.get("live") and "tr" in l]
    assert final and all(l["tr"]["Russian"] == "T" + l["text"] for l in final), "the window sees every line again with its translation"


def test_live_translation_with_claude_and_a_failure_that_ends_it_with_a_note(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"}, {"t": 1.0, "note": "x"}, {"t": 2.0, "lang": "ru", "conf": 1.0, "text": "Пока"}]
    worker = mt.LiveWorker([], targets=["English", "German"])
    worker.translate(lines)                                                  # 1.13.0: queued, made by the worker's steps
    _drain_translator(worker)
    assert lines[0]["tr"] == {"English": "T0", "German": "T0"} and lines[2]["tr"] == {"English": "T1", "German": "T1"}
    assert "tr" not in lines[1] and lines[0]["tr_offline"] is False

    def broken(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(mt, "claude_translate", broken)                      # the request itself fails (on the helper thread)
    worker.translate(lines)
    _drain_translator(worker)
    worker.translate(lines)
    _drain_translator(worker)
    assert worker.translation_off and capsys.readouterr().out.count("@@NOTE\tLive translation stopped (RuntimeError: no network)") == 1


def test_the_final_translation_reuses_live_lines_made_the_same_way_only(monkeypatch):
    calls = []

    class Counting(FakeAnthropic):
        def create(self, **kw):
            calls.append(kw["messages"][0]["content"].split("\n\n", 1)[1])
            return super().create(**kw)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Counting))
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "a", "tr": {"English": "LIVE"}, "tr_of": "a", "tr_offline": False},
                     {"t": 1.0, "lang": "ru", "conf": 1.0, "text": "b"},
                     {"t": 2.0, "lang": "ru", "conf": 1.0, "text": "c", "tr": {"English": "OFFLINE"}, "tr_of": "c", "tr_offline": True}],
           "translations": {}}
    assert mt.translate(doc, "English") == ["LIVE", "T0", "T1"]
    assert calls == ["0|b\n1|c"], "only the lines not translated live with Claude were sent"
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    assert mt.translate(doc, "English", offline=True) == ["Ta", "Tb", "OFFLINE"], "offline reuses only what the offline model made"


def test_follow_takes_the_live_translation_targets(monkeypatch, tmp_path):
    seen = {}

    class Worker:
        def __init__(self, tracks, targets=(), offline=False, source=None):
            seen.update(targets=list(targets), offline=offline)
            self.tracks = tracks

        def start(self):
            raise KeyboardInterrupt                            # out of follow() before its loop
    monkeypatch.setattr(mt, "LiveWorker", Worker)
    monkeypatch.setattr(mt, "LiveTrack", lambda *a, **k: object())
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(sys, "argv", ["mt.py", "follow", str(tmp_path / "mic.wav"), str(tmp_path / "system.wav"), "--into", str(tmp_path),
                                      "--to", "English", "--to", "German"])
    with pytest.raises(KeyboardInterrupt):
        mt.main()
    assert seen == {"targets": ["English", "German"], "offline": False}
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):            # the key is checked before the follower starts
        mt.main()


# ---------- review fixes (1.10.0): translations tied to their text, reruns in the output folder ----------
def test_a_live_translation_is_reused_only_for_the_text_it_was_made_from(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    theirs = [{"t": 10.0, "end": 12.0, "lang": "en", "conf": 1.0, "text": "we should meet on Friday"}]
    mine = [{"t": 10.1, "end": 13.0, "lang": "en", "conf": 1.0, "text": "we should meet on Friday that's fine by me",
             "tr": {"German": "STALE"}, "tr_of": "we should meet on Friday that's fine by me", "tr_offline": False}]
    lines = mt.merge_call(mine, theirs)                              # the echo is trimmed: the text changed, tr did not
    you = next(l for l in lines if l["speaker"] == "You")
    assert you["text"] == "that's fine by me" and you["tr"]["German"] == "STALE"
    assert mt.translated_live(lines, "German", False) == {}, "a translation of another text is not reused"
    assert "STALE" not in mt.translate({"lines": lines, "translations": {}}, "German")


def test_a_line_claude_left_out_live_gets_no_translation_and_is_sent_again_at_the_end(monkeypatch):
    class Partial:
        def __init__(self): self.messages = self
        def create(self, **kw):
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(text="0|Hello")])   # line 1 missing
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Partial))
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"}, {"t": 1.0, "lang": "ru", "conf": 1.0, "text": "Пока"}]
    mt.translate_live(lines, ["English"])
    assert lines[0]["tr"] == {"English": "Hello"} and lines[0]["tr_of"] == "Привет"
    assert "tr" not in lines[1], "no translation is made up from the original"
    assert mt.translated_live(lines, "English", False) == {0: "Hello"}


def test_a_reply_cut_off_at_the_token_limit_drops_its_last_line(monkeypatch):
    class Cut:
        def __init__(self): self.messages = self
        def create(self, **kw):
            return SimpleNamespace(stop_reason="max_tokens", content=[SimpleNamespace(text="0|Hello\n1|Goo")])
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=Cut, APIConnectionError=ConnectionError))
    got, whole = mt.claude_translate(Cut(), "English", ["Привет", "Пока"])
    assert got == {"0": "Hello"} and not whole


def test_offline_live_translation_leaves_a_language_the_model_lacks_to_the_final_note(monkeypatch):
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}, {"t": 1.0, "lang": "en", "conf": 1.0, "text": "hi"},
             {"t": 2.0, "lang": "xx", "conf": 1.0, "text": "???"}]
    mt.translate_live(lines, ["English"], offline=True)
    assert lines[0]["tr"] == {"English": "Tprivet"} and lines[1]["tr"] == {"English": "hi"} and "tr" not in lines[2]


def test_a_translation_model_that_will_not_load_leaves_the_live_transcript_running(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setattr(mt, "get_engine", lambda: object())

    def no_model(**kw):
        raise OSError("not downloaded and no network")
    monkeypatch.setattr(mt, "get_nllb", no_model)

    class Track:
        def __init__(self): self.pending, self.decoded, self.closed = [1], 0, False
        def cut(self, final=False): return 0
        def decode(self, engine, translate=None):
            if not self.pending:
                return None
            self.pending.pop()
            self.decoded += 1
            lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "x"}]
            if translate:
                translate(lines)
            return lines
        def close(self): self.closed = True
    track = Track()
    worker = mt.LiveWorker([track], targets=["English"], offline=True)
    worker.start()
    time.sleep(0.2)
    worker.stop(wait=2)
    out = capsys.readouterr().out
    assert track.decoded == 1 and worker.failed is None and worker.translation_off
    assert "@@NOTE\tLive translation is off (OSError" in out and "Live transcription stopped" not in out


class FrozenClock(mt.datetime):
    """datetime with now() stopped at one second, so that two jobs begin in the same second."""
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 11, 26, 15, 20, 11)


def test_two_jobs_of_one_name_in_the_same_second_get_data_folders_of_their_own(monkeypatch, tmp_path):
    """A batch of short clips with one file name in two folders (a/x.wav, b/x.wav), or a recording and a queued job,
    began in the same second: the second one failed with FileExistsError (Codex on #99). Now it takes -2, -3 ...; a
    folder another process made first is skipped the same way, and a job without a name numbers its folder alone."""
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "datetime", FrozenClock)
    made = [mt._new_dir("x") for _ in range(3)]
    assert [d.name for d in made] == ["20261126-152011-x", "20261126-152011-x-2", "20261126-152011-x-3"]
    assert all(d.is_dir() for d in made)
    (tmp_path / "20261126-152011-y").mkdir()                       # made by another process a moment before
    assert mt._new_dir("y").name == "20261126-152011-y-2"
    assert [mt._new_dir("").name, mt._new_dir("").name] == ["20261126-152011", "20261126-152011-recording-2"]
    assert mt._data_dir(None, "x").name == "20261126-152011-x-4"     # a new job's folder goes the same way
    # The readers of a folder's name keep the date and take the number as part of the name.
    assert re.sub(r"^\d{8}-\d{6}-?", "", made[1].name) == "x-2"
    assert mt.hour_stamp(made[1]) == mt.hour_stamp(made[0])


def test_a_second_nameless_folder_of_one_second_is_read_as_recording_2_not_2(monkeypatch, tmp_path):
    """A bare `<stamp>-2` made its readers call the meeting "2": translate wrote 2.<Language>.txt, and the app would list
    it as "2" and transcribe it with --stem 2. Numbered as recording-2, it reads as the hour folder's second recording."""
    _fake_model(monkeypatch)
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "datetime", FrozenClock)
    monkeypatch.setattr(mt, "OUT", None)
    monkeypatch.setattr(mt, "EVENTS", False)
    mt._new_dir(None)
    folder = mt._new_dir(None)
    assert folder.name == "20261126-152011-recording-2"
    mt._save(folder, {"source": "s", "whisper": "w", "lines": [{"t": 1.0, "lang": "ru", "conf": 1.0, "text": "Privet"}],
                      "translations": {}})
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", folder.name, "--offline", "--to", "English"])   # by its id
    mt.main()
    assert [p.name for p in folder.glob("*.English.txt")] == ["recording-2.English.txt"]


def test_a_data_folder_that_is_no_folder_is_named_in_the_error_not_counted_through(monkeypatch, tmp_path):
    """mkdir(parents=True) also raises FileExistsError when DATA itself exists but is no folder (a file; a dangling link,
    as a moved MT_HOME leaves): that is not a taken number, and must not run 999 times into "999 data folders of that
    name in one second"."""
    data = tmp_path / "data"
    monkeypatch.setattr(mt, "DATA", data)
    data.write_text("")
    with pytest.raises(FileExistsError) as e:
        mt._new_dir("x")
    assert Path(e.value.filename) == data and "999" not in str(e.value)
    data.unlink()
    try:
        data.symlink_to(tmp_path / "moved away")
    except OSError:                                  # Windows without the right to make links: the file case stands
        return
    with pytest.raises(FileExistsError) as e:
        mt._new_dir("x")
    assert Path(e.value.filename) == data and "999" not in str(e.value)


def test_record_takes_its_folder_the_same_way(monkeypatch, tmp_path):
    _fake_recording(monkeypatch, tmp_path)
    names = []
    real = mt._new_dir
    monkeypatch.setattr(mt, "_new_dir", lambda name, check=None: names.append(name) or real(name, check))
    folder = mt.record("x", 1 / 3600, None)[0]
    assert names == ["x"] and folder.parent == tmp_path and re.fullmatch(r"\d{8}-\d{6}-x(-\d+)?", folder.name)


def test_a_rerun_of_the_same_input_replaces_its_files_in_the_hour_folder_and_another_meeting_is_numbered(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    monkeypatch.setattr(mt, "DATA", tmp_path / "data")
    src = tmp_path / "call.m4a"
    src.write_bytes(b"x")
    out = mt.OUT / mt.hour_stamp()
    first = {"source": str(src), "lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "first"}], "translations": {}}
    mt.write_outputs(first, mt.result_folder(None, src.parent), "call", [], data=mt._new_dir("call"))
    again = {"source": str(src), "lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "again"}], "translations": {}}
    time.sleep(1.1)                                                   # another data folder: a plain `transcribe` run again
    paths = mt.write_outputs(again, mt.result_folder(None, src.parent), "call", [], data=mt._new_dir("call"))
    assert [p.name for p in paths] == ["call.transcript.txt"] and (out / "call.transcript.txt").read_text().endswith("again\n")
    assert "Another meeting" not in capsys.readouterr().err, "the same input is not another meeting"
    other = tmp_path / "elsewhere" / "call.m4a"
    other.parent.mkdir()
    other.write_bytes(b"y")
    doc = {"source": str(other), "lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "other"}], "translations": {}}
    time.sleep(1.1)
    assert mt.write_outputs(doc, out, "call", [], data=mt._new_dir("call"))[0].name == "call-2.transcript.txt"


def test_transcribing_a_meeting_again_in_its_data_folder_keeps_its_name_and_a_later_translation_keeps_the_file_list(monkeypatch, tmp_path):
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    d = tmp_path / "20261122-140512-weekly"
    d.mkdir()
    (d / "audio.wav").write_bytes(b"AUDIO")
    lines = [{"t": 0.0, "lang": "en", "conf": 1.0, "speaker": "Speaker 1", "text": "hello"}]
    monkeypatch.setattr(mt, "transcribe_lines", lambda *a, **k: [dict(l) for l in lines])
    out = mt.OUT / "26-11-22 14"
    (out).mkdir(parents=True)
    (out / "weekly.transcript.txt").write_text("another meeting\n")         # an earlier meeting of the hour, same name
    for _ in range(2):                                                        # transcribed, then transcribed again (--into)
        doc = mt.transcribe(d / "audio.wav", d)
        paths = mt.write_outputs(doc, mt.result_folder(None, d, d), "weekly", [], data=d, audio=[(d / "audio.wav", ".wav")])
    assert [p.name for p in paths] == ["weekly-2.transcript.txt"], "the rerun keeps its own name; the audio is not cloned twice"
    assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".")) == ["weekly-2.transcript.txt", "weekly-2.wav", "weekly.transcript.txt"]
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    mt.write_outputs(mt._load(d), out, "weekly", ["German"], transcript=False, data=d)        # the Translate button later
    files = mt._load(d)["outputs"][str(out)]["files"]
    assert [Path(f).name for f in files] == ["weekly-2.transcript.txt", "weekly-2.wav", "weekly-2.German.txt"]
    mt.rename_speakers(d, {"Speaker 1": "Anna"})
    assert "<Anna> hello" in (out / "weekly-2.transcript.txt").read_text() and "<Anna>" in (out / "weekly-2.German.txt").read_text()
    assert (out / "weekly.transcript.txt").read_text() == "another meeting\n"


def test_translating_a_text_file_into_the_output_folder_never_overwrites_a_meetings_translation(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    out = mt.OUT / mt.hour_stamp()
    out.mkdir(parents=True)
    (out / "call.transcript.txt").write_text(SAMPLE)
    (out / "call.Spanish.txt").write_text("the recording's own Spanish\n")
    downloaded = tmp_path / "Downloads" / "call.transcript.txt"
    downloaded.parent.mkdir()
    downloaded.write_text(SAMPLE)
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", str(downloaded), "--to", "Spanish"])
    mt.main()
    assert (out / "call.Spanish.txt").read_text() == "the recording's own Spanish\n"
    assert (out / "call-2.Spanish.txt").exists()
    mt.main()                                                                 # the same file again: its own name is reused
    assert not (out / "call-3.Spanish.txt").exists()
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", str(out / "call.transcript.txt"), "--to", "German"])
    mt.main()                                                                 # a meeting's own transcript: beside itself
    assert (out / "call.German.txt").exists()


def test_a_live_worker_that_fails_tells_the_window_with_a_check(monkeypatch, capsys):
    monkeypatch.setattr(mt, "EVENTS", True)

    def broken():
        raise OSError("no model")
    monkeypatch.setattr(mt, "get_engine", broken)
    worker = mt.LiveWorker([])
    worker.start()
    worker.stop(wait=2)
    out = capsys.readouterr().out.splitlines()
    assert "@@CHECK\tlive\tfailed" in out and any(o.startswith("@@NOTE\tLive transcription stopped (OSError") for o in out)


def test_lines_sent_again_with_a_file_translation_carry_only_translations_of_their_own_text(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setattr(mt, "EVENTS", True)
    doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "fine by me", "tr": {"German": "STALE"}, "tr_of": "we meet fine by me"},
                     {"t": 1.0, "lang": "en", "conf": 1.0, "text": "ok", "tr": {"German": "gut"}, "tr_of": "ok"}],
           "translations": {}}
    mt.write_outputs(doc, tmp_path, "call", ["Russian"])
    sent = [json.loads(o.split("\t", 1)[1]) for o in capsys.readouterr().out.splitlines() if o.startswith("@@LINE\t")]
    assert sent[0]["tr"] == {"Russian": "T0"} and sent[0]["tr_of"] == "fine by me", "the stale German is not sent with it"
    assert sent[1]["tr"] == {"German": "gut", "Russian": "T1"} and sent[1]["tr_of"] == "ok"


def test_lines_sent_again_after_each_target_keep_the_targets_translated_before(monkeypatch, tmp_path, capsys):
    # The window replaces a line's translations with the ones sent; German after English must not take English away.
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setattr(mt, "EVENTS", True)
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "a"},
                     {"t": 1.0, "lang": "ru", "conf": 1.0, "text": "b", "tr": {"French": "live"}, "tr_of": "b"}],
           "translations": {}}
    mt.write_outputs(doc, tmp_path, "m", ["English", "German"])
    sent = [json.loads(o.split("\t", 1)[1]) for o in capsys.readouterr().out.splitlines() if o.startswith("@@LINE\t")]
    assert [sorted(l["tr"]) for l in sent] == [["English"], ["English", "French"],
                                              ["English", "German"], ["English", "French", "German"]]
    assert "tr" not in doc["lines"][0], "the doc's own lines are not changed"


def test_same_named_transcript_files_translated_in_one_batch_into_the_output_folder_keep_both_results(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(mt, "OUT", tmp_path / "out")
    files = []
    for day in ("mon", "tue"):
        f = tmp_path / day / "audio_only.transcript.txt"
        f.parent.mkdir()
        f.write_text(f"[00:00:01] [ru] {day}\n")
        files.append(str(f))
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", *files, "--to", "English"])
    mt.main()
    out = mt.OUT / mt.hour_stamp()
    assert sorted(p.name for p in out.glob("*.txt")) == ["audio_only-2.English.txt", "audio_only.English.txt"]


# ---------- TranslateGemma: the second offline translator (--translator gemma) ----------
class FakeGemmaTokenizer:
    """Stands in for mlx-lm's TokenizerWrapper: the chat template's message is kept, its index is the prompt."""
    def __init__(self):
        self.messages, self.eos = [], set()

    def add_eos_token(self, token):
        self.eos.add(token)

    def apply_chat_template(self, msgs, add_generation_prompt=True):
        assert add_generation_prompt and len(msgs) == 1 and msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert isinstance(content, list) and len(content) == 1 and content[0]["type"] == "text"   # the template's rule
        self.messages.append(content[0])
        return [len(self.messages) - 1] * 10


def _gemma_module(generate):
    """An mlx_lm stand-in from a generate(): stream_generate yields its reply in two parts (1.13.0 streams)."""
    def stream_generate(model, tokenizer, prompt, **kw):
        out = generate(model, tokenizer, prompt, **kw)
        yield SimpleNamespace(text=out[:3], finish_reason=None)
        yield SimpleNamespace(text=out[3:], finish_reason="stop")
    return SimpleNamespace(generate=generate, stream_generate=stream_generate)


def _fake_gemma(monkeypatch, reply=lambda item: f"<{item['target_lang_code']}>{item['text']}"):
    tok = FakeGemmaTokenizer()
    calls = []

    def generate(model, tokenizer, prompt, max_tokens=256, **kw):
        item = tokenizer.messages[prompt[0]]
        calls.append((item, max_tokens))
        return reply(item)

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):      # 1.13.0 streams; generate() is its concatenation
        out = generate(model, tokenizer, prompt, max_tokens, **kw)
        yield SimpleNamespace(text=out[:3], finish_reason=None)
        yield SimpleNamespace(text=out[3:], finish_reason="stop")
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(generate=generate, stream_generate=stream_generate,
                                                                load=lambda repo: ("model", tok)))
    monkeypatch.setattr(mt, "_load_gemma", lambda: ("model", tok))
    monkeypatch.setattr(mt, "release_engine", lambda: None)
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    return tok, calls


def test_gemma_codes_cover_the_51_evaluated_languages_and_their_variants():
    assert len(mt.GEMMA_LANGS) == 51
    assert [mt.gemma_code(n) for n in ("English", "ru", "Russian", "Brazilian Portuguese", "Portuguese", "European Portuguese",
                                        "Traditional Chinese", "Chinese", "Filipino", "Tagalog", "Zulu", "zul_Latn", "German")] == \
        ["en", "ru", "ru", "pt-BR", "pt", "pt-PT", "zh-TW", "zh", "tl", "tl", "zu", "zu", "de"]
    assert [mt.gemma_code(n) for n in ("Khmer", "Basque", "yue", "Klingon", "", None)] == [None] * 6   # not evaluated: NLLB's


def test_the_template_language_table_knows_every_code_given_to_it():
    """The codes the 51 languages are sent as must be keys of the model's own template (checked against its file, kept
    here as the list of the template's codes for these languages)."""
    template_codes = set("en ar bg bn ca cs da de el es et fa fi tl fr gu he hi hr hu id is it ja kn ko lt lv ml mr nl no pa pl "
                         "pt ro ru sk sl sr sv sw ta te th tr uk ur vi zh zu pt-BR pt-PT zh-TW".split())
    assert set(mt.GEMMA_LANGS) | set(mt.GEMMA_VARIANTS.values()) <= template_codes


def test_translating_a_meeting_with_gemma_sends_the_templates_message_and_gives_other_languages_to_nllb(monkeypatch, capsys):
    tok, calls = _fake_gemma(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"},
                     {"t": 1.0, "lang": "en", "conf": 1.0, "text": "Hello"},
                     {"t": 2.0, "note": "a note"},
                     {"t": 3.0, "lang": "km", "conf": 1.0, "text": "khmer words"},
                     {"t": 4.0, "lang": "de", "conf": 1.0, "text": "Hallo"}], "translations": {}}
    out = mt.translate(doc, "English", offline=True)
    assert out == ["<en>Привет", "Hello", None, "Tkhmer Twords", "<en>Hallo"]
    assert [(c["source_lang_code"], c["target_lang_code"], c["text"]) for c, _ in calls] == [("ru", "en", "Привет"), ("de", "en", "Hallo")]
    sources = {sentence[0] for tokens, _ in translator.calls for sentence in tokens}
    assert sources == {"khm_Khmr"}, "only the Khmer line went to NLLB"
    assert f"en [{mt.GEMMA}]" in doc["translations"] and f"eng_Latn [{mt.NLLB}]" not in doc["translations"]
    err = capsys.readouterr().err
    assert "with TranslateGemma; 1 already in English; 1 in a language TranslateGemma was not evaluated on go to NLLB" in err


def test_the_stop_token_is_added_and_a_leaked_end_of_turn_is_cut(monkeypatch):
    tok = FakeGemmaTokenizer()
    module = _gemma_module(lambda *a, **k: "Hello<end_of_turn>junk")
    module.load = lambda repo: ("model", tok)
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: "/m"))
    mt._load_gemma.cache_clear()
    monkeypatch.setattr(mt, "release_engine", lambda: None)
    try:
        assert mt.gemma_translate([("ru", "Привет")], "en") == ["Hello"]
        assert tok.eos == {"<end_of_turn>"}
    finally:
        mt._load_gemma.cache_clear()


def test_a_long_text_goes_to_gemma_in_pieces_of_whole_sentences(monkeypatch):
    tok, calls = _fake_gemma(monkeypatch, reply=lambda item: f"[{len(item['text'])}]")
    text = " ".join(f"Sentence number {k} is here." for k in range(200))           # about 5,600 characters
    out = mt.gemma_translate([("en", text)], "de")[0]
    pieces = [c["text"] for c, _ in calls]
    assert len(pieces) >= 4 and all(len(p) <= mt.GEMMA_PIECE for p in pieces) and " ".join(pieces) == text
    assert out == " ".join(f"[{len(p)}]" for p in pieces)
    long_sentence = "word " * 700                                                 # one sentence of 3,500 characters
    assert all(len(p) <= mt.GEMMA_PIECE for p in mt._pieces(long_sentence.strip(), mt.GEMMA_PIECE))
    assert " ".join(mt._pieces(long_sentence.strip(), mt.GEMMA_PIECE)) == long_sentence.strip(), "cut at words, nothing lost"


def test_a_gemma_translation_cut_short_goes_on_from_its_log(monkeypatch, tmp_path):
    tok, calls = _fake_gemma(monkeypatch)
    lines = [{"t": float(k), "lang": "ru", "conf": 1.0, "text": f"строка {k}"} for k in range(4)]
    seen = []

    def reply(item):
        if len(seen) == 2:
            raise KeyboardInterrupt                                          # the run is stopped after two lines
        seen.append(item["text"])
        return "EN " + item["text"]
    monkeypatch.setitem(sys.modules, "mlx_lm", _gemma_module(lambda m, t, prompt, **k: reply(t.messages[prompt[0]])))
    with pytest.raises(KeyboardInterrupt):
        mt.translate({"lines": lines, "translations": {}}, "English", offline=True, checkpoint=tmp_path)
    seen.clear()
    monkeypatch.setitem(sys.modules, "mlx_lm", _gemma_module(lambda m, t, prompt, **k: "EN " + t.messages[prompt[0]]["text"]))
    calls_before = len(tok.messages)
    out = mt.translate({"lines": lines, "translations": {}}, "English", offline=True, checkpoint=tmp_path)
    assert out == [f"EN строка {k}" for k in range(4)] and len(tok.messages) - calls_before == 2, "only the last two were sent"


def test_live_gemma_translations_are_marked_and_reused_only_by_gemma(monkeypatch):
    tok, calls = _fake_gemma(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"}, {"t": 1.0, "lang": "km", "conf": 1.0, "text": "khmer"}]
    mt.translate_live(lines, ["English"], offline=True)
    assert lines[0]["tr"] == {"English": "<en>Привет"} and lines[0]["tr_by"] == {"English": mt.GEMMA}
    assert lines[1]["tr"] == {"English": "Tkhmer"} and lines[1]["tr_by"] == {"English": mt.NLLB}
    assert mt.translated_live(lines, "English", True) == {0: "<en>Привет", 1: "Tkhmer"}, "each line by the model that made it"
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    assert mt.translated_live(lines, "English", True) == {1: "Tkhmer"}, "with NLLB at the end, TranslateGemma's line is made again"


def test_translator_option_and_its_checks(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(mt, "OUT", None)
    seen = []
    monkeypatch.setattr(mt, "translate", lambda doc, target, offline=False, source=None, checkpoint=None:
                        seen.append((mt.TRANSLATOR, offline)) or [l.get("text") for l in doc["lines"]])
    (tmp_path / "a.txt").write_text("[00:00:01] [ru] привет\n")
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", str(tmp_path / "a.txt"), "--to", "English", "--offline", "--translator", "gemma"])
    mt.main()
    assert seen == [("gemma", True)], "--translator sets the run's translator"


def test_without_mlx_lm_the_gemma_choice_falls_back_to_nllb_with_a_note_instead_of_refusing(monkeypatch, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: False)
    mt._need_key(["English"], offline=True)                                    # a recording is not refused over it
    assert mt.TRANSLATOR == "nllb" and "TranslateGemma is not available here" in capsys.readouterr().err
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    mt._need_key([], offline=True)                                             # nothing to translate: nothing changes
    assert mt.TRANSLATOR == "gemma"


def test_the_regional_codes_are_accepted_by_both_models(monkeypatch):
    assert [mt.nllb_code(c) for c in ("pt-BR", "pt-PT", "zh-TW", "zh-Hant", "fil", "zh-CN")] == \
        ["por_Latn", "por_Latn", "zho_Hant", "zho_Hant", "tgl_Latn", "zho_Hans"]
    assert [mt.gemma_code(c) for c in ("zho_Hant", "zho_Hans", "zh-CN")] == ["zh-TW", "zh", "zh"]
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: True)
    mt._need_key(["pt-BR", "zh-TW", "fil"], offline=True)                     # no "does not know"


def test_which_models_a_translation_needs(monkeypatch):
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    assert mt.translation_models(["English"], True) == [(mt.GEMMA, "about 7 GB"), (mt.NLLB, "about 1.4 GB")]
    assert mt.translation_models(["Afrikaans"], True) == [(mt.NLLB, "about 1.4 GB")], "TranslateGemma only for its languages"
    assert mt.translation_models(["English"], False) == [] and mt.translation_models([], True) == []
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    assert mt.translation_models(["English"], True) == [(mt.NLLB, "about 1.4 GB")]


def test_a_recording_and_a_call_follower_do_not_wait_for_a_model_download(monkeypatch, tmp_path):
    fetched = []
    monkeypatch.setattr(mt, "fetch_model", lambda repo, size, stage=True: fetched.append(repo))
    mt.preflight(0, 1, ["English"], fetch=False)
    assert fetched == []
    mt.preflight(0, 1, ["English"])
    assert fetched == [mt.NLLB], "a processing job fetches before it starts"
    calls = []

    class Stop(Exception):
        pass

    def stop(*a, **k):
        calls.append(k)
        raise Stop
    monkeypatch.setattr(mt, "preflight", stop)
    monkeypatch.setattr(mt, "resolve_device", lambda *a, **k: None)
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: True)
    for argv in (["record", "--no-transcribe", "--live", "--offline", "--to", "English"],
                 ["follow", str(tmp_path / "m.wav"), str(tmp_path / "s.wav"), "--into", str(tmp_path), "--offline", "--to", "English"]):
        monkeypatch.setattr(sys, "argv", ["mt.py", *argv])
        with pytest.raises(Stop):
            mt.main()
    assert [c.get("fetch") for c in calls] == [False, False]


def test_live_translation_waits_for_the_download_then_loads_the_model_on_the_workers_thread(monkeypatch, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    import threading
    loaded_on, gate = [], threading.Event()
    translator = FakeTranslator()

    def load(**kw):
        loaded_on.append(threading.get_ident())
        return translator, FakeTokenizer()
    monkeypatch.setattr(mt, "get_nllb", load)
    monkeypatch.setattr(mt, "model_present", lambda repo: False)
    monkeypatch.setattr(mt, "fetch_model", lambda repo, size, stage=True: gate.wait(5))
    worker = mt.LiveWorker([], targets=["English"], offline=True)
    # What run() starts (1.13.0): the prepare thread downloads the missing model, then loads NLLB (a CPU model) itself;
    # the worker's thread (this one here) only queues and steps, and never waits.
    prepare = threading.Thread(target=worker.translator.prepare, daemon=True)
    prepare.start()
    first = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}]
    worker.translate(first)
    assert not worker.step()
    assert "tr" not in first and loaded_on == [], "nothing waits for the download"
    assert "starts once the translation model is downloaded" in capsys.readouterr().err
    gate.set()
    worker.models_here.wait(5)
    prepare.join(5)
    second = [{"t": 1.0, "lang": "ru", "conf": 1.0, "text": "poka"}]
    worker.translate(second)
    _drain_translator(worker)
    assert second[0]["tr"] == {"English": "Tpoka"} and loaded_on == [prepare.ident], "loaded once, on the prepare thread"


def test_gemma_with_a_target_it_lacks_and_with_an_nllb_that_cannot_load(monkeypatch, capsys):
    tok, calls = _fake_gemma(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}], "translations": {}}
    assert mt.translate(doc, "Afrikaans", offline=True) == ["Tprivet"] and calls == []
    assert "Afrikaans is not among TranslateGemma's languages: NLLB translates it" in capsys.readouterr().err

    def no_nllb(**kw):
        raise OSError("offline")
    monkeypatch.setattr(mt, "get_nllb", no_nllb)
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}, {"t": 1.0, "lang": "ms", "conf": 1.0, "text": "apa khabar"},
                     {"t": 2.0, "lang": "en", "conf": 1.0, "text": "hello"}], "translations": {}}
    out = mt.translate(doc, "English", offline=True)
    assert out == ["<en>privet", "apa khabar", "hello"], "TranslateGemma's lines are kept, NLLB's stay in the original"
    assert doc["translations"] == {} and "_incomplete" not in doc, "not kept as done: a rerun translates the rest"
    err = capsys.readouterr().err
    assert "NLLB could not translate the 1 line(s) TranslateGemma does not cover (OSError: offline)" in err
    live = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}, {"t": 1.0, "lang": "ms", "conf": 1.0, "text": "apa khabar"}]
    monkeypatch.setattr(mt, "_NLLB_FAILED", [])
    mt.translate_live(live, ["English"], offline=True)                       # live: the Gemma line keeps its translation
    assert live[0]["tr"] == {"English": "<en>privet"} and "tr" not in live[1]


def test_the_fallback_note_counts_what_was_done_before_not_all_of_it_as_translategemmas(monkeypatch, capsys):
    tok, calls = _fake_gemma(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "en", "conf": 1.0, "text": "hi"}, {"t": 1.0, "lang": "en", "conf": 1.0, "text": "yes"},
                     {"t": 2.0, "lang": "ms", "conf": 1.0, "text": "apa"}], "translations": {}}
    mt.translate(doc, "English", offline=True)
    err = capsys.readouterr().err
    assert "by TranslateGemma;" not in err and "2 done before (by TranslateGemma, live, or already in English)" in err


def test_gemma_keeps_a_line_in_the_target_but_turns_chinese_into_traditional_chinese(monkeypatch):
    tok, calls = _fake_gemma(monkeypatch)
    released = []
    monkeypatch.setattr(mt, "release_engine", lambda: released.append(1))
    doc = {"lines": [{"t": 0.0, "lang": "pt", "conf": 1.0, "text": "Olá"}, {"t": 1.0, "lang": "zh", "conf": 1.0, "text": "你好"},
                     {"t": 2.0, "lang": "ru", "conf": 1.0, "text": "Привет"}], "translations": {}}
    assert mt.translate(doc, "Brazilian Portuguese", offline=True) == ["Olá", "<pt-BR>你好", "<pt-BR>Привет"]
    assert mt.translate(doc, "Traditional Chinese", offline=True) == ["<zh-TW>Olá", "<zh-TW>你好", "<zh-TW>Привет"]
    assert len(released) == 2, "the speech model is released once per translation, not once per line"


# ---------- Hy-MT3: the third offline translator (--translator hy) ----------
# The real functions, before _no_model_downloads (below) replaces them in every test: the tests of fetching and checking use these.
_REAL = SimpleNamespace(hy_present=mt.hy_present, fetch_hy=mt.fetch_hy, fetch_model=mt.fetch_model, model_present=mt.model_present)


class FakeHyTokenizer:
    """Stands in for mlx-lm's TokenizerWrapper with Tencent's chat template: one user message whose content is the prompt
    itself (a string), its index is the prompt's ids; a text has one token per word."""
    def __init__(self):
        self.prompts, self.eos = [], []

    def add_eos_token(self, token):
        self.eos.append(token)

    def convert_tokens_to_ids(self, token):
        return 1

    def apply_chat_template(self, msgs, add_generation_prompt=True):
        assert add_generation_prompt and len(msgs) == 1 and msgs[0]["role"] == "user"
        assert isinstance(msgs[0]["content"], str), "the plain prompt, not TranslateGemma's structured content"
        self.prompts.append(msgs[0]["content"])
        return [len(self.prompts) - 1] * 10

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _hy_parts(prompt):
    """(target name, text) of a prompt made by hy_prompt."""
    m = re.fullmatch(r"Translate the following text into (.+?)\. Note that you should only output the translated result "
                     r"without any additional explanation:\n\n(.*)", prompt, re.S)
    return m.group(1), m.group(2)


def _fake_hy(monkeypatch, reply=lambda name, text: f"<{name}>{text}", finish="stop"):
    tok = FakeHyTokenizer()
    calls, processors = [], []

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):
        prompt_text = tokenizer.prompts[prompt[0]]
        name, text = _hy_parts(prompt_text)
        calls.append({"prompt": prompt_text, "name": name, "text": text, "max_tokens": max_tokens, "kw": kw})
        out = reply(name, text)
        yield SimpleNamespace(text=out[:3], finish_reason=None)             # the reply comes in parts, as mlx-lm streams it
        yield SimpleNamespace(text=out[3:], finish_reason=finish)

    def make_logits_processors(**kw):
        processors.append(kw)
        return ["the penalty"]
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=stream_generate))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_logits_processors=make_logits_processors))
    monkeypatch.setattr(mt, "_load_hy", lambda: ("model", tok))
    monkeypatch.setattr(mt, "release_engine", lambda: None)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    return tok, calls, processors


def test_hy_names_cover_89_whisper_languages_and_traditional_chinese():
    assert len(mt.HY_NAMES) == 90
    assert set(mt.HY_NAMES) - {"zh-TW"} <= mt.WHISPER_LANGS and len(set(mt.HY_NAMES) - {"zh-TW"}) == 89
    outside = "br bs fo ha haw la ln mi oc sn yo".split()
    assert sorted(mt.WHISPER_LANGS - set(mt.HY_NAMES)) == sorted(outside)
    assert [mt.hy_code(c) for c in outside] == [None] * 11
    assert set(mt.HY_VARIANTS.values()) <= set(mt.HY_NAMES) and set(mt.HY_DENSE_OK) - {"bs"} <= set(mt.HY_NAMES)
    assert mt.HY_NO_SPACES <= set(mt.HY_NAMES)


def test_hy_codes_and_the_names_its_prompt_takes():
    names = ("Russian", "ru", "Kazakh", "Mongolian", "Norwegian", "jw", "Tagalog", "Filipino", "Traditional Chinese",
             "zho_Hant", "Chinese", "zh-CN", "Brazilian Portuguese", "European Portuguese", "Cantonese", "rus_Cyrl")
    assert [mt.hy_code(n) for n in names] == "ru ru kk mn no jw tl tl zh-TW zh-TW zh zh pt pt yue ru".split()
    assert (mt.HY_NAMES["kk"], mt.HY_NAMES["mn"], mt.HY_NAMES["no"], mt.HY_NAMES["jw"], mt.HY_NAMES["zh-TW"]) == (
        "Cyrillic Kazakh", "Cyrillic Mongolian", "Norwegian", "Javanese", "Traditional Chinese")
    assert [mt.hy_code(n) for n in ("Zulu", "Yoruba", "Bosnian", "Klingon", "", None)] == [None] * 6
    assert mt.hy_pair("ru", "English") == ("ru", "en") and mt.hy_pair(None, "English", "de") == ("de", "en")
    assert mt.hy_pair("bs", "English") is None and mt.hy_pair("ru", "Yoruba") is None


def test_hy_sends_the_cards_plain_prompt(monkeypatch):
    tok, calls, _ = _fake_hy(monkeypatch)
    assert mt.hy_translate([("ru", "Привет")], "en") == (["<English>Привет"], 0)
    assert calls[0]["prompt"] == ("Translate the following text into English. Note that you should only output the "
                                  "translated result without any additional explanation:\n\nПривет")
    assert mt.hy_prompt("x", "kk").startswith("Translate the following text into Cyrillic Kazakh. ")


def test_hy_decodes_greedily_and_budgets_tokens_for_the_target_script(monkeypatch):
    tok, calls, processors = _fake_hy(monkeypatch)
    for tgt in ("en", "hi", "my"):
        mt.hy_translate([("ru", "раз два три четыре")], tgt)
    assert [c["max_tokens"] for c in calls] == [64 + 3 * 4, 64 + 9 * 4, 64 + 9 * 4]
    assert all(c["kw"] == {"logits_processors": ["the penalty"]} for c in calls), "no sampler: argmax"
    assert processors and all(p == {"repetition_penalty": 1.05, "repetition_context_size": 20} for p in processors)
    calls.clear()
    mt.hy_translate([("en", ("a " * 699).strip() + ".")], "hi")                   # 700 tokens in one piece: 64 + 9 * 700
    assert calls[0]["max_tokens"] == 4096


def test_a_long_text_goes_to_hy_in_pieces_and_is_joined_as_the_target_writes(monkeypatch):
    tok, calls, _ = _fake_hy(monkeypatch, reply=lambda name, text: f"[{len(text)}]")
    text = " ".join(f"Sentence number {k} is here." for k in range(200))           # about 5,600 characters
    out, cut = mt.hy_translate([("en", text)], "de")
    pieces = [c["text"] for c in calls]
    assert len(pieces) >= 4 and all(len(p) <= mt.HY_PIECE for p in pieces) and " ".join(pieces) == text and cut == 0
    assert out == [" ".join(f"[{len(p)}]" for p in pieces)]
    calls.clear()
    assert mt.hy_translate([("en", text)], "ja")[0] == ["".join(f"[{len(p)}]" for p in pieces)], "no spaces in Japanese"


def test_a_hy_piece_that_runs_to_the_limit_is_kept_and_counted(monkeypatch, capsys):
    tok, calls, _ = _fake_hy(monkeypatch, reply=lambda name, text: "loop loop loop", finish="length")
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "да да да"}], "translations": {}}
    assert mt.translate(doc, "English", offline=True) == ["loop loop loop"]
    assert "1 line(s) ran to the token limit and may be cut short or repeat" in capsys.readouterr().err
    live = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "да да да"}]
    mt.translate_live(live, ["English"], offline=True)
    assert live[0]["tr"] == {"English": "loop loop loop"}
    assert "1 line(s) ran to the token limit" in capsys.readouterr().err


def test_translating_a_meeting_with_hy_gives_other_languages_to_nllb(monkeypatch, capsys):
    tok, calls, _ = _fake_hy(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"},
                     {"t": 1.0, "lang": "en", "conf": 1.0, "text": "Hello"},
                     {"t": 2.0, "note": "a note"},
                     {"t": 3.0, "lang": "bs", "conf": 1.0, "text": "bosanske riječi"},
                     {"t": 4.0, "lang": "km", "conf": 1.0, "text": "ខ្មែរ"}], "translations": {}}
    out = mt.translate(doc, "English", offline=True)
    assert out == ["<English>Привет", "Hello", None, "Tbosanske Triječi", "<English>ខ្មែរ"], "Khmer is Hy-MT3's (not Gemma's)"
    assert [c["text"] for c in calls] == ["Привет", "ខ្មែរ"]
    assert {sentence[0] for tokens, _ in translator.calls for sentence in tokens} == {"bos_Latn"}, "only Bosnian to NLLB"
    assert list(doc["translations"]) == [f"en [{mt.HY}]"]
    err = capsys.readouterr().err
    assert "into English with Hy-MT3; 1 already in English; 1 in a language Hy-MT3 does not support go to NLLB" in err
    assert "3 done before (by Hy-MT3, live, or already in English)" in err


def test_the_note_when_no_line_is_in_hys_languages(monkeypatch, capsys):
    tok, calls, _ = _fake_hy(monkeypatch)
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (FakeTranslator(), FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "yo", "conf": 1.0, "text": "bawo"}], "translations": {}}
    assert mt.translate(doc, "English", offline=True) == ["Tbawo"] and calls == []
    err = capsys.readouterr().err
    assert "None of the 1 line(s) is in one of Hy-MT3's languages: NLLB translates them into English" in err
    assert "Translated 0 line(s)" not in err


def test_hy_with_a_target_it_lacks_and_with_an_nllb_that_cannot_load(monkeypatch, capsys):
    tok, calls, _ = _fake_hy(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}], "translations": {}}
    assert mt.translate(doc, "Yoruba", offline=True) == ["Tprivet"] and calls == []
    assert "Yoruba is not among Hy-MT3's languages: NLLB translates it" in capsys.readouterr().err

    def no_nllb(**kw):
        raise OSError("offline")
    monkeypatch.setattr(mt, "get_nllb", no_nllb)
    doc = {"lines": [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}, {"t": 1.0, "lang": "yo", "conf": 1.0, "text": "bawo"},
                     {"t": 2.0, "lang": "en", "conf": 1.0, "text": "hello"}], "translations": {}}
    out = mt.translate(doc, "English", offline=True)
    assert out == ["<English>privet", "bawo", "hello"], "Hy-MT3's lines are kept, NLLB's stay in the original"
    assert doc["translations"] == {} and "_incomplete" not in doc, "not kept as done: a rerun translates the rest"
    assert "NLLB could not translate the 1 line(s) Hy-MT3 does not cover (OSError: offline)" in capsys.readouterr().err
    live = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}, {"t": 1.0, "lang": "yo", "conf": 1.0, "text": "bawo"}]
    mt.translate_live(live, ["English"], offline=True)
    assert live[0]["tr"] == {"English": "<English>privet"} and "tr" not in live[1]
    assert "the lines Hy-MT3 does not cover are translated at the end" in capsys.readouterr().err


def test_a_hy_translation_cut_short_goes_on_from_its_log(monkeypatch, tmp_path):
    seen = []

    def reply(name, text):
        if len(seen) == 2:
            raise KeyboardInterrupt                                          # the run is stopped after two lines
        seen.append(text)
        return "EN " + text
    tok, calls, _ = _fake_hy(monkeypatch, reply=reply)
    lines = [{"t": float(k), "lang": "ru", "conf": 1.0, "text": f"строка {k}"} for k in range(4)]
    with pytest.raises(KeyboardInterrupt):
        mt.translate({"lines": lines, "translations": {}}, "English", offline=True, checkpoint=tmp_path)
    assert [p.name.startswith(f"en-{mt.HY.replace('/', '_')}-") for p in (tmp_path / "translations").iterdir()] == [True]
    tok, calls, _ = _fake_hy(monkeypatch, reply=lambda name, text: "EN " + text)
    out = mt.translate({"lines": lines, "translations": {}}, "English", offline=True, checkpoint=tmp_path)
    assert out == [f"EN строка {k}" for k in range(4)] and [c["text"] for c in calls] == ["строка 2", "строка 3"]


def test_live_hy_translations_are_marked_and_reused_only_by_hy(monkeypatch):
    tok, calls, _ = _fake_hy(monkeypatch)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    lines = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "Привет"}, {"t": 1.0, "lang": "bs", "conf": 1.0, "text": "zdravo"}]
    mt.translate_live(lines, ["English"], offline=True)
    assert lines[0]["tr"] == {"English": "<English>Привет"} and lines[0]["tr_by"] == {"English": mt.HY}
    assert lines[1]["tr"] == {"English": "Tzdravo"} and lines[1]["tr_by"] == {"English": mt.NLLB}
    assert mt.translated_live(lines, "English", True) == {0: "<English>Привет", 1: "Tzdravo"}
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    assert mt.translated_live(lines, "English", True) == {1: "Tzdravo"}, "with TranslateGemma at the end it is made again"
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    assert mt.translated_live(lines, "English", True) == {1: "Tzdravo"}


def test_the_live_worker_loads_hy_beside_the_speech_model_and_goes_on_without_nllb(monkeypatch, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    loaded = []
    monkeypatch.setattr(mt, "get_hy", lambda keep_engine=False: loaded.append(("hy", keep_engine)))
    monkeypatch.setattr(mt, "get_nllb", lambda keep_engine=False: loaded.append(("nllb", keep_engine)))
    mt.LiveWorker([], targets=["English"], offline=True).load_models()
    mt.LiveWorker([], targets=["Yoruba"], offline=True).load_models()
    assert loaded == [("hy", True), ("nllb", True)]
    fetched = []

    def fetch(repo, size, stage=True):
        fetched.append((repo, size, stage))
        if repo == mt.NLLB:
            raise OSError("offline")
    monkeypatch.setattr(mt, "fetch_model", fetch)
    worker = mt.LiveWorker([], targets=["English"], offline=True)
    worker.fetch_models()
    assert fetched == [(mt.HY, "about 4.2 GB", False), (mt.NLLB, "about 1.4 GB", False)], "never a stage while recording"
    assert not worker.translation_off and worker.models_here.is_set()
    assert "NLLB could not be fetched (OSError); the lines Hy-MT3 does not cover are translated at the end" in capsys.readouterr().err


def test_a_wrong_hy_file_is_not_downloaded_again_on_the_live_workers_thread(monkeypatch):
    """hy_present looks at presence only, so no fetch thread runs; a checksum mismatch found when the live worker loads the
    model must not download 4.2 GB there (no chunk would be decoded meanwhile): live translation is off, with the reason."""
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setattr(mt, "HY", mt.HY_DEFAULT)
    calls = []

    def fetch_hy(stage=True, refetch=True):
        calls.append((stage, refetch))
        if not refetch:
            raise RuntimeError("The Hy-MT3 file model.safetensors from x does not match its checksum; it is downloaded again "
                               "when the recording is translated at the end")
        return {}
    monkeypatch.setattr(mt, "fetch_hy", fetch_hy)
    monkeypatch.setattr(mt, "get_hy", lambda keep_engine=False: pytest.fail("not loaded after a failed check"))
    notes = []
    monkeypatch.setattr(mt, "note", notes.append)
    worker = mt.LiveWorker([], targets=["English"], offline=True)
    worker.translator.prepare()                                             # 1.13.0: the check runs on the prepare thread
    worker.translate([{"text": "привет", "lang": "ru"}])
    assert calls == [(False, False)] and worker.translation_off
    assert notes and "could not be loaded" in notes[0] and "downloaded again when the recording is translated" in notes[0]
    monkeypatch.setattr(mt, "HY", "someone/Hy-MT3-7B-8bit")                    # another copy: no pinned files to check
    loaded = []
    monkeypatch.setattr(mt, "get_hy", lambda keep_engine=False: loaded.append(keep_engine))
    mt.LiveWorker([], targets=["English"], offline=True).load_models()
    assert loaded == [True] and calls == [(False, False)]


def test_preflight_shows_the_advice_of_a_failed_check(monkeypatch, capsys):
    def fetch_model(repo, size, stage=True):
        raise RuntimeError("The Hy-MT3 file f from a/b does not match its checksum (...). Delete the folder ~/x")
    monkeypatch.setattr(mt, "fetch_model", fetch_model)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    mt.preflight(0, 1, ["English"])
    assert "could not be fetched now (RuntimeError: The Hy-MT3 file f from a/b does not match its checksum (...). Delete " \
           "the folder ~/x)" in capsys.readouterr().err
    monkeypatch.setattr(mt, "fetch_model", lambda repo, size, stage=True: (_ for _ in ()).throw(OSError("secret path")))
    mt.preflight(0, 1, ["English"])
    assert "could not be fetched now (OSError);" in capsys.readouterr().err, "other errors: the type only, as before"


def test_translator_option_accepts_hy(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(mt, "OUT", None)
    seen = []
    monkeypatch.setattr(mt, "translate", lambda doc, target, offline=False, source=None, checkpoint=None:
                        seen.append((mt.TRANSLATOR, offline)) or [l.get("text") for l in doc["lines"]])
    (tmp_path / "a.txt").write_text("[00:00:01] [ru] привет\n")
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", str(tmp_path / "a.txt"), "--to", "English", "--offline", "--translator", "hy"])
    mt.main()
    assert seen == [("hy", True)], "--translator sets the run's translator"
    monkeypatch.setattr(sys, "argv", ["mt.py", "translate", "--help"])
    with pytest.raises(SystemExit):
        mt.main()
    assert "{nllb,gemma,hy}" in capsys.readouterr().out


def test_without_mlx_lm_the_hy_choice_falls_back_to_nllb_with_a_note(monkeypatch, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setattr(mt, "mlx_lm_available", lambda: False)                # an Intel Mac, Linux, Windows, or no mlx-lm
    mt._need_key(["English"], offline=True)
    err = capsys.readouterr().err
    assert mt.TRANSLATOR == "nllb" and "Hy-MT3 is not available here (it needs Apple silicon and the mlx-lm package" in err
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mt._need_key(["English"], offline=False)                                   # with Claude: nothing changes
    mt._need_key([], offline=True)
    assert mt.TRANSLATOR == "hy"


def test_mlx_lm_is_looked_for_only_on_apple_silicon(monkeypatch):
    import importlib.util
    import platform
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "mlx_lm" else None)
    for system, machine, expected in (("darwin", "arm64", True), ("darwin", "x86_64", False), ("linux", "arm64", False),
                                      ("win32", "AMD64", False)):
        monkeypatch.setattr(sys, "platform", system)
        monkeypatch.setattr(platform, "machine", lambda m=machine: m)
        assert mt.mlx_lm_available() is expected, (system, machine)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert mt.mlx_lm_available() is False, "Apple silicon without mlx-lm"


def test_which_models_a_translation_needs_with_hy(monkeypatch):
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    assert mt.translation_models(["English"], True) == [(mt.HY, "about 4.2 GB"), (mt.NLLB, "about 1.4 GB")]
    assert mt.translation_models(["Yoruba"], True) == [(mt.NLLB, "about 1.4 GB")], "Hy-MT3 only for its languages"
    assert mt.line_translator("ru", "English", True) == mt.HY and mt.line_translator("bs", "English", True) == mt.NLLB
    assert mt.line_translator("ru", "English", False) == "claude"


def test_the_default_hy_files_are_pinned_and_checked(monkeypatch, tmp_path, capsys):
    assert {repo for repo, *_ in mt.HY_FILES} == {mt.HY_DEFAULT, "tencent/Hy-MT3-7B"}
    assert all(re.fullmatch(r"[0-9a-f]{40}", commit) and re.fullmatch(r"[0-9a-f]{64}", sha) for _, commit, _, _, sha in mt.HY_FILES)
    names = [name for _, _, name, _, _ in mt.HY_FILES]
    assert len(set(names)) == len(names) == 8 and not any(n.startswith("model-0000") for n in names), "no bf16 shard"
    assert {name for repo, _, name, _, _ in mt.HY_FILES if repo == mt.HY_DEFAULT} == {"config.json", "model.safetensors",
                                                                                     "LICENSE.txt", "NOTICE"}
    assert sum(size for _, _, _, size, _ in mt.HY_FILES) == 4238245157
    import hashlib
    monkeypatch.setattr(mt, "MODELS_DIR", tmp_path / "models")
    good = tmp_path / "good"
    good.write_bytes(b"hello")
    sha = hashlib.sha256(b"hello").hexdigest()
    downloads = []
    fixed = tmp_path / "fixed"

    def hf_hub_download(repo, name, revision=None, force_download=False):
        downloads.append((repo, name, revision, force_download))
        fixed.write_bytes(b"hello" if "fixes" in name else b"HELLO")
        return str(fixed)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=hf_hub_download))
    assert mt._check_hy_file("a/b", "c" * 40, "f", 5, sha, str(good)) == str(good) and downloads == []
    bad = tmp_path / "bad"
    bad.write_bytes(b"HELLO")
    assert mt._check_hy_file("a/b", "c" * 40, "fixes", 5, sha, str(bad)) == str(fixed), "downloaded again, then right"
    assert downloads == [("a/b", "fixes", "c" * 40, True)]
    with pytest.raises(RuntimeError, match=r"The Hy-MT3 file f from tencent/Hy-MT3-7B does not match its checksum \(expected "
                                           + sha + r", got [0-9a-f]{64}\)\. Delete the folder "
                                           r"~/\.cache/huggingface/hub/models--tencent--Hy-MT3-7B and translate again"):
        mt._check_hy_file("tencent/Hy-MT3-7B", "c" * 40, "f", 5, sha, str(bad))
    assert len(downloads) == 2, "downloaded again only once"
    with pytest.raises(RuntimeError, match="Delete the folder"):
        mt._check_hy_file("tencent/Hy-MT3-7B", "c" * 40, "f", 5, sha, str(bad))
    assert len(downloads) == 2, "a file that was wrong after its new download is not fetched a third time in the same run"
    with pytest.raises(RuntimeError, match=r"^The Hy-MT3 file g from a/b does not match its checksum; it is downloaded again "
                                           r"when the recording is translated at the end$"):
        mt._check_hy_file("a/b", "c" * 40, "g", 5, sha, str(bad), refetch=False)
    assert len(downloads) == 2, "on the live worker's thread nothing is downloaded"
    with pytest.raises(RuntimeError, match="got a file of 5 bytes instead of 6"):
        mt._check_hy_file("a/b", "c" * 40, "f", 6, sha, str(good))
    # The big file: hashed once, then known by its path, size and modification time.
    big = tmp_path / "blob"
    with open(big, "wb") as f:
        f.truncate(2**26 + 1)                                                 # sparse: 64 MB of zeros, written at once
    big_sha = hashlib.sha256(b"\0" * (2**26 + 1)).hexdigest()
    assert mt._check_hy_file("a/b", "c" * 40, "model.safetensors", 2**26 + 1, big_sha, str(big)) == str(big)
    assert "Checking the downloaded translation model (once)" in capsys.readouterr().err
    record = json.loads((tmp_path / "models" / mt.HY_CHECKED).read_text())
    assert list(record) == [os.path.realpath(big)] and record[os.path.realpath(big)][2] == big_sha

    def no_hash():
        raise AssertionError("hashed again")
    monkeypatch.setattr(mt, "hashlib", SimpleNamespace(sha256=no_hash))
    assert mt._check_hy_file("a/b", "c" * 40, "model.safetensors", 2**26 + 1, big_sha, str(big)) == str(big)
    assert "Checking" not in capsys.readouterr().err
    os.utime(big, ns=(0, 10**9))                                                # the file changed: hashed again
    with pytest.raises(AssertionError, match="hashed again"):
        mt._check_hy_file("a/b", "c" * 40, "model.safetensors", 2**26 + 1, big_sha, str(big))


def test_hy_present_needs_every_file(monkeypatch):
    monkeypatch.setattr(mt, "hy_present", _REAL.hy_present)
    monkeypatch.setattr(mt, "model_present", _REAL.model_present)
    missing = []
    asked = []

    def try_to_load_from_cache(repo, name, revision=None):
        asked.append((repo, name, revision))
        return None if name in missing else object() if name == "NOTICE" and "odd" in missing else f"/cache/{name}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=try_to_load_from_cache))
    assert mt.hy_present() and mt.model_present(mt.HY_DEFAULT)
    assert sorted(asked[:8]) == sorted((repo, name, commit) for repo, commit, name, _, _ in mt.HY_FILES), "each at its commit"
    missing.append("tokenizer_config.json")
    assert not mt.hy_present() and not mt.model_present(mt.HY_DEFAULT)
    missing[:] = ["odd"]                                                        # _CACHED_NO_EXIST: known to be missing
    assert not mt.hy_present()
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)                  # not installed
    assert mt.hy_present() is False


def test_hy_is_fetched_file_by_file_at_its_commits_before_a_long_job(monkeypatch, capsys):
    monkeypatch.setattr(mt, "fetch_hy", _REAL.fetch_hy)
    monkeypatch.setattr(mt, "fetch_model", _REAL.fetch_model)
    here = {name for _, _, name, _, _ in mt.HY_FILES} - {"model.safetensors", "chat_template.jinja"}
    downloads, checked = [], []

    def hf_hub_download(repo, name, revision=None, force_download=False):
        downloads.append((repo, name, revision))
        here.add(name)
        return f"/cache/{repo}/{revision}/{name}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=hf_hub_download,
        try_to_load_from_cache=lambda repo, name, revision=None: f"/cache/{repo}/{revision}/{name}" if name in here else None))
    monkeypatch.setattr(mt, "_check_hy_file", lambda repo, commit, name, size, sha, path, stage=True, refetch=True:
                        checked.append((name, stage)) or path)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    mt.preflight(0, 1, ["English"])                                            # NLLB: present (the fixture's model_present)
    assert downloads == [(mt.HY_DEFAULT, "model.safetensors", "dca6fb32e6d61d2657ed714b572307216df2cb5c"),
                         ("tencent/Hy-MT3-7B", "chat_template.jinja", "38d6c3c31d6b4dbdfc208e0621b822ae7bf6aeb2")]
    assert sorted(checked) == sorted((name, True) for _, _, name, _, _ in mt.HY_FILES), "every file checked, not only the new"
    assert "Downloading the translation model Hy-MT3 7B (first time only, about 4.2 GB)" in capsys.readouterr().err
    downloads.clear()
    checked.clear()
    paths = mt.fetch_hy(stage=False)
    assert downloads == [] and len(checked) == 8 and "Downloading" not in capsys.readouterr().err
    assert paths["model.safetensors"] == f"/cache/{mt.HY_DEFAULT}/dca6fb32e6d61d2657ed714b572307216df2cb5c/model.safetensors"
    here.discard("NOTICE")
    monkeypatch.setattr(mt, "EVENTS", True)
    mt.fetch_hy(stage=False)                                                   # during a recording: a plain line, no stage
    out, err = capsys.readouterr()
    assert "@@STAGE" not in out and "Downloading the translation model Hy-MT3" in err
    monkeypatch.setattr(mt, "HY", "someone/Hy-MT3-7B-8bit")                    # another copy: the usual whole-repo fetch
    pulled = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda repo, **k: pulled.append(repo)))
    monkeypatch.setattr(mt, "model_present", lambda repo: False)
    mt.fetch_model(mt.HY, "about 4.2 GB", stage=False)
    assert pulled == ["someone/Hy-MT3-7B-8bit"]


def test_the_pinned_hy_model_is_loaded_from_its_two_snapshots_with_its_end_tokens(monkeypatch):
    tok = FakeHyTokenizer()
    seen = {}

    def load_model(path):
        seen["model"] = path
        return "model", {"eos_token_id": [127960, 127967]}

    def load_tokenizer(path, eos_token_ids=None):
        seen["tokenizer"] = (path, eos_token_ids)
        return tok
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(load_model=load_model, load_tokenizer=load_tokenizer))
    monkeypatch.setattr(mt, "fetch_hy", lambda stage=True: {"model.safetensors": "/c/w/snap/model.safetensors",
                                                            "tokenizer.json": "/c/t/snap/tokenizer.json"})
    monkeypatch.setattr(mt, "HY", mt.HY_DEFAULT)
    mt._load_hy.cache_clear()
    try:
        assert mt._load_hy() == ("model", tok)
        assert seen == {"model": Path("/c/w/snap"), "tokenizer": (Path("/c/t/snap"), [127960, 127967])}
        assert tok.eos == ["<|eos|>", "<|extra_5|>"]
        mt._load_hy.cache_clear()
        tok.eos.clear()
        tok.unk_token_id = 1                                                   # <|extra_5|> not in this vocabulary
        mt._load_hy()
        assert tok.eos == ["<|eos|>"]
    finally:
        mt._load_hy.cache_clear()


def test_a_hy_model_from_a_folder_or_another_repo_is_loaded_as_it_is(monkeypatch, tmp_path):
    tok = FakeHyTokenizer()
    loaded = []
    failure = []

    def load(path):
        loaded.append(path)
        if failure:
            raise failure[0]
        return "model", tok

    def no_download(*a, **k):
        raise AssertionError("nothing is downloaded for a folder")
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=load))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=no_download, snapshot_download=no_download))
    monkeypatch.setattr(mt, "fetch_hy", no_download)
    folder = tmp_path / "hy-mt3-7b-4bit"
    folder.mkdir()
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    mt._load_hy.cache_clear()
    try:
        for given, expected in ((str(folder), str(folder)), ("someone/Hy-MT3-7B-8bit", "someone/Hy-MT3-7B-8bit")):
            monkeypatch.setattr(mt, "HY", given)
            mt._load_hy.cache_clear()
            assert mt._load_hy() == ("model", tok) and loaded[-1] == expected and tok.eos[-2:] == ["<|eos|>", "<|extra_5|>"]
        monkeypatch.setattr(mt, "HY", str(folder))
        name = mt.llm_model_name()
        assert re.fullmatch(r"hy-mt3-7b-4bit-[0-9a-f]{8}", name), "a folder's log is named after the folder, not its whole path"
        other = tmp_path / "b" / "hy-mt3-7b-4bit"
        other.mkdir(parents=True)
        monkeypatch.setattr(mt, "HY", str(other))
        assert mt.llm_model_name() != name, "two folders of the same name keep their logs apart"
        doc = {"lines": [{"lang": "ru", "text": "Привет"}]}
        logs = {mt._llm_log(tmp_path / "data", "en", doc["lines"])}
        monkeypatch.setattr(mt, "HY", str(folder))
        logs.add(mt._llm_log(tmp_path / "data", "en", doc["lines"]))
        assert len(logs) == 2
        assert mt.translation_models(["English"], True)[0] == (str(folder), "about 4.2 GB")
        failure.append(ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported."))
        mt._load_hy.cache_clear()
        with pytest.raises(RuntimeError, match=r"needs transformers 5 \('TokenizersBackend'\); put Tencent's "
                                               r"tokenizer_config\.json \(https://huggingface\.co/tencent/Hy-MT3-7B/blob/main/"
                                               r"tokenizer_config\.json\) in its place, or unset MT_HY_MODEL"):
            mt._load_hy()
        failure[:] = [ValueError("something else")]
        mt._load_hy.cache_clear()
        with pytest.raises(ValueError, match="something else"):
            mt._load_hy()
    finally:
        mt._load_hy.cache_clear()


def test_a_model_in_a_local_folder_is_present_and_never_fetched(monkeypatch, tmp_path, capsys):
    """MT_HY_MODEL (or MT_GEMMA_MODEL) naming a folder: huggingface_hub takes no path as a repo id (HFValidationError), so
    the folder must be taken as present before the cache is asked, or live translation is off and preflight says it
    downloads 4.2 GB."""
    monkeypatch.setattr(mt, "model_present", _REAL.model_present)
    monkeypatch.setattr(mt, "fetch_model", _REAL.fetch_model)

    def snapshot_download(repo, **k):
        if repo == mt.NLLB:
            return "/cache/nllb"
        raise ValueError(f"Repo id must be in the form 'repo_name' or 'namespace/repo_name': '{repo}'")   # as HFValidationError
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(mt, "_nllb_weights_here", lambda path: True)
    folder = tmp_path / "hy-mt3-7b-4bit"
    folder.mkdir()
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setattr(mt, "HY", str(folder))
    assert mt.model_present(mt.HY)
    mt.fetch_model(mt.HY, mt.llm_size(), stage=False)
    mt.preflight(0, 1, ["English"])
    assert "Fetching" not in capsys.readouterr().err and "could not be fetched" not in capsys.readouterr().err
    worker = mt.LiveWorker([], targets=["English"], offline=True)
    worker.fetch_models()
    assert not worker.translation_off and worker.models_here.is_set()
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    monkeypatch.setattr(mt, "GEMMA", str(folder))
    assert mt.model_present(mt.GEMMA), "TranslateGemma from a folder too"
    monkeypatch.setattr(mt, "HY", "~/no-such-folder-for-hy")                  # not a folder: asked as a repo id, as before
    assert not mt.model_present(mt.HY)


def test_an_nllb_download_cut_short_is_not_taken_as_present(monkeypatch, tmp_path, capsys):
    """An HTTP 429 or a lost network leaves NLLB's snapshot folder with the small files and no model.bin; it must be
    fetched again, not loaded ("Unable to open file 'model.bin'" at every run)."""
    monkeypatch.setattr(mt, "model_present", _REAL.model_present)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").write_text("{}")
    pulled = []

    def snapshot_download(repo, local_files_only=False):
        if not local_files_only:
            pulled.append(repo)
            (snap / "model.bin").write_bytes(b"w")
        return str(snap)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: "tok")))
    monkeypatch.setattr(mt, "import_ctranslate2", lambda: SimpleNamespace(Translator=lambda path, **k: ("translator", path)))
    monkeypatch.setattr(mt, "cuda_wanted", lambda ct2: False)
    assert not mt.model_present(mt.NLLB)
    mt._load_nllb.cache_clear()
    try:
        assert mt._load_nllb() == (("translator", str(snap)), "tok") and pulled == [mt.NLLB]
        assert f"Downloading the translation model {mt.NLLB}" in capsys.readouterr().err
        assert mt.model_present(mt.NLLB)
        mt._load_nllb.cache_clear()
        mt._load_nllb()
        assert pulled == [mt.NLLB], "complete: loaded from the cache"
    finally:
        mt._load_nllb.cache_clear()


def test_the_memory_note_for_hy(monkeypatch, capsys):
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setattr(mt, "total_memory", lambda: 8 * 2**30)
    mt.preflight(0, 1, ["English"], fetch=False)
    err = capsys.readouterr().err
    assert "This computer has 8 GB of memory: Hy-MT3 needs about 4.5 GB (about 6 GB beside the speech model" in err
    assert "TranslateGemma" not in err
    mt.preflight(0, 1, [], fetch=False)
    assert "Hy-MT3" not in capsys.readouterr().err, "nothing translated offline: no note"
    monkeypatch.setattr(mt, "total_memory", lambda: 16 * 2**30)
    mt.preflight(0, 1, ["English"], fetch=False)
    assert "Hy-MT3" not in capsys.readouterr().err
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    monkeypatch.setattr(mt, "total_memory", lambda: 8 * 2**30)
    mt.preflight(0, 1, ["English"], fetch=False)
    assert "TranslateGemma needs about 7 GB" in capsys.readouterr().err, "TranslateGemma's note as before (below 16 GB)"


def test_the_windows_path_check_counts_the_hy_log(monkeypatch):
    monkeypatch.setattr(mt, "windows_long_paths", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(mt, "NLLB", "a/b")                                     # a short NLLB: the Hy-MT3 log is the longest
    folder = Path(os.path.abspath(os.sep + "d" * 200))
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    problem = mt.windows_path_problem(folder, ["English"], True)
    assert problem and mt.translation_log_name("zho_Hant", mt.HY, "0" * 16) in problem
    monkeypatch.setattr(mt, "TRANSLATOR", "nllb")
    assert mt.windows_path_problem(folder, ["English"], True) is None


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch, request, tmp_path_factory):
    """No test downloads a translation model: the cache is taken as holding them (Hy-MT3's pinned files too), and a fetch
    does nothing (the tests of fetching and checking Hy-MT3 put the real functions back, from _REAL). The stream
    switch of mlx-lm needs the real MLX (it was run against MLX 0.29.3, 0.31.2, 0.32.0 and 0.32.4); it is left out except
    in its own test, which gives it a stand-in MLX."""
    monkeypatch.setattr(mt, "model_present", lambda repo: True)
    monkeypatch.setattr(mt, "fetch_model", lambda repo, size, stage=True: None)
    monkeypatch.setattr(mt, "hy_present", lambda: True)
    monkeypatch.setattr(mt, "fetch_hy", lambda stage=True, refetch=True: {})
    monkeypatch.setattr(mt, "_HY_REFETCHED", set())
    monkeypatch.setattr(mt, "_NLLB_FAILED", [])
    if "gets_a_stream_of_its_own" not in request.node.name:
        monkeypatch.setattr(mt, "_thread_stream", lambda: None)
    # 1.13.0: the lock files of one test are its own (the LLM slot, the live marker), a live model handed over or files
    # checked in one test are not seen by the next, and the machine counts as one with memory enough for a 7B model live
    # (a 16 GB runner reports a little less than 16 GiB, below TranslateGemma's threshold).
    monkeypatch.setattr(mt, "LOCK_DIR", str(tmp_path_factory.mktemp("locks")))
    slot, want = mt.FileLock("llm.lock"), mt.FileLock("llm-want.lock")
    monkeypatch.setattr(mt, "_SLOT", slot)
    monkeypatch.setattr(mt, "_WANT", want)
    monkeypatch.setattr(mt, "_LIVE_MODEL", {})
    monkeypatch.setattr(mt, "_HY_FILES_CHECKED", {})
    if "memory" not in request.node.name:
        monkeypatch.setattr(mt, "total_memory", lambda: 64 * 2**30)
    yield
    slot.free()
    want.free()


def test_each_thread_that_generates_gets_a_stream_of_its_own_in_the_mlx_lm_module(monkeypatch):
    """mlx_lm/__init__.py imports the function generate over the module mlx_lm.generate; the stream must be set on
    the module, and made anew only when another thread generates."""
    import threading
    made = []
    fake_mx = SimpleNamespace(new_stream=lambda device: made.append(threading.get_ident()) or f"stream-{len(made)}",
                              default_device=lambda: "gpu")
    module = SimpleNamespace(generation_stream="made at import")
    package = SimpleNamespace(generate=lambda *a, **k: "the function", __path__=[])
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx_lm", package)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", module)
    monkeypatch.setattr(mt, "_STREAM_THREAD", [])
    real = mt._thread_stream
    assert real.__name__ == "_thread_stream", "the real function, not the stand-in"
    real()
    real()                                                                    # the same thread: not made again
    assert module.generation_stream == "stream-1" and len(made) == 1
    t = threading.Thread(target=real)
    t.start(); t.join()
    assert module.generation_stream == "stream-2" and len(made) == 2 and made[1] != made[0]
    real()                                                                    # back on this thread: its own again
    assert module.generation_stream == "stream-3"


# ---------- 1.13.0: the live transcript never waits for a translation ----------
class _Events:
    """Stands in for mt.ev: every event with the monotonic time it was sent at (thread-safe)."""
    def __init__(self):
        self.items, self.lock = [], threading.Lock()

    def __call__(self, kind, *fields):
        with self.lock:
            self.items.append((time.monotonic(), kind, fields))

    def of(self, kind):
        with self.lock:
            return [(at, fields) for at, k, fields in self.items if k == kind]

    def lines(self):
        return [(at, json.loads(f[0])) for at, f in self.of("LINE")]

    def stats(self, kind=None):
        return [(at, f[0], json.loads(f[1])) for at, f in self.of("STAT") if kind is None or f[0] == kind]


class TwoLineEngine:
    """Decodes a chunk into two lines named after the second it starts at (speech samples carry their second + 1)."""
    name = "fake"

    def __init__(self, cost=0.0):
        self.calls, self.cost = [], cost

    def analyse(self, chunk, detect=True):
        return SimpleNamespace(chunk=chunk, probs=[("ru", 0.99)] if detect else None)

    def decode(self, state, language):
        import numpy as np
        if self.cost:
            time.sleep(self.cost)
        first = int(np.asarray(state.chunk)[np.flatnonzero(state.chunk)[0]] * 32768) - 1
        self.calls.append(first)
        return [{"start": 0.0, "end": 1.0, "text": f"c{first} one", "logprob": -0.1},
                {"start": 1.0, "end": 2.0, "text": f"c{first} two", "logprob": -0.3}]


def _live_hy(monkeypatch, tmp_path, token_s=0.0, tokens=1, load_s=0.0, fetch_s=0.0, slices=1, slice_s=0.0):
    """A stand-in Hy-MT3 for the live worker, at mlx-lm's narrowest points: stream_generate yields `tokens` tokens of
    `token_s` each; the pinned files' check (fetch_hy) takes fetch_s; the model's load takes load_s at once (the 1.12.0
    path, _load_hy) or `slices` slices of slice_s (the 1.13.0 path: load_model(lazy=True), then eval_in_slices).
    Returns (abort, seen): set abort to end every generation at once; seen records where things ran."""
    tok = FakeHyTokenizer()
    abort = threading.Event()
    seen = {"fetch_threads": [], "generate_threads": [], "load_lazy": [], "requests": 0}
    files = {"model.safetensors": str(tmp_path / "hy" / "model.safetensors"), "tokenizer.json": str(tmp_path / "hy" / "tokenizer.json")}
    (tmp_path / "hy").mkdir(exist_ok=True)
    (tmp_path / "hy" / "model.safetensors").write_bytes(b"\0" * 4096)
    (tmp_path / "hy" / "config.json").write_text(json.dumps({"eos_token_id": [127960]}))

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):
        seen["requests"] += 1
        seen["generate_threads"].append(threading.get_ident())
        for k in range(tokens):
            if abort.is_set():
                yield SimpleNamespace(text="", finish_reason="stop")
                return
            if token_s:
                time.sleep(token_s)
            yield SimpleNamespace(text="<tr>" if k == 0 else "x", finish_reason=None if k < tokens - 1 else "stop")

    def load_model(path, lazy=False):
        seen["load_lazy"].append(lazy)
        if not lazy and load_s:
            time.sleep(load_s)
        return "model", {"eos_token_id": [127960]}

    def load_tokenizer(path, tokenizer_config_extra=None, eos_token_ids=None):
        return tok

    def fetch_hy(stage=True, refetch=True):
        seen["fetch_threads"].append(threading.get_ident())
        if fetch_s:
            time.sleep(fetch_s)
        return files

    def old_load_hy():                                                       # 1.12.0: the whole load at once
        if load_s:
            time.sleep(load_s)
        return "model", tok

    def eval_in_slices(model, slice_bytes=None):
        for k in range(slices):
            if slice_s:
                time.sleep(slice_s)
            yield (k + 1) * 100, slices * 100
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=stream_generate))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_logits_processors=lambda **kw: []))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", SimpleNamespace(load_model=load_model, load_tokenizer=load_tokenizer))
    monkeypatch.setattr(mt, "_load_hy", old_load_hy)
    monkeypatch.setattr(mt, "fetch_hy", fetch_hy)
    monkeypatch.setattr(mt, "eval_in_slices", eval_in_slices, raising=False)
    monkeypatch.setattr(mt, "release_engine", lambda: None)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    monkeypatch.setattr(mt, "HY", mt.HY_DEFAULT)
    return abort, seen


def _chunk_of_speech(second, speech=6, pause=2):
    """int16 audio of `speech` seconds of speech (each second carries second + 1) and then `pause` seconds of silence."""
    import numpy as np
    audio = np.concatenate([np.full(16000, s + 1, dtype="int16") for s in range(second, second + speech)]
                           + [np.zeros(16000 * pause, dtype="int16")])
    return audio.reshape(-1, 1)


def _live_worker(monkeypatch, tmp_path, engine, targets):
    _content_vad(monkeypatch)
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    monkeypatch.setattr(mt, "EVENTS", True)
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["ru"], "normal", None, mt.engine_name())
    worker = mt.LiveWorker([track], targets=targets, offline=True)
    return worker, track, events


def _drain_translator(worker, timeout=10):
    """Steps the worker (on this thread, as its own thread would) until its translator has nothing left to do."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.step():
            continue
        if not worker.translator.has_work():
            return
        time.sleep(0.001)                                                     # a request on the helper thread
    raise AssertionError("the translator did not finish")


def _wait_for(condition, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_live_transcript_is_not_held_up_by_a_slow_translation(monkeypatch, tmp_path):
    """T1. 1.12.0 translated every line of a chunk into every target on the decode thread before it looked at the next
    chunk (8 s a line and target here: the second chunk came 16-32 s late). Now the second chunk's lines come within
    CUT_EVERY_S plus one token of its pause being fed, whatever the translation costs."""
    abort, seen = _live_hy(monkeypatch, tmp_path, token_s=0.02, tokens=400)
    worker, track, events = _live_worker(monkeypatch, tmp_path, TwoLineEngine(), ["English", "German"])
    worker.start()
    try:
        track.feed(_chunk_of_speech(0))                                       # speech 0-6 s, then 2 s of pause
        assert _wait_for(lambda: any(l.get("text") == "c0 one" for _, l in events.lines()), 5), "the first chunk is shown"
        assert _wait_for(lambda: seen["requests"] > 0, 10), "and its translation has started"
        track.feed(_chunk_of_speech(8))                                       # the second chunk and its pause
        fed_at = time.monotonic()
        assert _wait_for(lambda: any(l.get("text") == "c8 one" for _, l in events.lines()), 20), "the second chunk is shown"
        shown_at = next(at for at, l in events.lines() if l.get("text") == "c8 one")
        assert shown_at - fed_at <= 1.5, f"the second chunk waited {shown_at - fed_at:.1f} s for the translation"
    finally:
        abort.set()
        worker.stop(wait=30)


def test_the_first_model_load_does_not_stop_the_transcript(monkeypatch, tmp_path):
    """T2. 1.12.0 checked the pinned files and loaded Hy-MT3 on the decode thread at the first chunk with text: the
    transcript stopped for the whole load. Now the check runs on the prepare thread and the weights are evaluated in
    slices between chunks: the lines keep coming at the pace they come without translation, and @@STAT model goes
    checking -> loading (progress rising) -> ready with its load time."""
    def run(targets):
        _, seen = _live_hy(monkeypatch, tmp_path, load_s=2.0, fetch_s=2.0, slices=20, slice_s=0.1)
        worker, track, events = _live_worker(monkeypatch, tmp_path, TwoLineEngine(), targets)
        worker.start()
        try:
            for k in range(12):                                               # a chunk every 0.5 s (speech 14x real time)
                track.feed(_chunk_of_speech(8 * k))
                time.sleep(0.5)
            _wait_for(lambda: len([l for _, l in events.lines() if "text" in l and "tr" not in l]) >= 24, 5)
        finally:
            worker.stop(wait=30)
        firsts = {}
        for at, l in events.lines():
            if "text" in l:
                firsts.setdefault(l["text"], at)
        times = sorted(firsts.values())
        return max(b - a for a, b in zip(times, times[1:])), events, seen, worker
    plain_gap, _, _, _ = run([])
    gap, events, seen, worker = run(["English"])
    assert gap <= plain_gap + 1.5, f"a gap of {gap:.1f} s between two lines (without translation {plain_gap:.1f} s)"
    assert seen["fetch_threads"] and worker.thread.ident not in seen["fetch_threads"], "the files are checked off the worker's thread"
    states = [s["state"] for _, _, s in events.stats("model") if s.get("role") == "translator"]
    assert "checking" in states and "loading" in states and states[-1] in ("ready", "stopped", "released"), states
    assert states.index("checking") < states.index("loading") < states.index("ready")
    progress = [s["progress"] for _, _, s in events.stats("model") if s.get("state") == "loading" and "progress" in s]
    assert progress == sorted(progress) and progress[-1] == 1.0
    ready = next(s for _, _, s in events.stats("model") if s.get("state") == "ready" and s.get("role") == "translator")
    assert ready["load_ms"] >= 2000


class _VirtualEngine:
    """Decodes a chunk into the lines of the utterances in it: utterance u carries the sample value u + 1, and gives a
    line every `seg` seconds (as Whisper gives a segment), named after it, so the lines do not depend on how the speech
    was cut into chunks. Each decode costs `cost` seconds of the virtual clock."""
    name = "fake"

    def __init__(self, clock, cost, seg=6.0):
        self.clock, self.cost, self.seg, self.calls = clock, cost, seg, 0

    def analyse(self, chunk, detect=True):
        return SimpleNamespace(chunk=chunk, probs=[("ru", 0.99)] if detect else None)

    def decode(self, state, language):
        import numpy as np
        self.clock[0] += self.cost
        self.calls += 1
        values = np.rint(np.asarray(state.chunk) * 32768).astype(int)
        segs = []
        for u in sorted(set(values[values > 0].tolist())):
            where = np.flatnonzero(values == u)
            start, end = where[0] / 16000, (where[-1] + 1) / 16000
            n = max(1, int(round((end - start) / self.seg)))
            step = (end - start) / n
            segs.extend({"start": start + k * step, "end": start + (k + 1) * step, "text": f"u{u - 1}-{k}", "logprob": -0.2}
                        for k in range(n))
        return segs


def _virtual_speech(minutes, tracks=1, seed=1, short=(3, 12), gap=(0.8, 2.0)):
    """The lag driver's synthetic meeting: utterances of 3-12 s, pauses of 0.8-2 s (a call's other side answers 60 % of
    the time). Returns the int16 audio per track (utterance u carries u + 1) and the utterances (track, start, end)."""
    import random
    import numpy as np
    rnd = random.Random(seed)
    total = int(minutes * 60)
    audio = [np.zeros(total * 16000, dtype="int16") for _ in range(tracks)]
    pos, who, said = 0.5, 0, []
    while pos < total - 1:
        end = min(total - 0.5, pos + rnd.uniform(*short))
        audio[who][int(pos * 16000):int(end * 16000)] = len(said) + 1
        said.append((who, pos, end))
        pos = end + rnd.uniform(*gap)
        if tracks == 2 and rnd.random() < 0.6:
            who = 1 - who
    return audio, said


def _virtual_live(monkeypatch, tmp_path, minutes=10.0, d=0.3, t=6.0, tokens=30, targets=2, tracks=1, mode="hy", seg=6.0,
                  short=(3, 12), gap=(0.8, 2.0), seed=1):
    """T3's harness: the real LiveTrack, LiveWorker and LiveTranslator on a virtual clock, driven step by step on this
    thread. Before each step() the audio up to clock() is fed; a decode costs d, a Hy-MT3 request t (tokens tokens of
    t / tokens each), an NLLB batch t; an idle step costs 0.1 s. Then the stop: the drain on the same clock. Returns a
    dict of what was shown and when (virtual seconds)."""
    clock = [0.0]
    _content_vad(monkeypatch)
    events = []
    monkeypatch.setattr(mt, "ev", lambda kind, *fields: events.append((clock[0], kind, fields)))
    monkeypatch.setattr(mt, "EVENTS", True)
    engine = _VirtualEngine(clock, d, seg)
    tok = FakeHyTokenizer()

    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):
        for k in range(tokens):
            clock[0] += t / tokens
            yield SimpleNamespace(text="<tr>" if k == 0 else "x", finish_reason=None if k < tokens - 1 else "stop")

    class Batches(FakeTranslator):
        def translate_batch(self, tokens_, target_prefix, **kw):
            clock[0] += t
            return super().translate_batch(tokens_, target_prefix, **kw)
    nllb = (Batches(), FakeTokenizer())
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=stream_generate))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_logits_processors=lambda **kw: []))
    monkeypatch.setattr(mt, "llm_live_parts", lambda stopped=None: SimpleNamespace(path=tmp_path, tok=tok, bytes=100))
    monkeypatch.setattr(mt, "llm_live_model", lambda path: "model")
    monkeypatch.setattr(mt, "eval_in_slices", lambda model, slice_bytes=None: iter([(100, 100)]))
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: nllb)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy" if mode == "hy" else "nllb")
    audio, said = _virtual_speech(minutes, tracks, seed, short, gap)
    names = ["You", "Them"]
    live_tracks = [mt.LiveTrack(f"t{k}.wav", tmp_path, f".{k + 1}" if tracks > 1 else "", tmp_path / f"t{k}.wav", ["ru"], "normal",
                                names[k] if tracks > 1 else None, mt.engine_name()) for k in range(tracks)]
    langs = ["English", "German"][:targets] if mode != "none" else []
    worker = mt.LiveWorker(live_tracks, targets=langs, offline=True)
    worker.clock = lambda: clock[0]
    worker.engine = engine
    worker.translator.engine = engine
    worker.translator.started_at = 0.0
    worker.translator.prepare()                                              # its thread, run here: deterministic
    n, fed = len(audio[0]), 0

    def feed(upto):
        nonlocal fed
        upto = min(n, upto)
        if upto > fed:
            for k, track in enumerate(live_tracks):
                track.feed(audio[k][fed:upto].reshape(-1, 1))
            fed = upto
    while fed < n:
        feed(int(clock[0] * 16000))
        if not worker.step():
            clock[0] += 0.1
    stop_at = clock[0]
    pending_at_stop = sum(len(tr.pending) for tr in live_tracks)
    worker.stop_flag.set()
    worker.drain()
    for track in live_tracks:
        track.close()
    shown, translated, skipped = {}, {}, {}
    for at, kind, fields in events:
        if kind != "LINE":
            continue
        line = json.loads(fields[0])
        if "text" not in line:
            continue
        key = (line.get("speaker"), line["text"])
        shown.setdefault(key, (at, line["end"]))
        for target in line.get("tr", {}):
            translated.setdefault((key, target), at)
        for target in line.get("tr_skip", []):
            skipped.setdefault((key, target), at)
    stats = [(at, f[0], f[1]) for at, k, f in events if k == "STAT"]
    return dict(shown=shown, translated=translated, skipped=skipped, stats=stats, targets=langs, stop_at=stop_at,
                drain_s=clock[0] - stop_at, pending_at_stop=pending_at_stop, d=d, events=events, worker=worker,
                engine=engine, tracks=live_tracks)


def _lags(run):
    return {key: at - end for key, (at, end) in run["shown"].items()}


def _check_bounds(run, plain, token_s):
    """The bounds of G1 and G2 (spec A1) on one virtual run against the same audio without translation."""
    import numpy as np
    lag, lag0 = _lags(run), _lags(plain)
    assert set(lag0) <= set(lag), "every line of the run without translation is shown with it too"
    worst = max(lag[k] - lag0[k] for k in lag0)
    assert worst <= mt.CUT_EVERY_S + max(token_s, 0.2) + 1e-6, f"a line came {worst:.2f} s later than without translation"
    def slope(r, lags):
        late = sorted((end, lags[k]) for k, (_, end) in r["shown"].items() if r["stop_at"] - 300 <= end < r["stop_at"])
        return float(np.polyfit([e for e, _ in late], [l for _, l in late], 1)[0]) * 60 if len(late) > 3 else 0.0
    # The lag must not grow with the meeting's length. Measured against the same audio without translation: that run's
    # own slope over the last 5 minutes is noise of up to 0.18 s a minute (seed 1), so the slope is compared with it.
    grows, base = slope(run, lag), slope(plain, lag0)
    assert grows <= base + 0.25, f"the lag grows by {grows:.2f} s a minute over the last 5 minutes ({base:.2f} without translation)"
    for (key, target), at in run["translated"].items():
        shown_at, end = run["shown"][key]
        # G2: at most LIVE_TR_BEHIND_S + the longest job (6 s) + CUT_EVERY_S after the line was shown (a line of a 30 s
        # chunk is shown late already: its translation is measured from then).
        assert at - shown_at <= mt.LIVE_TR_BEHIND_S + 6 + mt.CUT_EVERY_S + 1e-6, (key, target, at - shown_at, at - end)
    for key in run["shown"]:
        for target in run["targets"]:
            assert (key, target) in run["translated"] or (key, target) in run["skipped"], (key, target)
    return worst


def test_live_lag_stays_bounded_when_translation_cannot_keep_up(monkeypatch, tmp_path):
    """T3. The worst row of the lag driver (1.12.0: lag 256 s median, growing 46 s a minute): 10 minutes of speech,
    0.3 s a chunk, Hy-MT3 6 s a line and target (30 tokens of 0.2 s), two targets, so the translation alone would need
    1.7 times the audio. The transcript keeps its pace (each line at most CUT_EVERY_S plus one token later than without
    translation, no growth), each line translated live comes at most 30 s + one job + CUT_EVERY_S after it was shown,
    the rest is marked for the final pass, and the stop does not wait for the translation."""
    plain = _virtual_live(monkeypatch, tmp_path / "plain", mode="none")
    run = _virtual_live(monkeypatch, tmp_path / "hy")
    _check_bounds(run, plain, 0.2)
    kinds = [kind for _, kind, _ in run["stats"]]
    skips = [json.loads(f) for _, k, f in run["stats"] if k == "trskip"]
    assert any(s["reason"] in ("behind", "duty") for s in skips), "the translator skipped to the newest chunk"
    assert any(json.loads(f)["state"] == "behind" for _, k, f in run["stats"] if k == "queue")
    assert run["translated"], "and translated live what it could"
    assert run["drain_s"] <= (run["pending_at_stop"] + 1) * run["d"] + 1e-6, "the drain decodes, it does not translate"
    if run["pending_at_stop"] or "trskip" in kinds:
        assert any(s["reason"] == "stop" for s in skips) or not run["worker"].translator.queue


def test_stat_events_are_small_valid_json_and_never_per_token(monkeypatch, tmp_path):
    """T12, on a 4-minute run of T3's harness."""
    run = _virtual_live(monkeypatch, tmp_path, minutes=4)
    events = run["events"]
    stats = [(at, f) for at, k, f in events if k == "STAT"]
    assert stats
    for at, (kind, body) in stats:
        assert len(("@@STAT\t" + kind + "\t" + body).encode("utf-8")) <= 1024, body
        assert isinstance(json.loads(body), dict) and kind in {"chunk", "tr", "trskip", "queue", "model", "sys"}
    by_kind = collections.Counter(kind for _, (kind, _) in stats)
    assert by_kind["tr"] <= len(run["shown"]) * len(run["targets"]), "at most one per line and target: never per token"
    queue_at = [at for at, (kind, _) in stats if kind == "queue"]
    assert max(sum(1 for b in queue_at if a <= b < a + 1) for a in queue_at) <= 2, "queue: at most 2 a second"
    sys_at = [at for at, (kind, _) in stats if kind == "sys"]
    assert sys_at and all(b - a >= mt.STAT_SYS_EVERY_S - 1e-6 for a, b in zip(sys_at, sys_at[1:])), "sys: every 2 s at most"


def test_a_stat_line_with_long_non_ascii_strings_stays_within_1024_bytes(monkeypatch):
    """A8: strings are cut at 200 characters, but 200 non-ASCII characters are up to 800 bytes: a model folder and an
    error in Japanese are shortened further so that the whole line is at most 1024 bytes, and it is still valid JSON."""
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    mt.stat("model", role="translator", name="/Users/ユーザー/モデル/" + "翻訳" * 100, engine="mlx", state="failed",
            error="RuntimeError: " + "モデルを読み込めません" * 30)
    mt.stat("trskip", reason="behind", lines=3, targets=["ポルトガル語（ブラジル）" * 3] * 40, total=10)
    mt.stat("queue", state="ready", jobs=0)
    for kind, body in [f for _, f in events.of("STAT")]:
        assert len(("@@STAT\t" + kind + "\t" + body).encode("utf-8")) <= 1024
        fields = json.loads(body)
        assert fields["state" if kind != "trskip" else "reason"]
    model = json.loads(events.of("STAT")[0][1][1])
    assert model["name"].startswith("/Users/ユーザー/") and model["error"].startswith("RuntimeError: ")
    assert json.loads(events.of("STAT")[2][1][1]) == {"state": "ready", "jobs": 0}, "a short line is left as it is"


def test_a_jobs_model_download_says_how_much_is_downloaded(monkeypatch):
    """The app turns tqdm's bars off (HF_HUB_DISABLE_PROGRESS_BARS): a job's download sends its STAGE again with the
    GB so far instead; during a recording (stage=False) the live translator's @@STAT model does that, not a STAGE."""
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    monkeypatch.setattr(mt, "DOWNLOAD_STAGE_EVERY_S", 0.02)
    monkeypatch.setattr(mt, "fetch_model", _REAL.fetch_model)
    monkeypatch.setattr(mt, "model_present", lambda repo: False)
    monkeypatch.setattr(mt, "hf_partial_bytes", lambda repos: 3 * 2**29)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda repo: time.sleep(0.2)))
    mt.fetch_model("org/some-model", "about 1 GB")
    stages = [f[0] for _, f in events.of("STAGE")]
    first = "Fetching the translation model org/some-model first (once, about 1 GB), so that it is here when it is needed"
    assert stages[0] == first and first + ": 1.5 GB so far" in stages[1:]
    count = len(stages)
    time.sleep(0.1)
    assert len(events.of("STAGE")) == count, "none after the download"
    events.items.clear()
    mt.fetch_model("org/some-model", "about 1 GB", stage=False)
    assert not events.of("STAGE")


def test_lines_that_aged_while_the_slot_was_held_elsewhere_are_skipped_as_slot(monkeypatch):
    """B2's trskip reasons: a job that aged while the translator waited for another process's slot says 'slot'; one that
    aged while the translator was ready says 'behind'."""
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    worker = mt.LiveWorker([], targets=["English"], offline=False)
    clock = [100.0]
    worker.clock = lambda: clock[0]
    tr = worker.translator
    for state, reason in (("waiting", "slot"), ("ready", "behind")):
        events.items.clear()
        tr.state = state
        tr.queue = [mt._Job("claude", "English", [{"t": 1.0, "text": "Да"}], None, None, 1, 50.0)]
        tr._skip_to_newest(clock[0])
        skips = [json.loads(f[1]) for _, f in events.of("STAT") if f[0] == "trskip"]
        assert [s["reason"] for s in skips] == [reason] and not tr.queue


def test_a_wrong_mlx_cache_size_is_a_warning_not_a_crash(capsys):
    assert mt._cache_mb(None) == 1024 and mt._cache_mb("512") == 512 and mt._cache_mb("0") == 0
    assert mt._cache_mb("1g") == 1024 and "MT_MLX_CACHE_MB='1g'" in capsys.readouterr().err


def test_a_translation_models_load_does_not_count_toward_the_duty_cap():
    """The load's slices are the worker's time (duty_tr) but not the trailing minute's duty: the jobs that waited during
    a long load are not held back (and skipped as 'duty') by it once the model is ready."""
    worker = mt.LiveWorker([], targets=["English"], offline=False)
    clock = [100.0]
    worker.clock = lambda: clock[0]
    tr = worker.translator
    tr.add_busy(40.0, duty=False)                      # 60 s of load slices
    assert tr.duty(clock[0]) == 0.0 and tr.take_busy() == 60.0
    tr.add_busy(90.0)                                  # 10 s of translation
    assert tr.duty(clock[0]) == 10.0 and tr.take_busy() == 10.0


def test_a_request_still_running_at_the_stop_does_not_keep_the_process_from_ending(tmp_path):
    """A Claude request (or a CPU NLLB batch) on the helper thread after the stop: the process ends without waiting
    for it (its result is not wanted; a ThreadPoolExecutor's thread would be joined at the exit)."""
    import subprocess
    code = ("import sys, time; sys.path.insert(0, %r); import mt\n"
            "tr = mt.LiveWorker([], targets=['English'], offline=False).translator\n"
            "f = tr._executor().submit(time.sleep, 30)\n"
            "time.sleep(0.2); tr.halt(); assert f.running()\n") % str(Path(mt.__file__).parent)
    t0 = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ, MT_HOME=str(tmp_path)),
                          timeout=25, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert time.monotonic() - t0 < 15
    helper = mt._DaemonHelper("test-helper")
    ok, bad = helper.submit(lambda a, b: a + b, 1, 2), helper.submit(lambda: 1 / 0)
    assert ok.result(5) == 3 and isinstance(bad.exception(5), ZeroDivisionError)
    helper.shutdown(wait=True)


@pytest.mark.parametrize("row", [
    dict(mode="none", t=0), dict(mode="nllb", t=0.2, targets=1), dict(mode="nllb", t=0.2, targets=2),
    dict(t=1, targets=1), dict(t=1, targets=2), dict(t=3, targets=1), dict(t=3, targets=2), dict(t=6, targets=1),
    dict(t=6, targets=2), dict(t=3, targets=2, tracks=2), dict(t=6, targets=2, tracks=2),
    dict(t=1.5, targets=1, seg=3.0, short=(2, 6), gap=(0.4, 1.0)), dict(t=2, targets=1, seg=3.0, short=(2, 6), gap=(0.4, 1.0)),
    dict(t=4, targets=1, seg=3.0, short=(2, 6), gap=(0.4, 1.0))],
    ids=lambda r: "-".join(f"{k}{v}" for k, v in r.items() if k not in ("short", "gap")))
def test_live_lag_grid(monkeypatch, tmp_path, row):
    """T4: the rows of the lag driver's grid (telemetry/lag/summary.tsv), 6 minutes each: none; NLLB-like (0.2 s a
    batch); Hy-MT3 1/3/6 s a line with 1-2 targets; a call (2 tracks) at 3/6 s; a dense meeting at 1.5/2/4 s. The bound
    of T3 on every row, and with the translation needing at most 0.7 of the audio's time, no skip at all."""
    row = dict(row)
    tokens = 30 if row.get("mode", "hy") == "hy" else 1
    common = {k: row[k] for k in ("tracks", "seg", "short", "gap") if k in row}
    plain = _virtual_live(monkeypatch, tmp_path / "plain", minutes=6, mode="none", **common)
    run = _virtual_live(monkeypatch, tmp_path / "run", minutes=6, tokens=tokens, **row)
    token_s = row["t"] / tokens if row.get("mode", "hy") == "hy" else row["t"]
    _check_bounds(run, plain, token_s)
    lines_per_s = len(plain["shown"]) / (6 * 60)
    rho = lines_per_s * row["t"] * row.get("targets", 0) if row.get("mode", "hy") == "hy" else 0.0
    if rho <= 0.7:
        assert not [1 for _, k, f in run["stats"] if k == "trskip" and json.loads(f)["reason"] != "stop"], f"rho {rho:.2f}: no skip"


def test_stop_leaves_translation_to_the_final_pass_and_caps_the_drain(monkeypatch, tmp_path):
    """T5. 20 chunks still to decode at the stop, 6 s each: the drain ends after LIVE_DRAIN_S of (virtual) time, and the
    chunks it did not get to are decoded by the final pass, each chunk once."""
    import numpy as np
    clock = [0.0]
    _content_vad(monkeypatch)
    monkeypatch.setattr(mt, "EVENTS", False)
    engine = _VirtualEngine(clock, 6.0)
    audio = np.concatenate([np.concatenate([np.full(16000 * 6, k + 1, dtype="int16"), np.zeros(16000 * 2, dtype="int16")])
                            for k in range(20)] + [np.zeros(16000 * 3, dtype="int16")])     # utterance k: 6 s, then a pause
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["ru"], "normal", None, mt.engine_name())
    worker = mt.LiveWorker([track], targets=["English"], offline=False)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    worker.clock = lambda: clock[0]
    worker.engine = engine
    track.feed(audio.reshape(-1, 1))
    worker.step()                                                            # one chunk decoded, its translation queued
    assert len(track.pending) == 19 and worker.translator.queue
    started = clock[0]
    worker.stop_flag.set()
    worker.drain()
    track.close()
    assert clock[0] - started <= mt.LIVE_DRAIN_S + 1e-6, clock[0] - started
    live_calls = engine.calls
    assert 1 < live_calls < 20 and track.pending, "the drain stopped at its cap"
    assert worker.translator.stopped and not worker.translator.queue
    engine.cost = 0.0
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio.astype("float32") / 32768)
    lines = mt.transcribe_lines(tmp_path / "x.wav", "ru", granularity="normal", checkpoint=tmp_path)
    assert engine.calls == 20, "the final pass decoded exactly the chunks the live run left"
    assert [l["text"] for l in lines if "text" in l] == [f"u{k}-0" for k in range(20)], "every chunk once, in order"


def _hy_translate_1_12(items, tgt, model, tok):
    """hy_translate as 1.12.0 had it (mt.py at bb8d15c), one stream_generate per piece run to its end: the reference
    the stepped path must equal."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors
    decoding = dict(logits_processors=make_logits_processors(repetition_penalty=1.05, repetition_context_size=20))
    out, cut = [], 0
    for src, text in items:
        done, short = [], False
        for p in mt._pieces(text, mt.HY_PIECE):
            ids = tok.apply_chat_template([{"role": "user", "content": mt.hy_prompt(p, tgt)}], add_generation_prompt=True)
            budget = mt.hy_budget(len(tok.encode(p, add_special_tokens=False)), tgt)
            reply, finish = "", None
            for r in stream_generate(model, tok, prompt=ids, max_tokens=budget, **decoding):
                reply += r.text
                finish = r.finish_reason
            short = short or finish == "length"
            done.append(reply.split("<|eos|>")[0].strip() or p)
        cut += short
        out.append(("" if tgt in mt.HY_NO_SPACES else " ").join(done))
    return out, cut


def _token_model(monkeypatch, calls, limit_words=None):
    """A deterministic token stream: the reply to a prompt is its text's words upper-cased, four tokens each, then
    <|eos|>; at max_tokens it stops with "length". Every request is recorded (prompt text, max_tokens, options)."""
    def stream_generate(model, tokenizer, prompt, max_tokens=256, **kw):
        text = tokenizer.prompts[prompt[0]].rsplit("\n\n", 1)[-1] if hasattr(tokenizer, "prompts") else \
            tokenizer.messages[prompt[0]]["text"]
        calls.append((text, max_tokens, sorted(kw)))
        words = [w.upper() + " " for w in text.split() for _ in range(4)] + ["<|eos|>"]     # 4 tokens a word
        n = min(len(words), max_tokens)
        for k in range(n):
            yield SimpleNamespace(text=words[k], finish_reason=None if k < n - 1 else "length" if n < len(words) else "stop")
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=stream_generate,
                                                                generate=lambda m, t, prompt, **kw: "".join(r.text for r in stream_generate(m, t, prompt, **kw))))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", SimpleNamespace(make_logits_processors=lambda **kw: ["penalty"]))


def _live_translate(monkeypatch, tmp_path, lines, targets, tok):
    """The live path on its own: a LiveWorker whose prepare and load are stand-ins, the lines queued as LiveTrack.decode
    queues them, and the worker's steps run here until the translator is idle."""
    monkeypatch.setattr(mt, "llm_live_parts", lambda stopped=None: SimpleNamespace(path=tmp_path, tok=tok, bytes=1))
    monkeypatch.setattr(mt, "llm_live_model", lambda path: "model")
    monkeypatch.setattr(mt, "eval_in_slices", lambda model, slice_bytes=None: iter([(1, 1)]))
    worker = mt.LiveWorker([], targets=targets, offline=True)
    worker.translator.prepare()
    worker.translate(lines)
    _drain_translator(worker)
    return worker


def test_stepped_hy_and_gemma_give_the_one_shot_text(monkeypatch, tmp_path):
    """T6. The live worker steps Hy-MT3 one token at a time between two decodes; its text, its count of lines cut at the
    token limit and its requests (same prompts, budgets and options, in the same order for a line) are those of
    1.12.0's one-shot hy_translate. Pieces and the limit included; TranslateGemma likewise against generate()."""
    calls = []
    _token_model(monkeypatch, calls)
    tok = FakeHyTokenizer()
    monkeypatch.setattr(mt, "_load_hy", lambda: ("model", tok))
    monkeypatch.setattr(mt, "release_engine", lambda: None)
    monkeypatch.setattr(mt, "TRANSLATOR", "hy")
    long = " ".join(f"Sentence number {k} is here." for k in range(120))       # in pieces
    texts = ["привет как дела", long, ("слово " * 300).strip()]               # the last runs to its token limit
    items = [("ru", t) for t in texts]
    reference, ref_calls = _hy_translate_1_12(items, "en", "model", tok), list(calls)
    assert reference[1] >= 1, "the stand-in does run to the token limit"
    calls.clear()
    assert mt.hy_translate(items, "en") == reference and calls == ref_calls
    stepped = []
    for src, text in items:
        gen = mt.hy_steps(src, text, "en", "model", tok)
        steps = 0
        while True:
            try:
                next(gen)
                steps += 1
            except StopIteration as done:
                stepped.append(done.value)
                break
        assert steps > 1, "one step per token"
    assert [t for t, _ in stepped] == reference[0] and sum(c for _, c in stepped) == reference[1]
    calls.clear()
    lines = [{"t": float(k), "end": float(k) + 1, "lang": "ru", "conf": 1.0, "text": t} for k, t in enumerate(texts)]
    _live_translate(monkeypatch, tmp_path, lines, ["English"], tok)
    assert [l["tr"]["English"] for l in lines] == reference[0] and calls == ref_calls, "the live path: the same text, the same requests"
    assert all(l["tr_by"] == {"English": mt.HY} and l["tr_of"] == l["text"] and l["tr_offline"] for l in lines)
    # TranslateGemma: 1.12.0 called generate(), whose body in mlx-lm 0.29.1 is the concatenation of stream_generate's text.
    gtok = FakeGemmaTokenizer()
    monkeypatch.setattr(mt, "_load_gemma", lambda: ("model", gtok))
    monkeypatch.setattr(mt, "TRANSLATOR", "gemma")
    calls.clear()
    one_shot = []
    for src, text in items[:2]:
        done = []
        for p in mt._pieces(text, mt.GEMMA_PIECE):
            msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": src, "target_lang_code": "en", "text": p}]}]
            ids = gtok.apply_chat_template(msgs, add_generation_prompt=True)
            reply = sys.modules["mlx_lm"].generate("model", gtok, prompt=ids, max_tokens=min(2048, 2 * len(ids) + 64))
            done.append(reply.split("<end_of_turn>")[0].strip() or p)
        one_shot.append(" ".join(done))
    ref_calls = list(calls)
    calls.clear()
    assert mt.gemma_translate(items[:2], "en") == one_shot and calls == ref_calls
    calls.clear()
    glines = [{"t": float(k), "end": float(k) + 1, "lang": "ru", "conf": 1.0, "text": t} for k, t in enumerate(texts[:2])]
    _live_translate(monkeypatch, tmp_path, glines, ["English"], gtok)
    assert [l["tr"]["English"] for l in glines] == one_shot and calls == ref_calls


class _PatternStream(FakeStream):
    """3 s of speech, 1 s of silence, 10x real time (as the record --live tests have it)."""
    def __init__(self, samplerate, channels, dtype, device, callback):
        super().__init__(samplerate, channels, dtype, device, callback)
        self.second = 0

    def _run(self):
        import numpy as np
        while not self.stop_flag.is_set():
            s = self.second
            self.block = np.full((1600, 1), 0 if s % 4 == 3 else s + 1, dtype="int16")
            for _ in range(10):
                if self.stop_flag.is_set():
                    break
                self.callback(self.block, 1600, None, None)
                time.sleep(0.01)
            self.second += 1


def _record_live_only(monkeypatch, tmp_path, capsys, targets):
    """`record --no-transcribe --live` of 40 s (10x real time) with live translation into `targets` (offline, NLLB
    stand-in), as the app runs it; then the queued `transcribe --into` with the same targets. Returns (folder, the
    records of the live checkpoint, the first run's events, the second run's events)."""
    pytest.importorskip("soundfile")
    _content_vad(monkeypatch)
    engine = SecondEngine()
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    monkeypatch.setattr(mt, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(mt, "DATA", tmp_path)
    monkeypatch.setattr(mt, "EVENTS", True)
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=_PatternStream))
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: __import__("soundfile").read(str(src), dtype="float32")[0])
    to = [arg for t in targets for arg in ("--to", t)]
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(40 / 3600), "--no-transcribe", "--live", "--name", "standup",
                                      "--language", "en", "--granularity", "coarse", *to, "--offline"])
    mt.main()
    first = capsys.readouterr().out.splitlines()
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    records = [json.loads(r) for r in (folder / "partial.jsonl").read_text().splitlines()]
    monkeypatch.setattr(sys, "argv", ["mt.py", "transcribe", str(folder / "audio.wav"), "--into", str(folder), "--out", str(folder),
                                      "--stem", "standup", "--language", "en", "--granularity", "coarse", *to, "--offline"])
    mt.main()
    second = capsys.readouterr().out.splitlines()
    return folder, records, first, second


def test_live_translations_are_byte_identical_to_the_one_shot_path_and_reused_at_the_end(monkeypatch, tmp_path, capsys):
    """T7. A line translated live has the text and tr_by the one-shot path gives it; the last checkpoint record of its
    chunk carries it; the final pass reuses it ("translated live") and writes it into the file unchanged."""
    folder, records, first, second = _record_live_only(monkeypatch, tmp_path, capsys, ["Russian"])
    last = {}
    for rec in records:
        last[rec["i"]] = rec["lines"]
    live = [l for lines in last.values() for l in lines if "tr" in l]
    assert live, "a chunk was translated while recording and written again with it"
    for line in live:
        fresh = {k: v for k, v in line.items() if k not in ("tr", "tr_of", "tr_by", "tr_offline")}
        assert line["tr"] == {"Russian": mt.translate_texts([fresh], "Russian", True)[0]} and line["tr_of"] == line["text"]
        assert line["tr_by"] == {"Russian": mt.line_translator(line.get("lang"), "Russian", True)} and line["tr_offline"] is True
    assert any(o.startswith("@@NOTE\tTranslated") and "translated live" in o for o in second)
    text = (folder / "standup.Russian.txt").read_text()
    assert all(line["tr"]["Russian"] in text for line in live)


def test_telemetry_fields_are_only_in_events(monkeypatch, tmp_path, capsys):
    """T13. tr_ms, tr_skip and dec_ms are sent in events only: never in transcript.json, the checkpoint or a text file."""
    folder, records, first, second = _record_live_only(monkeypatch, tmp_path, capsys, ["Russian"])
    assert any("tr_ms" in o for o in first) and any(o.startswith("@@STAT\tchunk\t") and "dec_ms" in o for o in first)
    assert not any(k in json.dumps(records) for k in ("tr_ms", "tr_skip", "dec_ms")), "not in the live checkpoint"
    for path in folder.iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".txt"):
            body = path.read_text(encoding="utf-8")
            assert not any(k in body for k in ("tr_ms", "tr_skip", "dec_ms", "@@STAT")), path.name


def test_live_scheduling_adds_no_note(monkeypatch, tmp_path, capsys):
    """T13. The app puts every NOTE at the top of the transcript: the scheduler's messages (behind, paused, the
    model's states) go to stderr and @@STAT, so a recording translated live has the notes of one that is not."""
    _, _, plain_first, plain_second = _record_live_only(monkeypatch, tmp_path / "plain", capsys, [])
    _, _, first, second = _record_live_only(monkeypatch, tmp_path / "live", capsys, ["Russian"])
    notes = lambda out: [o for o in out if o.startswith("@@NOTE\t")]
    assert notes(first) == notes(plain_first), "no note while recording"
    extra = [n for n in notes(second) if n not in notes(plain_second)]
    assert all(n.startswith("@@NOTE\tTranslated ") for n in extra), extra


_REAL_THREAD_STREAM = mt._thread_stream
_REAL_EVAL_IN_SLICES = mt.eval_in_slices


def test_every_mlx_call_of_a_live_run_is_on_the_worker_thread(monkeypatch, tmp_path):
    """T8. One thread owns the GPU: a stand-in mlx.core records the thread of every eval, new_stream, set_cache_limit,
    synchronize and clear_cache of a live run with Hy-MT3 (the engine on the GPU, the weights evaluated in slices, the
    generation stream switched): there is one, the worker's. The prepare thread (files, tokenizer, page cache) makes none."""
    import threading as th
    used = collections.defaultdict(set)

    def record(name, result=None):
        def call(*a, **k):
            used[name].add(th.get_ident())
            return result
        return call
    mx = SimpleNamespace(eval=record("eval"), new_stream=record("new_stream", "stream"), default_device=lambda: "gpu",
                         set_cache_limit=record("set_cache_limit"), synchronize=record("synchronize"),
                         clear_cache=record("clear_cache"))
    params = {"layers": [SimpleNamespace(nbytes=200 << 20) for _ in range(5)]}

    class Model:
        def parameters(self):
            return params
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx.utils", SimpleNamespace(tree_flatten=lambda tree: [(str(k), a) for k, a in enumerate(tree["layers"])]))
    abort, seen = _live_hy(monkeypatch, tmp_path, token_s=0.005, tokens=20, fetch_s=0.2)
    monkeypatch.setattr(mt, "eval_in_slices", _REAL_EVAL_IN_SLICES)
    monkeypatch.setattr(mt, "llm_live_model", lambda path: Model())
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", SimpleNamespace(generation_stream="made at import"))
    monkeypatch.setattr(mt, "_STREAM_THREAD", [])
    monkeypatch.setattr(mt, "_thread_stream", _REAL_THREAD_STREAM)
    engine = TwoLineEngine()
    engine.name, engine.mx = "mlx", mx
    worker, track, events = _live_worker(monkeypatch, tmp_path, engine, ["English"])
    worker.start()
    try:
        for k in range(4):
            track.feed(_chunk_of_speech(8 * k))
            time.sleep(0.3)
        assert _wait_for(lambda: any("tr" in l for _, l in events.lines()), 10), "translated live"
    finally:
        worker.stop(wait=30)
    assert used["eval"] and used["set_cache_limit"] and used["new_stream"], dict(used)
    assert set().union(*used.values()) == {worker.thread.ident}, {k: v for k, v in used.items()}
    assert seen["fetch_threads"] and worker.thread.ident not in seen["fetch_threads"]
    assert set(seen["generate_threads"]) == {worker.thread.ident}


def test_the_weights_are_evaluated_in_slices_of_at_most_256_mb(monkeypatch):
    """eval_in_slices: one mx.eval per slice of at most LOAD_SLICE_BYTES (a tensor bigger than that is a slice of its
    own), every array once, the progress rising to the total."""
    evals = []
    mx = SimpleNamespace(eval=lambda *arrays: evals.append(arrays))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    sizes = [100, 100, 100, 300, 50, 10, 10]
    arrays = [SimpleNamespace(nbytes=s << 20) for s in sizes]
    monkeypatch.setitem(sys.modules, "mlx.utils", SimpleNamespace(tree_flatten=lambda tree: list(enumerate(tree))))
    model = SimpleNamespace(parameters=lambda: arrays)
    progress = list(_REAL_EVAL_IN_SLICES(model))
    assert [sum(a.nbytes for a in group) >> 20 for group in evals] == [200, 100, 300, 70]
    assert [a for group in evals for a in group] == arrays and progress[-1] == (sum(sizes) << 20, sum(sizes) << 20)
    assert [d for d, _ in progress] == sorted(d for d, _ in progress)


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_one_llm_on_the_machine_and_the_final_job_hands_over(monkeypatch, tmp_path, capsys):
    """T9. Two holders of the LLM slot in one process (flock belongs to an open file description, so they exclude each
    other as two processes do). A final job translates 10 lines with Hy-MT3; a live recording asks for the slot during
    it: the job lets the model go after its current line, says "Paused…", the recording gets the slot, and once the
    recording gives it back the job reloads and goes on from its log, with no line twice."""
    tok, calls, _ = _fake_hy(monkeypatch)
    loads = []

    @functools.lru_cache(maxsize=1)                                           # cached as the real one is (release_hy clears it)
    def load():
        loads.append(1)
        return "model", tok
    monkeypatch.setattr(mt, "_load_hy", load)
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    started = threading.Event()
    real_generate = sys.modules["mlx_lm"].stream_generate

    def slow(*a, **k):
        if len(calls) == 2:
            started.set()                                                     # the job is at its third line
        time.sleep(0.05)
        yield from real_generate(*a, **k)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(stream_generate=slow))
    doc = {"lines": [{"t": float(k), "lang": "ru", "conf": 1.0, "text": f"строка {k}"} for k in range(10)], "translations": {}}
    out = []
    job = threading.Thread(target=lambda: out.append(mt.translate(doc, "English", offline=True, checkpoint=tmp_path)))
    job.start()
    assert started.wait(5)
    live_want, live_slot = mt.FileLock("llm-want.lock"), mt.FileLock("llm.lock")   # what a live recording holds
    assert live_want.try_take()
    deadline = time.monotonic() + 10
    while not live_slot.try_take():
        assert time.monotonic() < deadline, "the job never handed the slot over"
        time.sleep(0.02)
    live_want.free()
    translated_when_live_got_it = len(calls)
    time.sleep(0.3)
    assert len(calls) == translated_when_live_got_it, "the job waits while the recording has the model"
    assert any(f == ("Paused: the recording in progress translates live (one translation model at a time)",)
               for _, f in events.of("STAGE"))
    live_slot.free()                                                          # the recording ended
    job.join(10)
    assert out and out[0] == [f"<English>строка {k}" for k in range(10)]
    assert len(loads) == 2, "loaded again after the hand-over"
    log = next((tmp_path / "translations").glob("*.jsonl"))
    done = [json.loads(r)["i"] for r in log.read_text().splitlines()]
    assert sorted(done) == list(range(10)) and len(done) == 10, "every line once in the log"
    assert len(calls) == 10, "no line translated twice"


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_a_job_waits_for_the_slot_a_live_recording_holds(monkeypatch, tmp_path):
    """A final job that finds the slot held at its start says so and waits for it."""
    tok, calls, _ = _fake_hy(monkeypatch)
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    live_slot = mt.FileLock("llm.lock")
    assert live_slot.try_take()
    out = []
    job = threading.Thread(target=lambda: out.append(mt.hy_translate([("ru", "да")], "en")))
    job.start()
    time.sleep(0.3)
    assert not out and ("Waiting for the live translation of the recording in progress",) in [f for _, f in events.of("STAGE")]
    live_slot.free()
    job.join(5)
    assert out == [(["<English>да"], 0)]


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_a_job_waits_while_a_live_recording_waits_for_the_free_slot(monkeypatch, tmp_path):
    """A final job that starts while a live recording waits for the slot (llm-want.lock held, llm.lock free) does not
    take it before the recording: it waits until the recording has it, then until the recording lets it go."""
    tok, calls, _ = _fake_hy(monkeypatch)
    monkeypatch.setattr(mt, "EVENTS", True)
    events = _Events()
    monkeypatch.setattr(mt, "ev", events)
    live_want, live_slot = mt.FileLock("llm-want.lock"), mt.FileLock("llm.lock")
    assert live_want.try_take()
    out = []
    job = threading.Thread(target=lambda: out.append(mt.hy_translate([("ru", "да")], "en")))
    job.start()
    time.sleep(0.3)
    assert not out and not calls and ("Waiting for the live translation of the recording in progress",) in [
        f for _, f in events.of("STAGE")]
    assert live_slot.try_take(), "the slot is still free for the recording"
    live_want.free()
    time.sleep(0.8)
    assert not out, "the recording has the slot: the job waits for it"
    live_slot.free()
    job.join(5)
    assert out == [(["<English>да"], 0)]


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_a_job_that_holds_the_slot_already_does_not_wait_for_a_waiting_recording(monkeypatch):
    """After a hand-over the job takes the slot back and loads its model again: take_llm_slot then returns at once,
    also when a recording waits for the slot (waiting there, holding it, would be a deadlock)."""
    monkeypatch.setattr(mt, "_SLOT", mt.FileLock("llm.lock"))
    assert mt._SLOT.try_take()
    live_want = mt.FileLock("llm-want.lock")
    assert live_want.try_take()
    done = threading.Event()
    threading.Thread(target=lambda: (mt.take_llm_slot(), done.set()), daemon=True).start()
    try:
        assert done.wait(2), "returned at once"
    finally:
        live_want.free()
        mt._SLOT.free()


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_the_live_marker_is_held_although_a_job_probes_it_at_that_instant():
    """A job's live_running() holds live.lock exclusively for an instant: a recording that starts then still holds
    its marker (it waits for the probe), so the job yields to it."""
    probe = mt.FileLock("live.lock")
    assert probe.try_take()
    threading.Timer(0.2, probe.free).start()
    marker = mt.live_marker()
    try:
        assert marker.held and mt.live_running()
    finally:
        marker.free()
    assert not mt.live_running()


def test_without_fcntl_the_slot_and_the_marker_do_nothing(monkeypatch):
    """Windows has no fcntl (and no TranslateGemma or Hy-MT3): every slot is free, no recording counts as live."""
    monkeypatch.setitem(sys.modules, "fcntl", None)                           # import fcntl raises ImportError
    a, b = mt.FileLock("llm.lock"), mt.FileLock("llm.lock")
    assert a.try_take() and b.try_take() and not mt.live_running() and not mt.live_waiting()
    a.free()
    b.free()


@pytest.mark.skipif(sys.platform == "win32", reason="flock: the LLM slot and the live marker do nothing on Windows")
def test_a_final_job_yields_the_gpu_while_a_recording_runs(monkeypatch, tmp_path):
    """T10. With live.lock held (a live recording runs), a processing job sleeps after each chunk as long as the chunk
    took, at most 2 s; without it, never."""
    import numpy as np
    pauses = []
    monkeypatch.setattr(mt, "_pause", pauses.append)
    _content_vad(monkeypatch)
    engine = TwoLineEngine(cost=0.05)
    monkeypatch.setattr(mt, "get_engine", lambda: engine)
    audio = np.concatenate([_chunk_of_speech(8 * k).reshape(-1) for k in range(3)]).astype("float32") / 32768
    monkeypatch.setattr(mt, "load_audio", lambda src, track=None: audio)
    mt.transcribe_lines(tmp_path / "x.wav", "ru", granularity="normal")
    assert pauses == [], "no recording: no pause"
    marker = mt.live_marker()
    try:
        mt.transcribe_lines(tmp_path / "x.wav", "ru", granularity="normal")
        assert len(pauses) == 3 and all(0.05 <= p < 0.5 for p in pauses), pauses
        pauses.clear()
        assert mt.gpu_yield(5.0) and pauses == [2.0], "at most 2 s"
    finally:
        marker.free()
    assert not mt.live_running() and not mt.gpu_yield(1.0)


def test_low_memory_mac_translates_live_with_nllb(monkeypatch, tmp_path):
    """T11. On an 8 GB Mac with Hy-MT3 chosen the live translation is NLLB's (a 7B model beside the speech model would
    swap): Hy-MT3 is never loaded live, @@STAT model says nllb, and the final pass translates those lines again with
    Hy-MT3 (tr_by is NLLB, not Hy-MT3)."""
    monkeypatch.setattr(mt, "total_memory", lambda: 8 * 2**30)
    abort, seen = _live_hy(monkeypatch, tmp_path)
    translator = FakeTranslator()
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (translator, FakeTokenizer()))
    worker, track, events = _live_worker(monkeypatch, tmp_path, TwoLineEngine(), ["English"])
    worker.start()
    try:
        track.feed(_chunk_of_speech(0))
        assert _wait_for(lambda: any("tr" in l for _, l in events.lines()), 5)
    finally:
        worker.stop(wait=10)
    assert seen["load_lazy"] == [] and seen["requests"] == 0, "Hy-MT3 was not loaded live"
    assert any(s.get("state") == "nllb" for _, _, s in events.stats("model"))
    lines = [l for l in track.shown if "text" in l]
    assert lines and all(l["tr_by"] == {"English": mt.NLLB} and l["tr"]["English"].startswith("T") for l in lines)
    assert mt.translated_live(lines, "English", True) == {}, "the final pass makes them again with Hy-MT3"


def test_live_pos_reports_the_decoded_end(monkeypatch, tmp_path, capsys):
    """T14. Live: x of y counts decoded audio: with two chunks cut and one decoded, POS is the first chunk's end, not
    the end of what was cut."""
    import numpy as np
    _content_vad(monkeypatch)
    monkeypatch.setattr(mt, "EVENTS", True)
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["ru"], "normal", None, mt.engine_name())
    track.feed(np.concatenate([_chunk_of_speech(0), _chunk_of_speech(8)]))
    assert track.cut() == 2 and track.covered == 16000 * 14
    track.decode(TwoLineEngine())
    track.close()
    assert "@@POS\t6.0\t16.0" in capsys.readouterr().out.splitlines()


def test_chunk_stat_carries_whisper_signals_without_changing_lines(monkeypatch, tmp_path, capsys):
    """T15. decode_chunk gathers the chunk's mean log-probability, highest temperature and compression ratio for
    @@STAT chunk; the lines are the ones it gives without stats."""
    class Signals(TwoLineEngine):
        def decode(self, state, language):
            segs = super().decode(state, language)
            segs[0].update(temp=0.4, cr=2.1)
            segs[1].update(temp=0.0, cr=1.3)
            return segs
    chunk = _chunk_of_speech(0).reshape(-1).astype("float32") / 32768
    plain = mt.decode_chunk(Signals(), chunk, 0, ["ru"], "ru", None)
    stats = {}
    assert mt.decode_chunk(Signals(), chunk, 0, ["ru"], "ru", None, stats=stats) == plain
    assert stats == {"lp": pytest.approx(-0.2), "temp": 0.4, "cr": 2.1}
    assert not any(k.startswith("_") for l in plain for k in l)
    _content_vad(monkeypatch)
    monkeypatch.setattr(mt, "EVENTS", True)
    track = mt.LiveTrack("x.wav", tmp_path, "", tmp_path / "x.wav", ["ru"], "normal", None, mt.engine_name())
    track.feed(_chunk_of_speech(0))
    track.cut()
    track.decode(Signals())
    track.close()
    chunk_stat = [json.loads(o.split("\t")[2]) for o in capsys.readouterr().out.splitlines() if o.startswith("@@STAT\tchunk\t")]
    assert chunk_stat and chunk_stat[0]["lp"] == -0.2 and chunk_stat[0]["temp"] == 0.4 and chunk_stat[0]["cr"] == 2.1
    assert chunk_stat[0]["src"] == "live" and chunk_stat[0]["lines"] == 2 and chunk_stat[0]["b"] == 6.0


def test_claude_and_cpu_nllb_jobs_run_beside_the_decode(monkeypatch, tmp_path):
    """T16. A Claude request that takes 3 s runs on the helper thread: the next chunk is decoded and shown meanwhile.
    Likewise NLLB beside the GPU engine (it runs on the CPU)."""
    release = threading.Event()

    class SlowClaude(FakeAnthropic):
        def create(self, **kw):
            release.wait(3)
            return super().create(**kw)
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=SlowClaude))

    class SlowNLLB(FakeTranslator):
        def translate_batch(self, tokens, target_prefix, **kw):
            release.wait(3)
            return super().translate_batch(tokens, target_prefix, **kw)
    monkeypatch.setattr(mt, "get_nllb", lambda **kw: (SlowNLLB(), FakeTokenizer()))
    for offline, engine in ((False, TwoLineEngine()), (True, TwoLineEngine())):
        if offline:
            engine.name, engine.mx = "mlx", SimpleNamespace()               # NLLB beside the GPU engine: the helper thread
        release.clear()
        worker, track, events = _live_worker(monkeypatch, tmp_path / str(offline), engine, ["English"])
        worker.offline = worker.translator.offline = offline
        if not offline:
            worker.translator.ready, worker.translator.state = True, "ready"
            worker.translator.models_here.set()
            worker.translator.prepared.set()
        worker.start()
        try:
            track.feed(_chunk_of_speech(0))
            assert _wait_for(lambda: worker.translator.current is not None and worker.translator.current.future is not None, 5)
            track.feed(_chunk_of_speech(8))
            assert _wait_for(lambda: any(l.get("text") == "c8 one" for _, l in events.lines()), 2.5), "decoded beside the request"
            assert not any("tr" in l for _, l in events.lines()), "while the first chunk's request is still running"
            release.set()
            assert _wait_for(lambda: any("tr" in l for _, l in events.lines()), 5)
        finally:
            release.set()
            worker.stop(wait=10)
