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
    assert len({(mt.nllb_code(name), mt.gemma_code(name)) for name in targets}) == len(targets), "two names for one language"
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

    def fake_record(name, hours, device, system_device=None, live=None, notes=None):
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
    monkeypatch.setattr(sys, "argv", ["mt.py", "record", "--hours", str(24 / 3600), "--live", "--language", "en", "--granularity", "coarse",
                                      "--to", "Russian", "--offline"])
    mt.main()
    out = capsys.readouterr().out.splitlines()
    lines = [json.loads(o.split("\t", 1)[1]) for o in out if o.startswith("@@LINE\t")]
    live = [l for l in lines if l.get("live")]
    shown_first = [l for l in live if "tr" not in l]
    translated = [l for l in live if "tr" in l]
    assert shown_first and translated, "a line is shown at once, then again with its translation"
    assert all(l["tr"] == {"Russian": "T" + l["text"]} and l["tr_offline"] for l in translated)
    assert all(any(t["t"] == s["t"] for t in translated) for s in shown_first), "every live line got its translation"
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
    worker.translate(lines)
    assert lines[0]["tr"] == {"English": "T0", "German": "T0"} and lines[2]["tr"] == {"English": "T1", "German": "T1"}
    assert "tr" not in lines[1] and lines[0]["tr_offline"] is False

    def broken(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(mt, "translate_live", broken)
    worker.translate(lines)
    worker.translate(lines)
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


def _fake_gemma(monkeypatch, reply=lambda item: f"<{item['target_lang_code']}>{item['text']}"):
    tok = FakeGemmaTokenizer()
    calls = []

    def generate(model, tokenizer, prompt, max_tokens=256, **kw):
        item = tokenizer.messages[prompt[0]]
        calls.append((item, max_tokens))
        return reply(item)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(generate=generate, load=lambda repo: ("model", tok)))
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
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lambda repo: ("model", tok),
                                                               generate=lambda *a, **k: "Hello<end_of_turn>junk"))
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
    assert all(len(p) <= mt.GEMMA_PIECE for p in mt._gemma_pieces(long_sentence.strip()))
    assert " ".join(mt._gemma_pieces(long_sentence.strip())) == long_sentence.strip(), "cut at words, nothing lost"


def test_a_gemma_translation_cut_short_goes_on_from_its_log(monkeypatch, tmp_path):
    tok, calls = _fake_gemma(monkeypatch)
    lines = [{"t": float(k), "lang": "ru", "conf": 1.0, "text": f"строка {k}"} for k in range(4)]
    seen = []

    def reply(item):
        if len(seen) == 2:
            raise KeyboardInterrupt                                          # the run is stopped after two lines
        seen.append(item["text"])
        return "EN " + item["text"]
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(generate=lambda m, t, prompt, **k: reply(t.messages[prompt[0]])))
    with pytest.raises(KeyboardInterrupt):
        mt.translate({"lines": lines, "translations": {}}, "English", offline=True, checkpoint=tmp_path)
    seen.clear()
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(generate=lambda m, t, prompt, **k: "EN " + t.messages[prompt[0]]["text"]))
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
    monkeypatch.setattr(mt, "gemma_available", lambda: True)
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
    monkeypatch.setattr(mt, "gemma_available", lambda: False)
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
    monkeypatch.setattr(mt, "gemma_available", lambda: True)
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
    monkeypatch.setattr(mt, "gemma_available", lambda: True)
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
    threading.Thread(target=worker.fetch_models, daemon=True).start()       # what run() starts when the model is missing
    first = [{"t": 0.0, "lang": "ru", "conf": 1.0, "text": "privet"}]
    worker.translate(first)
    assert "tr" not in first and loaded_on == [], "nothing waits: the line is translated at the end"
    assert "starts once the translation model is downloaded" in capsys.readouterr().err
    gate.set()
    worker.models_here.wait(5)
    second = [{"t": 1.0, "lang": "ru", "conf": 1.0, "text": "poka"}]
    worker.translate(second)
    assert second[0]["tr"] == {"English": "Tpoka"} and loaded_on and loaded_on[0] == threading.get_ident()


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


@pytest.fixture(autouse=True)
def _no_model_downloads(monkeypatch, request):
    """No test downloads a translation model: the cache is taken as holding them, and a fetch does nothing. The stream
    switch of mlx-lm needs the real MLX (it was run against MLX 0.29.3, 0.31.2, 0.32.0 and 0.32.4); it is left out except
    in its own test, which gives it a stand-in MLX."""
    monkeypatch.setattr(mt, "model_present", lambda repo: True)
    monkeypatch.setattr(mt, "fetch_model", lambda repo, size, stage=True: None)
    monkeypatch.setattr(mt, "_NLLB_FAILED", [])
    if "gets_a_stream_of_its_own" not in request.node.name:
        monkeypatch.setattr(mt, "_thread_stream", lambda: None)


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
