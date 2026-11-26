# public-mini-projects

Public, finished or MVP projects made by me and Claude.

## Projects

| Project | What it does | Status |
|---|---|---|
| [`meeting-transcriber/`](meeting-transcriber/) | Record a meeting or load any audio/video file → full transcript with the **language of every line** → translation into any language on request | In use; see [`VERSION`](meeting-transcriber/VERSION) |

## meeting-transcriber

Speech to text for meetings in **several languages at once**, with translation. Speech recognition, speaker separation and
offline translation all run on your own computer; only translation with Claude sends text (never audio) over the internet.

```text
[00:00:04] [en] <You> Let's start with the budget.
[00:00:09] [ru] <Them> Хорошо, я подготовил цифры.
[00:00:15] [de?] <Them> Danke, das passt.
```

`[xx]` is the language of the line, `[xx?]` a low-confidence guess, `<You>` / `<Them>` / `<Speaker 1>` say who spoke.

**What it can do**

- Transcribe **any audio or video file** ffmpeg can read, or **record** from the microphone (up to 8 hours).
- Tag the **language of every line** automatically (about 100 languages), including meetings that switch languages.
- Record **online calls** as two tracks (your microphone and the sound the computer plays) and label `<You>` / `<Them>` (macOS app).
- Tell **several speakers** apart on one microphone (`--speakers auto`) and name them afterwards.
- **Translate** to any number of languages, with Claude or **offline** (NLLB-200; TranslateGemma 12B on Apple silicon), also live while recording.
- **Never lose work:** an interrupted run continues from its checkpoints.
- Use it as a **Mac app** (macOS 13+), from the **command line**, or from a Claude Code session.

**What it cannot do (the short list)**

- A language switch *inside* a sentence is not caught; offline translation is below Claude's quality and NLLB's licence is non-commercial.
- "Who spoke" is a guess from voices; the other side of a call is recorded only if you ask.
- The Mac app is macOS only; recording behaviour was measured on one Mac, and Windows is not tested.
- Output is plain text: no SRT/VTT subtitles, no summaries.

➡️ **[Read the full overview: features, possibilities, limits, requirements and licences](meeting-transcriber/OVERVIEW.md)**
· Command reference: [`meeting-transcriber/README.md`](meeting-transcriber/README.md)

### Quick start (command line)

```bash
cd meeting-transcriber
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 mt.py prepare                                   # once: checks ffmpeg, downloads the speech model
python3 mt.py transcribe call.m4a --to English          # → call.transcript.txt and call.English.txt
```

`--to` uses Claude and needs `ANTHROPIC_API_KEY`; add `--offline` to translate on your own computer instead (a 1.4 GB model is
downloaded on first use). The Mac app is built with `./meeting-transcriber/app/build_app.sh --install` (needs the Xcode Command
Line Tools); see the project README for the first-launch setup.

## How this repository is maintained

`meeting-transcriber/` is **copied automatically from a private development repository**. Please do not edit it here: the next
automatic copy overwrites it. Open an issue to suggest a change.

## Licence and consent

No licence is stated yet, so by default all rights are reserved. The speech and translation models the tool downloads have their
own licences (see section 8 of the overview). **Recording other people requires their consent in many places.**
