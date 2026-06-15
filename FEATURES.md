# Fish Speech S2-Pro — Feature Guide

Expressive control, multi-speaker dialogue, and voice cloning through the
OpenAI-compatible server (`tools/openai_api_server.py`). All examples below were
verified against the running server on `cuda:3 : 8770`.

Base URL used throughout: `http://localhost:8770`. If you set `FISH_API_KEY`,
add `-H "Authorization: Bearer $FISH_API_KEY"`.

> **How long text is handled (why it's robust):** S2-Pro 2.0 only chunks text
> that carries `<|speaker:N|>` turns. The server therefore tags each sentence
> with a speaker so the model's own `generate_long` batches it by `chunk_length`
> and generates each batch against the **shared running conversation** — giving
> a *consistent voice across the whole input* with *bounded per-batch VRAM*
> (no truncation, no OOM). Context budget is ~6k tokens (~5 min of audio) per
> request at the configured `max_seq_len=8192`.

---

## 0. OpenAI-client compatibility matrix

"OpenAI-compatible" here means it works through an **unmodified** OpenAI client
(verified against the official `openai` Python SDK pointed at this server). Every
*generation* feature rides inside the standard `input` text or the `voice` field,
so no custom client is needed. Only voice *enrollment* uses a side endpoint that
OpenAI's API has no concept of.

| Feature | Works via OpenAI `/v1/audio/speech`? | How (vanilla OpenAI client) | Verified |
|---|---|---|---|
| Emotion / prosody tags | ✅ Fully | `[tags]` inline in `input` | ✅ |
| `instructions` style (OpenAI field) | ✅ Fully | `instructions="excited"` → leading style tag | ✅ |
| Multi-speaker dialogue | ✅ Fully | `<\|speaker:N\|>` inline in `input` | ✅ |
| Use a cloned voice | ✅ Fully | `voice="narrator"` (any custom string) | ✅ |
| Streaming | ✅ Fully | `with_streaming_response.create(...)` / `stream=True` | ✅ |
| Audio formats | ✅ Fully | `response_format` = mp3/opus/aac/flac/wav/pcm | ✅ |
| Fish knobs (seed, temperature, top_p, repetition_penalty, reference_id) | ⚠️ Via `extra_body` | `extra_body={"seed":123, ...}` | ✅ |
| **Register/enroll a clone** | ❌ Not OpenAI | one-time `POST /v1/voices` (multipart upload) | ✅ |
| One-shot inline reference audio | ❌ Not OpenAI | native `POST /v1/tts` with `references=[...]` | ✅ |

**Bottom line:** all generation features work through a stock OpenAI client; only
voice *enrollment* (uploading reference audio) needs the one-time non-OpenAI
`/v1/voices` call. After enrollment you use the clone through the OpenAI client
via `voice="…"`.

Minimal proof (official SDK):
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8770/v1", api_key="x")
c.audio.speech.create(model="tts-1", voice="default",
    input="[whisper] a secret. [excited] now I am thrilled!",
    response_format="mp3").write_to_file("emotion.mp3")
c.audio.speech.create(model="tts-1", voice="narrator",      # a cloned voice
    input="<|speaker:0|>Ready? <|speaker:1|>Ready when you are.",
    response_format="mp3", extra_body={"seed": 123}).write_to_file("mix.mp3")
```

---

## 1. Emotion & prosody control

Emotion is **inline in the text** — just embed `[tag]` markers anywhere in your
`input`. S2-Pro supports 15,000+ free-form tags, so beyond the built-ins you can
write natural-language directions like `[professional broadcast tone]`,
`[whisper in a small voice]`, or `[pitch up]`.

**Built-in tags** (also returned by `GET /health`):

| Category | Tags |
|---|---|
| Pacing / breath | `[pause]` `[short pause]` `[inhale]` `[exhale]` `[panting]` |
| Laughter | `[laughing]` `[laughing tone]` `[chuckle]` `[chuckling]` `[audience laughter]` `[tsk]` |
| Emotion | `[excited]` `[excited tone]` `[angry]` `[sad]` `[delight]` `[surprised]` `[shocked]` `[moved]` `[crying]` `[moaning]` |
| Delivery | `[emphasis]` `[whisper]` `[low voice]` `[low volume]` `[loud]` `[shouting]` `[screaming]` `[sigh]` `[singing]` `[interrupting]` `[clearing throat]` `[with strong accent]` |
| Volume / FX | `[volume up]` `[volume down]` `[echo]` |

**Example (curl):**
```bash
curl -s http://localhost:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "input": "[whisper] I have a secret to tell you. [excited] But I cannot wait to share it with everyone! [laughing] It is going to be wonderful.",
        "response_format": "mp3"
      }' -o emotion.mp3
```

**OpenAI SDK** — the standard `instructions` field also works; it is prepended
as a style tag for the whole utterance:
```python
client.audio.speech.create(
    model="tts-1", voice="default", response_format="mp3",
    instructions="excited",          # -> applied as a leading [excited] tag
    input="We just shipped the release!",
).write_to_file("excited.mp3")
```

Tips:
- Tags affect the words that follow them, until the next tag.
- Combine freely: `[sad] ... [sigh] ... [low voice] ...`.
- Keep `normalize: true` (default) for clean numbers/dates; it does not strip tags.

---

## 2. Multi-speaker dialogue

Mark each turn with `<|speaker:N|>` (N = 0,1,2,…). The server detects these
tags and passes the text straight through, so each speaker keeps a **distinct,
consistent voice** across the whole dialogue (up to 5 speakers per batch).

**Example (curl):**
```bash
curl -s http://localhost:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
        "input": "<|speaker:0|>Did you finish the cabinet? <|speaker:1|>Almost. The left door still sticks a little. <|speaker:0|>Let me grab the plane, we will fix it in a minute.",
        "response_format": "mp3"
      }' -o dialogue.mp3
```

Notes:
- Emotion tags work **inside** speaker turns:
  `<|speaker:1|>[sigh] Fine, you were right.`
- Without references, each speaker id gets its own model-chosen voice that stays
  stable for that id throughout the request.

**Pin specific cloned voices to speakers — `voice_map`:** map each speaker id to
a registered voice id in one request (no per-line stitching):
```json
{
  "input": "<|speaker:0|>[low voice] Where were you?\n<|speaker:1|>I can explain.",
  "voice_map": {"0": "grace2", "1": "david_attenborough_cc3"},
  "response_format": "wav"
}
```

> **⚠️ Each mapped voice MUST have a transcript.** voice_map concatenates every
> speaker's reference audio into one blob; the model separates the speakers using
> each reference's *text*. If a referenced voice has an empty `.lab` the speakers
> collapse to a single voice (usually speaker 0). The server auto-transcribes
> voices that lack a transcript (see §3), so this normally just works — but if you
> add voices manually, give them a transcript or run
> `python tools/backfill_transcripts.py`.

---

## 3. Voice cloning (custom voices)

Provide a short, clean reference clip (~5–15 s) plus its exact transcript. The
transcript is **not optional for quality** — it anchors the clone and is what
makes multi-speaker `voice_map` work (§2). If you omit it, the server
auto-transcribes the clip with faster-whisper (`FISH_AUTO_TRANSCRIBE=1`), so
enrollment never leaves a voice with an empty transcript. Two ways:

### A. Register a reusable voice, then reference it by name

```bash
# Register
curl -s http://localhost:8770/v1/voices \
  -F id=narrator \
  -F text="The workshop smelled of cedar and warm varnish." \
  -F audio=@reference.wav

# Use it — any non-OpenAI `voice` name is treated as a registered voice id
curl -s http://localhost:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This new line is spoken in the cloned narrator voice.",
       "voice":"narrator","response_format":"mp3"}' -o cloned.mp3
```

Manage voices:
```bash
curl -s http://localhost:8770/v1/voices            # list
curl -s -X DELETE http://localhost:8770/v1/voices/narrator   # delete
```

### B. One-shot inline reference (no registration) — native endpoint

The native `/v1/tts` endpoint accepts a reference per request. `audio` is the
raw bytes (base64 is auto-decoded):
```python
import requests, base64
ref = base64.b64encode(open("reference.wav","rb").read()).decode()
r = requests.post("http://localhost:8770/v1/tts", json={
    "text": "Speak this in the reference voice.",
    "format": "mp3",
    "references": [{"audio": ref, "text": "Transcript of the reference clip."}],
})
open("cloned.mp3","wb").write(r.content)
```

Reference-quality tips:
- 5–15 s, single speaker, minimal background noise/music.
- The `text` must match what is actually said in the clip.
- WAV/mp3/flac all accepted; mono is fine.
- Re-use is cached, so repeated calls with the same registered voice are fast.

---

## 4. Combining features

Everything composes — clone a voice, drive emotion inline, across speakers:
```json
{
  "input": "<|speaker:0|>[excited] The grant came through! <|speaker:1|>[whisper] Keep it down, nobody else knows yet.",
  "response_format": "mp3"
}
```
For a single cloned narrator reading long expressive text, just set
`"voice":"narrator"` and sprinkle `[tags]` through the `input`.

---

## 5. Request parameters (`POST /v1/audio/speech`)

| Field | Default | Notes |
|---|---|---|
| `input` | — | Text; supports `[emotion]` tags and `<|speaker:N|>` turns |
| `voice` | `default` | OpenAI names (`alloy`,…) → model default voice; any other name → registered clone id |
| `response_format` | `mp3` | `mp3 \| opus \| aac \| flac \| wav \| pcm` (wav/pcm = lowest streaming latency) |
| `stream` | `false` | Chunked streaming (wav/pcm native; mp3/opus via live ffmpeg pipe) |
| `instructions` | — | OpenAI field; applied as a leading emotion/style tag |
| `speed` | `1.0` | 0.25–4.0 (time-stretch) |
| `temperature` | `0.8` | Sampling temperature |
| `top_p` | `0.8` | Nucleus sampling |
| `repetition_penalty` | `1.1` | Repetition penalty |
| `seed` | — | Fix for reproducible output |
| `reference_id` | — | Explicit registered voice id (alternative to `voice`) |

Endpoints: `POST /v1/audio/speech`, `POST /v1/tts`, `GET/POST /v1/voices`,
`DELETE /v1/voices/{id}`, `GET /v1/models`, `GET /health`. See
[OPENAI_API_README.md](OPENAI_API_README.md) for run/deploy details.
