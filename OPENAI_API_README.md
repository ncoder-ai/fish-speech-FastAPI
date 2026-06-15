# Fish Speech S2-Pro — OpenAI-compatible TTS API

Production wrapper around `fishaudio/s2-pro` (4B Dual-AR TTS) that embeds the
project's own `ModelManager` / `TTSInferenceEngine` in a single FastAPI process
and exposes an **OpenAI `/v1/audio/speech`** endpoint with streaming, multiple
audio formats, inline **emotion control**, and voice cloning.

## What's installed

| Item | Value |
|------|-------|
| Model | `checkpoints/s2-pro` (4B) + `codec.pth` decoder |
| Env | `.venv` (Python 3.12, torch 2.8.0+cu128) |
| Server | `tools/openai_api_server.py` |
| Default device | `cuda:0` (most free VRAM) |
| Default port | `8770` (8080 was already taken) |
| Steady-state speed | RTF ≈ 1.1 on one RTX 3090 with `--compile` |
| VRAM | ~9 GB |

## Run

```bash
cd fish-speech-FastAPI
./run_openai_api.sh start      # detached, waits until ready
./run_openai_api.sh status
./run_openai_api.sh logs
./run_openai_api.sh stop
```

Config via env vars (see top of `run_openai_api.sh`): `FISH_DEVICE`,
`FISH_LISTEN`, `FISH_COMPILE`, `FISH_HALF`, `FISH_API_KEY`.

For boot persistence use the included `fish-speech-openai.service` systemd unit.

> First start with `--compile` spends ~2 min building CUDA graphs during warm-up.
> Set `FISH_COMPILE=0` for instant startup at ~5× slower generation.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/audio/speech` | OpenAI TTS (streaming + non-streaming) |
| POST | `/v1/tts` | Native JSON passthrough (full `ServeTTSRequest`) |
| GET  | `/v1/voices` | List cloned reference voices |
| POST | `/v1/voices` | Add a reference voice (multipart: `id`, `text`, `audio`) |
| DELETE | `/v1/voices/{id}` | Delete a reference voice |
| GET  | `/v1/models` | Model list |
| GET  | `/health` | Health + supported emotion tags |

**Live interactive reference** (auto-generated, always in sync, includes the
speaker/emotion/voice guide and `voice_map`): `http://<host>:8770/docs` (Swagger),
`/redoc`, and `/openapi.json`.

## OpenAI client usage

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8770/v1", api_key="not-needed")

# Non-streaming
client.audio.speech.create(
    model="tts-1", voice="nova", response_format="mp3",
    input="Hello from Fish Speech S2 Pro.",
).write_to_file("out.mp3")

# Streaming
with client.audio.speech.with_streaming_response.create(
    model="tts-1-hd", voice="alloy", response_format="wav",
    input="This is streamed chunk by chunk.",
) as r:
    for chunk in r.iter_bytes(4096):
        ...  # feed your player
```

`response_format`: `mp3 | opus | aac | flac | wav | pcm`
(`wav`/`pcm` give the lowest streaming latency — native chunks; compressed
formats stream through a live ffmpeg pipe).

## Emotion / prosody control

Embed inline tags **directly in the text** (`input`). S2-Pro supports 15k+
free-form tags, e.g.:

```json
{"input": "[whisper] I have a secret. [excited] But I can't wait to share it!",
 "response_format": "wav"}
```

Common tags: `[pause] [emphasis] [laughing] [chuckle] [sigh] [whisper]
[excited] [angry] [sad] [surprised] [shouting] [screaming] [low voice] [loud]
[delight] [panting] [crying] [singing] [clearing throat] [shocked]
[with strong accent]`. Free-form descriptions also work, e.g.
`[professional broadcast tone]`, `[pitch up]`. You may also pass OpenAI's
`instructions` field — it's prepended as a style tag.

## Multi-speaker scenes (one request — recommended)

Tag each turn with `<|speaker:N|>` inside `input` and send the **whole scene in a
single request**. The model keeps each speaker's voice consistent and paces the
turns naturally. **Do not** render one line per request and stitch the clips —
that accumulates silence between clips and sounds choppy.

```json
{
  "input": "<|speaker:0|>[calm] Where were you all night?\n<|speaker:1|>[nervous] I can explain, really.\n<|speaker:0|>[angry] Then explain.",
  "voice_map": {"0": "voice_a", "1": "voice_b"},
  "response_format": "mp3"
}
```

- `voice_map` maps each speaker id → a registered voice id (per-character voices
  in one request). Omit it to let the model auto-assign distinct, scene-consistent
  voices.
- Plain prose (no `<|speaker|>` tags) = one narrator, auto-chunked by
  `chunk_length` (raise it for fewer pauses in long narration).
- Emotion tags work per turn, exactly as above.

## Voice cloning & auto-registration

```bash
# Register a reference voice from a short clean sample + its transcript
curl -X POST http://localhost:8770/v1/voices \
  -F id=my_voice -F text="This is the transcript of the sample." \
  -F audio=@sample.wav

# Then use it via `voice` (any non-OpenAI name = reference id) or `voice_map`
curl -X POST http://localhost:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Now I speak in the cloned voice.","voice":"my_voice","response_format":"mp3"}' \
  -o cloned.mp3
```

**Auto-register from a folder:** set `VOICES_DIR=/path/to/voices` (env, or
`--voices-dir`). On startup and every `FISH_VOICES_SCAN_INTERVAL` seconds
(default 30) the server scans it and registers any new voices. Each voice is
either `‹id›.wav` (+ optional `‹id›.lab`/`‹id›.txt` transcript) or a subfolder
`‹id›/` containing audio + `.lab`. Already-registered ids are skipped.

Inline per-request references (base64 audio + text, one per speaker) are also
accepted via `/v1/tts`.

## Tuning knobs (extra JSON fields on `/v1/audio/speech`)

`temperature` (0.8), `top_p` (0.8), `repetition_penalty` (1.1),
`seed`, `chunk_length` (200), `max_new_tokens` (1024), `normalize` (true),
`speed` (0.25–4.0, time-stretch).

## Performance / VRAM knobs

- `FISH_QUANTIZE=int8|int4` — torchao weight-only quant (forces bf16). int8
  ≈ 12.8 GB & faster; int4 ≈ 11 GB. Off by default.
- `FISH_MAX_SEQ_LEN=4096` — smaller context = less KV cache / lower peak VRAM;
  long-form still works via the sliding window.
- `FISH_CONCURRENCY` (1) — concurrent synths; `FISH_QUEUE_TIMEOUT` (300 s) —
  requests wait this long for a slot then get a fast **503** (no unbounded hang).

## Notes / limits

- Generation is serialized through one model queue (`concurrency=1` by default);
  excess concurrent requests queue and then **503** after `queue_timeout`. For
  more throughput, run instances on other GPUs (`FISH_DEVICE`) behind a balancer.
- License: weights & code under the **Fish Audio Research License** (see
  `checkpoints/s2-pro/LICENSE.md`).
