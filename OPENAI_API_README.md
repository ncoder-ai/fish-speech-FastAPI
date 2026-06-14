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
| Default device | `cuda:3` (most free VRAM) |
| Default port | `8770` (8080 was already taken) |
| Steady-state speed | RTF ≈ 1.1 on one RTX 3090 with `--compile` |
| VRAM | ~9 GB |

## Run

```bash
cd /home/nishant/App/fish-speech
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

## Voice cloning

```bash
# Register a reference voice from a short clean sample + its transcript
curl -X POST http://localhost:8770/v1/voices \
  -F id=my_voice -F text="This is the transcript of the sample." \
  -F audio=@sample.wav

# Then synthesize with it via the `voice` field (any non-OpenAI name = reference id)
curl -X POST http://localhost:8770/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Now I speak in the cloned voice.","voice":"my_voice","response_format":"mp3"}' \
  -o cloned.mp3
```

Or pass inline references per-request via `/v1/tts` with base64 `references`.

## Tuning knobs (extra JSON fields on `/v1/audio/speech`)

`temperature` (0.8), `top_p` (0.8), `repetition_penalty` (1.1),
`seed`, `chunk_length` (200), `max_new_tokens` (1024), `normalize` (true),
`speed` (0.25–4.0, time-stretch).

## Notes / limits

- Generation is serialized through one model queue → concurrent requests queue
  up. For higher throughput run more instances on other GPUs (`FISH_DEVICE`) +
  a load balancer, or use SGLang/vLLM-Omni recipes from the upstream README.
- License: weights & code under the **Fish Audio Research License** (see
  `checkpoints/s2-pro/LICENSE.md`).
