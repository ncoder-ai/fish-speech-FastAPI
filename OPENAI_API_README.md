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
cd /path/to/fish-speech
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

> **Transcripts matter — they are auto-generated if missing.** Voice cloning
> conditions on a `(transcript, audio)` pair, and **multi-speaker `voice_map`
> binding depends on it**: all per-speaker references are concatenated into one
> blob and the model uses each reference's *text* to tell the speakers apart. An
> empty transcript makes every speaker collapse to one voice. So when a voice is
> enrolled without a transcript (bare audio in the folder, or `POST /v1/voices`
> with no `text`), the server **auto-transcribes it with faster-whisper**
> (`FISH_AUTO_TRANSCRIBE=1`, model `FISH_ASR_MODEL=small`, on `FISH_ASR_DEVICE=cpu`).
> To backfill voices registered before this existed:
> `python tools/backfill_transcripts.py` (idempotent — only fills empty `.lab`s).

Inline per-request references (base64 audio + text, one per speaker) are also
accepted via `/v1/tts`.

## Tuning knobs (extra JSON fields on `/v1/audio/speech`)

`temperature` (0.8), `top_p` (0.8), `repetition_penalty` (1.1),
`seed`, `chunk_length` (200), `max_new_tokens` (1024), `normalize` (true),
`speed` (0.25–4.0, time-stretch).

## Performance / VRAM knobs

- `FISH_QUANTIZE=int8|int4` — torchao weight-only quant (forces bf16). int4 is
  the leaner option. On a 24 GB 3090: int8 ≈ 12.2 GB, int4 ≈ 11.1 GB at the
  default 8192 context, ≈ 9.85 GB at 4096. Only the backbone Linears are
  quantized (embeddings/head/codec stay full precision for audio quality), so
  the savings are modest. Off by default.
- `FISH_QUANTIZED_WEIGHTS=checkpoints/s2-pro/model.int4.g128.pt` — load
  **pre-quantized** weights instead of re-quantizing on every boot (model-load
  drops to ~5 s vs minutes; no CPU-RAM/swap pressure). See "Pre-quantized
  weights" below.
- `FISH_MAX_SEQ_LEN=4096` — smaller context = less KV cache / lower peak VRAM;
  long-form still works via the sliding window, but a ~400–500 word multi-speaker
  scene will trim older turns ~10× at 4096 (voices stay consistent; some
  long-range continuity is lost). Keep 8192 (default) for full-scene context.
- `FISH_TORCH_CACHE_HOST_DIR` (`./.torch-cache`) — persists the torch.compile /
  Triton kernel cache so a warm restart skips the ~4-min recompile (~44 s warm).
- `FISH_BATCH_SIZE` (1) — requests decoded together (continuous batching). Each
  request owns one KV-cache slot; one decode step serves every slot, and new
  requests join between steps. Decode is memory-bandwidth-bound, so a step for
  4 requests costs about the same as a step for 1. On an RTX 3090 (bf16):

  | Concurrent streams | Aggregate speed | Per-stream RTF | First audio |
  |---|---|---|---|
  | 1 (`FISH_BATCH_SIZE=1`, int8) | 1.5× real time | 0.66 | 1.2 s (later requests queue: 15 s at 4) |
  | 4 (`FISH_BATCH_SIZE=4`) | 4.5× real time | ~0.8 | ~1.6 s |
  | 6 (`FISH_BATCH_SIZE=6`) | 5.8× real time | ~0.95 | ~2.1 s |

  Use `FISH_QUANTIZE=none` with batching: torchao int8/int4 kernels are fused
  only at batch size 1 and dequantize every weight each step at larger sizes.
  VRAM grows ~1.2 GB per slot at `FISH_MAX_SEQ_LEN=8192`. `seed` is not
  reproducible while other requests share the batch.
- `FISH_CONCURRENCY` (= `FISH_BATCH_SIZE`) — concurrent synths;
  `FISH_QUEUE_TIMEOUT` (300 s) — requests wait this long for a slot then get a
  fast **503** (no unbounded hang).
- `FISH_CHUNK_PAUSE_MS` (400) — silence at each text-chunk boundary. Long input
  is generated in chunks (`chunk_length`), and each chunk ends ~0.15 s after
  its last word, so chunks joined edge to edge sound rushed next to the ~0.4 s
  the model leaves between sentences within a chunk. The engine measures each
  chunk's trailing silence and pads up to this target. `0` disables.
- `FISH_SPLIT_SPEAKERS` (1) — in multi-speaker input, start a new generation at
  every speaker change (consecutive lines by one speaker stay together). With
  several speakers in one generation, the model drifts other speakers toward the
  voice it opened with; a 3-voice scene had wrong-voice lines in 10/24 renders
  versus 0/24 with this on, at no speed cost. `0` restores the old grouping.
- `FISH_RAS` (1) — Repetition Aware Sampling; `0` disables it (diagnostics).
- `FISH_REF_CACHE` (on) — reuse encoded reference voices across requests. The
  voice add/delete endpoints invalidate it; set `off` if you edit files under
  `references/` by hand.

## Pre-quantized weights (quantize once; persist; move between boxes)

Quantizing at load runs torchao in CPU RAM **every boot** (slow; swap-thrashes
low-RAM hosts). Instead, quantize once and load the saved tensors:

```bash
# Generate once (uses this box's torch/torchao/GPU). int4 or int8.
FISH_QUANTIZE=int4 DEV=cuda:0 \
  .venv/bin/python tools/quantize_save.py checkpoints/s2-pro \
  checkpoints/s2-pro/model.int4.g128.pt

# Then point the server at it:
export FISH_QUANTIZE=int4
export FISH_QUANTIZED_WEIGHTS=checkpoints/s2-pro/model.int4.g128.pt
./run_openai_api.sh start
```

- **Auto-generate:** if `FISH_QUANTIZED_WEIGHTS` is set but the file is missing,
  `run_openai_api.sh` (and the container entrypoint) generates it once on
  startup, then loads it. So a fresh box self-provisions on first boot.
- **Persistence:** the file lives under `checkpoints/` (a **mounted volume** in
  the container — `FISH_CHECKPOINTS_HOST_DIR`), never baked into the image. It
  survives rebuilds, restarts, and reboots. Deleting the checkpoints dir is the
  only way to lose it.
- **Different box / sharing:** the `.pt` is **coupled to torch/torchao + GPU
  arch** (e.g. built with torch 2.8 + torchao 0.17 on Ampere). For a box with
  the **same stack**, copy the file (rsync, or a shared NFS/object store) to skip
  the one-time generation. For a **different stack**, don't copy it — let the box
  regenerate (just set the two env vars; first boot builds a compatible file).
  Regenerate after bumping torch/torchao.

## Notes / limits

- Generation is serialized through one model queue (`concurrency=1` by default);
  excess concurrent requests queue and then **503** after `queue_timeout`. For
  more throughput, run instances on other GPUs (`FISH_DEVICE`) behind a balancer.
- License: weights & code under the **Fish Audio Research License** (see
  `checkpoints/s2-pro/LICENSE.md`).
