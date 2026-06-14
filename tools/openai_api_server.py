"""
OpenAI-compatible FastAPI server for Fish Speech / OpenAudio S2-Pro.

Embeds the project's own (tested) ModelManager + TTSInferenceEngine in a single
process and exposes:

  POST /v1/audio/speech   - OpenAI-compatible TTS (streaming + non-streaming)
  POST /v1/tts            - Native JSON TTS (full ServeTTSRequest surface)
  GET  /v1/voices         - List reference voices (for cloning)
  POST /v1/voices         - Add a reference voice (multipart: id, audio, text)
  DELETE /v1/voices/{id}  - Delete a reference voice
  GET  /v1/models         - OpenAI-style model list
  GET  /health            - Health probe

Emotion / prosody control is inline in the text via tags such as
  [angry] [whisper] [excited] [sad] [laughing] [pause] [shouting] ...
(15k+ free-form tags supported by S2-Pro). Pass them right inside `input`.

Run:
  python tools/openai_api_server.py \
      --llama-checkpoint-path checkpoints/s2-pro \
      --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
      --decoder-config-name modded_dac_vq \
      --device cuda:3 --half --listen 0.0.0.0:8080
"""

import argparse
import asyncio
import io
import os
import re
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Generator, List, Optional

import numpy as np
import pyrootutils
import soundfile as sf
import uvicorn

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from tools.server.model_manager import ModelManager

# ----------------------------------------------------------------------------
# Globals (populated at startup)
# ----------------------------------------------------------------------------
ARGS: argparse.Namespace = None  # type: ignore
MODEL_MANAGER: Optional[ModelManager] = None

# Serializes GPU work across requests. The LLAMA worker is a single thread, so
# letting many requests fan out only thrashes VRAM and the executor pool and
# makes everything appear to hang. We gate synthesis behind this semaphore
# (default size 1) and fail fast with 503 once a request has waited too long,
# turning an unbounded hang into a clear, retryable error.
SYNTH_SEM: Optional[asyncio.Semaphore] = None

# OpenAI default voice names -> treated as "model default speaker" unless a
# reference voice with the same id has been registered.
_OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash",
                  "ballad", "coral", "sage", "verse", "default", ""}

# OpenAI response_format -> internal handling
_PCM_RATE_FALLBACK = 44100

# Fish-Speech S2-Pro emotion / prosody / tone tags (inline in `input`).
# S2-Pro accepts 15k+ free-form tags; these are the documented built-ins.
EMOTION_TAGS = [
    # pacing / breath
    "[pause]", "[short pause]", "[inhale]", "[exhale]", "[panting]",
    # laughter / amusement
    "[laughing]", "[laughing tone]", "[chuckle]", "[chuckling]",
    "[audience laughter]", "[tsk]",
    # core emotions
    "[excited]", "[excited tone]", "[angry]", "[sad]", "[delight]",
    "[surprised]", "[shocked]", "[moved]", "[crying]", "[moaning]",
    # emphasis / delivery
    "[emphasis]", "[whisper]", "[low voice]", "[low volume]", "[loud]",
    "[shouting]", "[screaming]", "[sigh]", "[singing]", "[interrupting]",
    "[clearing throat]", "[with strong accent]",
    # volume / effects
    "[volume up]", "[volume down]", "[echo]",
]


# ----------------------------------------------------------------------------
# OpenAI request schema
# ----------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    model: str = "s2-pro"
    input: str = Field(..., description="Text to synthesize. Inline emotion tags allowed, e.g. [whisper] hello.")
    voice: str = "default"
    response_format: str = Field("mp3", description="mp3|opus|aac|flac|wav|pcm")
    speed: float = Field(1.0, ge=0.25, le=4.0)
    # stream is accepted both as `stream` and OpenAI's newer `stream_format`
    stream: bool = False
    stream_format: Optional[str] = None

    # ---- Fish-Speech extras (optional, ignored by vanilla OpenAI clients) ----
    instructions: Optional[str] = None  # OpenAI field; prepended as a tag if given
    reference_id: Optional[str] = None
    seed: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    chunk_length: int = 200
    # 0 = generate the WHOLE input and stop at the model's natural end of speech
    # (bounded by the 32k context). A non-zero value HARD-CAPS the audio length
    # (1024 tokens is only ~47s), which silently truncates longer text.
    max_new_tokens: int = 0
    normalize: bool = True


# ----------------------------------------------------------------------------
# Audio encoding helpers
# ----------------------------------------------------------------------------
def _sample_rate(engine) -> int:
    dm = engine.decoder_model
    if hasattr(dm, "spec_transform"):
        return int(dm.spec_transform.sample_rate)
    return int(getattr(dm, "sample_rate", _PCM_RATE_FALLBACK))


def _sf_format(fmt: str) -> Optional[str]:
    return {"wav": "WAV", "flac": "FLAC", "pcm": "RAW"}.get(fmt)


_FFMPEG_CODEC = {
    "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "96k"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "flac"],
    "wav": ["-f", "wav"],
}


def _content_type(fmt: str) -> str:
    return {
        "mp3": "audio/mpeg",
        "opus": "audio/ogg",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/L16",
    }.get(fmt, "application/octet-stream")


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def _apply_speed(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """Resample-based time-stretch fallback for `speed` (pitch shifts slightly)."""
    if abs(speed - 1.0) < 1e-3:
        return audio
    try:
        import librosa

        return librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
    except Exception as e:  # pragma: no cover - best effort
        logger.warning(f"speed adjust failed ({e}); returning unmodified audio")
        return audio


def encode_full(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    """Encode a complete float32 [-1,1] mono array to the requested format."""
    if fmt == "pcm":
        return _float_to_pcm16(audio)
    sff = _sf_format(fmt)
    if sff in ("WAV", "FLAC"):
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=sff)
        return buf.getvalue()
    # compressed formats via ffmpeg
    return _ffmpeg_encode(_float_to_pcm16(audio), sr, fmt)


def _ffmpeg_encode(pcm16: bytes, sr: int, fmt: str) -> bytes:
    codec = _FFMPEG_CODEC.get(fmt, _FFMPEG_CODEC["mp3"])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           *codec, "pipe:1"]
    proc = subprocess.run(cmd, input=pcm16, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode()[:500]}")
    return proc.stdout


# ----------------------------------------------------------------------------
# Core synthesis -> byte stream
# ----------------------------------------------------------------------------
def _build_tts_request(r: SpeechRequest, text: str, streaming: bool,
                       seed=None, max_new_tokens=None) -> ServeTTSRequest:
    reference_id = r.reference_id
    if reference_id is None and r.voice and r.voice.lower() not in _OPENAI_VOICES:
        # Non-OpenAI voice name -> treat as a registered reference id.
        reference_id = r.voice

    # Fish internal format is wav/pcm/mp3/opus; we encode ourselves, so ask the
    # engine for raw audio (format="wav") and never let it constrain us.
    return ServeTTSRequest(
        text=text,
        chunk_length=max(100, min(1000, r.chunk_length)),
        format="wav",
        references=[],
        reference_id=reference_id,
        seed=seed if seed is not None else r.seed,
        normalize=r.normalize,
        streaming=streaming,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else r.max_new_tokens,
        top_p=r.top_p,
        repetition_penalty=r.repetition_penalty,
        temperature=r.temperature,
    )


def _iter_segments(
    req: ServeTTSRequest, sr: int, cancel_event: Optional[threading.Event] = None
) -> Generator[np.ndarray, None, None]:
    """Low-level: yield float32 audio segments from the engine for one request."""
    engine = MODEL_MANAGER.tts_inference_engine
    for result in engine.inference(req, cancel_event=cancel_event):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code == "segment" and isinstance(result.audio, tuple):
            yield result.audio[1].astype(np.float32)
        elif result.code == "final" and isinstance(result.audio, tuple):
            # In non-streaming mode the only audio comes through "final".
            if not req.streaming:
                yield result.audio[1].astype(np.float32)


# Split into sentences/lines, then regroup up to ~max_chars. Long-form text MUST
# be chunked: a single huge generation (a) can run away to the context cap and
# (b) forces one giant DAC decode that OOMs. Per-chunk generation stops at each
# sentence's natural end and keeps the decode small.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？…])\s+|\n+")
_PER_CHUNK_MAX_TOKENS = 1024  # ~45s of audio per chunk


def _split_text(text: str, max_chars: int = 200) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    pieces = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    chunks: List[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip() if cur else p
    if cur:
        chunks.append(cur)
    # Hard-split any monster piece with no sentence breaks.
    out: List[str] = []
    for c in chunks:
        while len(c) > max_chars * 2:
            out.append(c[: max_chars * 2])
            c = c[max_chars * 2:]
        out.append(c)
    return out


_SPEAKER_TAG = "<|speaker:0|>"
_SPEAKER_RE = re.compile(r"<\|speaker:\d+\|>")


def _iter_chunk_audio(
    r: SpeechRequest, sr: int, cancel_event: Optional[threading.Event] = None
) -> Generator[np.ndarray, None, None]:
    """Yield float32 audio per batch using the framework's OWN batched
    generation. Fish-Speech 2.0 only chunks text that carries <|speaker:X|>
    turns; for plain prose it would otherwise run the whole input as ONE giant
    generation + decode (runaway + OOM). So:

    - If the input ALREADY has <|speaker:N|> turns (multi-speaker dialogue),
      pass it straight through -- generate_long groups the turns by
      `chunk_length` and each speaker keeps its own voice.
    - Otherwise tag every sentence with a single speaker so the same batching
      kicks in for a single narrator.

    Either way each batch is generated against the SHARED running conversation
    -> bounded per-batch decode AND a consistent voice (no seed/reference
    hacks)."""
    text = r.input
    if r.instructions:
        text = f"[{r.instructions.strip().strip('[]')}] {text}"

    if _SPEAKER_RE.search(text):
        tagged = text  # caller drives the speaker turns
    else:
        sentences = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p and p.strip()]
        if not sentences:
            return
        tagged = "\n".join(f"{_SPEAKER_TAG}{s}" for s in sentences)

    # streaming=True makes the engine decode + emit each batch separately
    # (bounded VRAM); max_new_tokens=0 lets each short batch end naturally.
    req = _build_tts_request(r, text=tagged, streaming=True, max_new_tokens=0)
    for seg in _iter_segments(req, sr, cancel_event=cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            break
        yield seg


async def _stream_bytes(
    r: SpeechRequest, fmt: str, cancel_event: threading.Event
) -> AsyncGenerator[bytes, None]:
    """Stream encoded audio bytes for one request.

    Robustness contract:
    - If the client disconnects, the generator's finally fires, `cancel_event`
      is set, the GPU producer stops after its current batch, and any ffmpeg
      child is killed -- nothing is left running.
    - The compressed (ffmpeg) path reads stdout on a dedicated task while a
      separate task feeds stdin, so the pipe can never deadlock when generation
      outpaces draining.
    """
    sr = _sample_rate(MODEL_MANAGER.tts_inference_engine)
    speed_native = abs(r.speed - 1.0) < 1e-3

    loop = asyncio.get_event_loop()

    # Run blocking generation in a thread, push audio segments through a queue.
    q: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for seg in _iter_chunk_audio(r, sr, cancel_event=cancel_event):
                if cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(q.put_nowait, ("seg", seg))
        except Exception as e:  # noqa
            # A cancelled request can surface as "No audio generated" when it is
            # dropped before the first batch finishes -- that is expected, not a
            # failure, so don't spam a traceback or forward it as an error.
            if cancel_event.is_set():
                logger.info(f"producer stopped after cancellation ({e})")
            else:
                logger.exception("synthesis producer failed")
                loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("end", None))

    fut = loop.run_in_executor(None, producer)

    try:
        if fmt in ("wav", "pcm"):
            # Native low-latency path: WAV header (once) + PCM16 chunks.
            if fmt == "wav":
                from fish_speech.inference_engine.utils import wav_chunk_header

                yield bytes(wav_chunk_header(sample_rate=sr))
            while True:
                kind, payload = await q.get()
                if kind == "seg":
                    seg = payload if speed_native else _apply_speed(payload, r.speed, sr)
                    yield _float_to_pcm16(seg)
                elif kind == "err":
                    # Headers are already sent; we cannot change the status code
                    # mid-stream, so log and end cleanly rather than hang.
                    logger.error(f"streaming synthesis error: {payload}")
                    break
                else:
                    break
        else:
            async for chunk in _stream_compressed(q, fmt, sr, r, speed_native):
                yield chunk
    finally:
        # Client gone / stream ended: stop the GPU producer and wait for it to
        # unwind so we do not release the synthesis slot while work is still
        # running on the device.
        cancel_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("producer did not stop within 120s after cancel")
        except Exception:
            pass


async def _stream_compressed(q, fmt, sr, r, speed_native):
    """ffmpeg-backed streaming for mp3/opus/aac/flac with no pipe deadlock."""
    codec = _FFMPEG_CODEC.get(fmt, _FFMPEG_CODEC["mp3"])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           *codec, "pipe:1"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Drain stderr concurrently so ffmpeg never blocks writing diagnostics.
    async def _drain_stderr():
        try:
            data = await proc.stderr.read()
            if data:
                logger.debug(f"ffmpeg stderr: {data.decode(errors='replace')[:500]}")
        except Exception:
            pass

    # Feed PCM into ffmpeg on its own task, decoupled from stdout reading.
    async def _feed():
        try:
            while True:
                kind, payload = await q.get()
                if kind == "seg":
                    seg = payload if speed_native else _apply_speed(payload, r.speed, sr)
                    proc.stdin.write(_float_to_pcm16(seg))
                    await proc.stdin.drain()
                elif kind == "err":
                    logger.error(f"streaming synthesis error: {payload}")
                    break
                else:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logger.exception("ffmpeg feeder failed")
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    stderr_task = asyncio.create_task(_drain_stderr())
    feed_task = asyncio.create_task(_feed())

    try:
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        for t in (feed_task, stderr_task):
            t.cancel()
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            await proc.wait()
        except Exception:
            pass


def _synthesize_full(r: SpeechRequest, fmt: str) -> bytes:
    sr = _sample_rate(MODEL_MANAGER.tts_inference_engine)
    segs = list(_iter_chunk_audio(r, sr))
    if not segs:
        raise HTTPException(500, "No audio generated; check the input text.")
    audio = np.concatenate(segs, axis=0)
    audio = _apply_speed(audio, r.speed, sr)
    return encode_full(audio, sr, fmt)


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_MANAGER, SYNTH_SEM
    # Create the semaphore inside the running loop.
    SYNTH_SEM = asyncio.Semaphore(max(1, ARGS.concurrency))

    # Pass quant / context-window knobs to the engine via env (read in init_model).
    use_half = ARGS.half
    if ARGS.quantize and ARGS.quantize != "none":
        os.environ["FISH_QUANTIZE"] = ARGS.quantize
        if ARGS.quantize == "int4" and ARGS.half:
            logger.warning("int4 requires bf16; overriding --half -> running bf16.")
            use_half = False
    if ARGS.max_seq_len and ARGS.max_seq_len > 0:
        os.environ["FISH_MAX_SEQ_LEN"] = str(ARGS.max_seq_len)

    logger.info("Loading models (this can take a minute, includes warm-up)...")
    MODEL_MANAGER = ModelManager(
        mode="tts",
        device=ARGS.device,
        half=use_half,
        compile=ARGS.compile,
        llama_checkpoint_path=ARGS.llama_checkpoint_path,
        decoder_checkpoint_path=ARGS.decoder_checkpoint_path,
        decoder_config_name=ARGS.decoder_config_name,
    )
    logger.info(
        f"Ready on http://{ARGS.listen}  device={ARGS.device}  "
        f"concurrency={ARGS.concurrency}  queue_timeout={ARGS.queue_timeout}s  "
        f"quantize={ARGS.quantize}  max_seq_len={ARGS.max_seq_len or 'default'}"
    )
    yield
    MODEL_MANAGER = None


async def _acquire_slot():
    """Acquire the synthesis slot or fail fast.

    Waits up to ARGS.queue_timeout for the GPU to free up; on timeout returns a
    clear 503 instead of letting the request hang indefinitely behind a long
    in-flight generation.
    """
    try:
        await asyncio.wait_for(SYNTH_SEM.acquire(), timeout=ARGS.queue_timeout)
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy; retry shortly")


app = FastAPI(title="Fish Speech S2-Pro OpenAI API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


async def _check_auth(request: Request):
    if ARGS.api_key:
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else auth
        if token != ARGS.api_key:
            raise HTTPException(401, "Invalid API key")


@app.get("/health")
async def health():
    ready = MODEL_MANAGER is not None
    return {"status": "ok" if ready else "loading", "model": "s2-pro",
            "device": ARGS.device, "emotion_tags": EMOTION_TAGS}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "s2-pro", "object": "model", "owned_by": "fishaudio"},
        {"id": "tts-1", "object": "model", "owned_by": "fishaudio"},
        {"id": "tts-1-hd", "object": "model", "owned_by": "fishaudio"},
    ]}


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, body: SpeechRequest):
    await _check_auth(request)
    if MODEL_MANAGER is None:
        raise HTTPException(503, "Model still loading")

    fmt = (body.response_format or "mp3").lower()
    if fmt not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
        raise HTTPException(400, f"Unsupported response_format: {fmt}")

    want_stream = bool(body.stream or (body.stream_format == "audio"))

    # Acquire the synthesis slot up front (may 503 fast). For streaming we hold
    # it across the whole response and release in the generator's finally.
    await _acquire_slot()

    if want_stream:
        cancel_event = threading.Event()

        async def guarded():
            try:
                async for chunk in _stream_bytes(body, fmt, cancel_event):
                    yield chunk
            finally:
                cancel_event.set()
                SYNTH_SEM.release()

        return StreamingResponse(
            guarded(),
            media_type=_content_type(fmt),
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    # Non-streaming: run blocking synth in a thread.
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_full, body, fmt
        )
    finally:
        SYNTH_SEM.release()
    return StreamingResponse(iter([data]), media_type=_content_type(fmt))


def _synthesize_native(req: ServeTTSRequest, fmt: str) -> bytes:
    sr = _sample_rate(MODEL_MANAGER.tts_inference_engine)
    segs = list(_iter_segments(req, sr))
    if not segs:
        raise HTTPException(500, "No audio generated")
    audio = np.concatenate(segs, axis=0)
    return encode_full(audio, sr, fmt)


@app.post("/v1/tts")
async def native_tts(request: Request, body: dict = Body(...)):
    """Native ServeTTSRequest passthrough (JSON). Returns audio bytes."""
    await _check_auth(request)
    if MODEL_MANAGER is None:
        raise HTTPException(503, "Model still loading")
    req = ServeTTSRequest(**body)
    fmt = req.format if req.format in ("wav", "pcm", "mp3", "flac") else "wav"
    # Run the blocking generation off the event loop so it does not stall other
    # requests / health checks, and gate it behind the synthesis slot.
    await _acquire_slot()
    try:
        data = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_native, req, fmt
        )
    finally:
        SYNTH_SEM.release()
    return StreamingResponse(iter([data]), media_type=_content_type(fmt))


@app.get("/v1/voices")
async def list_voices(request: Request):
    await _check_auth(request)
    engine = MODEL_MANAGER.tts_inference_engine
    try:
        ids = engine.list_reference_ids()
    except Exception:
        ids = []
    return {"voices": ids, "default_openai_voices": sorted(_OPENAI_VOICES - {""})}


@app.post("/v1/voices")
async def add_voice(request: Request, id: str = Form(...),
                    text: str = Form(...), audio: UploadFile = File(...)):
    await _check_auth(request)
    engine = MODEL_MANAGER.tts_inference_engine
    content = await audio.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(content)
        path = f.name
    try:
        engine.add_reference(id, path, text)
    finally:
        os.unlink(path)
    return {"success": True, "voice": id}


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(request: Request, voice_id: str):
    await _check_auth(request)
    engine = MODEL_MANAGER.tts_inference_engine
    engine.delete_reference(voice_id)
    return {"success": True, "voice": voice_id}


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-checkpoint-path", default="checkpoints/s2-pro")
    p.add_argument("--decoder-checkpoint-path", default="checkpoints/s2-pro/codec.pth")
    p.add_argument("--decoder-config-name", default="modded_dac_vq")
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--listen", default="0.0.0.0:8080")
    p.add_argument("--api-key", default=os.environ.get("FISH_API_KEY"))
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("FISH_CONCURRENCY", "1")),
        help="Max simultaneous synthesis requests (GPU worker is single-threaded; "
             "keep at 1 unless you know you have headroom).",
    )
    p.add_argument(
        "--queue-timeout",
        type=float,
        default=float(os.environ.get("FISH_QUEUE_TIMEOUT", "300")),
        help="Seconds a request waits for a synthesis slot before returning 503.",
    )
    p.add_argument(
        "--quantize",
        choices=["none", "int8", "int4"],
        default=os.environ.get("FISH_QUANTIZE", "none") or "none",
        help="Weight-only quantization of the slow backbone (torchao). int8 ~"
             "12.8GB & faster; int4 ~11GB & fastest but needs bf16 (drop --half). "
             "Off by default.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=int(os.environ.get("FISH_MAX_SEQ_LEN", "0") or "0"),
        help="Override the model context window (default 8192). Smaller = less "
             "KV cache / lower peak VRAM; long-form still works via the sliding "
             "window. e.g. 4096. 0 = keep model default.",
    )
    return p.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    host, port = ARGS.listen.rsplit(":", 1)
    uvicorn.run(app, host=host, port=int(port), log_level="info")
