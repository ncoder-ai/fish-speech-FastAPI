"""Lazy faster-whisper transcriber for reference-voice enrollment.

WHY THIS EXISTS
---------------
Fish voice cloning conditions on (transcript, audio) pairs. In a multi-speaker
request every speaker's reference audio is concatenated into ONE VQ blob and the
ONLY thing telling the model where one speaker's audio ends and the next begins
is the per-speaker reference *text* aligned to it. An empty transcript removes
that anchor, so the model can't segment the blob and collapses every turn to the
first/dominant voice. Auto-transcribing references at enrollment keeps the
transcript non-empty so multi-speaker `voice_map` binding works.

Optional dependency: if faster-whisper isn't installed (or the model can't load)
`transcribe()` returns "" and the caller falls back to the prior behaviour
(register with an empty transcript) instead of crashing enrollment.

Config via env:
  FISH_AUTO_TRANSCRIBE  1/0       enable (default 1)
  FISH_ASR_MODEL        small     faster-whisper model name (default; multilingual
                                  so mixed-language voice folders transcribe in the
                                  sample's own language — use "small.en" if all your
                                  reference samples are English for a bit more speed)
  FISH_ASR_DEVICE       cpu       cpu | cuda | cuda:N (default cpu — avoids
                                  VRAM contention with the TTS model + cuDNN deps)
  FISH_ASR_COMPUTE      auto      ctranslate2 compute_type; default int8 on cpu,
                                  float16 on cuda
  FISH_ASR_LANGUAGE     (auto)    force a language code (e.g. en); blank = auto
                                  (".en" models are English-only regardless)
"""

import os
import threading

from loguru import logger

_LOCK = threading.Lock()
_MODEL = None
_LOAD_FAILED = False


def auto_transcribe_enabled() -> bool:
    return os.environ.get("FISH_AUTO_TRANSCRIBE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _resolve_device():
    dev = (os.environ.get("FISH_ASR_DEVICE", "cpu") or "cpu").strip().lower()
    if dev == "auto":
        try:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    device_index = 0
    if dev.startswith("cuda:"):
        device_index = int(dev.split(":", 1)[1])
        dev = "cuda"
    compute = (os.environ.get("FISH_ASR_COMPUTE", "") or "").strip()
    if not compute:
        compute = "float16" if dev == "cuda" else "int8"
    return dev, device_index, compute


def _get_model():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    with _LOCK:
        if _MODEL is not None or _LOAD_FAILED:
            return _MODEL
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            logger.warning(
                f"[asr] faster-whisper not installed ({e}); voices without a "
                "transcript register empty (multi-speaker voice binding will be "
                "degraded). `uv pip install faster-whisper` to enable."
            )
            _LOAD_FAILED = True
            return None
        name = (os.environ.get("FISH_ASR_MODEL", "small") or "small").strip()
        device, device_index, compute = _resolve_device()
        try:
            logger.info(
                f"[asr] loading faster-whisper '{name}' "
                f"(device={device}:{device_index} compute={compute})"
            )
            _MODEL = WhisperModel(
                name, device=device, device_index=device_index, compute_type=compute
            )
        except Exception as e:
            logger.warning(
                f"[asr] failed to load '{name}' on {device}: {e}; "
                "auto-transcription disabled for this session."
            )
            _LOAD_FAILED = True
    return _MODEL


def transcribe(audio_path: str) -> str:
    """Return a transcript for `audio_path`, or "" if ASR is unavailable/failed."""
    if not auto_transcribe_enabled():
        return ""
    model = _get_model()
    if model is None:
        return ""
    try:
        kwargs = {"beam_size": 5}
        lang = (os.environ.get("FISH_ASR_LANGUAGE", "") or "").strip()
        if lang:
            kwargs["language"] = lang
        segments, _ = model.transcribe(audio_path, **kwargs)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        logger.warning(f"[asr] transcription failed for {audio_path}: {e}")
        return ""
