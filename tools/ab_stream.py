#!/usr/bin/env python3
"""A/B: baseline per-batch decode vs incremental decode-and-flush.

Same scene, same seed, both ways. Measures time-to-first-audio and saves both
full WAVs (+ a diff metric) so you can hear whether incremental decode adds any
artifacts vs the clean per-batch decode. Generation is identical; only WHEN audio
is decoded/emitted differs.
"""
import os
import time
from pathlib import Path

import numpy as np
import pyrootutils
import soundfile as sf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS  # noqa: E402
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest  # noqa: E402
from tools.server.model_manager import ModelManager  # noqa: E402

DEV = os.environ.get("DEV", "cuda:3")
OUT = Path(os.environ.get("OUT", "/mnt/truenas_public"))
PREFIX = os.environ.get("PREFIX", "ab2")
OUT.mkdir(parents=True, exist_ok=True)


def refs_from_voice_map(vmap):
    refs = []
    for spk, vid in sorted(vmap.items()):
        d = Path("references") / vid
        audio = next(f for f in sorted(d.iterdir())
                     if f.suffix.lower() in AUDIO_EXTENSIONS)
        lab = audio.with_suffix(".lab")
        text = lab.read_text(encoding="utf-8").strip() if lab.exists() else ""
        refs.append(ServeReferenceAudio(audio=audio.read_bytes(),
                                        text=f"<|speaker:{spk}|>{text}"))
    return refs


# Generic fictional scene (no real names): long narration first turn (big batch 0)
# to exercise time-to-first-audio, then dialogue across three voices.
long_turn = ("The morning market was already crowded when the two travelers arrived, "
             "the wooden stalls overflowing with bright vegetables and the warm smell "
             "of fresh bread drifting between the carts, and for a long moment they "
             "simply stood at the edge of it all, taking in the noise and the color "
             "before deciding where on earth they ought to begin.")
scene = (f"<|speaker:1|>{long_turn}\n"
         f"<|speaker:0|>[excited] Look at those tomatoes, they are perfect for tonight.\n"
         f"<|speaker:2|>Let us grab a basket first, or we will be juggling everything again.\n"
         f"<|speaker:1|>She laughed and reached for one of the worn wicker baskets by the gate.")
VMAP = {"0": "grace2", "1": "david_attenborough_cc3", "2": "en-in-m-prabhat"}
refs = refs_from_voice_map(VMAP)

print(f"[setup] loading model on {DEV}…", flush=True)
mm = ModelManager(mode="tts", device=DEV, half=False, compile=True,
                  llama_checkpoint_path="checkpoints/s2-pro",
                  decoder_checkpoint_path="checkpoints/s2-pro/codec.pth",
                  decoder_config_name="modded_dac_vq")
engine = mm.tts_inference_engine
sr = engine.decoder_model.spec_transform.sample_rate if hasattr(
    engine.decoder_model, "spec_transform") else engine.decoder_model.sample_rate


def run(chunk_tokens, label):
    req = ServeTTSRequest(
        text=scene, references=refs, reference_id=None, chunk_length=200,
        max_new_tokens=0, top_p=0.8, repetition_penalty=1.1, temperature=0.8,
        seed=1234, format="wav", streaming=True, stream_chunk_tokens=chunk_tokens,
    )
    t0 = time.perf_counter()
    ttfa = None
    n_segments = 0
    final = None
    for r in engine.inference(req):
        if r.code == "segment":
            n_segments += 1
            if ttfa is None:
                ttfa = time.perf_counter() - t0
        elif r.code == "final":
            final = r.audio[1].astype(np.float32)
        elif r.code == "error":
            print(f"  [{label}] ERROR: {r.error}")
    total = time.perf_counter() - t0
    print(f"@@@ {label:10} chunk_tokens={chunk_tokens:>3}  "
          f"TTFA={ttfa:6.2f}s  segments={n_segments:>3}  total={total:6.2f}s  "
          f"dur={len(final)/sr:.1f}s")
    return final


# Warm the scene shape once (discard) so neither run pays one-time compile.
print("[warm] priming scene shape…", flush=True)
run(0, "warm")

print("@@@ ==== A/B ====", flush=True)
base = run(0, "baseline")     # per-batch decode (current behaviour)
new = run(32, "incremental")  # decode every 32 tokens

sf.write(OUT / f"{PREFIX}_baseline.wav", base, sr)
sf.write(OUT / f"{PREFIX}_incremental.wav", new, sr)

# Diff metric (align lengths) — large spikes = boundary artifacts.
n = min(len(base), len(new))
d = np.abs(base[:n] - new[:n])
print(f"\n[diff] len base={len(base)} new={len(new)}  "
      f"max|Δ|={d.max():.4f}  mean|Δ|={d.mean():.6f}  "
      f"(0 = bit-identical; small = clean; spikes = clicks)")
print(f"[saved] {OUT}/{PREFIX}_baseline.wav  and  {OUT}/{PREFIX}_incremental.wav")
