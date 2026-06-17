#!/usr/bin/env python3
"""Render a sibilant-heavy line at one precision (QUANT env) for a hiss A/B.

QUANT = int4 | int8 | bf16. Same text + seed across runs so the only variable is
weight precision. Single narrator (grace2) so sibilants are isolated, no
speaker-switching to confound the comparison.
"""
import io
import os

import numpy as np
import pyrootutils
import soundfile as sf

QUANT = os.environ.get("QUANT", "bf16")
if QUANT == "int4":
    os.environ["FISH_QUANTIZE"] = "int4"
    os.environ["FISH_QUANTIZED_WEIGHTS"] = "checkpoints/s2-pro/model.int4.g128.pt"
elif QUANT == "int8":
    os.environ["FISH_QUANTIZE"] = "int8"
    os.environ["FISH_QUANTIZED_WEIGHTS"] = "checkpoints/s2-pro/model.int8.pt"
else:  # bf16, no quant
    os.environ["FISH_QUANTIZE"] = "none"
    os.environ.pop("FISH_QUANTIZED_WEIGHTS", None)

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.schema import ServeTTSRequest  # noqa: E402
from tools.server.model_manager import ModelManager  # noqa: E402

DEV = os.environ.get("DEV", "cuda:3")
OUT = os.environ.get("OUT", f"/mnt/truenas_public/hiss_{QUANT}.wav")

# Sibilant/fricative-heavy — s/sh/z/f/st/ch, where low-bit hiss shows worst.
text = ("She sells seashells by the seashore, while the sixth sheikh's sixth sheep "
        "sleeps soundly nearby. Fresh fish sizzle and soft steam hisses from the "
        "simmering pots, as the shopkeeper whispers softly about this season's "
        "finest spices, silks, and sweets.")

print(f"[hiss] QUANT={QUANT} loading on {DEV}…", flush=True)
mm = ModelManager(mode="tts", device=DEV, half=False, compile=True,
                  llama_checkpoint_path="checkpoints/s2-pro",
                  decoder_checkpoint_path="checkpoints/s2-pro/codec.pth",
                  decoder_config_name="modded_dac_vq")
engine = mm.tts_inference_engine

req = ServeTTSRequest(text=text, reference_id="grace2", chunk_length=200,
                      max_new_tokens=0, top_p=0.8, repetition_penalty=1.1,
                      temperature=0.8, seed=777, format="wav", streaming=False,
                      stream_chunk_tokens=0)
final = None
sr = 44100
for r in engine.inference(req):
    if r.code == "final":
        sr = r.audio[0]
        final = r.audio[1].astype(np.float32)
    elif r.code == "error":
        print(f"[hiss] ERROR: {r.error}", flush=True)
if final is not None:
    sf.write(OUT, final, sr)
    print(f"@@@ {QUANT}: dur={len(final)/sr:.1f}s saved {OUT}", flush=True)
