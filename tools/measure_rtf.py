#!/usr/bin/env python3
"""Measure RTF and save a sample from the running server (for quant A/B).

Same generic market scene + seed as tools/ab_stream.py, so the output is
directly comparable to the int4 samples for the hiss check. Non-streaming so RTF
= pure generation (wall / audio_seconds).
"""
import io
import os
import time

import numpy as np
import requests
import soundfile as sf

BASE = os.environ.get("BASE", "http://localhost:8770")
OUT = os.environ.get("OUT", "/tmp/int8_baseline.wav")
LABEL = os.environ.get("LABEL", "int8")

long_turn = ("The morning market was already crowded when the two travelers arrived, "
             "the wooden stalls overflowing with bright vegetables and the warm smell "
             "of fresh bread drifting between the carts, and for a long moment they "
             "simply stood at the edge of it all, taking in the noise and the color "
             "before deciding where on earth they ought to begin.")
scene = (f"<|speaker:1|>{long_turn}\n"
         f"<|speaker:0|>[excited] Look at those tomatoes, they are perfect for tonight.\n"
         f"<|speaker:2|>Let us grab a basket first, or we will be juggling everything again.\n"
         f"<|speaker:1|>She laughed and reached for one of the worn wicker baskets by the gate.")
req = {"input": scene, "voice": "grace2",
       "voice_map": {"0": "grace2", "1": "david_attenborough_cc3", "2": "en-in-m-prabhat"},
       "response_format": "wav", "seed": 1234, "stream": False}

t0 = time.time()
r = requests.post(f"{BASE}/v1/audio/speech", json=req, timeout=900)
wall = time.time() - t0
r.raise_for_status()
a, sr = sf.read(io.BytesIO(r.content))
a = a.mean(1) if a.ndim > 1 else a
dur = len(a) / sr
sf.write(OUT, a.astype(np.float32), sr)
print(f"@@@ {LABEL}: audio={dur:.1f}s  wall={wall:.1f}s  RTF={wall/dur:.3f}  "
      f"(~{dur*21.5/wall:.0f} tok/s)  saved {OUT}")
