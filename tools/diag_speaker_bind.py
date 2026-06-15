#!/usr/bin/env python3
"""Diagnostic: does multi-speaker voice_map binding actually switch voices?

Generates a strict alternating 2-speaker dialogue (spk0=female voice,
spk1=male voice) and analyzes the F0 (pitch) contour. Proper binding ->
bimodal F0 (female ~180-260 Hz alternating with male ~90-150 Hz). A collapse
(all one voice) -> unimodal F0.

Usage: diag_speaker_bind.py <spk0_voice> <spk1_voice> [out_label]
"""
import io
import sys
import numpy as np
import requests
import soundfile as sf
import librosa

import os
BASE = os.environ.get("BASE", "http://localhost:8770")

spk0 = sys.argv[1] if len(sys.argv) > 1 else "grace2"
spk1 = sys.argv[2] if len(sys.argv) > 2 else "david_attenborough_cc3"
label = sys.argv[3] if len(sys.argv) > 3 else "test"

# Alternating turns, similar length so segments are comparable.
lines = [
    (0, "The morning light came softly through the kitchen window."),
    (1, "Yes, it was a calm and beautiful day all around us."),
    (0, "I have already made some fresh coffee for both of us."),
    (1, "Thank you so much, that is really very kind of you today."),
    (0, "We should take a long walk in the park this afternoon."),
    (1, "That sounds like a wonderful and relaxing idea to me."),
]
text = "\n".join(f"<|speaker:{s}|>{t}" for s, t in lines)

req = {
    "input": text,
    "voice": spk0,
    "voice_map": {"0": spk0, "1": spk1},
    "response_format": "wav",
    "seed": 12345,
}

print(f"[gen] spk0={spk0} (expect female) spk1={spk1} (expect male)")
r = requests.post(f"{BASE}/v1/audio/speech", json=req, timeout=600)
r.raise_for_status()
audio, sr = sf.read(io.BytesIO(r.content))
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)
dur = len(audio) / sr
print(f"[gen] got {dur:.1f}s @ {sr} Hz")

out = f"/tmp/diag_{label}.wav"
sf.write(out, audio, sr)
print(f"[gen] saved {out}")

# Frame-wise F0 via pyin (voiced frames only).
f0, voiced, _ = librosa.pyin(
    audio, fmin=70, fmax=350, sr=sr, frame_length=2048, hop_length=512
)
times = librosa.times_like(f0, sr=sr, hop_length=512)
v = f0[voiced & ~np.isnan(f0)]
if len(v) == 0:
    print("!! no voiced frames detected")
    sys.exit(1)

print(f"\n[F0] voiced frames: {len(v)}")
print(f"[F0] median={np.median(v):.0f} Hz  mean={np.mean(v):.0f} Hz  "
      f"p10={np.percentile(v,10):.0f}  p90={np.percentile(v,90):.0f}")

# Bimodality check: split at 160 Hz (rough female/male divider).
lo = v[v < 160]   # male-ish
hi = v[v >= 160]  # female-ish
print(f"[F0] frames <160Hz (male-ish): {len(lo)} ({100*len(lo)/len(v):.0f}%)  "
      f">=160Hz (female-ish): {len(hi)} ({100*len(hi)/len(v):.0f}%)")

# Timeline: median F0 per 1s window, to see alternation.
print("\n[timeline] median F0 per 1s window (F=female-range >=180, M=male <=150):")
row = []
for t in range(int(dur) + 1):
    mask = (times >= t) & (times < t + 1) & voiced & ~np.isnan(f0)
    seg = f0[mask]
    if len(seg) < 3:
        row.append(f"{t:>2}:  --- ")
        continue
    m = np.median(seg)
    tag = "F" if m >= 180 else ("M" if m <= 150 else "?")
    row.append(f"{t:>2}: {m:>4.0f}{tag}")
print("  " + "  ".join(row))

# Verdict
both = len(lo) > 0.15 * len(v) and len(hi) > 0.15 * len(v)
print(f"\n[verdict] {'BIMODAL — voices appear to switch (GOOD)' if both else 'UNIMODAL — voices likely COLLAPSED to one (BUG)'}")
