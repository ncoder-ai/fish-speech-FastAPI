#!/usr/bin/env python3
"""Fixed-text benchmark for quant A/B: prints RTF, saves audio for listening."""
import os
import subprocess
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8770"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = f"/mnt/truenas_public/fish_speech_tests/quant/{LABEL}.wav"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

TEXT = (
    "The harbor lights flickered against the evening fog as the last ferry pulled "
    "away from the dock. Somewhere in the distance a bell rang twice, slow and "
    "deliberate, marking the turn of the tide. She pulled her coat tighter and "
    "watched the water turn from gray to black, thinking of everything that had "
    "been left unsaid between them on that final, quiet afternoon."
)

t = time.time()
r = httpx.post(f"{BASE}/v1/audio/speech",
               json={"input": TEXT, "response_format": "wav", "stream": False,
                     "temperature": 0.7, "seed": 1234}, timeout=600)
dt = time.time() - t
if r.status_code != 200:
    print(f"{LABEL}: HTTP {r.status_code} {r.text[:200]}")
    sys.exit(1)
with open(OUT, "wb") as f:
    f.write(r.content)
dur = float(subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=nw=1:nk=1", OUT], stdout=subprocess.PIPE).stdout.decode().strip())
print(f"BENCH {LABEL}: audio={dur:.1f}s wall={dt:.1f}s RTF={dt/dur:.3f} -> {OUT}")
