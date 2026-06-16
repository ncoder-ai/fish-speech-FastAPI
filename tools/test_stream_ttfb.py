#!/usr/bin/env python3
"""Measure streaming latency: when does the header arrive vs the first AUDIO chunk?

A long first turn => big batch 0 => first audio is delayed by batch-0 generation
time even though we 'stream'. This is what makes Kahani time out before any audio.
"""
import os
import time

import requests

BASE = os.environ.get("BASE", "http://172.16.23.180:8770")

# Kahani-like scene: a LONG narration first turn (big batch 0), then dialogue.
long_turn = ("Radhika repeated the name slowly, testing the unfamiliar syllables "
             "on her tongue as she dried her hands on the worn kitchen towel, her "
             "mind already drifting toward the gleaming cabinets and the modern "
             "appliances she had only ever seen in magazines and never once dared "
             "to imagine inside her own small and perpetually cluttered home. ")
text = (f"<|speaker:1|>{long_turn}\n"
        f"<|speaker:0|>[low voice] These are beautiful, really beautiful.\n"
        f"<|speaker:2|>I thought so too, let us call him tomorrow.")

req = {"input": text, "voice": "grace2",
       "voice_map": {"0": "grace2", "1": "david_attenborough_cc3", "2": "en-in-m-prabhat"},
       "response_format": "pcm", "stream": True}

print(f"[stream] POST {BASE}/v1/audio/speech  (pcm, stream=True)")
t0 = time.time()
r = requests.post(f"{BASE}/v1/audio/speech", json=req, stream=True, timeout=600)
print(f"[stream] response headers at {time.time()-t0:.2f}s  status={r.status_code}")

first_byte = None
first_audio = None
total = 0
for chunk in r.iter_content(chunk_size=4096):
    if not chunk:
        continue
    now = time.time() - t0
    total += len(chunk)
    if first_byte is None:
        first_byte = now
        print(f"[stream] FIRST byte (header) at {first_byte:.2f}s ({len(chunk)} B)")
    # PCM: header is tiny; treat the first chunk that carries real audio (>1KB
    # cumulative beyond a header) as first audio.
    if first_audio is None and total > 2048:
        first_audio = now
        print(f"[stream] FIRST AUDIO chunk at {first_audio:.2f}s (cum {total} B)")
print(f"[stream] done at {time.time()-t0:.2f}s, {total} bytes total")
print(f"\nTTFB(header)={first_byte:.2f}s   time-to-first-audio={first_audio:.2f}s"
      if first_audio else "no audio")
