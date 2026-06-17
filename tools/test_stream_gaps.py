#!/usr/bin/env python3
"""Measure gaps in the audio stream (esp. between batches) for a long scene.

Streams PCM and tracks, per chunk: arrival time and cumulative audio seconds
received. The telling metric is BUFFER LEAD = (audio_received - wall_elapsed):
if it ever goes negative after the first chunk, a real-time player would starve
(audible gap). Large inter-chunk gaps usually fall on batch boundaries.
"""
import os
import time

import requests

BASE = os.environ.get("BASE", "http://localhost:8770")
SR = 44100  # pcm s16le mono
BYTES_PER_SEC = SR * 2

# Multi-turn scene -> several batches, so we exercise batch boundaries.
turns = [
    (1, "The afternoon light slanted through the tall windows of the old library, "
        "catching the dust that drifted between the towering shelves of forgotten books."),
    (0, "[whisper] I think the map is hidden somewhere in this section."),
    (2, "Keep your voice down, the librarian already suspects we are up to something."),
    (1, "She ran her fingers along the cracked leather spines, pausing at a volume "
        "thicker than the rest, its title worn almost to nothing."),
    (0, "This one. It has to be this one, look at the strange marking on the side."),
    (2, "Pull it slowly. The last time you yanked a book down, half the shelf came with it."),
    (1, "With a careful tug the book slid free, and behind it, set into the wood, "
        "was a small brass keyhole that had not seen a key in a very long time."),
    (0, "[excited] I knew it. I absolutely knew there was something back here."),
    (2, "Incredible. Now we just need to find whatever opens it before they close."),
]
text = "\n".join(f"<|speaker:{s}|>{t}" for s, t in turns)
req = {"input": text, "voice": "grace2",
       "voice_map": {"0": "grace2", "1": "david_attenborough_cc3", "2": "en-in-m-prabhat"},
       "response_format": "pcm", "stream": True, "seed": 99}

print(f"[gaps] streaming {len(turns)}-turn scene from {BASE}")
t0 = time.time()
r = requests.post(f"{BASE}/v1/audio/speech", json=req, stream=True, timeout=600)
r.raise_for_status()

total = 0
last_t = None
gaps = []  # (wall_time, gap_since_last_chunk, buffer_lead)
min_lead = None
first = None
for chunk in r.iter_content(chunk_size=16384):
    if not chunk:
        continue
    now = time.time() - t0
    total += len(chunk)
    if first is None:
        first = now
    audio_recv = total / BYTES_PER_SEC
    lead = audio_recv - now  # how far the buffer is ahead of real-time playback
    if last_t is not None:
        gaps.append((now, now - last_t, lead))
    if min_lead is None or lead < min_lead:
        min_lead = lead
    last_t = now

audio_total = total / BYTES_PER_SEC
wall = time.time() - t0
big = [g for g in gaps if g[1] > 0.4]
print(f"[gaps] audio={audio_total:.1f}s  wall={wall:.1f}s  first_chunk={first:.2f}s")
print(f"[gaps] inter-chunk gaps > 0.4s: {len(big)}")
for w, g, lead in big:
    print(f"   at {w:5.1f}s  gap={g:4.2f}s  buffer_lead={lead:+.2f}s")
print(f"[gaps] min buffer_lead after start = {min_lead:+.2f}s "
      f"({'NEGATIVE -> playback would stall (gap)' if min_lead < 0 else 'never starves -> smooth'})")
