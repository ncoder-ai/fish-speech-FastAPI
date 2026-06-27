#!/usr/bin/env python3
"""
Force the sliding-context-window path. A single-narrator script with many short
sentences becomes many speaker turns, so the running conversation overflows the
6144-token budget. BEFORE the fix this returned HTTP 500
("Prompt is too long"); AFTER, it must trim the oldest turns and complete 200.

Usage:
  .venv/bin/python tools/test_force_trim.py --base-url http://localhost:8770
"""
import argparse
import subprocess
import time

import httpx

OUT = "/tmp/fish_speech_tests/force_trim_long_narration.wav"

# ~95 short sentences -> ~95 turns -> guaranteed to exceed the context budget.
_POOL = [
    "The morning began like any other on the quiet harbor.",
    "Gulls circled above the cold gray water.",
    "A fisherman untied his small wooden boat.",
    "The rope was stiff and damp with salt.",
    "He checked the nets one final time.",
    "The engine coughed twice before it caught.",
    "Smoke drifted low across the calm surface.",
    "He steered carefully past the old stone pier.",
    "The town behind him was still asleep.",
    "Lights flickered in only a few windows.",
    "A dog barked somewhere far away.",
    "The horizon glowed a pale and tired orange.",
    "He had made this trip a thousand times.",
    "Yet every dawn still felt brand new.",
    "The water grew darker as he moved offshore.",
    "Waves slapped gently against the wooden hull.",
    "He poured coffee from a battered metal flask.",
    "The bitter warmth steadied his tired hands.",
    "A cormorant dove and vanished beneath the swell.",
    "Far out, a freighter crawled along the edge of the world.",
]


def build_text(n_sentences=95):
    out = []
    for i in range(n_sentences):
        out.append(_POOL[i % len(_POOL)])
    return " ".join(out)


def decoded_seconds(data: bytes) -> float:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "44100", "pipe:1"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
        )
        return len(p.stdout) / 2 / 44100
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8770")
    ap.add_argument("--sentences", type=int, default=95)
    args = ap.parse_args()

    text = build_text(args.sentences)
    words = len(text.split())
    print(f"sending {args.sentences} sentences (~{words} words) single-narrator...")
    t = time.time()
    r = httpx.post(f"{args.base_url}/v1/audio/speech",
                   json={"input": text, "response_format": "wav", "stream": False},
                   timeout=900)
    dt = time.time() - t
    if r.status_code != 200:
        print(f"FAIL: status={r.status_code} {r.text[:300]} in {dt:.1f}s")
        raise SystemExit(1)
    with open(OUT, "wb") as f:
        f.write(r.content)
    dur = decoded_seconds(r.content)
    print(f"status=200, {dur:.1f}s audio, {len(r.content)} bytes in {dt:.1f}s -> {OUT}")
    if dur > 30:
        print("PASS: long many-turn narration completed without crashing")
        raise SystemExit(0)
    print("FAIL: audio too short")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
