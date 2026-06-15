#!/usr/bin/env python3
"""Backfill transcripts (.lab) for registered reference voices that have none.

Fish voice cloning needs a transcript per reference; an empty .lab breaks
multi-speaker voice_map binding (speakers collapse to one voice). This scans
`references/<id>/sample.lab` and transcribes the matching audio in place for any
that are missing or empty, using the same faster-whisper config as the server
(FISH_ASR_MODEL / FISH_ASR_DEVICE / ...). Idempotent: re-running only touches
still-empty labs.

Usage:
  python tools/backfill_transcripts.py [voice_id ...]   # default: all empty
  python tools/backfill_transcripts.py --dry-run
"""
import sys
from pathlib import Path

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.utils.file import AUDIO_EXTENSIONS  # noqa: E402
from tools import asr  # noqa: E402

REF = Path("references")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv

    if not REF.is_dir():
        print("no references/ dir")
        return

    targets = []
    for d in sorted(REF.iterdir()):
        if not d.is_dir():
            continue
        if args and d.name not in args:
            continue
        lab = d / "sample.lab"
        if lab.exists() and lab.stat().st_size > 0:
            continue  # already has a transcript
        audio = next((f for f in sorted(d.iterdir())
                      if f.suffix.lower() in AUDIO_EXTENSIONS), None)
        if audio is None:
            continue
        targets.append((d.name, audio, lab))

    print(f"[backfill] {len(targets)} voice(s) need a transcript"
          + (" (dry run)" if dry else ""))
    if dry:
        for vid, audio, _ in targets:
            print(f"  would transcribe {vid} <- {audio.name}")
        return
    if not targets:
        return

    if not asr.auto_transcribe_enabled():
        print("[backfill] FISH_AUTO_TRANSCRIBE is off; nothing to do")
        return

    done = 0
    failed = []
    for i, (vid, audio, lab) in enumerate(targets, 1):
        text = asr.transcribe(str(audio))
        if not text:
            failed.append(vid)
            print(f"  [{i}/{len(targets)}] {vid}: FAILED (empty)")
            continue
        lab.write_text(text, encoding="utf-8")
        done += 1
        print(f"  [{i}/{len(targets)}] {vid}: {text[:80]!r}")

    print(f"[backfill] done: {done} transcribed, {len(failed)} failed")
    if failed:
        print("  failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
