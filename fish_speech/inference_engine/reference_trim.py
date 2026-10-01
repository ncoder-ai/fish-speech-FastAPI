"""Trim long reference clips to a usable length, at a pause, with a matching transcript.

Long references (tens of seconds) degrade voice cloning, and break multi-voice
scenes: a single 87 s clip scrambled a 7-voice scene that worked once that voice
was swapped. Clips longer than FISH_REF_MAX_S (default 20; 0 disables) are cut
in memory; files on disk are untouched.

The cut is placed in a real pause between words, never inside one: candidate
cuts are word ends from Whisper's timestamps, preferring sentence ends, then
clause ends, and requiring at least MIN_GAP_S of quiet before the next word. The
exact sample is the quietest point in that gap, followed by a short fade. The
transcript is cut to the kept words: the original transcript when it lines up
with what Whisper heard (keeps your wording and punctuation), otherwise
Whisper's own text.
"""

import io
import os
import re
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf
from loguru import logger

MAX_S = float(os.environ.get("FISH_REF_MAX_S", "20") or "0")
MIN_KEEP_S = 6.0  # never trim below this; prefer a shorter clip only if it ends cleanly
MIN_GAP_S = 0.15  # quiet needed between the last kept word and the next one
MIN_QUIET_DB = 20.0  # the pause must dip this far below the clip's speech level
FADE_S = 0.02
_TAG_RE = re.compile(r"^((?:<\|speaker:\d+\|>\s*)*)")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def needs_trim(duration_s: float) -> bool:
    return MAX_S > 0 and duration_s > MAX_S + 1.0


def _norm(t: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(t)]


def plan_trim(wav: np.ndarray, sr: int):
    """Decide where to cut `wav`. Returns a plan dict, or None to keep it whole."""
    import librosa

    from tools import asr

    words, lang = asr.transcribe_words(librosa.resample(wav, orig_sr=sr, target_sr=16000))
    if not words:
        logger.warning(f"[ref] {len(wav) / sr:.1f}s reference left untrimmed: no word timings (ASR unavailable)")
        return None

    # A repetition loop ("bread, bread, bread...") means the transcription failed;
    # cutting with it would pair the audio with a garbage transcript.
    norm_words = [re.sub(r"[^\w]", "", w.lower()) for w, _, _ in words]
    run = longest = 1
    for a, b in zip(norm_words, norm_words[1:]):
        run = run + 1 if a and a == b else 1
        longest = max(longest, run)
    if longest > 5:
        logger.warning(f"[ref] {len(wav) / sr:.1f}s reference left untrimmed: unreliable transcription "
                       f"(a word repeats {longest} times)")
        return None

    def gap_after(i):
        nxt = words[i + 1][1] if i + 1 < len(words) else len(wav) / sr
        return nxt - words[i][2]

    hop = max(1, int(0.01 * sr))
    speech_rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)

    def quietest(i):
        """(sample index, dB below speech) of the quietest 10 ms in the pause after word i."""
        a = int(words[i][2] * sr)
        b = int((words[i + 1][1] if i + 1 < len(words) else len(wav) / sr) * sr)
        if b - a <= 2 * hop:
            return None, 0.0
        levels = [float(np.sqrt(np.mean(wav[s:s + hop] ** 2)) + 1e-9) for s in range(a, b - hop, hop)]
        k = int(np.argmin(levels))
        return a + hop * k + hop // 2, 20 * np.log10(speech_rms / levels[k])

    def rank(i):
        w = words[i][0]
        return 2 if re.search(r"[.!?。！？…]$", w) else 1 if re.search(r"[,;:—–\-]$", w) else 0

    # Timestamps alone are not trusted: a pause only counts if the audio in it
    # is actually quiet, so the cut can never land inside a word.
    candidates = {}
    for i, (_, _, end) in enumerate(words):
        if MIN_KEEP_S <= end <= MAX_S and gap_after(i) >= MIN_GAP_S:
            at, depth = quietest(i)
            if at is not None and depth >= MIN_QUIET_DB:
                candidates[i] = at
    if not candidates:
        logger.warning(f"[ref] {len(wav) / sr:.1f}s reference left untrimmed: no pause between "
                       f"{MIN_KEEP_S:.0f}s and {MAX_S:.0f}s to cut at")
        return None
    best = max(candidates, key=lambda i: (rank(i), words[i][2]))

    # Cut at the quietest 10 ms window inside the pause after the kept word.
    cut = candidates[best]
    out = wav[:cut].copy()
    fade = min(len(out), int(FADE_S * sr))
    out[len(out) - fade:] *= np.linspace(1.0, 0.0, fade, dtype=out.dtype)
    buf = io.BytesIO()
    sf.write(buf, out, sr, format="WAV")
    return {"audio": buf.getvalue(), "kept": words[: best + 1], "lang": lang,
            "from_s": len(wav) / sr, "to_s": cut / sr}


def trim_text(text: str, plan) -> str:
    """Cut `text` (optionally starting with speaker tags) to the plan's kept words."""
    tags = _TAG_RE.match(text).group(1)
    body = text[len(tags):]
    kept = plan["kept"]
    asr_text = " ".join(w for w, _, _ in kept).strip()
    toks = list(_TOKEN_RE.finditer(body))
    heard = [t for w, _, _ in kept for t in _norm(w)]
    if toks and heard:
        sm = SequenceMatcher(None, [t.group(0).lower() for t in toks], heard, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        matched = sum(b.size for b in blocks)
        if blocks and matched >= 0.6 * len(heard):
            last = blocks[-1]
            end_tok = toks[last.a + last.size - 1]
            # Keep trailing punctuation that belongs to the last kept word.
            m = re.match(r"[^\w\s]*", body[end_tok.end():])
            return tags + body[: end_tok.end() + (m.end() if m else 0)].strip()
    return tags + asr_text
