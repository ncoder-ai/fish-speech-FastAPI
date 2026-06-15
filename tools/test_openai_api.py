#!/usr/bin/env python3
"""
End-to-end smoke/stress tests for the Fish-Speech OpenAI-compatible API.

Targets the hang/crash fixes:
  - all response formats produce valid, decodable audio (non-stream + stream)
  - the ffmpeg compressed-streaming path does not deadlock on long input
  - concurrency is serialized cleanly (no crash) and excess load fails fast (503)
  - a client disconnect cancels the in-flight job and the server recovers fast

Usage:
  .venv/bin/python tools/test_openai_api.py --base-url http://localhost:8770
"""
import argparse
import asyncio
import io
import subprocess
import sys
import time

import httpx
import numpy as np
import soundfile as sf

BASE = "http://localhost:8770"
PASS, FAIL = [], []


def ok(name, msg=""):
    PASS.append(name)
    print(f"  \033[32mPASS\033[0m {name} {msg}")


def bad(name, msg=""):
    FAIL.append(name)
    print(f"  \033[31mFAIL\033[0m {name} {msg}")


def probe_duration(data: bytes, fmt: str) -> float:
    """Return audio duration in seconds, or -1 if undecodable.

    Decodes through ffmpeg to raw s16le and counts samples. This is robust to
    non-seekable pipes and to streaming containers (e.g. a WAV written with a
    placeholder length header), which trip up ffprobe/soundfile.
    """
    if fmt == "pcm":
        # raw s16le mono @ 44100 (server fallback rate); no container to decode.
        return len(data) / 2 / 44100
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "44100", "pipe:1"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        return len(p.stdout) / 2 / 44100
    except Exception:
        return -1


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=10)
    j = r.json()
    if r.status_code == 200 and j.get("status") == "ok":
        ok("health", f"(device={j.get('device')})")
    else:
        bad("health", str(j))


def test_nonstream(fmt):
    payload = {"input": "Hello there. This is a non streaming test.",
               "response_format": fmt, "stream": False}
    t = time.time()
    try:
        r = httpx.post(f"{BASE}/v1/audio/speech", json=payload, timeout=300)
    except Exception as e:
        bad(f"nonstream:{fmt}", f"request error {e}")
        return
    dt = time.time() - t
    if r.status_code != 200:
        bad(f"nonstream:{fmt}", f"status={r.status_code} {r.text[:200]}")
        return
    dur = probe_duration(r.content, fmt)
    if dur > 0.3:
        ok(f"nonstream:{fmt}", f"{len(r.content)}B {dur:.1f}s audio in {dt:.1f}s")
    else:
        bad(f"nonstream:{fmt}", f"bad audio dur={dur} bytes={len(r.content)}")


def test_stream(fmt, text=None, name=None):
    name = name or f"stream:{fmt}"
    text = text or "Streaming test. One two three four five."
    payload = {"input": text, "response_format": fmt, "stream": True}
    t = time.time()
    buf = bytearray()
    nchunks = 0
    try:
        with httpx.stream("POST", f"{BASE}/v1/audio/speech", json=payload,
                          timeout=300) as r:
            if r.status_code != 200:
                bad(name, f"status={r.status_code}")
                return
            for chunk in r.iter_bytes():
                buf += chunk
                nchunks += 1
    except Exception as e:
        bad(name, f"stream error {e}")
        return
    dt = time.time() - t
    dur = probe_duration(bytes(buf), fmt)
    if dur > 0.3:
        ok(name, f"{len(buf)}B {dur:.1f}s in {nchunks} chunks, {dt:.1f}s")
    else:
        bad(name, f"bad audio dur={dur} bytes={len(buf)} chunks={nchunks}")


async def test_concurrency(n=4):
    """Fire n streaming requests at once; with concurrency=1 they must all
    succeed (serialized) without the server crashing."""
    payload = {"input": "Concurrent request stress test sentence.",
               "response_format": "mp3", "stream": True}

    async def one(i):
        try:
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream("POST", f"{BASE}/v1/audio/speech",
                                    json=payload) as r:
                    b = bytearray()
                    async for chunk in r.aiter_bytes():
                        b += chunk
                    return r.status_code, len(b)
        except Exception as e:
            return ("ERR", str(e))

    t = time.time()
    res = await asyncio.gather(*[one(i) for i in range(n)])
    dt = time.time() - t
    good = sum(1 for s, b in res if s == 200 and isinstance(b, int) and b > 0)
    s503 = sum(1 for s, _ in res if s == 503)
    if good + s503 == n and good >= 1:
        ok("concurrency", f"{good}/{n} ok, {s503} fast-503, {dt:.1f}s, results={res}")
    else:
        bad("concurrency", f"results={res}")


async def test_disconnect():
    """Start a long streaming job, drop it after the first chunk, then confirm
    the server answers a fresh short request promptly (worker not wedged)."""
    long_text = " ".join(
        [f"This is sentence number {i} of a long disconnect test." for i in range(40)]
    )
    payload = {"input": long_text, "response_format": "wav", "stream": True}
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            async with c.stream("POST", f"{BASE}/v1/audio/speech",
                                json=payload) as r:
                async for _ in r.aiter_bytes():
                    break  # got first bytes -> abruptly disconnect
    except Exception as e:
        bad("disconnect:drop", f"{e}")
        return
    ok("disconnect:drop", "client dropped after first chunk")

    # Server must recover and serve a short request reasonably soon. Allow time
    # for the in-flight batch to finish + cancel to take effect.
    await asyncio.sleep(2)
    t = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{BASE}/v1/audio/speech",
                             json={"input": "Recovered.", "response_format": "wav",
                                   "stream": False})
        dt = time.time() - t
        if r.status_code == 200 and probe_duration(r.content, "wav") > 0.1:
            ok("disconnect:recover", f"served fresh request in {dt:.1f}s")
        else:
            bad("disconnect:recover", f"status={r.status_code} dt={dt:.1f}s")
    except Exception as e:
        bad("disconnect:recover", f"{e}")


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8770")
    ap.add_argument("--quick", action="store_true", help="skip long/concurrency tests")
    args = ap.parse_args()
    BASE = args.base_url

    print("== health ==")
    test_health()

    print("== non-streaming formats ==")
    for fmt in ["wav", "pcm", "flac", "mp3", "opus", "aac"]:
        test_nonstream(fmt)

    print("== streaming formats ==")
    for fmt in ["wav", "pcm", "mp3", "opus"]:
        test_stream(fmt)

    if not args.quick:
        print("== long streaming (ffmpeg deadlock stress) ==")
        long_text = " ".join(
            [f"Paragraph chunk {i}, a moderately long sentence to stress the pipe."
             for i in range(25)]
        )
        test_stream("mp3", text=long_text, name="stream:mp3:long")

        print("== concurrency ==")
        asyncio.run(test_concurrency(4))

        print("== disconnect / recovery ==")
        asyncio.run(test_disconnect())

    print(f"\n==== {len(PASS)} passed, {len(FAIL)} failed ====")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
