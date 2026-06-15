#!/usr/bin/env python3
"""
Prove the fast-503 path: run against a server started with a SHORT
--queue-timeout. One long request occupies the single synthesis slot; concurrent
requests must return 503 quickly instead of hanging until the long one finishes.

Usage (server must run with e.g. --queue-timeout 8):
  .venv/bin/python tools/test_503.py --base-url http://localhost:8770
"""
import argparse
import asyncio
import time

import httpx

LONG = " ".join(
    [f"This is sentence number {i} in a deliberately long occupying request."
     for i in range(60)]
)


async def occupy(base):
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{base}/v1/audio/speech",
                         json={"input": LONG, "response_format": "wav",
                               "stream": False})
        return ("occupier", r.status_code, time.time())


async def contender(base, i):
    t0 = time.time()
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(f"{base}/v1/audio/speech",
                         json={"input": "Quick contender request.",
                               "response_format": "wav", "stream": False})
        return ("contender", i, r.status_code, round(time.time() - t0, 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8770")
    args = ap.parse_args()
    base = args.base_url

    occ = asyncio.create_task(occupy(base))
    await asyncio.sleep(2.0)  # let the occupier grab the slot
    contenders = await asyncio.gather(*[contender(base, i) for i in range(3)])
    occ_res = await occ

    print("occupier:", occ_res[1])
    for c in contenders:
        print(f"contender {c[1]}: status={c[2]} after {c[3]}s")

    n503 = sum(1 for c in contenders if c[2] == 503)
    fast = all(c[3] <= 20 for c in contenders if c[2] == 503)
    if n503 >= 1 and fast and occ_res[1] == 200:
        print(f"\nPASS: {n503}/3 contenders got fast 503 while occupier ran; "
              f"occupier completed 200")
        raise SystemExit(0)
    print(f"\nFAIL: n503={n503} occupier={occ_res[1]} contenders={contenders}")
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
