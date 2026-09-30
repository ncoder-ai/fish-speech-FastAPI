"""Fire N concurrent /v1/audio/speech requests; report TTFB, wall, audio length, RTF.

BASE=http://localhost:8771 N=4 STREAM=1 OUT=dir
"""
import concurrent.futures as cf, io, os, time, wave
import requests

BASE = os.environ.get("BASE", "http://localhost:8771"); N = int(os.environ.get("N", "4"))
STREAM = os.environ.get("STREAM", "1") == "1"; OUT = os.environ.get("OUT", "/tmp/fish_load")
os.makedirs(OUT, exist_ok=True)
VOICES = ["grace2", "david_attenborough_cc3", "en-in-m-prabhat", "alice", "andy", "angie1", "grace2", "alice"]
TEXTS = [
    "The morning market was already crowded when the two travelers arrived, and the wooden stalls overflowed with bright vegetables.",
    "[excited] Look at those tomatoes, they are perfect for tonight. Let us grab a basket before they are all gone!",
    "In the quiet hours before dawn, the old lighthouse keeper climbed the spiral stairs one last time and watched the sea.",
    "Please remember to bring your identification documents and arrive fifteen minutes early for the appointment tomorrow.",
    "She laughed and reached for one of the worn wicker baskets by the gate, then turned back to wave at her brother.",
    "Scientists have discovered a new species of deep sea fish that glows faintly blue in the darkness of the ocean trench.",
    "[whisper] Keep your voice down, the baby is finally asleep and I do not want to start the whole routine over again.",
    "Our quarterly results exceeded expectations, driven by strong growth in the services division and lower costs overall.",
]

def one(i):
    body = {"model": "s2-pro", "input": TEXTS[i % len(TEXTS)], "voice": VOICES[i % len(VOICES)],
            "response_format": "wav", "stream": STREAM, "seed": 100 + i}
    t0 = time.time(); ttfb = None; buf = bytearray()
    with requests.post(f"{BASE}/v1/audio/speech", json=body, stream=True, timeout=900) as r:
        r.raise_for_status()
        for chunk in r.iter_content(8192):
            buf += chunk
            if ttfb is None and len(buf) > 44: ttfb = time.time() - t0  # first audio past WAV header
    wall = time.time() - t0
    path = f"{OUT}/req{i}.wav"
    open(path, "wb").write(buf)
    pcm = len(buf) - 44
    dur = pcm / 2 / 44100
    return i, ttfb, wall, dur

t0 = time.time()
with cf.ThreadPoolExecutor(N) as ex:
    res = list(ex.map(one, range(N)))
total = time.time() - t0
for i, ttfb, wall, dur in res:
    print(f"@@@ req{i}: ttfb={ttfb:.2f}s wall={wall:.1f}s audio={dur:.1f}s RTF={wall/dur:.2f}")
aud = sum(r[3] for r in res)
print(f"@@@ N={N} stream={STREAM}: total wall {total:.1f}s for {aud:.1f}s audio -> {aud/total:.2f}x realtime aggregate")
