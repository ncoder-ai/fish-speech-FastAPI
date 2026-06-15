#!/usr/bin/env python3
"""Send a ~450-word 3-speaker scene and report duration + F0 spread.

Trim events ("Context window full; dropped oldest turn pair") must be read from
the server logs separately. Usage: BASE=http://host:port test_long_scene.py v0 v1 v2
"""
import io
import os
import sys

import librosa
import numpy as np
import requests
import soundfile as sf

BASE = os.environ.get("BASE", "http://localhost:8770")
v0 = sys.argv[1] if len(sys.argv) > 1 else "en-ca-f-clara"
v1 = sys.argv[2] if len(sys.argv) > 2 else "en-au-m-william"
v2 = sys.argv[3] if len(sys.argv) > 3 else "en-ca-m-liam"

# ~450-word scene: spk0 female (dialogue), spk1 male (narration), spk2 male (dialogue)
turns = [
    (1, "The rain had not stopped for three days, and the small harbor town wore the grey like an old coat."),
    (0, "Do you really think the ferry will run in this weather?"),
    (2, "They said it would. The captain has crossed worse than this, believe me."),
    (1, "She pulled her scarf tighter and looked out at the water, where the waves climbed over the breakwater."),
    (0, "I just want to be home before dark. My mother worries when the storms come in like this."),
    (2, "We will be fine. Forty minutes across, and then a short walk up the hill, and you are at your door."),
    (1, "He said it with the easy confidence of a man who had made the crossing a hundred times."),
    (0, "You always say that, and then something goes wrong and we are stuck in the cafe for hours."),
    (2, "That happened once. Once! And the coffee was good, you cannot deny that the coffee was very good."),
    (1, "A gull cried somewhere above them, invisible in the low cloud, and the ferry horn answered from the dock."),
    (0, "There it is. Come on, before all the dry seats are taken by the people from the bus."),
    (2, "After you. Watch the step, it gets slick right at the edge where the planks meet the ramp."),
    (1, "They moved together toward the gangway, two friends and the sea, and the town faded behind the rain."),
    (0, "When we get across, I am buying. You carried my bag the whole way down the hill without complaining."),
    (2, "I will not argue with that. A warm drink and a window seat sound like exactly the right reward."),
    (1, "And so the little ferry pushed out into the grey, steady against the swell, carrying them home at last."),
]
text = "\n".join(f"<|speaker:{s}|>{t}" for s, t in turns)
words = sum(len(t.split()) for _, t in turns)
print(f"scene: {len(turns)} turns, ~{words} words, voices {v0}/{v1}/{v2}")

req = {"input": text, "voice": v0, "voice_map": {"0": v0, "1": v1, "2": v2},
       "response_format": "wav", "seed": 7}
r = requests.post(f"{BASE}/v1/audio/speech", json=req, timeout=1800)
r.raise_for_status()
a, sr = sf.read(io.BytesIO(r.content))
a = a.mean(1) if a.ndim > 1 else a
a = a.astype(np.float32)
out = "/tmp/long_scene.wav"
sf.write(out, a, sr)
dur = len(a) / sr
print(f"audio: {dur:.1f}s ({dur/60:.1f} min), saved {out}")

f0, vo, _ = librosa.pyin(a, fmin=70, fmax=350, sr=sr, frame_length=2048, hop_length=512)
v = f0[vo & ~np.isnan(f0)]
print(f"F0 median={np.median(v):.0f}Hz  male<160={100*np.mean(v<160):.0f}%  "
      f"female>=160={100*np.mean(v>=160):.0f}%  (mixed = voices switching)")
