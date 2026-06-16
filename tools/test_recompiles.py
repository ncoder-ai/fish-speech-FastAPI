#!/usr/bin/env python3
"""Definitive test: does the compiled decode recompile PER scene shape, or once?

Mimics prod: build ModelManager (loads + compiles + warms up on 'Hello world'
with NO references), then run distinct-length scenes WITH a reference and count
torch._dynamo recompiles per scene.

Verdict:
  - recompiles only on the FIRST with-ref scene -> warmup just didn't cover the
    reference path (one-time; fix = prime warmup with a ref). NOT per-shape.
  - recompiles on EACH new length -> genuine per-shape recompile (needs dynamic).
  - ~zero after warmup -> it's purely one-time cold compile + generation time.
"""
import os
os.environ.setdefault("TORCH_LOGS", "recompiles")  # must precede torch import

import sys
import time

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch._dynamo as dyn  # noqa: E402
from fish_speech.utils.schema import ServeTTSRequest  # noqa: E402
from tools.server.model_manager import ModelManager  # noqa: E402

DEV = os.environ.get("DEV", "cuda:3")
REF = os.environ.get("REF", "grace2")

print(f"[setup] loading + compiling + warmup (Hello world, no ref) on {DEV}…",
      flush=True)
mm = ModelManager(
    mode="tts", device=DEV, half=False, compile=True,
    llama_checkpoint_path="checkpoints/s2-pro",
    decoder_checkpoint_path="checkpoints/s2-pro/codec.pth",
    decoder_config_name="modded_dac_vq",
)
engine = mm.tts_inference_engine


def n_compiles():
    return dyn.utils.counters["stats"].get("unique_graphs", 0)


def run(tag, text, ref_id=REF):
    req = ServeTTSRequest(
        text=text, reference_id=ref_id, chunk_length=200, max_new_tokens=0,
        top_p=0.8, repetition_penalty=1.1, temperature=0.8, format="wav",
        streaming=False,
    )
    before = n_compiles()
    t0 = time.perf_counter()
    for r in engine.inference(req):
        if r.code == "error":
            print(f"  [{tag}] ERROR: {r.error}", flush=True)
    dt = time.perf_counter() - t0
    after = n_compiles()
    print(f"@@@ {tag:14} chars={len(text):4d}  wall={dt:6.1f}s  "
          f"NEW_COMPILES={after - before}", flush=True)


print(f"[setup] warmup done. unique_graphs so far = {n_compiles()}", flush=True)
print("@@@ ==== counting per-scene recompiles below ====", flush=True)

scenes = [
    ("ref_first", "<|speaker:0|>Hi there friend."),
    ("len_A", "<|speaker:0|>" + "This is one distinct prompt length to test. " * 3),
    ("len_B", "<|speaker:0|>" + "A clearly different and longer prompt length here. " * 9),
    ("len_C", "<|speaker:0|>" + "Yet another unique length, longer still than before. " * 18),
    ("len_A_again", "<|speaker:0|>" + "This is one distinct prompt length to test. " * 3),
    ("multispk", "<|speaker:0|>Where are you going?\n<|speaker:1|>To the market, want anything?"),
]
for tag, txt in scenes:
    run(tag, txt)

print(f"@@@ FINAL unique_graphs = {n_compiles()}", flush=True)
print("@@@ If NEW_COMPILES is 0 for len_B/len_C/multispk, there is NO per-shape "
      "recompile — only one-time compile + generation time.", flush=True)
