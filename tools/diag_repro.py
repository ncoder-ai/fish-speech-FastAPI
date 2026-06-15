#!/usr/bin/env python3
"""Reproduce the live-server runaway under exact server conditions:
warmup generation first, then the trigger with max_new_tokens=0 (server path)
vs max_new_tokens=700 (diagnostic path). Isolates whether the trigger is the
unbounded cap and/or the warmup/recompile, not int8 quality."""
import os
import torch

from fish_speech.models.text2semantic.inference import generate_long, init_model

MODE = os.environ.get("FISH_QUANTIZE", "none")
DEV = "cuda:0"
TRIG = ("<|speaker:0|>[excited] Request number 1.\n"
        "<|speaker:0|>The quick brown fox jumps over the lazy dog near the riverbank at dawn.")


def gen(model, dot, text, max_new_tokens, label):
    mx = tot = 0
    for r in generate_long(model=model, device=DEV, decode_one_token=dot, text=text,
                           max_new_tokens=max_new_tokens, top_p=0.8, temperature=0.8,
                           compile=True, chunk_length=200, iterative_prompt=True):
        if r.action == "sample" and r.codes is not None:
            n = int(r.codes.size(1)); mx = max(mx, n); tot += n
    flag = "RUNAWAY" if mx > 1000 else "ok"
    print(f"[{label}] mnt={max_new_tokens} -> max_batch={mx} total={tot} {flag}", flush=True)


def main():
    print(f"=== load (mode={MODE}) ===", flush=True)
    model, dot = init_model("checkpoints/s2-pro", DEV, torch.bfloat16, compile=True)
    with torch.device(DEV):
        model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len,
                           dtype=next(model.parameters()).dtype)

    print("=== warmup (mnt=1024, like model_manager.warm_up) ===", flush=True)
    gen(model, dot, "<|speaker:0|>Hello world.", 1024, "warmup")

    print("=== trigger mnt=0 (SERVER path) x3 ===", flush=True)
    for i in range(3):
        gen(model, dot, TRIG, 0, f"trig_mnt0_{i}")

    print("=== trigger mnt=700 (diag path) x3 ===", flush=True)
    for i in range(3):
        gen(model, dot, TRIG, 700, f"trig_mnt700_{i}")


if __name__ == "__main__":
    main()
