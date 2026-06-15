#!/usr/bin/env python3
"""
Diagnose int8/int4 runaway: does the quantized model fail to emit the stop token?
Loads the LLM only (no codec), runs generate_long on a fixed trigger across a
temperature/seed sweep, and reports the max tokens any batch generated.
A "runaway" = a batch that hits the cap instead of stopping (~150 normally).

  CUDA_VISIBLE_DEVICES=3 FISH_QUANTIZE=none  uv-or-venv python tools/diag_quant.py
  CUDA_VISIBLE_DEVICES=3 FISH_QUANTIZE=int8  ...
  CUDA_VISIBLE_DEVICES=3 FISH_QUANTIZE=int4  ...
"""
import os
import torch

from fish_speech.models.text2semantic.inference import generate_long, init_model

MODE = os.environ.get("FISH_QUANTIZE", "none")
CAP = int(os.environ.get("DIAG_CAP", "700"))
DEV = "cuda:0"  # via CUDA_VISIBLE_DEVICES

# The exact input that ran away on the live int8 server (server tags each
# sentence with <|speaker:0|>).
TEXT = (
    "<|speaker:0|>[excited] Request number one.\n"
    "<|speaker:0|>The quick brown fox jumps over the lazy dog near the riverbank at dawn."
)


def main():
    print(f"=== loading model (mode={MODE}) ===", flush=True)
    model, decode_one_token = init_model("checkpoints/s2-pro", DEV, torch.bfloat16,
                                         compile=True)

    def trial(seed, temperature, top_p):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        batches = []
        for r in generate_long(model=model, device=DEV,
                               decode_one_token=decode_one_token, text=TEXT,
                               max_new_tokens=CAP, top_p=top_p,
                               temperature=temperature, compile=True,
                               chunk_length=200, iterative_prompt=True):
            if r.action == "sample" and r.codes is not None:
                batches.append(int(r.codes.size(1)))
        return batches

    print(f"=== sweep (mode={MODE}, cap={CAP}; runaway if max batch >= {CAP-20}) ===",
          flush=True)
    for temperature, top_p in [(0.8, 0.8), (0.3, 0.9), (0.1, 1.0)]:
        maxes = []
        runaways = 0
        for s in range(4):
            b = trial(s, temperature, top_p)
            mx = max(b) if b else 0
            maxes.append(mx)
            if mx >= CAP - 20:
                runaways += 1
        print(f"MODE={MODE} temp={temperature} top_p={top_p}: "
              f"max_batch_per_seed={maxes}  runaways={runaways}/4", flush=True)


if __name__ == "__main__":
    main()
