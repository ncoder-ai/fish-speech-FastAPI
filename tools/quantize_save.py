#!/usr/bin/env python3
"""Quantize the S2-Pro backbone ONCE and save the quantized weights.

The server otherwise re-runs torchao weight-only quant in CPU RAM on every boot
(~slow, and on low-RAM boxes it thrashes swap). This produces the quantized
state_dict once so the server can load it straight to GPU (set
FISH_QUANTIZED_WEIGHTS=<out>) and skip the per-boot re-quant. Same VRAM as
quantizing live; the win is startup time + CPU RAM.

Usage:
  FISH_QUANTIZE=int4 DEV=cuda:0 python tools/quantize_save.py [checkpoint_dir] [out_path]

int4 ~= half the backbone weight bytes of int8 -> lowest VRAM (Ampere-safe
tile_packed path). int8 is the conservative/validated option.
"""
import os
import sys
from pathlib import Path

import pyrootutils
import torch

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from loguru import logger  # noqa: E402

from fish_speech.models.text2semantic.inference import _apply_quantization  # noqa: E402
from fish_speech.models.text2semantic.llama import DualARTransformer  # noqa: E402

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/s2-pro"
mode = os.environ.get("FISH_QUANTIZE", "int4").strip().lower()
gs = os.environ.get("FISH_QUANT_GROUPSIZE", "128")
dev = os.environ.get("DEV", "cuda:0")
if mode not in ("int8", "int4"):
    raise SystemExit(f"FISH_QUANTIZE must be int8|int4, got {mode!r}")

default_out = Path(ckpt) / (
    f"model.{mode}.g{gs}.pt" if mode == "int4" else f"model.{mode}.pt")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out

logger.info(f"Loading bf16 model from {ckpt} on {dev} …")
model = DualARTransformer.from_pretrained(ckpt, load_weights=True)
model = model.to(device=dev, dtype=torch.bfloat16)

os.environ["FISH_QUANTIZE"] = mode  # _apply_quantization reads this
os.environ["FISH_QUANT_GROUPSIZE"] = str(gs)
_apply_quantization(model)  # quantizes layers.* in place (on GPU = fast)

logger.info(f"Saving quantized state_dict -> {out}")
# Move to CPU for a portable, device-independent file.
sd = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in model.state_dict().items()}
torch.save(sd, out)
sz = out.stat().st_size / 1e9
logger.info(f"Done: {out} ({sz:.2f} GB). "
            f"Set FISH_QUANTIZED_WEIGHTS={out} (with FISH_QUANTIZE={mode}) to load it.")
