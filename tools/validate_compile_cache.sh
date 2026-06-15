#!/usr/bin/env bash
# A/B test: does persisting the Inductor/Triton cache cut torch.compile time?
# Run 1 = cold cache (recompiles). Run 2 = warm cache (should load kernels).
# Uses a free GPU + non-prod port; kills each server as soon as it logs the
# compile time so it doesn't squat the GPU.
set -u
cd "$(dirname "$0")/.."

DEV="${DEV:-cuda:2}"
PORT="${PORT:-8771}"
CACHE="/tmp/fish_cc_cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
PY=".venv/bin/python"

run() {
  local tag="$1"
  local log="/tmp/fish_cc_${tag}.log"
  : > "$log"
  setsid nohup "$PY" tools/openai_api_server.py \
    --device "$DEV" --listen "0.0.0.0:$PORT" \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq --half --compile \
    > "$log" 2>&1 &
  local pid=$!
  local t=0
  while [ $t -lt 480 ]; do
    if grep -q "Compilation time" "$log" 2>/dev/null; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[$tag] DIED early"; tail -5 "$log"; return 1; fi
    sleep 3; t=$((t+3))
  done
  local ct; ct=$(grep -oE "Compilation time: [0-9.]+ seconds" "$log" | head -1)
  echo "[$tag] $ct (waited ${t}s)"
  kill -9 "$pid" 2>/dev/null
  # kill any children (uvicorn/model workers) on this port's process group
  pkill -9 -f "openai_api_server.py --device $DEV --listen 0.0.0.0:$PORT" 2>/dev/null
  sleep 5
}

echo "=== cold cache (wiped) ==="; rm -rf "$CACHE"; run cold
echo "=== warm cache (reuse) ==="; run warm
echo "=== cache size ==="; du -sh "$CACHE" 2>/dev/null
