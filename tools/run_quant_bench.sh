#!/usr/bin/env bash
# Measure VRAM + RTF for baseline / int8 / int4 weight-only quant on GPU 3.
# bf16 throughout (int4 tinygemm needs bf16). Compiled (fullgraph) each time.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=.venv/bin/python
RESULTS=/tmp/quant_results.txt
: > "$RESULTS"

run_cfg() {
  local label="$1" quant="$2"
  local log="logs/quant_${label}.log"
  : > "$log"
  echo "=========== $label (FISH_QUANTIZE='${quant}') ==========="
  FISH_QUANTIZE="$quant" $PY tools/openai_api_server.py \
      --device cuda:3 --listen 0.0.0.0:8770 --compile \
      --concurrency 1 --queue-timeout 120 >> "$log" 2>&1 &
  local pid=$!

  # wait for ready or failure
  local ready=0
  for i in $(seq 1 120); do
    if grep -qE "Ready on http" "$log" 2>/dev/null; then ready=1; break; fi
    if grep -qiE "Traceback|Error compiling|raise|CUDA error" "$log" 2>/dev/null; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 3
  done

  if [ "$ready" != "1" ]; then
    echo "$label: FAILED to start (compile/quant error)" | tee -a "$RESULTS"
    echo "---- last log ----"; tail -15 "$log"
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    sleep 3
    return
  fi

  # benchmark
  $PY tools/bench_quant.py http://localhost:8770 "$label" | tee -a "$RESULTS"
  # peak VRAM: server's reserved log + live nvidia-smi
  local reserved toks smi
  reserved=$(grep -oE "GPU Memory used: [0-9.]+ GB" "$log" | tail -1)
  toks=$(grep -oE "[0-9.]+ tokens/sec" "$log" | tail -1)
  smi=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n '4p')
  echo "$label: nvidia-smi=${smi}MiB | ${reserved} | ${toks}" | tee -a "$RESULTS"

  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 4
}

run_cfg baseline ""
run_cfg int8 int8
run_cfg int4 int4

echo ""
echo "================== SUMMARY =================="
cat "$RESULTS"
echo "QUANT_BENCH_DONE"
