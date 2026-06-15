#!/usr/bin/env bash
# Generate the 3 showcase scenes at int8 and int4 (bf16, compiled) for A/B vs
# the existing 16-bit scenes. Saves to quant_scenes/<q>/.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ONLY="multivoice_dialogue,emotion_narration,multivoice_emotion_combined"
OUTBASE=/tmp/quant_scenes

run() {
  local q="$1" log="logs/qscene_$1.log"
  : > "$log"
  echo "=========== scenes @ $q ==========="
  FISH_QUANTIZE="$q" $PY tools/openai_api_server.py \
      --device cuda:0 --listen 0.0.0.0:8770 --compile \
      --concurrency 1 --queue-timeout 180 >> "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 140); do
    grep -qE "Ready on http" "$log" 2>/dev/null && break
    grep -qiE "Traceback|ImportError|RuntimeError|CUDA error" "$log" 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 3
  done
  if ! grep -qE "Ready on http" "$log" 2>/dev/null; then
    echo "$q: FAILED to start"; tail -15 "$log"
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 3; return
  fi
  $PY tools/test_long_scenes.py --base-url http://localhost:8770 \
      --formats wav --out-dir "$OUTBASE/$q" --only "$ONLY"
  # graceful stop, escalate to -9
  kill "$pid" 2>/dev/null
  for i in 1 2 3 4 5 6; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 4
}

run int8
run int4
echo "QSCENES_DONE -> $OUTBASE"
ls -R "$OUTBASE"
