#!/usr/bin/env bash
# Launch the Fish-Speech S2-Pro OpenAI-compatible TTS API server.
# Usage:
#   ./run_openai_api.sh start [--fg]   # start detached (or foreground with --fg)
#   ./run_openai_api.sh stop
#   ./run_openai_api.sh status
#   ./run_openai_api.sh logs
set -euo pipefail

cd "$(dirname "$0")"

# ---- Config (override via environment) -------------------------------------
DEVICE="${FISH_DEVICE:-cuda:3}"
LISTEN="${FISH_LISTEN:-0.0.0.0:8770}"
LLAMA_CKPT="${FISH_LLAMA_CKPT:-checkpoints/s2-pro}"
DECODER_CKPT="${FISH_DECODER_CKPT:-checkpoints/s2-pro/codec.pth}"
DECODER_CFG="${FISH_DECODER_CFG:-modded_dac_vq}"
COMPILE="${FISH_COMPILE:-1}"          # 1 = torch.compile (faster steady state)
HALF="${FISH_HALF:-1}"                # 1 = fp16 (auto-disabled for int4)
QUANTIZE="${FISH_QUANTIZE:-none}"     # none | int8 (~12.8GB) | int4 (~11GB, bf16)
MAX_SEQ_LEN="${FISH_MAX_SEQ_LEN:-0}"  # 0 = model default (8192); e.g. 4096 = lower peak VRAM
VOICES_DIR="${VOICES_DIR:-}"          # folder to auto-register + watch voices from (e.g. /home/nishant/apps/voices)
# export FISH_API_KEY=... to require a bearer token.

PY=".venv/bin/python"
LOG="logs/server.log"
PIDFILE="logs/server.pid"
mkdir -p logs

build_args() {
  ARGS=(tools/openai_api_server.py
        --device "$DEVICE" --listen "$LISTEN"
        --llama-checkpoint-path "$LLAMA_CKPT"
        --decoder-checkpoint-path "$DECODER_CKPT"
        --decoder-config-name "$DECODER_CFG")
  [[ "$HALF" == "1" ]] && ARGS+=(--half)
  [[ "$COMPILE" == "1" ]] && ARGS+=(--compile)
  [[ "$QUANTIZE" != "none" ]] && ARGS+=(--quantize "$QUANTIZE")
  [[ "$MAX_SEQ_LEN" != "0" ]] && ARGS+=(--max-seq-len "$MAX_SEQ_LEN")
  [[ -n "$VOICES_DIR" ]] && ARGS+=(--voices-dir "$VOICES_DIR")
  return 0   # never let a false [[ ]] && ... above abort under `set -e`
}

case "${1:-start}" in
  start)
    build_args
    if [[ "${2:-}" == "--fg" ]]; then
      exec "$PY" "${ARGS[@]}"
    fi
    setsid nohup "$PY" "${ARGS[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started (device=$DEVICE listen=$LISTEN compile=$COMPILE). Tailing log until ready..."
    for i in $(seq 1 90); do
      grep -q "Application startup complete" "$LOG" 2>/dev/null && { echo "READY: http://$LISTEN"; exit 0; }
      grep -qiE "Traceback|address already in use" "$LOG" 2>/dev/null && { echo "FAILED — see $LOG"; tail -5 "$LOG"; exit 1; }
      sleep 4
    done
    echo "Timed out waiting for startup; check $LOG"; exit 1
    ;;
  stop)
    pkill -f "openai_api_server.py" && echo "Stopped." || echo "Not running."
    ;;
  status)
    if pgrep -f "openai_api_server.py" >/dev/null; then
      echo "RUNNING"; curl -s "http://${LISTEN/0.0.0.0/localhost}/health" || true; echo
    else echo "STOPPED"; fi
    ;;
  logs) tail -f "$LOG" ;;
  *) echo "usage: $0 {start [--fg]|stop|status|logs}"; exit 1 ;;
esac
