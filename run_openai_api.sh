#!/usr/bin/env bash
# Launch the Fish-Speech S2-Pro OpenAI-compatible TTS API server.
# Usage:
#   ./run_openai_api.sh start [--fg]   # start detached (or foreground with --fg)
#   ./run_openai_api.sh stop
#   ./run_openai_api.sh status
#   ./run_openai_api.sh logs
set -euo pipefail

cd "$(dirname "$0")"

# Load box-local overrides if present (gitignored; mirrors the container's .env).
# This is how bare-metal hosts pin their config (quant, device, streaming, etc.)
# persistently. Not present inside the container image (.env is dockerignored), so
# the containerized path keeps using compose-provided env.
if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

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
# Optional pre-quantized weights: load these instead of re-quantizing every boot.
# If set but the file is missing, it's generated ONCE from the bf16 checkpoint
# below (using THIS box's torch/torchao/GPU, so it's always compatible) and then
# persists in the (mounted) checkpoints dir. export FISH_QUANTIZED_WEIGHTS=...
export FISH_QUANTIZED_WEIGHTS="${FISH_QUANTIZED_WEIGHTS:-}"
export FISH_QUANT_GROUPSIZE="${FISH_QUANT_GROUPSIZE:-128}"  # int4 group size
VOICES_DIR="${VOICES_DIR:-}"          # folder to auto-register + watch voices from (e.g. /path/to/voices)
# Auto-transcribe voices enrolled without a transcript (needed for cloning +
# multi-speaker voice_map binding). Read directly by tools/asr.py; exported here.
export FISH_AUTO_TRANSCRIBE="${FISH_AUTO_TRANSCRIBE:-1}"   # 1=on 0=off
export FISH_ASR_MODEL="${FISH_ASR_MODEL:-small}"          # small=multilingual; small.en=English-only
export FISH_ASR_DEVICE="${FISH_ASR_DEVICE:-cpu}"          # cpu | cuda:N
# Incremental decode for streaming requests: emit audio every N tokens so the
# first audio arrives in ~1.5s instead of ~12s on long turns. 0 = off. Read by
# tools/openai_api_server.py at import; only affects streaming requests.
export FISH_STREAM_CHUNK_TOKENS="${FISH_STREAM_CHUNK_TOKENS:-32}"
# Persist torch.compile (Inductor) + Triton kernel caches across restarts so a
# warm start skips the ~4-min CPU kernel recompile. Cache is keyed on
# torch/triton/GPU-arch/code; a bump triggers ONE cold recompile, fast after.
TORCH_CACHE_DIR="${FISH_TORCH_CACHE_DIR:-$PWD/.torch-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TORCH_CACHE_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TORCH_CACHE_DIR/triton}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
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

ensure_prequant() {
  # Generate pre-quantized weights once if a path is configured but missing.
  # Done with the local stack so the saved file is always loadable here.
  [[ -n "$FISH_QUANTIZED_WEIGHTS" && "$QUANTIZE" != "none" \
     && ! -f "$FISH_QUANTIZED_WEIGHTS" ]] || return 0
  echo "Pre-quantized weights missing ($FISH_QUANTIZED_WEIGHTS); generating once ($QUANTIZE)…"
  FISH_QUANTIZE="$QUANTIZE" DEV="$DEVICE" \
    "$PY" tools/quantize_save.py "$LLAMA_CKPT" "$FISH_QUANTIZED_WEIGHTS"
  echo "Pre-quantized weights ready: $FISH_QUANTIZED_WEIGHTS"
}

case "${1:-start}" in
  start)
    ensure_prequant
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
