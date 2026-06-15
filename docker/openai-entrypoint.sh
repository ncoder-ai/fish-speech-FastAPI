#!/usr/bin/env bash
# Container entrypoint for the Fish Speech S2-Pro OpenAI-compatible API server.
# All tunables come from the environment (see docker-compose.yml / .env). The
# server itself is fully env-driven via run_openai_api.sh.
set -euo pipefail
cd /app

CKPT_DIR="${FISH_LLAMA_CKPT:-checkpoints/s2-pro}"

# Checkpoints are NOT baked into the image (11GB). Either mount them at
# /app/checkpoints, or let this download them once into the mounted volume.
if [ ! -f "${CKPT_DIR}/codec.pth" ]; then
  echo "[entrypoint] No checkpoints at ${CKPT_DIR} — downloading fishaudio/s2-pro (~11GB, one time)…"
  mkdir -p "${CKPT_DIR}"
  uv run python - "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("fishaudio/s2-pro", local_dir=sys.argv[1])
PY
  echo "[entrypoint] Checkpoint download complete."
fi

# Voices: if a voices dir is mounted/configured, the server auto-registers and
# watches it (FISH_VOICES_SCAN_INTERVAL). Log what we found.
if [ -n "${VOICES_DIR:-}" ] && [ -d "${VOICES_DIR}" ]; then
  echo "[entrypoint] Auto-registering voices from ${VOICES_DIR} ($(find "${VOICES_DIR}" -maxdepth 1 -type f 2>/dev/null | wc -l) files)."
fi

echo "[entrypoint] Starting server: device=${FISH_DEVICE:-cuda:0} listen=${FISH_LISTEN:-0.0.0.0:8770} quantize=${FISH_QUANTIZE:-none} compile=${FISH_COMPILE:-1}"
exec bash run_openai_api.sh start --fg
