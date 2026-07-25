#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FUGU_CONDUCTOR_BASE_MODEL_PATH="${FUGU_CONDUCTOR_BASE_MODEL_PATH:-${FUGU_CONDUCTOR_MODEL_PATH:-}}"
: "${FUGU_CONDUCTOR_BASE_MODEL_PATH:?set FUGU_CONDUCTOR_BASE_MODEL_PATH to the local base model}"

VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"
FUGU_CONDUCTOR_PORT="${FUGU_CONDUCTOR_PORT:-8010}"
FUGU_CONDUCTOR_NAME="${FUGU_CONDUCTOR_NAME:-fugu-27b-conductor}"
FUGU_CONDUCTOR_BASE_NAME="${FUGU_CONDUCTOR_BASE_NAME:-fugu-27b-base}"
FUGU_CONDUCTOR_ADAPTER_PATH="${FUGU_CONDUCTOR_ADAPTER_PATH:-$REPO_ROOT/scratchpad/fugu_27b_ale_accepted_r2}"

exec "$VLLM_BIN" serve "$FUGU_CONDUCTOR_BASE_MODEL_PATH" \
  --port "$FUGU_CONDUCTOR_PORT" \
  --served-model-name "$FUGU_CONDUCTOR_BASE_NAME" \
  --tensor-parallel-size "${FUGU_CONDUCTOR_TP:-2}" \
  --max-model-len "${FUGU_CONDUCTOR_CONTEXT:-8192}" \
  --gpu-memory-utilization "${FUGU_CONDUCTOR_GPU_UTILIZATION:-0.85}" \
  --max-num-seqs "${FUGU_CONDUCTOR_MAX_SEQS:-64}" \
  --enable-prefix-caching \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --enable-lora \
  --lora-modules "$FUGU_CONDUCTOR_NAME=$FUGU_CONDUCTOR_ADAPTER_PATH" \
  --enforce-eager \
  "$@" \
  --logprobs-mode processed_logprobs
