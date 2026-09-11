#!/usr/bin/env bash
# Run from the repository root. Each scenario starts ONE foreground process.
set -euo pipefail
scenario=${1:-chat}
if [[ $# -gt 0 ]]; then shift; fi
case "$scenario" in
  -h|--help)
    echo 'Usage: bash examples/serve/launch.sh SCENARIO [extra engine flags]'
    echo 'Scenarios: chat concurrent lora vision offload moe multi-gpu multi-model mtp dflash generate embeddings'
    echo 'See examples/serve/README.md for model inputs and prerequisites.'
    exit 0 ;;
esac
server=(--host 127.0.0.1 --port "${PORT:-8080}" --served-model-name demo)
if [[ -n ${SUROGATE_API_KEY:-} ]]; then server+=(--api-key "$SUROGATE_API_KEY"); fi
case "$scenario" in
  chat)
    exec surogate serve "${MODEL:-Qwen/Qwen3-0.6B}" "${server[@]}" \
      --max-model-len 4096 --kv-capacity auto --no-thinking \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml "$@" ;;
  concurrent)
    mkdir -p outputs/serve
    exec surogate serve "${MODEL:-Qwen/Qwen3.5-0.8B}" "${server[@]}" \
      --max-model-len 4096 --kv-capacity auto --kv-cache-dtype fp8 \
      --max-num-seqs 16 --max-num-batched-tokens 2048 \
      --max-pending-requests 64 --pending-timeout-ms 60000 \
      --enable-prefix-caching \
      --request-log-jsonl outputs/serve/requests.jsonl "$@" ;;
  lora)
    adapter=${ADAPTER:-./outputs/training/runtime-lora}
    exec surogate serve "${MODEL:-Qwen/Qwen3-0.6B}" "${server[@]}" \
      --max-model-len 4096 --kv-capacity auto --enable-lora \
      --max-loras 2 --max-lora-rank 32 --lora-modules "tuned=$adapter" "$@" ;;
  vision)
    exec surogate serve "${MODEL:-Qwen/Qwen3-VL-2B-Instruct}" "${server[@]}" \
      --vision --max-model-len 8192 --kv-capacity auto \
      --media-cache-mib 1024 --media-live-mib 2048 --media-preprocess-threads 4 "$@" ;;
  offload)
    exec surogate serve "${MODEL:-Qwen/Qwen3.5-4B}" "${server[@]}" \
      --gpu-layers "${GPU_LAYERS:-8}" --max-model-len 4096 --kv-capacity auto "$@" ;;
  moe)
    : "${MODEL:?Set MODEL to a supported MoE checkpoint or local GGUF (first shard for split files)}"
    exec surogate serve "$MODEL" "${server[@]}" --host-moe-layers auto \
      --expert-slots 0 --host-expert-bank auto --cpu-moe-share auto \
      --cpu-moe-prefill-share 0.25 --max-model-len 4096 --kv-capacity auto "$@" ;;
  multi-gpu)
    exec surogate serve "${MODEL:-Qwen/Qwen3.5-4B}" "${server[@]}" \
      --devices "${DEVICES:-0,1}" --max-model-len 4096 --kv-capacity auto "$@" ;;
  multi-model)
    : "${SECOND_MODEL:?Set SECOND_MODEL to a prepared .sinfer file; see README}"
    exec surogate serve "${MODEL:-Qwen/Qwen3-0.6B}" "${server[@]}" \
      --model "small=$SECOND_MODEL,priority=low,max-model-len=4096,max-num-seqs=4" \
      --model-priority high --enable-sleep-mode --elastic-kv-overcommit \
      --kv-capacity auto --max-model-len 4096 --max-num-seqs 4 "$@" ;;
  mtp)
    : "${MODEL:?Set MODEL to a supported checkpoint containing MTP weights}"
    exec surogate serve "$MODEL" "${server[@]}" --spec mtp --draft-tokens 3 \
      --spec-max-lanes 1 --max-model-len 4096 --kv-capacity auto "$@" ;;
  dflash)
    : "${MODEL:?Set MODEL to an artifact prepared with a compatible DFlash drafter; see README}"
    exec surogate serve "$MODEL" "${server[@]}" --spec dflash --draft-tokens 7 \
      --kv-cache-dtype bf16 --max-model-len 4096 --kv-capacity auto "$@" ;;
  generate)
    exec surogate serve --generate "${MODEL:-Qwen/Qwen3-0.6B}" \
      --prompt 'Explain gradient accumulation in two sentences.' --max-new 128 \
      --max-context 4096 --kv-dtype bf16 --no-thinking "$@" ;;
  embeddings)
    : "${MODEL:?Set MODEL to an EmbeddingGemma GGUF}"
    : "${FRONTEND:?Set FRONTEND to its Hugging Face tokenizer directory}"
    exec surogate serve --embed "$MODEL" --frontend "$FRONTEND" \
      --host 127.0.0.1 --port "${PORT:-8413}" --device "${DEVICE:-0}" "$@" ;;
  *) echo "Unknown scenario: $scenario (use --help)" >&2; exit 2 ;;
esac
