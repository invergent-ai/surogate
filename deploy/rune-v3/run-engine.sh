#!/usr/bin/env bash
set -euo pipefail
# The environment file names paths and an immutable local image ID; it contains no key value.
source "${RUNE_CONFIG:-/etc/rune-v3/environment}"
: "${RUNE_RUNTIME_IMAGE:?}" "${RUNE_RELEASE:?}" "${RUNE_ARTIFACT:?}" "${RUNE_API_KEY_FILE:?}"
: "${RUNE_GPU_LOCK:?}" "${RUNE_UID:?}" "${RUNE_GID:?}"
test -r "$RUNE_API_KEY_FILE"
test -f "$RUNE_ARTIFACT"
spec_args=()
spec_env=()
case "${RUNE_SPEC:-none}" in
    none) ;;
    dflash)
        spec_args=(--spec dflash --draft-tokens "${RUNE_DRAFT_TOKENS:-7}")
        spec_env=(--env SUROGATE_SERVE_DFLASH_PACKED_PREFILL=1)
        ;;
    *) printf 'RUNE_SPEC must be none or dflash\n' >&2; exit 2 ;;
esac
exec flock -n "$RUNE_GPU_LOCK" docker run --rm --init --name rune-v3 \
    --gpus all --network host --read-only --cap-drop ALL --security-opt no-new-privileges \
    --user "$RUNE_UID:$RUNE_GID" --tmpfs /tmp:rw,nosuid,nodev,size=512m \
    --mount "type=bind,src=$RUNE_RELEASE,dst=/opt/rune,readonly" \
    --mount "type=bind,src=$RUNE_ARTIFACT,dst=/models/rune-v3.sinfer,readonly" \
    --mount "type=bind,src=$RUNE_API_KEY_FILE,dst=/run/secrets/rune-api-key,readonly" \
    --env LD_LIBRARY_PATH=/opt/rune --env CUDA_MODULE_LOADING=LAZY \
    "${spec_env[@]}" \
    "$RUNE_RUNTIME_IMAGE" /opt/rune/surogate-engine /models/rune-v3.sinfer \
    --host 127.0.0.1 --port 8460 --served-model-name rune-v3 \
    --api-key-file /run/secrets/rune-api-key \
    "${spec_args[@]}" \
    --vision --gemma-image-tokens 1120 --max-num-seqs 64 --max-model-len 16384 \
    --kv-capacity auto --max-num-batched-tokens 8192 \
    --max-request-mib 16 --media-live-mib 512 --media-cache-mib 1024 \
    --rate-limit-rps 16 --rate-limit-burst 16 --max-inflight-requests 16 \
    --max-thinking-requests 2 --max-image-requests 4 --max-pending-requests 8 --pending-timeout-ms 2000 \
    --default-max-tokens 512
