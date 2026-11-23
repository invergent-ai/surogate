#!/bin/bash
# Run inkling_ref_driver.py MODE on IN.jsonl inside the vLLM Inkling image on a remote Docker host (CPU only, no card,
# no network), result to OUT.jsonl locally. Files go over ssh into $REF_DIR/io on that host.
# Required environment: REF_HOST (ssh destination with Docker and the vLLM Inkling image) and REF_DIR (a directory on
# REF_HOST holding the model under model/; io/ is created beside it).
set -euo pipefail
MODE=$1; IN=$2; OUT=$3
: "${REF_HOST:?set REF_HOST to the ssh destination that runs the vLLM Inkling image}"
: "${REF_DIR:?set REF_DIR to the directory on REF_HOST that holds model/ (io/ is created there)}"
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$(basename "$IN" .jsonl)-$$
ssh "$REF_HOST" "mkdir -p $REF_DIR/io"
cat "$HERE/inkling_ref_driver.py" | ssh "$REF_HOST" "cat > $REF_DIR/io/inkling_ref_driver.py"
cat "$IN" | ssh "$REF_HOST" "cat > $REF_DIR/io/$TAG.in.jsonl"
ssh "$REF_HOST" "docker run --rm --name inkling-ref-$$ --label purpose=surogate-inkling-ref --network none --cpus 4 --memory 16g \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e VLLM_TARGET_DEVICE=cpu -e REF_TRACE=\${REF_TRACE:-} \
  -v $REF_DIR/model:/model:ro -v $REF_DIR/io:/io --entrypoint python3 \
  vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae -I /io/inkling_ref_driver.py $MODE /io/$TAG.in.jsonl /io/$TAG.out.jsonl" 2>&1 | grep -vE "No CUDA runtime|vllm._C|^\s*$" | tail -5
ssh "$REF_HOST" "cat $REF_DIR/io/$TAG.out.jsonl && rm -f $REF_DIR/io/$TAG.in.jsonl $REF_DIR/io/$TAG.out.jsonl" > "$OUT"
wc -l "$OUT"
