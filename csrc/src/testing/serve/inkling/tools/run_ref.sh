#!/bin/bash
# agent-EN-f2: run inkling_ref_driver.py MODE on IN.jsonl inside the vLLM Inkling image on gpubox2 (CPU only, no card,
# no network), result to OUT.jsonl on NFS. gpubox2 does not mount NFS: files go over ssh into /home/densemax2/en-f2/io.
set -euo pipefail
MODE=$1; IN=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=$(basename "$IN" .jsonl)-$$
ssh gpubox2 "mkdir -p /home/densemax2/en-f2/io"
cat "$HERE/inkling_ref_driver.py" | ssh gpubox2 "cat > /home/densemax2/en-f2/io/inkling_ref_driver.py"
cat "$IN" | ssh gpubox2 "cat > /home/densemax2/en-f2/io/$TAG.in.jsonl"
ssh gpubox2 "docker run --rm --name en-f2-ref-$$ --label owner=agent-EN-f2 --network none --cpus 4 --memory 16g \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 -e VLLM_TARGET_DEVICE=cpu -e REF_TRACE=\${REF_TRACE:-} \
  -v /home/densemax2/en-f2/model:/model:ro -v /home/densemax2/en-f2/io:/io --entrypoint python3 \
  vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae -I /io/inkling_ref_driver.py $MODE /io/$TAG.in.jsonl /io/$TAG.out.jsonl" 2>&1 | grep -vE "No CUDA runtime|vllm._C|^\s*$" | tail -5
ssh gpubox2 "cat /home/densemax2/en-f2/io/$TAG.out.jsonl && rm -f /home/densemax2/en-f2/io/$TAG.in.jsonl /home/densemax2/en-f2/io/$TAG.out.jsonl" > "$OUT"
wc -l "$OUT"
