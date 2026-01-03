#!/bin/bash

MODEL=$1
RECIPE=$2

if [ -z "$MODEL" ] || [ -z "$RECIPE" ]; then
    echo "Usage: $0 <model_id: Qwen/Qwen3-0.6B> <recipe: bf16 | fp8 | qfp8 | fp4 | qfp4>"
    exit 1
fi

rm -rf ./output /tmp/benchmark_${RECIPE}.yaml
cp examples/sft/qwen3-lora-${RECIPE}.yaml /tmp/benchmark_${RECIPE}.yaml
sed -i "s|^model: .*|model: ${MODEL}|" /tmp/benchmark_${RECIPE}.yaml
surogate sft --config /tmp/benchmark_${RECIPE}.yaml