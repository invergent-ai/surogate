#!/bin/bash

MODEL=$1
RECIPE=$2

if [ -z "$MODEL" ] || [ -z "$RECIPE" ]; then
    echo "Usage: $0 <model_id: Qwen/Qwen3-0.6B> <recipe: bf16 | fp8 | fp4>"
    exit 1
fi

rm -rf ./output/benchmark_pt_${RECIPE} /tmp/benchmark_${RECIPE}.yaml
cp examples/pt/qwen3-lora-${RECIPE}.yaml /tmp/benchmark_${RECIPE}.yaml
sed -i "s|^model: .*|model: ${MODEL}|" /tmp/benchmark_${RECIPE}.yaml
sed -i "s|^output_dir: .*|output_dir: ./output/benchmark_pt_${RECIPE}|" /tmp/benchmark_${RECIPE}.yaml
surogate pt --config /tmp/benchmark_${RECIPE}.yaml