#!/bin/bash

MODEL=$1
RECIPE=$2

if [ -z "$MODEL" ] || [ -z "$RECIPE" ]; then
    echo "Usage: $0 <model_id: Qwen/Qwen3-0.6B> <recipe: bf16 | fp8 | qfp8 | fp4 | qfp4 | bnb>"
    exit 1
fi

rm -rf ./output/benchmark_sft_${RECIPE} /tmp/bench_${USER}_${RECIPE}.yaml
cp examples/sft/qwen3-lora-${RECIPE}.yaml /tmp/bench_${USER}_${RECIPE}.yaml
sed -i "s|^model: .*|model: ${MODEL}|" /tmp/bench_${USER}_${RECIPE}.yaml
sed -i "s|^output_dir: .*|output_dir: ./output/benchmark_sft_${RECIPE}|" /tmp/bench_${USER}_${RECIPE}.yaml
surogate sft /tmp/bench_${USER}_${RECIPE}.yaml