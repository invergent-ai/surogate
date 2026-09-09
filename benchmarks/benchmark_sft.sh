#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: benchmark_sft.sh MODEL bf16|fp8|qfp8|fp4|qfp4|qbnb}
recipe=${2:?Specify a recipe}
[[ "$recipe" == bnb ]] && recipe=qbnb
source_config="examples/sft/qwen3/qwen3-lora-${recipe}.yaml"
[[ -f "$source_config" ]] || { echo "Unknown recipe: $recipe" >&2; exit 2; }
config=$(mktemp --suffix=.yaml)
trap 'rm -f "$config"' EXIT
cp "$source_config" "$config"
sed -i "s|^model: .*|model: ${model}|" "$config"
sed -i "s|^output_dir: .*|output_dir: ./outputs/benchmark_sft_${recipe}|" "$config"
surogate sft "$config"
