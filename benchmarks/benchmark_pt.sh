#!/usr/bin/env bash
set -euo pipefail
model=${1:?Usage: benchmark_pt.sh MODEL bf16|fp8|fp4}
recipe=${2:?Specify a recipe}
case "$recipe" in
  bf16) compute=bf16 ;;
  fp8) compute=fp8-hybrid ;;
  fp4) compute=nvfp4 ;;
  *) echo "Unknown recipe: $recipe" >&2; exit 2 ;;
esac
config=$(mktemp --suffix=.yaml)
trap 'rm -f "$config"' EXIT
cp examples/pt/qwen3.yaml "$config"
sed -i "s|^model: .*|model: ${model}|" "$config"
sed -i "s|^recipe: .*|recipe: ${compute}|" "$config"
sed -i "s|^output_dir: .*|output_dir: ./outputs/benchmark_pt_${recipe}|" "$config"
surogate pt "$config"
