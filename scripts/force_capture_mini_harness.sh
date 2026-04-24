#!/usr/bin/env bash
# Validation harness for the mem_eff attention port (see
# design/capture-safe-runtime-plan.md, Phase 4 attempt 1). Runs
# Gemma4-E2B LoRA training at bs=1, seq_len=1024 for 3 steps under
# normal and force-full-capture modes, then checks per-step losses
# match within bf16 tolerance (absolute diff < 5e-3).
#
# This is the smaller sibling of scripts/loss_diff_harness.sh — the
# full bs=2, seq_len=2048 config hits a 9.3 GiB stack-resize
# cudaMalloc cliff that's blocked on further memory-budget work.

set -uo pipefail
CFG=examples/sft/gemma4/gemma4-e2b-lora-mini.yaml
LOG_A=/tmp/force_capture_mini_a.log
LOG_B=/tmp/force_capture_mini_b.log
METRICS=/tmp/force_capture_mini_metrics_$$.jsonl
trap 'rm -f "$METRICS"' EXIT

run() {
    local label="$1" log="$2"; shift 2
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
        SUROGATE_METRICS_PATH="$METRICS" \
        env "$@" timeout 300 .venv/bin/surogate sft "$CFG" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "$label: CRASH rc=$rc"
        tail -10 "$log"
        return 1
    fi
}

extract() {
    local log="$1" i="$2"
    grep -E "^:: step\s+$i" "$log" | sed -n 's/.*loss \([0-9nan.]*\).*/\1/p' | head -1
}

echo "normal mode..."
run normal "$LOG_A" || exit 1
echo "force-capture mode..."
run force-capture "$LOG_B" SUROGATE_FORCE_FULL_GRAPH_CAPTURE=1 || exit 1

echo "per-step comparison:"
fail=0
for i in 0 1 2; do
    a=$(extract "$LOG_A" "$i")
    b=$(extract "$LOG_B" "$i")
    if [[ -z "$a" || -z "$b" ]]; then
        echo "  step $i: missing loss (normal=$a force=$b)"
        fail=1; continue
    fi
    diff=$(python3 -c "print(abs(float('$a') - float('$b')))")
    ok=$(python3 -c "print('ok' if abs(float('$a') - float('$b')) < 5e-3 else 'DIVERGED')")
    printf "  step %d: normal=%s force=%s diff=%s %s\n" "$i" "$a" "$b" "$diff" "$ok"
    [[ "$ok" != "ok" ]] && fail=1
done
if [[ $fail -eq 0 ]]; then
    echo "FORCE-CAPTURE MINI HARNESS: PASS"
    exit 0
fi
echo "FORCE-CAPTURE MINI HARNESS: FAIL"
exit 1
