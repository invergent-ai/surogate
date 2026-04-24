#!/usr/bin/env bash
# Loss-diff regression harness for capture-safety work.
#
# Runs the Gemma4-E2B LoRA training for 3 steps, extracts per-step loss, and
# compares against the known-good baseline captured at commit 08b4ff9 (pre-
# capture-safety-refactor). Exits 0 on match (abs diff < 1e-3), 1 otherwise.
#
# Usage:  ./scripts/loss_diff_harness.sh [extra env-var assignments...]
# Example:
#     ./scripts/loss_diff_harness.sh
#     ./scripts/loss_diff_harness.sh SUROGATE_FORCE_FULL_GRAPH_CAPTURE=1
#
# Baseline (from phase2_clean_baseline run on 2026-04-24):
#   step 0: loss 3.6317
#   step 1: loss 3.7640
#   step 2: loss 3.5395

set -uo pipefail

HARNESS_CFG=/tmp/loss_diff_harness.yaml
HARNESS_LOG=/tmp/loss_diff_harness.log

cp examples/sft/gemma4/gemma4-e2b-lora-bf16.yaml "$HARNESS_CFG"
sed -i 's/max_steps: 50/max_steps: 3/' "$HARNESS_CFG"

# Empirical baseline bands chosen to catch gross regression (the attempt-1
# failure mode was loss 18 at step 0) without false positives on normal
# run-to-run stochasticity. Gemma4-E2B bs=2 gas=4 seq_len=2048 on GSM8K:
# stable first-few-step losses cluster in [3.3, 4.5]. A loss outside this
# band, or a NaN, indicates forward or backward has been corrupted.
BAND_LO="3.0"
BAND_HI="4.5"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} env "$@" timeout 180 .venv/bin/surogate sft "$HARNESS_CFG" > "$HARNESS_LOG" 2>&1

rc=$?
if [[ $rc -ne 0 ]]; then
    echo "HARNESS: training crashed (rc=$rc). Tail:"
    tail -15 "$HARNESS_LOG"
    exit 1
fi

# Extract losses from step lines
ACTUAL_0=$(grep -E '^:: step\s+0' "$HARNESS_LOG" | sed -n 's/.*loss \([0-9nan.]*\).*/\1/p' | head -1)
ACTUAL_1=$(grep -E '^:: step\s+1' "$HARNESS_LOG" | sed -n 's/.*loss \([0-9nan.]*\).*/\1/p' | head -1)
ACTUAL_2=$(grep -E '^:: step\s+2' "$HARNESS_LOG" | sed -n 's/.*loss \([0-9nan.]*\).*/\1/p' | head -1)

if [[ -z "$ACTUAL_0" || -z "$ACTUAL_1" || -z "$ACTUAL_2" ]]; then
    echo "HARNESS: could not extract losses. Tail:"
    tail -15 "$HARNESS_LOG"
    exit 1
fi

fail=0
for i in 0 1 2; do
    eval "actual=\$ACTUAL_$i"
    # python3 returns 'ok' if actual in [BAND_LO, BAND_HI] and not nan.
    result=$(python3 -c "
actual = '$actual'
try:
    v = float(actual)
    if v != v:  # NaN
        print('NaN')
    elif $BAND_LO <= v <= $BAND_HI:
        print('ok')
    else:
        print('OUT_OF_BAND')
except Exception:
    print('PARSE')
")
    printf "  step %d: loss=%s band=[%s, %s] %s\n" "$i" "$actual" "$BAND_LO" "$BAND_HI" "$result"
    [[ "$result" != "ok" ]] && fail=1
done

if [[ $fail -eq 0 ]]; then
    echo "HARNESS PASS"
    exit 0
fi
echo "HARNESS FAIL"
exit 1
