#!/bin/bash
# KL-drift watcher: the revert rule is about SUSTAINED drift, so take a second reading two
# steps into the 1e-5 regime. Fires when step=66 lands; prints the last trainer lines and the
# orch eviction/parse metrics for steps 65-66. Pure grep/sed — no inline python.
cd /home/densemax/work/flavius/surogate
TRAIN_LOG=output/fugu_ultra_paper/process_logs/train.log
TRAIN_PID=1852589
while ! grep -q "step=66 loss" "$TRAIN_LOG"; do
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "ALERT: trainer pid $TRAIN_PID DIED before step 66 ($(date +%H:%M:%S))"; exit 1; }
  sleep 120
done
echo "=== KL-DRIFT CHECK @ $(date +%H:%M:%S): steps 64-66 at lr 1e-5 ==="
grep -E "step=6[456] loss" "$TRAIN_LOG" | tail -3
echo "=== orch metrics (steps 65-66): eviction + parse ==="
for S in 65 66; do
  L=$(grep "\"step\": $S," /tmp/grpo_metrics.jsonl | tail -1)
  [ -z "$L" ] && continue
  echo -n "step $S: "
  echo "$L" | grep -oE '"(reward/mean|metrics/ultra_workflow_parse_valid|evicted_examples/hard|evicted_examples/easy|filtered_rollouts/easy)": [0-9.]+' | tr '\n' ' '
  echo
done
