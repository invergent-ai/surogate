#!/bin/bash
# Step-70 eval watcher (re-armed 2026-07-05 after the intervention restart consumed the old one).
# Waits for the trainer's step-70 checkpoint, settles, fires the standard single-pass n=60 live
# eval against the training vLLM's 'default' adapter (no-stop protocol). Alerts on trainer death.
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a
CKP=output/fugu_ultra_paper/step_00000070
TRAIN_PID=1852589
while [ ! -d "$CKP" ]; do
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "ALERT: trainer pid $TRAIN_PID DIED before step 70 ($(date +%H:%M:%S))"; exit 1; }
  sleep 120
done
echo "step-70 checkpoint detected at $(date +%H:%M:%S); settling 90s then firing single-pass eval"
sleep 90
EVAL_TREND_LOG=output/fugu_ultra_paper/heldout_trend.log \
EVAL_MANIFEST=heldout_trend60_taskspecs.jsonl \
ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra \
.venv/bin/python scratch_eval_live_throttled.py --label step70_fullmix --conc 3 --n 60
echo "==== trend tail ===="
tail -8 output/fugu_ultra_paper/heldout_trend.log
