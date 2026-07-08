#!/bin/bash
# Step-80 eval watcher: first row where the corrected LR has had ~15 steps — the intervention
# verdict window opens here. Same single-pass protocol as the whole series. Alerts on trainer death.
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a
CKP=output/fugu_ultra_paper/step_00000140
TRAIN_PID=2906828
while [ ! -d "$CKP" ]; do
  kill -0 "$TRAIN_PID" 2>/dev/null || { echo "ALERT: trainer pid $TRAIN_PID DIED before step 140 ($(date +%H:%M:%S))"; exit 1; }
  sleep 180
done
echo "step-80 checkpoint detected at $(date +%H:%M:%S); settling 90s then firing single-pass eval"
sleep 90
EVAL_TREND_LOG=output/fugu_ultra_paper/heldout_trend.log \
EVAL_MANIFEST=heldout_trend60_taskspecs.jsonl \
ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra \
.venv/bin/python scratch_eval_live_throttled.py --label step140_fullmix --conc 3 --n 60
echo "==== trend tail ===="
tail -4 output/fugu_ultra_paper/heldout_trend.log
echo "==== trainer 75-80 ===="
grep -E "step=(7[5-9]|80) loss" output/fugu_ultra_paper/process_logs/train.log | tail -6
