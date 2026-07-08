#!/bin/bash
# RLPR-lane deploy bounce (user GO): fires at the next orch checkpoint boundary, kills the
# orch BY PID (never by pattern - the 04:5x lesson), relaunches with the 3-lane config.
# Math/code eviction pools SURVIVE this bounce (their task names + prompts are unchanged,
# so the buffer's content-hash load matches) - first restart with full pool carryover.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
CKDIR=$OUT/run_default/checkpoints
ORCH_PID=3009725
BASE=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1); BASE=${BASE:-0}
echo "armed $(date +%H:%M:%S); orch pid $ORCH_PID, latest orch ckpt step_$BASE"
while true; do
  CUR=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1); CUR=${CUR:-0}
  [ "$CUR" -gt "$BASE" ] && [ -f "$CKDIR/step_$CUR/orchestrator/progress.pt" ] && break
  kill -0 $ORCH_PID 2>/dev/null || { echo "ABORT: orch died while waiting"; exit 1; }
  sleep 20
done
sleep 5
echo "boundary ckpt step_$CUR at $(date +%H:%M:%S); deploying RLPR lane"
kill $ORCH_PID 2>/dev/null
for i in $(seq 1 20); do kill -0 $ORCH_PID 2>/dev/null || break; sleep 1; done
kill -0 $ORCH_PID 2>/dev/null && { kill -9 $ORCH_PID; sleep 2; }
echo "orch stopped"
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=$CUR
{ echo "==== RLPR LANE BOUNCE @ $(date +%H:%M:%S): 3 lanes [0.2 math, 0.6 code, 0.2 reason]; resume $CUR ===="; } >> $LOG/launch.log
nohup .venv/bin/surogate grpo-orch director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml >> $LOG/orch.log 2>&1 &
echo "new orch pid $!" | tee -a $LOG/launch.log
sleep 90
echo "==== SNAPSHOT ===="
tail -c 80000 $LOG/orch.log | grep -aE "Loading 3 training|Setting up buffer|Resuming training|Restored|Could not move|Loaded" | tail -8
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3}'
echo "RLPR BOUNCE COMPLETE"
