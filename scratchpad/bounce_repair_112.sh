#!/bin/bash
# Repair-lane deploy bounce (user-directed): fires when the orchestrator checkpoint for
# step >= 112 exists (batch 112 binned + orch state saved), kills the orch BY PID (never by
# pattern), relaunches on the 4-lane config. Pools + in-flight spool survive: env names for
# the 3 existing lanes are unchanged (content-hash load matches), repair joins fresh.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
CKDIR=$OUT/run_default/checkpoints
ORCH_PID=3741299
# orch ckpts are written at the TOP of each step, so "batch 112 finished" == step_113 ckpt
# present AND the step_112 bin on disk.
TARGET=113
echo "armed $(date +%H:%M:%S); orch pid $ORCH_PID; firing when step_$TARGET ckpt + step_112 bin exist"
while true; do
  CUR=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1); CUR=${CUR:-0}
  [ "$CUR" -ge "$TARGET" ] && [ -f "$CKDIR/step_$CUR/orchestrator/progress.pt" ] \
    && [ -f "$OUT/run_default/rollouts/step_112/rollouts.bin" ] && break
  kill -0 $ORCH_PID 2>/dev/null || { echo "ABORT: orch died while waiting (ckpt at $CUR)"; exit 1; }
  sleep 20
done
sleep 5
echo "boundary ckpt step_$CUR at $(date +%H:%M:%S); deploying REPAIR lane (4-lane, ratios 0.2/0.5/0.2/0.1)"
kill $ORCH_PID 2>/dev/null
for i in $(seq 1 20); do kill -0 $ORCH_PID 2>/dev/null || break; sleep 1; done
kill -0 $ORCH_PID 2>/dev/null && { kill -9 $ORCH_PID; sleep 2; }
echo "orch stopped"
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=$CUR
{ echo "==== REPAIR LANE BOUNCE @ $(date +%H:%M:%S): 4 lanes [0.2 math, 0.5 code, 0.2 reason, 0.1 repair]; resume $CUR ===="; } >> $LOG/launch.log
nohup .venv/bin/surogate grpo-orch director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml >> $LOG/orch.log 2>&1 &
echo "new orch pid $!" | tee -a $LOG/launch.log
sleep 90
echo "==== SNAPSHOT ===="
tail -c 100000 $LOG/orch.log | grep -aE "Loading 4 training|Loading [0-9]+ training|Setting up buffer|Resuming|Restored|Loaded|repair" | tail -10
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3}'
echo "REPAIR BOUNCE COMPLETE"
