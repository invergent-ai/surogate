#!/bin/bash
# Plain orch bounce at the next batch boundary to load the effort-only GPT amendment
# (providers.py already amended on disk; env servers are orch children and reload it).
# No config/prompt changes. Spool restores in-flight groups.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
CKDIR=$OUT/run_default/checkpoints
BASE=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1); BASE=${BASE:-0}
echo "armed $(date +%H:%M:%S); latest orch ckpt step_$BASE"
while true; do
  CUR=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1); CUR=${CUR:-0}
  [ "$CUR" -gt "$BASE" ] && [ -f "$CKDIR/step_$CUR/orchestrator/progress.pt" ] && break
  pgrep -f "[g]rpo-orch" >/dev/null || { echo "ABORT: orch died while waiting"; exit 1; }
  sleep 20
done
sleep 5
echo "boundary ckpt step_$CUR at $(date +%H:%M:%S); bouncing orch for effort-only GPT params"
pkill -f "[g]rpo-orch"
for i in $(seq 1 20); do pgrep -f "[g]rpo-orch" >/dev/null || break; sleep 1; done
pgrep -f "[g]rpo-orch" >/dev/null && { pkill -9 -f "[g]rpo-orch"; sleep 2; }
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=$CUR
{ echo "==== EFFORT-ONLY GPT BOUNCE @ $(date +%H:%M:%S): resume $CUR ===="; } >> $LOG/launch.log
.venv/bin/surogate grpo-orch director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml >> $LOG/orch.log 2>&1 &
echo "new orch pid $!" | tee -a $LOG/launch.log
sleep 75
tail -c 50000 $LOG/orch.log | grep -aE "Resuming training|Restored|spool" | tail -4
echo "EFFORT-ONLY BOUNCE COMPLETE"
