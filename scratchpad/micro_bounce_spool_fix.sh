#!/bin/bash
# Orch-only micro-bounce: deploys the spool compaction fix (mkdir before tmp write) with
# seconds of loss. Fires when the orch writes its first interval checkpoint (step_73), i.e.
# right after bin-72 lands. Trainer + vLLM untouched. The new orch resumes via
# ckpt.resume_step=-1 -- validating the persistence path end-to-end -- with
# SUROGATE_GRPO_START_STEP=73 exported only as a belt-and-braces fallback.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
ORCH_PID=2369179
CKPT=$OUT/run_default/checkpoints/step_73/orchestrator
while [ ! -f "$CKPT/progress.pt" ]; do
  kill -0 $ORCH_PID 2>/dev/null || { echo "ABORT: orch died before the step-73 checkpoint ($(date +%H:%M:%S))"; exit 1; }
  sleep 60
done
sleep 5
echo "orch checkpoint step_73 complete at $(date +%H:%M:%S); restarting orch only"
kill $ORCH_PID 2>/dev/null
for i in $(seq 1 20); do kill -0 $ORCH_PID 2>/dev/null || break; sleep 1; done
kill -0 $ORCH_PID 2>/dev/null && { kill -9 $ORCH_PID; sleep 2; }
echo "orch stopped"
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=73
{ echo "==== ORCH MICRO-BOUNCE @ $(date +%H:%M:%S): spool fix; resume via orch ckpt (-1) ===="; } >> $LOG/launch.log
.venv/bin/surogate grpo-orch director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml >> $LOG/orch.log 2>&1 &
echo "new orch pid $!" | tee -a $LOG/launch.log
sleep 90
echo "==== SNAPSHOT ===="
tail -c 60000 $LOG/orch.log | grep -aE "Resuming training from checkpoint|spool|Restored|Setting up buffer|Loaded|pool" | tail -8
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3}'
echo "MICRO-BOUNCE COMPLETE"
