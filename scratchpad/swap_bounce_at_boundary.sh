#!/bin/bash
# OOD few-shot swap at the next batch boundary (user GO; waiting for the step-90 eval row
# dropped as low-value — decisions pre-committed). Trigger: the orch saves its next interval
# checkpoint (= previous batch banked, in-flight is seconds old). Orch-only bounce; trainer
# and vLLM untouched. Deploys: Countdown few-shots + the staged spool accepted-group fix.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
CKDIR=$OUT/run_default/checkpoints
BASE=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1)
BASE=${BASE:-0}
echo "armed at $(date +%H:%M:%S); latest orch ckpt step_$BASE, waiting for the next one"
DEADLINE=$(( $(date +%s) + 2100 ))   # 35 min: take the free boundary if it comes, else accept mid-batch cost
MODE=boundary
while true; do
  CUR=$(ls -d $CKDIR/step_* 2>/dev/null | sed -E 's/.*step_([0-9]+)$/\1/' | sort -n | tail -1)
  CUR=${CUR:-0}
  [ "$CUR" -gt "$BASE" ] && [ -f "$CKDIR/step_$CUR/orchestrator/progress.pt" ] && break
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then MODE=deadline; break; fi
  pgrep -f "[g]rpo-orch" >/dev/null || { echo "ABORT: orch died while waiting"; exit 1; }
  sleep 20
done
sleep 5
echo "trigger=$MODE (ckpt step_$CUR) at $(date +%H:%M:%S); applying few-shot swap"
[ "$MODE" = deadline ] && echo "mid-batch bounce: banked groups of the in-flight batch are forfeit (known spool hole, fix deploys now); live partial groups restore from spool"
.venv/bin/python /tmp/claude-1000/-home-densemax-work-flavius-surogate/1636be7a-c882-47c0-8ed5-6ece7392008f/scratchpad/apply_fewshot_swap.py || { echo "SWAP FAILED - orch left running on old prompt"; exit 1; }
pkill -f "[g]rpo-orch"
for i in $(seq 1 20); do pgrep -f "[g]rpo-orch" >/dev/null || break; sleep 1; done
pgrep -f "[g]rpo-orch" >/dev/null && { pkill -9 -f "[g]rpo-orch"; sleep 2; }
echo "orch stopped"
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=$CUR
{ echo "==== OOD FEW-SHOT SWAP BOUNCE @ $(date +%H:%M:%S): Countdown examples live; resume $CUR ===="; } >> $LOG/launch.log
.venv/bin/surogate grpo-orch director/manifests/fugu_clean_v1/grpo_pilot_train/orch_paper.yaml >> $LOG/orch.log 2>&1 &
echo "new orch pid $!" | tee -a $LOG/launch.log
sleep 90
echo "==== SNAPSHOT ===="
tail -c 60000 $LOG/orch.log | grep -aE "Resuming training|Restored|Discarded|spool|Could not move|Loaded" | tail -8
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3}'
echo "SWAP BOUNCE COMPLETE"
