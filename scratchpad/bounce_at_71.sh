#!/bin/bash
# BOUNCE at the step-71 boundary (user GO 2026-07-05): deploys mix 60/40 + easy_threshold 1.0
# + orch persistence + in-flight spool + progress line. Proven step-64 pattern: manufacture the
# trainer checkpoint from the post-71 broadcast, restart orch+trainer, vLLM stays up.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
TRAIN_PID=1852589
D=director/manifests/fugu_clean_v1/grpo_pilot_train

# 1) wait for the boundary: step 71 applied AND its successor broadcast STABLE
while true; do
  kill -0 $TRAIN_PID 2>/dev/null || { echo "ABORT: trainer died before the boundary ($(date +%H:%M:%S))"; exit 1; }
  if grep -q "step=71 loss" $LOG/train.log && [ -f $OUT/run_default/broadcasts/step_72/STABLE ]; then
    break
  fi
  sleep 60
done
echo "BOUNDARY at $(date +%H:%M:%S): step 71 applied, broadcast 72 STABLE"
grep -oE "step=71 loss=[0-9.]+ grad_norm=[0-9.e-]+ lr=[0-9.e-]+ kl=[0-9.]+" $LOG/train.log | tail -1
sleep 10

# 2) manufacture trainer checkpoint 72 from the broadcast (integrity-checked)
CKP=$OUT/step_00000072; BC=$OUT/run_default/broadcasts/step_72
mkdir -p "$CKP"
cp "$BC/adapter_config.json" "$BC/adapter_model.safetensors" "$CKP/"
REF=$(stat -c%s $OUT/step_00000070/adapter_model.safetensors)
S=$(stat -c%s "$CKP/adapter_model.safetensors")
[ "$S" -ge $((REF * 8 / 10)) ] || { echo "ABORT: adapter $S bytes vs reference $REF"; exit 1; }
python3 - <<'PY'
import json
t = json.load(open("output/fugu_ultra_paper/step_00000070/checkpoint.json"))
t["run"]["step"] = 72
json.dump(t, open("output/fugu_ultra_paper/step_00000072/checkpoint.json", "w"), indent=2)
print("checkpoint.json written with run.step=72")
PY
echo "manufactured $CKP (adapter $S bytes, ref $REF)"

# 3) stop orch + trainer (vLLM stays on :8007)
pkill -f "[g]rpo-orch"; pkill -f "[g]rpo-train"
for i in $(seq 1 20); do pgrep -f "[g]rpo-(orch|train)" >/dev/null || break; sleep 1; done
pgrep -f "[g]rpo-(orch|train)" >/dev/null && { pkill -9 -f "[g]rpo-orch"; pkill -9 -f "[g]rpo-train"; sleep 3; }
echo "orch+trainer stopped"

# 4) prune stale rollout batches at/after the resume step
for RD in $OUT/run_default/rollouts/step_*; do
  [ -d "$RD" ] || continue
  N=${RD##*step_}
  [ "$N" -ge 72 ] && rm -rf "$RD" && echo "pruned stale rollout batch $RD"
done

# 5) relaunch with the staged config
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=72
{ echo "==== BOUNCE @ $(date +%H:%M:%S): mix 60/40 + easy_threshold + persistence + spool, resume 72 ===="; } >> $LOG/launch.log
.venv/bin/surogate grpo-orch "$D/orch_paper.yaml" >> $LOG/orch.log 2>&1 &
NEW_ORCH=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_paper.yaml" >> $LOG/train.log 2>&1 &
NEW_TRAIN=$!
echo "new orch pid $NEW_ORCH, new train pid $NEW_TRAIN" | tee -a $LOG/launch.log
echo "$NEW_ORCH $NEW_TRAIN" > $OUT/process_logs/bounce_pids.txt

# 6) acceptance snapshot after warm-up
sleep 60
echo "==== ACCEPTANCE SNAPSHOT ===="
grep -aE "training environment|Setting up buffer|spool|override step|Restored" $LOG/orch.log | tail -8
grep -aE "checkpoint at step|Learning rate" $LOG/train.log | tail -3
echo "==== procs ===="
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3, $4}'
echo "BOUNCE COMPLETE"
