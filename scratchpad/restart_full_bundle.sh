#!/bin/bash
# FULL-BUNDLE RESTART after the emergency stop (user GO): GPT param fix (providers.py, loads
# fresh in new env servers) + Countdown few-shot swap + spool accepted-group fix. Resume point:
# trainer applied through 85, broadcast step_86 STABLE -> manufacture trainer ckpt 86; orch
# resumes via its own ckpt step_86 (aligned, no pruning needed); spool restores batch-86 groups.
set -u
cd /home/densemax/work/flavius/surogate
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
D=director/manifests/fugu_clean_v1/grpo_pilot_train

# 1) few-shot swap (validated dry-run; auto-backup .pre_ood_swap)
.venv/bin/python /tmp/claude-1000/-home-densemax-work-flavius-surogate/1636be7a-c882-47c0-8ed5-6ece7392008f/scratchpad/apply_fewshot_swap.py || { echo "SWAP FAILED - aborting restart"; exit 1; }

# 2) manufacture trainer checkpoint 86 from the post-85 broadcast (proven pattern)
CKP=$OUT/step_00000086; BC=$OUT/run_default/broadcasts/step_86
[ -f "$BC/STABLE" ] || { echo "ABORT: broadcast 86 not STABLE"; exit 1; }
mkdir -p "$CKP"
cp "$BC/adapter_config.json" "$BC/adapter_model.safetensors" "$CKP/"
REF=$(stat -c%s $OUT/step_00000080/adapter_model.safetensors)
S=$(stat -c%s "$CKP/adapter_model.safetensors")
[ "$S" -ge $((REF * 8 / 10)) ] || { echo "ABORT: adapter $S vs ref $REF"; exit 1; }
python3 - <<'PY'
import json
t = json.load(open("output/fugu_ultra_paper/step_00000080/checkpoint.json"))
t["run"]["step"] = 86
json.dump(t, open("output/fugu_ultra_paper/step_00000086/checkpoint.json", "w"), indent=2)
PY
echo "manufactured ckpt 86 (adapter $S bytes)"

# 3) prune any stale rollout bins >= 86 (none expected) ; orch ckpts > 86 (none expected)
for RD in $OUT/run_default/rollouts/step_*; do
  [ -d "$RD" ] || continue; N=${RD##*step_}
  [ "$N" -ge 86 ] && rm -rf "$RD" && echo "pruned stale bin $RD"
done
for OC in $OUT/run_default/checkpoints/step_*; do
  [ -d "$OC" ] || continue; N=${OC##*step_}
  [ "$N" -gt 86 ] && rm -rf "$OC" && echo "pruned orch ckpt $OC"
done

# 4) relaunch (vLLM already up on :8007)
set -a; source .env 2>/dev/null; set +a
export ULTRA_ALLOW_YUNWU=1 SUROGATE_GRPO_START_STEP=86
{ echo "==== FULL-BUNDLE RESTART @ $(date +%H:%M:%S): GPT param fix + OOD few-shots + spool fix; resume 86 ===="; } >> $LOG/launch.log
.venv/bin/surogate grpo-orch "$D/orch_paper.yaml" >> $LOG/orch.log 2>&1 &
echo "orch pid $!" | tee -a $LOG/launch.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_paper.yaml" >> $LOG/train.log 2>&1 &
echo "train pid $!" | tee -a $LOG/launch.log
sleep 90
echo "==== SNAPSHOT ===="
tail -c 80000 $LOG/orch.log | grep -aE "Resuming training|Restored|spool|Could not move|Loading 2|Setting up buffer" | tail -8
grep -aE "checkpoint at step|Learning rate" $LOG/train.log | tail -2
pgrep -af "grpo-(orch|train|infer)" | awk '{print $1, $3}'
echo "RESTART COMPLETE"
