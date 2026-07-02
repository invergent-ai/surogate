#!/bin/bash
# RESUME the LCB GRPO run from the latest checkpoint (for the pause->eval->resume-every-10 loop).
# Unlike scratch_launch_lcb.sh, this does NOT rm checkpoints/run_default, and it sets the orch
# start-step override so orch + trainer resume from step N (not restart at 0). Logs are APPENDED
# to preserve the reward/grad trajectory across chunks.
# Requires: train_lcb.yaml has `resume_from_checkpoint: true` (added at pause time).
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1
D=director/manifests/fugu_clean_v1/grpo_pilot_train
OUT=output/fugu_ultra_lcb; LOG=$OUT/process_logs; mkdir -p "$LOG"
# KEEP all run_default state (broadcasts + rollouts) -- orch + trainer resume from it.
# Do NOT rm broadcasts: the orch loads step-N weights from run_default/broadcasts/step_N at startup.
STEP=$(ls -d $OUT/step_* 2>/dev/null | sed -E 's/.*step_0*([0-9]+)$/\1/' | sort -n | tail -1)
[ -z "$STEP" ] && { echo "NO_CHECKPOINT_TO_RESUME" | tee -a "$LOG/launch.log"; exit 1; }
export SUROGATE_GRPO_START_STEP=$STEP
# Remove stale rollout batches at/after the resume step. Otherwise the trainer instantly
# re-consumes the old step_$STEP batch (double-applying an update) and then ignores the
# orch's freshly regenerated step_$STEP batch entirely -- one full generation step wasted.
for RD in $OUT/run_default/rollouts/step_*; do
  [ -d "$RD" ] || continue
  N=${RD##*step_}
  [ "$N" -ge "$STEP" ] && rm -rf "$RD" && echo "pruned stale rollout batch $RD (>= resume step $STEP)" >> "$LOG/launch.log"
done
# Self-heal: the orch's get_weight_dir needs run_default/broadcasts/step_$STEP/STABLE at startup, but
# the broadcast cleanup prunes old steps -- reconstruct it from the trainer checkpoint if missing.
BC=$OUT/run_default/broadcasts/step_$STEP; CKP=$OUT/step_$(printf '%08d' "$STEP")
if [ ! -f "$BC/STABLE" ] && [ -d "$CKP" ]; then
  mkdir -p "$BC"; cp "$CKP/adapter_config.json" "$CKP/adapter_model.safetensors" "$BC/" && touch "$BC/STABLE"
  echo "reconstructed broadcast $BC from $CKP" >> "$LOG/launch.log"
fi
{ echo "==== RESUME @ $(date +%H:%M:%S) from step $STEP ===="; } >> "$LOG/launch.log"
.venv/bin/surogate grpo-infer "$D/infer_singleturn.yaml" >> "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" >> "$LOG/launch.log"
for i in $(seq 1 120); do curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 && { echo "VLLM_READY $((i*3))s">>"$LOG/launch.log"; break; }; sleep 3; done
curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 || { echo VLLM_FAILED>>"$LOG/launch.log"; exit 1; }
.venv/bin/surogate grpo-orch "$D/orch_lcb.yaml" >> "$LOG/orch.log" 2>&1 &
echo "orch pid $!" >> "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_lcb.yaml" >> "$LOG/train.log" 2>&1 &
echo "train pid $!" >> "$LOG/launch.log"
echo "LAUNCHED (resume from $STEP)" >> "$LOG/launch.log"; wait
