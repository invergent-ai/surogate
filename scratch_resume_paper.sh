#!/bin/bash
# RESUME the paper-track GRPO run from the latest checkpoint (hardened pattern from the
# lcb-track resume script). Handles: broadcast self-heal (orch loads step-N weights from
# run_default/broadcasts/step_N, which cleanup may have pruned) and the resume off-by-one
# (stale rollouts/step_>=N would be re-consumed by the trainer, double-applying an update
# and orphaning the orch's regenerated batch -- prune them). Logs are APPENDED.
# Requires train_paper.yaml: resume_from_checkpoint: true.
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1
D=director/manifests/fugu_clean_v1/grpo_pilot_train
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs; mkdir -p "$LOG"
STEP=$(ls -d $OUT/step_* 2>/dev/null | sed -E 's/.*step_0*([0-9]+)$/\1/' | sort -n | tail -1)
[ -z "$STEP" ] && { echo "NO_CHECKPOINT_TO_RESUME" | tee -a "$LOG/launch.log"; exit 1; }
export SUROGATE_GRPO_START_STEP=$STEP
# prune stale rollout batches at/after the resume step (off-by-one guard)
for RD in $OUT/run_default/rollouts/step_*; do
  [ -d "$RD" ] || continue
  N=${RD##*step_}
  [ "$N" -ge "$STEP" ] && rm -rf "$RD" && echo "pruned stale rollout batch $RD (>= resume step $STEP)" >> "$LOG/launch.log"
done
# prune orchestrator checkpoints AHEAD of the trainer's step (alignment contract with
# orch_paper.yaml ckpt.resume_step=-1: latest surviving orch ckpt == trainer step, so the
# orch resumes with the eviction pools + surplus rollout cache from exactly that step; if
# none survive, the SUROGATE_GRPO_START_STEP fallback below behaves as before)
for OC in $OUT/run_default/checkpoints/step_*; do
  [ -d "$OC" ] || continue
  N=${OC##*step_}
  [ "$N" -gt "$STEP" ] && rm -rf "$OC" && echo "pruned orch checkpoint $OC (> resume step $STEP)" >> "$LOG/launch.log"
done
# broadcast self-heal: reconstruct run_default/broadcasts/step_$STEP from the checkpoint if missing
BC=$OUT/run_default/broadcasts/step_$STEP; CKP=$OUT/step_$(printf '%08d' "$STEP")
if [ ! -f "$BC/STABLE" ] && [ -d "$CKP" ]; then
  mkdir -p "$BC"; cp "$CKP/adapter_config.json" "$CKP/adapter_model.safetensors" "$BC/" && touch "$BC/STABLE"
  echo "reconstructed broadcast $BC from $CKP" >> "$LOG/launch.log"
fi
{ echo "==== RESUME @ $(date +%H:%M:%S) from step $STEP ===="; } >> "$LOG/launch.log"
.venv/bin/surogate grpo-infer "$D/infer_paper.yaml" >> "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" >> "$LOG/launch.log"
for i in $(seq 1 120); do curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 && { echo "VLLM_READY $((i*3))s">>"$LOG/launch.log"; break; }; sleep 3; done
curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 || { echo VLLM_FAILED>>"$LOG/launch.log"; exit 1; }
.venv/bin/surogate grpo-orch "$D/orch_paper.yaml" >> "$LOG/orch.log" 2>&1 &
echo "orch pid $!" >> "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_paper.yaml" >> "$LOG/train.log" 2>&1 &
echo "train pid $!" >> "$LOG/launch.log"
echo "RESUMED (from $STEP)" >> "$LOG/launch.log"; wait
