#!/bin/bash
# STAGE-2 LAUNCH — 2-turn feedback GRPO, resuming the CONDUCTOR from paper-track step 145.
# Pre-registered GO conditions (MISSION 2026-07-08): rep2 replicate holds + sealed confirmation
# clean. DO NOT run before both — this script REPLACES the vLLM on :8007 (evals depend on it).
#
# Mechanics (the proven manufactured-checkpoint pattern):
#   1. seed output/fugu_ultra_stage2 with checkpoint step_00000145 built from the paper run's
#      atomic broadcast step_145;
#   2. launch vLLM (infer_stage2) -> orch (orch_stage2: max_turns=2 all lanes, token client)
#      -> trainer (train_stage2: same LoRA/recipe, resumes at 145).
set -euo pipefail
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1

D=director/manifests/fugu_clean_v1/grpo_pilot_train
SRC=output/fugu_ultra_paper
OUT=output/fugu_ultra_stage2
LOG=$OUT/process_logs
STEP=145
CKP=$OUT/step_$(printf '%08d' $STEP)
BC_SRC=$SRC/run_default/broadcasts/step_$STEP

if pgrep -f "grpo-orch|grpo-train" >/dev/null; then
  echo "orch/trainer already running — refusing" >&2; exit 1
fi
[ -f "$BC_SRC/STABLE" ] || { echo "source broadcast $BC_SRC not STABLE" >&2; exit 1; }

mkdir -p "$LOG" "$OUT/run_default/broadcasts/step_$STEP"
if [ ! -d "$CKP" ]; then
  mkdir -p "$CKP"
  cp "$BC_SRC/adapter_config.json" "$BC_SRC/adapter_model.safetensors" "$CKP/"
  # checkpoint.json in the trainer's own format (mirrors the paper-run manufacture pattern)
  REF=$(ls -d $SRC/step_* | tail -1)/checkpoint.json
  .venv/bin/python - "$REF" "$CKP/checkpoint.json" $STEP <<'PY'
import json, sys
ref, out, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(ref))
d["run"]["step"] = step  # nested (see any step_*/checkpoint.json); loss/norm history resets are benign
json.dump(d, open(out, "w"))
PY
  echo "manufactured $CKP from $BC_SRC"
fi
cp "$BC_SRC/adapter_config.json" "$BC_SRC/adapter_model.safetensors" "$OUT/run_default/broadcasts/step_$STEP/" 2>/dev/null || true
touch "$OUT/run_default/broadcasts/step_$STEP/STABLE"

# stop the old eval vLLM (port 8007) by PID
OLDV=$(pgrep -f "grpo-infer" | head -1 || true)
if [ -n "$OLDV" ]; then kill "$OLDV"; sleep 5; echo "stopped old vLLM pid $OLDV"; fi

export SUROGATE_GRPO_START_STEP=$STEP
{ echo "==== STAGE-2 LAUNCH @ $(date +%H:%M:%S) from step $STEP ===="; } >> "$LOG/launch.log"
.venv/bin/surogate grpo-infer "$D/infer_stage2.yaml" >> "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" | tee -a "$LOG/launch.log"
for i in $(seq 1 120); do curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 && { echo "VLLM_READY $((i*3))s" | tee -a "$LOG/launch.log"; break; }; sleep 3; done
curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 || { echo VLLM_FAILED | tee -a "$LOG/launch.log"; exit 1; }
.venv/bin/surogate grpo-orch "$D/orch_stage2.yaml" >> "$LOG/orch.log" 2>&1 &
echo "orch pid $!" | tee -a "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_stage2.yaml" >> "$LOG/train.log" 2>&1 &
echo "train pid $!" | tee -a "$LOG/launch.log"
echo "STAGE-2 LAUNCHED (from $STEP)" | tee -a "$LOG/launch.log"
wait