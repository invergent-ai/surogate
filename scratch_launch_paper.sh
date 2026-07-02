#!/bin/bash
# FRESH launch of the paper-track GRPO run (raw Qwen3-8B, paper prompt, no-think,
# batch 256 = 4x64, std advantage). Refuses to start over an existing run dir.
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1
D=director/manifests/fugu_clean_v1/grpo_pilot_train
OUT=output/fugu_ultra_paper; LOG=$OUT/process_logs
if ls $OUT/step_* >/dev/null 2>&1; then
  echo "REFUSING: $OUT already has checkpoints -- use a resume script, not the fresh launcher."
  exit 1
fi
mkdir -p "$LOG"
{ echo "==== FRESH LAUNCH @ $(date +%H:%M:%S) ===="; } >> "$LOG/launch.log"
.venv/bin/surogate grpo-infer "$D/infer_paper.yaml" >> "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" >> "$LOG/launch.log"
for i in $(seq 1 120); do curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 && { echo "VLLM_READY $((i*3))s">>"$LOG/launch.log"; break; }; sleep 3; done
curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 || { echo VLLM_FAILED>>"$LOG/launch.log"; exit 1; }
.venv/bin/surogate grpo-orch "$D/orch_paper.yaml" >> "$LOG/orch.log" 2>&1 &
echo "orch pid $!" >> "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_paper.yaml" >> "$LOG/train.log" 2>&1 &
echo "train pid $!" >> "$LOG/launch.log"
echo "LAUNCHED" >> "$LOG/launch.log"; wait
