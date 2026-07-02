#!/bin/bash
cd /home/densemax/work/flavius/surogate
D=director/manifests/fugu_clean_v1/grpo_pilot_train
LOG=output/fugu_ultra_swe_replay/process_logs
mkdir -p "$LOG"
: > "$LOG/launch.log"
# clean trainer state (avoid resume contamination)
rm -rf output/fugu_ultra_swe_replay/run_default output/fugu_ultra_swe_replay/step_* \
       output/fugu_ultra_swe_replay/checkpoints output/fugu_ultra_swe_replay/rollouts \
       output/fugu_ultra_swe_replay/broadcasts 2>/dev/null
# 1) vLLM
.venv/bin/surogate grpo-infer "$D/infer_swe_replay.yaml" > "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" >> "$LOG/launch.log"
# 2) wait for ready (up to ~5 min)
for i in $(seq 1 100); do
  if curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1; then echo "VLLM_READY after $((i*3))s" >> "$LOG/launch.log"; break; fi
  sleep 3
done
if ! curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1; then echo "VLLM_FAILED" >> "$LOG/launch.log"; exit 1; fi
# 3) orch + train
.venv/bin/surogate grpo-orch "$D/orch_swe_replay.yaml" > "$LOG/orch.log" 2>&1 &
echo "orch pid $!" >> "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_swe_replay.yaml" > "$LOG/train.log" 2>&1 &
echo "train pid $!" >> "$LOG/launch.log"
echo "ORCH_TRAIN_LAUNCHED" >> "$LOG/launch.log"
wait
