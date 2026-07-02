#!/bin/bash
cd /home/densemax/work/flavius/surogate
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1
D=director/manifests/fugu_clean_v1/grpo_pilot_train
OUT=output/fugu_ultra_grpo_pilot_qwen3_8b_after_parent_repair_sft_tight_repo_tool
LOG=$OUT/process_logs; mkdir -p "$LOG"; : > "$LOG/launch.log"
rm -rf $OUT/run_default $OUT/step_* $OUT/checkpoints $OUT/rollouts $OUT/broadcasts 2>/dev/null
.venv/bin/surogate grpo-infer "$D/infer_pilot_qwen3_8b_after_parent_repair_sft.yaml" > "$LOG/infer.log" 2>&1 &
echo "vLLM pid $!" >> "$LOG/launch.log"
for i in $(seq 1 120); do curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 && { echo "VLLM_READY $((i*3))s" >> "$LOG/launch.log"; break; }; sleep 3; done
curl -s -m2 http://localhost:8007/v1/models >/dev/null 2>&1 || { echo "VLLM_FAILED" >> "$LOG/launch.log"; exit 1; }
.venv/bin/surogate grpo-orch "$D/orch_tight_repo_tool_after_parent_repair_sft_commercial.yaml" > "$LOG/orch.log" 2>&1 &
echo "orch pid $!" >> "$LOG/launch.log"
CUDA_VISIBLE_DEVICES=1 .venv/bin/surogate grpo-train "$D/train_tight_repo_tool_after_parent_repair_sft.yaml" > "$LOG/train.log" 2>&1 &
echo "train pid $!" >> "$LOG/launch.log"
echo "ORCH_TRAIN_LAUNCHED" >> "$LOG/launch.log"
wait
