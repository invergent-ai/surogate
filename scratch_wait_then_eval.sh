#!/bin/bash
cd /home/densemax/work/flavius/surogate
while pgrep -f "grpo-orch.*singleturn" >/dev/null 2>&1; do sleep 60; done
echo "=== training finished $(date) ===" > /tmp/heldout_verdict.log
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|tr -d ' '); do kill -9 "$p" 2>/dev/null; done
sleep 10
CKPT=$(ls -td output/fugu_ultra_singleturn/checkpoints/step_* 2>/dev/null | head -1)
echo "=== eval TRAINED checkpoint: $CKPT (full held-out, 600s timeout) ===" >> /tmp/heldout_verdict.log
set -a; source .env 2>/dev/null; set +a; export ULTRA_ALLOW_YUNWU=1
PYTHONPATH=ultra CUDA_VISIBLE_DEVICES=0 .venv/bin/python scratch_eval_heldout.py --adapter "$CKPT" --n 25 >> /tmp/heldout_verdict.log 2>&1
echo "=== VERDICT_COMPLETE $(date) ===" >> /tmp/heldout_verdict.log
