#!/usr/bin/env bash
# Launch the FULL 60-task full-strength verdict eval + the MISSION.md progress watcher.
# Resumable: smoke rows in scratchpad/fs_verdict_rows.jsonl are reused (same tasks/arms).
set -euo pipefail
cd /home/densemax/work/flavius/surogate

if pgrep -f "eval_fullstrength_verdict.py" >/dev/null; then
  echo "an eval is already running; refusing to double-launch" >&2
  exit 1
fi

mkdir -p output/fugu_ultra_paper
EVAL_TREND_LOG=output/fugu_ultra_paper/heldout_trend.log \
ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra \
nohup .venv/bin/python scratchpad/eval_fullstrength_verdict.py \
  --label fs_verdict60 --conc 8 --n 60 --budget-usd 120 \
  > scratchpad/fs_verdict_full.log 2>&1 &
echo "eval pid: $!"

nohup .venv/bin/python scratchpad/fs_progress_watch.py --interval 120 \
  > scratchpad/fs_progress_watch.log 2>&1 &
echo "watcher pid: $!"
