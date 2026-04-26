# Regression Baselines

This directory is the tracked home for first-month refactor regression artifacts.

Use:

```bash
python -m surogate.regression.baseline_runner --report
python -m surogate.regression.baseline_runner --run --steps 50 --out regression_baselines/current --report
python -m surogate.regression.baseline_runner --compare
```

`locked/` is for blessed baselines. `current/` is ignored by convention in review
unless a baseline update is intentional.
