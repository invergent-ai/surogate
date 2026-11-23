"""Inkling-2-Mini RL glue.

The kit's RL loop (an external RL recipe, branch ow-template) talks to its trainer through files only:
  - rollouts:   <run_dir>/rollouts/step_<N>/rollouts.bin  (msgpack TrainingBatch, the kit's sg_types schema)
  - broadcasts: <run_dir>/broadcasts/step_<N>/{adapter_model.safetensors, adapter_config.json, STABLE}
  - metrics:    <surogate_metrics_path> (JSON lines) and train.log lines the kit's guards parse
  - checkpoints:<output_dir>/step_<NNNNNNNN>/ every save_steps
This package implements that contract for Inkling with a pluggable forward / backward backend:
  records.py  the TrainingBatch reader and the per-visit routing-replay rows (48 visits x top-8)
  kitsrc.py   the deployed GSPO loss (byte copy, sha-pinned) and its config class, loaded without imports
  step.py     one kit step: step-start score, engine_mismatch_kl, off-policy drop, token TIS, GSPO seeds
  adapter.py  expert-LoRA export / import in the engine's names, the engine's bank fold, bf16 RNE (numpy only)
  shim.py     the trainer process (wait for a batch, step, broadcast, log, checkpoint, resume)
New files only: nothing here is imported by any other part of surogate.
"""

VISITS = 48            # layer applications per token: prelude 0-5, core 6-17 x 3 passes, coda 18-23
PHYSICAL_LAYERS = 24
TOP_K = 8
EXPERTS = 128
HIDDEN = 2304
INTER = 1152
RANK = 16
ALPHA = 32
