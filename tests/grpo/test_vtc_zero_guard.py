"""The vtc==0 optimizer guard: a corrupt-accounting step must not move the policy.

Observed 2026-08-03 on a chunked-GRPO run: one step's loss op computed nothing
(stuck chunk-sweep state), leaving ValidTokenCount at 0. That buffer is also
the gradient normalization denominator, so gradients came out ~1e5x too large
(grad_norm 1.88 vs a 0.003-0.03 band) and the metrics denominators produced an
impossible masked=2198%. The guard swaps in a zero-lr, zero-decay optimizer
config so the poisoned accumulation flushes without a policy update.
"""

import ast
from pathlib import Path

TRAINER = Path(__file__).resolve().parents[2] / "surogate/grpo/trainer.py"


def test_guard_precedes_optimizer_update():
    src = TRAINER.read_text()
    guard = src.index("if vtc == 0 and n_mb > 0:")
    update = src.index("result = self.trainer.update_with_config(opt_config, step + 1)")
    read = src.index("vtc = self.trainer.get_valid_token_count(0)")
    assert read < guard < update, "guard must sit between the vtc read and the optimizer update"


def test_guard_zeroes_both_lr_and_decay():
    src = TRAINER.read_text()
    block = src[src.index("if vtc == 0 and n_mb > 0:"):src.index("result = self.trainer.update_with_config")]
    assert "learning_rate=0.0" in block
    assert "weight_decay=0.0" in block, "decoupled decay must be zeroed too — lr=0 alone is not sufficient by contract"


def test_guard_flushes_through_normal_path_not_skip():
    """The guard must still call update_with_config (state flush), not skip it —
    skipping would leak the poisoned gradient accumulation into the next step."""
    src = TRAINER.read_text()
    block = src[src.index("if vtc == 0 and n_mb > 0:"):]
    # no `continue` between the guard and the update call
    upto_update = block[:block.index("update_with_config")]
    assert "\ncontinue" not in upto_update and " continue\n" not in upto_update
