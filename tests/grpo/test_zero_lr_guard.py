"""The zero-LR optimizer guard: a step with no loss tokens must not move the policy.

The guard swaps in a zero-lr, zero-decay optimizer config so the step's (empty)
accumulation flushes through the normal optimizer path without a policy update.

It is keyed on the step's own loss-token total, the number the gradients are
normalized by. It used to key on the engine's ValidTokenCount instead, which in
chunked GRPO is the chunk-0 count of the LAST micro-batch: 0 whenever that
sample's prompt fills chunk 0. On agentic data that is every step, so every
update was discarded (issue #264). The native step also divided the gradient by
that count a second time, on top of loss_scale.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "surogate/grpo/trainer.py"
DSL_EXECUTION = REPO_ROOT / "csrc/src/runtime/dsl/dsl_model_execution.cpp"

GUARD = "if loss_tokens == 0 and n_mb > 0:"


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for idx in range(brace, len(source)):
        char = source[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace : idx + 1]
    raise AssertionError(f"could not parse body for {signature}")


def test_guard_precedes_optimizer_update():
    src = TRAINER.read_text()
    guard = src.index(GUARD)
    update = src.index("result = self.trainer.update_with_config(opt_config, step + 1)")
    total = src.index('loss_tokens = int(sum(int(mb["loss_mask"].sum()) for mb in micro_batches))')
    assert total < guard < update, "guard must sit between the loss-token total and the optimizer update"


def test_guard_is_keyed_on_a_step_total_not_on_the_engine_token_count():
    """ValidTokenCount is re-zeroed by every chunk invocation, so at step end it
    holds one chunk of one micro-batch. Nothing in the trainer may decide on it."""
    src = TRAINER.read_text()
    assert "get_valid_token_count" not in src
    assert "vtc == 0" not in src


def test_guard_zeroes_both_lr_and_decay():
    src = TRAINER.read_text()
    block = src[src.index(GUARD):src.index("result = self.trainer.update_with_config")]
    assert "learning_rate=0.0" in block
    assert "weight_decay=0.0" in block, "decoupled decay must be zeroed too — lr=0 alone is not sufficient by contract"


def test_guard_flushes_through_normal_path_not_skip():
    """The guard must still call update_with_config (state flush), not skip it —
    skipping would leak the step's gradient accumulation into the next step."""
    src = TRAINER.read_text()
    block = src[src.index(GUARD):]
    # no `continue` between the guard and the update call
    upto_update = block[:block.index("update_with_config")]
    assert "\ncontinue" not in upto_update and " continue\n" not in upto_update


def test_native_grpo_step_does_not_scale_gradients_by_the_engine_token_count():
    """custom_dloss already carries 1/loss_scale. With mUseTokenScale left on, the
    optimizer divided the gradient again by whatever ValidTokenCount held at step
    end; the chunked and unchunked native paths both go through this function."""
    source = DSL_EXECUTION.read_text()
    window = _function_body(source, "void DslModel::step_grpo_native_window(")
    assert "mUseTokenScale = false;" in window
    assert window.index("mUseTokenScale = false;") < window.index("execute_forward(")
    native = _function_body(source, "void DslModel::step_grpo_native(")
    assert "step_grpo_native_window(" in native
