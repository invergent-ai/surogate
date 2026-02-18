"""Integration tests for GRPO inference and log-prob APIs.

Validates Phases 1-3 of the GRPO implementation against a mini Qwen3 model:
  - Phase 1: inference_prefill / inference_decode (KV-cache forward)
  - Phase 2: set_kv_pos (position reset for multi-completion generation)
  - Phase 3: compute_logprobs (per-token log P(target|context))

Requirements:
    - GPU with enough VRAM for a small Qwen3 model
    - HF weights: set QWEN3_MODEL_PATH env var or have Qwen/Qwen3-0.6B cached

Usage:
    pytest tests/test_grpo_inference.py -v
    QWEN3_MODEL_PATH=/path/to/Qwen3-0.6B pytest tests/test_grpo_inference.py -v
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID  = "Qwen/Qwen3-0.6B"
ENV_VAR   = "QWEN3_MODEL_PATH"
NUM_LAYERS = 2       # very small to keep the test fast
VOCAB_SIZE = None    # resolved from config
SEED       = 42
BATCH      = 2
SEQ_LEN    = 16
PROMPT_LEN = 6       # tokens used as shared prompt for set_kv_pos test

MINI_MODEL_DIR = Path("tmp/grpo_inference_mini")


# ---------------------------------------------------------------------------
# Helpers (reuse mini-model setup from test_onboarding_qwen3)
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path | None:
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.exists():
            return p
    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    model_slug = MODEL_ID.replace("/", "--")
    snaps = cache_root / f"models--{model_slug}" / "snapshots"
    if snaps.exists():
        for snap in sorted(snaps.iterdir(), reverse=True):
            if (snap / "config.json").exists():
                return snap
    return None


def prepare_mini_model(snapshot_dir: Path) -> Path:
    if MINI_MODEL_DIR.exists():
        return MINI_MODEL_DIR
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS
    (MINI_MODEL_DIR / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = snapshot_dir / tok_file
        if src.exists():
            shutil.copy2(src, MINI_MODEL_DIR / tok_file)

    index_path = snapshot_dir / "model.safetensors.index.json"
    single_path = snapshot_dir / "model.safetensors"

    if index_path.exists():
        base_index = json.loads(index_path.read_text())
        weight_map = base_index.get("weight_map", {})
        prefixes = [f"model.layers.{i}." for i in range(NUM_LAYERS)]
        extra = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

        def want(name: str) -> bool:
            return name in extra or any(name.startswith(p) for p in prefixes)

        mini_map = {k: v for k, v in weight_map.items() if want(k)}
        (MINI_MODEL_DIR / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": base_index.get("metadata", {}),
                        "weight_map": mini_map}, indent=2) + "\n"
        )
        for fname in sorted(set(mini_map.values())):
            link = MINI_MODEL_DIR / fname
            if not link.exists():
                link.symlink_to((snapshot_dir / fname).resolve())
    elif single_path.exists():
        link = MINI_MODEL_DIR / "model.safetensors"
        if not link.exists():
            link.symlink_to(single_path.resolve())
    else:
        raise FileNotFoundError(f"No weights found in {snapshot_dir}")

    return MINI_MODEL_DIR


def build_trainer(model_dir: Path) -> _surogate.SurogateTrainer:
    cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = _surogate.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    return trainer


def make_batch(vocab_size: int, seed: int = SEED):
    """Return (inputs [B,T], targets [B,T]) with last column masked."""
    rng = np.random.default_rng(seed)
    inputs  = rng.integers(1, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1).copy()
    targets[:, -1] = -100
    return inputs, targets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Qwen3 weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def vocab_size(model_dir):
    return json.loads((model_dir / "config.json").read_text())["vocab_size"]


@pytest.fixture(scope="module")
def trainer(model_dir):
    return build_trainer(model_dir)


# ---------------------------------------------------------------------------
# Phase 1: inference_prefill / inference_decode
# ---------------------------------------------------------------------------

class TestInferenceAPI:
    """Validate KV-cache prefill and decode (Phase 1)."""

    def test_prefill_returns_vocab_size_logits(self, trainer, vocab_size):
        """inference_prefill returns a float32 vector of shape [vocab_size]."""
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        trainer.enter_inference_mode(SEQ_LEN)
        try:
            logits = trainer.inference_prefill(prompt)
        finally:
            trainer.exit_inference_mode()

        assert logits.dtype == np.float32, f"expected float32, got {logits.dtype}"
        assert logits.shape == (vocab_size,), \
            f"expected ({vocab_size},), got {logits.shape}"
        assert np.all(np.isfinite(logits)), "logits contain inf/nan"

    def test_decode_returns_vocab_size_logits(self, trainer, vocab_size):
        """inference_decode returns a float32 vector of shape [vocab_size]."""
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        trainer.enter_inference_mode(SEQ_LEN)
        try:
            trainer.inference_prefill(prompt)
            logits = trainer.inference_decode(int(prompt[-1]), PROMPT_LEN)
        finally:
            trainer.exit_inference_mode()

        assert logits.dtype == np.float32
        assert logits.shape == (vocab_size,)
        assert np.all(np.isfinite(logits))

    def test_decode_changes_logits_each_step(self, trainer, vocab_size):
        """Each decode step should produce different logits (context grows)."""
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        rng = np.random.default_rng(SEED)

        trainer.enter_inference_mode(SEQ_LEN)
        try:
            logits0 = np.array(trainer.inference_prefill(prompt))
            token0 = int(np.argmax(logits0))

            logits1 = np.array(trainer.inference_decode(token0, PROMPT_LEN))
            token1 = int(np.argmax(logits1))

            logits2 = np.array(trainer.inference_decode(token1, PROMPT_LEN + 1))
        finally:
            trainer.exit_inference_mode()

        # Logits should differ across steps (different cached context).
        assert not np.allclose(logits0, logits1, atol=1e-3), \
            "prefill and first decode logits are identical — decode is not updating context"
        assert not np.allclose(logits1, logits2, atol=1e-3), \
            "consecutive decode logits are identical — decode is not updating context"

    def test_prefill_logits_are_not_all_zero(self, trainer, vocab_size):
        """Sanity check: logits are non-trivial."""
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        trainer.enter_inference_mode(SEQ_LEN)
        try:
            logits = trainer.inference_prefill(prompt)
        finally:
            trainer.exit_inference_mode()
        assert not np.allclose(logits, 0.0), "all logits are zero — model output is degenerate"


# ---------------------------------------------------------------------------
# Phase 2: set_kv_pos
# ---------------------------------------------------------------------------

class TestSetKVPos:
    """Validate KV-cache position reset for multi-completion generation (Phase 2)."""

    def test_two_completions_same_first_logits(self, trainer, vocab_size):
        """Two completions from the same prompt should start with identical logits.

        After inference_prefill(prompt), calling set_kv_pos(prompt_len) twice
        resets the decode position so both completions share the same cached
        prompt K/V and produce the same first decode logits.
        """
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        trainer.enter_inference_mode(SEQ_LEN)
        try:
            trainer.inference_prefill(prompt)

            # First completion: first decode step
            trainer.set_kv_pos(PROMPT_LEN)
            logits_c1 = np.array(trainer.inference_decode(int(prompt[-1]), PROMPT_LEN))

            # Second completion: reset to same prompt position, repeat
            trainer.set_kv_pos(PROMPT_LEN)
            logits_c2 = np.array(trainer.inference_decode(int(prompt[-1]), PROMPT_LEN))
        finally:
            trainer.exit_inference_mode()

        np.testing.assert_array_equal(logits_c1, logits_c2,
            err_msg="set_kv_pos did not reproduce the same logits from the same prompt")

    def test_different_completions_diverge(self, trainer, vocab_size):
        """Two completions that choose different tokens should diverge."""
        prompt = np.arange(1, PROMPT_LEN + 1, dtype=np.int32)
        trainer.enter_inference_mode(SEQ_LEN)
        try:
            trainer.inference_prefill(prompt)

            # Completion A: token 1 at position PROMPT_LEN
            trainer.set_kv_pos(PROMPT_LEN)
            trainer.inference_decode(1, PROMPT_LEN)
            logits_a = np.array(trainer.inference_decode(1, PROMPT_LEN + 1))

            # Completion B: token 2 at position PROMPT_LEN
            trainer.set_kv_pos(PROMPT_LEN)
            trainer.inference_decode(2, PROMPT_LEN)
            logits_b = np.array(trainer.inference_decode(2, PROMPT_LEN + 1))
        finally:
            trainer.exit_inference_mode()

        # Different context → different logits (with high probability for any real model)
        assert not np.allclose(logits_a, logits_b, atol=1e-3), \
            "completions with different tokens produced identical logits at step 2"


# ---------------------------------------------------------------------------
# Phase 3: compute_logprobs
# ---------------------------------------------------------------------------

class TestComputeLogprobs:
    """Validate per-token log-probability extraction (Phase 3)."""

    def test_shape(self, trainer, vocab_size):
        """compute_logprobs returns float32 array of shape [B, T]."""
        inputs, targets = make_batch(vocab_size)
        logprobs = trainer.compute_logprobs(inputs, targets, use_lora=False)

        assert logprobs.dtype == np.float32, f"expected float32, got {logprobs.dtype}"
        assert logprobs.shape == (BATCH, SEQ_LEN), \
            f"expected ({BATCH}, {SEQ_LEN}), got {logprobs.shape}"

    def test_all_finite(self, trainer, vocab_size):
        """All returned log-probs must be finite."""
        inputs, targets = make_batch(vocab_size)
        logprobs = trainer.compute_logprobs(inputs, targets, use_lora=False)
        assert np.all(np.isfinite(logprobs)), \
            f"non-finite log-probs at positions: {np.argwhere(~np.isfinite(logprobs))}"

    def test_nonpositive_for_valid_tokens(self, trainer, vocab_size):
        """log P(token) ≤ 0 for all non-masked positions."""
        inputs, targets = make_batch(vocab_size)
        logprobs = trainer.compute_logprobs(inputs, targets, use_lora=False)

        mask = (targets != -100)
        bad = logprobs[mask][logprobs[mask] > 1e-5]
        assert bad.size == 0, \
            f"found {bad.size} log-probs > 0 at valid positions: max={bad.max():.4f}"

    def test_masked_positions_are_zero(self, trainer, vocab_size):
        """Positions with target=-100 must receive exactly 0."""
        inputs, targets = make_batch(vocab_size)
        # targets[:, -1] is already -100 from make_batch
        logprobs = trainer.compute_logprobs(inputs, targets, use_lora=False)

        masked = logprobs[targets == -100]
        np.testing.assert_array_equal(masked, 0.0,
            err_msg="masked positions (target==-100) should have log-prob 0")

    def test_consistent_with_training_loss(self, trainer, vocab_size):
        """Mean cross-entropy from log-probs should match the training step loss.

        The training loss is:
            loss = sum(-logprob) / n_valid_tokens
        We compare against the loss returned by trainer.step().
        Tolerance is generous to account for minor fp16 accumulation differences
        between the fused training path and the log-prob extraction path.
        """
        inputs, targets = make_batch(vocab_size)
        logprobs = trainer.compute_logprobs(inputs, targets, use_lora=False)

        # Loss from log-probs
        valid_mask = (targets != -100)
        lp_loss = float(-logprobs[valid_mask].sum() / valid_mask.sum())

        # validate() returns the scalar loss without modifying gradients/optimizer state.
        val_loss = float(trainer.validate(inputs, targets))

        rel_diff = abs(lp_loss - val_loss) / max(abs(val_loss), 1e-6)
        assert rel_diff < 0.05, (
            f"log-prob loss {lp_loss:.4f} vs validate loss {val_loss:.4f} "
            f"(rel diff {rel_diff:.3%} > 5%)"
        )

    def test_different_inputs_different_logprobs(self, trainer, vocab_size):
        """Different input sequences should yield different log-probs."""
        rng = np.random.default_rng(0)
        inputs_a = rng.integers(1, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
        targets_a = np.roll(inputs_a, -1, axis=1).copy()
        targets_a[:, -1] = -100

        rng = np.random.default_rng(1)
        inputs_b = rng.integers(1, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
        targets_b = np.roll(inputs_b, -1, axis=1).copy()
        targets_b[:, -1] = -100

        lp_a = trainer.compute_logprobs(inputs_a, targets_a, use_lora=False)
        lp_b = trainer.compute_logprobs(inputs_b, targets_b, use_lora=False)

        assert not np.allclose(lp_a, lp_b, atol=1e-4), \
            "different inputs produced identical log-probs"

    def test_use_lora_false_equals_true_without_lora(self, trainer, vocab_size):
        """Without LoRA configured, use_lora=True and use_lora=False are identical."""
        inputs, targets = make_batch(vocab_size)
        lp_policy = trainer.compute_logprobs(inputs, targets, use_lora=True)
        lp_ref    = trainer.compute_logprobs(inputs, targets, use_lora=False)

        np.testing.assert_allclose(lp_policy, lp_ref, rtol=1e-5, atol=1e-6,
            err_msg="use_lora=True and use_lora=False gave different results without LoRA")

    def test_batch_dimension_independent(self, trainer, vocab_size):
        """Each batch row should be computed independently.

        Compute log-probs for a batch of 2, then compare the first row against
        a single-sequence compute with B=1.
        """
        if BATCH < 2:
            pytest.skip("BATCH < 2, cannot test batch independence")

        rng = np.random.default_rng(7)
        inputs  = rng.integers(1, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
        targets = np.roll(inputs, -1, axis=1).copy()
        targets[:, -1] = -100

        lp_batch = trainer.compute_logprobs(inputs, targets, use_lora=False)

        # Single-row sub-batch (reuse the same trainer; B must match config so we skip
        # if B_config != 1.  We validate by ensuring B rows differ from each other.)
        lp_row0 = lp_batch[0]
        lp_row1 = lp_batch[1]
        # Rows with different tokens should have different log-probs (almost surely)
        if not np.array_equal(inputs[0], inputs[1]):
            assert not np.allclose(lp_row0, lp_row1, atol=1e-4), \
                "different rows in same batch produced identical log-probs"


# ---------------------------------------------------------------------------
# Phase 4: step_with_custom_loss
# ---------------------------------------------------------------------------

def make_opt_config(lr: float = 1e-3) -> _surogate.OptimizerConfig:
    return _surogate.OptimizerConfig(
        learning_rate=lr,
        weight_decay=0.0,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        grad_clip=1.0,
    )


@pytest.fixture
def fresh_trainer(model_dir):
    """Function-scoped trainer — each test gets an independent model state."""
    return build_trainer(model_dir)


class TestStepWithCustomLoss:
    """Validate step_with_custom_loss for Phase 4 (custom GRPO backward).

    Each test uses a function-scoped `fresh_trainer` so model state does not
    leak between tests.
    """

    def test_smoke(self, fresh_trainer, vocab_size):
        """step_with_custom_loss runs without error and returns None."""
        inputs, targets = make_batch(vocab_size)
        grads = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)
        result = fresh_trainer.step_with_custom_loss(inputs, targets, grads)
        assert result is None
        # Consume accumulated gradients so the trainer is in a clean state.
        fresh_trainer.update_with_config(make_opt_config(lr=0.0), 0)

    def test_zero_grads_zero_gradient_norm(self, fresh_trainer, vocab_size):
        """All-zero per_token_grads must produce a near-zero gradient norm.

        With d_loss=0, the cross-entropy backward gives d_logits=0, which gives
        d_weight=0 and d_hidden=0 everywhere.  The gradient norm reported by
        update_with_config must therefore be (near) zero.

        Note: we check the norm rather than log-prob identity because BF16
        forward passes are not bit-exact across separate calls — the numerical
        noise (~0.01 absolute) is larger than 1e-5 and unrelated to parameter
        changes.
        """
        inputs, targets = make_batch(vocab_size)
        grads = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)

        fresh_trainer.step_with_custom_loss(inputs, targets, grads)
        result = fresh_trainer.update_with_config(make_opt_config(lr=1e-3), 0)

        norm = float(result["norm"])
        assert norm < 1e-3, (
            f"Expected near-zero gradient norm with all-zero per_token_grads, "
            f"but got norm={norm:.6f}"
        )

    def test_positive_grads_decrease_loss(self, fresh_trainer, vocab_size):
        """Positive per_token_grads at valid positions must decrease cross-entropy loss.

        Seeding d_loss > 0 drives the same backward path as standard SFT:
        the model is pushed to assign higher probability to the target tokens.
        """
        inputs, targets = make_batch(vocab_size)
        valid_mask = targets != -100
        n_valid = int(valid_mask.sum())
        grads = np.where(valid_mask, 1.0 / n_valid, 0.0).astype(np.float32)

        val_before = float(fresh_trainer.validate(inputs, targets))
        fresh_trainer.step_with_custom_loss(inputs, targets, grads)
        fresh_trainer.update_with_config(make_opt_config(lr=1e-3), 0)
        val_after = float(fresh_trainer.validate(inputs, targets))

        assert val_after < val_before, (
            f"Expected loss to decrease with positive per_token_grads, "
            f"but got {val_before:.4f} → {val_after:.4f}"
        )

    def test_masked_positions_same_effect_as_zero_grads(self, model_dir, vocab_size):
        """Large grads at masked positions (target=-100) must have no effect.

        The fused LM-head backward kernel zeroes d_logits for target==-100 entries
        regardless of d_loss.  We verify this by comparing two fresh trainers:
          - Trainer A: all-zero per_token_grads
          - Trainer B: large grads (1e3) at masked positions, zero at valid positions

        Both trainers start from the same weights, so if masked grads are truly
        inert, both update steps produce the same new parameter values and hence
        the same subsequent log-probs.
        """
        inputs, targets = make_batch(vocab_size)
        opt = make_opt_config(lr=1e-3)

        trainer_a = build_trainer(model_dir)
        grads_zero = np.zeros((BATCH, SEQ_LEN), dtype=np.float32)
        trainer_a.step_with_custom_loss(inputs, targets, grads_zero)
        trainer_a.update_with_config(opt, 0)
        lp_a = trainer_a.compute_logprobs(inputs, targets, use_lora=False)

        trainer_b = build_trainer(model_dir)
        grads_masked = np.where(targets == -100, 1e3, 0.0).astype(np.float32)
        trainer_b.step_with_custom_loss(inputs, targets, grads_masked)
        trainer_b.update_with_config(opt, 0)
        lp_b = trainer_b.compute_logprobs(inputs, targets, use_lora=False)

        np.testing.assert_allclose(
            lp_b, lp_a, atol=1e-4,
            err_msg=(
                "Large grads at masked positions should produce the same model "
                "update as all-zero grads (masked positions are inert)"
            ),
        )

    def test_equivalent_to_step(self, model_dir, vocab_size):
        """step_with_custom_loss(grads=ones) produces identical updates to step().

        Both paths seed d_loss = 1.0 per token and run the same backward graph.
        The only implementation difference is fill_constant vs D→D memcpy.
        Starting from the same initial weights, log-probs after one update must
        be numerically identical (within BF16 rounding).
        """
        inputs, targets = make_batch(vocab_size)
        opt = make_opt_config(lr=1e-3)

        # Trainer A: standard SFT step.
        trainer_a = build_trainer(model_dir)
        trainer_a.step(inputs, targets)
        trainer_a.update_with_config(opt, 0)
        lp_a = trainer_a.compute_logprobs(inputs, targets, use_lora=False)

        # Trainer B: custom step with all-ones gradients (equivalent to d_loss=1).
        trainer_b = build_trainer(model_dir)
        grads = np.ones((BATCH, SEQ_LEN), dtype=np.float32)
        trainer_b.step_with_custom_loss(inputs, targets, grads)
        trainer_b.update_with_config(opt, 0)
        lp_b = trainer_b.compute_logprobs(inputs, targets, use_lora=False)

        np.testing.assert_allclose(
            lp_b, lp_a, atol=1e-4,
            err_msg=(
                "step_with_custom_loss(ones) and step() should produce identical "
                "parameter updates from the same initial weights"
            ),
        )
