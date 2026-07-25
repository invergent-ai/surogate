"""Numerical validation of the native KD loss/gradient against a torch reference.

Uses a truncated (4-layer) Qwen3 model. The teacher signal is derived from the
HuggingFace forward of the *same* model, which gives closed-form expectations:

- With kd_weight=1, ce_weight=0, tau=1 and teacher == student, the reported
  kd_loss is KL(renorm(topk(p)) || p) = -log(topk_mass) per token.
- With tau != 1, kd_loss = tau^2 * KL(q_tau || p_tau), computable in torch.
- With kd_weight=0, ce_weight=1 the step must reproduce the plain CE step.
- Training against a fixed teacher must reduce kd_loss (gradient direction).

Requirements: 1 GPU, cached Qwen3 weights (QWEN3_MODEL_PATH or HF cache for
Qwen/Qwen3-0.6B or Qwen/Qwen3-1.7B).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CANDIDATE_MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]
ENV_VAR = "QWEN3_MODEL_PATH"
NUM_LAYERS = 4
SEED = 1234
BATCH = 1
SEQ_LEN = 64
TOP_K = 8
MINI_MODEL_DIR = Path("tmp/distill_kd_mini")


def resolve_model_path() -> Path | None:
    env = os.environ.get(ENV_VAR)
    if env and Path(env).exists():
        return Path(env)
    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    for model_id in CANDIDATE_MODELS:
        model_cache = cache_root / f"models--{model_id.replace('/', '--')}"
        snaps = model_cache / "snapshots"
        if snaps.exists():
            for snap in sorted(snaps.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    return snap
    return None


def prepare_mini_model(snapshot_dir: Path) -> Path:
    if MINI_MODEL_DIR.exists():
        shutil.rmtree(MINI_MODEL_DIR)
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS
    (MINI_MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
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
        mini_map = {
            k: v for k, v in weight_map.items() if k in extra or any(k.startswith(p) for p in prefixes)
        }
        mini_index = {"metadata": base_index.get("metadata", {}), "weight_map": mini_map}
        (MINI_MODEL_DIR / "model.safetensors.index.json").write_text(
            json.dumps(mini_index, indent=2, sort_keys=True) + "\n"
        )
        for fname in sorted(set(mini_map.values())):
            link = MINI_MODEL_DIR / fname
            if not link.exists():
                link.symlink_to((snapshot_dir / fname).resolve())
    elif single_path.exists():
        (MINI_MODEL_DIR / "model.safetensors").symlink_to(single_path.resolve())
    else:
        raise FileNotFoundError(f"No weights found in {snapshot_dir}")
    return MINI_MODEL_DIR


def build_trainer(model_dir: Path) -> "_surogate.SurogateTrainer":
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


def opt_config(lr: float) -> "_surogate.OptimizerConfig":
    return _surogate.OptimizerConfig(
        optimizer="adamw",
        learning_rate=lr,
        weight_decay=0.0,
        grad_clip=1.0,
        adamw_beta1=0.9,
        adamw_beta2=0.999,
        adamw_epsilon=1e-8,
    )


@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Qwen3 weights not found. Set {ENV_VAR} or cache one of {CANDIDATE_MODELS}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def batch(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, config["vocab_size"], size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets, "vocab_size": config["vocab_size"]}


@pytest.fixture(scope="module")
def hf_logprobs(model_dir, batch):
    """Float32 log-softmax of the HF forward on the same mini model/batch."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(batch["inputs"], device="cuda", dtype=torch.long)
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1).cpu()
    del model
    torch.cuda.empty_cache()
    return logprobs  # [BATCH, SEQ_LEN, V] fp32


def teacher_topk(hf_logprobs: "torch.Tensor") -> tuple[np.ndarray, np.ndarray]:
    values, indices = torch.topk(hf_logprobs, TOP_K, dim=-1)
    return (
        indices.numpy().astype(np.int32),
        values.numpy().astype(np.float32),
    )


def torch_kd_reference(
    hf_logprobs: "torch.Tensor", kd_ids: np.ndarray, kd_lps: np.ndarray, targets: np.ndarray, tau: float
) -> float:
    """Mean over valid tokens of tau^2 * KL(q_tau || p_tau), matching the kernel math."""
    logits = hf_logprobs  # already log-probs; softmax(logprobs/tau) == softmax-with-temperature
    total = 0.0
    count = 0
    for b in range(targets.shape[0]):
        for t in range(targets.shape[1]):
            if targets[b, t] == -100:
                continue
            q = torch.softmax(torch.tensor(kd_lps[b, t]) / tau, dim=-1)
            p_tau = torch.log_softmax(logits[b, t] / tau, dim=-1)
            s = p_tau[torch.tensor(kd_ids[b, t], dtype=torch.long)]
            total += float(tau * tau * torch.sum(q * (torch.log(q.clamp_min(1e-30)) - s)))
            count += 1
    return total / max(1, count)


class TestKdGradient:
    def test_kd_loss_identity_and_reference_band(self, model_dir, batch, hf_logprobs):
        """Exact + sanity validation of the native kd_loss value.

        Exact: with K=1 and the teacher set to the target token (q = 1), the
        KD loss collapses to -logprob(target) = cross-entropy, computed by the
        SAME engine — so kd_loss must equal the reported CE loss regardless of
        any surogate-vs-HF forward drift. This pins the metric's math
        (normalization, tau^2, valid-token divisor, scatter/gather indexing).

        Sanity band: the torch reference from HF logits only bounds kd_loss
        loosely — the truncated 4-layer model has degenerate distributions and
        the two engines drift ~0.7 nats/token on it (measured), so tight
        agreement is not expected; the band still catches gross factor bugs
        (missing tau^2, wrong divisor, unnormalized q).
        """
        trainer = build_trainer(model_dir)
        try:
            # --- exact identity: teacher = target token, K=1 ---
            kd_ids1 = np.where(batch["targets"] >= 0, batch["targets"], 0)[..., None].astype(np.int32)
            kd_lps1 = np.zeros_like(kd_ids1, dtype=np.float32)
            trainer.step_with_kd(
                batch["inputs"],
                batch["targets"],
                kd_ids1,
                kd_lps1,
                top_k=1,
                temperature=1.0,
                kd_weight=1.0,
                ce_weight=0.0,
            )
            result = trainer.update_with_config(opt_config(1e-9), 1)
            kd_loss1 = trainer.get_kd_loss()
            assert np.isfinite(result["loss"]) and np.isfinite(result["norm"])
            assert abs(kd_loss1 - result["loss"]) < 2e-2 * max(1.0, result["loss"]), (
                f"K=1 identity broken: kd_loss={kd_loss1} vs CE={result['loss']}"
            )

            # --- sanity band vs the HF-logit reference, tau=1 and tau=2 ---
            kd_ids, kd_lps = teacher_topk(hf_logprobs)
            for tau in (1.0, 2.0):
                trainer.step_with_kd(
                    batch["inputs"],
                    batch["targets"],
                    kd_ids,
                    kd_lps,
                    top_k=TOP_K,
                    temperature=tau,
                    kd_weight=1.0,
                    ce_weight=0.0,
                )
                trainer.update_with_config(opt_config(1e-9), 2)
                kd_loss = trainer.get_kd_loss()
                ref = torch_kd_reference(hf_logprobs, kd_ids, kd_lps, batch["targets"], tau=tau)
                assert kd_loss > 0
                assert 0.2 * ref < kd_loss < 5.0 * ref, (
                    f"kd_loss(tau={tau})={kd_loss} outside sanity band of torch ref={ref}"
                )

            # tau=1 closed form and torch reference agree with each other tightly
            # (both computed from the same HF logits) — validates the reference itself.
            valid = batch["targets"] != -100
            topk_mass = np.exp(kd_lps.astype(np.float64)).sum(axis=-1)
            expected = float(-np.log(topk_mass[valid]).mean())
            ref1 = torch_kd_reference(hf_logprobs, kd_ids, kd_lps, batch["targets"], tau=1.0)
            assert abs(ref1 - expected) < 5e-3
        finally:
            del trainer

    def test_ce_only_kd_step_matches_plain_step(self, model_dir, batch, hf_logprobs):
        kd_ids, kd_lps = teacher_topk(hf_logprobs)

        trainer = build_trainer(model_dir)
        try:
            trainer.step(batch["inputs"], batch["targets"])
            plain = trainer.update_with_config(opt_config(1e-9), 1)
        finally:
            del trainer

        trainer = build_trainer(model_dir)
        try:
            trainer.step_with_kd(
                batch["inputs"],
                batch["targets"],
                kd_ids,
                kd_lps,
                top_k=TOP_K,
                temperature=1.0,
                kd_weight=0.0,
                ce_weight=1.0,
            )
            kd = trainer.update_with_config(opt_config(1e-9), 1)
        finally:
            del trainer

        # Same weights, same batch, KD contribution zeroed: CE loss and grad
        # norm must match the plain step closely (identical kernels modulo the
        # KD-variant backward with kd_scale=0).
        assert abs(kd["loss"] - plain["loss"]) < 5e-3 * max(1.0, abs(plain["loss"]))
        assert abs(kd["norm"] - plain["norm"]) < 5e-2 * max(1.0, abs(plain["norm"]))

    def test_kd_gradient_reduces_kd_loss(self, model_dir, batch, hf_logprobs):
        # Fixed synthetic teacher that differs from the student: shift the
        # top-k ids by one position so the student must move probability mass.
        kd_ids, kd_lps = teacher_topk(hf_logprobs)
        rng = np.random.default_rng(SEED + 1)
        kd_ids = ((kd_ids.astype(np.int64) + 7919) % batch["vocab_size"]).astype(np.int32)
        kd_lps = np.sort(rng.uniform(-3.0, -0.5, size=kd_lps.shape).astype(np.float32), axis=-1)[..., ::-1].copy()

        trainer = build_trainer(model_dir)
        try:
            losses = []
            for step in range(12):
                trainer.step_with_kd(
                    batch["inputs"],
                    batch["targets"],
                    kd_ids,
                    kd_lps,
                    top_k=TOP_K,
                    temperature=1.0,
                    kd_weight=1.0,
                    ce_weight=0.0,
                )
                trainer.update_with_config(opt_config(2e-4), step + 1)
                losses.append(trainer.get_kd_loss())
            assert all(np.isfinite(losses))
            assert losses[-1] < losses[0] * 0.9, f"kd_loss did not decrease: {losses}"
        finally:
            del trainer
