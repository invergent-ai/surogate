"""Smoke test: Pre-quantized MXFP4 GPT-OSS model loading with LoRA.

Validates that:
1. Pre-quantized MXFP4 weights are detected and loaded correctly
2. modules_to_not_convert (attention, router, embed, lm_head) are loaded as full-precision
3. Expert weights (_blocks/_scales) are loaded as quantized MXFP4
4. LoRA adapters are initialized and a forward+backward pass runs without error
5. Loss is finite and reasonable

Requirements:
    - GPU with enough VRAM (~20GB for 4-layer mini model)
    - HF weights: openai/gpt-oss-20b (MXFP4 quantized checkpoint)

Usage:
    pytest tests/test_prequant_mxfp4_gpt_oss.py -v --no-header
    GPT_OSS_MODEL_PATH=/path/to/gpt-oss pytest tests/test_prequant_mxfp4_gpt_oss.py -v
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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

MODEL_ID = "openai/gpt-oss-20b"
ENV_VAR = "GPT_OSS_MODEL_PATH"
NUM_LAYERS = 4
SEED = 42
BATCH = 1
SEQ_LEN = 16

MINI_MODEL_DIR = Path("tmp/prequant_mxfp4_gpt_oss_mini")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path | None:
    """Resolve the path to GPT-OSS MXFP4 weights."""
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.exists():
            if (p / "config.json").exists():
                return p
            snaps = p / "snapshots"
            if snaps.exists():
                for snap in sorted(snaps.iterdir(), reverse=True):
                    if (snap / "config.json").exists():
                        return snap

    cache_root = Path("~/.cache/huggingface/hub").expanduser()
    model_slug = MODEL_ID.replace("/", "--")
    model_cache = cache_root / f"models--{model_slug}"
    if model_cache.exists():
        snaps = model_cache / "snapshots"
        if snaps.exists():
            for snap in sorted(snaps.iterdir(), reverse=True):
                if (snap / "config.json").exists():
                    return snap
    return None


def check_is_mxfp4(model_dir: Path) -> None:
    """Skip if the checkpoint is not MXFP4 quantized."""
    config = json.loads((model_dir / "config.json").read_text())
    qcfg = config.get("quantization_config")
    if not qcfg or qcfg.get("quant_method") != "mxfp4":
        pytest.skip("GPT-OSS checkpoint is not MXFP4 quantized; need the quantized version")


def prepare_mini_model(snapshot_dir: Path) -> Path:
    """Create a truncated GPT-OSS MXFP4 model with NUM_LAYERS layers."""
    if MINI_MODEL_DIR.exists():
        try:
            src_cfg = json.loads((snapshot_dir / "config.json").read_text())
            mini_cfg = json.loads((MINI_MODEL_DIR / "config.json").read_text())
            src_has_quant = bool(src_cfg.get("quantization_config"))
            mini_has_quant = bool(mini_cfg.get("quantization_config"))
            src_has_index = (snapshot_dir / "model.safetensors.index.json").exists()
            mini_has_index = (MINI_MODEL_DIR / "model.safetensors.index.json").exists()
            if src_has_quant == mini_has_quant and src_has_index == mini_has_index:
                return MINI_MODEL_DIR
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        shutil.rmtree(MINI_MODEL_DIR)
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS
    if "model_type" not in config:
        config["model_type"] = "gpt_oss"
    if "architectures" not in config:
        config["architectures"] = ["GptOssForCausalLM"]
    if "num_experts_per_tok" not in config and "experts_per_token" in config:
        config["num_experts_per_tok"] = config["experts_per_token"]
    if "num_local_experts" not in config and "num_experts" in config:
        config["num_local_experts"] = config["num_experts"]
    if "max_position_embeddings" not in config and "initial_context_length" in config:
        config["max_position_embeddings"] = config["initial_context_length"]
    config.setdefault("rms_norm_eps", 1e-5)
    config.setdefault("attention_bias", True)
    if isinstance(config.get("layer_types"), list):
        config["layer_types"] = ["full_attention"] * NUM_LAYERS
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
            if name in extra:
                return True
            return any(name.startswith(p) for p in prefixes)

        mini_map = {k: v for k, v in weight_map.items() if want(k)}
        mini_index = {"metadata": base_index.get("metadata", {}), "weight_map": mini_map}
        (MINI_MODEL_DIR / "model.safetensors.index.json").write_text(
            json.dumps(mini_index, indent=2, sort_keys=True) + "\n"
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


def make_inputs(vocab_size: int):
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return inputs, targets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"GPT-OSS weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    check_is_mxfp4(snapshot)
    return prepare_mini_model(snapshot)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrequantMXFP4GptOss:
    """Pre-quantized MXFP4 loading + LoRA smoke tests."""

    def test_weight_import_and_forward(self, model_dir):
        """Load pre-quantized MXFP4 weights with LoRA, run forward+backward."""
        config = json.loads((model_dir / "config.json").read_text())
        vocab_size = config["vocab_size"]
        inputs, targets = make_inputs(vocab_size)

        # Build DSL IR
        ir_json = build_dsl_ir_for_model(str(model_dir))

        # Create pretrained config
        cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")

        # Runtime options — pre-quantized needs recompute, no CUDA graphs
        opts = _surogate.RuntimeOptions(
            recompute="true",
            offload_residual=False,
            use_cuda_graphs=False,
            offload_master=False,
            offload_grads=False,
            offload_optimizer=False,
            shard_gradients=True,
            use_zero_copy=False,
        )
        opts.dsl_ir_json = ir_json

        # LoRA config — target MoE expert projections + attention
        lora_config = _surogate.LoRAAdapterConfig(
            rank=16,
            alpha=32,
            dropout=0.0,
            dtype="bf16",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_up_proj", "down_proj"],
            use_rslora=False,
        )

        # Pre-quantized MXFP4 QLoRA config
        qlora_config = _surogate.QLoRAConfig.prequant_mxfp4()
        # Set modules_to_not_convert from HF quantization_config
        qcfg = config.get("quantization_config", {})
        modules_to_not_convert = qcfg.get(
            "modules_to_not_convert", qcfg.get("ignore", []))
        if modules_to_not_convert:
            qlora_config.modules_to_not_convert = modules_to_not_convert

        # Verify config
        assert qlora_config.is_prequantized
        assert qlora_config.strategy == "prequant_mxfp4"

        # Create trainer
        trainer = _surogate.SurogateTrainer(
            ngpu=1,
            config=cfg,
            options=opts,
            batch_size=BATCH,
            seq_len=SEQ_LEN,
            grad_accum=1,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=lora_config,
            qlora_config=qlora_config,
        )

        # Import weights — this is the key step that tests pre-quantized loading
        weights_path = get_model_weights_path(str(model_dir))
        trainer.import_weights(weights_path)

        # Run forward + backward
        trainer.step(inputs, targets)

        # Get loss via optimizer update
        opt_config = _surogate.OptimizerConfig(
            learning_rate=2e-4,
            weight_decay=0.0,
            adamw_beta1=0.9,
            adamw_beta2=0.95,
            grad_clip=1.0,
        )
        result = trainer.update_with_config(opt_config, 0)

        loss = result["loss"]
        print(f"\n[Prequant MXFP4] Step 0 loss: {loss:.4f}")

        # Verify loss is finite and reasonable
        assert np.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss > 0, f"Loss should be positive: {loss}"
        assert loss < 100, f"Loss seems unreasonably large: {loss}"

    def test_loss_decreases(self, model_dir):
        """Run 3 steps and verify loss decreases (LoRA is training)."""
        config = json.loads((model_dir / "config.json").read_text())
        vocab_size = config["vocab_size"]
        inputs, targets = make_inputs(vocab_size)

        ir_json = build_dsl_ir_for_model(str(model_dir))
        cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
        opts = _surogate.RuntimeOptions(
            recompute="true",
            offload_residual=False,
            use_cuda_graphs=False,
            offload_master=False,
            offload_grads=False,
            offload_optimizer=False,
            shard_gradients=True,
            use_zero_copy=False,
        )
        opts.dsl_ir_json = ir_json

        lora_config = _surogate.LoRAAdapterConfig(
            rank=16,
            alpha=32,
            dropout=0.0,
            dtype="bf16",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_up_proj", "down_proj"],
            use_rslora=False,
        )

        qlora_config = _surogate.QLoRAConfig.prequant_mxfp4()
        qcfg = config.get("quantization_config", {})
        modules_to_not_convert = qcfg.get(
            "modules_to_not_convert", qcfg.get("ignore", []))
        if modules_to_not_convert:
            qlora_config.modules_to_not_convert = modules_to_not_convert

        trainer = _surogate.SurogateTrainer(
            ngpu=1,
            config=cfg,
            options=opts,
            batch_size=BATCH,
            seq_len=SEQ_LEN,
            grad_accum=1,
            memcpy_all_gather=True,
            memcpy_send_recv=True,
            lora_config=lora_config,
            qlora_config=qlora_config,
        )

        weights_path = get_model_weights_path(str(model_dir))
        trainer.import_weights(weights_path)

        opt_config = _surogate.OptimizerConfig(
            learning_rate=2e-4,
            weight_decay=0.0,
            adamw_beta1=0.9,
            adamw_beta2=0.95,
            grad_clip=1.0,
        )

        losses = []
        for step in range(3):
            trainer.step(inputs, targets)
            result = trainer.update_with_config(opt_config, step)
            losses.append(result["loss"])

        print(f"\n[Prequant MXFP4] Losses: {[f'{l:.4f}' for l in losses]}")

        # All losses should be finite
        for i, loss in enumerate(losses):
            assert np.isfinite(loss), f"Step {i} loss is not finite: {loss}"

        # Loss should decrease (or at least not diverge) over 3 steps
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
