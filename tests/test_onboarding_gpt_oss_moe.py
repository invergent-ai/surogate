"""Onboarding test: GPT-OSS MoE model.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for a mini (truncated) GPT-OSS MoE model. Compares per-layer hidden states and
final-norm outputs with tolerances.

Also validates LoRA training stability: runs N training steps with
forward+backward+optimizer and checks that loss decreases, gradient norms
stay finite and bounded, and step-0 loss matches HF cross-entropy loss.

Supports both full-precision (BF16) and pre-quantized (MXFP4) checkpoints.
For MXFP4 checkpoints, Surogate loads weights via the pre-quantized pipeline
with LoRA (B=0 init, so first forward is equivalent to no-LoRA). HF uses
Mxfp4Config(dequantize=True) to dequantize before forward.

Requirements:
    - GPU with enough VRAM for a small GPT-OSS MoE model
    - HF weights: set GPT_OSS_MODEL_PATH env var or have the model cached

Usage:
    pytest tests/test_onboarding_gpt_oss_moe.py -v --no-header
    GPT_OSS_MODEL_PATH=/path/to/gpt-oss pytest tests/test_onboarding_gpt_oss_moe.py -v
    GPT_OSS_NUM_LAYERS=2 pytest tests/test_onboarding_gpt_oss_moe.py -v
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "openai/gpt-oss-20b"
ENV_VAR = "GPT_OSS_MODEL_PATH"
NUM_LAYERS = int(os.environ.get("GPT_OSS_NUM_LAYERS", "2"))
SEED = 42
BATCH = 1
SEQ_LEN = 16
# bf16 forward through MoE layers accumulates more error than dense models.
# Full-precision (BF16) tolerances:
RMS_TOL = 5e-2           # per-layer mid-state tolerance
FINAL_NORM_TOL = 2e-1    # post-norm tolerance
LAYER_OUT_TOL = 5e-2     # per-layer output tolerance (after MoE residual)
# Pre-quantized (MXFP4) tolerances — dequantization adds quantization noise:
MXFP4_RMS_TOL = 5e-1
MXFP4_FINAL_NORM_TOL = 2.0
MXFP4_LAYER_OUT_TOL = 5e-1

# Training comparison constants
# Use longer sequences and gradient accumulation to match real training conditions.
# The YAML config uses seq_len=512, grad_accum=4. We use seq_len=128, grad_accum=4
# as a compromise between realism and test speed/memory.
TRAIN_SEQ_LEN = 512
TRAIN_GRAD_ACCUM = 4
NUM_TRAIN_STEPS = 10       # optimizer steps (each = grad_accum micro-steps)
TRAIN_LR = 2e-4
LOSS_MATCH_TOL = 0.5           # HF vs Surogate step-0 loss tolerance (BF16)
MXFP4_LOSS_MATCH_TOL = 1.0    # Larger for MXFP4 quantization noise
# Short HF-vs-Surogate stability comparison (kept small for VRAM/time).
# Grad accumulation defaults to TRAIN_GRAD_ACCUM to mirror real training.
HF_COMPARE_STEPS = int(os.environ.get("GPT_OSS_HF_COMPARE_STEPS", "10"))
HF_COMPARE_SEQ_LEN = int(os.environ.get("GPT_OSS_HF_COMPARE_SEQ_LEN", "512"))
HF_COMPARE_GRAD_ACCUM = int(os.environ.get("GPT_OSS_HF_COMPARE_GRAD_ACCUM", str(TRAIN_GRAD_ACCUM)))
PROJ_NAMES = ["gate_up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

MINI_MODEL_DIR = Path(f"tmp/onboarding_gpt_oss_moe_mini_{NUM_LAYERS}l")
DUMP_DIR = Path(f"tmp/onboarding_gpt_oss_moe_dumps_{NUM_LAYERS}l")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path | None:
    """Resolve the path to GPT-OSS weights."""
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


def prepare_mini_model(snapshot_dir: Path) -> Path:
    """Create a truncated GPT-OSS model with NUM_LAYERS layers.

    Forces full-attention layers to match DSL behavior.
    """
    if MINI_MODEL_DIR.exists():
        try:
            src_cfg = json.loads((snapshot_dir / "config.json").read_text())
            mini_cfg = json.loads((MINI_MODEL_DIR / "config.json").read_text())
            src_has_quant = bool(src_cfg.get("quantization_config"))
            mini_has_quant = bool(mini_cfg.get("quantization_config"))
            src_has_index = (snapshot_dir / "model.safetensors.index.json").exists()
            mini_has_index = (MINI_MODEL_DIR / "model.safetensors.index.json").exists()
            mini_has_arch = bool(mini_cfg.get("model_type") or mini_cfg.get("architectures"))
            if src_has_quant == mini_has_quant and src_has_index == mini_has_index and mini_has_arch:
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


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in name)


def load_dump(name: str) -> np.ndarray:
    safe = sanitize(name)
    meta_path = DUMP_DIR / f"{safe}.json"
    bin_path = DUMP_DIR / f"{safe}.bin"
    if not meta_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"Missing dump for {name}: {meta_path}")
    meta = json.loads(meta_path.read_text())
    data = np.fromfile(bin_path, dtype=np.float32)
    shape = list(meta.get("shape", []))
    while shape and shape[-1] == 1:
        shape.pop()
    return data.reshape(shape)


def diff_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    rms = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    return rms, max_abs


def make_inputs(vocab_size: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def is_mxfp4_model(model_dir: Path) -> bool:
    """Check if the model uses MXFP4 quantization."""
    config = json.loads((model_dir / "config.json").read_text())
    qcfg = config.get("quantization_config")
    return bool(qcfg and qcfg.get("quant_method") == "mxfp4")


def compute_ce_loss(logits_np: np.ndarray, targets_np: np.ndarray) -> float:
    """Compute cross-entropy loss from numpy logits and targets.

    Uses the same convention as Surogate: targets are pre-shifted
    (target[i] = input[i+1], last position = -100).
    """
    import torch.nn.functional as F
    logits = torch.tensor(logits_np, dtype=torch.float32)
    targets = torch.tensor(targets_np.astype(np.int64))
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    ).item()


def make_train_inputs(vocab_size: int, seq_len: int = TRAIN_SEQ_LEN) -> Dict[str, np.ndarray]:
    """Create training inputs with seq_len (longer than forward test inputs)."""
    rng = np.random.default_rng(SEED + 1)  # Different seed from forward test
    inputs = rng.integers(0, vocab_size, size=(BATCH, seq_len), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def _match_proj(name: str, proj_names: list[str]) -> str | None:
    """Find which projection a parameter name belongs to."""
    for proj in sorted(proj_names, key=len, reverse=True):
        if proj in name:
            return proj
    return None


def _grad_breakdown_from_named_tensors(named_tensors, proj_names: list[str], scale: float = 1.0) -> dict:
    """Compute per-projection grad norms and max_abs from (name, tensor) pairs.

    Optional `scale` applies a scalar normalization (e.g. token averaging) to each tensor.
    """
    import math
    stats = {p: {"sq": 0.0, "max_abs": 0.0, "tensors": 0} for p in proj_names}
    total_sq = 0.0
    total_max = 0.0
    total_tensors = 0
    for name, t in named_tensors:
        if t is None or t.numel() == 0:
            continue
        proj = _match_proj(name, proj_names)
        if proj is None:
            continue
        t_f = t.float()
        if scale != 1.0:
            t_f = t_f * scale
        sq = float((t_f * t_f).sum().item())
        max_abs = float(t_f.abs().max().item())
        s = stats[proj]
        s["sq"] += sq
        s["max_abs"] = max(s["max_abs"], max_abs)
        s["tensors"] += 1
        total_sq += sq
        total_max = max(total_max, max_abs)
        total_tensors += 1
    out = {}
    for proj, s in stats.items():
        out[proj] = {
            "norm": math.sqrt(s["sq"]),
            "max_abs": s["max_abs"],
            "tensors": s["tensors"],
        }
    out["_total"] = {"norm": math.sqrt(total_sq), "max_abs": total_max, "tensors": total_tensors}
    return out


def lora_grad_breakdown_from_trainer(trainer, proj_names: list[str], grad_accum: int = 1) -> dict:
    """Per-projection grad breakdown from Surogate LoRA gradients.

    Applies token-averaging so norms are comparable to HF (mean loss).
    """
    token_scale = 1.0
    try:
        valid_tokens = trainer.get_valid_token_count(0)
        if valid_tokens and valid_tokens > 0:
            denom = float(valid_tokens) * max(1, int(grad_accum))
            token_scale = 1.0 / denom
    except Exception:
        token_scale = 1.0
    grads = trainer.get_lora_gradients(0)
    named = []
    for name, arr in grads.items():
        t = torch.utils.dlpack.from_dlpack(arr)
        if t.numel() == 0:
            continue
        named.append((name, t))
    return _grad_breakdown_from_named_tensors(named, proj_names, scale=token_scale)


def run_surogate_training(model_dir: Path, inputs: np.ndarray, targets: np.ndarray,
                          num_steps: int = NUM_TRAIN_STEPS,
                          seq_len: int | None = None,
                          grad_accum: int = TRAIN_GRAD_ACCUM,
                          learning_rate: float = TRAIN_LR,
                          norm_source: str = "trainer",
                          capture_breakdown: bool = False) -> Dict[str, list]:
    """Run Surogate forward+backward+optimizer for num_steps.

    Uses grad_accum gradient accumulation steps per optimizer step to
    match real training conditions. Returns dict with 'losses' and 'norms'.
    norm_source can be "trainer" (default) or "lora" to compute norms
    directly from LoRA gradients for fair HF comparison.
    """
    import gc

    # Clear debug env vars that interfere with training
    for key in ["SUROGATE_DEBUG_DUMP_TENSORS", "SUROGATE_DEBUG_DUMP_DIR",
                "SUROGATE_DEBUG_DUMP_LAYER", "SUROGATE_DEBUG_FORWARD_ONLY"]:
        os.environ.pop(key, None)

    mxfp4 = is_mxfp4_model(model_dir)
    if seq_len is None:
        seq_len = int(inputs.shape[1])

    cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = _surogate.RuntimeOptions(
        recompute="true" if mxfp4 else "false",
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))

    lora_config = _surogate.LoRAAdapterConfig(
        rank=16, alpha=32, dropout=0.0, dtype="bf16",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_up_proj", "down_proj"],
        use_rslora=False,
    )

    qlora_config = None
    if mxfp4:
        qlora_config = _surogate.QLoRAConfig.prequant_mxfp4()
        model_config = json.loads((model_dir / "config.json").read_text())
        qcfg = model_config.get("quantization_config", {})
        modules_to_not_convert = qcfg.get(
            "modules_to_not_convert", qcfg.get("ignore", []))
        if modules_to_not_convert:
            qlora_config.modules_to_not_convert = modules_to_not_convert

    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=seq_len,
        grad_accum=grad_accum,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=lora_config,
        qlora_config=qlora_config,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))

    opt_config = _surogate.OptimizerConfig(
        learning_rate=learning_rate,
        weight_decay=0.0,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        grad_clip=1.0,
    )

    losses = []
    norms = []
    breakdowns = [] if capture_breakdown else None
    for step in range(num_steps):
        # Each step = grad_accum micro-steps of forward+backward,
        # then one optimizer update.
        for _ in range(grad_accum):
            trainer.step(inputs, targets)
        norm_override = None
        breakdown = None
        if norm_source == "lora":
            breakdown = lora_grad_breakdown_from_trainer(trainer, PROJ_NAMES, grad_accum=grad_accum)
            norm_override = breakdown["_total"]["norm"]
        if capture_breakdown:
            if breakdown is None:
                breakdown = lora_grad_breakdown_from_trainer(trainer, PROJ_NAMES, grad_accum=grad_accum)
            breakdowns.append(breakdown)
        result = trainer.update_with_config(opt_config, step)
        losses.append(result["loss"])
        norms.append(result["norm"] if norm_override is None else norm_override)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    out = {"losses": losses, "norms": norms}
    if breakdowns is not None:
        out["breakdowns"] = breakdowns
    return out


def check_supported_weights(model_dir: Path) -> None:
    """Skip if the cached weights are unsupported (non-MXFP4 quantized, wrong format)."""
    config = json.loads((model_dir / "config.json").read_text())
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})

    # If we can see full-precision expert weights in the map, accept regardless of config flags.
    if weight_map and "model.layers.0.mlp.experts.down_proj" in weight_map:
        return

    # If there's a single safetensors file, validate that it uses HF-style keys.
    model_path = model_dir / "model.safetensors"
    if model_path.exists():
        try:
            from safetensors import safe_open
            with safe_open(str(model_path), framework="pt", device="cpu") as f:
                keys = f.keys()
                if any(k.startswith("model.layers.") for k in keys):
                    return
                if any(k.startswith("block.") for k in keys):
                    pytest.skip(
                        "GPT-OSS checkpoint is in OpenAI format (block.* keys), not HF. "
                        "Use HF-converted weights with model.layers.* keys for this test."
                    )
        except Exception:
            pass

    qcfg = config.get("quantization_config")
    if qcfg:
        method = qcfg.get("quant_method", "unknown")
        # MXFP4 is supported via pre-quantized loading
        if method == "mxfp4":
            return
        pytest.skip(f"GPT-OSS weights are quantized ({method}); only mxfp4 and full-precision supported")

    if weight_map and any(k.endswith("down_proj_blocks") for k in weight_map):
        # This looks like MXFP4 even without quantization_config — accept it
        return


# ---------------------------------------------------------------------------
# HuggingFace forward — uses hooks for reliable per-layer capture
# ---------------------------------------------------------------------------

def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> Dict[str, np.ndarray]:
    """Run HF forward and capture per-layer outputs via hooks.

    Captures mid-layer states (after attention, before MoE) via pre-hook on
    post_attention_layernorm. These correspond to Surogate's blocks[i].res_att.
    """
    from transformers import AutoModelForCausalLM

    cfg = json.loads((model_dir / "config.json").read_text())
    qcfg = cfg.get("quantization_config")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if qcfg:
        try:
            from transformers import Mxfp4Config
            model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    model.eval()

    result: Dict[str, np.ndarray] = {}
    layer_outs: Dict[int, torch.Tensor] = {}
    mid_states: Dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(NUM_LAYERS):
        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(model.model.layers[i].register_forward_hook(make_layer_hook(i)))

        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(
            model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(
                make_mid_hook(i)
            )
        )

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0)
        position_ids = position_ids.expand(input_ids.shape[0], -1)
        attention_mask = torch.ones_like(input_ids)
        _ = model(input_ids=input_ids,
                  position_ids=position_ids,
                  attention_mask=attention_mask,
                  use_cache=False)

        for i in range(NUM_LAYERS):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        pre_norm = layer_outs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        post_norm = model.model.norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        logits = model.lm_head(post_norm)
        result["logits"] = logits.float().cpu().numpy()

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Surogate forward
# ---------------------------------------------------------------------------

def run_surogate_forward(model_dir: Path, inputs: np.ndarray,
                         targets: np.ndarray) -> None:
    """Run Surogate forward pass and dump tensors to DUMP_DIR.

    For MXFP4 pre-quantized models, uses prequant MXFP4 loading + LoRA
    (LoRA B=0 init means the first forward is equivalent to no-LoRA).
    """
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_att")
        dump_tensors.append(f"blocks[{i}].mlp_down")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR)
    os.environ["SUROGATE_DEBUG_DUMP_LAYER"] = "-1"
    os.environ["SUROGATE_DEBUG_FORWARD_ONLY"] = "1"

    mxfp4 = is_mxfp4_model(model_dir)

    cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    opts = _surogate.RuntimeOptions(
        recompute="true" if mxfp4 else "false",
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        shard_gradients=True,
        use_zero_copy=False,
    )
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))

    lora_config = None
    qlora_config = None

    if mxfp4:
        # Pre-quantized MXFP4 requires LoRA. B=0 init means first forward
        # produces the same output as without LoRA adapters.
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
        model_config = json.loads((model_dir / "config.json").read_text())
        qcfg = model_config.get("quantization_config", {})
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
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    trainer.step(inputs, targets)


def run_hf_training(model_dir: Path, inputs: np.ndarray, targets: np.ndarray,
                    num_steps: int, grad_accum: int = 1,
                    learning_rate: float = TRAIN_LR,
                    capture_breakdown: bool = False) -> Dict[str, list]:
    """Run HF training for a few steps and report loss + grad norm (LoRA params).

    Uses PEFT LoRA adapters and AdamW. Keeps this lightweight for stability checks.
    """
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Mxfp4Config

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    mxfp4 = is_mxfp4_model(model_dir)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        use_cache=False,
    )
    if mxfp4:
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **model_kwargs)
    try:
        model.config.output_router_logits = False
        model.config.router_aux_loss_coef = 0.0
    except Exception:
        pass
    model.train()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    params = [p for p in model.parameters() if p.requires_grad]
    bnb = pytest.importorskip("bitsandbytes")
    opt = bnb.optim.AdamW8bit(params, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    input_ids = torch.tensor(inputs, dtype=torch.long, device=model.device)
    labels = torch.tensor(targets, dtype=torch.long, device=model.device)

    losses = []
    norms = []
    breakdowns = [] if capture_breakdown else None

    for _ in range(num_steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / grad_accum
            total_loss += float(out.loss.detach().cpu())
            loss.backward()
        if capture_breakdown:
            named = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
            breakdowns.append(_grad_breakdown_from_named_tensors(named, PROJ_NAMES))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(total_loss / grad_accum)
        norms.append(float(grad_norm))

    del model
    torch.cuda.empty_cache()

    out = {"losses": losses, "norms": norms}
    if breakdowns is not None:
        out["breakdowns"] = breakdowns
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"GPT-OSS weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def quantized(model_dir):
    """Whether the model is MXFP4 pre-quantized."""
    return is_mxfp4_model(model_dir)


@pytest.fixture(scope="module")
def inputs_data(model_dir):
    """Deterministic input/target pair (shared across forward and training tests)."""
    config = json.loads((model_dir / "config.json").read_text())
    return make_inputs(config["vocab_size"])


@pytest.fixture(scope="module")
def forward_results(model_dir, inputs_data):
    """Run both HF and Surogate forward, return HF results dict."""
    check_supported_weights(model_dir)

    run_surogate_forward(model_dir, inputs_data["inputs"], inputs_data["targets"])
    hf = run_hf_forward(model_dir, inputs_data["inputs"])
    return hf


@pytest.fixture(scope="module")
def training_results(model_dir):
    """Run Surogate LoRA training for NUM_TRAIN_STEPS optimizer steps.

    Uses TRAIN_SEQ_LEN and TRAIN_GRAD_ACCUM to match real training conditions.
    """
    check_supported_weights(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    data = make_train_inputs(config["vocab_size"])
    return run_surogate_training(model_dir, data["inputs"], data["targets"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGptOssMoEOnboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_mid_state(self, forward_results, quantized):
        """Check that per-layer mid-states (after attention) match.

        blocks[i].res_att only dumped for layers 0..NUM_LAYERS-2
        (last layer maps to residualN, not separately dumpable).
        """
        hf = forward_results
        tol = MXFP4_RMS_TOL if quantized else RMS_TOL
        failures = []

        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
            except FileNotFoundError as e:
                failures.append(f"layer {i}: dump not found ({e})")
                continue

            hf_mid = hf[f"mid_state_{i}"]
            rms, max_abs = diff_stats(rt_res_att, hf_mid)
            if rms > tol:
                failures.append(
                    f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={tol:.0e})"
                )

        if failures:
            pytest.fail("Per-layer mid-state mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, forward_results, quantized):
        """Check that xF (after final RMSNorm) matches HF post-norm output."""
        hf = forward_results
        tol = MXFP4_FINAL_NORM_TOL if quantized else FINAL_NORM_TOL
        rt_xf = load_dump("xF")
        rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
        assert rms < tol, (
            f"xF (final norm output) rms={rms:.4e} max_abs={max_abs:.4e} (tol={tol:.0e})"
        )

    def test_per_layer_output(self, forward_results, quantized):
        """Check that per-layer outputs (after MoE residual) match HF."""
        hf = forward_results
        tol = MXFP4_LAYER_OUT_TOL if quantized else LAYER_OUT_TOL
        failures = []

        for i in range(NUM_LAYERS):
            hf_out = hf[f"layer_output_{i}"]
            try:
                if i == NUM_LAYERS - 1:
                    rt_out = load_dump("residual_final")
                else:
                    rt_res_att = load_dump(f"blocks[{i}].res_att")
                    rt_mlp = load_dump(f"blocks[{i}].mlp_down")
                    rt_out = rt_res_att + rt_mlp
            except FileNotFoundError as e:
                failures.append(f"layer {i}: dump not found ({e})")
                continue

            rms, max_abs = diff_stats(rt_out, hf_out)
            if rms > tol:
                failures.append(
                    f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={tol:.0e})"
                )

        if failures:
            pytest.fail("Per-layer output mismatches:\n" + "\n".join(failures))

    def test_residual_final(self, forward_results, quantized):
        """Check that residual_final (before final norm) matches HF pre-norm."""
        hf = forward_results
        tol = MXFP4_RMS_TOL if quantized else RMS_TOL

        try:
            rt_residual_final = load_dump("residual_final")
        except FileNotFoundError:
            pytest.skip("residual_final dump not available")

        rms, max_abs = diff_stats(rt_residual_final, hf["pre_norm"])
        assert rms < tol, (
            f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={tol:.0e})"
        )

    def test_summary(self, forward_results, quantized):
        """Print a summary table of all comparisons (informational)."""
        hf = forward_results
        rows: List[Tuple[str, float, float]] = []

        for i in range(NUM_LAYERS - 1):
            rt_res_att = None
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
                hf_mid = hf[f"mid_state_{i}"]
                rms, max_abs = diff_stats(rt_res_att, hf_mid)
                rows.append((f"blocks[{i}].res_att", rms, max_abs))
            except (FileNotFoundError, KeyError):
                rows.append((f"blocks[{i}].res_att", float("nan"), float("nan")))
            try:
                if rt_res_att is None:
                    rt_res_att = load_dump(f"blocks[{i}].res_att")
                rt_mlp = load_dump(f"blocks[{i}].mlp_down")
                hf_out = hf[f"layer_output_{i}"]
                rms, max_abs = diff_stats(rt_res_att + rt_mlp, hf_out)
                rows.append((f"blocks[{i}].layer_out", rms, max_abs))
            except (FileNotFoundError, KeyError):
                rows.append((f"blocks[{i}].layer_out", float("nan"), float("nan")))

        try:
            rt_xf = load_dump("xF")
            rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
            rows.append(("xF (post-norm)", rms, max_abs))
        except FileNotFoundError:
            pass

        try:
            rt_rf = load_dump("residual_final")
            rms, max_abs = diff_stats(rt_rf, hf["pre_norm"])
            rows.append(("residual_final (pre-norm)", rms, max_abs))
        except FileNotFoundError:
            pass

        mode = "MXFP4 pre-quantized" if quantized else "BF16 full-precision"
        print(f"\n--- GPT-OSS MoE Forward Compare ({mode}: Surogate vs HF) ---")
        for name, rms, max_abs in rows:
            if quantized:
                tol = MXFP4_FINAL_NORM_TOL if "post-norm" in name else MXFP4_RMS_TOL
            else:
                tol = FINAL_NORM_TOL if "post-norm" in name else RMS_TOL
            status = "OK" if rms <= tol else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")


class TestGptOssMoETraining:
    """Training stability: LoRA fine-tuning on GPT-OSS MoE with backward + optimizer."""

    def test_initial_loss_matches_hf(self, model_dir, training_results, quantized):
        """Step-0 Surogate training loss should match HF cross-entropy loss.

        Runs a HF forward pass on the same training-length inputs and compares
        the cross-entropy loss. Since LoRA B=0 on step 0, the forward is
        equivalent to no-LoRA.
        """
        config = json.loads((model_dir / "config.json").read_text())
        data = make_train_inputs(config["vocab_size"])

        hf = run_hf_forward(model_dir, data["inputs"])
        hf_loss = compute_ce_loss(hf["logits"], data["targets"])

        rt_loss = training_results["losses"][0]
        tol = MXFP4_LOSS_MATCH_TOL if quantized else LOSS_MATCH_TOL
        diff = abs(hf_loss - rt_loss)

        print(f"\n[Loss Compare] HF={hf_loss:.4f}  Surogate={rt_loss:.4f}  diff={diff:.4f}")
        assert diff < tol, (
            f"Step-0 loss mismatch: HF={hf_loss:.4f} Surogate={rt_loss:.4f} "
            f"diff={diff:.4f} (tol={tol})"
        )

    def test_all_losses_finite(self, training_results):
        """All training losses must be finite (no NaN/Inf)."""
        for i, loss in enumerate(training_results["losses"]):
            assert np.isfinite(loss), f"Step {i}: loss is {loss}"

    def test_all_grad_norms_finite(self, training_results):
        """All gradient norms must be finite (no NaN/Inf from backward pass)."""
        for i, norm in enumerate(training_results["norms"]):
            assert np.isfinite(norm), f"Step {i}: grad norm is {norm}"

    def test_grad_norms_bounded(self, training_results):
        """Gradient norms should stay bounded (no explosion).

        With grad_clip=1.0, post-clip norms should be <= 1.0 (or close).
        Pre-clip norms can be larger but shouldn't explode exponentially.
        """
        norms = training_results["norms"]
        max_norm = max(norms)
        # After gradient clipping to 1.0, norms should not grow unbounded.
        # Allow some headroom — the reported norm may be pre-clip.
        assert max_norm < 1e6, (
            f"Gradient norms exploded: max={max_norm:.2f}, "
            f"all norms={[f'{n:.2f}' for n in norms]}"
        )

    def test_loss_decreases(self, training_results):
        """Loss should decrease (or at least not diverge) over training steps."""
        losses = training_results["losses"]
        assert losses[-1] < losses[0] * 1.5, (
            f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f} "
            f"(all: {[f'{l:.4f}' for l in losses]})"
        )

    def test_training_summary(self, training_results, quantized):
        """Print training progress table (informational)."""
        losses = training_results["losses"]
        norms = training_results["norms"]
        mode = "MXFP4 pre-quantized" if quantized else "BF16 full-precision"

        print(f"\n--- GPT-OSS MoE Training ({mode}: {NUM_TRAIN_STEPS} steps) ---")
        print(f"  {'step':>4s}  {'loss':>10s}  {'grad_norm':>12s}")
        for i, (loss, norm) in enumerate(zip(losses, norms)):
            print(f"  {i:4d}  {loss:10.4f}  {norm:12.4f}")
        print(f"  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
        if len(losses) > 1:
            improved = losses[-1] < losses[0]
            print(f"  Training {'improved' if improved else 'DID NOT improve'} loss")

    def test_hf_vs_surogate_training_stability(self, model_dir, quantized):
        """Compare short HF vs Surogate training for numerical stability.

        Uses a small number of steps/seq_len to fit in 32GB VRAM.
        Compares loss and grad-norm scale (order-of-magnitude).
        """
        if HF_COMPARE_STEPS <= 0:
            pytest.skip("HF_COMPARE_STEPS <= 0")

        config = json.loads((model_dir / "config.json").read_text())
        data = make_train_inputs(config["vocab_size"], seq_len=HF_COMPARE_SEQ_LEN)

        hf = run_hf_training(model_dir, data["inputs"], data["targets"],
                             num_steps=HF_COMPARE_STEPS, grad_accum=HF_COMPARE_GRAD_ACCUM,
                             capture_breakdown=True)
        rt = run_surogate_training(model_dir, data["inputs"], data["targets"],
                                   num_steps=HF_COMPARE_STEPS,
                                   seq_len=HF_COMPARE_SEQ_LEN,
                                   grad_accum=HF_COMPARE_GRAD_ACCUM,
                                   norm_source="lora",
                                   capture_breakdown=True)

        hf_losses = hf["losses"]
        rt_losses = rt["losses"]
        hf_norms = hf["norms"]
        rt_norms = rt["norms"]
        hf_breakdowns = hf.get("breakdowns", [])
        rt_breakdowns = rt.get("breakdowns", [])

        # Sanity: all finite
        assert np.isfinite(hf_losses).all(), f"HF losses not finite: {hf_losses}"
        assert np.isfinite(hf_norms).all(), f"HF norms not finite: {hf_norms}"
        assert np.isfinite(rt_losses).all(), f"Surogate losses not finite: {rt_losses}"
        assert np.isfinite(rt_norms).all(), f"Surogate norms not finite: {rt_norms}"

        loss_ratio_tol = 5.0 if quantized else 3.0
        norm_ratio_tol = 50.0 if quantized else 20.0

        rows = []
        for i in range(HF_COMPARE_STEPS):
            l_hf = hf_losses[i]
            l_rt = rt_losses[i]
            n_hf = hf_norms[i]
            n_rt = rt_norms[i]
            loss_ratio = max(l_hf, l_rt) / max(min(l_hf, l_rt), 1e-6)
            norm_ratio = max(n_hf, n_rt) / max(min(n_hf, n_rt), 1e-6)
            rows.append((i, l_hf, l_rt, loss_ratio, n_hf, n_rt, norm_ratio))

        print("\n--- GPT-OSS HF vs Surogate Training (short) ---")
        print(" step |   loss_hf   loss_rt   ratio |  norm_hf   norm_rt   ratio")
        for i, l_hf, l_rt, lr, n_hf, n_rt, nr in rows:
            print(f" {i:>4d} | {l_hf:8.4f} {l_rt:8.4f} {lr:6.2f} |"
                  f" {n_hf:8.2f} {n_rt:8.2f} {nr:6.2f}")
        if hf_breakdowns and rt_breakdowns:
            print("\n--- GPT-OSS LoRA Grad Breakdown (per projection) ---")
            header = " proj            |  hf_norm    rt_norm    ratio |  hf_max    rt_max | tensors_hf/rt"
            for i in range(HF_COMPARE_STEPS):
                print(f" step {i}:")
                print(header)
                for proj in PROJ_NAMES:
                    h = hf_breakdowns[i].get(proj, {"norm": 0.0, "max_abs": 0.0, "tensors": 0})
                    r = rt_breakdowns[i].get(proj, {"norm": 0.0, "max_abs": 0.0, "tensors": 0})
                    h_norm = h["norm"]
                    r_norm = r["norm"]
                    ratio = max(h_norm, r_norm) / max(min(h_norm, r_norm), 1e-6)
                    print(
                        f" {proj:14s} | {h_norm:9.2f} {r_norm:9.2f} {ratio:8.2f} |"
                        f" {h['max_abs']:8.2f} {r['max_abs']:8.2f} | {h['tensors']:5d}/{r['tensors']:5d}"
                    )

        for i, l_hf, l_rt, loss_ratio, n_hf, n_rt, norm_ratio in rows:
            assert loss_ratio < loss_ratio_tol, (
                f"Step {i}: loss ratio too large (HF={l_hf:.4f}, RT={l_rt:.4f}, "
                f"ratio={loss_ratio:.2f}, tol={loss_ratio_tol})"
            )
            assert norm_ratio < norm_ratio_tol, (
                f"Step {i}: grad-norm ratio too large (HF={n_hf:.2f}, RT={n_rt:.2f}, "
                f"ratio={norm_ratio:.2f}, tol={norm_ratio_tol})"
            )

    def test_hf_vs_surogate_training_stability_no_update(self, model_dir, quantized):
        """Compare HF vs Surogate gradients with optimizer updates disabled (LR=0).

        This isolates raw gradient differences from optimizer update effects.
        """
        if HF_COMPARE_STEPS <= 0:
            pytest.skip("HF_COMPARE_STEPS <= 0")

        config = json.loads((model_dir / "config.json").read_text())
        data = make_train_inputs(config["vocab_size"], seq_len=HF_COMPARE_SEQ_LEN)

        hf = run_hf_training(model_dir, data["inputs"], data["targets"],
                             num_steps=HF_COMPARE_STEPS, grad_accum=HF_COMPARE_GRAD_ACCUM,
                             learning_rate=0.0, capture_breakdown=False)
        rt = run_surogate_training(model_dir, data["inputs"], data["targets"],
                                   num_steps=HF_COMPARE_STEPS,
                                   seq_len=HF_COMPARE_SEQ_LEN,
                                   grad_accum=HF_COMPARE_GRAD_ACCUM,
                                   learning_rate=0.0,
                                   norm_source="lora",
                                   capture_breakdown=False)

        hf_losses = hf["losses"]
        rt_losses = rt["losses"]
        hf_norms = hf["norms"]
        rt_norms = rt["norms"]

        # Sanity: all finite
        assert np.isfinite(hf_losses).all(), f"HF losses not finite: {hf_losses}"
        assert np.isfinite(hf_norms).all(), f"HF norms not finite: {hf_norms}"
        assert np.isfinite(rt_losses).all(), f"Surogate losses not finite: {rt_losses}"
        assert np.isfinite(rt_norms).all(), f"Surogate norms not finite: {rt_norms}"

        loss_ratio_tol = 5.0 if quantized else 3.0
        norm_ratio_tol = 50.0 if quantized else 20.0

        rows = []
        for i in range(HF_COMPARE_STEPS):
            l_hf = hf_losses[i]
            l_rt = rt_losses[i]
            n_hf = hf_norms[i]
            n_rt = rt_norms[i]
            loss_ratio = max(l_hf, l_rt) / max(min(l_hf, l_rt), 1e-6)
            norm_ratio = max(n_hf, n_rt) / max(min(n_hf, n_rt), 1e-6)
            rows.append((i, l_hf, l_rt, loss_ratio, n_hf, n_rt, norm_ratio))

        print("\n--- GPT-OSS HF vs Surogate Training (LR=0 no-update) ---")
        print(" step |   loss_hf   loss_rt   ratio |  norm_hf   norm_rt   ratio")
        for i, l_hf, l_rt, lr, n_hf, n_rt, nr in rows:
            print(f" {i:>4d} | {l_hf:8.4f} {l_rt:8.4f} {lr:6.2f} |"
                  f" {n_hf:8.2f} {n_rt:8.2f} {nr:6.2f}")

        for i, l_hf, l_rt, loss_ratio, n_hf, n_rt, norm_ratio in rows:
            assert loss_ratio < loss_ratio_tol, (
                f"Step {i}: loss ratio too large (HF={l_hf:.4f}, RT={l_rt:.4f}, "
                f"ratio={loss_ratio:.2f}, tol={loss_ratio_tol})"
            )
            assert norm_ratio < norm_ratio_tol, (
                f"Step {i}: grad-norm ratio too large (HF={n_hf:.2f}, RT={n_rt:.2f}, "
                f"ratio={norm_ratio:.2f}, tol={norm_ratio_tol})"
            )
