"""Onboarding test: Qwen3.5 dense model (conditional architecture).

Validates that the Surogate DSL forward+backward pass matches HuggingFace
reference for a mini (truncated) Qwen3.5 model.

Forward checks:
- Per-layer mid-state (input to post_attention_layernorm)
- Final residual stream before norm
- Final norm output

Backward checks:
- Selected parameter gradients from one training step (Surogate vs HF)

Requirements:
    - GPU
    - HF weights: set QWEN3_5_MODEL_PATH env var or cache Qwen/Qwen3.5-0.8B

Usage:
    pytest tests/test_onboarding_qwen3_5.py -v --no-header
    QWEN3_5_MODEL_PATH=/path/to/Qwen3.5-0.8B pytest tests/test_onboarding_qwen3_5.py -v
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

MODEL_ID = "Qwen/Qwen3.5-0.8B"
ENV_VAR = "QWEN3_5_MODEL_PATH"
ARCH = "Qwen3_5ForConditionalGeneration"
NUM_LAYERS = 4
MINI_VOCAB_SIZE = 8192
SEED = 42
BATCH = 1
SEQ_LEN = 16
RMS_TOL = 5e-2
POST_NORM_REL_RMS_TOL = 7e-2

# Backward parity tolerances are on sampled elements.
GRAD_REL_RMS_TOL = 1e-1
GRAD_RMS_TOL = 5e-2
GRAD_SAMPLE_SIZE = 131072

MINI_MODEL_DIR = Path("tmp/onboarding_qwen3_5_mini")
DUMP_DIR = Path("tmp/onboarding_qwen3_5_dumps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_path() -> Path | None:
    """Resolve the path to Qwen3.5 weights."""
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.exists():
            return p

    cache_root = Path("~/.cache/huggingface/hub").expanduser()

    # Preferred default model id.
    model_slug = MODEL_ID.replace("/", "--")
    model_cache = cache_root / f"models--{model_slug}"
    if model_cache.exists():
        snaps = model_cache / "snapshots"
        if snaps.exists():
            for snap in sorted(snaps.iterdir(), reverse=True):
                cfg = snap / "config.json"
                if cfg.exists():
                    try:
                        doc = json.loads(cfg.read_text())
                        if ARCH in doc.get("architectures", []):
                            return snap
                    except Exception:
                        pass

    # Fallback scan for any cached snapshot with matching architecture.
    for model_dir in sorted(cache_root.glob("models--*")):
        snaps = model_dir / "snapshots"
        if not snaps.exists():
            continue
        for snap in sorted(snaps.iterdir(), reverse=True):
            cfg = snap / "config.json"
            if not cfg.exists():
                continue
            try:
                doc = json.loads(cfg.read_text())
                if ARCH in doc.get("architectures", []):
                    return snap
            except Exception:
                continue
    return None


def prepare_mini_model(snapshot_dir: Path) -> Path:
    """Create a truncated Qwen3.5 model with NUM_LAYERS text layers."""
    if MINI_MODEL_DIR.exists():
        cfg_path = MINI_MODEL_DIR / "config.json"
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
                text_cfg = existing.get("text_config", {})
                if (
                    text_cfg.get("num_hidden_layers") == NUM_LAYERS
                    and text_cfg.get("vocab_size") == MINI_VOCAB_SIZE
                    and text_cfg.get("max_position_embeddings") == max(256, SEQ_LEN * 8)
                    and existing.get("tie_word_embeddings") is False
                    and text_cfg.get("tie_word_embeddings") is False
                ):
                    return MINI_MODEL_DIR
            except Exception:
                pass
        shutil.rmtree(MINI_MODEL_DIR)
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    text_cfg = config.get("text_config", {})
    text_cfg["num_hidden_layers"] = NUM_LAYERS
    text_cfg["vocab_size"] = MINI_VOCAB_SIZE
    text_cfg["max_position_embeddings"] = max(256, SEQ_LEN * 8)
    # Match HF checkpoint behavior for Qwen3.5-0.8B: keep embedding and lm_head untied
    # when both tensors are present in weights.
    text_cfg["tie_word_embeddings"] = False
    if isinstance(text_cfg.get("layer_types"), list):
        text_cfg["layer_types"] = text_cfg["layer_types"][:NUM_LAYERS]
    config["text_config"] = text_cfg
    config["vocab_size"] = MINI_VOCAB_SIZE
    config["tie_word_embeddings"] = False

    # Keep vision tiny (unused in this text-only parity test) to reduce memory.
    vis_cfg = config.get("vision_config")
    if isinstance(vis_cfg, dict):
        vis_cfg["depth"] = 1
        vis_cfg["hidden_size"] = 64
        vis_cfg["intermediate_size"] = 128
        vis_cfg["num_heads"] = 4
        vis_cfg["out_hidden_size"] = text_cfg.get("hidden_size", vis_cfg.get("out_hidden_size", 3584))
        vis_cfg["num_position_embeddings"] = 256
        config["vision_config"] = vis_cfg

    (MINI_MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    for tok_file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ]:
        src = snapshot_dir / tok_file
        if src.exists():
            shutil.copy2(src, MINI_MODEL_DIR / tok_file)

    index_path = snapshot_dir / "model.safetensors.index.json"
    single_path = snapshot_dir / "model.safetensors"

    def _write_mini_vocab_shard(weight_map: dict[str, str]) -> str:
        from safetensors import safe_open
        from safetensors.torch import save_file

        out_name = "model_vocab.safetensors"
        out_path = MINI_MODEL_DIR / out_name
        if out_path.exists():
            return out_name

        emb_key = "model.language_model.embed_tokens.weight"
        lm_key = "lm_head.weight"

        if emb_key not in weight_map:
            raise KeyError(f"Missing required embedding tensor '{emb_key}' in weight map")

        emb_file = snapshot_dir / weight_map[emb_key]

        with safe_open(str(emb_file), framework="pt", device="cpu") as f:
            emb = f.get_tensor(emb_key)

        # Some tied-embedding checkpoints omit lm_head.weight.
        # Synthesize lm_head from embedding when absent.
        if lm_key in weight_map:
            lm_file = snapshot_dir / weight_map[lm_key]
            with safe_open(str(lm_file), framework="pt", device="cpu") as f:
                lm = f.get_tensor(lm_key)
        else:
            lm = emb.clone()

        mini_tensors = {
            emb_key: emb[:MINI_VOCAB_SIZE].contiguous(),
            lm_key: lm[:MINI_VOCAB_SIZE].contiguous(),
        }
        save_file(mini_tensors, str(out_path))
        return out_name

    if index_path.exists():
        base_index = json.loads(index_path.read_text())
        weight_map = base_index.get("weight_map", {})
        prefixes = [f"model.language_model.layers.{i}." for i in range(NUM_LAYERS)]
        extra = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        }

        def want(name: str) -> bool:
            if name in extra:
                return True
            return any(name.startswith(p) for p in prefixes)

        mini_map = {k: v for k, v in weight_map.items() if want(k)}
        if "model.language_model.embed_tokens.weight" in mini_map:
            vocab_shard = _write_mini_vocab_shard(weight_map)
            mini_map["model.language_model.embed_tokens.weight"] = vocab_shard
            mini_map["lm_head.weight"] = vocab_shard

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


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    rms = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    return rms, max_abs


def make_inputs(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def sample_flat(t: torch.Tensor, max_elems: int = GRAD_SAMPLE_SIZE) -> torch.Tensor:
    flat = t.float().reshape(-1)
    n = int(flat.numel())
    if n <= max_elems:
        return flat
    step = max(1, n // max_elems)
    return flat[::step][:max_elems]


def grad_diff_stats(rt: torch.Tensor, hf: torch.Tensor) -> tuple[float, float, float]:
    rt_s = sample_flat(rt)
    hf_s = sample_flat(hf)
    diff = rt_s - hf_s
    rms = float(torch.sqrt(torch.mean(diff * diff)).item())
    max_abs = float(torch.max(torch.abs(diff)).item())
    ref_rms = float(torch.sqrt(torch.mean(hf_s * hf_s)).item())
    rel_rms = rms / max(ref_rms, 1e-12)
    return rel_rms, rms, max_abs


def build_grad_mapping(layer_types: list[str]) -> dict[str, str | tuple[str, str, str]]:
    """Map Surogate param names to HF param names for selected tensors."""
    mapping: dict[str, str | tuple[str, str, str]] = {}

    linear_idx = next((i for i, t in enumerate(layer_types) if t == "linear_attention"), None)
    attn_idx = next((i for i, t in enumerate(layer_types) if t == "full_attention"), None)

    if linear_idx is not None:
        p = f"model.language_model.layers.{linear_idx}"
        mapping[f"blocks[{linear_idx}].ln1_weight"] = f"{p}.input_layernorm.weight"
        mapping[f"blocks[{linear_idx}].mlp_up_weight"] = (
            "fuse",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.gate_proj.weight",
        )
        mapping[f"blocks[{linear_idx}].mlp_down_weight"] = f"{p}.mlp.down_proj.weight"
        mapping[f"blocks[{linear_idx}].lin_in_proj_qkv_weight"] = f"{p}.linear_attn.in_proj_qkv.weight"
        mapping[f"blocks[{linear_idx}].lin_A_log"] = f"{p}.linear_attn.A_log"
        mapping[f"blocks[{linear_idx}].lin_dt_bias"] = f"{p}.linear_attn.dt_bias"
        mapping[f"blocks[{linear_idx}].lin_norm_weight"] = f"{p}.linear_attn.norm.weight"

    if attn_idx is not None:
        p = f"model.language_model.layers.{attn_idx}"
        mapping[f"blocks[{attn_idx}].ln1_weight"] = f"{p}.input_layernorm.weight"
        mapping[f"blocks[{attn_idx}].mlp_up_weight"] = (
            "fuse",
            f"{p}.mlp.up_proj.weight",
            f"{p}.mlp.gate_proj.weight",
        )
        mapping[f"blocks[{attn_idx}].mlp_down_weight"] = f"{p}.mlp.down_proj.weight"
        mapping[f"blocks[{attn_idx}].full_q_proj_weight"] = f"{p}.self_attn.q_proj.weight"
        mapping[f"blocks[{attn_idx}].full_out_weight"] = f"{p}.self_attn.o_proj.weight"
        mapping[f"blocks[{attn_idx}].q_norm_weight"] = f"{p}.self_attn.q_norm.weight"
        mapping[f"blocks[{attn_idx}].k_norm_weight"] = f"{p}.self_attn.k_norm.weight"

    mapping["final_norm"] = "model.language_model.norm.weight"
    return mapping


# ---------------------------------------------------------------------------
# HuggingFace forward/backward
# ---------------------------------------------------------------------------


def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Run HF forward and capture per-layer outputs via hooks."""
    from transformers import Qwen3_5ForConditionalGeneration

    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pytest.skip(f"HF forward skipped due to CUDA OOM: {e}")
        raise
    model.eval()

    result: dict[str, np.ndarray] = {}
    layer_outs: dict[int, torch.Tensor] = {}
    mid_states: dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(NUM_LAYERS):

        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(model.model.language_model.layers[i].register_forward_hook(make_layer_hook(i)))

        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(
            model.model.language_model.layers[i].post_attention_layernorm.register_forward_pre_hook(make_mid_hook(i))
        )

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

        for i in range(NUM_LAYERS):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        pre_norm = layer_outs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()
        post_norm = model.model.language_model.norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()
        logits = model.lm_head(post_norm)
        result["logits"] = logits.float().cpu().numpy()

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result


def run_hf_backward(
    model_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    grad_mapping: dict[str, str | tuple[str, str, str]],
) -> dict[str, torch.Tensor]:
    """Run HF backward once and collect selected gradients."""
    import torch.nn.functional as F
    from transformers import Qwen3_5ForConditionalGeneration

    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pytest.skip(f"HF backward skipped due to CUDA OOM: {e}")
        raise
    model.eval()
    model.zero_grad(set_to_none=True)

    input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
    labels = torch.tensor(targets, device="cuda", dtype=torch.long)

    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss.backward()

    named_params = dict(model.named_parameters())
    grads: dict[str, torch.Tensor] = {}

    for rt_name, hf_spec in grad_mapping.items():
        if isinstance(hf_spec, tuple):
            _, up_name, gate_name = hf_spec
            up = named_params.get(up_name)
            gate = named_params.get(gate_name)
            if up is None or gate is None or up.grad is None or gate.grad is None:
                continue
            grads[rt_name] = torch.cat([up.grad, gate.grad], dim=0).detach().clone()
        else:
            p = named_params.get(hf_spec)
            if p is None or p.grad is None:
                continue
            grads[rt_name] = p.grad.detach().clone()

    del model
    torch.cuda.empty_cache()
    return grads


# ---------------------------------------------------------------------------
# Surogate step (forward + backward)
# ---------------------------------------------------------------------------


def run_surogate_step(
    model_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    grad_mapping: dict[str, str | tuple[str, str, str]],
) -> dict[str, torch.Tensor]:
    """Run one Surogate training step, dump forward tensors, return selected grads."""
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_att")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR)
    os.environ["SUROGATE_DEBUG_DUMP_LAYER"] = "-1"
    stack_override_set = False
    if "SUROGATE_MIN_STACK_MB" not in os.environ:
        os.environ["SUROGATE_MIN_STACK_MB"] = "512"
        stack_override_set = True

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
    try:
        trainer.import_weights(get_model_weights_path(str(model_dir)))
        trainer.step(inputs, targets)
    except RuntimeError as e:
        if "oom" in str(e).lower() or "out of memory" in str(e).lower():
            pytest.skip(f"Surogate backward skipped due to CUDA OOM: {e}")
        raise

    raw = trainer.get_gradients(0)
    out: dict[str, torch.Tensor] = {}
    for rt_name in grad_mapping:
        if rt_name not in raw:
            continue
        out[rt_name] = torch.utils.dlpack.from_dlpack(raw[rt_name]).detach().clone()

    for key in ["SUROGATE_DEBUG_DUMP_TENSORS", "SUROGATE_DEBUG_DUMP_DIR", "SUROGATE_DEBUG_DUMP_LAYER"]:
        os.environ.pop(key, None)
    if stack_override_set:
        os.environ.pop("SUROGATE_MIN_STACK_MB", None)

    del trainer
    torch.cuda.empty_cache()
    return out


def run_surogate_forward(model_dir: Path, inputs: np.ndarray, targets: np.ndarray) -> None:
    """Run one Surogate forward-only pass and dump tensors."""
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_att")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR)
    os.environ["SUROGATE_DEBUG_DUMP_LAYER"] = "-1"
    stack_override_set = False
    if "SUROGATE_MIN_STACK_MB" not in os.environ:
        os.environ["SUROGATE_MIN_STACK_MB"] = "512"
        stack_override_set = True

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
    try:
        trainer.import_weights(get_model_weights_path(str(model_dir)))
        trainer.validate(inputs, targets)
    except RuntimeError as e:
        if "oom" in str(e).lower() or "out of memory" in str(e).lower():
            pytest.skip(f"Surogate forward skipped due to CUDA OOM: {e}")
        raise
    finally:
        for key in [
            "SUROGATE_DEBUG_DUMP_TENSORS",
            "SUROGATE_DEBUG_DUMP_DIR",
            "SUROGATE_DEBUG_DUMP_LAYER",
        ]:
            os.environ.pop(key, None)
        if stack_override_set:
            os.environ.pop("SUROGATE_MIN_STACK_MB", None)

    del trainer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Qwen3.5 weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def layer_types(model_dir) -> list[str]:
    config = json.loads((model_dir / "config.json").read_text())
    return config.get("text_config", {}).get("layer_types", [])[:NUM_LAYERS]


@pytest.fixture(scope="module")
def inputs_data(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    vocab_size = config["text_config"]["vocab_size"]
    return make_inputs(vocab_size)


@pytest.fixture(scope="module")
def grad_mapping(layer_types):
    mapping = build_grad_mapping(layer_types)
    if not mapping:
        pytest.skip("No gradient mapping available for this layer_types pattern")
    return mapping


@pytest.fixture(scope="module")
def rt_results(model_dir, inputs_data, grad_mapping):
    return run_surogate_step(model_dir, inputs_data["inputs"], inputs_data["targets"], grad_mapping)


@pytest.fixture(scope="module")
def rt_forward_done(model_dir, inputs_data):
    run_surogate_forward(model_dir, inputs_data["inputs"], inputs_data["targets"])
    return True


@pytest.fixture(scope="module")
def hf_forward_results(model_dir, inputs_data, rt_forward_done):
    del rt_forward_done
    return run_hf_forward(model_dir, inputs_data["inputs"])


@pytest.fixture(scope="module")
def hf_backward_grads(model_dir, inputs_data, grad_mapping):
    return run_hf_backward(model_dir, inputs_data["inputs"], inputs_data["targets"], grad_mapping)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQwen35OnboardingForward:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_mid_state(self, hf_forward_results):
        hf = hf_forward_results
        failures = []

        # Last layer residual stream is represented by residual_final.
        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
            except FileNotFoundError as e:
                failures.append(f"layer {i}: dump not found ({e})")
                continue

            hf_mid = hf[f"mid_state_{i}"]
            rms, max_abs = diff_stats(rt_res_att, hf_mid)
            if rms > RMS_TOL:
                failures.append(f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})")

        if failures:
            pytest.fail("Per-layer mid-state mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, hf_forward_results):
        rt_xf = load_dump("xF")
        hf_post = hf_forward_results["post_norm"]
        rms, max_abs = diff_stats(rt_xf, hf_post)
        ref_rms = float(np.sqrt(np.mean(hf_post.astype(np.float32) * hf_post.astype(np.float32))))
        rel_rms = rms / max(ref_rms, 1e-12)
        assert (rms < RMS_TOL) or (rel_rms < POST_NORM_REL_RMS_TOL), (
            f"xF (final norm output) rms={rms:.4e} rel_rms={rel_rms:.4e} "
            f"max_abs={max_abs:.4e} (tol abs={RMS_TOL:.0e} or rel={POST_NORM_REL_RMS_TOL:.0e})"
        )

    def test_residual_final(self, hf_forward_results):
        try:
            rt_residual_final = load_dump("residual_final")
        except FileNotFoundError:
            pytest.skip("residual_final dump not available")

        rms, max_abs = diff_stats(rt_residual_final, hf_forward_results["pre_norm"])
        assert rms < RMS_TOL, f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"

    def test_summary(self, hf_forward_results):
        rows: list[tuple[str, float, float]] = []
        hf = hf_forward_results

        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
                hf_mid = hf[f"mid_state_{i}"]
                rms, max_abs = diff_stats(rt_res_att, hf_mid)
                rows.append((f"blocks[{i}].res_att", rms, max_abs))
            except (FileNotFoundError, KeyError):
                rows.append((f"blocks[{i}].res_att", float("nan"), float("nan")))

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

        print("\n--- Qwen3.5 Forward Compare (Surogate vs HF) ---")
        for name, rms, max_abs in rows:
            status = "OK" if rms <= RMS_TOL else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")


class TestQwen35OnboardingBackward:
    """Selected gradient comparison: Surogate vs HuggingFace."""

    def test_selected_gradients(self, rt_results, hf_backward_grads, grad_mapping):
        failures = []
        rows: list[tuple[str, float, float, float]] = []

        for rt_name in sorted(grad_mapping.keys()):
            rt = rt_results.get(rt_name)
            hf = hf_backward_grads.get(rt_name)
            if rt is None or hf is None:
                failures.append(f"{rt_name}: missing gradient (rt={rt is not None}, hf={hf is not None})")
                continue

            rel_rms, rms, max_abs = grad_diff_stats(rt, hf)
            rows.append((rt_name, rel_rms, rms, max_abs))
            if rel_rms > GRAD_REL_RMS_TOL and rms > GRAD_RMS_TOL:
                failures.append(
                    f"{rt_name}: rel_rms={rel_rms:.4e} rms={rms:.4e} max_abs={max_abs:.4e} "
                    f"(tol rel={GRAD_REL_RMS_TOL:.0e}, abs={GRAD_RMS_TOL:.0e})"
                )

        print("\n--- Qwen3.5 Backward Compare (sampled grads) ---")
        for name, rel_rms, rms, max_abs in rows:
            status = "OK" if (rel_rms <= GRAD_REL_RMS_TOL or rms <= GRAD_RMS_TOL) else "FAIL"
            print(f"  {name:40s} rel_rms={rel_rms:.4e}  rms={rms:.4e}  max={max_abs:.4e}  [{status}]")

        if failures:
            pytest.fail("Gradient mismatches:\n" + "\n".join(failures))
