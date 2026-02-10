"""Onboarding test: Nemotron-H hybrid model.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for a mini (truncated) Nemotron-H model. Compares per-layer hidden states and
final-norm outputs with tolerances.

Nemotron-H is a hybrid architecture with interleaved block types:
    M = Mamba2, * = Attention, - = MLP, E = MoE

Requirements:
    - GPU with enough VRAM for a small Nemotron-H model
    - HF weights: set NEMOTRON_H_MODEL_PATH env var or have the model cached
    - trust_remote_code=True (Nemotron-H uses custom HF code)

Usage:
    pytest tests/test_onboarding_nemotron_h.py -v --no-header
    NEMOTRON_H_MODEL_PATH=/path/to/model pytest tests/test_onboarding_nemotron_h.py -v
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

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ENV_VAR = "NEMOTRON_H_MODEL_PATH"
NUM_LAYERS = 6
SEED = 42
BATCH = 1
SEQ_LEN = 16
# Nemotron-H has MoE blocks; MoE routing introduces extra numerical variation,
# and RMSNorm amplifies small residual-stream differences.
RMS_TOL = 5e-2           # per-layer residual tolerance
FINAL_NORM_TOL = 2e-1    # post-norm tolerance (RMSNorm amplifies diffs)

MINI_MODEL_DIR = Path("tmp/onboarding_nemotron_h_mini")
DUMP_DIR = Path("tmp/onboarding_nemotron_h_dumps")

# Map NemotronH pattern characters to block type names
CHAR_TO_TYPE = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}


def residual_name_for_block_type(block_type: str) -> str:
    """Return the Surogate dump name suffix for the residual output.

    Attention blocks name their residual output 'res_att', while all other
    block types (Mamba, MLP, MoE) use 'res_in'.
    """
    if block_type == "attention":
        return "res_att"
    return "res_in"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path:
    """Resolve the path to Nemotron-H weights."""
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env)
        if p.exists():
            return p

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
    """Create a truncated Nemotron-H model with NUM_LAYERS layers.

    Nemotron-H specific: also truncates hybrid_override_pattern and copies
    custom modeling code for trust_remote_code.
    """
    if MINI_MODEL_DIR.exists():
        return MINI_MODEL_DIR
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS

    pattern = config.get("hybrid_override_pattern")
    if pattern is not None:
        config["hybrid_override_pattern"] = pattern[:NUM_LAYERS]

    (MINI_MODEL_DIR / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    # Copy custom modeling code (Nemotron-H requires trust_remote_code)
    for code_file in snapshot_dir.glob("*.py"):
        shutil.copy2(code_file, MINI_MODEL_DIR / code_file.name)

    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = snapshot_dir / tok_file
        if src.exists():
            shutil.copy2(src, MINI_MODEL_DIR / tok_file)

    # Filter weight map: Nemotron uses backbone.layers prefix
    index_path = snapshot_dir / "model.safetensors.index.json"
    single_path = snapshot_dir / "model.safetensors"

    if index_path.exists():
        base_index = json.loads(index_path.read_text())
        weight_map = base_index.get("weight_map", {})
        prefixes = [f"backbone.layers.{i}." for i in range(NUM_LAYERS)]
        extra = {"backbone.embeddings.weight", "backbone.norm_f.weight", "lm_head.weight"}

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


# ---------------------------------------------------------------------------
# HuggingFace forward — uses hooks for reliable per-layer capture
# ---------------------------------------------------------------------------

def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> Dict[str, np.ndarray]:
    """Run HF forward and capture per-layer block inputs via hooks.

    Captures per-layer block inputs (= residual stream) via pre-hooks on
    each backbone.layers[i]. These correspond to Surogate's blocks[i].res_in
    (or blocks[i].res_att for attention blocks).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    result: Dict[str, np.ndarray] = {}
    layer_inputs: Dict[int, torch.Tensor] = {}
    layer_outputs: Dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(NUM_LAYERS):
        # Pre-hook: captures block input (= residual stream at this layer)
        def make_pre_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                layer_inputs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(model.backbone.layers[i].register_forward_pre_hook(make_pre_hook(i)))

        # Post-hook: captures block output
        def make_post_hook(idx):
            def hook_fn(module, args, output):
                hs = output if isinstance(output, torch.Tensor) else output[0]
                layer_outputs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(model.backbone.layers[i].register_forward_hook(make_post_hook(i)))

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

        for i in range(NUM_LAYERS):
            result[f"layer_input_{i}"] = layer_inputs[i].float().cpu().numpy()
            result[f"layer_output_{i}"] = layer_outputs[i].float().cpu().numpy()

        # Pre-norm = output of last layer (before final RMSNorm)
        pre_norm = layer_outputs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        # Post-norm = apply final norm explicitly
        post_norm = model.backbone.norm_f(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        # Logits
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
                         targets: np.ndarray, block_types: List[str]) -> None:
    """Run Surogate forward pass and dump tensors to DUMP_DIR."""
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        res_name = residual_name_for_block_type(block_types[i])
        dump_tensors.append(f"blocks[{i}].{res_name}")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR)
    os.environ["SUROGATE_DEBUG_DUMP_LAYER"] = "-1"

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
    trainer.step(inputs, targets)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Nemotron-H weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def block_types(model_dir):
    """Parse the hybrid_override_pattern to get block types for the mini model."""
    config = json.loads((model_dir / "config.json").read_text())
    pattern = config.get("hybrid_override_pattern", "")
    return [CHAR_TO_TYPE.get(c, "unknown") for c in pattern[:NUM_LAYERS]]


@pytest.fixture(scope="module")
def forward_results(model_dir, block_types):
    """Run both HF and Surogate forward, return HF results dict."""
    config = json.loads((model_dir / "config.json").read_text())
    vocab_size = config["vocab_size"]
    data = make_inputs(vocab_size)

    run_surogate_forward(model_dir, data["inputs"], data["targets"], block_types)
    hf = run_hf_forward(model_dir, data["inputs"])
    return hf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNemotronHOnboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_residual(self, forward_results, block_types):
        """Check that per-layer residual inputs (block inputs) match.

        Surogate: blocks[i].res_in (Mamba/MLP/MoE) or blocks[i].res_att (Attention)
        HF: input to backbone.layers[i] (captured via pre-hook)

        Note: last layer's residual maps to residualN (not dumpable separately),
        so we only check layers 0..NUM_LAYERS-2 here.
        """
        hf = forward_results
        failures = []

        for i in range(NUM_LAYERS - 1):
            res_name = residual_name_for_block_type(block_types[i])
            dump_name = f"blocks[{i}].{res_name}"
            try:
                rt_residual = load_dump(dump_name)
            except FileNotFoundError as e:
                failures.append(f"layer {i} ({block_types[i]}): dump {dump_name} not found ({e})")
                continue

            hf_input = hf[f"layer_input_{i}"]
            rms, max_abs = diff_stats(rt_residual, hf_input)
            if rms > RMS_TOL:
                failures.append(
                    f"layer {i} ({block_types[i]}): rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
                )

        if failures:
            pytest.fail("Per-layer residual mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, forward_results):
        """Check that xF (after final RMSNorm) matches HF post-norm output.

        Uses relaxed tolerance because RMSNorm amplifies small residual-stream
        differences, and MoE routing introduces extra numerical variation.
        """
        hf = forward_results
        rt_xf = load_dump("xF")
        rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
        assert rms < FINAL_NORM_TOL, (
            f"xF (final norm output) rms={rms:.4e} max_abs={max_abs:.4e} (tol={FINAL_NORM_TOL:.0e})"
        )

    def test_residual_final(self, forward_results):
        """Check that residual_final (before final norm) matches HF pre-norm."""
        hf = forward_results

        try:
            rt_residual_final = load_dump("residual_final")
        except FileNotFoundError:
            pytest.skip("residual_final dump not available")

        rms, max_abs = diff_stats(rt_residual_final, hf["pre_norm"])
        assert rms < RMS_TOL, (
            f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
        )

    def test_summary(self, forward_results, block_types, model_dir):
        """Print a summary table of all comparisons (informational)."""
        hf = forward_results
        config = json.loads((model_dir / "config.json").read_text())
        pattern = config.get("hybrid_override_pattern", "")

        rows: List[Tuple[str, str, float, float]] = []

        for i in range(NUM_LAYERS - 1):
            btype = block_types[i]
            res_name = residual_name_for_block_type(btype)
            dump_name = f"blocks[{i}].{res_name}"
            try:
                rt_residual = load_dump(dump_name)
                hf_input = hf[f"layer_input_{i}"]
                rms, max_abs = diff_stats(rt_residual, hf_input)
                rows.append((dump_name, btype, rms, max_abs))
            except (FileNotFoundError, KeyError):
                rows.append((dump_name, btype, float("nan"), float("nan")))

        try:
            rt_xf = load_dump("xF")
            rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
            rows.append(("xF (post-norm)", "-", rms, max_abs))
        except (FileNotFoundError, KeyError):
            pass

        try:
            rt_rf = load_dump("residual_final")
            rms, max_abs = diff_stats(rt_rf, hf["pre_norm"])
            rows.append(("residual_final (pre-norm)", "-", rms, max_abs))
        except (FileNotFoundError, KeyError):
            pass

        print(f"\n--- Nemotron-H Forward Compare (pattern={pattern[:NUM_LAYERS]}) ---")
        for name, btype, rms, max_abs in rows:
            tol = FINAL_NORM_TOL if "post-norm" in name else RMS_TOL
            status = "OK" if rms <= tol else "FAIL"
            print(f"  {name:30s} [{btype:9s}] rms={rms:.4e}  max={max_abs:.4e}  [{status}]")
