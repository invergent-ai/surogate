"""Onboarding test: Qwen3 dense model.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for a mini (truncated) Qwen3 model. Compares per-layer hidden states and
final-norm outputs with tolerances.

Requirements:
    - GPU with enough VRAM for a small Qwen3 model
    - HF weights: set QWEN3_MODEL_PATH env var or have Qwen/Qwen3-0.6B cached

Usage:
    pytest tests/test_onboarding_qwen3.py -v --no-header
    QWEN3_MODEL_PATH=/path/to/Qwen3-0.6B pytest tests/test_onboarding_qwen3.py -v
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

MODEL_ID = "Qwen/Qwen3-0.6B"
ENV_VAR = "QWEN3_MODEL_PATH"
NUM_LAYERS = 4
SEED = 42
BATCH = 1
SEQ_LEN = 16
# bf16 forward through multiple layers accumulates error; 5e-2 is a reasonable
# threshold for implementation-level differences (fused kernels, cuDNN attention).
RMS_TOL = 5e-2

MINI_MODEL_DIR = Path("tmp/onboarding_qwen3_mini")
DUMP_DIR = Path("tmp/onboarding_qwen3_dumps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path:
    """Resolve the path to Qwen3 weights."""
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
    """Create a truncated Qwen3 model with NUM_LAYERS layers."""
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
    """Run HF forward and capture per-layer outputs via hooks.

    Captures two types of hidden states:
    - Full layer outputs (post-hook on each decoder layer)
    - Mid-layer states after attention (pre-hook on post_attention_layernorm)

    The mid-layer states correspond to Surogate's blocks[i].res_att (the
    hidden state after the first residual add + attention, before MLP).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    result: Dict[str, np.ndarray] = {}
    layer_outs: Dict[int, torch.Tensor] = {}
    mid_states: Dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(NUM_LAYERS):
        # Post-hook on full layer: captures layer output (after both residual adds)
        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(model.model.layers[i].register_forward_hook(make_layer_hook(i)))

        # Pre-hook on post_attention_layernorm: captures mid-layer state
        # = hidden_states after first residual add (attention output + input residual)
        # This corresponds to Surogate's blocks[i].res_att
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
        _ = model(input_ids=input_ids, use_cache=False)

        # Per-layer full outputs
        for i in range(NUM_LAYERS):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()

        # Per-layer mid-states (= blocks[i].res_att in Surogate)
        for i in range(NUM_LAYERS):
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        # Pre-norm = last layer output (before final RMSNorm)
        pre_norm = layer_outs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        # Post-norm = explicitly apply final norm
        post_norm = model.model.norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        # Logits for reference
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

    Dumpable tensors for Qwen3:
    - blocks[i].res_att: mid-layer hidden state (after attention, before MLP)
      Available for layers 0..NUM_LAYERS-2. Last layer maps to residualN.
    - residual_final: full hidden state before final norm (= residualN + xN)
    - xF: after final RMSNorm

    NOT dumpable (shared/recomputed buffers):
    - layer{i}.mlp_down: MLP output (share_policy="when_recomputed")
    - xN, residualN: consumed by fused_residual_rmsnorm
    """
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_att")

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
        pytest.skip(f"Qwen3 weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def forward_results(model_dir):
    """Run both HF and Surogate forward, return HF results dict."""
    config = json.loads((model_dir / "config.json").read_text())
    vocab_size = config["vocab_size"]
    data = make_inputs(vocab_size)

    run_surogate_forward(model_dir, data["inputs"], data["targets"])
    hf = run_hf_forward(model_dir, data["inputs"])
    return hf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQwen3Onboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_mid_state(self, forward_results):
        """Check that per-layer mid-states (after attention) match.

        Surogate: blocks[i].res_att = hidden_state + attn_out (before MLP)
        HF: input to post_attention_layernorm (captured via pre-hook)

        Note: last layer's res_att maps to residualN (not dumpable separately),
        so we only check layers 0..NUM_LAYERS-2 here.
        """
        hf = forward_results
        failures = []

        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
            except FileNotFoundError as e:
                failures.append(f"layer {i}: dump not found ({e})")
                continue

            hf_mid = hf[f"mid_state_{i}"]
            rms, max_abs = diff_stats(rt_res_att, hf_mid)
            if rms > RMS_TOL:
                failures.append(
                    f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
                )

        if failures:
            pytest.fail("Per-layer mid-state mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, forward_results):
        """Check that xF (after final RMSNorm) matches HF post-norm output."""
        hf = forward_results
        rt_xf = load_dump("xF")
        rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
        assert rms < RMS_TOL, (
            f"xF (final norm output) rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
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

    def test_summary(self, forward_results):
        """Print a summary table of all comparisons (informational)."""
        hf = forward_results
        rows: List[Tuple[str, float, float]] = []

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

        print("\n--- Qwen3 Forward Compare (Surogate vs HF) ---")
        for name, rms, max_abs in rows:
            status = "OK" if rms <= RMS_TOL else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")
