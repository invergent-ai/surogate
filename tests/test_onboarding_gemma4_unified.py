"""Onboarding test: Gemma4 Unified (gemma4_unified) model.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for the google/gemma-4-12B model (Gemma4UnifiedForConditionalGeneration).
Compares per-layer hidden states and final-norm outputs with tolerances.

What this exercises beyond test_onboarding_gemma4.py (E2B-it):
  - attention_k_eq_v=true: full-attention layers ship q_proj+k_proj only,
    V reuses the raw K projection (V-norm still applied)
  - num_global_key_value_heads (1) != num_key_value_heads (8): per-block-type
    KV head counts, not just per-block-type head sizes
  - global_head_dim 512 vs sliding head_dim 256
  - no per-layer input embeddings (hidden_size_per_layer_input=0)

Requirements:
    - GPU with enough VRAM for gemma-4-12B (~24GB bf16; HF side runs on CPU)
    - HF weights: set GEMMA4_UNIFIED_MODEL_PATH env var or have
      google/gemma-4-12B cached

Usage:
    pytest tests/test_onboarding_gemma4_unified.py -v --no-header
    GEMMA4_UNIFIED_MODEL_PATH=/path/to/gemma-4-12B pytest tests/test_onboarding_gemma4_unified.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.utils.hf import get_model_weights_path

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "google/gemma-4-12B"
ENV_VAR = "GEMMA4_UNIFIED_MODEL_PATH"
SEED = 42
BATCH = 1
SEQ_LEN = 16
# Use cosine similarity (direction) as primary metric — robust to signal
# magnitude differences across layers.
#
# Tolerances are calibrated against the model's INHERENT bf16 drift: HF's
# own bf16 forward deviates from its fp32 forward by cos ≈ 0.987 (layer 30),
# 0.979 (layer 41), 0.975 (xF) on this checkpoint — round-off compounds
# through 48 layers with large late-layer activations. Surogate-bf16 vs
# HF-bf16 measures at or slightly tighter than that envelope, so demanding
# 0.999 at depth is unachievable in bf16. Early layers haven't accumulated
# drift yet and stay strict.
COS_TOL_EARLY = 0.999  # layers < EARLY_LAYERS
COS_TOL_LATE = 0.975  # deep layers + final norm (inherent drift floor)
EARLY_LAYERS = 16
# A correctness bug shows up as a sharp per-layer CLIFF on top of the smooth
# drift (e.g. the layer-41 LoRA OOB-read bug measured cos 0.918 against
# ~0.99 neighbors). Flag any layer dropping more than this below the median
# of its neighbors.
CLIFF_TOL = 0.02
CLIFF_WINDOW = 3  # neighbors on each side

DUMP_DIR = Path("tmp/onboarding_gemma4_unified_dumps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_path() -> Path:
    """Resolve the path to Gemma4 Unified weights."""
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


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a_f = a.astype(np.float32).ravel()
    b_f = b.astype(np.float32).ravel()
    diff = a_f - b_f
    rms = float(np.sqrt(np.mean(diff * diff)))
    signal_rms = float(np.sqrt(np.mean(b_f * b_f)))
    rel_rms = rms / signal_rms if signal_rms > 1e-8 else rms
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    cos = float(np.dot(a_f, b_f) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
    return cos, rel_rms, rms


def make_inputs(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def _get_hf_model_internals(model):
    """Return (language_model_core, layers, final_norm, lm_head).

    Gemma4UnifiedForConditionalGeneration layout:
        model.model -> Gemma4UnifiedModel -> .language_model -> Gemma4UnifiedTextModel
        model.lm_head -> Linear
    """
    lm_head = model.lm_head
    inner = model.model
    if hasattr(inner, "language_model"):
        core = inner.language_model
    else:
        core = inner
    return core, core.layers, core.norm, lm_head


# ---------------------------------------------------------------------------
# HuggingFace forward — uses hooks for reliable per-layer capture
# ---------------------------------------------------------------------------


def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Run HF forward on CPU and capture per-layer outputs via hooks.

    Captures:
    - Full layer outputs (post-hook on each decoder layer)
    - Mid-layer states after attention (pre-hook on pre_feedforward_layernorm)

    The mid-layer states correspond to Surogate's blocks[i].res_att (the
    hidden state after the first residual add + attention, before MLP).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    core, layers, final_norm, lm_head = _get_hf_model_internals(model)
    num_layers = len(layers)

    result: dict[str, np.ndarray] = {}
    layer_outs: dict[int, torch.Tensor] = {}
    mid_states: dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(num_layers):
        # Post-hook on full layer: captures layer output (after both residual adds)
        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(layers[i].register_forward_hook(make_layer_hook(i)))

        # Pre-hook on pre_feedforward_layernorm: captures mid-layer state
        # = hidden_states after first residual add (attention output + input residual)
        # This corresponds to Surogate's blocks[i].res_att
        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(layers[i].pre_feedforward_layernorm.register_forward_pre_hook(make_mid_hook(i)))

    with torch.no_grad():
        input_ids = torch.tensor(inputs, dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

        # Per-layer full outputs
        for i in range(num_layers):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()

        # Per-layer mid-states (= blocks[i].res_att in Surogate)
        for i in range(num_layers):
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        # Pre-norm = last layer output (before final RMSNorm)
        pre_norm = layer_outs[num_layers - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        # Post-norm = explicitly apply final norm
        post_norm = final_norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        # Logits for reference
        logits = lm_head(post_norm)
        result["logits"] = logits.float().cpu().numpy()

    result["num_layers"] = num_layers

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Surogate forward
# ---------------------------------------------------------------------------


def run_surogate_forward(model_dir: Path, inputs: np.ndarray, targets: np.ndarray, num_layers: int) -> None:
    """Run Surogate forward pass and dump tensors to DUMP_DIR.

    Dumpable tensors:
    - blocks[i].res_att: mid-layer hidden state (after attention, before MLP)
      Available for layers 0..num_layers-2. Last layer maps to residualN.
    - residual_final: full hidden state before final norm
    - xF: after final RMSNorm
    """
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(num_layers):
        dump_tensors.append(f"blocks[{i}].res_att")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR)
    os.environ["SUROGATE_DEBUG_DUMP_LAYER"] = "-1"
    os.environ["SUROGATE_MIN_STACK_MB"] = "512"

    hf_config = json.loads((model_dir / "config.json").read_text())

    cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    # offload_master=True: the 12B bf16 weights (~24GB) plus the persistent
    # arena exceed a single 32GB GPU; keep masters on CPU like the stock
    # gemma4-12b training config (cpu_training: true) does.
    opts = _surogate.RuntimeOptions(
        offload_residual=False,
        use_cuda_graphs=False,
        offload_master=True,
        offload_grads=True,
        offload_optimizer=True,
        shard_gradients=True,
        use_zero_copy=False,
    )
    from surogate.dsl import models as _  # noqa: F401 - register models
    from surogate.dsl.ir_builder import resolve_architecture
    from surogate.dsl.py_compiler import compile_model_for_hf

    arch = resolve_architecture(hf_config)
    opts.dsl_ir_json = compile_model_for_hf(arch, hf_config)

    # Use LoRA to avoid allocating full parameter gradients (12B model
    # gradient footprint exceeds single 32GB GPU memory).
    lora_cfg = _surogate.LoRAAdapterConfig()
    lora_cfg.rank = 8

    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=cfg,
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=lora_cfg,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    try:
        trainer.step(inputs, targets)
    except RuntimeError as e:
        # Forward dump tensors are captured before backward.
        # Allow backward OOM/errors for the onboarding validation.
        if "OOM" in str(e) or "backward" in str(e).lower():
            pass
        else:
            raise


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Gemma4 Unified weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return snapshot


@pytest.fixture(scope="module")
def layer_types(model_dir):
    """HF layer_types list — used to report sliding vs full (k_eq_v) layers."""
    config = json.loads((model_dir / "config.json").read_text())
    tc = config.get("text_config", config)
    return tc.get("layer_types") or []


@pytest.fixture(scope="module")
def forward_results(model_dir):
    """Run both HF and Surogate forward, return HF results dict."""
    config = json.loads((model_dir / "config.json").read_text())
    vocab_size = config.get("vocab_size") or config.get("text_config", {}).get("vocab_size", 262144)
    data = make_inputs(vocab_size)

    hf = run_hf_forward(model_dir, data["inputs"])
    num_layers = hf.pop("num_layers")
    run_surogate_forward(model_dir, data["inputs"], data["targets"], num_layers)
    hf["_num_layers"] = num_layers
    return hf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _layer_tag(layer_types, i: int) -> str:
    if i < len(layer_types):
        return "full/k_eq_v" if layer_types[i] == "full_attention" else "sliding"
    return "?"


def _cos_tol(i: int) -> float:
    return COS_TOL_EARLY if i < EARLY_LAYERS else COS_TOL_LATE


def _collect_layer_cos(hf, num_layers) -> dict[int, float]:
    cos_by_layer: dict[int, float] = {}
    for i in range(num_layers - 1):
        try:
            rt = load_dump(f"blocks[{i}].res_att")
        except FileNotFoundError:
            continue
        cos, _, _ = diff_stats(rt, hf[f"mid_state_{i}"])
        cos_by_layer[i] = cos
    return cos_by_layer


def _cliff_failures(cos_by_layer: dict[int, float], layer_types) -> list[str]:
    """Flag layers whose cos drops sharply below the median of their neighbors.

    Inherent bf16 drift is smooth across depth; a correctness bug in one
    layer shows as a localized cliff.
    """
    failures = []
    layers = sorted(cos_by_layer)
    for i in layers:
        neighbors = [cos_by_layer[j] for j in layers if j != i and abs(j - i) <= CLIFF_WINDOW]
        if len(neighbors) < 3:
            continue
        med = float(np.median(neighbors))
        if cos_by_layer[i] < med - CLIFF_TOL:
            failures.append(
                f"layer {i} [{_layer_tag(layer_types, i)}]: cos={cos_by_layer[i]:.6f} is "
                f"{med - cos_by_layer[i]:.4f} below neighbor median {med:.6f} (cliff_tol={CLIFF_TOL})"
            )
    return failures


class TestGemma4UnifiedOnboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_mid_state(self, forward_results, layer_types):
        """Check that per-layer mid-states (after attention) match.

        Surogate: blocks[i].res_att = hidden_state + attn_out (before MLP)
        HF: input to pre_feedforward_layernorm (captured via pre-hook)

        Two criteria: a depth-aware absolute floor (strict for early layers,
        the measured inherent-bf16-drift floor for deep layers), and a cliff
        detector that flags any layer dropping sharply below its neighbors.
        """
        hf = forward_results
        num_layers = hf["_num_layers"]
        failures = []

        cos_by_layer = _collect_layer_cos(hf, num_layers)
        for i in range(num_layers - 1):
            if i not in cos_by_layer:
                failures.append(f"layer {i}: dump not found")
                continue
            cos = cos_by_layer[i]
            tol = _cos_tol(i)
            if cos < tol:
                failures.append(f"layer {i} [{_layer_tag(layer_types, i)}]: cos={cos:.6f} (cos_tol={tol})")

        failures.extend(_cliff_failures(cos_by_layer, layer_types))

        if failures:
            pytest.fail("Per-layer mid-state mismatches:\n" + "\n".join(failures))

    def test_full_attention_layers_mid_state(self, forward_results, layer_types):
        """Focused check on the k_eq_v full-attention layers only.

        These layers exercise the V-reuses-K projection path, the 2-source
        qkv fuse import, and the per-block-type KV head count — the parts
        unique to gemma4_unified. A failure here with passing sliding layers
        points squarely at the k_eq_v implementation.
        """
        hf = forward_results
        num_layers = hf["_num_layers"]
        full_layers = [i for i, t in enumerate(layer_types[: num_layers - 1]) if t == "full_attention"]
        if not full_layers:
            pytest.skip("no full-attention layers reported in config")

        cos_by_layer = _collect_layer_cos(hf, num_layers)
        failures = []
        for i in full_layers:
            if i not in cos_by_layer:
                failures.append(f"layer {i}: dump not found")
                continue
            cos = cos_by_layer[i]
            tol = _cos_tol(i)
            if cos < tol:
                failures.append(f"layer {i}: cos={cos:.6f} (cos_tol={tol})")

        # Cliff check restricted to full layers, against all-layer neighbors
        failures.extend(f for f in _cliff_failures(cos_by_layer, layer_types) if "full" in f)

        if failures:
            pytest.fail("k_eq_v full-attention layer mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, forward_results):
        """Check that xF (after final RMSNorm) matches HF post-norm output."""
        hf = forward_results
        rt_xf = load_dump("xF")
        cos, rel_rms, abs_rms = diff_stats(rt_xf, hf["post_norm"])
        assert cos > COS_TOL_LATE, f"xF (final norm output) cos={cos:.6f} rel_rms={rel_rms:.4e}"

    def test_residual_final(self, forward_results):
        """Check that residual_final (before final norm) matches HF pre-norm."""
        hf = forward_results

        try:
            rt_residual_final = load_dump("residual_final")
        except FileNotFoundError:
            pytest.skip("residual_final dump not available")

        cos, rel_rms, abs_rms = diff_stats(rt_residual_final, hf["pre_norm"])
        assert cos > COS_TOL_LATE, f"residual_final cos={cos:.6f} rel_rms={rel_rms:.4e}"

    def test_summary(self, forward_results, layer_types):
        """Print a summary table of all comparisons (informational)."""
        hf = forward_results
        num_layers = hf["_num_layers"]
        rows: list[tuple[str, float, float, float]] = []

        for i in range(num_layers - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att")
                hf_mid = hf[f"mid_state_{i}"]
                cos, rel_rms, abs_rms = diff_stats(rt_res_att, hf_mid)
                rows.append((f"blocks[{i}].res_att [{_layer_tag(layer_types, i)}]", cos, rel_rms, abs_rms))
            except (FileNotFoundError, KeyError):
                rows.append((f"blocks[{i}].res_att", float("nan"), float("nan"), float("nan")))

        try:
            rt_xf = load_dump("xF")
            cos, rel_rms, abs_rms = diff_stats(rt_xf, hf["post_norm"])
            rows.append(("xF (post-norm)", cos, rel_rms, abs_rms))
        except FileNotFoundError:
            pass

        try:
            rt_rf = load_dump("residual_final")
            cos, rel_rms, abs_rms = diff_stats(rt_rf, hf["pre_norm"])
            rows.append(("residual_final (pre-norm)", cos, rel_rms, abs_rms))
        except FileNotFoundError:
            pass

        print("\n--- Gemma4 Unified Forward Compare (Surogate vs HF) ---")
        for idx, (name, cos, rel_rms, abs_rms) in enumerate(rows):
            tol = _cos_tol(idx) if name.startswith("blocks[") else COS_TOL_LATE
            status = "OK" if cos >= tol else "FAIL"
            print(f"  {name:42s} cos={cos:.6f}  rel_rms={rel_rms:.4e}  abs_rms={abs_rms:.4e}  [{status}]")
