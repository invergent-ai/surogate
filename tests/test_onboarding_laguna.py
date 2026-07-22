"""Onboarding test: Laguna model (LagunaForCausalLM, Poolside).

Validates that the Surogate DSL forward AND backward pass match the
HuggingFace reference for a mini (truncated) Laguna model. Compares per-layer
hidden states, final-norm outputs, and selected parameter gradients with
tolerances.

The first NUM_LAYERS=5 layers of Laguna-XS-2.1 cover all block variants:
layer 0 full_attention+dense, layers 1-3 sliding_attention+sparse,
layer 4 full_attention+sparse — plus both rope configs (YaRN full / default
sliding) and both query head counts (48 full / 64 sliding).

The mini config overrides sliding_window to 8 (< SEQ_LEN) so the sliding
window masking is numerically exercised — with the checkpoint's window of 512
a 16-token sequence would make sliding and full attention bit-identical.

Requirements:
    - GPU with enough VRAM for a 5-layer Laguna model
    - HF weights: set LAGUNA_MODEL_PATH env var or have poolside/Laguna-XS-2.1 cached

Usage:
    pytest tests/test_onboarding_laguna.py -v --no-header
    LAGUNA_MODEL_PATH=/path/to/Laguna-XS-2.1 pytest tests/test_onboarding_laguna.py -v
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

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "poolside/Laguna-XS-2.1"
ENV_VAR = "LAGUNA_MODEL_PATH"
NUM_LAYERS = 5
SEED = 42
BATCH = 1
SEQ_LEN = 16
# Override the checkpoint's sliding_window (512) so the window masking is
# actually exercised at SEQ_LEN=16. Both HF and Surogate read the same mini
# config, so parity comparisons remain valid.
SLIDING_WINDOW_OVERRIDE = 8
# bf16 forward through MoE layers accumulates more error than dense models
# due to routing (sigmoid + topk + correction bias) numerical variation.
RMS_TOL = 5e-2  # per-layer mid-state tolerance
FINAL_NORM_TOL = 2e-1  # post-norm tolerance (RMSNorm amplifies small diffs)
GRAD_REL_RMS_TOL = 1e-1
GRAD_RMS_TOL = 5e-2
GRAD_SAMPLE_SIZE = 131072

MINI_MODEL_DIR = Path("tmp/onboarding_laguna_mini")
DUMP_DIR = Path("tmp/onboarding_laguna_dumps")

# Per-layer config lists that must be truncated together with num_hidden_layers.
_PER_LAYER_KEYS = ("layer_types", "mlp_layer_types", "num_attention_heads_per_layer", "gating_types")

_DUMP_ENV_KEYS = ("SUROGATE_DEBUG_DUMP_TENSORS", "SUROGATE_DEBUG_DUMP_DIR", "SUROGATE_DEBUG_DUMP_LAYER")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_path() -> Path | None:
    """Resolve the path to Laguna weights."""
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
    """Create (or refresh) a truncated Laguna model with NUM_LAYERS layers.

    Idempotent: config.json and symlinks are (re)written on every call, so a
    changed truncation/override or a refreshed snapshot never leaves a stale
    cached mini model behind.
    """
    import shutil

    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS
    for key in _PER_LAYER_KEYS:
        if key in config and isinstance(config[key], list):
            config[key] = config[key][:NUM_LAYERS]
    config["sliding_window"] = SLIDING_WINDOW_OVERRIDE
    (MINI_MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    for tok_file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "configuration_laguna.py",
        "modeling_laguna.py",
    ]:
        src = snapshot_dir / tok_file
        if src.exists():
            shutil.copy2(src, MINI_MODEL_DIR / tok_file)

    def relink(link: Path, target: Path) -> None:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())

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
            relink(MINI_MODEL_DIR / fname, snapshot_dir / fname)
    elif single_path.exists():
        relink(MINI_MODEL_DIR / "model.safetensors", single_path)
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


def sample_flat(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().float().reshape(-1)
    if flat.numel() <= GRAD_SAMPLE_SIZE:
        return flat.cpu()
    stride = flat.numel() // GRAD_SAMPLE_SIZE
    return flat[:: max(stride, 1)][:GRAD_SAMPLE_SIZE].cpu()


def grad_diff_stats(rt: torch.Tensor, hf: torch.Tensor) -> tuple[float, float, float]:
    rt_s = sample_flat(rt)
    hf_s = sample_flat(hf)
    diff = rt_s - hf_s
    rms = float(torch.sqrt(torch.mean(diff * diff)).item())
    max_abs = float(torch.max(torch.abs(diff)).item())
    ref_rms = float(torch.sqrt(torch.mean(hf_s * hf_s)).item())
    rel_rms = rms / max(ref_rms, 1e-12)
    return rel_rms, rms, max_abs


def make_inputs(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def build_grad_mapping() -> dict[str, object]:
    """Map Surogate gradient names to HF parameter names.

    HF spec forms:
      - str: direct parameter name
      - ("fuse", up_name, gate_name): surogate weight is [up; gate] rows
      - ("experts_gate_up", name): HF batched [E, 2M, C] is [gate; up] per
        expert while surogate stores [up; gate] — halves are swapped.
    """
    l0 = "model.layers.0"  # full_attention + dense
    l1 = "model.layers.1"  # sliding_attention + sparse
    return {
        # Layer 0: full attention (48 heads, YaRN partial rope) + dense SwiGLU
        "blocks[0].ln1_weight": f"{l0}.input_layernorm.weight",
        "blocks[0].q_proj_weight": f"{l0}.self_attn.q_proj.weight",
        "blocks[0].g_proj_weight": f"{l0}.self_attn.g_proj.weight",
        "blocks[0].q_norm_weight": f"{l0}.self_attn.q_norm.weight",
        "blocks[0].mlp_up_weight": ("fuse", f"{l0}.mlp.up_proj.weight", f"{l0}.mlp.gate_proj.weight"),
        "blocks[0].mlp_down_weight": f"{l0}.mlp.down_proj.weight",
        # Layer 1: sliding attention (64 heads, windowed) + sparse MoE
        "blocks[1].g_proj_weight": f"{l1}.self_attn.g_proj.weight",
        "blocks[1].out_weight": f"{l1}.self_attn.o_proj.weight",
        "blocks[1].k_norm_weight": f"{l1}.self_attn.k_norm.weight",
        "blocks[1].router_weight": f"{l1}.mlp.gate.weight",
        # The checkpoint's auto_map remote code names this ``shared_expert``;
        # the transformers-native port uses ``shared_experts`` (lookup tries both).
        "blocks[1].shared_expert_up": f"{l1}.mlp.shared_expert.up_proj.weight",
        "blocks[1].shared_expert_down": f"{l1}.mlp.shared_expert.down_proj.weight",
        "blocks[1].experts_gate_up": ("experts_gate_up", f"{l1}.mlp.experts.gate_up_proj"),
        "blocks[1].experts_down": f"{l1}.mlp.experts.down_proj",
        # Model level
        "final_norm": "model.norm.weight",
    }


# ---------------------------------------------------------------------------
# HuggingFace forward — uses hooks for reliable per-layer capture
# ---------------------------------------------------------------------------


def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> dict[str, np.ndarray]:
    """Run HF forward and capture per-layer outputs via hooks.

    Captures mid-layer states (after attention, before MLP/MoE) via pre-hook on
    post_attention_layernorm. These correspond to Surogate's blocks[i].res_att.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
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

        hooks.append(model.model.layers[i].register_forward_hook(make_layer_hook(i)))

        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(make_mid_hook(i)))

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

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


def run_hf_backward(
    model_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    grad_mapping: dict[str, object],
) -> dict[str, torch.Tensor]:
    """Run HF backward once and collect selected gradients."""
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="eager",
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

    def lookup(name: str):
        p = named_params.get(name)
        if p is None and ".shared_expert." in name:
            p = named_params.get(name.replace(".shared_expert.", ".shared_experts."))
        return p

    grads: dict[str, torch.Tensor] = {}

    for rt_name, hf_spec in grad_mapping.items():
        if isinstance(hf_spec, tuple) and hf_spec[0] == "fuse":
            _, up_name, gate_name = hf_spec
            up = named_params.get(up_name)
            gate = named_params.get(gate_name)
            if up is None or gate is None or up.grad is None or gate.grad is None:
                continue
            grads[rt_name] = torch.cat([up.grad, gate.grad], dim=0).float().cpu()
        elif isinstance(hf_spec, tuple) and hf_spec[0] == "experts_gate_up":
            p = lookup(hf_spec[1])
            if p is None or p.grad is None:
                continue
            g = p.grad  # [E, 2M, C], [gate; up] rows per expert (HF chunk(2))
            m = g.shape[1] // 2
            grads[rt_name] = torch.cat([g[:, m:, :], g[:, :m, :]], dim=1).float().cpu()
        else:
            p = lookup(hf_spec)
            if p is None or p.grad is None:
                continue
            grads[rt_name] = p.grad.float().cpu()

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
    grad_mapping: dict[str, object],
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

    try:
        cfg = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
        # Offload master/optimizer: the 5-layer mini model still carries
        # 4 x 256-expert MoE layers (~3.7B params). Gradients stay on the GPU
        # so trainer.get_gradients() can read them.
        opts = _surogate.RuntimeOptions(
            offload_residual=False,
            use_cuda_graphs=False,
            offload_master=True,
            offload_grads=False,
            offload_optimizer=True,
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
                pytest.skip(f"Surogate step skipped due to CUDA OOM: {e}")
            raise

        raw = trainer.get_gradients(0)
        grads: dict[str, torch.Tensor] = {}
        for rt_name in grad_mapping:
            if rt_name not in raw:
                continue
            grads[rt_name] = torch.utils.dlpack.from_dlpack(raw[rt_name]).detach().float().cpu()

        del trainer
        torch.cuda.empty_cache()
        return grads
    finally:
        for key in _DUMP_ENV_KEYS:
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Laguna weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def inputs_data(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    return make_inputs(config["vocab_size"])


@pytest.fixture(scope="module")
def grad_mapping():
    return build_grad_mapping()


@pytest.fixture(scope="module")
def rt_grads(model_dir, inputs_data, grad_mapping):
    """Run the Surogate step (also produces the forward dumps)."""
    return run_surogate_step(model_dir, inputs_data["inputs"], inputs_data["targets"], grad_mapping)


@pytest.fixture(scope="module")
def forward_results(model_dir, inputs_data, rt_grads):
    """HF forward reference (Surogate dumps already written by rt_grads)."""
    del rt_grads  # ordering only: ensure the Surogate step ran first
    return run_hf_forward(model_dir, inputs_data["inputs"])


@pytest.fixture(scope="module")
def hf_grads(model_dir, inputs_data, grad_mapping):
    return run_hf_backward(model_dir, inputs_data["inputs"], inputs_data["targets"], grad_mapping)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLagunaOnboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace."""

    def test_per_layer_mid_state(self, forward_results):
        """Check that per-layer mid-states (after attention) match.

        blocks[i].res_att only dumped for layers 0..NUM_LAYERS-2
        (last layer maps to residualN, not separately dumpable).
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
                failures.append(f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})")

        if failures:
            pytest.fail("Per-layer mid-state mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, forward_results):
        """Check that xF (after final RMSNorm) matches HF post-norm output."""
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
        assert rms < RMS_TOL, f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"

    def test_summary(self, forward_results):
        """Print a summary table of all comparisons (informational)."""
        hf = forward_results
        rows: list[tuple[str, float, float]] = []

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

        print("\n--- Laguna Forward Compare (Surogate vs HF) ---")
        for name, rms, max_abs in rows:
            tol = FINAL_NORM_TOL if "post-norm" in name else RMS_TOL
            status = "OK" if rms <= tol else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")


class TestLagunaOnboardingBackward:
    """Selected-gradient comparison: Surogate vs HF autograd.

    Covers the backward paths new to Laguna: softplus gate (g_proj), the
    per-head broadcast-gate multiply, GLM-prefix partial rope backward
    (layer 0), sliding-window attention backward (layer 1), the fused dense
    SwiGLU, the sigmoid+bias router, and the grouped expert GEMMs.
    """

    def test_selected_gradients(self, rt_grads, hf_grads, grad_mapping):
        failures = []
        rows: list[tuple[str, float, float, float]] = []

        for rt_name in sorted(grad_mapping.keys()):
            rt = rt_grads.get(rt_name)
            hf = hf_grads.get(rt_name)
            if rt is None or hf is None:
                failures.append(f"{rt_name}: missing gradient (rt={rt is not None}, hf={hf is not None})")
                continue
            if rt.numel() != hf.numel():
                failures.append(f"{rt_name}: shape mismatch rt={tuple(rt.shape)} hf={tuple(hf.shape)}")
                continue

            rel_rms, rms, max_abs = grad_diff_stats(rt.reshape(hf.shape), hf)
            rows.append((rt_name, rel_rms, rms, max_abs))
            if rel_rms > GRAD_REL_RMS_TOL and rms > GRAD_RMS_TOL:
                failures.append(
                    f"{rt_name}: rel_rms={rel_rms:.4e} rms={rms:.4e} max_abs={max_abs:.4e} "
                    f"(tol rel={GRAD_REL_RMS_TOL:.0e}, abs={GRAD_RMS_TOL:.0e})"
                )

        print("\n--- Laguna Backward Compare (sampled grads) ---")
        for name, rel_rms, rms, max_abs in rows:
            status = "OK" if (rel_rms <= GRAD_REL_RMS_TOL or rms <= GRAD_RMS_TOL) else "FAIL"
            print(f"  {name:40s} rel_rms={rel_rms:.4e}  rms={rms:.4e}  max={max_abs:.4e}  [{status}]")

        if failures:
            pytest.fail("Gradient mismatches:\n" + "\n".join(failures))
