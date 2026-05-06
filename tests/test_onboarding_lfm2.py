"""Onboarding test: LFM2 hybrid model.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for a mini (truncated) LFM2 model. Compares per-layer residual inputs and
final-norm outputs with tolerances.

Requirements:
    - GPU with enough VRAM for a small LFM2 model
    - HF weights: set LFM2_MODEL_PATH env var or have LiquidAI/LFM2-1.2B cached

Usage:
    pytest tests/test_onboarding_lfm2.py -v --no-header
    LFM2_MODEL_PATH=/path/to/LFM2-1.2B pytest tests/test_onboarding_lfm2.py -v
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_ID = "LiquidAI/LFM2-1.2B"
ENV_VAR = "LFM2_MODEL_PATH"
NUM_LAYERS = 4
SEED = 42
BATCH = 1
SEQ_LEN = 16
RMS_TOL = 5e-2

MINI_MODEL_DIR = Path("tmp/onboarding_lfm2_mini")
DUMP_DIR = Path("tmp/onboarding_lfm2_dumps")


def resolve_model_path() -> Path | None:
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
    if MINI_MODEL_DIR.exists():
        cfg_path = MINI_MODEL_DIR / "config.json"
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
                if existing.get("num_hidden_layers") == NUM_LAYERS:
                    return MINI_MODEL_DIR
            except Exception:
                pass
        shutil.rmtree(MINI_MODEL_DIR)

    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    config["num_hidden_layers"] = NUM_LAYERS
    if isinstance(config.get("layer_types"), list):
        config["layer_types"] = config["layer_types"][:NUM_LAYERS]
    if isinstance(config.get("full_attn_idxs"), list):
        config["full_attn_idxs"] = [i for i in config["full_attn_idxs"] if i < NUM_LAYERS]
    config["max_position_embeddings"] = min(config.get("max_position_embeddings", 256), max(256, SEQ_LEN * 8))

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

    if index_path.exists():
        base_index = json.loads(index_path.read_text())
        weight_map = base_index.get("weight_map", {})
        prefixes = [f"model.layers.{i}." for i in range(NUM_LAYERS)]
        extra = {"model.embed_tokens.weight", "model.embedding_norm.weight", "lm_head.weight"}

        def want(name: str) -> bool:
            if name in extra:
                return True
            return any(name.startswith(prefix) for prefix in prefixes)

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


def diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a.astype(np.float32) - b.astype(np.float32)
    rms = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    return rms, max_abs


def load_hf_tensor(model_dir: Path, name: str) -> np.ndarray:
    from safetensors import safe_open

    single_path = model_dir / "model.safetensors"
    if single_path.exists():
        with safe_open(single_path, framework="pt", device="cpu") as sf:
            return sf.get_tensor(name).float().numpy()

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = index["weight_map"][name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as sf:
        return sf.get_tensor(name).float().numpy()


def make_inputs(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    inputs = rng.integers(0, vocab_size, size=(BATCH, SEQ_LEN), dtype=np.int32)
    targets = inputs.copy()
    targets[:, :-1] = inputs[:, 1:]
    targets[:, -1] = -100
    return {"inputs": inputs, "targets": targets}


def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> dict[str, np.ndarray]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    result: dict[str, np.ndarray] = {}
    layer_inputs: dict[int, torch.Tensor] = {}
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = []

    for i in range(NUM_LAYERS):

        def make_pre_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                layer_inputs[idx] = hs.detach().clone()

            return hook_fn

        def make_post_hook(idx):
            def hook_fn(module, args, output):
                hs = output if isinstance(output, torch.Tensor) else output[0]
                layer_outputs[idx] = hs.detach().clone()

            return hook_fn

        hooks.append(model.model.layers[i].register_forward_pre_hook(make_pre_hook(i)))
        hooks.append(model.model.layers[i].register_forward_hook(make_post_hook(i)))

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

        for i in range(NUM_LAYERS):
            result[f"layer_input_{i}"] = layer_inputs[i].float().cpu().numpy()
            result[f"layer_output_{i}"] = layer_outputs[i].float().cpu().numpy()

        pre_norm = layer_outputs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()
        post_norm = model.model.embedding_norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result


def run_surogate_forward(model_dir: Path, inputs: np.ndarray, targets: np.ndarray) -> None:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_operator")

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


@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"LFM2 weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def forward_results(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    data = make_inputs(config["vocab_size"])

    run_surogate_forward(model_dir, data["inputs"], data["targets"])
    return run_hf_forward(model_dir, data["inputs"])


class TestLfm2Onboarding:
    def test_per_layer_residual_input(self, forward_results):
        hf = forward_results
        failures = []

        for i in range(NUM_LAYERS - 1):
            dump_name = f"blocks[{i}].res_operator"
            try:
                rt_residual = load_dump(dump_name)
            except FileNotFoundError as e:
                failures.append(f"layer {i}: dump {dump_name} not found ({e})")
                continue

            hf_input = hf[f"layer_input_{i}"]
            rms, max_abs = diff_stats(rt_residual, hf_input)
            if rms > RMS_TOL:
                failures.append(f"layer {i}: rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})")

        if failures:
            pytest.fail("Per-layer residual input mismatches:\n" + "\n".join(failures))

    def test_final_norm_output(self, model_dir, forward_results):
        config = json.loads((model_dir / "config.json").read_text())
        rt_residual_final = load_dump("residual_final")
        rt_xf = load_dump("xF")
        weight = load_hf_tensor(model_dir, "model.embedding_norm.weight")
        expected = (
            rt_residual_final
            * np.reciprocal(
                np.sqrt(np.mean(rt_residual_final * rt_residual_final, axis=-1, keepdims=True) + config["norm_eps"])
            )
            * weight
        )
        rms, max_abs = diff_stats(rt_xf, expected)
        assert rms < 2e-2, f"xF rms={rms:.4e} max_abs={max_abs:.4e} (tol=2e-2)"

    def test_residual_final(self, forward_results):
        hf = forward_results
        rt_residual_final = load_dump("residual_final")
        rms, max_abs = diff_stats(rt_residual_final, hf["pre_norm"])
        assert rms < RMS_TOL, f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
