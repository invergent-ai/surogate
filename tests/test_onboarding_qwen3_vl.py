"""Onboarding test: Qwen3-VL text backbone.

Validates that the Surogate DSL forward pass matches HuggingFace reference
for a mini (truncated) Qwen3-VL model. This test uses text-only inputs and
expects visual buffers to be zeroed (no visual injection).

Requirements:
    - GPU with enough VRAM for a small Qwen3-VL model
    - HF weights: set QWEN3_VL_MODEL_PATH env var or have Qwen/Qwen3-VL-2B-Instruct cached

Usage:
    pytest tests/test_onboarding_qwen3_vl.py -v --no-header
    QWEN3_VL_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct pytest tests/test_onboarding_qwen3_vl.py -v
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

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ENV_VAR = "QWEN3_VL_MODEL_PATH"
NUM_LAYERS = 2
SEED = 42
BATCH = 1
SEQ_LEN = 16
SEQ_LEN_MM = 32
IMAGE_SIZE = 64
RMS_TOL = 5e-2

MINI_MODEL_DIR = Path("tmp/onboarding_qwen3_vl_mini")
DUMP_DIR = Path("tmp/onboarding_qwen3_vl_dumps")
DUMP_DIR_MM = Path("tmp/onboarding_qwen3_vl_dumps_mm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_model_path() -> Path:
    """Resolve the path to Qwen3-VL weights."""
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
    """Create a truncated Qwen3-VL model with NUM_LAYERS text layers."""
    if MINI_MODEL_DIR.exists():
        return MINI_MODEL_DIR
    MINI_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads((snapshot_dir / "config.json").read_text())
    if "text_config" in config and isinstance(config["text_config"], dict):
        config["text_config"]["num_hidden_layers"] = NUM_LAYERS
    if "num_hidden_layers" in config:
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
        prefixes = [f"model.language_model.layers.{i}." for i in range(NUM_LAYERS)]
        extra = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        }

        def want(name: str) -> bool:
            if name in extra:
                return True
            if name.startswith("model.language_model."):
                return any(name.startswith(p) for p in prefixes)
            return False

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


def load_dump(name: str, dump_dir: Path = DUMP_DIR) -> np.ndarray:
    safe = sanitize(name)
    meta_path = dump_dir / f"{safe}.json"
    bin_path = dump_dir / f"{safe}.bin"
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


def build_multimodal_payload(model, tokenizer) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    config = model.config
    image_token_id = config.image_token_id
    vision_start_id = config.vision_start_token_id
    vision_end_id = config.vision_end_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    patch_size = model.visual.patch_size
    merge = model.visual.spatial_merge_size
    grid_h = IMAGE_SIZE // patch_size
    grid_w = IMAGE_SIZE // patch_size
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)

    num_visual = (grid_h * grid_w) // (merge * merge)
    prefix = tokenizer.encode("Describe", add_special_tokens=False)
    suffix = tokenizer.encode("image.", add_special_tokens=False)
    tokens = prefix + [vision_start_id] + [image_token_id] * num_visual + [vision_end_id] + suffix
    if len(tokens) > SEQ_LEN_MM:
        raise ValueError(f"Multimodal input length {len(tokens)} exceeds SEQ_LEN_MM={SEQ_LEN_MM}")
    if len(tokens) < SEQ_LEN_MM:
        tokens = tokens + [eos_id] * (SEQ_LEN_MM - len(tokens))

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    pixel_values = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
    image_embeds = image_embeds[0]

    expected_layers = len(getattr(model.visual, "deepstack_visual_indexes", []))
    if expected_layers and len(deepstack_image_embeds) != expected_layers:
        raise ValueError(
            f"deepstack layers mismatch: expected {expected_layers}, got {len(deepstack_image_embeds)}"
        )

    visual_mask = (input_ids == image_token_id).squeeze(0)
    num_visual_actual = int(visual_mask.sum().item())
    if num_visual_actual != image_embeds.shape[0]:
        raise ValueError(
            f"visual token mismatch: mask has {num_visual_actual}, embeds have {image_embeds.shape[0]}"
        )

    rope_fn = model.get_rope_index if hasattr(model, "get_rope_index") else model.model.get_rope_index
    position_ids, _ = rope_fn(
        input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )

    inputs_np = input_ids.cpu().numpy().astype(np.int32)
    targets = inputs_np.copy()
    targets[:, :-1] = inputs_np[:, 1:]
    targets[:, -1] = -100

    visual_pos_masks = visual_mask.cpu().numpy().astype(np.int32)[None, :]
    hidden = config.text_config.hidden_size
    packed = np.zeros((SEQ_LEN_MM, hidden), dtype=np.float32)
    packed[:num_visual_actual] = image_embeds.float().cpu().numpy()

    deepstack_packed: List[np.ndarray] = []
    for ds in deepstack_image_embeds:
        ds_buf = np.zeros_like(packed)
        ds_buf[:num_visual_actual] = ds.float().cpu().numpy()
        deepstack_packed.append(ds_buf)

    payload = {
        "inputs": inputs_np,
        "targets": targets,
        "position_ids": position_ids.cpu().numpy().astype(np.int32),
        "visual_pos_masks": visual_pos_masks,
        "visual_embeds": packed,
        "deepstack_visual_embeds": deepstack_packed,
    }
    hf_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    return payload, hf_inputs


# ---------------------------------------------------------------------------
# HuggingFace forward
# ---------------------------------------------------------------------------

def run_hf_forward(model_dir: Path, inputs: np.ndarray) -> Dict[str, np.ndarray]:
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
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

    text_model = model.model.language_model

    for i in range(NUM_LAYERS):
        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(text_model.layers[i].register_forward_hook(make_layer_hook(i)))

        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(
            text_model.layers[i].post_attention_layernorm.register_forward_pre_hook(
                make_mid_hook(i)
            )
        )

    with torch.no_grad():
        input_ids = torch.tensor(inputs, device="cuda", dtype=torch.long)
        _ = model(input_ids=input_ids, use_cache=False)

        for i in range(NUM_LAYERS):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        pre_norm = layer_outs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        post_norm = text_model.norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        logits = model.lm_head(post_norm)
        result["logits"] = logits.float().cpu().numpy()

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result


def run_hf_forward_multimodal(model_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model.eval()

    payload, hf_inputs = build_multimodal_payload(model, tokenizer)

    result: Dict[str, np.ndarray] = {}
    layer_outs: Dict[int, torch.Tensor] = {}
    mid_states: Dict[int, torch.Tensor] = {}
    hooks = []

    text_model = model.model.language_model

    for i in range(NUM_LAYERS):
        def make_layer_hook(idx):
            def hook_fn(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                layer_outs[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(text_model.layers[i].register_forward_hook(make_layer_hook(i)))

        def make_mid_hook(idx):
            def hook_fn(module, args):
                hs = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                mid_states[idx] = hs.detach().clone()
            return hook_fn
        hooks.append(
            text_model.layers[i].post_attention_layernorm.register_forward_pre_hook(
                make_mid_hook(i)
            )
        )

    with torch.no_grad():
        _ = model(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
            pixel_values=hf_inputs["pixel_values"],
            image_grid_thw=hf_inputs["image_grid_thw"],
            use_cache=False,
        )

        for i in range(NUM_LAYERS):
            result[f"layer_output_{i}"] = layer_outs[i].float().cpu().numpy()
            result[f"mid_state_{i}"] = mid_states[i].float().cpu().numpy()

        pre_norm = layer_outs[NUM_LAYERS - 1]
        result["pre_norm"] = pre_norm.float().cpu().numpy()

        post_norm = text_model.norm(pre_norm)
        result["post_norm"] = post_norm.float().cpu().numpy()

        logits = model.lm_head(post_norm)
        result["logits"] = logits.float().cpu().numpy()

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return result, payload


# ---------------------------------------------------------------------------
# Surogate forward
# ---------------------------------------------------------------------------

def run_surogate_forward(model_dir: Path, inputs: np.ndarray, targets: np.ndarray) -> None:
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


def run_surogate_forward_multimodal(model_dir: Path, payload: Dict[str, np.ndarray]) -> None:
    DUMP_DIR_MM.mkdir(parents=True, exist_ok=True)
    for p in DUMP_DIR_MM.glob("*"):
        p.unlink()

    dump_tensors = ["xF", "residual_final", "ln_final_rstd"]
    for i in range(NUM_LAYERS):
        dump_tensors.append(f"blocks[{i}].res_att")

    os.environ["SUROGATE_DEBUG_DUMP_TENSORS"] = ",".join(dump_tensors)
    os.environ["SUROGATE_DEBUG_DUMP_DIR"] = str(DUMP_DIR_MM)
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
        seq_len=SEQ_LEN_MM,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=None,
        qlora_config=None,
    )
    trainer.import_weights(get_model_weights_path(str(model_dir)))
    trainer.set_visual_inputs(
        payload["visual_pos_masks"],
        payload["visual_embeds"],
        payload["deepstack_visual_embeds"],
    )
    trainer.step(payload["inputs"], payload["targets"], payload["position_ids"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_dir():
    snapshot = resolve_model_path()
    if snapshot is None:
        pytest.skip(f"Qwen3-VL weights not found. Set {ENV_VAR} or cache {MODEL_ID}")
    return prepare_mini_model(snapshot)


@pytest.fixture(scope="module")
def forward_results(model_dir):
    config = json.loads((model_dir / "config.json").read_text())
    if "text_config" in config and isinstance(config["text_config"], dict):
        vocab_size = config["text_config"]["vocab_size"]
    else:
        vocab_size = config["vocab_size"]
    data = make_inputs(vocab_size)

    run_surogate_forward(model_dir, data["inputs"], data["targets"])
    hf = run_hf_forward(model_dir, data["inputs"])
    return hf


@pytest.fixture(scope="module")
def forward_results_multimodal(model_dir):
    hf, payload = run_hf_forward_multimodal(model_dir)
    run_surogate_forward_multimodal(model_dir, payload)
    return hf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQwen3VLOnboarding:
    """Per-layer forward comparison: Surogate vs HuggingFace (text-only)."""

    def test_per_layer_mid_state(self, forward_results):
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
        hf = forward_results
        rt_xf = load_dump("xF")
        rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
        assert rms < RMS_TOL, (
            f"xF (final norm output) rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
        )

    def test_residual_final(self, forward_results):
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

        print("\n--- Qwen3-VL Forward Compare (Surogate vs HF) ---")
        for name, rms, max_abs in rows:
            status = "OK" if rms <= RMS_TOL else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")


class TestQwen3VLOnboardingMultimodal:
    """Per-layer forward comparison: Surogate vs HuggingFace (with visual inputs)."""

    def test_per_layer_mid_state(self, forward_results_multimodal):
        hf = forward_results_multimodal
        failures = []

        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att", dump_dir=DUMP_DIR_MM)
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

    def test_final_norm_output(self, forward_results_multimodal):
        hf = forward_results_multimodal
        rt_xf = load_dump("xF", dump_dir=DUMP_DIR_MM)
        rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
        assert rms < RMS_TOL, (
            f"xF (final norm output) rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
        )

    def test_residual_final(self, forward_results_multimodal):
        hf = forward_results_multimodal

        try:
            rt_residual_final = load_dump("residual_final", dump_dir=DUMP_DIR_MM)
        except FileNotFoundError:
            pytest.skip("residual_final dump not available")

        rms, max_abs = diff_stats(rt_residual_final, hf["pre_norm"])
        assert rms < RMS_TOL, (
            f"residual_final rms={rms:.4e} max_abs={max_abs:.4e} (tol={RMS_TOL:.0e})"
        )

    def test_summary(self, forward_results_multimodal):
        hf = forward_results_multimodal
        rows: List[Tuple[str, float, float]] = []

        for i in range(NUM_LAYERS - 1):
            try:
                rt_res_att = load_dump(f"blocks[{i}].res_att", dump_dir=DUMP_DIR_MM)
                hf_mid = hf[f"mid_state_{i}"]
                rms, max_abs = diff_stats(rt_res_att, hf_mid)
                rows.append((f"blocks[{i}].res_att", rms, max_abs))
            except (FileNotFoundError, KeyError):
                rows.append((f"blocks[{i}].res_att", float("nan"), float("nan")))

        try:
            rt_xf = load_dump("xF", dump_dir=DUMP_DIR_MM)
            rms, max_abs = diff_stats(rt_xf, hf["post_norm"])
            rows.append(("xF (post-norm)", rms, max_abs))
        except FileNotFoundError:
            pass

        try:
            rt_rf = load_dump("residual_final", dump_dir=DUMP_DIR_MM)
            rms, max_abs = diff_stats(rt_rf, hf["pre_norm"])
            rows.append(("residual_final (pre-norm)", rms, max_abs))
        except FileNotFoundError:
            pass

        print("\n--- Qwen3-VL Forward Compare (Surogate vs HF, multimodal) ---")
        for name, rms, max_abs in rows:
            status = "OK" if rms <= RMS_TOL else "FAIL"
            print(f"  {name:30s} rms={rms:.4e}  max={max_abs:.4e}  [{status}]")
