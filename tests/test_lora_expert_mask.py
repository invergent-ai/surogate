"""`lora_expert_mask`: expert LoRA on a per-layer subset of the routed experts (ESFT design, section 3).

CPU:
  * the mask parser and its validation (ids out of range or repeated, layers past the model, the run options a
    mask cannot share: ep_size > 1, selective expert dequant, gate_up_proj targets, lora: false);
  * the adapter's parameter / tensor counts from a checkpoint config, which the experiment's jobs check at step 0:
    wp1's real target set on the 35B-A3B config still gives 1,265,008,640 parameters in 480 tensors, and the arms'
    (a) 1,279,293,440 / 860, (b) 21,002,240 / 620, (c) 21,002,240 + 122,880 * sum(K) / 620 + 6 per layer with K > 0.

GPU (a tiny qwen3_5_moe: 3 linear-attention + 1 full-attention layer, 8 experts, top-2), each scenario in a fresh
process:
  1. without a mask the trainer allocates what it allocated before; with one, what the count formula says;
  2. a masked run equals a dense run whose unselected experts carry A = B = 0: per-token log-probs, every LoRA
     gradient (the selected experts' rows) and, after one AdamW 8-bit step, every adapter weight -- bit for bit
     against the dense cuBLAS path (SUROGATE_FUSED_EXPERT_LORA=0; the fused kernel rounds differently);
  3. the export holds exactly the selected experts, under their global ids, with their dense-run initial lora_A
     (gathered init); adapter_config.json records the mask; the adapter imports into the same mask and is refused,
     naming the masks, by another mask, by `all`, and an `all` adapter by the mask;
  4. `none` keeps the shared-expert, attention, linear-attention and router adapters and no `mlp.experts.*`;
  5. the trainer refuses a mask with an expert id >= E, a layer >= L, or repeated ids;
  6. chunked training (sequence_chunks 4) gives the loss and the LoRA gradients (GDN adapters included) of the
     unchunked step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surogate.core.config.lora_expert_mask import (
    MASK_FORMAT,
    LoRAExpertMask,
    check_lora_expert_mask_options,
    parse_lora_expert_mask,
    qwen3_5_moe_lora_counts,
    validate_lora_expert_mask,
)

# ---------------------------------------------------------------------------------------------- CPU

WP1_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ARM_TARGETS = WP1_TARGETS + ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

# The geometry of Surogate3.7-35B-A3B (hf_release/Surogate3.7-35B-A3B/config.json, text_config).
SUROGATE37_TEXT = {
    "hidden_size": 2048, "num_hidden_layers": 40, "num_attention_heads": 16, "num_key_value_heads": 2,
    "head_dim": 256, "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_num_value_heads": 32,
    "linear_value_head_dim": 128, "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512, "full_attention_interval": 4,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10,
}
RELEASE_CONFIG = Path("/mnt/data/work/hf_release/Surogate3.7-35B-A3B/config.json")


def _write(tmp_path, obj, name="mask.json"):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return path


class TestParse:
    def test_all_and_unset_mean_every_expert(self):
        assert parse_lora_expert_mask(None) is None
        assert parse_lora_expert_mask("all") is None
        assert parse_lora_expert_mask(" ALL ") is None

    def test_none_selects_no_expert(self):
        mask = parse_lora_expert_mask("none")
        assert mask.source == "none" and mask.layers == {} and mask.per_layer() == [] and mask.total() == 0

    def test_file(self, tmp_path):
        path = _write(tmp_path, {"format": MASK_FORMAT, "num_layers": 4, "num_experts": 8, "score": "esft_token",
                                 "layers": {"0": [5, 1], "1": [], "3": [3, 0, 2]}})
        mask = parse_lora_expert_mask(str(path))
        assert mask.source == str(path.resolve())
        assert mask.layers == {0: (1, 5), 1: (), 3: (0, 2, 3)}
        assert mask.per_layer() == [[1, 5], [], [], [0, 2, 3]]  # unlisted layer 2: none
        assert mask.total() == 5
        validate_lora_expert_mask(mask, num_layers=4, num_experts=8)
        assert mask.to_json(4, 8)["layers"] == {"0": [1, 5], "1": [], "2": [], "3": [0, 2, 3]}

    def test_inline_mapping_and_list(self):
        assert parse_lora_expert_mask({"layers": {"2": [7]}}).per_layer() == [[], [], [7]]
        assert parse_lora_expert_mask({"layers": [[1], [], [0, 3]]}).layers == {0: (1,), 1: (), 2: (0, 3)}

    @pytest.mark.parametrize("obj,match", [
        ({"layers": {"0": [1, 1]}}, "more than once"),
        ({"layers": {"0": [-1]}}, "negative"),
        ({"layers": {"0": [1.5]}}, "integer expert ids"),
        ({"layers": {"0": [True]}}, "integer expert ids"),
        ({"layers": {"x": [1]}}, "not an integer"),
        ({"layers": {"-1": [1]}}, "not a layer index"),
        ({"format": "other", "layers": {}}, "unknown format"),
        ({"num_layers": 4}, "no 'layers'"),
        ({"layers": {"0": [1]}, "num_experts": 0}, "positive integer"),
    ])
    def test_malformed(self, tmp_path, obj, match):
        with pytest.raises(ValueError, match=match):
            parse_lora_expert_mask(str(_write(tmp_path, obj)))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="neither 'all', 'none' nor an existing mask file"):
            parse_lora_expert_mask(str(tmp_path / "absent.json"))


class TestValidate:
    def test_against_the_model(self):
        ok = LoRAExpertMask("m", {0: (0, 7), 3: (1,)})
        validate_lora_expert_mask(ok, num_layers=4, num_experts=8)
        with pytest.raises(ValueError, match=r"layer 3 lists expert\(s\) \[8\]"):
            validate_lora_expert_mask(LoRAExpertMask("m", {3: (1, 8)}), num_layers=4, num_experts=8)
        with pytest.raises(ValueError, match="lists layer 4, but the model has 4 layers"):
            validate_lora_expert_mask(LoRAExpertMask("m", {4: ()}), num_layers=4, num_experts=8)
        with pytest.raises(ValueError, match="no routed experts"):
            validate_lora_expert_mask(ok, num_layers=4, num_experts=0)
        with pytest.raises(ValueError, match="mask is for 40 layers"):
            validate_lora_expert_mask(LoRAExpertMask("m", {}, declared_num_layers=40), num_layers=4, num_experts=8)
        with pytest.raises(ValueError, match="mask is for 256 experts"):
            validate_lora_expert_mask(LoRAExpertMask("m", {}, declared_num_experts=256), num_layers=4, num_experts=8)
        with pytest.raises(ValueError, match="layer 1 lists experts, but it has no routed experts"):
            validate_lora_expert_mask(LoRAExpertMask("m", {1: (2,)}), num_layers=4, num_experts=8, moe_layers={0, 2})

    @pytest.mark.parametrize("kwargs,match", [
        (dict(ep_size=2), "ep_size > 1"),
        (dict(selective_expert_dequant=True), "qlora_selective_expert_dequant"),
        (dict(target_modules=["gate_up_proj", "down_proj"]), "gate_up_proj"),
        (dict(lora=False), "requires lora: true"),
        (dict(rollout_parity=True), "rollout parity"),
    ])
    def test_refused_options(self, kwargs, match):
        args = dict(lora=True, target_modules=WP1_TARGETS, ep_size=1, selective_expert_dequant=False)
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            check_lora_expert_mask_options(LoRAExpertMask("none"), **args)
        check_lora_expert_mask_options(LoRAExpertMask("none"), lora=True, target_modules=WP1_TARGETS, ep_size=1,
                                       selective_expert_dequant=False)


class TestCounts:
    """The adapter geometry of the ESFT arms (design section 2.3) from the checkpoint config alone."""

    def test_wp1_is_unchanged(self):
        assert qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=WP1_TARGETS,
                                       train_router=False) == (1_265_008_640, 480)

    @pytest.mark.skipif(not RELEASE_CONFIG.is_file(), reason="release config not on this machine")
    def test_wp1_from_the_release_config(self):
        config = json.loads(RELEASE_CONFIG.read_text())
        assert qwen3_5_moe_lora_counts(config, rank=16, target_modules=WP1_TARGETS,
                                       train_router=False) == (1_265_008_640, 480)

    def test_arms(self):
        a = qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=ARM_TARGETS, train_router=True)
        b = qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=ARM_TARGETS, train_router=True,
                                    mask=LoRAExpertMask("none"))
        assert a == (1_279_293_440, 860)
        assert b == (21_002_240, 620)
        rng = np.random.default_rng(0)
        layers = {layer: tuple(sorted(rng.choice(256, size=int(rng.integers(0, 30)), replace=False).tolist()))
                  for layer in range(40)}
        layers[7] = ()
        mask = LoRAExpertMask("c", layers)
        k_total = sum(len(v) for v in layers.values())
        nonempty = sum(1 for v in layers.values() if v)
        c = qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=ARM_TARGETS, train_router=True,
                                    mask=mask)
        assert c == (21_002_240 + 122_880 * k_total, 620 + 6 * nonempty)

    def test_all_targets(self):
        everything = qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=["all"], train_router=True)
        assert everything == qwen3_5_moe_lora_counts(SUROGATE37_TEXT, rank=16, target_modules=ARM_TARGETS,
                                                     train_router=True)


def test_binding_carries_the_mask():
    sg = pytest.importorskip("surogate._surogate")
    cfg = sg.LoRAAdapterConfig(rank=8, alpha=16, target_modules=WP1_TARGETS, train_router=True)
    assert cfg.expert_mask is None and cfg.expert_mask_source == ""
    assert "expert_mask" not in repr(cfg)  # the default repr is unchanged
    cfg.set_expert_mask([[1, 5], [], [7]], "m.json")
    assert cfg.expert_mask == [[1, 5], [], [7]] and cfg.expert_mask_source == "m.json"
    assert "expert_mask='m.json'" in repr(cfg)
    cfg.set_expert_mask([], "none")
    assert cfg.expert_mask == []
    cfg.clear_expert_mask()
    assert cfg.expert_mask is None
    for bad in ([[5, 1]], [[1, 1]], [[-1]]):
        with pytest.raises(ValueError, match="sorted, unique, non-negative"):
            cfg.set_expert_mask(bad, "bad")


# ---------------------------------------------------------------------------------------------- GPU

E = 8
LAYERS = 4
RANK = 8
SEQ = 128
MASK = [[1, 5], [], [7], [0, 2, 3]]  # the design's test mask
OTHER_MASK = [[1, 6], [], [7], [0, 2, 3]]  # same sizes, other ids


def tiny_config():
    return dict(architectures=["Qwen3_5MoeForCausalLM"], model_type="qwen3_5_moe_text",
                hidden_size=128, intermediate_size=256, num_hidden_layers=LAYERS, num_attention_heads=2,
                num_key_value_heads=1, head_dim=128, vocab_size=512, max_position_embeddings=4096,
                rms_norm_eps=1e-6, tie_word_embeddings=False, linear_num_key_heads=2, linear_key_head_dim=32,
                linear_num_value_heads=4, linear_value_head_dim=64, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"], mtp_num_hidden_layers=1, eos_token_id=1,
                num_experts=E, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                rope_parameters=dict(rope_type="default", rope_theta=1e7, partial_rotary_factor=.5,
                                     mrope_section=[11, 11, 10], mrope_interleaved=True))


def _make_trainer(model_dir, *, mask=None, targets=("all",), train_router=True, chunks=1, recipe="bf16"):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    opts = sg.RuntimeOptions(recipe=recipe, use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    opts.recompute = "true"
    if chunks > 1:
        opts.sequence_chunks = chunks
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora = sg.LoRAAdapterConfig(rank=RANK, alpha=2 * RANK, dropout=0, dtype="bf16", target_modules=list(targets),
                                train_router=train_router)
    if mask is not None:
        lora.set_expert_mask(mask, "none" if mask == [] else "other-mask" if mask == OTHER_MASK else "test-mask")
    trainer = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                                 options=opts, batch_size=1, seq_len=SEQ // chunks, grad_accum=1,
                                 memcpy_all_gather=True, memcpy_send_recv=True, lora_config=lora, qlora_config=None)
    trainer.init_weights()
    return trainer


def _tokens(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.integers(3, 500, size=(1, SEQ), dtype=np.int32)
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
    return x, y


def _numpy(tensors):
    import torch

    return {k: torch.from_dlpack(v).float().cpu().numpy().copy() for k, v in tensors.items()}


def _expert_key(name):
    """(layer, expert, rest) of an exported routed-expert adapter key, else None."""
    import re

    m = re.search(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.lora_([AB])\.weight$", name)
    return (int(m[1]), int(m[2]), m[3], m[4]) if m else None


def _live_adapter(init, mask):
    """A nonzero adapter over `init`'s keys; routed experts outside `mask` get A = B = 0."""
    import torch

    g = torch.Generator().manual_seed(20261018)
    live = {}
    for k in sorted(init):
        v = init[k]
        t = (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
        key = _expert_key(k)
        if key is not None and mask is not None and key[1] not in mask[key[0]]:
            t = torch.zeros_like(t)
        live[k] = t
    return live


# Scenarios: run in a fresh process (`python tests/test_lora_expert_mask.py <scenario> <model_dir> <out_dir>`).

def scenario_counts(model_dir, out):
    result = {}
    for name, mask, targets, router in (("all", None, ("all",), True), ("none", [], ("all",), True),
                                        ("mask", MASK, ("all",), True), ("wp1_mask", MASK, WP1_TARGETS, False),
                                        ("wp1", None, WP1_TARGETS, False)):
        trainer = _make_trainer(model_dir, mask=mask, targets=targets, train_router=router)
        weights = trainer.get_lora_weights(0)
        shapes = {k: list(v.shape) for k, v in _numpy(weights).items()}
        result[name] = {"params": int(sum(np.prod(s) for s in shapes.values())), "tensors": len(shapes),
                        "shapes": shapes}
        del trainer
    return result


def scenario_export_import(model_dir, out):
    from safetensors.torch import load_file

    out = Path(out)
    result = {}
    for name, mask in (("dense", None), ("mask", MASK), ("other", OTHER_MASK), ("none", [])):
        trainer = _make_trainer(model_dir, mask=mask)
        trainer.export_adapter(str(out / name), str(model_dir))
        del trainer
    # The masked trainer loads its own adapter (and a checkpoint round trip: save, load).
    trainer = _make_trainer(model_dir, mask=MASK)
    trainer.import_adapter(str(out / "mask" / "adapter_model.safetensors"))
    x, y = _tokens()
    trainer.step(x, y)
    from surogate import _surogate as sg

    trainer.update_with_config(sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=1e-3), 1)
    trainer.save_checkpoint(str(out / "ckpt"), 1)
    del trainer
    trainer = _make_trainer(model_dir, mask=MASK)
    trainer.load_checkpoint(str(out / "ckpt"), 1)
    result["checkpoint_resume_ok"] = True
    del trainer

    def refused(mask, adapter):
        trainer = _make_trainer(model_dir, mask=mask)
        try:
            trainer.import_adapter(str(adapter))
        except Exception as exc:  # noqa: BLE001 -- the message is what is checked
            return str(exc)
        finally:
            del trainer
        return None

    result["other_into_mask"] = refused(MASK, out / "other" / "adapter_model.safetensors")
    result["mask_into_all"] = refused(None, out / "mask" / "adapter_model.safetensors")
    result["all_into_mask"] = refused(MASK, out / "dense" / "adapter_model.safetensors")
    trainer = _make_trainer(model_dir, mask=OTHER_MASK)
    try:
        trainer.load_checkpoint(str(out / "ckpt"), 1)
        result["ckpt_into_other"] = None
    except Exception as exc:  # noqa: BLE001
        result["ckpt_into_other"] = str(exc)
    del trainer
    result["files"] = {n: str(out / n) for n in ("dense", "mask", "other", "none")}
    _ = load_file  # imported here so a missing safetensors fails the scenario, not the parent
    return result


def scenario_equivalence(model_dir, out):
    """Masked vs dense (unselected experts A = B = 0); run with SUROGATE_FUSED_EXPERT_LORA=0."""
    from safetensors.torch import load_file, save_file
    from surogate import _surogate as sg

    out = Path(out)
    x, y = _tokens()
    valid = y[0] != -100
    dense = _make_trainer(model_dir)
    dense.export_adapter(str(out / "dense_init"), str(model_dir))
    init = load_file(str(out / "dense_init" / "adapter_model.safetensors"))
    live = _live_adapter(init, MASK)
    save_file(live, str(out / "dense_live.safetensors"))
    masked = _make_trainer(model_dir, mask=MASK)
    masked.export_adapter(str(out / "mask_init"), str(model_dir))
    masked_keys = load_file(str(out / "mask_init" / "adapter_model.safetensors")).keys()
    save_file({k: live[k] for k in masked_keys}, str(out / "mask_live.safetensors"))
    del masked
    optimizer = sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=1e-3, weight_decay=0.0, grad_clip=0.0)
    arrays = {}
    # dense2 repeats the dense run: the run-to-run noise floor the masked run is held to.
    for name, mask in (("dense", None), ("dense2", None), ("mask", MASK)):
        live_name = "mask" if name == "mask" else "dense"
        trainer = dense if name == "dense" else _make_trainer(model_dir, mask=mask)
        trainer.import_adapter(str(out / f"{live_name}_live.safetensors"))
        arrays[f"{name}/logprobs"] = trainer.compute_logprobs(x, y, use_lora=True)[0][valid].astype(np.float64)
        arrays[f"{name}/base_logprobs"] = trainer.compute_logprobs(x, y, use_lora=False)[0][valid].astype(np.float64)
        trainer.step(x, y)
        for k, v in _numpy(trainer.get_lora_gradients(0)).items():
            arrays[f"{name}/grad/{k}"] = v
        trainer.update_with_config(optimizer, 1)
        for k, v in _numpy(trainer.get_lora_weights(0)).items():
            arrays[f"{name}/weight/{k}"] = v
        del trainer
        dense = None
    np.savez(out / "equivalence.npz", **arrays)
    return {"npz": str(out / "equivalence.npz")}


def scenario_chunked(model_dir, out):
    """Masked training, sequence_chunks 4 against 1: loss and LoRA gradients."""
    from safetensors.torch import load_file, save_file
    from surogate import _surogate as sg

    out = Path(out)
    x, y = _tokens(11)
    trainer = _make_trainer(model_dir, mask=MASK)
    trainer.export_adapter(str(out / "init"), str(model_dir))
    live = _live_adapter(load_file(str(out / "init" / "adapter_model.safetensors")), None)
    save_file(live, str(out / "live.safetensors"))
    del trainer
    flush = sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, weight_decay=0.0, grad_clip=0.0)
    arrays = {}
    for chunks in (1, 4):
        trainer = _make_trainer(model_dir, mask=MASK, chunks=chunks)
        trainer.import_adapter(str(out / "live.safetensors"))
        trainer.step(x, y)
        for k, v in _numpy(trainer.get_lora_gradients(0)).items():
            arrays[f"{chunks}/grad/{k}"] = v
        arrays[f"{chunks}/loss"] = np.array([trainer.update_with_config(flush, 1)["loss"]], dtype=np.float64)
        del trainer
    np.savez(out / "chunked.npz", **arrays)
    return {"npz": str(out / "chunked.npz")}


def scenario_validation(model_dir, out):
    result = {}
    for name, mask in (("expert_ge_E", [[1, E]]), ("layer_ge_L", [[1]] * LAYERS + [[]])):
        try:
            trainer = _make_trainer(model_dir, mask=mask)
            del trainer
            result[name] = None
        except Exception as exc:  # noqa: BLE001
            result[name] = str(exc)
    return result


def scenario_fingerprint(model_dir, out, recipe="bf16", train_router="1"):
    """Default (no mask) numerics, for comparing two builds bit for bit: log-probs, LoRA gradients and the
    adapter after one AdamW 8-bit step. Uses no API the baseline build lacks."""
    import hashlib

    from safetensors.torch import load_file, save_file
    from surogate import _surogate as sg

    out = Path(out)
    router = train_router == "1"
    x, y = _tokens()
    trainer = _make_trainer(model_dir, train_router=router, recipe=recipe)
    trainer.export_adapter(str(out / "fp_init"), str(model_dir))
    init = load_file(str(out / "fp_init" / "adapter_model.safetensors"))
    save_file(_live_adapter(init, None), str(out / "fp_live.safetensors"))
    trainer.import_adapter(str(out / "fp_live.safetensors"))
    digest, saved = {}, {}

    def h(name, arrays):
        m = hashlib.sha256()
        for k in sorted(arrays):
            m.update(k.encode())
            m.update(np.ascontiguousarray(arrays[k]).tobytes())
            saved[f"{name}/{k}"] = arrays[k]
        digest[name] = m.hexdigest()

    h("init", {k: v.float().numpy() for k, v in init.items()})
    h("logprobs", {"lp": np.asarray(trainer.compute_logprobs(x, y, use_lora=True)[0])})
    trainer.step(x, y)
    h("grads", _numpy(trainer.get_lora_gradients(0)))
    result = trainer.update_with_config(sg.OptimizerConfig(optimizer="adamw_8bit", learning_rate=1e-3), 1)
    digest["loss"] = repr(float(result["loss"]))
    digest["norm"] = repr(float(result["norm"]))
    h("weights", _numpy(trainer.get_lora_weights(0)))
    np.savez(out / "fingerprint.npz", **saved)
    return digest


SCENARIOS = {f.__name__[len("scenario_"):]: f for f in (scenario_counts, scenario_export_import, scenario_equivalence,
                                                         scenario_chunked, scenario_validation, scenario_fingerprint)}


@pytest.fixture(scope="module")
def tiny_moe(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    try:
        import surogate._surogate  # noqa: F401
    except ImportError:
        pytest.skip("surogate._surogate C++ extension not built")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    model = tmp_path_factory.mktemp("tiny_qwen3_5_moe")
    (model / "config.json").write_text(json.dumps(tiny_config()))
    return model


def run_scenario(name, model_dir, *, env=None, args=()):
    out = Path(tempfile.mkdtemp(prefix=f"lora_mask_{name}_"))
    root = Path(__file__).resolve().parent.parent
    full_env = dict(os.environ)
    full_env.update(env or {})
    full_env["SUROGATE_ROPE_MAX_SEQ"] = str(SEQ)
    # The script's own directory is tests/; import this checkout's package, as the parent does.
    full_env["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), full_env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), name, str(model_dir), str(out), *args],
                          env=full_env, cwd=str(root), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AssertionError(f"scenario {name} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-8000:]}")
    return json.loads((out / "result.json").read_text())


def _counts_for(config, mask, targets, router):
    return qwen3_5_moe_lora_counts(config, rank=RANK, target_modules=targets, train_router=router,
                                   mask=None if mask is None else LoRAExpertMask("m", dict(enumerate(map(tuple, mask)))))


@pytest.mark.gpu
@pytest.mark.slow
def test_counts_match_the_formula(tiny_moe):
    result = run_scenario("counts", tiny_moe)
    config = tiny_config()
    for name, mask, targets, router in (("all", None, ["all"], True), ("none", [], ["all"], True),
                                        ("mask", MASK, ["all"], True), ("wp1_mask", MASK, WP1_TARGETS, False),
                                        ("wp1", None, WP1_TARGETS, False)):
        assert (result[name]["params"], result[name]["tensors"]) == _counts_for(config, mask, targets, router), name
    # Compact expert tensors: [K_l, r, in] / [K_l, out, r]; no grouped expert tensor in a layer with K_l = 0.
    shapes = result["mask"]["shapes"]
    for layer, ids in enumerate(MASK):
        key = f"base_model.model.model.layers.{layer}.mlp.experts.gate_proj.lora_A.weight"
        if ids:
            assert shapes[key] == [len(ids), RANK, 128]
            assert shapes[key.replace("gate_proj.lora_A", "down_proj.lora_B")] == [len(ids), 128, RANK]
        else:
            assert key not in shapes
    assert not any(".mlp.experts." in k for k in result["none"]["shapes"])
    assert all(v[0] == E for k, v in result["all"]["shapes"].items() if ".mlp.experts." in k)


@pytest.mark.gpu
@pytest.mark.slow
def test_masked_equals_dense_with_zeroed_experts(tiny_moe):
    result = run_scenario("equivalence", tiny_moe, env={"SUROGATE_FUSED_EXPERT_LORA": "0"})
    data = np.load(result["npz"])
    lp_d, lp_d2, lp_m = data["dense/logprobs"], data["dense2/logprobs"], data["mask/logprobs"]
    effect = float(np.abs(lp_d - data["dense/base_logprobs"]).mean())

    def rel(a, b):
        return float(np.linalg.norm((a - b).ravel()) / max(float(np.linalg.norm(b.ravel())), 1e-30))

    def same(masked, dense, dense2, what):
        """Masked == dense to 1e-6 relative, or within twice the dense run's own run-to-run noise when the
        dense run itself is not bit-reproducible."""
        floor = rel(dense2, dense)
        assert rel(masked, dense) <= max(1e-6, 2.0 * floor), (what, rel(masked, dense), floor)
        return floor

    report = {"adapter_effect": effect, "logprob_max_abs_diff": float(np.abs(lp_d - lp_m).max()),
              "logprob_dense_rerun_max_abs_diff": float(np.abs(lp_d - lp_d2).max())}
    assert effect > 1e-3  # the adapter moves the model
    if np.array_equal(lp_d, lp_d2):
        np.testing.assert_allclose(lp_m, lp_d, rtol=1e-6, atol=0)
    else:
        same(lp_m, lp_d, lp_d2, "logprobs")
    for kind in ("grad", "weight"):
        def part(run, kind=kind):
            return {k.split("/", 2)[2]: data[k] for k in data.files if k.startswith(f"{run}/{kind}/")}

        dense, dense2, masked = part("dense"), part("dense2"), part("mask")
        assert set(masked) == {k for k in dense if not (".layers.1.mlp.experts." in k)}, kind
        compared, floors = 0, []
        for k, m in masked.items():
            d, d2 = dense[k], dense2[k]
            if ".mlp.experts." in k:
                layer = int(k.split(".layers.")[1].split(".")[0])
                ids = MASK[layer]
                assert m.shape[0] == len(ids) and d.shape[0] == E, k
                floors.append(same(m, d[ids], d2[ids], f"{kind} {k}"))
                if kind == "grad":  # unselected experts (A = B = 0) get no gradient in the dense run
                    others = [e for e in range(E) if e not in ids]
                    assert not np.any(d[others]), k
            else:
                floors.append(same(m, d, d2, f"{kind} {k}"))
            compared += 1
        report[f"{kind}_dense_rerun_worst_relL2"] = max(floors)
        report[f"{kind}_masked_bit_identical"] = all(
            np.array_equal(m, dense[k][MASK[int(k.split(".layers.")[1].split(".")[0])]] if ".mlp.experts." in k
                           else dense[k]) for k, m in masked.items())
        assert compared > 20, compared
    print(json.dumps(report))


@pytest.mark.gpu
@pytest.mark.slow
def test_export_import_and_refusals(tiny_moe):
    from safetensors.torch import load_file

    result = run_scenario("export_import", tiny_moe)
    files = {n: Path(p) for n, p in result["files"].items()}
    dense = load_file(str(files["dense"] / "adapter_model.safetensors"))
    masked = load_file(str(files["mask"] / "adapter_model.safetensors"))
    # Exactly the selected experts, under their global ids, with the HF/PEFT per-expert shapes.
    experts = {_expert_key(k) for k in masked if _expert_key(k)}
    expect = {(layer, e, proj, ab) for layer, ids in enumerate(MASK) for e in ids
              for proj in ("gate_proj", "up_proj", "down_proj") for ab in "AB"}
    assert experts == expect
    for k, v in masked.items():
        assert tuple(v.shape) == tuple(dense[k].shape), k
        # Gathered init: a selected expert's lora_A is the dense run's (same seed and subsequence); B = 0.
        assert np.array_equal(v.float().numpy(), dense[k].float().numpy()), k
    assert {k for k in dense if not _expert_key(k)} == {k for k in masked if not _expert_key(k)}
    cfg = json.loads((files["mask"] / "adapter_config.json").read_text())
    rec = cfg["surogate_expert_mask"]
    assert rec["format"] == MASK_FORMAT and rec["num_layers"] == LAYERS and rec["num_experts"] == E
    assert rec["layers"] == {str(i): ids for i, ids in enumerate(MASK)} and rec["source"] == "test-mask"
    assert "surogate_expert_mask" not in json.loads((files["dense"] / "adapter_config.json").read_text())
    # The recorded mask is itself a valid mask file.
    assert parse_lora_expert_mask(rec).per_layer() == MASK
    # `none`: shared-expert, attention, linear-attention and router adapters, no routed expert.
    none = load_file(str(files["none"] / "adapter_model.safetensors"))
    assert not any(".mlp.experts." in k for k in none)
    for part in (".mlp.shared_experts.up_proj.", ".mlp.shared_experts.down_proj.", ".self_attn.q_proj.",
                 ".linear_attn.in_proj_qkv.", ".linear_attn.out_proj.", ".mlp.gate.lora_"):
        assert any(part in k for k in none), part
    assert {k for k in none} == {k for k in dense if not _expert_key(k)}
    # Imports: the own mask loads (and a checkpoint resumes); everything else is refused naming the masks.
    assert result["checkpoint_resume_ok"]
    assert result["other_into_mask"] and "lora_expert_mask 'test-mask'" in result["other_into_mask"]
    assert "lacks" in result["other_into_mask"] and "surogate_expert_mask 'other-mask'" in result["other_into_mask"]
    assert result["mask_into_all"] and "surogate_expert_mask 'test-mask'" in result["mask_into_all"]
    assert "every routed expert (no lora_expert_mask)" in result["mask_into_all"]
    assert result["all_into_mask"] and "outside this trainer's lora_expert_mask 'test-mask'" in result["all_into_mask"]
    assert result["ckpt_into_other"] and "lora_expert_mask" in result["ckpt_into_other"]


@pytest.mark.gpu
@pytest.mark.slow
def test_trainer_refuses_invalid_masks(tiny_moe):
    result = run_scenario("validation", tiny_moe)
    assert result["expert_ge_E"] and f"lists expert {E}" in result["expert_ge_E"]
    assert result["layer_ge_L"] and f"lists layer {LAYERS}, but the model has {LAYERS} layers" in result["layer_ge_L"]


@pytest.mark.gpu
@pytest.mark.slow
def test_chunked_equals_unchunked(tiny_moe):
    result = run_scenario("chunked", tiny_moe)
    data = np.load(result["npz"])
    loss1, loss4 = float(data["1/loss"][0]), float(data["4/loss"][0])
    keys = sorted(k.split("/", 2)[2] for k in data.files if k.startswith("1/grad/"))
    assert keys == sorted(k.split("/", 2)[2] for k in data.files if k.startswith("4/grad/"))
    report = {}
    for k in keys:
        a, b = data[f"1/grad/{k}"].ravel(), data[f"4/grad/{k}"].ravel()
        ref = max(float(np.linalg.norm(a)), 1e-12)
        cos = float(a @ b / (ref * max(float(np.linalg.norm(b)), 1e-12)))
        report[k] = (float(np.linalg.norm(a - b)) / ref, cos)
    worst = max(report, key=lambda k: report[k][0])
    print(json.dumps({"loss": [loss1, loss4], "worst_relL2": [worst, report[worst]],
                      "linear_attn_worst": max(v[0] for k, v in report.items() if ".linear_attn." in k)}))
    assert any(".linear_attn." in k for k in keys) and any(".mlp.experts." in k for k in keys)
    assert loss4 == pytest.approx(loss1, rel=1e-2)
    # The GDN LoRA test's bar against its fp32 reference (cosine > 0.98, relative L2 <= 0.1).
    for k, (rel, cos) in report.items():
        assert cos > 0.98 and rel <= 0.1, (k, rel, cos)


if __name__ == "__main__":
    scenario, model_dir, out_dir = sys.argv[1:4]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = SCENARIOS[scenario](Path(model_dir), out, *sys.argv[4:])
    (out / "result.json").write_text(json.dumps(res))
