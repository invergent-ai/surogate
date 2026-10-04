"""The graph compiler gives every output that is read a real shape (#235).

An op output in a Mapped slot whose op has no shape rule in GraphCompiler::compile takes the
{B, T, C} default. When a later op (or the backward) reads it, or a backward op writes it, the
default is a wrong shape, or at best a lost planned buffer: the dispatch falls back to a runtime
temp, a persistent fallback allocation under CUDA-graph capture. The compiler refuses those, and
keeps a warning for outputs nothing reads.

This compiles the forward and backward graphs of every onboarding architecture, from the model's
own config cut to a few layers, and checks that only discarded outputs fall back. It needs a GPU
(the trainer allocates the weights) but no checkpoint: nothing runs, so the weights stay random.

    pytest tests/test_graph_shape_fallbacks.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.dsl.ir_builder import build_dsl_ir_for_model

pytestmark = [pytest.mark.gpu]

# The HF config.json of each onboarding test's model (tests/test_onboarding_<name>.py).
CONFIGS = Path(__file__).parent / "fixtures" / "onboarding_configs"

# Layers kept: the onboarding test's own cut where it has one, else enough for every kind of
# layer the model mixes.
LAYERS = {
    "gemma4": 10,  # the last 5 share their KV, as the last 20 of 35 do
    "gemma4_unified": 6,
    "gpt_oss_moe": 2,
    "laguna": 5,
    "lfm2": 4,
    "nemotron_h": 6,
    "qwen3": 4,
    "qwen3_5": 4,
    "qwen3_moe": 4,
    "qwen3_vl": 2,
}
VOCAB = 4096
EXPERTS = 8
BATCH = 1
SEQ_LEN = 64

PER_LAYER_LISTS = ("layer_types", "mlp_layer_types", "gating_types", "num_attention_heads_per_layer")


def mini_config(name: str) -> dict:
    """The model's config with fewer layers, a small vocabulary and few experts."""
    config = json.loads((CONFIGS / f"{name}.json").read_text())
    config.pop("quantization_config", None)  # the weights are bf16 here
    text = config.get("text_config", config)
    layers = LAYERS[name]
    text["num_hidden_layers"] = layers
    for key in PER_LAYER_LISTS:
        if isinstance(text.get(key), list):
            text[key] = text[key][:layers]
    if "hybrid_override_pattern" in text:
        text["hybrid_override_pattern"] = text["hybrid_override_pattern"][:layers]
    if "full_attn_idxs" in text:
        text["full_attn_idxs"] = [i for i in text["full_attn_idxs"] if i < layers]
    if text.get("num_kv_shared_layers"):
        text["num_kv_shared_layers"] = layers // 2
    for key in ("vocab_size", "vocab_size_per_layer_input"):
        if text.get(key):
            text[key] = VOCAB
    for key in ("num_experts", "num_local_experts", "n_routed_experts"):
        if text.get(key):
            text[key] = EXPERTS
    return config


def has_gpu() -> bool:
    try:
        return len(_surogate.SystemInfo.get_gpu_info()) > 0
    except Exception:
        return False


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_only_discarded_outputs_take_the_shape_default(name, tmp_path, monkeypatch):
    if not has_gpu():
        pytest.skip("needs a GPU")
    # Let the compile through so every fallback of both graphs is listed, not just the first
    # graph's.
    monkeypatch.setenv("SUROGATE_ALLOW_SHAPE_FALLBACK", "1")
    (tmp_path / "config.json").write_text(json.dumps(mini_config(name)))

    opts = _surogate.RuntimeOptions(recipe="bf16", use_cuda_graphs=False)
    opts.dsl_ir_json = build_dsl_ir_for_model(str(tmp_path))
    trainer = _surogate.SurogateTrainer(
        ngpu=1,
        config=_surogate.PretrainedConfig.from_pretrained(str(tmp_path), "bf16"),
        options=opts,
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        grad_accum=1,
        memcpy_all_gather=False,
        memcpy_send_recv=False,
        lora_config=None,
        qlora_config=None,
    )
    try:
        fallbacks = trainer.get_debug_shape_fallbacks()
    finally:
        del trainer

    refused = [f for f in fallbacks if f["consumed"] or (f["graph"] == "backward" and f["output"])]
    assert not refused, "outputs without a shape rule:\n" + "\n".join(
        f"  {f['graph']} op {f['op_id']} ({f['op_type']}) output[{f['output_index']}] {f['output']}"
        + (" (read)" if f["consumed"] else "")
        for f in refused
    )
