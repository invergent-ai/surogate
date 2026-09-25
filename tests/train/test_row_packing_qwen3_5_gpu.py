"""Packed documents on Qwen3.5 linear-attention (Gated DeltaNet) layers (one GPU, ~1-2 min).

A Qwen3.5 linear-attention block mixes tokens twice outside attention: a causal depthwise
convolution (kernel 4) over the projected q/k/v, and the gated delta rule's recurrent state,
which the chunked kernels carry from one 64-token chunk to the next. Row packing and sample
packing mark documents only through position ids (attention's document masking); before this
test's fix both token mixers ignored them, so every packed document started from the previous
document's convolution tail and recurrent state.

* Document isolation, bit for bit: on a model of linear-attention layers only, every packed
  document's per-token log-probs equal the same document alone at the start of an unpacked row
  (which takes the dense convolution kernel and the whole-row delta-rule pipeline). Documents
  of 2, 3, 64 and 65 tokens exercise the convolution's first taps and the chunk edges.
* Hybrid model (three linear-attention layers and one full-attention layer): the same, with
  the lone document also document-masked (its padding its own document), so both layouts take
  the same attention kernel and only the linear-attention path is under test.
* Training step on the linear-attention model: the same rows padded and packed give
  bit-identical per-row losses, gradients no further from padded than padded training is from
  itself with its micro-steps reordered, and the padded step run twice is bit-identical.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

T = 2048
K = 8
LENGTHS = [700, 3, 64, 900, 65, 2, 37]  # 1,771 tokens: chunk-aligned and unaligned starts, conv-edge rows


def _config(layer_types):
    return dict(
        architectures=["Qwen3_5ForCausalLM"],
        model_type="qwen3_5_text",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        vocab_size=1024,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_num_value_heads=4,  # two value heads per key head, as Qwen3.8-27B has three
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
        eos_token_id=1,
        pad_token_id=0,
        rope_parameters=dict(
            rope_type="default", rope_theta=1e7, partial_rotary_factor=0.25, mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ),
    )


def _write_fixture(out, layer_types, seed):
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        pytest.skip("transformers without Qwen3.5")
    cfg = _config(layer_types)
    torch.manual_seed(seed)
    model = transformers.Qwen3_5ForCausalLM(transformers.Qwen3_5TextConfig(**cfg)).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and not name.endswith(("A_log", "dt_bias")):
                p.copy_(0.1 * torch.randn_like(p, dtype=torch.float32).to(p.dtype))
    model.save_pretrained(out, safe_serialization=True)
    (out / "config.json").write_text(json.dumps({**json.loads((out / "config.json").read_text()), **cfg}))
    return str(out)


@pytest.fixture(scope="module")
def linear_only(tmp_path_factory):
    return _write_fixture(tmp_path_factory.mktemp("qwen35-linear"), ["linear_attention"] * 3, 351)


@pytest.fixture(scope="module")
def hybrid(tmp_path_factory):
    return _write_fixture(
        tmp_path_factory.mktemp("qwen35-hybrid"),
        ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        352,
    )


def make_trainer(fixture, adapter=None):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = sg.RuntimeOptions(
        recipe="bf16",
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_gradients=True,
    )
    opts.recompute = "true"
    opts.dsl_ir_json = build_dsl_ir_for_model(fixture)
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    tr = sg.SurogateTrainer(
        ngpu=1,
        config=sg.PretrainedConfig.from_pretrained(fixture, "bf16"),
        options=opts,
        batch_size=1,
        seq_len=T,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
        lora_config=sg.LoRAAdapterConfig(rank=8, alpha=8, dropout=0, dtype="bf16", target_modules=["all"]),
        qlora_config=None,
    )
    tr.import_weights(get_model_weights_path(fixture))
    if adapter:
        tr.import_adapter(adapter)
    return tr


def nonzero_adapter(fixture, out):
    """Non-zero A and B so every LoRA tensor gets a gradient from step 0."""
    from safetensors.torch import load_file, save_file

    tr = make_trainer(fixture)
    tr.export_adapter(str(out), fixture)
    del tr
    gc.collect()
    init = load_file(out / "adapter_model.safetensors")
    g = torch.Generator().manual_seed(20260925)
    live = {
        k: (torch.randn(v.shape, generator=g) * (0.02 if "lora_A" in k else 0.005)).to(v.dtype) for k, v in init.items()
    }
    path = out / "nonzero.safetensors"
    save_file(live, path)
    return str(path)


@pytest.fixture(scope="module")
def linear_adapter(linear_only, tmp_path_factory):
    return nonzero_adapter(linear_only, tmp_path_factory.mktemp("adapter-linear"))


def rows():
    rng = np.random.default_rng(2026)
    out = []
    for n in LENGTHS:
        cands = rng.choice(np.arange(10, 1000), size=4, replace=False).astype(np.int32)
        toks = rng.integers(3, 1000, size=n + 1).astype(np.int32)
        toks[-1] = cands[int(rng.integers(0, 4))]
        out.append({"tokens": toks, "cands": cands})
    return out


def window(rs, idxs, tail_document=False):
    """The rows as documents of one sequence (positions restart at 0); the tail its own document.

    A lone row is one dense document with its padding (no document boundary at all) unless
    tail_document is set."""
    x = np.zeros((1, T), np.int32)
    y = np.full((1, T), -100, np.int32)
    p = np.zeros((1, T), np.int32)
    ids = np.zeros((1, T, K), np.int32)
    o, sup = 0, {}
    for i in idxs:
        n = len(rs[i]["tokens"]) - 1
        x[0, o : o + n] = rs[i]["tokens"][:n]
        p[0, o : o + n] = np.arange(n)
        y[0, o + n - 1] = rs[i]["tokens"][-1]
        ids[0, o + n - 1] = -1
        ids[0, o + n - 1, :4] = rs[i]["cands"]
        sup[i] = o + n - 1
        o += n
    p[0, o:] = np.arange(T - o) if (len(idxs) > 1 or tail_document) else np.arange(o, T)
    return x, y, p, ids, sup


def full_targets(x, n):
    """Next-token targets for the first n - 1 positions."""
    return np.where(np.arange(T) < n - 1, np.roll(x, -1, axis=1), -100).astype(np.int32)


def _isolation(fixture, tail_document):
    rs = rows()
    tr = make_trainer(fixture)
    x, _, p, _, _ = window(rs, range(len(rs)))
    total = sum(len(r["tokens"]) - 1 for r in rs)
    packed = np.asarray(tr.compute_logprobs(x, full_targets(x, total), True, p))[0]
    o = 0
    for i, r in enumerate(rs):
        n = len(r["tokens"]) - 1
        xa, _, pa, _, _ = window(rs, [i], tail_document=tail_document)
        alone = np.asarray(tr.compute_logprobs(xa, full_targets(xa, n), True, pa))[0]
        np.testing.assert_array_equal(packed[o : o + n - 1], alone[: n - 1], err_msg=f"document {i} (n={n}) leaked")
        o += n
    del tr
    gc.collect()


def test_linear_attention_documents_match_each_document_alone(linear_only):
    _isolation(linear_only, tail_document=False)


def test_hybrid_documents_match_each_document_alone(hybrid):
    _isolation(hybrid, tail_document=True)


def kd_step(tr, x, y, p, ids):
    tr.step_with_kd(
        x, y, ids, np.zeros(ids.shape, np.float32), position_ids=p, top_k=K, temperature=1.0, kd_weight=1.0,
        ce_weight=0.0, candidate_only=True,
    )


def grads(tr):
    return {k: torch.from_dlpack(v).double().cpu().numpy() for k, v in tr.get_lora_gradients(0).items()}


def rel_l2(a, b):
    return float(np.sqrt(sum(np.square(a[k] - b[k]).sum() for k in a) / sum(np.square(a[k]).sum() for k in a)))


def train_step(tr, rs, layout):
    if layout == "packed":
        batches = [list(range(len(rs)))]
    elif layout == "reversed":
        batches = [[i] for i in reversed(range(len(rs)))]
    else:
        batches = [[i] for i in range(len(rs))]
    tr.set_grad_accumulation(len(batches))
    losses = {}
    for b in batches:
        x, y, p, ids, sup = window(rs, b)
        kd_step(tr, x, y, p, ids)
        buf = np.asarray(tr.get_token_losses(0))
        losses.update({i: float(buf[at]) for i, at in sup.items()})
    g = grads(tr)
    tr.get_kd_loss()
    return losses, g


def test_linear_attention_training_step_packed_equals_padded(linear_only, linear_adapter):
    rs = rows()
    tr = make_trainer(linear_only, adapter=linear_adapter)
    results = {layout: train_step(tr, rs, "padded" if layout == "padded_again" else layout)
               for layout in ("padded", "padded_again", "reversed", "packed")}
    del tr
    gc.collect()
    padded, again, reversed_, packed = (results[k] for k in ("padded", "padded_again", "reversed", "packed"))
    assert padded[0] == again[0]
    assert all(np.array_equal(padded[1][k], again[1][k]) for k in padded[1])
    assert packed[0] == padded[0], {i: (padded[0][i], packed[0][i]) for i in padded[0] if packed[0][i] != padded[0][i]}
    floor = rel_l2(padded[1], reversed_[1])
    assert rel_l2(padded[1], packed[1]) <= max(2.0 * floor, 1e-6), (rel_l2(padded[1], packed[1]), floor)
