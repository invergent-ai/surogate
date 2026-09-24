"""Packed rows vs the same rows padded, on a small random Gemma-4 MoE (one GPU, ~1 min).

The fixture keeps what packing has to get right about Gemma 4: sliding-window layers
(FlashAttention varlen, window shorter than the documents), a global layer whose 512-wide
heads take the mem-efficient kernel, grouped-query attention with K=V sharing, routed experts
with a dense MLP beside them, and the final-logit softcap.

* Document isolation: every packed document's per-token log-probs equal the document run
  alone, bit for bit (bf16).
* Offset invariance of the backward: a row's attention input/output gradients are the same
  bits whether the row starts its window or sits after another document. (Two engine fixes
  made this hold: the deterministic dQ reduction's split count no longer depends on the
  document count, and autodiff gradient accumulation no longer rounds stochastically by flat
  buffer index.)
* Training step: the same rows padded and packed give bit-identical per-row losses, and
  gradients no further from padded than padded training is from itself with its micro-steps
  reordered (the order in which bf16 gradient accumulation sums rows).
* Determinism: the padded step run twice is bit-identical.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

T = 512
K = 8
LENGTHS = [150, 37, 211, 90, 12]  # > the 64-token window, and short ones


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    if not hasattr(transformers, "Gemma4ForCausalLM"):
        pytest.skip("transformers without Gemma 4")
    out = tmp_path_factory.mktemp("gemma4-mini")
    cfg = transformers.Gemma4TextConfig(
        vocab_size=1024,
        hidden_size=512,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=128,
        global_head_dim=512,
        attention_k_eq_v=True,
        sliding_window=64,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=128,
        final_logit_softcapping=30.0,
        max_position_embeddings=2048,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=1024,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        rope_parameters={
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    )
    torch.manual_seed(771)
    model = transformers.Gemma4ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.1 * torch.randn_like(p, dtype=torch.float32).to(p.dtype))
    model.save_pretrained(out, safe_serialization=True)
    config = json.loads((out / "config.json").read_text())
    config.update(global_head_dim=512, num_global_key_value_heads=1)
    (out / "config.json").write_text(json.dumps(config))
    return str(out)


def make_trainer(fixture, recipe="bf16", adapter=None):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.utils.hf import get_model_weights_path

    opts = sg.RuntimeOptions(
        recipe=recipe,
        use_cuda_graphs=False,
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        offload_residual=False,
        shard_gradients=True,
    )
    opts.recompute = "true"
    opts.router_aux_loss_coef = 0.0
    opts.router_z_loss_coef = 0.0
    opts.dsl_ir_json = build_dsl_ir_for_model(fixture)
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


@pytest.fixture(scope="module")
def adapter(fixture_dir, tmp_path_factory):
    """Non-zero A and B so every LoRA tensor gets a gradient from step 0."""
    from safetensors.torch import load_file, save_file

    out = tmp_path_factory.mktemp("adapter")
    tr = make_trainer(fixture_dir)
    tr.export_adapter(str(out), fixture_dir)
    del tr
    gc.collect()
    init = load_file(out / "adapter_model.safetensors")
    g = torch.Generator().manual_seed(20260919)
    live = {
        k: (torch.randn(v.shape, generator=g) * (0.02 if "lora_A" in k else 0.005)).to(v.dtype) for k, v in init.items()
    }
    path = out / "nonzero.safetensors"
    save_file(live, path)
    return str(path)


def rows():
    """Candidate-CE rows: the answer (one of the candidates) follows the last input token."""
    rng = np.random.default_rng(2026)
    out = []
    for n in LENGTHS:
        cands = rng.choice(np.arange(10, 1000), size=4, replace=False).astype(np.int32)
        toks = rng.integers(3, 1000, size=n + 1).astype(np.int32)
        toks[-1] = cands[int(rng.integers(0, 4))]
        out.append({"tokens": toks, "cands": cands})
    return out


def window(rs, idxs, supervise=None):
    """One sequence of the given rows, each a document (positions restart at 0), tail its own document."""
    x = np.zeros((1, T), np.int32)
    y = np.full((1, T), -100, np.int32)
    p = np.zeros((1, T), np.int32)
    ids = np.zeros((1, T, K), np.int32)
    o, sup = 0, {}
    for i in idxs:
        n = len(rs[i]["tokens"]) - 1
        x[0, o : o + n] = rs[i]["tokens"][:n]
        p[0, o : o + n] = np.arange(n)
        if supervise is None or i in supervise:
            y[0, o + n - 1] = rs[i]["tokens"][-1]
            ids[0, o + n - 1] = -1
            ids[0, o + n - 1, :4] = rs[i]["cands"]
            sup[i] = o + n - 1
        o += n
    p[0, o:] = np.arange(T - o) if len(idxs) > 1 else np.arange(o, T)  # padded row: one dense document
    return x, y, p, ids, sup


def kd_step(tr, x, y, p, ids):
    tr.step_with_kd(
        x,
        y,
        ids,
        np.zeros(ids.shape, np.float32),
        position_ids=p,
        top_k=K,
        temperature=1.0,
        kd_weight=1.0,
        ce_weight=0.0,
        candidate_only=True,
    )


def grads(tr):
    return {k: torch.from_dlpack(v).double().cpu().numpy() for k, v in tr.get_lora_gradients(0).items()}


def rel_l2(a, b):
    return float(np.sqrt(sum(np.square(a[k] - b[k]).sum() for k in a) / sum(np.square(a[k]).sum() for k in a)))


def test_packed_documents_match_each_document_alone(fixture_dir, adapter):
    rs = rows()
    tr = make_trainer(fixture_dir, adapter=adapter)

    def full_targets(x, n):
        """Next-token targets for the first n - 1 positions."""
        return np.where(np.arange(T) < n - 1, np.roll(x, -1, axis=1), -100).astype(np.int32)

    x, _, p, _, _ = window(rs, range(len(rs)))
    total = sum(len(r["tokens"]) - 1 for r in rs)
    packed = np.asarray(tr.compute_logprobs(x, full_targets(x, total), True, p))[0]
    o = 0
    for i, r in enumerate(rs):
        n = len(r["tokens"]) - 1
        xa, _, pa, _, _ = window(rs, [i])
        alone = np.asarray(tr.compute_logprobs(xa, full_targets(xa, n), True, pa))[0]
        np.testing.assert_array_equal(packed[o : o + n - 1], alone[: n - 1], err_msg=f"document {i} leaked")
        o += n
    del tr
    gc.collect()


def test_row_attention_gradients_do_not_depend_on_its_offset(fixture_dir, adapter, tmp_path, monkeypatch):
    """Row 0 alone vs row 0 after row 1 (row 1 unsupervised): per-token attention gradients, bit for bit."""
    rs = rows()
    tr = make_trainer(fixture_dir, adapter=adapter)
    n = len(rs[0]["tokens"]) - 1
    o = len(rs[1]["tokens"]) - 1
    dumps = {}
    for tag, (x, y, p, ids, _) in {"alone": window(rs, [0]), "offset": window(rs, [1, 0], supervise={0})}.items():
        for layer in range(4):
            d = tmp_path / f"{tag}{layer}"
            d.mkdir()
            monkeypatch.setenv("SUROGATE_DEBUG_DUMP_DIR", str(d))
            monkeypatch.setenv("SUROGATE_DEBUG_ATTENTION_DUMP_LAYER", str(layer))
            tr.set_grad_accumulation(1)
            kd_step(tr, x, y, p, ids)
            tr.get_kd_loss()
            for name in ("d_out", "d_qkv"):
                f = d / f"attn_live.layer{layer}.{name}.bin"
                dumps[(tag, layer, name)] = np.fromfile(f, dtype=np.float32).reshape(T, -1)
    for layer in range(4):
        for name in ("d_out", "d_qkv"):
            np.testing.assert_array_equal(
                dumps[("alone", layer, name)][:n],
                dumps[("offset", layer, name)][o : o + n],
                err_msg=f"layer {layer} {name}",
            )
    del tr
    gc.collect()


def train_step(tr, rs, layout):
    """One optimizer step's micro-steps; returns per-row losses (token-loss buffer) and raw gradients."""
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
    return losses, grads(tr)


def test_training_step_packed_equals_padded(fixture_dir, adapter):
    rs = rows()
    results = {}
    for layout in ("padded", "padded_again", "reversed", "packed"):
        tr = make_trainer(fixture_dir, adapter=adapter)
        losses, g = train_step(tr, rs, "padded" if layout == "padded_again" else layout)
        from surogate import _surogate as sg

        opt = sg.OptimizerConfig(
            optimizer="adamw",
            learning_rate=1e-4,
            weight_decay=0.0,
            grad_clip=1.0,
            adamw_beta1=0.9,
            adamw_beta2=0.999,
            adamw_epsilon=1e-8,
        )
        step = tr.update_with_config(opt, 1)
        results[layout] = (losses, g, float(step["loss"]), float(step["norm"]))
        del tr
        gc.collect()
    padded, again, reversed_, packed = (results[k] for k in ("padded", "padded_again", "reversed", "packed"))
    # Determinism: the deterministic dQ reduction is the default.
    assert padded[0] == again[0]
    assert all(np.array_equal(padded[1][k], again[1][k]) for k in padded[1])
    # Per-row losses: bit-identical (document masking + batch-invariant forward).
    assert packed[0] == padded[0]
    # Step loss is the same sum over the same supervised tokens, normalised by the same count.
    assert packed[2] == pytest.approx(padded[2], rel=1e-6)
    # Gradients: packing sums rows inside one GEMM where padded training sums them across
    # micro-steps in bf16; the difference must stay within that summation-order noise.
    floor = rel_l2(padded[1], reversed_[1])
    assert 0 < floor < 2e-2
    assert rel_l2(padded[1], packed[1]) <= 2.0 * floor
    assert packed[3] == pytest.approx(padded[3], rel=2e-2)
