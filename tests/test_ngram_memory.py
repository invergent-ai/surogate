"""Hashed n-gram memory, transplant arm (surogate/dsl/modules/ngram_memory.py, kernels/ngram_memory.h,
runtime/ngram/ngram_memory.h; contract rl/ngram-transplant-2026-10-18/INTERFACE.md).

CPU:
  * no ``ngram_memory`` option -> the IR carries nothing of the memory; with it only three config keys differ;
  * option parsing (the Swift-1.5 GGUF manifest, validation, the memory-only PEFT mode);
  * hash parity of the trainer's rule (positions + chunk carry) against the engine's rows_kernel semantics
    (slots with token history, serving segments of random length) and against probe.py's reference on 10^6
    tokens with EOS, conversation starts and chunk boundaries.
GPU, op level (fixtures in surogate._surogate):
  * the rows kernel = the NumPy reference bit for bit (10^6 tokens, chunked with carry);
  * the IQ4_NL gather = the exact decode rounded to bf16;
  * the forward (bf16) against an FP64 reference of the engine algebra; the 3-input residual norm is the plain
    fused norm bit for bit at u = 0;
  * the backward (fp32 path) against FP64 central finite differences for W_k, W_v, g_k, g_q and the residual.
GPU, model level (a random 4-layer qwen3_5_moe, recompute on):
  * zero init (W_v = 0) is the base model bit for bit: validation loss, per-token log-probs and every LoRA
    gradient, unchunked and with sequence_chunks 4;
  * live memory gradients match an fp32 HF reference (the memory as a pre-hook on layer 1); chunked = unchunked;
  * per-tensor learning rate and no weight decay for the memory tensors;
  * can it learn: memory alone on a frozen random model, synthetic n-gram language;
  * export round trip (trainer adapter -> engine export -> reload; refusals).
GPU, the SFT CLI's step (train_step_graphed, its own chunk loop):
  * zero init is the base model bit for bit (bf16 and fp8-hybrid, 4 chunks, grad accumulation 3, document cuts);
  * its chunk carry is step_chunked's bit for bit (real and row-shuffled table) and chunked = unchunked;
  * the SFT CLI end to end with smoke.yaml's and quick_{real,shuf}.yaml's option sets on a tiny full-vocabulary
    model (all 32 chunks, panels, per-token dumps, checkpoint + resume; NGRAM_SFT_TEMPLATES, NGRAM_TOKENIZER_DIR).
"""

from __future__ import annotations

import collections
import gzip
import json
import math
import os
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from surogate.dsl.modules.ngram_memory import (
    NgramMemoryConfig,
    TableMeta,
    iq4nl_decode_rows,
    iq4nl_encode_rows,
    ngram_rows_reference,
)

ARCH = "Qwen3_5MoeForConditionalGeneration"

# Swift-1.5's table (rl/ngram-transplant-2026-10-18/swift15-ple/manifest.json), hash constants only.
SWIFT15_MANIFEST = {
    "meta": {
        "qwen4exp.embedding_length": 2560,
        "qwen4exp.ple.layers": [1],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 8,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.ple.eos_token_id": 248044,
        "qwen4exp.ple.image_token_id": 248056,
        "qwen4exp.embedding_length_per_layer_input": 160,
        "qwen4exp.ple.layer_multipliers": [23703573157769, 20109073645365, 8052911324071],
        "qwen4exp.ple.head_offsets": [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                                      160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                                      300001275],
        "qwen4exp.ple.head_vocab_sizes": [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
                                          20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
                                          20000161, 20000171],
    },
    "tensors": [{"name": "per_layer_token_embd.weight", "type": "IQ4_NL", "dims": [160, 320001536]}],
}
SWIFT15 = TableMeta.from_manifest(SWIFT15_MANIFEST)
EOS = 248044
VOCAB = 248320


def _text_config(n_layers=4, hidden=64, experts=8):
    return {
        "hidden_size": hidden, "num_hidden_layers": n_layers, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 32, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
        "num_experts": experts, "num_experts_per_tok": 2, "vocab_size": 512, "max_position_embeddings": 256,
        "rms_norm_eps": 1e-6, "attention_bias": False, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
        "linear_value_head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (n_layers // 4),
        "full_attention_interval": 4, "tie_word_embeddings": False,
        "rope_parameters": {"partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1], "rope_theta": 10000.0,
                            "rope_type": "default", "mrope_interleaved": True},
    }


def _small_meta(vocab=1009, heads_per_ngram=8, head_dim=32, eos=500):
    heads = 2 * heads_per_ngram
    sizes = [vocab + 2 * h for h in range(heads)]
    offsets = list(np.cumsum([0] + sizes[:-1]))
    return TableMeta(multipliers=SWIFT15.multipliers, head_offsets=tuple(int(o) for o in offsets),
                     head_vocab_sizes=tuple(sizes), ngram_size=3, heads_per_ngram=heads_per_ngram,
                     head_dim=head_dim, row_count=int(sum(sizes)), eos_token_id=eos, layer=1)


def _meta_dict(meta: TableMeta):
    return {"multipliers": list(meta.multipliers), "head_offsets": list(meta.head_offsets),
            "head_vocab_sizes": list(meta.head_vocab_sizes), "ngram_size": meta.ngram_size,
            "heads_per_ngram": meta.heads_per_ngram, "head_dim": meta.head_dim, "row_count": meta.row_count,
            "eos_token_id": meta.eos_token_id, "layer": meta.layer}


def _ir(cfg, ngram_memory=None):
    from surogate.dsl.py_compiler import compile_model_for_hf

    raw = compile_model_for_hf(ARCH, cfg, extra_config={"ngram_memory": ngram_memory} if ngram_memory else None)
    ir = json.loads(raw)
    assert ir["success"], ir.get("errors")
    return ir


def _config(**kw):
    return {"architectures": [ARCH], "model_type": "qwen3_5_moe", "tie_word_embeddings": False,
            "text_config": _text_config(**kw)}


# ---------------------------------------------------------------------------------------------- CPU: IR, config


class TestIR:
    def test_off_carries_nothing_and_on_adds_three_keys(self):
        cfg = _config()
        off = _ir(cfg)
        assert "ngram" not in json.dumps(off)
        assert json.dumps(off, sort_keys=True) == json.dumps(_ir(cfg, ngram_memory=None), sort_keys=True)
        option = {"table": "/nonexistent/table.iq4nl", "meta": _meta_dict(_small_meta())}
        on = _ir(cfg, ngram_memory=option)
        off_cfg, on_cfg = off["modules"][0]["config"], on["modules"][0]["config"]
        assert set(on_cfg) - set(off_cfg) == {"ngram_memory", "ngram_memory_layer", "ngram_memory_spec"}
        assert {k: v for k, v in on_cfg.items() if not k.startswith("ngram_memory")} == off_cfg
        on_rest = dict(on)
        on_rest["modules"] = [dict(m) for m in on["modules"]]
        on_rest["modules"][0]["config"] = off_cfg
        assert json.dumps(on_rest, sort_keys=True) == json.dumps(off, sort_keys=True)
        spec = json.loads(on_cfg["ngram_memory_spec"])
        assert on_cfg["ngram_memory_layer"] == 1 == spec["layer"]
        assert spec["heads"] == 16 and spec["head_dim"] == 32 and spec["embed_dim"] == 512
        assert spec["lr_scale"] == 1.0 and spec["zero_token_ids"] == [248056, 248057]

    def test_layer_must_exist(self):
        with pytest.raises(Exception, match="not a layer"):
            _ir(_config(), ngram_memory={"table": "/x", "meta": _meta_dict(_small_meta()), "layer": 4})


class TestConfig:
    def test_swift15_manifest(self):
        m = SWIFT15
        m.validate()
        assert m.heads == 16 and m.head_dim == 160 and m.embed_dim == 2560 and m.row_bytes == 90
        assert m.row_count == 320_001_536 and m.layer == 1 and m.eos_token_id == EOS
        assert m.head_offsets[-1] + m.head_vocab_sizes[-1] <= m.row_count

    def test_parse_and_validation(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(SWIFT15_MANIFEST))
        cfg = NgramMemoryConfig.parse({"table": str(tmp_path / "table.iq4nl"), "lr": 2e-4, "base_lr": 1e-4})
        assert cfg.meta == SWIFT15 and cfg.layer == 1 and cfg.only and cfg.lr_scale() == 2.0
        assert NgramMemoryConfig.parse(cfg.to_dict()) == cfg
        assert NgramMemoryConfig.parse(None) is None and NgramMemoryConfig.parse({}) is None
        with pytest.raises(ValueError, match="unknown keys"):
            NgramMemoryConfig.parse({"table": "x", "meta": _meta_dict(_small_meta()), "lr_scale": 2})
        with pytest.raises(ValueError, match="table_residency"):
            NgramMemoryConfig.parse({"table": "x", "meta": _meta_dict(_small_meta()), "table_residency": "disk"})
        with pytest.raises(ValueError, match="manifest"):
            NgramMemoryConfig.parse({"table": str(tmp_path / "missing" / "t.iq4nl")})
        bad = _meta_dict(_small_meta())
        bad["head_dim"] = 48
        with pytest.raises(ValueError, match="multiple of 32"):
            NgramMemoryConfig.parse({"table": "x", "meta": bad})
        with pytest.raises(ValueError, match="needs the run's learning_rate"):
            NgramMemoryConfig.parse({"table": "x", "meta": _meta_dict(_small_meta()), "lr": 1e-3}).spec()


# ---------------------------------------------------------------------------------------------- CPU: hash parity


def _conversations(rng, total, mean_len=900, eos_rate=0.02, vocab=VOCAB):
    """Random conversations whose lengths sum to `total`: random ids, EOS at random (also as first token)."""
    convs, n = [], 0
    while n < total:
        length = int(min(total - n, max(1, rng.geometric(1.0 / mean_len))))
        ids = rng.integers(0, vocab, size=length, dtype=np.int64)
        ids[rng.random(length) < eos_rate] = EOS
        convs.append(ids)
        n += length
    return convs


def _rows_layout(convs, B):
    """Concatenate conversations into B rows of equal length; positions restart at every conversation."""
    ids = np.concatenate(convs)
    pos = np.concatenate([np.arange(len(c)) for c in convs])
    assert len(ids) % B == 0
    T = len(ids) // B
    # A conversation must not continue across rows: split it into a new conversation at the row start.
    pos = pos.reshape(B, T).copy()
    for b in range(1, B):
        if pos[b, 0] != 0:
            k = int(np.argmax(pos[b] == 0)) if (pos[b] == 0).any() else T
            pos[b, :k] = np.arange(k)
    return ids.reshape(B, T), pos


def _engine_rows(ids, pos, meta: TableMeta, rng):
    """The engine's rows_kernel + history_update_kernel semantics: one slot per conversation (history EOS when
    fresh), each conversation served in segments of random length (prefill chunks, single decode tokens)."""
    n = meta.ngram_size - 1
    eos = meta.eos_token_id
    flat_ids, flat_pos = ids.reshape(-1), pos.reshape(-1)
    starts = list(np.flatnonzero(flat_pos == 0)) + [len(flat_ids)]
    out = np.empty((len(flat_ids), meta.heads), dtype=np.int64)
    per = meta.heads // n
    for c0, c1 in zip(starts[:-1], starts[1:]):
        history = [eos] * n  # a fresh slot
        t = c0
        while t < c1:
            seg = int(min(c1 - t, rng.choice([1, 1, 2, 7, 64, 513, 4096])))
            seg_ids = flat_ids[t:t + seg]
            ext = np.concatenate([np.array(history, dtype=np.int64), seg_ids])  # history oldest first
            ctx = [seg_ids]
            cut = np.zeros(seg, dtype=bool)
            for k in range(1, meta.ngram_size):
                tok = ext[n - k: n - k + seg]  # predecessor k: in the segment or from the history
                tok = np.where(cut, -1, tok)
                cut = cut | (tok < 0) | (tok == eos)
                ctx.append(np.where(cut, eos, tok))
            with np.errstate(over="ignore"):
                m2 = (ctx[0].astype(np.uint64) * np.uint64(meta.multipliers[0])) ^ (
                    ctx[1].astype(np.uint64) * np.uint64(meta.multipliers[1]))
                m3 = m2 ^ (ctx[2].astype(np.uint64) * np.uint64(meta.multipliers[2]))
            for h in range(meta.heads):
                mixed = m2 if h < per else m3
                out[t:t + seg, h] = (mixed % np.uint64(meta.head_vocab_sizes[h])).astype(np.int64) + meta.head_offsets[h]
            history = list(ext[-n:])  # the segment's last tokens (older from the old history)
            t += seg
    return out


def _probe_rows(ids, start, meta: TableMeta):
    """rl/ngram-transplant-2026-10-18/probe.py's hash, verbatim (flat stream, document-start flags)."""
    MULT = np.array(meta.multipliers, dtype=np.uint64)
    OFFS = np.array(meta.head_offsets, dtype=np.int64)
    VSZ = np.array(meta.head_vocab_sizes, dtype=np.uint64)
    EOS_, HEADS, NG = meta.eos_token_id, meta.heads, meta.ngram_size
    ds = np.maximum.accumulate(np.where(start, np.arange(len(ids)), 0))
    pos = np.arange(len(ids))
    p1 = np.where(pos - 1 >= ds, np.roll(ids, 1), -1); cut1 = (p1 < 0) | (p1 == EOS_)
    p2 = np.where(pos - 2 >= ds, np.roll(ids, 2), -1); cut2 = cut1 | (p2 < 0) | (p2 == EOS_)
    c0 = ids; c1 = np.where(cut1, EOS_, p1); c2 = np.where(cut2, EOS_, p2)
    with np.errstate(over="ignore"):
        bi = (c0.astype(np.uint64) * MULT[0]) ^ (c1.astype(np.uint64) * MULT[1])
        tri = bi ^ (c2.astype(np.uint64) * MULT[2])
    per = HEADS // (NG - 1)
    rows = np.empty((len(ids), HEADS), dtype=np.int64)
    for h in range(HEADS):
        mixed = bi if h < per else tri
        rows[:, h] = (mixed % VSZ[h]).astype(np.int64) + OFFS[h]
    return rows


def _chunked_reference(ids, pos, meta, T):
    """The trainer's rows, chunk by chunk with the two-token carry (as the chunked step feeds them)."""
    B, T_full = ids.shape
    out = []
    for c in range(T_full // T):
        carry = np.full((B, 2), -1, dtype=np.int64)
        if c > 0:
            carry = ids[:, c * T - 2: c * T]
        rows = ngram_rows_reference(ids[:, c * T:(c + 1) * T], pos[:, c * T:(c + 1) * T], meta, carry)
        out.append(rows.reshape(B, T, -1))
    return np.concatenate(out, axis=1).reshape(B * T_full, -1)


@pytest.fixture(scope="module")
def token_stream():
    rng = np.random.default_rng(2026)
    B = 4
    convs = _conversations(rng, total=1 << 20)
    ids, pos = _rows_layout(convs, B)
    return ids, pos


def test_rows_match_engine_semantics(token_stream):
    """10^6 tokens: the trainer's rule (positions, unchunked and in 32 chunks with carry) = the engine's slot
    + segment semantics = probe.py's reference, for Swift-1.5's hash constants."""
    ids, pos = token_stream
    B, T_full = ids.shape
    ref = ngram_rows_reference(ids, pos, SWIFT15)
    chunked = _chunked_reference(ids, pos, SWIFT15, T_full // 32)
    engine = _engine_rows(ids, pos, SWIFT15, np.random.default_rng(7))
    flat = ids.reshape(-1)
    start = (pos.reshape(-1) == 0)
    probe = _probe_rows(flat, start, SWIFT15)
    assert np.array_equal(ref, chunked)
    assert np.array_equal(ref, engine)
    assert np.array_equal(ref, probe)
    # The stream exercises what it should.
    assert (flat == EOS).mean() > 0.01 and start.sum() > 500
    assert (ref >= 0).all() and (ref < SWIFT15.row_count).all()


# ---------------------------------------------------------------------------------------------- GPU helpers


def _gpu():
    global torch
    torch = pytest.importorskip("torch")
    try:
        from surogate import _surogate  # noqa: F401
    except ImportError:
        pytest.skip("surogate._surogate C++ extension not built")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from surogate import _surogate as sg

    return sg


def _spec_json(meta: TableMeta, table="/unused", zero=(248056, 248057), **kw):
    cfg = NgramMemoryConfig(table=table, meta=meta, layer=1, zero_token_ids=tuple(zero), **kw)
    return json.dumps(cfg.spec())


# ---------------------------------------------------------------------------------------------- GPU: op level


@pytest.mark.gpu
def test_rows_kernel_is_the_reference(token_stream):
    sg = _gpu()
    ids, pos = token_stream
    B, T_full = ids.shape
    T = T_full // 32
    spec = _spec_json(SWIFT15)
    want = _chunked_reference(ids, pos, SWIFT15, T).reshape(B, T_full, -1)
    got = np.empty_like(want)
    for c in range(32):
        i = torch.from_numpy(ids[:, c * T:(c + 1) * T].astype(np.int32)).cuda()
        p = torch.from_numpy(pos[:, c * T:(c + 1) * T].astype(np.int32)).cuda()
        carry = None if c == 0 else torch.from_numpy(ids[:, c * T - 2:c * T].astype(np.int32)).contiguous().cuda()
        rows = torch.empty((B * T, 16), dtype=torch.int32, device="cuda")
        sg._ngram_memory_rows(i, p, carry, rows, spec)
        got[:, c * T:(c + 1) * T] = rows.cpu().numpy().reshape(B, T, -1)
    assert np.array_equal(got, want)


def _random_table(meta: TableMeta, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    values = (rng.standard_normal((meta.row_count, meta.head_dim)) * scale).astype(np.float32)
    return iq4nl_encode_rows(values)


@pytest.mark.gpu
def test_gather_is_the_exact_decode():
    _gpu()
    from surogate import _surogate as sg

    meta = _small_meta(vocab=211, head_dim=64)
    raw = _random_table(meta, seed=1)
    table = torch.from_numpy(raw.reshape(-1)).cuda()
    rng = np.random.default_rng(3)
    N = 777
    rows = rng.integers(0, meta.row_count, size=(N, meta.heads)).astype(np.int32)
    emb = torch.empty((N, meta.embed_dim), dtype=torch.bfloat16, device="cuda")
    sg._ngram_memory_gather(torch.from_numpy(rows).cuda(), table, emb, _spec_json(meta))
    want = iq4nl_decode_rows(raw[rows.reshape(-1)], meta.head_dim).reshape(N, -1)
    want_bf16 = torch.from_numpy(want).to(torch.bfloat16)  # round-to-nearest-even, as __float2bfloat16_rn
    assert torch.equal(emb.cpu(), want_bf16)


def _bf16(a):
    """Round FP64 values to bf16 (nearest even), back to FP64."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)
    return t.double().numpy()


def _ref_forward(r, e, Wk, Wv, gk, gq, zero_mask, eps=1e-6, round_kv=False):
    """FP64 reference of the engine algebra (ple_reference.py, one stream, conv zero); round_kv rounds k and v to
    bf16 as the op (and the engine's cuBLASLt bf16 output) does."""
    k = e @ Wk.T
    v = e @ Wv.T
    if round_kv:
        k, v = _bf16(k), _bf16(v)
    C = k.shape[1]
    kn = k / np.sqrt((k * k).mean(1, keepdims=True) + eps) * gk
    qn = r / np.sqrt((r * r).mean(1, keepdims=True) + eps) * gq
    z = (kn * qn).sum(1) / np.sqrt(C)
    g = 1.0 / (1.0 + np.exp(-(np.copysign(np.sqrt(np.maximum(np.abs(z), 1e-6)), z))))
    g = np.where(zero_mask, 0.0, g)
    return g[:, None] * v, g, z


def _op_inputs(N, C, E, seed, dtype):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((N, C)) * 0.7
    e = rng.standard_normal((N, E)) * 0.05
    Wk = rng.standard_normal((C, E)) * 0.02 * 4
    Wv = rng.standard_normal((C, E)) * 0.02
    gk = 1.0 + 0.2 * rng.standard_normal(C)
    gq = 1.0 + 0.2 * rng.standard_normal(C)
    ids = rng.integers(0, 1000, size=N).astype(np.int32)
    ids[::17] = 248056  # image pads: zero output
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float32
    q = lambda a: torch.from_numpy(a.astype(np.float32)).to(tdt)  # noqa: E731
    vals = {"r": q(r), "e": q(e), "Wk": q(Wk), "Wv": q(Wv)}
    f64 = {k: v.double().numpy() for k, v in vals.items()}  # the exact operand values in FP64
    f64["gk"], f64["gq"] = gk.astype(np.float32).astype(np.float64), gq.astype(np.float32).astype(np.float64)
    vals["gk"], vals["gq"] = torch.from_numpy(gk.astype(np.float32)), torch.from_numpy(gq.astype(np.float32))
    return vals, f64, ids


def _run_forward(sg, vals, ids, spec, dtype):
    tdt = torch.bfloat16 if dtype == "bf16" else torch.float32
    N, C = vals["r"].shape
    k = torch.empty((N, C), dtype=tdt, device="cuda")
    v = torch.empty_like(k)
    u = torch.empty_like(k)
    stats = torch.empty((N, 4), dtype=torch.float32, device="cuda")
    ins = [vals[n].cuda().contiguous() for n in ("r", "e", "Wk", "Wv", "gk", "gq")] + [torch.from_numpy(ids).cuda()]
    sg._ngram_memory_forward(ins, [k, v, stats, u], spec)
    return ins, k, v, stats, u


@pytest.mark.gpu
def test_forward_matches_fp64_reference():
    sg = _gpu()
    N, C, E = 300, 2048, 2560  # the 35B's geometry
    vals, f64, ids = _op_inputs(N, C, E, seed=5, dtype="bf16")
    _, k, v, stats, u = _run_forward(sg, vals, ids, _spec_json(SWIFT15), "bf16")
    zero = ids == 248056
    u_ref, g_ref, z_ref = _ref_forward(f64["r"], f64["e"], f64["Wk"], f64["Wv"], f64["gk"], f64["gq"], zero)
    u_rb, g_rb, z_rb = _ref_forward(f64["r"], f64["e"], f64["Wk"], f64["Wv"], f64["gk"], f64["gq"], zero,
                                    round_kv=True)
    u_got = u.double().cpu().numpy()
    g_got = np.where(zero, 0, stats[:, 3].double().cpu().numpy())
    z_got = stats[:, 2].double().cpu().numpy()
    rel = np.linalg.norm(u_got - u_ref) / np.linalg.norm(u_ref)
    rel_rb = np.linalg.norm(u_got - u_rb) / np.linalg.norm(u_rb)
    report = {"exact": {"u_rel_l2": rel, "g_max_abs": float(np.abs(g_got - g_ref).max()),
                        "z_max_abs": float(np.abs(z_got - z_ref).max())},
              "bf16_kv": {"u_rel_l2": rel_rb, "g_max_abs": float(np.abs(g_got - g_rb).max()),
                          "z_max_abs": float(np.abs(z_got - z_rb).max())}}
    print(json.dumps(report))
    assert np.all(u_got[zero] == 0.0)
    # Against the exact algebra: bf16 k, v and u (the gate is steep near z = 0: d g / dz ~ 1 / (8 sqrt|z|)).
    assert rel < 1e-2 and report["exact"]["g_max_abs"] < 2e-2, report
    # Against the same rounding points: only fp32 accumulation order and GEMM roundings differ.
    assert rel_rb < 4e-3 and report["bf16_kv"]["g_max_abs"] < 2e-3 and report["bf16_kv"]["z_max_abs"] < 1e-3, report
    # fp32 path: tight
    vals32, f64_32, _ = _op_inputs(64, 96, 128, seed=6, dtype="fp32")
    _, _, _, stats32, u32 = _run_forward(sg, vals32, ids[:64], _spec_json(SWIFT15), "fp32")
    u_ref32, g_ref32, _ = _ref_forward(f64_32["r"], f64_32["e"], f64_32["Wk"], f64_32["Wv"], f64_32["gk"],
                                       f64_32["gq"], ids[:64] == 248056)
    assert np.linalg.norm(u32.double().cpu().numpy() - u_ref32) / np.linalg.norm(u_ref32) < 1e-5


@pytest.mark.gpu
def test_residual3_norm_is_the_plain_norm_at_zero():
    sg = _gpu()
    rng = np.random.default_rng(11)
    N, C = 333, 2048
    a = torch.from_numpy(rng.standard_normal((N, C)).astype(np.float32)).to(torch.bfloat16).cuda()
    b = torch.from_numpy(rng.standard_normal((N, C)).astype(np.float32) * 0.3).to(torch.bfloat16).cuda()
    w = torch.from_numpy(1 + 0.1 * rng.standard_normal(C).astype(np.float32)).to(torch.bfloat16).cuda()
    zero = torch.zeros_like(a)
    neg_zero = -torch.zeros_like(a)

    def run(u, plain):
        res, y = torch.empty_like(a), torch.empty_like(a)
        rr = torch.empty(N, dtype=torch.float32, device="cuda")
        sg._ngram_memory_residual3_rmsnorm([a, b, u, w], [res, y, rr], 1e-6, plain)
        return res, y, rr

    plain = run(zero, True)
    for u in (zero, neg_zero):
        got = run(u, False)
        for x, yv in zip(plain, got):
            assert torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
                               yv.view(torch.int16) if yv.dtype == torch.bfloat16 else yv.view(torch.int32))
    u = torch.from_numpy(rng.standard_normal((N, C)).astype(np.float32) * 0.01).to(torch.bfloat16).cuda()
    res, y, rr = run(u, False)
    s = a.double() + b.double() + u.double()
    assert torch.allclose(res.double(), s, atol=0, rtol=2 ** -8)
    rms = 1 / torch.sqrt((s.float() ** 2).mean(1) + 1e-6)
    assert torch.allclose(rr, rms, rtol=1e-5)


@pytest.mark.gpu
def test_backward_matches_fp64_finite_differences():
    """C = 32, E = 64: the fp32 backward of W_k, W_v, g_k, g_q and the residual against central differences
    of the FP64 reference (h = 1e-6), on random coordinates, image-pad tokens included."""
    sg = _gpu()
    N, C, E = 48, 32, 64
    vals, f64, ids = _op_inputs(N, C, E, seed=21, dtype="fp32")
    spec = _spec_json(SWIFT15)
    ins, k, v, stats, u = _run_forward(sg, vals, ids, spec, "fp32")
    rng = np.random.default_rng(4)
    G = rng.standard_normal((N, C))
    zero = ids == 248056

    def loss(p):
        uu, _, _ = _ref_forward(p["r"], p["e"], p["Wk"], p["Wv"], p["gk"], p["gq"], zero)
        return float((G * uu).sum())  # L = <G, u>: dL/dr is the gate path alone (the op adds it to G)

    d_res = torch.from_numpy(G.astype(np.float32)).cuda()
    dWk = torch.zeros((C, E), dtype=torch.float32, device="cuda")
    dWv, dgk, dgq = torch.zeros_like(dWk), torch.zeros(C, device="cuda"), torch.zeros(C, device="cuda")
    sg._ngram_memory_backward([ins[0], ins[1], k, v, stats, ins[4], ins[5], ins[6]], [d_res, dWk, dWv, dgk, dgq],
                              spec)
    analytic = {"Wk": dWk, "Wv": dWv, "gk": dgk, "gq": dgq,
                "r": d_res.double() - torch.from_numpy(G.astype(np.float32)).double().cuda()}
    report = {}
    for name, grad in analytic.items():
        g = grad.double().cpu().numpy()
        shape = f64[name].shape
        coords = [tuple(int(rng.integers(0, s)) for s in shape) for _ in range(40)]
        errs, scale = [], np.abs(g).max()
        for c in coords:
            p_plus = {kk: vv.copy() for kk, vv in f64.items()}
            p_minus = {kk: vv.copy() for kk, vv in f64.items()}
            p_plus[name][c] += 1e-6
            p_minus[name][c] -= 1e-6
            fd = (loss(p_plus) - loss(p_minus)) / 2e-6
            errs.append(abs(fd - g[c]) / max(abs(fd), 1e-3 * scale))
        report[name] = max(errs)
    print(json.dumps(report))
    for name, err in report.items():
        assert err < 2e-3, (name, err)
    # pad tokens: no gradient through them (the direct path only)
    assert np.allclose(d_res.cpu().numpy()[zero], G[zero].astype(np.float32))


# ---------------------------------------------------------------------------------------------- GPU: model level

HIDDEN = 256
TINY_META = None  # set by the tiny_model fixture


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A random 4-layer qwen3_5_moe (3 GDN + 1 attention layer) and a small random frozen n-gram table."""
    _gpu()
    pytest.importorskip("transformers")
    import transformers

    out = tmp_path_factory.mktemp("qwen35moe_ngram")
    text = _text_config(n_layers=4, hidden=HIDDEN, experts=4)
    text.update(num_experts_per_tok=4, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [4, 2, 2], "mrope_interleaved": True})
    vision = {"depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4,
              "out_hidden_size": HIDDEN, "num_position_embeddings": 256, "patch_size": 16,
              "spatial_merge_size": 2, "temporal_patch_size": 2, "in_channels": 3}
    cfg = transformers.Qwen3_5MoeConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)
    torch.manual_seed(2026)
    model = transformers.Qwen3_5MoeForConditionalGeneration(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p, dtype=torch.float32).uniform_(0.5, 2.0)).to(p.dtype))
            elif name.endswith("dt_bias"):
                p.zero_()
            elif p.dim() == 1:
                p.copy_((0.1 * torch.randn_like(p, dtype=torch.float32)).to(p.dtype))
    model.save_pretrained(out, safe_serialization=True)
    _fuse_experts(out, text["num_experts"])
    meta = _small_meta(vocab=1009, head_dim=32, eos=500)
    raw = _random_table(meta, seed=9, scale=0.5)
    (out / "table.iq4nl").write_bytes(raw.tobytes())
    (out / "table_meta.json").write_text(json.dumps(_meta_dict(meta)))
    return out


def _fuse_experts(root, experts):
    import re

    from safetensors.torch import load_file, save_file

    shards = sorted(Path(root).glob("*.safetensors"))
    tensors = {}
    for shard in shards:
        tensors.update(load_file(str(shard)))
    fused = {}
    for key in list(tensors):
        m = re.match(r"(.*\.mlp\.experts)\.0\.gate_proj\.weight$", key)
        if not m:
            continue
        base = m.group(1)
        gate = [tensors.pop(f"{base}.{e}.gate_proj.weight") for e in range(experts)]
        up = [tensors.pop(f"{base}.{e}.up_proj.weight") for e in range(experts)]
        down = [tensors.pop(f"{base}.{e}.down_proj.weight") for e in range(experts)]
        fused[f"{base}.gate_up_proj"] = torch.stack([torch.cat([g, u], 0) for g, u in zip(gate, up)]).contiguous()
        fused[f"{base}.down_proj"] = torch.stack(down).contiguous()
    if not fused:
        return
    tensors.update(fused)
    for shard in shards:
        shard.unlink()
    index = Path(root) / "model.safetensors.index.json"
    if index.exists():
        index.unlink()
    save_file(tensors, str(Path(root) / "model.safetensors"), metadata={"format": "pt"})


def _memory_option(model_dir, residency="host", **kw):
    meta = json.loads((Path(model_dir) / "table_meta.json").read_text())
    return {"table": str(Path(model_dir) / "table.iq4nl"), "meta": meta, "table_residency": residency,
            "zero_token_ids": [], **kw}


def _trainer(model_dir, targets, seq_len, ngram=None, chunks=1, recompute=True, lora_dtype="fp32", train_router=False,
             recipe="bf16", grad_accum=1):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    opts = sg.RuntimeOptions(recipe=recipe, use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    if recompute:
        opts.recompute = "true"
    if chunks > 1:
        opts.sequence_chunks = chunks
        os.environ["SUROGATE_ROPE_MAX_SEQ"] = str(seq_len)
    opts.dsl_ir_json = build_dsl_ir_for_model(str(model_dir), extra_config={"ngram_memory": ngram} if ngram else None)
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    lora_config = sg.LoRAAdapterConfig(rank=8, alpha=16, dropout=0, dtype=lora_dtype, target_modules=targets,
                                       train_router=train_router)
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(str(model_dir), "bf16"),
                            options=opts, batch_size=1, seq_len=seq_len // chunks, grad_accum=grad_accum,
                            memcpy_all_gather=True, memcpy_send_recv=True, lora_config=lora_config, qlora_config=None)
    tr.import_weights(get_model_weights_path(str(model_dir)))
    return tr


def _tokens(seq, seed=7, vocab=512, eos=500, starts=(0,)):
    rng = np.random.default_rng(seed)
    x = rng.integers(10, vocab, size=(1, seq), dtype=np.int32)
    x[0, rng.random(seq) < 0.03] = eos
    y = np.concatenate([x[:, 1:], np.full((1, 1), -100, np.int32)], axis=1).astype(np.int32)
    return x, y


NON_EXPERT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a",
                      "in_proj_b", "out_proj"]
KEY = "base_model.model.model.layers.1.ple.key_proj.weight"
VALUE = "base_model.model.model.layers.1.ple.value_proj.weight"
NORM_K = "base_model.model.model.layers.1.ple.norm_key.weight"
NORM_Q = "base_model.model.model.layers.1.ple.norm_query.weight"
MEMORY = (KEY, VALUE, NORM_K, NORM_Q)


def _adapter(tr, model_dir, tmp, name):
    from safetensors.torch import load_file

    tr.export_adapter(str(tmp / name), str(model_dir))
    return load_file(str(tmp / name / "adapter_model.safetensors")), json.loads(
        (tmp / name / "adapter_config.json").read_text())


def _randomized(adapter, seed, memory_scale=None):
    """Nonzero LoRA (and, with memory_scale, a nonzero W_v and perturbed gains); per-tensor generators."""
    out = {}
    for k, v in adapter.items():
        g = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(k.encode()))
        if k == VALUE:
            out[k] = (torch.randn(v.shape, generator=g) * (memory_scale or 0.0)).to(v.dtype)
        elif k in (NORM_K, NORM_Q):
            out[k] = (1.0 + (torch.randn(v.shape, generator=g) * 0.2 if memory_scale else 0.0) * torch.ones(v.shape)).to(v.dtype)
        elif k == KEY:
            out[k] = v.clone()
        else:
            out[k] = (torch.randn(v.shape, generator=g) * (0.05 if "lora_A" in k else 0.02)).to(v.dtype)
    return out


def _grads(tr):
    return {k: torch.from_dlpack(v).float().cpu().clone() for k, v in tr.get_lora_gradients(0).items()}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("chunks,recipe,recompute", [(1, "bf16", True), (4, "bf16", True), (4, "fp8-hybrid", True),
                                                     (1, "bf16", False)])
def test_zero_init_is_the_base_model_bit_for_bit(tiny_model, chunks, recipe, recompute):
    """W_v = 0: loss, per-token log-probs and every LoRA gradient equal the memory-off run bit for bit; the
    W_v gradient is live, the W_k / gain gradients are exactly zero (no gate gradient through v = 0)."""
    from safetensors.torch import save_file

    seq = 128
    x, y = _tokens(seq)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        results = {}
        for label, ngram in (("base", None), ("memory", _memory_option(tiny_model, only=False))):
            # bf16 adapters (expert LoRA needs them): the memory's masters are bf16 here too.
            tr = _trainer(tiny_model, ["all"], seq, ngram=ngram, chunks=chunks, lora_dtype="bf16", recipe=recipe,
                          recompute=recompute)
            init, cfg = _adapter(tr, tiny_model, tmp, label)
            if ngram:
                assert init[KEY].shape == (HIDDEN, 512) == init[VALUE].shape
                assert not init[VALUE].any() and 0.015 < init[KEY].float().std() < 0.025
                assert torch.equal(init[NORM_K], torch.ones(HIDDEN)) and torch.equal(init[NORM_Q], torch.ones(HIDDEN))
                assert cfg["surogate_ngram_memory"]["layer"] == 1
            else:
                assert KEY not in init and "surogate_ngram_memory" not in cfg
            live = _randomized(init, seed=11)
            save_file(live, str(tmp / f"{label}.safetensors"))
            tr.import_adapter(str(tmp / f"{label}.safetensors"))
            res = {"loss": tr.validate(x, y)}
            if chunks == 1:
                res["lp"] = tr.compute_logprobs(x, y, use_lora=True)[0].copy()
            tr.step(x, y)
            res["grads"] = _grads(tr)
            if ngram:
                res["stats"] = tr.get_ngram_memory_stats(0, True)
            results[label] = res
            del tr
            torch.cuda.empty_cache()
    base, on = results["base"], results["memory"]
    assert np.float32(base["loss"]).tobytes() == np.float32(on["loss"]).tobytes(), (base["loss"], on["loss"])
    if chunks == 1:
        assert np.array_equal(base["lp"], on["lp"])
    common = set(base["grads"]) & set(on["grads"])
    assert common == set(base["grads"]) and set(on["grads"]) - common == set(MEMORY)
    for k in sorted(common):
        assert torch.equal(base["grads"][k], on["grads"][k]), k
    g = on["grads"]
    assert g[VALUE].norm() > 0
    assert not g[KEY].any() and not g[NORM_K].any() and not g[NORM_Q].any()
    st = on["stats"]
    print(json.dumps(st))
    assert st["tokens"] > 0 and 0.0 < st["gate_mean"] < 1.0 and st["out_rms_over_residual_rms"] == 0.0


def _reference_grads(model_dir, adapter, x, y, dtype=None):
    """HF forward (fp32 unless `dtype`) of the base model with the memory as a forward pre-hook on decoder
    layer 1 (its residual input r -> r + u(r)): per-token log-probs and the gradients of the mean loss with
    respect to the memory's four tensors. The table embedding is the exact decode rounded to bf16 (as the
    trainer gathers it), the projections are the trainer's bf16 work copies."""
    from transformers import AutoModelForImageTextToText

    dtype = dtype or torch.float32
    meta = TableMeta.from_manifest(json.loads((Path(model_dir) / "table_meta.json").read_text()))
    raw = np.frombuffer((Path(model_dir) / "table.iq4nl").read_bytes(), dtype=np.uint8).reshape(meta.row_count, -1)
    rows = ngram_rows_reference(x.astype(np.int64), np.arange(x.shape[1])[None, :], meta)
    e = iq4nl_decode_rows(raw[rows.reshape(-1)], meta.head_dim).reshape(x.shape[1], -1)
    e = torch.from_numpy(e).to(torch.bfloat16).to(dtype).cuda()
    hf = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=dtype).cuda().eval()
    lm = hf.model.language_model
    wk = adapter[KEY].to(torch.bfloat16).to(dtype).cuda().requires_grad_(True)
    wv = adapter[VALUE].to(torch.bfloat16).to(dtype).cuda().requires_grad_(True)
    gk = adapter[NORM_K].float().cuda().requires_grad_(True)
    gq = adapter[NORM_Q].float().cuda().requires_grad_(True)

    def memory(r):
        rf = r.float()
        k, v = (e @ wk.t()).float(), (e @ wv.t()).float()
        C = k.shape[-1]
        kn = k * torch.rsqrt((k * k).mean(-1, keepdim=True) + 1e-6) * gk
        qn = rf * torch.rsqrt((rf * rf).mean(-1, keepdim=True) + 1e-6) * gq
        z = (kn * qn).sum(-1) / math.sqrt(C)
        a = torch.sign(z) * torch.sqrt(torch.clamp(z.abs(), min=1e-6))
        return (r + (torch.sigmoid(a)[..., None] * v).to(r.dtype))

    def pre_hook(module, args, kwargs):
        if args:
            return (memory(args[0]), *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = memory(kwargs["hidden_states"])
        return args, kwargs

    h = lm.layers[1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    logits = hf(input_ids=torch.from_numpy(x).long().cuda()).logits.float()
    labels = torch.from_numpy(y).long().cuda()
    lp = torch.log_softmax(logits[0], -1)
    valid = labels[0] != -100
    token_lp = lp[torch.arange(lp.shape[0], device=lp.device), labels[0].clamp_min(0)]
    loss = -(token_lp[valid]).mean()
    loss.backward()
    h.remove()
    grads = {KEY: wk.grad.float().cpu(), VALUE: wv.grad.float().cpu(), NORM_K: gk.grad.float().cpu(),
             NORM_Q: gq.grad.float().cpu()}
    out = token_lp.detach().cpu().numpy()
    del hf
    torch.cuda.empty_cache()
    return out, grads


@pytest.mark.gpu
@pytest.mark.slow
def test_live_memory_matches_an_fp32_reference_chunked_and_not(tiny_model):
    """Memory only (no LoRA), nonzero W_v and gains: per-token log-probs and the four memory gradients
    agree with an fp32 HF reference as closely as bf16 rounding allows (the same reference in bf16), and the
    chunked step (4 chunks, the two-token carry) equals the unchunked one."""
    from safetensors.torch import save_file

    seq = 128
    x, y = _tokens(seq, seed=3)
    valid = y[0] != -100
    n_valid = int(valid.sum())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        runs = {}
        for chunks in (1, 4):
            tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model), chunks=chunks)
            init, _ = _adapter(tr, tiny_model, tmp, f"init{chunks}")
            assert set(init) == set(MEMORY)  # memory-only: no LoRA tensors
            live = _randomized(init, seed=5, memory_scale=0.05)
            save_file(live, str(tmp / "live.safetensors"))
            tr.import_adapter(str(tmp / "live.safetensors"))
            run = {"loss": tr.validate(x, y)}
            if chunks == 1:
                run["lp"] = tr.compute_logprobs(x, y, use_lora=True)[0][valid].astype(np.float64)
                run["lp_base"] = tr.compute_logprobs(x, y, use_lora=False)[0][valid].astype(np.float64)
            tr.step(x, y)
            run["grads"] = _grads(tr)
            runs[chunks] = run
            del tr
            torch.cuda.empty_cache()
        ref_lp, ref = _reference_grads(tiny_model, live, x, y)
        bf_lp, bf = _reference_grads(tiny_model, live, x, y, dtype=torch.bfloat16)

    def rel(ours, theirs):
        ours, theirs = ours.flatten(), theirs.flatten()
        return float((ours - theirs).norm() / theirs.norm().clamp_min(1e-12)), float(
            torch.nn.functional.cosine_similarity(ours, theirs, dim=0))

    effect = np.abs(runs[1]["lp"] - runs[1]["lp_base"]).mean()
    diff = np.abs(runs[1]["lp"] - ref_lp[valid])
    bf_diff = np.abs(bf_lp[valid] - ref_lp[valid])
    report = {k: rel(runs[1]["grads"][k] / n_valid, ref[k]) for k in MEMORY}
    noise = {k: rel(bf[k], ref[k]) for k in MEMORY}
    chunk = {k: rel(runs[4]["grads"][k], runs[1]["grads"][k]) for k in MEMORY}
    chunk_ref = {k: rel(runs[4]["grads"][k] / n_valid, ref[k]) for k in MEMORY}
    print(json.dumps({"memory_effect_mean": float(effect), "lp_vs_ref_mean": float(diff.mean()),
                      "hf_bf16_lp_vs_ref_mean": float(bf_diff.mean()), "grads_vs_ref": report,
                      "hf_bf16_grads_vs_ref": noise, "chunked_vs_unchunked": chunk, "chunked_vs_ref": chunk_ref,
                      "loss": [runs[1]["loss"], runs[4]["loss"]]}, indent=1))
    assert effect > 0.01, effect
    assert diff.mean() <= max(1.5 * bf_diff.mean(), 0.1 * effect), (diff.mean(), bf_diff.mean())
    for k, (err, cos) in report.items():
        assert cos > 0.98 and err <= max(0.1, 1.5 * noise[k][0]), (k, err, cos, noise[k])
    # Chunked (the two-token carry at every chunk start): as close to the reference as unchunked. W_k and the
    # gains are second-order in the bf16 noise of the residual gradient, so chunked vs unchunked differs like two
    # independent bf16 runs (~sqrt(2) x the noise); W_v, which reads every boundary row directly, stays tight.
    # (measured: W_v 0.2% chunked vs unchunked, W_k / gains 4-10% where the bf16 reference itself is 5% off;
    # the existing LoRA gradients of the same model differ by up to 2.4% chunked vs unchunked).
    for k, (err, cos) in chunk_ref.items():
        assert cos > 0.98 and err <= max(0.15, 2.5 * noise[k][0]), (k, err, cos, noise[k])
    for k, (err, cos) in chunk.items():
        assert cos > 0.98 and err <= max(0.02, 2.5 * noise[k][0]), (k, err, cos)
    assert chunk[VALUE][0] <= 0.02 and chunk_ref[VALUE][0] <= max(0.02, 1.5 * noise[VALUE][0])
    # The carry: a wrong row for the first two tokens of chunks 1-3 would move the loss by ~0.05 x effect.
    assert abs(runs[1]["loss"] - runs[4]["loss"]) <= 0.01 * effect, (runs[1]["loss"], runs[4]["loss"], effect)


@pytest.mark.gpu
@pytest.mark.slow
def test_memory_lr_scale_and_no_weight_decay(tiny_model):
    """LoRA + memory, AdamW, weight decay 0.1, lr 1e-3, memory lr 2.5e-4: after one step W_k is exactly
    unchanged (zero gradient at W_v = 0 and no decay), |dW_v| = 2.5e-4 where its gradient is not tiny, a LoRA B
    moves by 1e-3 and a LoRA A with zero gradient decays."""
    from surogate import _surogate as sg

    seq = 64
    x, y = _tokens(seq, seed=9)
    opt = _memory_option(tiny_model, only=False, lr=2.5e-4, base_lr=1e-3)
    tr = _trainer(tiny_model, NON_EXPERT_TARGETS, seq, ngram=opt)  # fp32 masters (expert LoRA would need bf16)
    with tempfile.TemporaryDirectory() as tmp:  # the export holds the fp32 masters (get_lora_weights: bf16 work)
        tmp = Path(tmp)
        before = {k: v.float() for k, v in _adapter(tr, tiny_model, tmp, "before")[0].items()}
        tr.step(x, y)
        grads = _grads(tr)
        tr.update_with_config(sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.1), 1)
        after = {k: v.float() for k, v in _adapter(tr, tiny_model, tmp, "after")[0].items()}
    del tr
    torch.cuda.empty_cache()
    assert torch.equal(before[KEY], after[KEY])
    # AdamW's first step moves every parameter with a non-negligible gradient by the same |step| (its effective
    # learning rate); the memory's is 0.25 x the adapters'.
    lora_b = next(k for k in grads if k.endswith("lora_B.weight") and grads[k].abs().max() > 0)
    db = (after[lora_b] - before[lora_b]).abs()
    gb = grads[lora_b].abs() > 1e-3 * grads[lora_b].abs().max()
    step = float(db[gb].median())
    assert step > 0 and torch.allclose(db[gb], torch.full_like(db[gb], step), rtol=2e-2)
    dv = (after[VALUE] - before[VALUE]).abs()
    big = grads[VALUE].abs() > 1e-3 * grads[VALUE].abs().max()
    assert big.sum() > 100
    assert torch.allclose(dv[big], torch.full_like(dv[big], 0.25 * step), rtol=2e-2), (dv[big].min(), dv[big].max(), step)
    # (update_with_config(..., 1) runs Adam's step 2: |step| = lr * 0.526 / 0.707 = 0.744 lr.)
    assert abs(step / 1e-3 - 0.7441) < 0.01, step
    # Weight decay reaches the adapters (a zero-gradient lora_A shrinks by lr * wd) but not the memory (W_k above).
    lora_a = next(k for k in grads if k.endswith("lora_A.weight") and not grads[k].any() and before[k].abs().max() > 0)
    assert torch.allclose(after[lora_a], before[lora_a] * (1 - 1e-3 * 0.1), rtol=1e-6)
    assert not torch.allclose(after[lora_a], before[lora_a], rtol=1e-6, atol=0)


def _trigram_language(rng, n_seq, seq, alphabet, vocab=512):
    """Sequences over `alphabet` (token ids) with x[t+1] = f(x[t-1], x[t]) for a random f."""
    A = len(alphabet)
    f = rng.integers(0, A, size=(A, A))
    xs = []
    for _ in range(n_seq):
        s = list(rng.integers(0, A, size=2))
        while len(s) < seq + 1:
            s.append(int(f[s[-2], s[-1]]))
        xs.append(np.array(alphabet)[s])
    xs = np.array(xs, dtype=np.int32)
    return xs[:, :-1], xs[:, 1:].copy()


@pytest.mark.gpu
@pytest.mark.slow
def test_memory_alone_learns_an_ngram_language(tiny_model):
    """Frozen random backbone, memory only, a random frozen table: the next token is a function of the last
    two. The projections alone must cut the loss clearly (below half of the step-0 loss)."""
    from surogate import _surogate as sg

    seq = 64
    rng = np.random.default_rng(13)
    alphabet = list(rng.choice(np.arange(10, 499), size=12, replace=False))
    X, Y = _trigram_language(rng, 64, seq, alphabet)
    Y[:, :1] = -100  # the first target follows one token only
    tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model, init_std=0.02))
    held_x, held_y = X[-8:], Y[-8:]

    def held_loss():
        return float(np.mean([tr.validate(held_x[i:i + 1], held_y[i:i + 1]) for i in range(len(held_x))]))

    start = held_loss()
    losses = []
    for step in range(300):
        i = step % 56
        tr.step(X[i:i + 1], Y[i:i + 1])
        tr.update_with_config(sg.OptimizerConfig(optimizer="adamw", learning_rate=3e-3, weight_decay=0.0,
                                                 grad_clip=1.0), step + 1)
        if step % 50 == 49:
            losses.append(held_loss())
    stats = tr.get_ngram_memory_stats(0, True)
    end = losses[-1]
    print(json.dumps({"held_loss_start": start, "held_losses": losses, "stats": stats}))
    del tr
    torch.cuda.empty_cache()
    assert end < 0.5 * start, (start, losses)
    assert stats["out_rms_over_residual_rms"] > 0


@pytest.mark.gpu
@pytest.mark.slow
def test_export_round_trip_and_refusals(tiny_model):
    """One step trains the memory; the PEFT export carries it with its spec; the engine export holds the
    Flash-Next names, bf16 projections equal to the masters rounded, folded fp32 gains, zero conv, ones norm,
    hash constants; a fresh trainer re-imports the adapter exactly; a memory-less trainer and the merge refuse."""
    from safetensors.torch import load_file

    from surogate import _surogate as sg
    from surogate.dsl.ngram_memory_export import export_ngram_memory
    from surogate.utils.adapter_merge import AdapterMergeError, merge_adapter

    seq = 64
    x, y = _tokens(seq, seed=17)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
        tr.step(x, y)
        tr.update_with_config(sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3), 1)
        tr.step(x, y)
        tr.update_with_config(sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3), 2)
        lp = tr.compute_logprobs(x, y, use_lora=True)[0].copy()
        masters = {k: torch.from_dlpack(v).float().cpu().clone() for k, v in tr.get_lora_weights(0).items()}
        adapter, cfg = _adapter(tr, tiny_model, tmp, "trained")
        del tr
        torch.cuda.empty_cache()
        assert masters[VALUE].abs().sum() > 0 and masters[KEY].abs().sum() > 0
        for k in MEMORY:
            assert torch.equal(adapter[k].float(), masters[k]), k
        spec = cfg["surogate_ngram_memory"]
        assert spec["tensors"]["value_proj"] == VALUE and spec["norm_gain_convention"] == "folded"

        meta = export_ngram_memory(tmp / "trained", tmp / "engine", table_sha256="0" * 64, hidden_size=HIDDEN)
        out = load_file(str(tmp / "engine" / "ngram_memory.safetensors"))
        p = "model.language_model.layers.1.ple."
        assert set(out) == {p + n for n in ("key_proj.weight", "value_proj.weight", "norm_key.weight",
                                             "norm_query.weight", "norm_conv.weight", "conv1d.weight",
                                             "ple_embedding.layer_multipliers", "ple_embedding.ngram_heads_offsets",
                                             "ple_embedding.ngram_heads_vocab_sizes")}
        assert out[p + "key_proj.weight"].dtype == torch.bfloat16
        assert torch.equal(out[p + "key_proj.weight"], masters[KEY].to(torch.bfloat16))
        assert torch.equal(out[p + "value_proj.weight"], masters[VALUE].to(torch.bfloat16))
        assert torch.equal(out[p + "norm_key.weight"], masters[NORM_K])
        assert torch.equal(out[p + "norm_query.weight"], masters[NORM_Q])
        assert torch.equal(out[p + "norm_conv.weight"], torch.ones(HIDDEN))
        assert out[p + "conv1d.weight"].shape == (HIDDEN, 1, 4) and not out[p + "conv1d.weight"].any()
        tm = TableMeta.from_manifest(json.loads((Path(tiny_model) / "table_meta.json").read_text()))
        assert out[p + "ple_embedding.layer_multipliers"].tolist() == list(tm.multipliers)
        assert out[p + "ple_embedding.ngram_heads_offsets"].tolist() == list(tm.head_offsets)
        assert out[p + "ple_embedding.ngram_heads_vocab_sizes"].tolist() == list(tm.head_vocab_sizes)
        jm = json.loads((tmp / "engine" / "ngram_memory.json").read_text())
        assert jm == json.loads(json.dumps(meta))
        assert jm["format"] == "surogate-ngram-memory-v1" and jm["streams"] == 1 and jm["hidden"] == HIDDEN
        assert jm["embed_dim"] == 512 and jm["table"]["row_bytes"] == 18 and jm["table"]["sha256"] == "0" * 64
        assert jm["table"]["bytes"] == jm["table"]["file_bytes"] == tm.row_count * 18

        fresh = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
        fresh.import_adapter(str(tmp / "trained" / "adapter_model.safetensors"))
        assert np.array_equal(fresh.compute_logprobs(x, y, use_lora=True)[0], lp)
        base_lp = fresh.compute_logprobs(x, y, use_lora=False)[0].copy()
        del fresh
        torch.cuda.empty_cache()

        # The G2 scorer rebuilds the same trainer from the adapter's own spec and scores given ids.
        from surogate.dsl.ngram_memory_score import build_scorer, score_rows

        scorer = build_scorer(str(tiny_model), str(tmp / "trained"), seq)
        rec = next(score_rows(scorer, [{"id": 0, "ids": [int(t) for t in x[0]]}], seq))
        del scorer
        torch.cuda.empty_cache()
        assert rec["logprobs"][0] is None and len(rec["logprobs"]) == seq
        assert np.array_equal(np.array(rec["logprobs"][1:], dtype=np.float32), lp[: seq - 1])
        assert np.array_equal(np.array(rec["logprobs_base"][1:], dtype=np.float32), base_lp[: seq - 1])
        plain = _trainer(tiny_model, ["all"], seq)
        with pytest.raises(RuntimeError, match="carries an n-gram memory"):
            plain.import_adapter(str(tmp / "trained" / "adapter_model.safetensors"))
        del plain
        torch.cuda.empty_cache()
        with pytest.raises(AdapterMergeError, match="n-gram memory"):
            merge_adapter(str(tiny_model), str(tmp / "trained"), str(tmp / "merged"))


@pytest.mark.gpu
@pytest.mark.slow
def test_device_residency_matches_host(tiny_model):
    from safetensors.torch import save_file

    seq = 64
    x, y = _tokens(seq, seed=23)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for residency in ("host", "device"):
            tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model, residency=residency))
            init, _ = _adapter(tr, tiny_model, tmp, residency)
            save_file(_randomized(init, seed=3, memory_scale=0.05), str(tmp / "a.safetensors"))
            tr.import_adapter(str(tmp / "a.safetensors"))
            out[residency] = tr.compute_logprobs(x, y, use_lora=True)[0].copy()
            del tr
            torch.cuda.empty_cache()
    assert np.array_equal(out["host"], out["device"])


@pytest.mark.gpu
@pytest.mark.slow
def test_checkpoint_resume_is_exact(tiny_model):
    """Memory only, AdamW: two steps, a checkpoint, a third step; a fresh trainer resumed from the checkpoint
    gives the same loss before the third step and the same memory tensors after it, bit for bit (the
    preemptible H200 run resumes from such checkpoints)."""
    from surogate import _surogate as sg

    seq = 64
    xs = [_tokens(seq, seed=s) for s in (31, 32, 33)]
    opt = sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
        for step in (1, 2):
            tr.step(*xs[step - 1])
            tr.update_with_config(opt, step)
        tr.save_checkpoint(str(tmp / "ckpt"), 2)
        loss_a = tr.validate(*xs[2])
        tr.step(*xs[2])
        tr.update_with_config(opt, 3)
        after_a, _ = _adapter(tr, tiny_model, tmp, "a")
        del tr
        torch.cuda.empty_cache()

        tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
        tr.load_checkpoint(str(tmp / "ckpt"), 2)
        loss_b = tr.validate(*xs[2])
        tr.step(*xs[2])
        tr.update_with_config(opt, 3)
        after_b, _ = _adapter(tr, tiny_model, tmp, "b")
        del tr
        torch.cuda.empty_cache()
    assert np.float32(loss_a).tobytes() == np.float32(loss_b).tobytes(), (loss_a, loss_b)
    assert set(after_a) == set(MEMORY)
    for k in MEMORY:
        assert torch.equal(after_a[k], after_b[k]), k
    assert after_a[VALUE].abs().sum() > 0


# ---------------------------------------------------------------------------------------------- row shuffle (control)

SHUFFLE = {"a": 2654435761, "b": 97531}


class TestRowShuffleConfig:
    def test_off_is_the_spec_of_today_and_on_adds_two_keys(self):
        meta = _meta_dict(_small_meta())
        base = NgramMemoryConfig.parse({"table": "/x", "meta": meta})
        off = NgramMemoryConfig.parse({"table": "/x", "meta": meta, "row_shuffle": None})
        on = NgramMemoryConfig.parse({"table": "/x", "meta": meta, "row_shuffle": SHUFFLE})
        assert base.spec() == off.spec() and "row_shuffle_a" not in base.spec()
        assert set(on.spec()) - set(base.spec()) == {"row_shuffle_a", "row_shuffle_b"}
        assert {k: v for k, v in on.spec().items() if not k.startswith("row_shuffle")} == base.spec()
        assert NgramMemoryConfig.parse(on.to_dict()) == on
        cfg = _config()
        ir_base = _ir(cfg, ngram_memory={"table": "/x", "meta": meta})
        ir_off = _ir(cfg, ngram_memory={"table": "/x", "meta": meta, "row_shuffle": {}})
        assert ir_base["modules"][0]["config"]["ngram_memory_spec"] == ir_off["modules"][0]["config"]["ngram_memory_spec"]

    def test_validation_and_reference(self):
        from surogate.dsl.modules.ngram_memory import parse_row_shuffle, shuffle_rows

        R = SWIFT15.row_count
        assert parse_row_shuffle(SHUFFLE, R) == (2654435761, 97531)
        for bad in ({"a": 2}, {"a": 0}, {"a": 2**33 + 1}, {"a": 3, "b": -1}, {"c": 1}):
            with pytest.raises(ValueError):
                parse_row_shuffle(bad, R)
        with pytest.raises(ValueError, match="not a bijection"):
            parse_row_shuffle({"a": 3}, 3 * 101)
        rows = np.arange(0, R, 9973, dtype=np.int64)
        got = shuffle_rows(rows, (2654435761, 97531), R)
        want = np.array([(int(r) * 2654435761 + 97531) % R for r in rows])
        assert np.array_equal(got, want)
        # a bijection of the rows (small table, full check)
        small = 16 * 1009 + 240
        perm = shuffle_rows(np.arange(small), (2654435761, 97531), small)
        assert len(np.unique(perm)) == small


@pytest.mark.gpu
def test_gather_with_row_shuffle_is_the_remapped_decode():
    sg = _gpu()
    from surogate.dsl.modules.ngram_memory import shuffle_rows

    meta = _small_meta(vocab=211, head_dim=64)
    raw = _random_table(meta, seed=2)
    table = torch.from_numpy(raw.reshape(-1)).cuda()
    rng = np.random.default_rng(5)
    N = 513
    rows = rng.integers(0, meta.row_count, size=(N, meta.heads)).astype(np.int32)
    out = {}
    for label, sh in (("off", None), ("on", SHUFFLE)):
        cfg = NgramMemoryConfig.parse({"table": "/unused", "meta": _meta_dict(meta), "row_shuffle": sh})
        emb = torch.empty((N, meta.embed_dim), dtype=torch.bfloat16, device="cuda")
        sg._ngram_memory_gather(torch.from_numpy(rows).cuda(), table, emb, json.dumps(cfg.spec()))
        out[label] = emb.cpu()
    plain = torch.from_numpy(iq4nl_decode_rows(raw[rows.reshape(-1)], meta.head_dim).reshape(N, -1)).to(torch.bfloat16)
    remapped_rows = shuffle_rows(rows.reshape(-1), (SHUFFLE["a"], SHUFFLE["b"]), meta.row_count)
    remapped = torch.from_numpy(iq4nl_decode_rows(raw[remapped_rows], meta.head_dim).reshape(N, -1)).to(torch.bfloat16)
    assert torch.equal(out["off"], plain)
    assert torch.equal(out["on"], remapped)
    assert not torch.equal(out["on"], out["off"])


@pytest.mark.gpu
@pytest.mark.slow
def test_row_shuffle_is_a_permuted_table_end_to_end(tiny_model):
    """The shuffled arm on table T is exactly the plain memory on the permuted table T'[r] = T[(r a + b) mod R]:
    per-token log-probs and every memory gradient bit for bit (the whole trainer path, not just the gather).
    The shuffled adapter records its shuffle; a plain trainer refuses it and the engine export refuses it."""
    from safetensors.torch import save_file

    from surogate.dsl.modules.ngram_memory import shuffle_rows
    from surogate.dsl.ngram_memory_export import ShuffledMemoryError, export_ngram_memory

    meta = TableMeta.from_manifest(json.loads((Path(tiny_model) / "table_meta.json").read_text()))
    raw = np.frombuffer((Path(tiny_model) / "table.iq4nl").read_bytes(), dtype=np.uint8).reshape(meta.row_count, -1)
    perm = shuffle_rows(np.arange(meta.row_count), (SHUFFLE["a"], SHUFFLE["b"]), meta.row_count)
    (Path(tiny_model) / "table_perm.iq4nl").write_bytes(raw[perm].tobytes())
    seq = 128
    x, y = _tokens(seq, seed=41)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for label, opt in (("shuffled", _memory_option(tiny_model, row_shuffle=SHUFFLE)),
                           ("permuted", {**_memory_option(tiny_model), "table": str(Path(tiny_model) / "table_perm.iq4nl")}),
                           ("plain", _memory_option(tiny_model))):
            tr = _trainer(tiny_model, [], seq, ngram=opt)
            init, cfg = _adapter(tr, tiny_model, tmp, label)
            live = _randomized(init, seed=7, memory_scale=0.05)
            save_file(live, str(tmp / label / "adapter_model.safetensors"))  # beside its own adapter_config.json
            tr.import_adapter(str(tmp / label / "adapter_model.safetensors"))
            out[label] = {"lp": tr.compute_logprobs(x, y, use_lora=True)[0].copy(), "cfg": cfg}
            tr.step(x, y)
            out[label]["grads"] = _grads(tr)
            del tr
            torch.cuda.empty_cache()
        assert out["shuffled"]["cfg"]["surogate_ngram_memory"]["row_shuffle_a"] == SHUFFLE["a"]
        assert "row_shuffle_a" not in out["plain"]["cfg"]["surogate_ngram_memory"]
        assert np.array_equal(out["shuffled"]["lp"], out["permuted"]["lp"])
        for k in MEMORY:
            assert torch.equal(out["shuffled"]["grads"][k], out["permuted"]["grads"][k]), k
        assert not np.array_equal(out["shuffled"]["lp"], out["plain"]["lp"])
        # refusals: a plain trainer cannot load the shuffled adapter (and vice versa); the export refuses it
        tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
        with pytest.raises(RuntimeError, match="row_shuffle"):
            tr.import_adapter(str(tmp / "shuffled" / "adapter_model.safetensors"))
        del tr
        tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model, row_shuffle=SHUFFLE))
        with pytest.raises(RuntimeError, match="row_shuffle"):
            tr.import_adapter(str(tmp / "plain" / "adapter_model.safetensors"))
        del tr
        torch.cuda.empty_cache()
        with pytest.raises(ShuffledMemoryError):
            export_ngram_memory(tmp / "shuffled", tmp / "export")
        meta_json = export_ngram_memory(tmp / "shuffled", tmp / "export_analysis", allow_shuffled=True)
        assert meta_json["servable"] is False and meta_json["row_shuffle"] == SHUFFLE
        assert export_ngram_memory(tmp / "plain", tmp / "export_plain")["servable"] is True


# ---------------------------------------------------------------------------------------------- chunked log-probs, tail skip


@pytest.mark.gpu
@pytest.mark.slow
def test_chunked_logprobs_and_validate_tail_skip(tiny_model):
    """compute_logprobs over sequence chunks = the unchunked one (bf16 tolerance) and its mean over the scored
    tokens = validate; with the scored tokens in chunk 0 only, validate is unchanged bit for bit by any tokens in
    the later chunks (they are not run, and could not change chunk 0 anyway)."""
    from safetensors.torch import save_file

    seq = 128
    x, y = _tokens(seq, seed=43)
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for chunks in (1, 4):
            tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model), chunks=chunks)
            init, _ = _adapter(tr, tiny_model, tmp, f"c{chunks}")
            save_file(_randomized(init, seed=9, memory_scale=0.05), str(tmp / f"c{chunks}" / "adapter_model.safetensors"))
            tr.import_adapter(str(tmp / f"c{chunks}" / "adapter_model.safetensors"))
            lp = tr.compute_logprobs(x, y, use_lora=True)[0].astype(np.float64)
            r = {"lp": lp, "val": tr.validate(x, y)}
            if chunks == 4:
                y0 = y.copy()
                y0[0, 30:] = -100
                x2 = x.copy()
                x2[0, 40:] = np.random.default_rng(1).integers(10, 500, size=seq - 40)
                r["val0"], r["val0b"] = tr.validate(x, y0), tr.validate(x2, y0)
                lp0 = tr.compute_logprobs(x, y0, use_lora=True)[0].astype(np.float64)
                r["lp0"] = lp0
            res[chunks] = r
            del tr
            torch.cuda.empty_cache()
    valid = y[0] != -100
    for c in (1, 4):
        nll = -res[c]["lp"][valid].mean()
        assert abs(nll - res[c]["val"]) < 2e-3 * abs(res[c]["val"]), (c, nll, res[c]["val"])
    rel = np.abs(res[4]["lp"][valid] - res[1]["lp"][valid]).mean() / np.abs(res[1]["lp"][valid]).mean()
    assert rel < 2e-2, rel
    assert np.float32(res[4]["val0"]).tobytes() == np.float32(res[4]["val0b"]).tobytes()
    assert np.array_equal(res[4]["lp0"][:30], res[4]["lp"][:30])  # chunk 0 alone gives chunk 0's log-probs
    assert not res[4]["lp0"][40:].any()                          # the skipped chunks are left at 0


# ---------------------------------------------------------------------------------------------- eval panels


def test_bin_windows_and_ngram_counts(tmp_path):
    from surogate.train.eval_panels import NgramCounts, context_keys, read_bin_windows
    from surogate.train.tokenize import TokenizedDataFileWriter

    rng = np.random.default_rng(3)
    docs = [rng.integers(0, 50, size=n).astype(np.int32) for n in (17, 40, 5)]
    docs[1][10] = 500  # an EOS inside a document cuts the context
    T = 48
    path = tmp_path / "train-000.bin"
    with TokenizedDataFileWriter(str(path), 1000, masking=True, non_overlapping=True) as w:
        for d in docs:
            tok = np.pad(d, (0, T - len(d)), constant_values=500)  # EOS padding with a zero mask
            mask = np.pad(np.ones(len(d), dtype=np.int32), (0, T - len(d)))
            w.add_document(tokens=tok, position_ids=np.arange(T), mask=mask)
    wins = read_bin_windows(str(path), eos=500)
    assert [list(a) for a in wins] == [list(d) for d in docs]
    counts = NgramCounts(wins, eos=500)
    brute_bi, brute_tri = collections.Counter(), collections.Counter()
    for d in docs:
        for j in range(len(d)):
            c1 = 500 if j < 1 or d[j - 1] == 500 else int(d[j - 1])
            c2 = 500 if (j < 2 or c1 == 500 or d[j - 2] == 500) else int(d[j - 2])
            brute_bi[(int(d[j]), c1)] += 1
            brute_tri[(int(d[j]), c1, c2)] += 1
    q = docs[0]
    bi, tri = counts.lookup(q, 500)
    for j in range(len(q)):
        c1 = 500 if j < 1 or q[j - 1] == 500 else int(q[j - 1])
        c2 = 500 if (j < 2 or c1 == 500 or q[j - 2] == 500) else int(q[j - 2])
        assert bi[j] == brute_bi[(int(q[j]), c1)] and tri[j] == brute_tri[(int(q[j]), c1, c2)], j
    b2, t2 = counts.lookup(np.array([999, 998, 997]), 500)
    assert not b2.any() and not t2.any()


@pytest.mark.gpu
@pytest.mark.slow
def test_eval_panels_rows_and_tokens(tiny_model, tmp_path):
    """EvalPanels on pre-tokenized rows: per-row NLL = the mean of the per-token log-probs it dumps = validate;
    the per-token dump carries the training counts of the n-grams ending at each scored input position."""
    from types import SimpleNamespace

    from surogate.train.eval_panels import EvalPanels, NgramCounts, read_bin_windows
    from surogate.train.tokenize import TokenizedDataFileWriter

    seq = 128
    rng = np.random.default_rng(11)
    train = [rng.integers(10, 60, size=seq).astype(np.int32) for _ in range(3)]
    with TokenizedDataFileWriter(str(tmp_path / "train-000.bin"), 512, masking=True, non_overlapping=True) as w:
        for d in train:
            w.add_document(tokens=d, position_ids=np.arange(seq), mask=np.ones(seq, dtype=np.int32))
    rows = [{"id": f"r{k}", "ids": rng.integers(10, 60, size=n).tolist(), "spans": [[3, n]]} for k, n in enumerate((90, 128, 40))]
    (tmp_path / "panel.tok.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tr = _trainer(tiny_model, [], seq, ngram=_memory_option(tiny_model))
    cfg = SimpleNamespace(sequence_len=seq, eval_steps=4, eval_panels_steps=2, output_dir=str(tmp_path), gpus=1,
                          per_device_train_batch_size=1, ngram_memory={"meta": {"eos_token_id": 500}},
                          eval_panels=[{"name": "P_in_test", "path": str(tmp_path / "panel.tok.jsonl"), "per_token": True}],
                          model_dir=str(tiny_model))
    panels = EvalPanels(cfg, tr)
    assert panels.due(0, 0) and panels.due(2, 0) and not panels.due(3, 0)
    res = panels.run(0)
    val = []
    for r in rows:
        ids = np.asarray(r["ids"], dtype=np.int32)
        x = np.zeros((1, seq), dtype=np.int32)
        y = np.full((1, seq), -100, dtype=np.int32)
        x[0, : len(ids)] = ids
        y[0, 2: len(ids) - 1] = ids[3:]
        val.append(tr.validate(x, y))
    del tr
    torch.cuda.empty_cache()
    out = [json.loads(l) for l in open(tmp_path / "eval_panels.jsonl")]
    assert [o["id"] for o in out] == ["r0", "r1", "r2"] and all(o["step"] == 0 for o in out)
    for o, v, r in zip(out, val, rows):
        assert o["tokens"] == len(r["ids"]) - 3
        assert abs(o["nll"] - v) < 2e-3 * abs(v), (o, v)
    toks = [json.loads(l) for l in gzip.open(tmp_path / "eval_tokens_step0.jsonl.gz", "rt")]
    counts = NgramCounts(read_bin_windows(str(tmp_path / "train-000.bin"), eos=500), 500)
    for t, o, r in zip(toks, out, rows):
        assert t["pos"] == list(range(2, len(r["ids"]) - 1))
        assert abs(-np.mean(t["lp"]) - o["nll"]) < 1e-5
        bi, tri = counts.lookup(np.asarray(r["ids"]), 500)
        assert t["bi"] == bi[t["pos"]].tolist() and t["tri"] == tri[t["pos"]].tolist()
    assert any(c > 0 for t in toks for c in t["bi"])
    summ = [json.loads(l) for l in open(tmp_path / "eval_panels_summary.jsonl")]
    assert summ[0]["panel"] == "P_in_test" and summ[0]["rows"] == 3 and "memory" in summ[0]
    assert abs(res["P_in_test"] - summ[0]["nll"]) < 1e-12


def test_compare_script(tmp_path):
    """train/compare.py on two synthetic arms: step 0 identical, a planted real - shuf difference recovered."""
    import subprocess
    import sys

    script = Path(os.environ.get("NGRAM_COMPARE_SCRIPT", "/mnt/data/work/rl/ngram-transplant-2026-10-18/train/compare.py"))
    if not script.exists():
        pytest.skip("compare.py is outside the repo (rl/ngram-transplant-2026-10-18/train)")
    rng = np.random.default_rng(0)
    for arm, shift in (("real", -0.05), ("shuf", 0.0)):
        d = tmp_path / arm
        d.mkdir()
        with open(d / "eval_panels.jsonl", "w") as f:
            for step in (0, 4):
                for panel in ("P_in_a", "P_teach", "P_ret_b"):
                    for k in range(30):
                        base = 1.0 + 0.01 * k
                        planted = {"P_in_a": shift, "P_teach": 2 * shift}.get(panel, 0.0)
                        nll = base + (planted if step == 4 else 0.0) + (0.01 if step == 4 else 0.0)
                        f.write(json.dumps({"step": step, "panel": panel, "id": str(k), "tokens": 100 + k,
                                            "nll": nll}) + "\n")
        for step in (0, 4):
            with gzip.open(d / f"eval_tokens_step{step}.jsonl.gz", "wt") as f:
                for panel, gap in (("P_in_a", 0.05), ("P_teach", 0.10)):
                    for k in range(30):
                        lp = (-np.ones(10) * (1.0 + (gap if (arm == "shuf" or step == 0) else 0.0))).tolist()
                        f.write(json.dumps({"step": step, "panel": panel, "id": str(k), "pos": list(range(10)),
                                            "lp": lp, "bi": [0, 1, 2, 5, 9, 0, 3, 7, 1, 0], "tri": [0] * 10}) + "\n")
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "real"), str(tmp_path / "shuf"), "--out",
                        str(tmp_path / "c.json"), "--boot", "500"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "c.json").read_text())
    assert rep["step0_max_abs_diff"] == 0.0
    assert abs(rep["final"]["P_in"]["real-shuf"]["mean"] + 0.05) < 1e-9
    lo, hi = rep["final"]["P_in"]["real-shuf"]["ci95"]
    assert lo - 1e-9 <= -0.05 <= hi + 1e-9
    assert abs(rep["final"]["P_ret"]["real-shuf"]["mean"]) < 1e-12
    assert abs(rep["strata"]["bi"]["unseen"]["real-shuf"]["mean"] + 0.05) < 1e-9          # the pre-P_teach layout = P_in
    assert abs(rep["strata"]["P_in"]["tri"]["unseen"]["real-shuf"]["mean"] + 0.05) < 1e-9
    assert abs(rep["strata"]["P_teach"]["bi"]["unseen"]["real-shuf"]["mean"] + 0.10) < 1e-9
    assert abs(rep["final"]["P_teach"]["real-shuf"]["mean"] + 0.10) < 1e-9


# ---------------------------------------------------------------------------------------------- the full-step path
# The SFT CLI never calls tr.step: every optimizer step is train_step_graphed, which for sequence_chunks > 1 runs its
# own eager chunk loop (a KV sweep left to right, then a reverse re-forward + backward per chunk). Its chunks need the
# same two-token carry as step_chunked's (the H200 smoke of 2026-10-18 stopped at "chunk 1 has no token carry").


def _graphed_rows(seq, n, seed, chunks=4):
    """n rows of seq tokens for one optimizer step; the last restarts its positions inside chunk 1 and exactly at the
    start of chunk 2 (document cuts the memory honors across the carry, as packed rows have them)."""
    xs, ys = zip(*(_tokens(seq, seed=seed + i) for i in range(n)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    pos = np.tile(np.arange(seq, dtype=np.int32), (n, 1))
    T = seq // chunks
    for start in (T + 7, 2 * T):
        pos[-1, start:] = np.arange(seq - start, dtype=np.int32)
    return x, y, pos


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("recipe", ["bf16", "fp8-hybrid"])
def test_full_step_zero_init_is_the_base_model_bit_for_bit(tiny_model, recipe):
    """train_step_graphed with sequence_chunks 4, grad accumulation 3 and document cuts inside a chunk and at a chunk
    start, 3 optimizer steps. W_v = 0: the first step's loss and every LoRA gradient equal the memory-off run bit for
    bit, W_k's and the gains' gradients are exactly zero and W_v's is live; the update leaves the LoRA tensors as in
    the base run (no clipping) and W_k / gains untouched, makes W_v live, and the next losses move."""
    from safetensors.torch import save_file

    sg = _gpu()
    seq, ga = 128, 3
    x, y, pos = _graphed_rows(seq, ga, seed=21)
    opt = sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.0)  # grad_clip 0 = off
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for label, ngram in (("base", None), ("memory", _memory_option(tiny_model, only=False))):
            tr = _trainer(tiny_model, ["all"], seq, ngram=ngram, chunks=4, lora_dtype="bf16", recipe=recipe,
                          grad_accum=ga)
            init, _ = _adapter(tr, tiny_model, tmp, label)
            live = _randomized(init, seed=11)
            save_file(live, str(tmp / f"{label}.safetensors"))
            tr.import_adapter(str(tmp / f"{label}.safetensors"))
            res = {"live": live, "steps": []}
            for s in range(3):
                r = tr.train_step_graphed(x, y, pos, opt, s + 1)
                res["steps"].append({"loss": float(r["loss"]), "norm": float(r["norm"]),
                                     "grads": _grads(tr) if s == 0 else None})
                if s == 0:
                    res["after1"] = _adapter(tr, tiny_model, tmp, label + "_after1")[0]
            if ngram:
                res["stats"] = tr.get_ngram_memory_stats(0, True)
            results[label] = res
            del tr
            torch.cuda.empty_cache()
    base, on = results["base"], results["memory"]
    b0, m0 = base["steps"][0], on["steps"][0]
    print(json.dumps({"loss": [[s["loss"] for s in base["steps"]], [s["loss"] for s in on["steps"]]],
                      "norm": [b0["norm"], m0["norm"]], "stats": on["stats"]}))
    assert np.float32(b0["loss"]).tobytes() == np.float32(m0["loss"]).tobytes(), (b0["loss"], m0["loss"])
    common = set(b0["grads"]) & set(m0["grads"])
    assert common == set(b0["grads"]) and set(m0["grads"]) - common == set(MEMORY)
    for k in sorted(common):
        assert torch.equal(b0["grads"][k], m0["grads"][k]), k
    assert m0["grads"][VALUE].norm() > 0
    assert not m0["grads"][KEY].any() and not m0["grads"][NORM_K].any() and not m0["grads"][NORM_Q].any()
    assert m0["norm"] > b0["norm"]  # the global norm covers W_v's gradient too
    for k in base["after1"]:
        assert torch.equal(base["after1"][k], on["after1"][k]), k
    for k in (KEY, NORM_K, NORM_Q):
        assert torch.equal(on["after1"][k], on["live"][k]), k
    assert on["after1"][VALUE].float().abs().max() > 0
    for s in (1, 2):
        lb, lm = base["steps"][s]["loss"], on["steps"][s]["loss"]
        assert math.isfinite(lb) and math.isfinite(lm) and lb != lm, (s, lb, lm)
    assert on["stats"]["tokens"] > 0 and on["stats"]["out_rms_over_residual_rms"] > 0.0


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("arm", ["real", "shuf"])
def test_full_step_carry_is_step_chunked_and_unchunked(tiny_model, arm):
    """Live memory (W_v != 0, perturbed gains), memory-only PEFT with fp32 masters, the real table and the
    row-shuffled control arm, 3 rows per optimizer step.
    (a) The quick configs' recipe (fp8-hybrid), 4 chunks: the memory gradients of train_step_graphed equal those of
        the eager chunked step (tr.step per row) bit for bit -- the full-step loop gives every chunk the carry
        step_chunked gives it (test_live_memory_matches_an_fp32_reference_chunked_and_not ties that one to the
        unchunked step and an fp32 reference). The two paths' reported loss / norm differ by step_chunked's own
        reductions (LoRA-only too), so only the gradients are compared.
    (b) bf16, document cuts inside a chunk and at a chunk start, 2 optimizer steps: the chunked full step agrees
        with the unchunked one (doc-masked attention) in loss, norm, memory gradients and the updated memory."""
    from safetensors.torch import save_file

    sg = _gpu()
    seq, ga = 128, 3
    opt = sg.OptimizerConfig(optimizer="adamw", learning_rate=1e-3, weight_decay=0.0, grad_clip=1.0)
    ngram = _memory_option(tiny_model, only=True, **({"row_shuffle": SHUFFLE} if arm == "shuf" else {}))
    x, y, pos_cut = _graphed_rows(seq, ga, seed=31)
    pos_plain = np.tile(np.arange(seq, dtype=np.int32), (ga, 1))
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for label, chunks, recipe, pos, path, n_steps in (("graphed4_fp8", 4, "fp8-hybrid", pos_plain, "graphed", 1),
                                                          ("eager4_fp8", 4, "fp8-hybrid", pos_plain, "eager", 1),
                                                          ("graphed4", 4, "bf16", pos_cut, "graphed", 2),
                                                          ("graphed1", 1, "bf16", pos_cut, "graphed", 2)):
            tr = _trainer(tiny_model, [], seq, ngram=ngram, chunks=chunks, recipe=recipe, grad_accum=ga)
            init, cfg = _adapter(tr, tiny_model, tmp, label)
            assert set(init) == set(MEMORY)
            assert ("row_shuffle_a" in cfg["surogate_ngram_memory"]) == (arm == "shuf")
            live = _randomized(init, seed=5, memory_scale=0.05)
            save_file(live, str(tmp / label / "adapter_model.safetensors"))  # beside its adapter_config.json
            tr.import_adapter(str(tmp / label / "adapter_model.safetensors"))
            steps = []
            for s in range(n_steps):
                if path == "eager":
                    for i in range(ga):
                        tr.step(x[i:i + 1], y[i:i + 1], pos[i:i + 1])
                    r = tr.update_with_config(opt, s + 1)
                else:
                    r = tr.train_step_graphed(x, y, pos, opt, s + 1)
                steps.append({"loss": float(r["loss"]), "norm": float(r["norm"]), "grads": _grads(tr)})
            runs[label] = {"steps": steps, "live": live, "after": _adapter(tr, tiny_model, tmp, label + "_after")[0]}
            del tr
            torch.cuda.empty_cache()

    def rel(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float((a - b).norm() / b.norm().clamp_min(1e-30)), float(torch.nn.functional.cosine_similarity(a, b, 0))

    g8, e8 = runs["graphed4_fp8"]["steps"][0], runs["eager4_fp8"]["steps"][0]
    c4, c1 = runs["graphed4"], runs["graphed1"]
    report = {"fp8": {"graphed4": (g8["loss"], g8["norm"]), "eager4": (e8["loss"], e8["norm"])},
              "bf16": [{"loss": (c4["steps"][s]["loss"], c1["steps"][s]["loss"]),
                        "norm": (c4["steps"][s]["norm"], c1["steps"][s]["norm"]),
                        "grads": {k.split(".")[-2]: rel(c4["steps"][s]["grads"][k], c1["steps"][s]["grads"][k])
                                  for k in MEMORY}} for s in range(2)],
              "update": {k.split(".")[-2]: rel(c4["after"][k].float() - c4["live"][k].float(),
                                               c1["after"][k].float() - c1["live"][k].float()) for k in MEMORY}}
    print(json.dumps(report))
    # (a) the carry of the full-step loop = step_chunked's, bit for bit
    for k in MEMORY:
        assert torch.equal(g8["grads"][k], e8["grads"][k]), k
    assert g8["grads"][KEY].norm() > 0 and g8["grads"][VALUE].norm() > 0
    # (b) chunked = unchunked within bf16 noise. Measured (real / shuf): loss 5e-6 / 6e-5, norm <= 1.2e-3 relative,
    # W_v 0.3-0.5%; W_k and the gains, second-order in the residual gradient's bf16 noise (see the live-memory test),
    # 2-3% at step 1 and 15-18% (cosine >= 0.984) at step 2; the updates' cosines >= 0.989.
    for s in range(2):
        l4, l1 = c4["steps"][s]["loss"], c1["steps"][s]["loss"]
        n4, n1 = c4["steps"][s]["norm"], c1["steps"][s]["norm"]
        assert abs(l4 - l1) <= 2e-4, (s, l4, l1)
        assert abs(n4 - n1) <= 5e-3 * n1, (s, n4, n1)
        for k, (err, cos) in report["bf16"][s]["grads"].items():
            if k == "value_proj":
                assert cos > 0.9999 and err <= 0.01, (s, k, err, cos)
            else:
                assert cos > 0.97 and err <= 0.25, (s, k, err, cos)
    for k in MEMORY:
        assert torch.equal(c4["live"][k], c1["live"][k]), k
    for k, (err, cos) in report["update"].items():  # the two updates (Adam's first steps ~ lr * sign(g))
        assert cos > (0.999 if k == VALUE.split(".")[-2] else 0.98), (k, err, cos)
    assert c4["steps"][1]["loss"] != c4["steps"][0]["loss"]


# ---------------------------------------------------------------------------------------------- the SFT CLI end to end
# The H200 configs (rl/ngram-transplant-2026-10-18/train/{smoke,quick_real,quick_shuf}.yaml) on a tiny full-vocabulary
# qwen3_5_moe, a small table with Swift-1.5's hash constants and synthetic agentic conversations long enough to fill
# all 32 chunks. Every option of the H200 config is kept except paths and sizes (_TINY_SIZES). NGRAM_SFT_TEMPLATES=<dir>
# reads the configs from that directory instead of the copies below. Needs a Qwen3.5 tokenizer: NGRAM_TOKENIZER_DIR or
# QWEN3_5_MODEL_PATH.

_SMOKE_TEMPLATE = """
model: /mnt/data/work/merges/surogate37-wp1
output_dir: /mnt/data/work/rl/sft/runs/ngram-transplant-smoke
resume_from_checkpoint: false
save_steps: 1000
save_total_limit: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
gpus: 1
sample_packing: false
sequence_len: 131072
sequence_chunks: 32
lmhead_chunks: 512
num_epochs: 2
max_steps: 3
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_steps: 1
final_lr_fraction: 0.1
max_grad_norm: 1.0
weight_decay: 0.0
optimizer: adamw
adamw_beta1: 0.9
adamw_beta2: 0.999
recipe: fp8-hybrid
lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: fp32
lora_dropout: 0.0
recompute: true
train_router: false
train_vision: false
offload_residual: false
cpu_training: false
qlora_fp8: false
skip_quant_first_layers: 1
skip_quant_last_layers: 2
ngram_memory:
  table: /mnt/data/work/rl/ngram-transplant-2026-10-18/swift15-ple/table.iq4nl
  manifest: /mnt/data/work/rl/ngram-transplant-2026-10-18/swift15-ple/manifest.json
  only: true
  layer: 1
  init_std: 0.02
  table_residency: host
  zero_token_ids: [248056, 248057]
validation_split_ratio: 0.0
eval_steps: 1000
logging_steps: 1
dataloader_num_workers: 4
datasets:
  - {path: /mnt/data/work/rl/ngram-transplant-2026-10-18/train/smoke_rows.jsonl, type: conversation,
     messages_field: messages, tools_field: tools}
"""

_QUICK_TEMPLATE = """
model: /mnt/data/work/merges/surogate37-wp1
output_dir: /mnt/data/work/rl/sft/runs/ngram-quick-real
resume_from_checkpoint: true
save_steps: 4
save_total_limit: 2
train_seed: 1234
eval_seed: 1234
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gpus: 1
sample_packing: false
sequence_len: 131072
sequence_chunks: 32
lmhead_chunks: 512
num_epochs: 1
max_steps: 17
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_steps: 2
final_lr_fraction: 0.1
max_grad_norm: 1.0
weight_decay: 0.0
optimizer: adamw
adamw_beta1: 0.9
adamw_beta2: 0.999
recipe: fp8-hybrid
lora: true
lora_rank: 16
lora_alpha: 32
lora_dtype: fp32
lora_dropout: 0.0
recompute: true
train_router: false
train_vision: false
offload_residual: false
cpu_training: false
qlora_fp8: false
skip_quant_first_layers: 1
skip_quant_last_layers: 2
ngram_memory:
  table: /mnt/data/work/rl/ngram-transplant-2026-10-18/swift15-ple/table.iq4nl
  manifest: /mnt/data/work/rl/ngram-transplant-2026-10-18/swift15-ple/manifest.json
  only: true
  layer: 1
  init_std: 0.02
  table_residency: host
  zero_token_ids: [248056, 248057]
validation_split_ratio: 0.0
eval_steps: 1000
eval_panels:
  - {name: P_in_workplace, path: /d/pin_workplace.jsonl, per_token: true}
  - {name: P_in_telecom, path: /d/pin_telecom.jsonl, per_token: true}
  - {name: P_in_terminal, path: /d/pin_terminal.jsonl, per_token: true}
  - {name: P_teach, path: /d/pteach.jsonl, per_token: true}
  - {name: P_seen, path: /d/pseen.jsonl}
  - {name: P_ret_mergecheck, path: /d/pret_mergecheck.tok.jsonl}
  - {name: P_ret_ifeval, path: /d/pret_ifeval.jsonl}
  - {name: P_ret_bfcl, path: /d/pret_bfcl.jsonl}
eval_panels_steps: 4
logging_steps: 1
dataloader_num_workers: 4
datasets:
  - {path: /d/train_w1.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_t1.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_box_telecom.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_south_telecom.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_box_terminal.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_south_terminal.jsonl, type: conversation, messages_field: messages, tools_field: tools}
  - {path: /d/train_own_rst.jsonl, type: conversation, messages_field: messages, tools_field: tools}
"""

# H200: 131072 tokens in 32 chunks of 4096, 512 logit chunks of 8 tokens. Here: 1024 in 32 chunks of 32, 4 of 8.
_TINY_SIZES = {"sequence_len": 1024, "lmhead_chunks": 4}

_WORDS = ("the order account invoice ticket server deploy config network user status report quarterly customer update "
          "database query result error retry timeout cluster region policy budget meeting schedule draft review "
          "approve reject refund shipment warehouse inventory forecast metric latency throughput line plan device "
          "billing roaming data voice outage restart service log file directory permission process").split()
_TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a record by id.",
                                            "parameters": {"type": "object",
                                                           "properties": {"id": {"type": "string"}},
                                                           "required": ["id"]}}}]


def _cli_text(rng, words):
    out = []
    for _ in range(words):
        out.append(_WORDS[int(rng.integers(len(_WORDS)))])
        if rng.random() < 0.08:
            out.append(f"#{int(rng.integers(1000))}.")
    return " ".join(out)


def _cli_conversation(rng, k, words, tools=True):
    msgs = [{"role": "system", "content": "You are a workplace assistant."},
            {"role": "user", "content": f"Task {k}: " + _cli_text(rng, 25)},
            {"role": "assistant", "content": _cli_text(rng, max(words // 3, 5))},
            {"role": "tool", "content": json.dumps({"id": str(k), "status": _cli_text(rng, 12)})},
            {"role": "assistant", "content": _cli_text(rng, words)}]
    row = {"id": f"row{k}", "messages": msgs}
    if tools:
        row["tools"] = _TOOLS
    else:
        msgs.pop(0)
        del msgs[1:3]  # user, assistant: a plain chat row (the IFEval / BFCL retention panels)
    return row


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture(scope="module")
def cli_model(tmp_path_factory):
    """A random 4-layer full-vocabulary qwen3_5_moe with the Qwen3.5 tokenizer + a 16-head IQ4_NL table (head_dim 160)
    described by a Swift-1.5 manifest with its own head sizes."""
    import shutil

    _gpu()
    pytest.importorskip("transformers")
    pytest.importorskip("yaml")
    import transformers

    tok_dir = os.environ.get("NGRAM_TOKENIZER_DIR") or os.environ.get("QWEN3_5_MODEL_PATH")
    if not tok_dir or not (Path(tok_dir) / "tokenizer.json").exists():
        pytest.skip("SFT CLI test: needs a Qwen3.5 tokenizer (NGRAM_TOKENIZER_DIR or QWEN3_5_MODEL_PATH)")
    root = tmp_path_factory.mktemp("ngram_cli")
    mdir = root / "model"
    text = _text_config(n_layers=4, hidden=256, experts=4)
    text.update(vocab_size=VOCAB, num_experts_per_tok=2, head_dim=64, eos_token_id=EOS,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                 "mrope_section": [4, 2, 2], "mrope_interleaved": True})
    vision = {"depth": 1, "hidden_size": 64, "intermediate_size": 128, "num_heads": 4, "out_hidden_size": 256,
              "num_position_embeddings": 256, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
              "in_channels": 3}
    cfg = transformers.Qwen3_5MoeConfig(text_config=text, vision_config=vision, tie_word_embeddings=False,
                                        image_token_id=248056, video_token_id=248057)
    torch.manual_seed(1)
    model = transformers.Qwen3_5MoeForConditionalGeneration(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p, dtype=torch.float32).uniform_(0.5, 2.0)).to(p.dtype))
            elif name.endswith("dt_bias"):
                p.zero_()
            elif p.dim() == 1:
                p.copy_((0.1 * torch.randn_like(p, dtype=torch.float32)).to(p.dtype))
    model.save_pretrained(mdir, safe_serialization=True)
    del model
    _fuse_experts(mdir, 4)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
              "special_tokens_map.json"):
        if (Path(tok_dir) / f).exists():
            shutil.copy(Path(tok_dir) / f, mdir / f)
    sizes = [4099 + 2 * h for h in range(16)]
    offsets = [int(v) for v in np.cumsum([0] + sizes[:-1])]
    rows = sum(sizes)
    raw = iq4nl_encode_rows((np.random.default_rng(0).standard_normal((rows, 160)) * 0.5).astype(np.float32))
    (root / "table.iq4nl").write_bytes(raw.tobytes())
    manifest = json.loads(json.dumps(SWIFT15_MANIFEST))
    manifest["meta"]["qwen4exp.ple.head_offsets"] = offsets
    manifest["meta"]["qwen4exp.ple.head_vocab_sizes"] = sizes
    manifest["tensors"][0]["dims"] = [160, rows]
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _cli_templates():
    import yaml

    src = os.environ.get("NGRAM_SFT_TEMPLATES")
    if src:
        smoke = yaml.safe_load((Path(src) / "smoke.yaml").read_text())
        real = yaml.safe_load((Path(src) / "quick_real.yaml").read_text())
        shuf = yaml.safe_load((Path(src) / "quick_shuf.yaml").read_text())
    else:
        smoke, real = yaml.safe_load(_SMOKE_TEMPLATE), yaml.safe_load(_QUICK_TEMPLATE)
        shuf = json.loads(json.dumps(real))
        shuf["ngram_memory"]["row_shuffle"] = dict(SHUFFLE)
    shuffle = shuf["ngram_memory"]["row_shuffle"]
    smoke_shuf = json.loads(json.dumps(smoke))
    smoke_shuf["ngram_memory"]["row_shuffle"] = dict(shuffle)
    assert "row_shuffle" not in real["ngram_memory"] and "row_shuffle" not in smoke["ngram_memory"]
    return {"smoke_real": smoke, "smoke_shuf": smoke_shuf, "quick_real": real, "quick_shuf": shuf}


def _tiny_cli_config(template, root, name, max_steps):
    """The template with paths and sizes swapped for the tiny model; its data and panel files written (same file
    count and kinds as the template's). Steps: max_steps; panels every step and checkpoints every step (the H200
    cadence of 4 would never fire in 2 steps)."""
    cfg = json.loads(json.dumps(template))
    work = root / name
    work.mkdir(exist_ok=True)
    cfg["model"] = str(root / "model")
    cfg["output_dir"] = str(work / "out")
    cfg.update(_TINY_SIZES)
    cfg["max_steps"] = max_steps
    assert cfg["sequence_len"] % cfg["sequence_chunks"] == 0
    if cfg.get("resume_from_checkpoint"):
        cfg["save_steps"] = 1
    if cfg.get("eval_panels"):
        cfg["eval_panels_steps"] = 1
    cfg["ngram_memory"]["table"] = str(root / "table.iq4nl")
    cfg["ngram_memory"]["manifest"] = str(root / "manifest.json")
    rng = np.random.default_rng(zlib.crc32(name.split("_")[0].encode()))  # the same data for both arms
    data = work / "data"
    data.mkdir(exist_ok=True)
    lengths = (30, 120, 300, 500, 800, 1200)  # assistant words: 1 to 32 chunks (1200 words: truncated to 1024 tokens)
    k = 0
    per_file = max(3, -(-cfg["gradient_accumulation_steps"] * (max_steps + 1) // len(cfg["datasets"])))
    for i, ds in enumerate(cfg["datasets"]):
        rows = [_cli_conversation(rng, k + j, lengths[(k + j) % len(lengths)] if len(cfg["datasets"]) > 1
                                  else (1200, 500, 30)[j % 3]) for j in range(per_file)]
        k += per_file
        ds["path"] = str(data / f"train_{i}.jsonl")
        _write_jsonl(ds["path"], rows)
    for i, panel in enumerate(cfg.get("eval_panels") or []):
        tok = str(panel["path"]).endswith(".tok.jsonl")
        path = data / (f"panel_{i}.tok.jsonl" if tok else f"panel_{i}.jsonl")
        if tok:
            rows = []
            for j, n in enumerate((90, 700, 1500)):
                ids = rng.integers(1000, 200000, size=n).tolist()
                ids[n // 2] = EOS
                rows.append({"id": f"tok{j}", "ids": ids, "spans": [[10, n // 3], [n // 2 + 1, n]]})
        else:
            plain = "ifeval" in panel["name"] or "bfcl" in panel["name"]
            rows = [_cli_conversation(rng, 1000 * (i + 1) + j, n, tools=not plain) for j, n in enumerate((40, 400, 900))]
        _write_jsonl(path, rows)
        panel["path"] = str(path)
    return cfg


def _run_sft(cfg, name, root):
    """The SFT CLI as h200_run.sh runs it (python -m surogate.cli.main sft <yaml>), with the per-step loss curve
    (SUROGATE_REGRESSION_ARTIFACT) and the chunk trace of train_step_graphed's loop."""
    import re
    import shutil
    import subprocess
    import sys

    import yaml

    work = root / name.split("@")[0]
    path = work / f"{name.replace('@', '_')}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    curve_path = work / f"{name.replace('@', '_')}.curve.json"
    log_path = work / f"{name.replace('@', '_')}.log"
    short_tmp = tempfile.mkdtemp(prefix="ngcli", dir="/tmp")  # NCCL's AF_UNIX socket paths must stay short
    env = {**os.environ, "TMPDIR": short_tmp, "SUROGATE_CHUNK_TRACE": "1", "PYTHONUNBUFFERED": "1",
           "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "SUROGATE_REGRESSION_ARTIFACT": str(curve_path)}
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run([sys.executable, "-m", "surogate.cli.main", "sft", str(path)], stdout=log,
                                  stderr=subprocess.STDOUT, env=env, timeout=1800, cwd=str(work))
    finally:
        shutil.rmtree(short_tmp, ignore_errors=True)
    text = log_path.read_text(errors="replace")
    assert proc.returncode == 0, f"{name}: rc {proc.returncode}\n" + text[-6000:]
    curve = json.loads(curve_path.read_text())["convergence_curve"]
    chunks = [int(c) for c in re.findall(r"\[chunk\] phaseB j=\d+ c=(\d+)", text)]
    out = Path(cfg["output_dir"])
    stats = [json.loads(line) for line in open(out / "ngram_memory_stats.jsonl")]
    return {"curve": curve, "max_chunk": max(chunks, default=-1), "stats": stats, "out": out, "log": text}


def _check_cli_arm(run, cfg, steps, shuffled):
    from safetensors.torch import load_file

    assert [int(p["step"]) for p in run["curve"]] == list(range(steps[0], steps[1] + 1)), run["curve"]
    assert all(math.isfinite(p["loss"]) and math.isfinite(p["grad_norm"]) for p in run["curve"])
    assert run["max_chunk"] == cfg["sequence_chunks"] - 1, run["max_chunk"]  # the 1024-token rows: all 32 chunks
    assert "Traceback" not in run["log"]
    adapter = load_file(str(run["out"] / "adapter_model.safetensors"))
    assert set(adapter) == set(MEMORY), sorted(adapter)  # memory-only PEFT: no LoRA tensors
    assert adapter[KEY].shape == (256, 2560) and adapter[VALUE].float().abs().max() > 0
    acfg = json.loads((run["out"] / "adapter_config.json").read_text())["surogate_ngram_memory"]
    assert ("row_shuffle_a" in acfg) == shuffled


def _first_live(curve):
    """The first step (1-based) whose forward follows an update with lr > 0. Warmup starts at lr 0 (step 1 of every
    run updates nothing), so until then W_v = 0 and the arms -- which read different rows -- are one model."""
    for i, p in enumerate(curve):
        if p["lr"] > 0:
            return i + 2
    return len(curve) + 1


@pytest.mark.gpu
@pytest.mark.slow
def test_sft_cli_smoke_config_real_and_shuffled(cli_model):
    """smoke.yaml's option set (1 conversation per step, 32 chunks, fp8-hybrid, memory-only, host-resident table), 3
    steps through the SFT CLI, real table and row-shuffled: every chunk of the 1024-token rows runs (the carry); while
    W_v = 0 (steps 1-2: step 1's lr is 0) the arms' losses and memory output are identical bit for bit although they
    read different rows (W_v's gradient differs); after the first real update they part."""
    tmpl = _cli_templates()
    runs = {}
    for arm in ("smoke_real", "smoke_shuf"):
        cfg = _tiny_cli_config(tmpl[arm], cli_model, arm, max_steps=3)
        runs[arm] = _run_sft(cfg, arm, cli_model)
        _check_cli_arm(runs[arm], cfg, (1, 3), shuffled=arm.endswith("shuf"))
    real, shuf = runs["smoke_real"], runs["smoke_shuf"]
    print(json.dumps({arm: [(p["loss"], p["grad_norm"], p["lr"]) for p in r["curve"]] for arm, r in runs.items()}))
    live = _first_live(real["curve"])
    assert live == 3 and [p["lr"] for p in real["curve"]] == [p["lr"] for p in shuf["curve"]]
    for k in range(live - 1):
        assert real["curve"][k]["loss"] == shuf["curve"][k]["loss"], k
    assert real["curve"][0]["grad_norm"] != shuf["curve"][0]["grad_norm"]  # W_v's gradient reads the rows
    assert real["curve"][live - 1]["loss"] != shuf["curve"][live - 1]["loss"]
    for r in (real, shuf):  # one line per step (logging_steps 1)
        assert len(r["stats"]) == 3 and all(st["tokens"] > 0 for st in r["stats"]), r["stats"]
        assert all(st["out_rms_over_residual_rms"] == 0.0 for st in r["stats"][: live - 1])
        assert r["stats"][live - 1]["out_rms_over_residual_rms"] > 0.0


@pytest.mark.gpu
@pytest.mark.slow
def test_sft_cli_quick_configs_with_panels_and_resume(cli_model):
    """quick_real.yaml / quick_shuf.yaml's option sets (16 conversations per step over 7 files, 32 chunks, 8 panels,
    per-token dumps), 2 steps through the SFT CLI, then a resume of each arm to step 3 from its step-1 checkpoint.
    The per-token panels dump step 0 and the final step with training n-gram counts; the resumed step repeats the
    original step 2 (the loss bit for bit; norm and the step-2 panels up to fp8-hybrid's delayed-scaling history,
    which a checkpoint does not hold; warmup makes step 2's lr independent of max_steps); while W_v = 0 the arms'
    losses and panel rows are identical bit for bit, after it they part."""
    tmpl = _cli_templates()
    runs = {}
    for arm in ("quick_real", "quick_shuf"):
        cfg = _tiny_cli_config(tmpl[arm], cli_model, arm, max_steps=2)
        run = _run_sft(cfg, arm, cli_model)
        _check_cli_arm(run, cfg, (1, 2), shuffled=arm.endswith("shuf"))
        assert (run["out"] / "step_00000001").is_dir()
        rows = [json.loads(line) for line in open(run["out"] / "eval_panels.jsonl")]
        panels = [p["name"] for p in cfg["eval_panels"]]
        assert len(panels) == 8
        assert sorted({(r["step"], r["panel"]) for r in rows}) == sorted((s, p) for s in (0, 1, 2) for p in panels)
        assert all(math.isfinite(r["nll"]) and r["tokens"] > 0 for r in rows)
        per_token = {p["name"] for p in cfg["eval_panels"] if p.get("per_token")}
        for step in (0, 2):
            with gzip.open(run["out"] / f"eval_tokens_step{step}.jsonl.gz", "rt") as f:
                dumped = [json.loads(line) for line in f]
            assert {d["panel"] for d in dumped} == per_token
            assert all(len(d["pos"]) == len(d["lp"]) == len(d["bi"]) == len(d["tri"]) > 0 for d in dumped)
            assert any(b > 0 for d in dumped for b in d["bi"])  # training bigrams found in the panels
        assert not (run["out"] / "eval_tokens_step1.jsonl.gz").exists()
        # resume: the same config to step 3 (the checkpoint after step 1; the loader's position, Adam's moments)
        cfg3 = json.loads(json.dumps(cfg))
        cfg3["max_steps"] = 3
        resumed = _run_sft(cfg3, arm + "@resume", cli_model)
        assert [int(p["step"]) for p in resumed["curve"]] == [2, 3], resumed["curve"]
        assert resumed["max_chunk"] == cfg["sequence_chunks"] - 1
        # fp8-hybrid's delayed scaling keeps an amax history the checkpoint does not hold: after a resume the loss
        # is exact, the scales after it differ (measured: 3e-4 of the norm, panel rows up to 2.5e-3 nats, mean
        # 8e-4); the same resume in bf16 is bit for bit (loss, norm, all 24 panel rows). Not the memory's: it has no
        # FP8 state.
        exact = cfg["recipe"] == "bf16"
        r2, o2 = resumed["curve"][0], run["curve"][1]
        assert (r2["step"], r2["lr"], r2["loss"]) == (o2["step"], o2["lr"], o2["loss"]), (r2, o2)
        assert r2["grad_norm"] == o2["grad_norm"] if exact else abs(r2["grad_norm"] - o2["grad_norm"]) <= 2e-3 * o2[
            "grad_norm"], (r2, o2)
        rows3 = [json.loads(line) for line in open(run["out"] / "eval_panels.jsonl")][len(rows):]
        assert sorted({r["step"] for r in rows3}) == [2, 3]
        before = {(r["panel"], r["id"]): r["nll"] for r in rows if r["step"] == 2}
        after = {(r["panel"], r["id"]): r["nll"] for r in rows3 if r["step"] == 2}
        assert before.keys() == after.keys()
        assert all(abs(before[k] - after[k]) <= (0.0 if exact else 5e-3) for k in before), (before, after)
        runs[arm] = {"curve": run["curve"] + resumed["curve"][1:], "rows": rows + rows3}
    real, shuf = runs["quick_real"], runs["quick_shuf"]
    print(json.dumps({arm: [(p["loss"], p["grad_norm"], p["lr"]) for p in r["curve"]] for arm, r in runs.items()}))
    live = _first_live(real["curve"])
    assert live == 3 and [p["lr"] for p in real["curve"]] == [p["lr"] for p in shuf["curve"]]
    for k in range(live - 1):
        assert real["curve"][k]["loss"] == shuf["curve"][k]["loss"], k
    assert real["curve"][live - 1]["loss"] != shuf["curve"][live - 1]["loss"]

    def panel(r, step):
        return {(x["panel"], x["id"]): x["nll"] for x in r["rows"] if x["step"] == step}

    for step in range(live - 1):  # the panels at step s follow s updates
        assert panel(real, step) == panel(shuf, step) and len(panel(real, step)) == 3 * 8, step
    for step in range(live - 1, 4):
        assert panel(real, step) != panel(shuf, step), step
