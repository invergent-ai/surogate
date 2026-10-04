"""`surogate debug diff`: packed documents and the placement of the HF reference (#236).

CPU only: the DSL side needs a card, so these cover the row layout, the per-document HF
reference (a tiny random Llama), the region compare, and the reference placement.
"""

import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from surogate.debug import diff  # noqa: E402
from surogate.debug.schema import DiffStatus, Severity  # noqa: E402


def test_unpacked_layout_is_one_sequence():
    docs = diff._documents(64, packed=False, pack_split=None)
    assert docs == [diff._Document(None, 0, 64)]
    np.testing.assert_array_equal(diff._position_ids(2, docs), np.tile(np.arange(64), (2, 1)))


def test_packed_layout_splits_in_half_and_restarts_positions():
    docs = diff._documents(64, packed=True, pack_split=None)
    assert docs == [diff._Document("doc1", 0, 32), diff._Document("doc2", 32, 64)]
    pos = diff._position_ids(3, diff._documents(10, packed=True, pack_split=3))
    assert pos.shape == (3, 10)
    np.testing.assert_array_equal(pos[1], [0, 1, 2, 0, 1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize("split", [0, 64, -1, 100])
def test_packed_split_must_leave_both_documents_non_empty(split):
    with pytest.raises(ValueError, match="leaves a document empty"):
        diff._documents(64, packed=True, pack_split=split)


def test_doc_region_reads_bsh_and_flat_dumps():
    rows, seq, hidden = 2, 6, 4
    full = np.arange(rows * seq * hidden, dtype=np.float32).reshape(rows, seq, hidden)
    doc2 = diff._Document("doc2", 4, 6)
    np.testing.assert_array_equal(diff._doc_region(full, rows, seq, doc2), full[:, 4:6])
    flat = full.reshape(rows * seq, hidden)
    np.testing.assert_array_equal(diff._doc_region(flat, rows, seq, doc2), full[:, 4:6])
    assert diff._doc_region(np.zeros(7, dtype=np.float32), rows, seq, doc2) is None


def test_zero_reference_is_flagged_not_blamed_on_the_dsl():
    hf = np.zeros((1, 4, 8), dtype=np.float32)
    dsl = np.ones((1, 4, 8), dtype=np.float32)
    stats, status = diff._diff_tensors(hf, dsl, 4)
    assert status == DiffStatus.HF_ZERO
    assert diff._severity_from_diff(stats, status, 1e-2, 1e-3) == Severity.WARN
    assert diff._hf_side_broken(stats, status)


def test_nonfinite_reference_is_broken_but_nonfinite_dsl_is_not():
    hf = np.ones((1, 4, 8), dtype=np.float32)
    dsl = np.ones((1, 4, 8), dtype=np.float32)
    hf_bad = hf.copy()
    hf_bad[0, 1, 2] = np.nan
    stats, status = diff._diff_tensors(hf_bad, dsl, 4)
    assert status == DiffStatus.NONFINITE and diff._hf_side_broken(stats, status)
    dsl_bad = dsl.copy()
    dsl_bad[0, 0, 0] = np.inf
    stats, status = diff._diff_tensors(hf, dsl_bad, 4)
    assert status == DiffStatus.NONFINITE and not diff._hf_side_broken(stats, status)


def test_leak_verdict_ignores_bf16_drift_shared_by_both_documents():
    # Qwen3.5-0.8B on an L4: both documents at cos_sim ~0.99988 when isolated; with document
    # masking off doc2 drops to 0.68 while doc1 stays put.
    assert not diff._crosses_boundary(0.99987, 0.99988)
    assert not diff._crosses_boundary(0.9990, 0.9985)  # both drift alike
    assert diff._crosses_boundary(0.99987, 0.68278)
    assert diff._crosses_boundary(0.99999, 0.995)
    assert not diff._crosses_boundary(None, 0.5)


def test_broken_reference_note_points_at_sharding():
    sharded = diff._hf_broken_note("cuda:1", ["cuda:0", "cuda:1"])
    assert "--ref-device-map cuda:0" in sharded and "cuda:0, cuda:1" in sharded
    single = diff._hf_broken_note("cuda:0", ["cuda:0"])
    assert "says nothing about the DSL" in single and "--ref-device-map" not in single


def test_reference_goes_on_one_card_when_it_fits():
    gib = 1 << 30
    device_map, why = diff._choose_ref_device_map(2 * gib, 20 * gib, rows=1, doc_len=64, vocab_size=151_936)
    assert device_map == "cuda:0" and "one card" in why
    device_map, why = diff._choose_ref_device_map(54 * gib, 40 * gib, rows=1, doc_len=64, vocab_size=151_936)
    assert device_map == "auto" and "does not fit" in why
    # The logits count: a model that barely fits by weights alone does not fit with them.
    device_map, _ = diff._choose_ref_device_map(18 * gib, 20 * gib, rows=8, doc_len=4096, vocab_size=151_936)
    assert device_map == "auto"
    device_map, why = diff._choose_ref_device_map(None, 20 * gib, rows=1, doc_len=64, vocab_size=1000)
    assert device_map == "cuda:0" and "unknown" in why


def _write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


def test_weight_bytes_count_floats_as_bf16(tmp_path):
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "a.weight": ("F32", (4, 8), 4 * 8 * 4),  # fp32 on disk, bf16 once loaded
            "b.weight": ("BF16", (16,), 16 * 2),
            "c.weight": ("F8_E4M3", (10,), 10),  # counted as if dequantised to bf16
            "d.blocks": ("U8", (6,), 6),  # packed 4-bit, dequantised worst case
        },
    )
    assert diff._reference_weight_bytes(str(tmp_path)) == 32 * 2 + 16 * 2 + 10 * 2 + 6 * 4
    assert diff._reference_weight_bytes(str(tmp_path / "missing")) is None
    assert diff._reference_weight_bytes("org/not-a-local-dir") is None


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    path = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(path)
    return str(path)


def test_packed_reference_runs_each_document_on_its_own(tiny_llama):
    """Region compare against per-document references: a runtime that isolates the documents
    matches both; one that lets doc2 see doc1 (the whole row as one sequence) matches doc1 and
    diverges on doc2, which is the leak `--packed` exists to catch."""
    arch_map = diff._ARCH_MAPS["LlamaForCausalLM"]
    rows, seq_len = 2, 24
    docs = diff._documents(seq_len, packed=True, pack_split=10)
    tokens = np.random.default_rng(0).integers(0, 128, size=(rows, seq_len), dtype=np.int32)

    ref = diff._run_hf_reference(tiny_llama, [tokens[:, d.start : d.end] for d in docs], arch_map, device_map="cpu")
    assert ref.device_map == "cpu" and ref.placement == "requested"
    assert [sorted(o) for o in ref.outputs] == [[0, 1, 2], [0, 1, 2]]
    assert ref.outputs[0][0].shape == (rows, 10, 32) and ref.outputs[1][0].shape == (rows, 14, 32)
    assert set(ref.layer_devices.values()) == {"cpu"}

    isolated = {layer: np.concatenate([ref.outputs[0][layer], ref.outputs[1][layer]], axis=1) for layer in range(3)}
    leaky = diff._run_hf_reference(tiny_llama, [tokens], arch_map, device_map="cpu").outputs[0]

    for layer in range(3):
        for doc, hf_doc in zip(docs, (ref.outputs[0][layer], ref.outputs[1][layer]), strict=True):
            region = diff._doc_region(isolated[layer], rows, seq_len, doc)
            stats, status = diff._diff_tensors(hf_doc, region, doc.end - doc.start)
            assert diff._severity_from_diff(stats, status, 1e-2, 1e-3) == Severity.INFO

        stats, status = diff._diff_tensors(
            ref.outputs[0][layer], diff._doc_region(leaky[layer], rows, seq_len, docs[0]), 10
        )
        assert diff._severity_from_diff(stats, status, 1e-2, 1e-3) == Severity.INFO
        stats, status = diff._diff_tensors(
            ref.outputs[1][layer], diff._doc_region(leaky[layer], rows, seq_len, docs[1]), 14
        )
        assert diff._severity_from_diff(stats, status, 1e-2, 1e-3) == Severity.ERROR


def test_default_placement_without_a_card_loads_where_accelerate_puts_it(tiny_llama):
    if torch.cuda.is_available():
        pytest.skip("covers the CPU-only fallback")
    tokens = np.zeros((1, 8), dtype=np.int32)
    ref = diff._run_hf_reference(tiny_llama, [tokens], diff._ARCH_MAPS["LlamaForCausalLM"], vocab_size=128)
    assert ref.device_map == "auto" and ref.placement == "no CUDA device"
    assert set(ref.layer_devices.values()) == {"cpu"}
