"""Paged resident-policy decode with changing membership on every GRPO family."""

import gc
import json
import os

import numpy as np
import pytest
import torch

from tests.grpo.shared_model_configs import configurations

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]
CASES = configurations()
SELECTED = os.environ.get("SUROGATE_SHARED_CASES", "all").split(",")


def make_trainer(root, case, *, graphs=False, sequence=256, config=None, rollout_parity=None):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    if case == "glm":
        from examples.sft.glm.create_dummy import create_dummy

        create_dummy(root, index_topk=32, max_sequence_length=sequence)
    else:
        (root / "config.json").write_text(json.dumps(CASES[case] if config is None else config))
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    moe = text.get("num_experts", text.get("num_local_experts", text.get("n_routed_experts", 0)))
    if rollout_parity is None:
        rollout_parity = bool(moe)
    options = ext.RuntimeOptions(
        recompute="true",
        use_cuda_graphs=graphs,
        master_dtype="bf16",
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
        doc_masking=True,
    )
    options.glm_rollout_parity = case == "glm"
    options.moe_rollout_parity = rollout_parity
    options.dsl_ir_json = build_dsl_ir_for_model(str(root))
    options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json, rollout_parity=rollout_parity)
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
        options=options,
        batch_size=2,
        seq_len=sequence,
        grad_accum=1,
        lora_config=ext.LoRAAdapterConfig(
            rank=8, alpha=13, dropout=0, dtype="bf16" if moe else "fp32", target_modules=["all"]
        ),
    )
    if case == "glm":
        trainer.import_weights(str(root / "model.safetensors"))
    else:
        trainer.init_weights()
    return trainer


def decode(trainer, sessions, chunks, resets):
    return trainer.decode_batch_logits(
        np.asarray(sessions, dtype=np.int64),
        np.concatenate(chunks).astype(np.int32),
        np.cumsum([0] + [len(c) for c in chunks], dtype=np.int32),
        np.asarray(resets, dtype=np.int32),
    )


@pytest.mark.parametrize("case", list(CASES) + ["glm"] if SELECTED == ["all"] else SELECTED)
def test_paged_decode_membership_growth_reuse_and_invalidation(tmp_path, case):
    from surogate import _surogate as ext

    trainer = make_trainer(tmp_path, case)
    rng = np.random.default_rng(84)
    # These remain within the vocabulary of all miniature configs.
    history = {i: rng.integers(3, 97, size=n, dtype=np.int32) for i, n in [(11, 5), (22, 9), (33, 7)]}
    decode(trainer, [11, 22], [history[11], history[22]], [1, 1])
    base_ptrs = {n: torch.from_dlpack(t).data_ptr() for n, t in trainer.get_shared_base_weights().items()}
    # Different prefix lengths, row reordering, a new request, and a page boundary.
    additions = [
        rng.integers(3, 97, size=128, dtype=np.int32),
        history[33],
        rng.integers(3, 97, size=128, dtype=np.int32),
    ]
    history[22] = np.concatenate([history[22], additions[0]])
    history[11] = np.concatenate([history[11], additions[2]])
    decode(trainer, [22, 33, 11], additions, [0, 1, 0])
    for step in range(2):
        sessions = [33, 11, 22] if step == 0 else [22, 33, 11]
        chunks = [rng.integers(3, 97, size=1, dtype=np.int32) for _ in sessions]
        actual = decode(trainer, sessions, chunks, [0] * 3)
        for row, session in enumerate(sessions):
            history[session] = np.concatenate([history[session], chunks[row]])
            expected = trainer.decode_logits(history[session], reset=True)
            assert np.isfinite(actual[row]).all(), f"{case}: nonfinite batched logits"
            assert np.isfinite(expected).all(), f"{case}: nonfinite full-prefix logits"
            # BF16 GEMM shapes and attention reduction order can differ; GLM
            # deliberately fixes both for sparse selection/GRPO parity.
            # Compare policy log-probabilities, as in the existing shared-model
            # GRPO regression, rather than arbitrary additive logit offsets.
            actual_logprobs = actual[row].astype(np.float64)
            actual_logprobs -= np.logaddexp.reduce(actual_logprobs)
            expected_logprobs = expected.astype(np.float64)
            expected_logprobs -= np.logaddexp.reduce(expected_logprobs)
            # The full-vocabulary maximum is stricter than checking only the
            # sampled token. Bound rare BF16/MoE outliers as well as RMS error.
            if case != "glm":
                assert np.sqrt(np.mean((actual_logprobs - expected_logprobs) ** 2)) < 0.03
            np.testing.assert_allclose(
                actual_logprobs,
                expected_logprobs,
                atol=1e-5 if case == "glm" else 0.1,
                rtol=0,
                err_msg=f"{case} session {session}, step {step}",
            )
    stats = trainer.get_decode_batch_stats()
    assert stats["sessions"] == 3 and stats["pages"] > 0 and stats["pool_used_bytes"] > 0
    assert stats["length"] == sum(map(len, history.values()))
    trainer.release_decode_sessions([11, 22, 33])
    released = trainer.get_decode_batch_stats()
    assert released["pages"] == 0
    # The legacy reference above retains its own history in the same budget.
    assert released["pool_used_bytes"] > 0
    assert released["cache_allocated_bytes"] >= trainer.get_decode_cache_stats()["bytes"]
    with pytest.raises(RuntimeError, match="prefill"):
        decode(trainer, [11], [np.array([7])], [0])
    decode(trainer, [44], [history[22]], [1])
    assert trainer.get_decode_batch_stats()["reused_pages"] > stats["reused_pages"]
    # Invalid submissions leave other active requests intact.
    with pytest.raises(ValueError):
        decode(trainer, [44, 44], [np.array([7]), np.array([8])], [0, 0])
    assert trainer.get_decode_batch_stats()["sessions"] == 1
    ids = np.zeros((2, 256), dtype=np.int32)
    targets = np.full_like(ids, -100)
    ids[0, :16] = rng.integers(3, 97, size=16)
    targets[0, :15] = ids[0, 1:16]
    scales = (targets != -100).astype(np.float32) / 15
    trainer.step_with_custom_loss(ids, targets, scales)
    trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-4), 1)
    assert trainer.get_decode_batch_stats()["sessions"] == 0
    assert trainer.get_decode_batch_stats()["pool_used_bytes"] == 0
    assert base_ptrs == {n: torch.from_dlpack(t).data_ptr() for n, t in trainer.get_shared_base_weights().items()}
    with pytest.raises(RuntimeError, match="prefill"):
        decode(trainer, [44], [np.array([7])], [0])
    del trainer
    gc.collect()


def test_gemma_moe_adapter_import_refreshes_dense_and_grouped_projections(tmp_path):
    trainer = make_trainer(tmp_path, "gemma4_moe")
    prompt = np.array([7, 13, 19, 23], dtype=np.int32)
    expected = decode(trainer, [1], [prompt], [1])
    adapter = tmp_path / "adapter"
    trainer.export_adapter(str(adapter))
    # Poison the work copies, leaving the exported/master adapter untouched.
    # This detects skipped synchronization even on freshly zeroed GPU memory.
    work = {name: torch.from_dlpack(t) for name, t in trainer.get_lora_weights(0).items()}
    dense = [tensor for name, tensor in work.items() if ".mlp." in name and ".experts." not in name]
    grouped = [tensor for name, tensor in work.items() if ".experts." in name]
    assert dense and grouped
    for tensor in dense + grouped:
        tensor.fill_(float("nan"))
    torch.cuda.synchronize()
    trainer.import_adapter(str(adapter / "adapter_model.safetensors"))
    assert trainer.get_decode_batch_stats()["sessions"] == 0
    actual = decode(trainer, [2], [prompt], [1])
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    del dense, grouped, work, tensor, trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "gemma4_moe", "glm"])
def test_decode_between_captured_training_replays(tmp_path, case):
    from surogate import _surogate as ext

    trainer = make_trainer(tmp_path, case, graphs=True)
    rng = np.random.default_rng(35)
    ids = rng.integers(3, 97, size=(2, 256), dtype=np.int32)
    targets = np.full_like(ids, -100)
    targets[:, :15] = ids[:, 1:16]
    scales = (targets != -100).astype(np.float32) / 30
    norms = []
    for step in range(4):
        trainer.step_with_custom_loss(ids, targets, scales)
        # Keep the policy fixed so replay gradients can be compared across
        # decoding with different shapes and independent activation arenas.
        result = trainer.update_with_config(ext.OptimizerConfig(learning_rate=0), step + 1)
        norms.append(result["norm"])
        assert np.isfinite(norms[-1]) and norms[-1] > 0
        assert trainer.get_decode_batch_stats()["sessions"] == 0
        if step in (1, 2):
            decode(trainer, [1, 2], [ids[0, :5], ids[1, :9]], [1, 1])
            decode(trainer, [2, 3, 1], [ids[1, 9:10], ids[0, :7], ids[0, 5:6]], [0, 1, 0])
            logits = decode(trainer, [3, 1, 2], [ids[0, 7:8], ids[0, 6:7], ids[1, 10:11]], [0, 0, 0])
            assert np.isfinite(logits).all()
            for _ in range(4):
                logits = decode(trainer, [2, 3, 1], [np.array([7], dtype=np.int32)] * 3, [0, 0, 0])
                assert np.isfinite(logits).all()
            stats = trainer.get_decode_batch_stats()
            assert stats["decode_graph_captures"] > 0
            assert stats["decode_graph_replays"] > stats["decode_graph_captures"]
    np.testing.assert_allclose(norms[2:], norms[1], rtol=0.01, atol=1e-6)
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", list(CASES) + ["glm"] if SELECTED == ["all"] else SELECTED)
def test_cache_byte_budget_accounts_for_pages_tables_and_recurrent_state(tmp_path, case):
    trainer = make_trainer(tmp_path, case)

    def admit(ids, counts, resets):
        return trainer.admit_decode_sessions(
            np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.int32), np.asarray(resets, dtype=np.int32)
        )

    assert admit([1], [5], [1]) == [True]
    budget = trainer.get_decode_batch_stats()["cache_allocated_bytes"]
    trainer.set_decode_cache_budget(budget)
    assert admit([2], [5], [1]) == [False]
    prompt = np.array([7, 13, 19, 23, 29], dtype=np.int32)
    before = decode(trainer, [1], [prompt], [1])
    assert np.isfinite(before).all()
    assert admit([2, 1], [1, 1], [1, 0]) == [False, True]
    with pytest.raises(ValueError, match="below live"):
        trainer.set_decode_cache_budget(budget - 1)
    assert admit([1], [124], [0]) == [False]
    assert trainer.get_decode_batch_stats()["length"] == 5
    assert admit([1], [1], [0]) == [True]
    actual = decode(trainer, [1], [np.array([31], dtype=np.int32)], [0])
    assert np.isfinite(actual).all()
    assert trainer.get_decode_batch_stats()["cache_allocated_bytes"] <= budget
    trainer.release_decode_sessions([1])
    assert admit([3], [5], [1]) == [True]
    assert trainer.get_decode_batch_stats()["cache_allocated_bytes"] <= budget
    trainer = None
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "gemma4_moe", "glm"])
def test_gpu_sampling_tracks_chunked_history_reordering_and_reset(tmp_path, case):
    from surogate.grpo.shared_model import sample_logits
    from tests.grpo.test_decode_sampling_gpu import DEFAULTS

    trainer = make_trainer(tmp_path, case, graphs=True)
    history = {}
    rounds = [([1, 2], [[7, 7, 9] * 42, [9, 13] * 60], [1, 1])]
    rounds += [([2, 1], [[7], [9]], [0, 0])] * 4
    rounds += [([1, 2], [[13, 13, 13], [19]], [1, 0])]
    for step, (sessions, chunks, resets) in enumerate(rounds):
        logits = decode(trainer, [s + 100 for s in sessions], chunks, resets)
        requests, expected = [], []
        for row, (session, chunk, reset) in enumerate(zip(sessions, chunks, resets, strict=True)):
            history[session] = ([] if reset else history[session]) + chunk
            params = DEFAULTS | dict(
                temperature=0.7,
                top_p=0.8,
                repetition_penalty=1.3,
                presence_penalty=0.4,
                frequency_penalty=0.15,
                top_logprobs=5,
                logit_bias={7: 2.0},
            )
            seed = step * 10 + session
            requests.append(params | dict(uniform=float(np.random.default_rng(seed).random())))
            expected.append(sample_logits(logits[row], history[session], params, [], np.random.default_rng(seed)))
        sampled = trainer.decode_batch_sample(
            np.asarray(sessions, dtype=np.int64),
            np.concatenate(chunks).astype(np.int32),
            np.cumsum([0] + [len(c) for c in chunks], dtype=np.int32),
            np.asarray(resets, dtype=np.int32),
            requests,
        )
        for actual, reference in zip(sampled, expected, strict=True):
            assert actual["status"] == 0
            assert actual["token"] == reference["token"]
            assert actual["top_ids"] == reference["top_ids"]
            np.testing.assert_allclose(actual["logprob"], reference["logprob"], atol=1e-8, rtol=0)
        assert trainer.get_decode_batch_stats()["length"] == 2 * sum(map(len, history.values()))
    before = trainer.get_decode_batch_stats()
    with pytest.raises(ValueError, match="sampling parameters"):
        trainer.decode_batch_sample(
            np.array([1], dtype=np.int64),
            np.array([7], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            [dict(top_p=0)],
        )
    assert trainer.get_decode_batch_stats()["length"] == before["length"]
    assert before["sampling_counts_bytes"] > 0 and before["decode_graph_replays"] > 0
    del trainer
    gc.collect()


def test_decode_shape_cache_is_bounded_and_reuses_recent_shapes(tmp_path):
    trainer = make_trainer(tmp_path, "llama", graphs=True)
    for length in range(2, 12):
        decode(trainer, [1], [[7] * length], [1])
    before = trainer.get_decode_batch_stats()
    assert before["resident_decode_shapes"] == 8 and before["compiled_decode_shapes"] == 10
    decode(trainer, [2], [[13] * 10], [1])
    assert trainer.get_decode_batch_stats()["compiled_decode_shapes"] == 10
    decode(trainer, [1], [[13] * 2], [1])
    assert trainer.get_decode_batch_stats()["compiled_decode_shapes"] == 11
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "glm"])
def test_workspace_admission_rejects_without_advancing_healthy_history(tmp_path, case):
    trainer = make_trainer(tmp_path, case, graphs=True)

    def admit(ids, counts, resets, sampling=None):
        return trainer.admit_decode_sessions(np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.int32),
                                            np.asarray(resets, dtype=np.int32), sampling)

    trainer.set_decode_memory_budget(1)
    assert admit([1], [1], [1]) == [False]
    assert trainer.get_decode_batch_stats()["sessions"] == 0
    trainer.set_decode_memory_budget(0)
    decode(trainer, [1], [[7]], [1])
    before = trainer.get_decode_batch_stats()
    assert before["decode_workspace_bytes"] > 0
    assert before["decode_execution_headroom_bytes"] == 64 << 20
    budget = before["decode_memory_reserved_bytes"]
    trainer.set_decode_memory_budget(budget)
    assert admit([2, 1], [129, 1], [1, 0]) == [False, True]
    assert trainer.get_decode_batch_stats()["length"] == 1
    assert trainer.get_decode_batch_stats()["decode_memory_reserved_bytes"] <= budget
    # Sampling scratch is also reserved during admission, before history writes.
    assert admit([1], [1], [0], [dict(top_p=0.8, logit_bias={i: 1.0 for i in range(512)})]) == [False]
    assert trainer.get_decode_batch_stats()["length"] == 1
    assert admit([1], [1], [0]) == [True]
    trainer.set_decode_memory_budget(0)
    actual = decode(trainer, [1], [[13]], [0])
    reference = decode(trainer, [3], [[7, 13]], [1])
    actual -= np.logaddexp.reduce(actual, axis=-1, keepdims=True)
    reference -= np.logaddexp.reduce(reference, axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, reference, atol=1e-5 if case == "glm" else 0.1, rtol=0)
    trainer = None
    gc.collect()


def test_workspace_pressure_splits_batches_without_losing_rows(tmp_path):
    trainer = make_trainer(tmp_path, "llama", graphs=True)
    sessions = np.arange(1, 5, dtype=np.int64)
    ones = np.ones(4, dtype=np.int32)
    assert trainer.admit_decode_sessions(sessions, ones, ones) == [True] * 4
    before = trainer.get_decode_batch_stats()
    budget = before["decode_memory_reserved_bytes"] + 4096
    trainer.set_decode_memory_budget(budget)
    actual = decode(trainer, sessions, [[7], [13], [19], [23]], ones)
    stats = trainer.get_decode_batch_stats()
    assert stats["decode_batch_splits"] > 0
    assert stats["length"] == 4 and stats["sessions"] == 4
    assert stats["decode_memory_reserved_bytes"] <= budget
    assert stats["decode_shape_evictions"] > 0
    trainer.set_decode_memory_budget(0)
    reference = decode(trainer, sessions + 10, [[7], [13], [19], [23]], ones)
    np.testing.assert_array_equal(actual, reference)
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["qwen3_5", "lfm2", "glm"])
def test_recurrent_and_sparse_batches_cross_query_tile_boundaries(tmp_path, case):
    trainer = make_trainer(tmp_path, case, graphs=True)
    sessions = list(range(1, 11))
    prefixes = [[7 + row] * (row % 3 + 1) for row in range(10)]
    decode(trainer, sessions, prefixes, [1] * 10)
    # Ragged histories, a fresh row, and >8 queries exercise tiled MLA gathers,
    # batched convolution, and recurrent-state resets together.
    order = [10, 2, 9, 4, 8, 6, 7, 5, 3, 11]
    chunks = [[31 + row, 47 + row] for row in range(10)]
    resets = [int(session == 11) for session in order]
    actual = decode(trainer, order, chunks, resets)
    references = []
    for row, session in enumerate(order):
        prefix = prefixes[session - 1] if session != 11 else []
        references.append(decode(trainer, [100 + row], [prefix + chunks[row]], [1])[0])
    reference = np.asarray(references)
    actual -= np.logaddexp.reduce(actual, axis=-1, keepdims=True)
    reference -= np.logaddexp.reduce(reference, axis=-1, keepdims=True)
    np.testing.assert_allclose(actual, reference, atol=1e-5 if case == "glm" else 0.1, rtol=0)
    del trainer
    gc.collect()


def test_workspace_admission_recovers_after_physical_vram_pressure(tmp_path):
    trainer = make_trainer(tmp_path, "llama", graphs=True)
    decode(trainer, [1], [[7]], [1])
    torch.cuda.empty_cache()
    available, _ = torch.cuda.mem_get_info()
    pressure = torch.empty(available - (32 << 20), dtype=torch.uint8, device="cuda")
    try:
        admitted = trainer.admit_decode_sessions(np.array([1], dtype=np.int64),
                                                np.array([3], dtype=np.int32), np.array([0], dtype=np.int32))
        assert admitted == [False]
        assert trainer.get_decode_batch_stats()["length"] == 1
    finally:
        del pressure
        torch.cuda.empty_cache()
    actual = decode(trainer, [1], [[13, 19, 23]], [0])
    reference = decode(trainer, [2], [[7, 13, 19, 23]], [1])
    np.testing.assert_allclose(actual, reference, atol=0.1, rtol=0)
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "glm"])
def test_batched_vocabulary_projection_selects_independent_positions(tmp_path, case):
    trainer = make_trainer(tmp_path, case)
    tokens = np.random.default_rng(87).integers(3, 97, size=(2, 256), dtype=np.int32)
    actual = trainer.next_token_logits(tokens, np.array([0, 3], dtype=np.int32))
    # The legacy API requires the trainer's full input shape. Compare selected
    # positions against fresh cached prefixes from the same policy.
    reference = np.concatenate([decode(trainer, [row + 1], [tokens[row, :pos + 1]], [1])
                                for row, pos in enumerate([0, 3])])
    np.testing.assert_allclose(actual, reference, atol=1e-5 if case == "glm" else 0.1, rtol=0)
    del trainer
    gc.collect()
