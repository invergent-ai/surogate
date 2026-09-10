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


def make_trainer(root, case, *, graphs=False):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    if case == "glm":
        from examples.sft.glm.create_dummy import create_dummy

        create_dummy(root, index_topk=32, max_sequence_length=256)
    else:
        (root / "config.json").write_text(json.dumps(CASES[case]))
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
    options.dsl_ir_json = build_dsl_ir_for_model(str(root))
    options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json)
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    moe = text.get("num_experts", text.get("num_local_experts", text.get("n_routed_experts", 0)))
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
        options=options,
        batch_size=2,
        seq_len=256,
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
    assert trainer.get_decode_batch_stats()["pool_used_bytes"] == 0
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
    np.testing.assert_allclose(norms[2:], norms[1], rtol=0.01, atol=1e-6)
    del trainer
    gc.collect()
