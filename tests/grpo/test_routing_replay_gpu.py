"""Rollout Routing Replay through the native GRPO step, unchunked and chunked.

Replay rows cover the whole step sequence; under chunked-sequence training each chunk's forward (and its backward
re-forward) must read its own slice (`step_grpo_native_chunked` sets the token offset per chunk). Without the offset
every chunk would read the rows from token 0 - chunk 1 would silently replay chunk 0's experts - and the chunked step
would no longer reproduce the replayed unchunked step. A tiny random-weight Qwen3-MoE (2 layers, 4 experts, top-2) with LoRA on
attention makes layer 1's gradients depend on layer 0's routing.
"""

import json
import os

import numpy as np
import pytest

CHUNK = 16
SEQ_CHUNKS = 2
LAYERS, EXPERTS, TOP_K = 2, 4, 2


def _mini_qwen3_moe(root):
    config = dict(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe", hidden_size=256,
                  intermediate_size=512, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=64, vocab_size=512, max_position_embeddings=256, rms_norm_eps=1e-6, hidden_act="silu",
                  tie_word_embeddings=True, rope_theta=10000.0, attention_bias=False, eos_token_id=1,
                  num_experts=EXPERTS, num_experts_per_tok=TOP_K, moe_intermediate_size=128, norm_topk_prob=True,
                  decoder_sparse_step=1, mlp_only_layers=[], torch_dtype="bfloat16")
    (root / "config.json").write_text(json.dumps(config))
    return config


@pytest.mark.gpu
@pytest.mark.slow
def test_chunked_grpo_step_replays_the_same_routing_as_the_unchunked_step(tmp_path):
    _surogate = pytest.importorskip("surogate._surogate", reason="needs the built extension")
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    _mini_qwen3_moe(model_dir)
    full_len = CHUNK * SEQ_CHUNKS
    os.environ["SUROGATE_ROPE_MAX_SEQ"] = str(full_len)

    rng = np.random.default_rng(0)
    inputs = rng.integers(2, 512, size=(1, full_len), dtype=np.int32)
    targets = np.roll(inputs, -1, axis=1).astype(np.int32)
    position_ids = np.arange(full_len, dtype=np.int32).reshape(1, full_len)
    loss_mask = np.zeros(full_len, dtype=np.uint8)
    loss_mask[4:] = 1  # one sample spanning both chunks
    advantages = (rng.normal(0.0, 1.0, size=full_len) * loss_mask).astype(np.float32)
    inference_logprobs = (rng.uniform(-8.0, -6.0, size=full_len) * loss_mask).astype(np.float32)

    # Forced experts: a fixed, distinct pair per token and layer (seeded), unlike a random router's own picks.
    rows = np.empty((1, LAYERS, full_len, TOP_K), dtype=np.int16)
    for layer in range(LAYERS):
        for t in range(full_len):
            rows[0, layer, t] = rng.choice(EXPERTS, size=TOP_K, replace=False)

    config = _surogate.PretrainedConfig.from_pretrained(str(model_dir), "bf16")
    lora = _surogate.LoRAAdapterConfig(rank=8, alpha=16, dropout=0.0, dtype="bf16",
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                       use_rslora=False, train_router=False)
    flush = _surogate.OptimizerConfig(optimizer="adamw_8bit", learning_rate=0.0, weight_decay=0.0, grad_clip=0.0)
    weights_dir = tmp_path / "weights"

    def make_trainer(seq_chunks):
        options = _surogate.RuntimeOptions(offload_residual=False, use_cuda_graphs=False, offload_master=False,
                                           offload_grads=False, offload_optimizer=False, shard_gradients=True,
                                           use_zero_copy=False)
        options.sequence_chunks = seq_chunks
        options.dsl_ir_json = build_dsl_ir_for_model(str(model_dir))
        manifests = compile_jit_kernels(options.dsl_ir_json, rollout_parity=False)
        if manifests:
            options.jit_kernel_manifests = manifests
        trainer = _surogate.SurogateTrainer(ngpu=1, config=config, options=options, batch_size=1,
                                            seq_len=full_len // seq_chunks, grad_accum=1, memcpy_all_gather=True,
                                            memcpy_send_recv=True, lora_config=lora, qlora_config=None)
        if weights_dir.exists():
            trainer.import_weights(get_model_weights_path(str(weights_dir)))
        else:
            trainer.init_weights()
            trainer.export_model(str(weights_dir))
            (weights_dir / "config.json").write_text((model_dir / "config.json").read_text())
        return trainer

    def grad_norm(seq_chunks, replay):
        trainer = make_trainer(seq_chunks)
        if replay:
            trainer.set_routing_replay(rows)
        trainer.step_grpo_native(inputs, targets, inference_logprobs, advantages, loss_mask,
                                 np.array([0], dtype=np.int32), np.array([full_len], dtype=np.int32),
                                 position_ids=position_ids, loss_scale=float(loss_mask.sum()), kl_tau=0.1)
        diagnostics = trainer.routing_replay_diagnostics() if replay else None
        if seq_chunks > 1 and replay:
            # Rows of one chunk's length are refused: chunked steps slice the full-sequence rows.
            with pytest.raises(Exception):
                trainer.set_routing_replay(np.ascontiguousarray(rows[:, :, :CHUNK, :]))
        trainer.clear_routing_replay()
        norm = float(trainer.update_with_config(flush, 1)["norm"])
        del trainer
        return norm, diagnostics

    replayed, d_unchunked = grad_norm(1, replay=True)
    chunked, d_chunked = grad_norm(SEQ_CHUNKS, replay=True)
    own, _ = grad_norm(1, replay=False)

    assert d_unchunked == [(0, 0)], d_unchunked
    assert d_chunked == [(0, 0)], d_chunked  # every chunk forward found rows of the right shape
    assert replayed > 0.0 and own > 0.0
    print(f"grad norms: replayed unchunked {replayed:.6f}, replayed chunked {chunked:.6f}, own routing {own:.6f}")
    # Replay changed the routing (forced experts move the gradient away from the router's own picks), and the
    # chunked step reproduces the replayed step, not the own-routing one: its gap to the replayed norm is a small
    # fraction of the replay effect (measured 2026-10-16 on Spark 1: 4.7628 / 4.7678 / 4.5668).
    effect = abs(replayed - own)
    assert effect > 0.01 * own, (replayed, own)
    assert abs(chunked - replayed) < 0.25 * effect, (chunked, replayed, own)
