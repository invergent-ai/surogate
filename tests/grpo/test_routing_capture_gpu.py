"""Colocate Rollout Routing Replay: SharedModelServer returns the experts its own serving forwards selected.

A tiny random-weight Qwen3-MoE (2 layers, 4 experts, top-2) serves through each SharedModelServer path: the
continuous-batching scheduler (with and without decode CUDA graphs), the persistent decode state, and full
re-forwards. Checks: (1) capture changes nothing - the same request gives the same tokens and logprobs before capture
was ever switched on and with it on, and only requests asking for it carry the payload; (2) the captured ids are the
model's selection - they equal the routing a separate full forward over the same tokens selects; (3) round trip -
the payload, stitched and laid out by the trainer-side R3 plumbing and replayed in the trainer, gives no bad rows and
no shape mismatches and reproduces the served logprobs, while corrupted rows do not (the replay is live).
"""

import base64
import json

import numpy as np
import pytest
import requests

from surogate.grpo.routing_replay import SampleRouting, moe_layer_ids, rows_from_bytes, trainer_rows

LAYERS, EXPERTS, TOP_K = 2, 4, 2
SEQ_LEN = 128
PROMPT_A = [5, 7, 9, 11, 13, 17, 19]
PROMPT_B = [23, 29, 31, 37, 41]


class Tokenizer:
    eos_token_id = 1

    def decode(self, ids, **kwargs):
        return "".join(chr(65 + token % 26) for token in ids)


class NoBatchedDecode:
    """The trainer without continuous batching: SharedModelServer falls back to decode_logits / full re-forwards."""

    hidden = ("decode_batch_logits", "decode_batch_sample", "admit_decode_sessions")

    def __init__(self, trainer):
        self._trainer = trainer

    def __getattr__(self, name):
        if name in self.hidden:
            raise AttributeError(name)
        return getattr(self._trainer, name)


def _config():
    return dict(architectures=["Qwen3MoeForCausalLM"], model_type="qwen3_moe", hidden_size=256,
                intermediate_size=512, num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
                head_dim=64, vocab_size=512, max_position_embeddings=256, rms_norm_eps=1e-6, hidden_act="silu",
                tie_word_embeddings=True, rope_theta=10000.0, attention_bias=False, eos_token_id=1,
                num_experts=EXPERTS, num_experts_per_tok=TOP_K, moe_intermediate_size=128)


def _make_trainer(root, graphs):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels

    options = ext.RuntimeOptions(recompute="true", use_cuda_graphs=graphs, master_dtype="bf16", offload_master=False,
                                 offload_grads=False, offload_optimizer=False)
    options.dsl_ir_json = build_dsl_ir_for_model(str(root))
    options.moe_rollout_parity = True
    manifests = compile_jit_kernels(options.dsl_ir_json, rollout_parity=True)
    if manifests:
        options.jit_kernel_manifests = manifests
    lora = ext.LoRAAdapterConfig(rank=8, alpha=16, dropout=0.0, dtype="bf16", target_modules=["all"])
    trainer = ext.SurogateTrainer(ngpu=1, config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
                                  options=options, batch_size=1, seq_len=SEQ_LEN, grad_accum=1, lora_config=lora)
    trainer.init_weights()
    return trainer


def _payload_rows(choice):
    payload = choice["routed_experts"]
    assert list(payload) == ["data", "shape", "start", "dtype"] and payload["dtype"] == "int16"
    rows = np.frombuffer(base64.b64decode(payload["data"]), dtype="<i2").reshape(payload["shape"])
    return payload["start"], rows


def _scoring_batch(prompt, generated):
    inputs = np.zeros((1, SEQ_LEN), dtype=np.int32)
    inputs[0, : len(prompt) + len(generated)] = prompt + generated
    targets = np.full_like(inputs, -100)
    targets[0, len(prompt) - 1 : len(prompt) + len(generated) - 1] = generated
    return inputs, targets


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("mode,graphs", [("scheduler", True), ("scheduler", False), ("persistent", False),
                                         ("full", False)])
def test_colocate_capture_round_trips_through_replay(tmp_path, mode, graphs):
    pytest.importorskip("surogate._surogate", reason="needs the built extension")
    import torch

    from surogate.grpo.shared_model import SharedModelServer

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = _config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    trainer = _make_trainer(tmp_path, graphs)
    served = trainer if mode == "scheduler" else NoBatchedDecode(trainer)
    server = SharedModelServer(served, Tokenizer(), config,
                               dict(model="base", max_context=SEQ_LEN, max_concurrency=4, host="127.0.0.1", port=0))
    if mode == "persistent":
        server.persistent_decode = True
    assert (server.scheduler is not None) == (mode == "scheduler")
    url = f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions/tokens"

    def generate(prompt, routed):
        body = dict(model="policy", tokens=prompt, max_tokens=24, temperature=1.0, seed=43, logprobs=True,
                    ignore_eos=True)
        if routed:
            body["return_routed_experts"] = True
        response = requests.post(url, json=body, timeout=300)
        assert response.ok, response.text
        return response

    try:
        server.publish("policy", [], 0)
        # (1) The first request runs before capture was ever switched on.
        off = generate(PROMPT_A, routed=False).json()
        on_response = generate(PROMPT_A, routed=True)
        on = on_response.json()
        assert '"routed_experts": {"data": "' in on_response.text
        assert "routed_experts" not in off["choices"][0]
        assert on["choices"][0]["token_ids"] == off["choices"][0]["token_ids"]
        assert ([s["logprob"] for s in on["choices"][0]["logprobs"]["content"]] ==
                [s["logprob"] for s in off["choices"][0]["logprobs"]["content"]])
        again = generate(PROMPT_A, routed=False).json()
        assert "routed_experts" not in again["choices"][0]
        assert again["choices"][0]["token_ids"] == off["choices"][0]["token_ids"]
        fresh = generate(PROMPT_B, routed=True).json()
        server.begin_training()

        moe_layers = moe_layer_ids(config, lambda c, name: c[name])
        for result in (on, fresh):
            prompt, generated = result["prompt_token_ids"], result["choices"][0]["token_ids"]
            ran = len(prompt) + len(generated) - 1  # the final sampled token never runs through the model
            start, rows = _payload_rows(result["choices"][0])
            # Positions [start, ran): the scheduler's prefix cache may cover the start of a repeated prompt.
            assert rows.shape == (ran - start, LAYERS, TOP_K)
            assert 0 <= start <= len(prompt) - 1 and (start == 0 or mode == "scheduler")
            assert (rows >= 0).all() and (rows < EXPERTS).all()
            assert all(len(set(r)) == TOP_K for r in rows.reshape(-1, TOP_K).tolist())

            # (2) A full forward over the same tokens (captured through next_token_logits) selects the same experts.
            inputs = np.zeros((1, SEQ_LEN), dtype=np.int32)
            inputs[0, :ran] = (prompt + generated)[:ran]
            trainer.next_token_logits(inputs, np.asarray([ran - 1], dtype=np.int32))
            reference = trainer.last_routing_capture()[0, start:ran]
            mismatched = int((np.sort(reference, axis=-1) != np.sort(rows, axis=-1)).any(axis=-1).sum())
            print(f"{mode} graphs={graphs} start={start} rows={rows.shape[0]} "
                  f"token-layer selections differing from a full forward: {mismatched}")
            assert mismatched == 0

            # (3) Round trip through the trainer-side plumbing and replay.
            sample = SampleRouting()
            sample.add_step(len(prompt), len(generated), result["choices"][0]["routed_experts"])
            data, dims = sample.result()
            routing = rows_from_bytes(data, dims, len(prompt) + len(generated))
            replay = trainer_rows(routing, SEQ_LEN, LAYERS, moe_layers)
            assert replay is not None and (replay[:, start:ran] >= 0).all() and (replay[:, ran:, 0] == -1).all()
            inputs, targets = _scoring_batch(prompt, generated)
            served_scores = np.asarray([s["logprob"] for s in result["choices"][0]["logprobs"]["content"]])
            span = slice(len(prompt) - 1, len(prompt) - 1 + len(generated))

            trainer.set_routing_replay(replay[None])
            replayed = trainer.compute_logprobs(inputs, targets)[0, span]
            diagnostics = trainer.routing_replay_diagnostics()
            trainer.clear_routing_replay()
            print(f"replayed max |logprob - served| = {np.max(np.abs(replayed - served_scores)):.3g}, "
                  f"diagnostics {diagnostics}")
            assert diagnostics == [(0, 0)]
            np.testing.assert_allclose(replayed, served_scores, atol=1e-5, rtol=0)

            # Negative control: the other two experts everywhere replay changes the logprobs.
            corrupt = replay.copy()
            for layer in range(LAYERS):
                for t in range(start, ran):
                    corrupt[layer, t] = sorted(set(range(EXPERTS)) - set(replay[layer, t].tolist()))
            trainer.set_routing_replay(corrupt[None])
            corrupted = trainer.compute_logprobs(inputs, targets)[0, span]
            trainer.clear_routing_replay()
            print(f"corrupted max |logprob - served| = {np.max(np.abs(corrupted - served_scores)):.3g}")
            assert np.max(np.abs(corrupted - served_scores)) > 1e-3
    finally:
        server.close()
