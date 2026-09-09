"""GPU integration: tool trajectory -> native GRPO update, plus optimized Qwen."""

import asyncio
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

from surogate.grpo.shared_model import SharedModelServer
from tests.grpo.test_tool_protocol import CALLS, TOOLS, tool_rollout

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def native_trainer(root, seq_len):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    config = json.loads((root / "config.json").read_text())
    glm = config.get("model_type") == "glm5_next"
    options = ext.RuntimeOptions(
        recompute="true",
        use_cuda_graphs=False,
        master_dtype="bf16",
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(root))
    options.jit_kernel_manifests = compile_jit_kernels(options.dsl_ir_json)
    options.glm_rollout_parity = glm
    targets = ["all"] if glm else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = ext.LoRAAdapterConfig(rank=8, alpha=16, dropout=0.0, dtype="bf16" if glm else "fp32", target_modules=targets)
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
        options=options,
        batch_size=1,
        seq_len=seq_len,
        grad_accum=1,
        lora_config=lora,
    )
    trainer.import_weights(get_model_weights_path(str(root)))
    return trainer, config


def test_glm_tool_rollout_scores_and_updates_native_grpo(tmp_path, monkeypatch):
    from examples.sft.glm.create_dummy import create_dummy, dummy_tokenizer
    from surogate import _surogate as ext
    from surogate.grpo.batch import prepare_sample
    from surogate.grpo.orchestrator.advantage import AdvantageInputs, default_advantage_fn
    from surogate.grpo.orchestrator.trajectories import interleave_rollout

    seq_len = 2048
    create_dummy(tmp_path, index_topk=32, max_sequence_length=seq_len)
    tokenizer = dummy_tokenizer()
    trainer, config = native_trainer(tmp_path, seq_len)
    server = SharedModelServer(
        trainer,
        tokenizer,
        config,
        dict(
            model="base",
            host="127.0.0.1",
            port=0,
            max_context=seq_len,
            max_concurrency=1,
            eos_token_id=[1, tokenizer.convert_tokens_to_ids("<|observation|>")],
        ),
    )
    server.publish("policy", [], 0)
    raw = "compute</think>" + CALLS["arg_xml"]
    first = tokenizer.encode(raw, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|observation|>")]
    last = tokenizer.encode("done</think>5", add_special_tokens=False) + [1]
    chosen = iter(first + last)
    original_rng = np.random.default_rng

    # The random dummy model has no learned tool behavior. Force exploration in
    # this fixture's sampler only; every logit and returned policy logprob comes
    # from native GLM. This tests training plumbing, not learned agent quality.
    def test_rng(seed=None):
        return SimpleNamespace(choice=lambda *args, **kwargs: next(chosen)) if seed == 73191 else original_rng(seed)

    monkeypatch.setattr(np.random, "default_rng", test_rng)
    try:
        output = asyncio.run(tool_rollout(server, temperature=1, seed=73191))
        server.begin_training()
        assert [t for step in output["trajectory"] for t in step["tokens"]["completion_ids"]] == first + last
        samples = interleave_rollout(output)
        assert len(samples) == 1
        sample = samples[0]
        # A successful rollout versus the zero-reward baseline in its group.
        sample.advantage = (
            default_advantage_fn(
                AdvantageInputs(
                    rewards=torch.tensor([[output["reward"], 0.0]]),
                    completion_lengths=torch.tensor([[len(first) + len(last), 1]]),
                )
            )
            .advantages[0, 0]
            .item()
        )
        batch = prepare_sample(sample, seq_len)
        length = len(batch.input_ids)
        assert length == len(sample.prompt_ids) + len(sample.completion_ids) < seq_len
        inputs = np.zeros((1, seq_len), dtype=np.int32)
        targets = np.full_like(inputs, -100)
        mask = np.zeros(seq_len, dtype=np.uint8)
        inference = np.zeros(seq_len, dtype=np.float32)
        advantages = np.zeros(seq_len, dtype=np.float32)
        positions = np.arange(seq_len, dtype=np.int32)[None, :]
        inputs[0, :length] = batch.input_ids
        mask[:length] = batch.loss_mask
        inference[:length] = batch.inference_logprobs
        advantages[:length] = batch.advantages
        targets[0, : length - 1] = np.where(mask[1:length], inputs[0, 1:length], -100)
        assert int(mask.sum()) == len(first) + len(last)
        before = trainer.compute_logprobs(inputs, targets, position_ids=positions).copy()
        logical_scores = np.roll(before[0], 1)
        np.testing.assert_allclose(logical_scores[mask.astype(bool)], inference[mask.astype(bool)], atol=1e-5, rtol=0)
        trainer.step_grpo_native(
            inputs,
            targets,
            inference,
            advantages,
            mask,
            np.array([0], dtype=np.int32),
            np.array([length], dtype=np.int32),
            position_ids=positions,
            temperatures=None,
            teacher_logprobs=None,
            loss_scale=float(mask.sum()),
            ipo_mask_low=0.2,
            ipo_mask_high=0.2,
            adv_tau=1.0,
            teacher_tau=0.0,
            kl_tau=0.0,
            ratio_clip=0.2,
        )
        metrics = trainer.get_grpo_native_metrics()
        assert metrics and all(np.isfinite(v) for v in metrics.values())
        assert abs(metrics["policy_loss"]) > 0
        trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-3), 1)
        after = trainer.compute_logprobs(inputs, targets, position_ids=positions).copy()
        assert np.isfinite(after).all()
        assert not np.allclose(before[targets != -100], after[targets != -100], atol=1e-4, rtol=0)
        server.publish("policy", [], 1)
        print(
            f"tool GRPO: {length} tokens, {int(mask.sum())} model tokens, reward={output['reward']}, metrics={metrics}"
        )
    finally:
        server.close()


@pytest.mark.skipif(
    not os.environ.get("SUROGATE_SHARED_MODEL"), reason="set SUROGATE_SHARED_MODEL to a local Qwen checkpoint"
)
def test_optimized_qwen_tool_loop(tmp_path):
    from surogate import _surogate_serve
    from surogate.grpo.shared_weights import adapter_modules, borrow_weights, write_shared_artifact

    root = Path(os.environ["SUROGATE_SHARED_MODEL"])
    trainer, _ = native_trainer(root, 2048)
    artifact = tmp_path / "shared.sinfer"
    weights = borrow_weights(trainer, write_shared_artifact(str(root), artifact))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = _surogate_serve.SharedServer(
        str(artifact),
        weights,
        dict(
            host="127.0.0.1",
            port=port,
            device=0,
            model="base",
            max_context=2048,
            prefill_chunk=128,
            max_concurrency=1,
            kv_capacity=2048,
            use_cuda_graph=True,
            rank=8,
        ),
    )
    try:
        server.publish("policy", adapter_modules(trainer, 2.0), 0)
        output = asyncio.run(
            tool_rollout(
                SimpleNamespace(http=SimpleNamespace(server_port=port)),
                extra_body=dict(chat_template_kwargs=dict(enable_thinking=False)),
            )
        )
        assert output["reward"] == 1
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=dict(
                model="policy",
                messages=[
                    dict(
                        role="user",
                        content=(
                            "First call the add tool with a=2 and b=3. Wait for its result before answering. "
                            "Then reply with only the resulting number."
                        ),
                    )
                ],
                tools=TOOLS,
                stream=True,
                logprobs=True,
                return_token_ids=True,
                temperature=0,
                max_tokens=128,
                chat_template_kwargs=dict(enable_thinking=False),
            ),
            timeout=30,
        )
        assert response.status_code == 200, response.text
        choices = [
            json.loads(line[6:])["choices"][0]
            for line in response.text.splitlines()
            if line.startswith("data: {") and json.loads(line[6:])["choices"]
        ]
        ids = [t for choice in choices for t in choice.get("token_ids", [])]
        scores = [s for choice in choices for s in (choice.get("logprobs") or {}).get("content", [])]
        assert ids and len(ids) == len(scores)
        assert choices[-1]["finish_reason"] == "tool_calls"
        assert any(choice["delta"].get("tool_calls") for choice in choices)
        assert server.summary()["serving_base_allocated_bytes"] == 0
        assert server.summary()["base_upload_bytes"] == 0
    finally:
        server.close()
