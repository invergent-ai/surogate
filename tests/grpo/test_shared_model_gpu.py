"""Opt-in generation/scoring/update checks on every training family, excluding Nemotron."""

import concurrent.futures
import json
import os
from pathlib import Path

import numpy as np
import pytest
import requests
import torch

from surogate.grpo.shared_model import SharedModelServer
from tests.grpo.shared_model_configs import configurations

# Real checkpoint, real cards: minutes, not milliseconds. Runs under `--slow`.
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

CASES = configurations()
SELECTED = os.environ.get("SUROGATE_SHARED_CASES", "").split(",")
REAL = os.environ.get("SUROGATE_SHARED_MODEL", "")


class Tokenizer:
    eos_token_id = 1

    def decode(self, ids, **kwargs):
        return "".join(chr(65 + token % 26) for token in ids)


@pytest.mark.skipif(
    not os.environ.get("SUROGATE_SHARED_CASES") and not REAL,
    reason="set SUROGATE_SHARED_CASES=all or SUROGATE_SHARED_MODEL and select a test GPU",
)
@pytest.mark.parametrize(
    "case", ["real"] if REAL else list(CASES) if SELECTED == ["all"] else [s for s in SELECTED if s in CASES]
)
def test_generation_scoring_and_training_share_the_policy(tmp_path, case):
    root = Path(REAL) if case == "real" else tmp_path
    if case != "real":
        (root / "config.json").write_text(json.dumps(CASES[case]))
    _check_policy(
        root,
        load_weights=case == "real",
        batch_size=int(os.environ.get("SUROGATE_SHARED_BATCH", "1")),
        seq_len=int(os.environ.get("SUROGATE_SHARED_SEQ_LEN", "128")),
        rollout_tokens=int(os.environ.get("SUROGATE_SHARED_TOKENS", "4")),
        graphs=os.environ.get("SUROGATE_SHARED_GRAPHS") == "1",
        recompute=os.environ.get("SUROGATE_SHARED_RECOMPUTE", "true"),
    )


@pytest.fixture(scope="module")
def glm_checkpoint(tmp_path_factory):
    from examples.sft.glm.create_dummy import create_dummy

    root = tmp_path_factory.mktemp("glm_rollout")
    # Keep the raw random checkpoint, including its nearly tied expert scores.
    create_dummy(root, index_topk=32, max_sequence_length=512)
    return root


@pytest.mark.parametrize("batch_size,graphs,alpha,temperature", [(1, False, 16, 0), (2, True, 13, 1)])
def test_glm_long_rollout_matches_packed_scoring_after_updates(glm_checkpoint, batch_size, graphs, alpha, temperature):
    _check_policy(
        glm_checkpoint,
        load_weights=True,
        batch_size=batch_size,
        seq_len=512,
        rollout_tokens=248,
        graphs=graphs,
        alpha=alpha,
        packed=True,
        temperature=temperature,
    )


def _check_policy(
    root,
    *,
    load_weights,
    batch_size,
    seq_len,
    rollout_tokens,
    graphs,
    recompute="true",
    alpha=16,
    packed=False,
    temperature=0,
):
    from surogate import _surogate as ext
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    config = json.loads((root / "config.json").read_text())
    options = ext.RuntimeOptions(
        recompute=recompute,
        use_cuda_graphs=graphs,
        master_dtype="bf16",
        offload_master=False,
        offload_grads=False,
        offload_optimizer=False,
    )
    options.dsl_ir_json = build_dsl_ir_for_model(str(root))
    manifests = compile_jit_kernels(options.dsl_ir_json)
    if manifests:
        options.jit_kernel_manifests = manifests
    targets = ["all"]
    text_config = config.get("text_config", config)
    glm = config.get("model_type") == "glm5_next" or text_config.get("model_type") in ("glm5_next", "glm5_next_text")
    if glm:
        options.glm_rollout_parity = True
    dtype = (
        "bf16"
        if text_config.get("num_experts", text_config.get("num_local_experts", text_config.get("n_routed_experts", 0)))
        else "fp32"
    )
    lora = ext.LoRAAdapterConfig(rank=8, alpha=alpha, dropout=0.0, dtype=dtype, target_modules=targets)
    trainer = ext.SurogateTrainer(
        ngpu=1,
        config=ext.PretrainedConfig.from_pretrained(str(root), "bf16"),
        options=options,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum=1,
        lora_config=lora,
    )
    if load_weights:
        trainer.import_weights(get_model_weights_path(str(root)))
    else:
        trainer.init_weights()
    original = {n: torch.from_dlpack(t).data_ptr() for n, t in trainer.get_shared_base_weights().items()}
    server = SharedModelServer(
        trainer,
        Tokenizer(),
        config,
        dict(model="base", max_context=seq_len, max_concurrency=4, host="127.0.0.1", port=0),
    )
    url = f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions/tokens"
    body = dict(
        model="policy",
        tokens=[5, 7, 9, 11, 13],
        max_tokens=rollout_tokens,
        temperature=temperature,
        seed=43,
        logprobs=True,
        ignore_eos=True,
    )

    def generate():
        response = requests.post(url, json=body, timeout=120)
        assert response.ok, response.text
        return response.json()

    def scoring_batch(result):
        prompt, generated = result["prompt_token_ids"], result["choices"][0]["token_ids"]
        inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
        inputs[0, : len(prompt) + len(generated)] = prompt + generated
        targets = np.full_like(inputs, -100)
        targets[0, len(prompt) - 1 : len(prompt) + len(generated) - 1] = generated
        return inputs, targets

    def packed_batch(result, shift):
        prompt, generated = result["prompt_token_ids"], result["choices"][0]["token_ids"]
        document = prompt + generated[:-1]
        inputs = np.zeros((batch_size, seq_len), dtype=np.int32)
        targets = np.full_like(inputs, -100)
        positions = np.tile(np.arange(seq_len, dtype=np.int32), (batch_size, 1))
        selections = []
        for row in range(batch_size):
            offset = shift + 2 * row
            for _ in range(2):
                end = offset + len(document)
                assert end < seq_len
                inputs[row, offset:end] = document
                positions[row, offset:end] = np.arange(len(document))
                start = offset + len(prompt) - 1
                targets[row, start:end] = generated
                selections.append((row, slice(start, end)))
                offset = end
            positions[row, offset:] = np.arange(seq_len - offset)
        return inputs, targets, positions, selections

    def check_scores(result):
        inputs, targets = scoring_batch(result)
        start = len(result["prompt_token_ids"]) - 1
        teacher = server.trainer.compute_logprobs(inputs, targets)[
            0, start : start + len(result["choices"][0]["token_ids"])
        ]
        scores = [s["logprob"] for s in result["choices"][0]["logprobs"]["content"]]
        print(f"rollout tokens={len(scores)} max_logprob_error={np.max(np.abs(scores - teacher)):.6g}")
        # GLM uses the same recurrence and fixed GEMM reduction at every batch
        # size. Only the CPU/GPU log-softmax calculation should differ.
        np.testing.assert_allclose(scores, teacher, atol=1e-5 if glm else 0.08, rtol=0)
        if packed:
            # Shift identical rollouts around a packed batch, keeping document
            # count fixed so the second call can replay captured CUDA graphs.
            for shift in (1, 3):
                ids, targets, positions, selections = packed_batch(result, shift)
                teacher = server.trainer.compute_logprobs(ids, targets, position_ids=positions)
                for row, tokens in selections:
                    np.testing.assert_allclose(scores, teacher[row, tokens], atol=1e-5, rtol=0)

    try:
        assert requests.post(url, json=body, timeout=10).status_code == 429
        server.publish("policy", [], 0)
        before = generate()
        server.begin_training()
        check_scores(before)
        inputs, target_ids = scoring_batch(before)
        kwargs = {}
        if packed:
            inputs, target_ids, positions, _ = packed_batch(before, 1)
            kwargs["position_ids"] = positions
        old_scores = trainer.compute_logprobs(inputs, target_ids, **kwargs).copy()
        gradients = (target_ids != -100).astype(np.float32)
        # A real optimizer update, followed by generation from the live adapter.
        for step in range(2):
            trainer.step_with_custom_loss(inputs, target_ids, gradients, **kwargs)
            update = trainer.update_with_config(ext.OptimizerConfig(learning_rate=1e-3), step + 1)
            assert np.isfinite(update["norm"]) and update["norm"] > 0
            for name, value in trainer.get_lora_weights(0).items():
                assert torch.isfinite(torch.from_dlpack(value)).all(), (step, name, update)
        new_scores = trainer.compute_logprobs(inputs, target_ids, **kwargs).copy()
        assert np.max(np.abs(new_scores - old_scores)) > 1e-3
        server.publish("policy", [], 1)
        after = generate()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            repeated = list(pool.map(lambda _: generate(), range(2)))
        assert all(r["choices"][0]["token_ids"] == after["choices"][0]["token_ids"] for r in repeated)
        server.begin_training()
        check_scores(after)
        summary = server.summary()
        assert summary["base_upload_bytes"] == summary["serving_base_allocated_bytes"] == 0
        assert summary["shared_base_bytes"] > 0
        assert original == {n: torch.from_dlpack(t).data_ptr() for n, t in trainer.get_shared_base_weights().items()}
        with pytest.raises(ValueError, match="version"):
            server.publish("policy", [], 1)
        # The server itself retains the model owner until its admitted work drains.
        server.publish("policy", [], 2)
        del trainer
        assert generate()["choices"][0]["token_ids"] == after["choices"][0]["token_ids"]
    finally:
        server.close()
        server.close()
