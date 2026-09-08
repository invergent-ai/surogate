"""Real trainer/serving parity and handoff checks for supported shared models."""

import concurrent.futures
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
import requests
import torch

from surogate.grpo.shared_weights import adapter_modules, borrow_weights, write_shared_artifact

MODEL = os.environ.get("SUROGATE_SHARED_MODEL", "")


@pytest.mark.skipif(not MODEL, reason="set SUROGATE_SHARED_MODEL and CUDA_VISIBLE_DEVICES to one test GPU")
def test_shared_generation_adapter_update_and_owner_lifetime(tmp_path):
    from surogate import _surogate, _surogate_serve
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.utils.hf import get_model_weights_path

    options = _surogate.RuntimeOptions(use_cuda_graphs=False, master_dtype="bf16", offload_master=False,
                                       offload_grads=False, offload_optimizer=False)
    options.dsl_ir_json = build_dsl_ir_for_model(MODEL)
    from surogate.kernels.jit_compile import compile_jit_kernels
    manifests = compile_jit_kernels(options.dsl_ir_json)
    if manifests:
        options.jit_kernel_manifests = manifests
    config = _surogate.PretrainedConfig.from_pretrained(MODEL, "bf16")
    hf_config = json.loads((Path(MODEL) / "config.json").read_text())
    # Hybrid BF16 recurrence and different prefill batch widths amplify rounding.
    # On 0.8B, independent HF comparisons put the final hidden-state difference
    # at 2–3%, with a 0.26 logprob spread between training and graphed serving.
    score_tolerance = .30 if "Qwen3_5" in hf_config["architectures"][0] else .15
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora = _surogate.LoRAAdapterConfig(rank=8, alpha=16, dropout=0., dtype="fp32",
        target_modules=targets)
    trainer = _surogate.SurogateTrainer(ngpu=1, config=config, options=options, batch_size=1,
                                       seq_len=128, grad_accum=1, lora_config=lora)
    trainer.import_weights(get_model_weights_path(MODEL))
    artifact = tmp_path / "shared.sinfer"
    weights = borrow_weights(trainer, write_shared_artifact(MODEL, artifact))
    base = trainer.get_shared_base_weights()
    if hf_config.get("text_config", hf_config).get("tie_word_embeddings", hf_config.get("tie_word_embeddings")):
        assert torch.from_dlpack(base["embedding"]).data_ptr() == torch.from_dlpack(base["lm_head"]).data_ptr()
    settings = dict(host="127.0.0.1", port=int(os.environ.get("SUROGATE_SHARED_PORT", "18653")), device=0,
                    model="qwen3", max_context=256, prefill_chunk=128, max_concurrency=4,
                    kv_capacity=1024, use_cuda_graph=True, rank=8)
    bad = dict(weights)
    del bad["text/final_norm"]
    with pytest.raises(Exception, match="cover every device tensor"):
        _surogate_serve.SharedServer(str(artifact), bad, settings)
    del bad
    server = _surogate_serve.SharedServer(str(artifact), weights, settings)
    url = f"http://127.0.0.1:{settings['port']}/v1/chat/completions"
    body = dict(model="default", messages=[dict(role="user", content="What is 7 + 8? Answer briefly.")],
                max_tokens=16, temperature=0, logprobs=True, return_token_ids=True,
                chat_template_kwargs=dict(enable_thinking=False))

    def generate(payload=body):
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def score_in_trainer(result):
        prompt = result["prompt_token_ids"]
        completion = result["choices"][0]["token_ids"]
        tokens = prompt + completion
        ids = np.zeros((1, 128), dtype=np.int32)
        targets = np.full_like(ids, -100)
        ids[0, :len(tokens)] = tokens
        targets[0, len(prompt)-1:len(tokens)-1] = completion
        scores = trainer.compute_logprobs(ids, targets)
        return scores[0, len(prompt)-1:len(tokens)-1].copy()

    def serving_scores(result):
        return np.array([t["logprob"] for t in result["choices"][0]["logprobs"]["content"]])

    try:
        assert requests.post(url, json=body | {"model": "qwen3"}, timeout=10).status_code == 429
        server.publish("default", adapter_modules(trainer, 2.), 0)
        assert {m["module"] for m in adapter_modules(trainer, 2.)} == set(targets)
        first = generate()
        assert server.summary()["base_upload_bytes"] == 0
        assert server.summary()["serving_base_allocated_bytes"] == 0
        server.begin_training()
        initial_scores = score_in_trainer(first)
        np.testing.assert_allclose(serving_scores(first), initial_scores, atol=score_tolerance, rtol=0)

        # Update every projection, including the opposite halves of the fused
        # training MLP; publishing stale tensors or swapped halves breaks parity.
        from safetensors.torch import load_file, save_file

        trainer.export_adapter(str(tmp_path / "adapter"))
        adapter_path = tmp_path / "adapter/adapter_model.safetensors"
        torch.manual_seed(17)
        live = load_file(adapter_path)
        for name, tensor in live.items():
            if "lora_B" in name:
                tensor.normal_(0, .02)
        save_file(live, adapter_path)
        trainer.import_adapter(str(adapter_path))
        server.publish("default", adapter_modules(trainer, 2.), 1)
        updated = generate()
        server.begin_training()
        updated_scores = score_in_trainer(updated)
        (tmp_path / "scores.json").write_text(json.dumps(dict(response=updated, trainer=updated_scores.tolist())))
        np.testing.assert_allclose(serving_scores(updated), updated_scores, atol=score_tolerance, rtol=0)
        assert not np.allclose(score_in_trainer(first), initial_scores, atol=1e-3, rtol=0)
        with pytest.raises(ValueError, match="version"):
            server.publish("default", adapter_modules(trainer, 2.), 1)
        server.publish("default", adapter_modules(trainer, 2.), 2)

        # Concurrent prompts exercise independent recurrent states and prefix
        # reuse after publication.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            repeated = list(executor.map(lambda _: generate(), range(4)))
        assert all(result["choices"][0]["token_ids"] == updated["choices"][0]["token_ids"] for result in repeated)

        fresh_body = body | dict(messages=[dict(role="user", content="What is the capital of France?")])
        fresh = generate(fresh_body)
        server.begin_training()
        server.publish("default", adapter_modules(trainer, 2.), 3)

        # Pause while a streamed response is in flight. Already-admitted work
        # must finish before sleeping; the worker must not park before draining.
        long_body = body | dict(max_tokens=96, stream=True,
            messages=[dict(role="user", content="Count from 1 to 100, one number per line.")])
        with requests.post(url, json=long_body, stream=True, timeout=60) as response:
            response.raise_for_status()
            lines = response.iter_lines()
            next(line for line in lines if line.startswith(b"data:"))
            # A new prompt arriving during decode must retain its adapter too.
            overlapping = generate(fresh_body)
            assert overlapping["choices"][0]["token_ids"] == fresh["choices"][0]["token_ids"]
            # Prefill retains its shape; decode now uses a wider BF16 batch.
            np.testing.assert_allclose(serving_scores(overlapping)[:1], serving_scores(fresh)[:1], atol=1e-3, rtol=0)
            np.testing.assert_allclose(serving_scores(overlapping), serving_scores(fresh),
                                       atol=score_tolerance, rtol=0)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                paused = executor.submit(server.begin_training)
                assert any(line == b"data: [DONE]" for line in lines)
                paused.result(timeout=15)
        assert server.summary()["sleeping"]
        assert requests.post(url, json=body, timeout=10).status_code == 429
        server.publish("default", adapter_modules(trainer, 2.), 4)
        del live, base, weights, trainer
        gc.collect()
        after_release = generate()
        assert after_release["choices"][0]["token_ids"] == updated["choices"][0]["token_ids"]
        np.testing.assert_allclose(serving_scores(after_release), serving_scores(updated), atol=.05, rtol=0)
    finally:
        server.close()
        server.close()
