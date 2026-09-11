"""Opt-in Qwen3.5 MTP scoring regression with a small embedding/head adapter.

Set SUROGATE_MTP_SCORE_TEST_ARTIFACT to a prepared Qwen3.5 artifact with MTP.
Reserve the test GPU using CUDA_VISIBLE_DEVICES.
"""

import json
import os

import pytest

from tests.serve import test_mixed_prefill_decode as mixed

pytestmark = pytest.mark.skipif(
    not os.getenv("SUROGATE_MTP_SCORE_TEST_ARTIFACT"), reason="requires a prepared MTP artifact"
)


@pytest.mark.parametrize("active_slots", [1, 4])
def test_adapter_scores_across_chunk_boundaries(active_slots, tmp_path, tmp_path_factory, monkeypatch):
    import torch
    from safetensors.torch import save_file

    from surogate.serve.artifact.container import Artifact

    artifact = os.environ["SUROGATE_MTP_SCORE_TEST_ARTIFACT"]
    with Artifact.open(artifact) as model:
        embedding = next(obj for obj in model.objects if obj.name == "text/token_embedding")
        vocab, hidden = embedding.shape

    adapters = []
    names = ["adapter-a", "adapter-b"]
    for index, name in enumerate(names):
        root = tmp_path / name
        root.mkdir()
        (root / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "r": 2, "lora_alpha": 2,
            "target_modules": ["embed_tokens", "lm_head"], "lora_bias": True,
        }))
        embedding_a = torch.zeros((2, vocab), dtype=torch.bfloat16)
        embedding_b = torch.zeros((hidden, 2), dtype=torch.bfloat16)
        bias = torch.zeros(vocab, dtype=torch.bfloat16)
        embedding_a[0, 100 + index] = .25
        embedding_b[:32, 0] = .125
        # Keep the adapter's token competitive, without making its probability
        # saturate and conceal changes in the target logits.
        bias[100 + index] = 12
        save_file({
            "base_model.model.model.embed_tokens.lora_embedding_A": embedding_a,
            "base_model.model.model.embed_tokens.lora_embedding_B": embedding_b,
            "base_model.model.lm_head.lora_A.weight": torch.zeros((2, hidden), dtype=torch.bfloat16),
            "base_model.model.lm_head.lora_B.weight": torch.zeros((vocab, 2), dtype=torch.bfloat16),
            "base_model.model.lm_head.lora_B.bias": bias,
        }, str(root / "adapter_model.safetensors"))
        adapters.append(f"{name}={root}")

    monkeypatch.setenv("SUROGATE_MIXED_TEST_ARTIFACT", artifact)
    monkeypatch.setenv("SUROGATE_MIXED_TEST_MODELS", json.dumps([names[0], "test", names[1]]))
    monkeypatch.setenv("SUROGATE_MIXED_TEST_ARGS", json.dumps([
        "--spec", "mtp", "--draft-tokens", "3", "--kv-cache-dtype", "bf16",
        "--enable-lora", "--lora-modules", ",".join(adapters), "--max-loras", "2",
        "--max-lora-rank", "2", "--max-num-seqs", str(active_slots),
        "--kv-capacity", str(2048 * active_slots),
    ]))
    running = mixed.server.__wrapped__(tmp_path_factory)
    try:
        server = next(running)
        if active_slots > 1:
            mixed.test_overlapping_prefill_preserves_decode_and_adapters(server, lambda *_: None)
        else:
            prompts = [mixed.body(names[0], 0, 8, 192)] + [
                mixed.body(name, index + 1, 170 + index * 7, 24)
                for index, name in enumerate([names[0], "test", names[1]])
            ]
            for prompt in prompts:
                result = mixed.ask(server, prompt)
                choice = result["choices"][0]
                mixed.check_scores(server, prompt, result["prompt_token_ids"],
                                   choice["token_ids"], choice["logprobs"]["content"])
    finally:
        running.close()
