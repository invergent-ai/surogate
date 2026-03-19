"""End-to-end test for native generation engine.

Validates that trainer.generate() produces valid tokens, is deterministic
in greedy mode, and that the arena is properly restored for training afterward.

Requirements:
    - GPU with enough VRAM for Qwen3-0.6B
    - HF weights cached or QWEN3_MODEL_PATH set

Usage:
    pytest tests/test_native_generation.py -v --no-header
"""

from __future__ import annotations

import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    import surogate._surogate as _surogate
except ImportError:
    pytest.skip("surogate._surogate C++ extension not built", allow_module_level=True)

from surogate.dsl.ir_builder import build_dsl_ir_for_model
from surogate.utils.hf import get_model_weights_path

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_ID = "Qwen/Qwen3-0.6B"
ENV_VAR = "QWEN3_MODEL_PATH"


def get_model_path() -> str:
    if path := os.environ.get(ENV_VAR):
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID)


@pytest.fixture(scope="module")
def trainer():
    """Create a trainer with Qwen3-0.6B for generation tests."""
    model_path = get_model_path()
    ir_json = build_dsl_ir_for_model(model_path)

    config = _surogate.PretrainedConfig.from_pretrained(model_path, "bf16")
    options = _surogate.RuntimeOptions()
    options.dsl_ir_json = ir_json

    t = _surogate.SurogateTrainer(
        ngpu=1,
        config=config,
        options=options,
        batch_size=1,
        seq_len=128,
        grad_accum=1,
        memcpy_all_gather=True,
        memcpy_send_recv=True,
    )

    weights_path = get_model_weights_path(model_path)
    t.import_weights(weights_path)
    return t


@pytest.fixture(scope="module")
def tokenizer():
    model_path = get_model_path()
    return transformers.AutoTokenizer.from_pretrained(model_path)


class TestNativeGeneration:
    """Test native generation via trainer.generate()."""

    def test_generate_produces_tokens(self, trainer, tokenizer):
        """Basic smoke test: generate() returns non-empty token sequences."""
        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt)

        tokens, logprobs, prompt_lens, completion_lens = trainer.generate(
            prompts=[input_ids],
            num_completions=1,
            max_gen_len=16,
            temperature=0.0,  # greedy
            eos_token_id=tokenizer.eos_token_id or 151643,
            use_lora=False,
        )

        assert len(tokens) == 1, "Expected 1 trajectory"
        assert len(tokens[0]) > len(input_ids), "Expected completion tokens beyond prompt"
        assert prompt_lens[0] == len(input_ids)
        assert completion_lens[0] > 0

        # Decode and verify it's not garbage
        full_text = tokenizer.decode(tokens[0])
        print(f"Generated: {full_text}")
        assert len(full_text) > len(prompt)

    def test_greedy_is_deterministic(self, trainer, tokenizer):
        """Greedy decoding (temperature=0) should produce identical output twice."""
        prompt = "Once upon a time"
        input_ids = tokenizer.encode(prompt)

        results = []
        for _ in range(2):
            tokens, logprobs, _, _ = trainer.generate(
                prompts=[input_ids],
                num_completions=1,
                max_gen_len=8,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id or 151643,
                use_lora=False,
            )
            results.append(tokens[0])

        assert results[0] == results[1], (
            f"Greedy decode not deterministic:\n"
            f"  Run 1: {results[0]}\n"
            f"  Run 2: {results[1]}"
        )

    def test_arena_restored_after_generation(self, trainer, tokenizer):
        """Verify training still works after generation (arena properly restored)."""
        prompt = "Hello world"
        input_ids = tokenizer.encode(prompt)

        # Run generation (uses arena for KV-cache)
        trainer.generate(
            prompts=[input_ids],
            num_completions=1,
            max_gen_len=4,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id or 151643,
            use_lora=False,
        )

        # Now run a training forward — should work if arena was restored
        seq_len = trainer.seq_length
        B = 1
        input_np = np.zeros((B, seq_len), dtype=np.int32)
        target_np = np.full((B, seq_len), -100, dtype=np.int32)

        # Fill with some tokens
        for i, tok in enumerate(input_ids[:seq_len]):
            input_np[0, i] = tok
        for i in range(min(len(input_ids) - 1, seq_len - 1)):
            target_np[0, i] = input_ids[i + 1]

        trainer.set_grad_accumulation(1)
        logprobs = trainer.forward_for_grpo(input_np, target_np)
        assert logprobs is not None
        assert len(logprobs.flatten()) == B * seq_len

    def test_multi_completion(self, trainer, tokenizer):
        """Test M prompts × N completions generation."""
        prompts = [
            tokenizer.encode("The sky is"),
            tokenizer.encode("Water boils at"),
        ]

        tokens, logprobs, prompt_lens, completion_lens = trainer.generate(
            prompts=prompts,
            num_completions=2,
            max_gen_len=8,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id or 151643,
            use_lora=False,
        )

        # M=2 prompts × N=2 completions = 4 sequences
        assert len(tokens) == 4
        assert len(logprobs) == 4
        assert prompt_lens[0] == len(prompts[0])
        assert prompt_lens[1] == len(prompts[0])  # same prompt group
        assert prompt_lens[2] == len(prompts[1])
        assert prompt_lens[3] == len(prompts[1])

        # Each completion should have generated tokens
        for i in range(4):
            assert completion_lens[i] > 0, f"Sequence {i} has no completion tokens"
            assert len(logprobs[i]) > 0, f"Sequence {i} has no logprobs"
