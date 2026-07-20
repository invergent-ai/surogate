"""Deterministic direct-model scoring for matched ordinary/hindsight branches."""

from __future__ import annotations

import inspect
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DeterministicScorerError(RuntimeError):
    """A matched score cannot be trusted for training."""


@dataclass(frozen=True)
class MatchedCompletionScores:
    """Exactly repeated scores for one fixed completion under two prompts."""

    reference_logprobs: tuple[float, ...]
    hindsight_logprobs: tuple[float, ...]
    reference_repeat_count: int
    hindsight_repeat_count: int

    @property
    def shifts(self) -> tuple[float, ...]:
        return tuple(
            hindsight - reference
            for reference, hindsight in zip(
                self.reference_logprobs,
                self.hindsight_logprobs,
                strict=True,
            )
        )


def configure_deterministic_scoring(*, seed: int = 0) -> None:
    """Configure PyTorch before loading a scorer model onto CUDA."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


class DeterministicAdapterScorer:
    """Score fixed tokens directly with one frozen adapter and deterministic SDPA."""

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cuda:0",
        adapter_name: str | None = "parent",
        direct_token_limit: int = 3072,
        prefill_chunk_tokens: int = 512,
        repeat_count: int = 2,
    ) -> None:
        if direct_token_limit < 2:
            raise ValueError("direct_token_limit must be at least 2")
        if prefill_chunk_tokens < 1:
            raise ValueError("prefill_chunk_tokens must be positive")
        if repeat_count < 2:
            raise ValueError("repeat_count must be at least 2")
        self.model = model
        self.device = device
        self.adapter_name = adapter_name
        self.direct_token_limit = direct_token_limit
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.repeat_count = repeat_count
        self.model.eval()

        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        parameters = inspect.signature(base_model.forward).parameters
        self._supports_logits_to_keep = "logits_to_keep" in parameters

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str | Path,
        adapter_path: str | Path,
        device: str = "cuda:0",
        adapter_name: str = "parent",
        **kwargs: Any,
    ) -> DeterministicAdapterScorer:
        """Load one frozen parent adapter on a deterministic attention path."""
        configure_deterministic_scoring()

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(
            base,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=False,
        )
        return cls(
            model,
            device=device,
            adapter_name=adapter_name,
            **kwargs,
        )

    @staticmethod
    def _validate_ids(token_ids: Sequence[int], *, label: str) -> list[int]:
        values = list(token_ids)
        if not values:
            raise DeterministicScorerError(f"{label} token sequence is empty")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise DeterministicScorerError(f"{label} contains a non-integer token ID")
        return values

    def _forward(self, *, input_ids: Any, **kwargs: Any) -> Any:
        if not self._supports_logits_to_keep:
            kwargs.pop("logits_to_keep", None)
        return self.model(input_ids=input_ids, **kwargs)

    def score_completion(
        self,
        *,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
    ) -> tuple[float, ...]:
        """Return one finite log-probability per submitted completion token."""
        prompt = self._validate_ids(prompt_ids, label="prompt")
        completion = self._validate_ids(completion_ids, label="completion")

        import torch

        if self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)
        targets = torch.tensor(completion, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            if len(prompt) + len(completion) <= self.direct_token_limit:
                token_ids = [*prompt, *completion]
                inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                positions = torch.arange(
                    len(prompt) - 1,
                    len(token_ids) - 1,
                    dtype=torch.long,
                    device=self.device,
                )
                output = self._forward(
                    input_ids=inputs,
                    use_cache=False,
                    logits_to_keep=positions,
                )
                logits = output.logits[0].float()
                if not self._supports_logits_to_keep:
                    logits = logits[positions]
                values = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1))[:, 0]
            else:
                values = self._score_long_completion(
                    prompt_ids=prompt,
                    targets=targets,
                )

        result = tuple(float(value) for value in values.cpu().tolist())
        if len(result) != len(completion) or any(not math.isfinite(value) or value > 1e-6 for value in result):
            raise DeterministicScorerError("direct scorer returned invalid completion-aligned logprobs")
        return result

    def _score_long_completion(self, *, prompt_ids: list[int], targets: Any) -> Any:
        import torch

        past_key_values = None
        next_logits = None
        for start in range(0, len(prompt_ids), self.prefill_chunk_tokens):
            chunk = torch.tensor(
                [prompt_ids[start : start + self.prefill_chunk_tokens]],
                dtype=torch.long,
                device=self.device,
            )
            output = self._forward(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values
            next_logits = output.logits[0, -1].float()
        if next_logits is None:
            raise DeterministicScorerError("cannot score after an empty prompt")

        token_values = []
        for index, target in enumerate(targets):
            token_values.append(torch.log_softmax(next_logits, dim=-1)[target])
            if index + 1 < len(targets):
                output = self._forward(
                    input_ids=target.reshape(1, 1),
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                past_key_values = output.past_key_values
                next_logits = output.logits[0, -1].float()
        del past_key_values
        return torch.stack(token_values)

    def score_matched(
        self,
        *,
        ordinary_prompt_ids: Sequence[int],
        hindsight_prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
    ) -> MatchedCompletionScores:
        """Score and exactly repeat both branches before returning any values."""
        ordinary = self._validate_ids(ordinary_prompt_ids, label="ordinary prompt")
        hindsight = self._validate_ids(hindsight_prompt_ids, label="hindsight prompt")
        completion = self._validate_ids(completion_ids, label="completion")
        if ordinary == hindsight:
            raise DeterministicScorerError("ordinary and hindsight prompt token sequences are identical")

        reference = self.score_repeatable_completion(
            prompt_ids=ordinary,
            completion_ids=completion,
            branch_label="ordinary branch",
        )
        hindsight_values = self.score_repeatable_completion(
            prompt_ids=hindsight,
            completion_ids=completion,
            branch_label="hindsight branch",
        )
        return MatchedCompletionScores(
            reference_logprobs=reference,
            hindsight_logprobs=hindsight_values,
            reference_repeat_count=self.repeat_count,
            hindsight_repeat_count=self.repeat_count,
        )

    def score_repeatable_completion(
        self,
        *,
        prompt_ids: Sequence[int],
        completion_ids: Sequence[int],
        branch_label: str = "completion branch",
    ) -> tuple[float, ...]:
        """Return a fixed-token score only after exact repeated evaluation."""
        runs = [
            self.score_completion(
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
            )
            for _ in range(self.repeat_count)
        ]
        if any(values != runs[0] for values in runs[1:]):
            raise DeterministicScorerError(f"{branch_label} is not exactly repeatable")
        return runs[0]
