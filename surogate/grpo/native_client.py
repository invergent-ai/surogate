"""Native generation client for the verifiers library.

Implements the verifiers `Client` protocol using the C++ engine's built-in
`trainer.generate()`, enabling multi-turn rollouts without vLLM.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from verifiers.clients.client import Client
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    Tool,
    Usage,
)

from surogate.utils.logger import get_logger

logger = get_logger()


class NativeClient(Client[None, list[dict], dict, None]):
    """Verifiers-compatible client that generates using the native C++ engine.

    Wraps `trainer.generate()` behind the `Client.get_response()` interface,
    enabling multi-turn rollouts in verifiers environments without vLLM.
    """

    def __init__(
        self,
        trainer: Any,
        tokenizer: Any,
        max_gen_len: int = 512,
        use_lora: bool = True,
    ):
        # Skip parent __init__ since we don't use ClientConfig
        self._client = None
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.max_gen_len = max_gen_len
        self.use_lora = use_lora
        self.eos_id = tokenizer.eos_token_id or 151643
        self.default_temperature = 1.0  # Updated by trainer with scheduled temperature
        self.logger = logger

    def setup_client(self, config: ClientConfig) -> None:
        return None

    async def to_native_tool(self, tool: Tool) -> None:
        # Tool calling not supported in native mode
        return None

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict], dict]:
        """Convert vf.Messages to a list of dicts for tokenizer."""
        native_msgs = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                d = msg.model_dump(exclude_none=True)
            elif isinstance(msg, dict):
                d = msg
            else:
                d = {"role": getattr(msg, "role", "user"), "content": str(msg)}
            # Flatten content if it's a list of text parts
            content = d.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                if text_parts:
                    d["content"] = "\n".join(text_parts)
            native_msgs.append(d)
        return native_msgs, {}

    async def get_native_response(
        self,
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[None] | None = None,
        **kwargs,
    ) -> dict:
        """Generate a response using the native C++ engine."""
        # Apply chat template to get token IDs
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: concatenate message contents
            prompt_text = "\n".join(
                m.get("content", "") for m in prompt if isinstance(m, dict)
            )

        prompt_ids = self.tokenizer.encode(prompt_text)

        # Extract sampling parameters (fall back to scheduled default_temperature)
        temperature = sampling_args.get("temperature", self.default_temperature) if sampling_args else self.default_temperature
        top_p = sampling_args.get("top_p", 1.0) if sampling_args else 1.0
        top_k = sampling_args.get("top_k", 0) if sampling_args else 0
        max_tokens = sampling_args.get("max_tokens", self.max_gen_len) if sampling_args else self.max_gen_len

        # Generate
        tokens, logprobs, prompt_lens, completion_lens = self.trainer.generate(
            prompts=[prompt_ids],
            num_completions=1,
            max_gen_len=max_tokens or self.max_gen_len,
            temperature=max(temperature, 1e-6),  # Avoid exact 0 (use small temp for ~greedy)
            eos_token_id=self.eos_id,
            use_lora=self.use_lora,
            top_k=top_k,
            top_p=top_p,
        )

        pl = prompt_lens[0]
        cl = completion_lens[0]
        completion_ids = list(tokens[0][pl : pl + cl])
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Build completion logprobs
        comp_logprobs = list(logprobs[0][:cl]) if len(logprobs[0]) >= cl else list(logprobs[0])

        # Determine finish reason
        if cl > 0 and completion_ids[-1] == self.eos_id:
            finish_reason = "stop"
        elif cl >= (max_tokens or self.max_gen_len):
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return {
            "completion_text": completion_text,
            "completion_ids": completion_ids,
            "completion_logprobs": comp_logprobs,
            "prompt_ids": list(prompt_ids),
            "prompt_len": pl,
            "completion_len": cl,
            "finish_reason": finish_reason,
            "temperature": temperature,
        }

    async def raise_from_native_response(self, response: dict) -> None:
        # No error conditions to check for native generation
        pass

    async def from_native_response(self, response: dict) -> Response:
        """Convert native generation output to a vf.Response."""
        completion_ids = response["completion_ids"]
        prompt_ids = response["prompt_ids"]
        pl = response["prompt_len"]
        cl = response["completion_len"]

        # Build ResponseTokens for the verifiers trajectory
        response_tokens = ResponseTokens(
            prompt_ids=prompt_ids,
            prompt_mask=[1] * len(prompt_ids),
            completion_ids=completion_ids,
            completion_mask=[1] * len(completion_ids),
            completion_logprobs=response["completion_logprobs"],
        )

        message = ResponseMessage(
            role="assistant",
            content=response["completion_text"],
            finish_reason=response["finish_reason"],
            is_truncated=(response["finish_reason"] == "length"),
            tokens=response_tokens,
        )

        usage = Usage(
            prompt_tokens=pl,
            reasoning_tokens=0,
            completion_tokens=cl,
            total_tokens=pl + cl,
        )

        return Response(
            id=f"native-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model="native",
            usage=usage,
            message=message,
        )

    async def close(self) -> None:
        pass
