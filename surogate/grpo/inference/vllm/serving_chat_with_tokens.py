"""Token-in chat completion endpoint.

`create_chat_completion_with_tokens` mirrors vLLM's own
`OpenAIServingChat._create_chat_completion`, with exactly one delta: the rendered
prompt's token ids are replaced by the caller-supplied `tokens` list. GRPO needs
this so the trainer can score the *exact* token sequence a rollout produced,
without a detokenize/retokenize round trip that would not be guaranteed to be
identity.

Because this tracks an upstream private method, it must be re-checked whenever
vLLM is upgraded. Ported to vLLM 0.25.1: the reasoning parser was replaced by a
unified `Parser` (tools + reasoning), `engine_prompts` became `engine_inputs`,
and both chat generators changed their trailing keyword arguments.
"""

from collections.abc import AsyncGenerator
from typing import ClassVar

from fastapi import Request
from pydantic import Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat, _get_mm_token_counts
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.parser.abstract_parser import Parser
from vllm.sampling_params import BeamSearchParams, SamplingParams

logger = init_logger(__name__)


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    field_names: ClassVar[set[str] | None] = None
    tokens: list[int] = Field(description=("Prompt tokens to use for the request."))


class OpenAIServingChatWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in."""

    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """Chat Completion API that takes prompt token ids directly.

        See https://platform.openai.com/docs/api-reference/chat/create for the
        base API specification.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion_with_tokens(request, raw_request), request, raw_request
        )

    async def _create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None

        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        parser: Parser | None = None
        try:
            if self.parser_cls is not None:
                parser = self.parser_cls(
                    tokenizer,
                    request.tools,
                    chat_template_kwargs=chat_template_kwargs,
                    model_config=self.model_config,
                )
        except RuntimeError as e:
            logger.exception("Error in parser creation.")
            return self.create_error_response(str(e))

        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        # THE DELTA: override the rendered prompt with the caller's exact tokens.
        # VLM conversations use MITO (message-based) instead of TITO, so
        # multi_modal_data is not expected here.
        engine_inputs[0]["prompt_token_ids"] = request.tokens  # type: ignore[index]

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        mm_token_counts: dict[str, int] | None = None
        try:
            for i, engine_input in enumerate(engine_inputs):
                prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
                mm_token_counts = _get_mm_token_counts(engine_input)

                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"

                # Kept from surogate's original: the token-override path is
                # exactly where an over-long caller-supplied sequence can arrive,
                # so check it explicitly rather than relying on the renderer.
                prompt_len = self._extract_prompt_len(engine_input)
                if prompt_len >= max_model_len:
                    raise VLLMValidationError(
                        f"This model's maximum context length is "
                        f"{max_model_len} tokens. However, your request has "
                        f"{prompt_len} input tokens. Please reduce the length of "
                        "the input messages.",
                        parameter="input_tokens",
                        value=prompt_len,
                    )

                max_tokens = get_max_tokens(
                    max_model_len,
                    request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
                    prompt_len,
                    self.default_sampling_params,
                    self.override_max_tokens,
                    truncate_prompt_tokens=request.truncate_prompt_tokens,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.default_sampling_params,
                    )

                self._log_inputs(
                    sub_request_id,
                    engine_input,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_input,
                        request_id=sub_request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                else:
                    has_reasoning = parser is not None and parser.reasoning_parser is not None
                    if not request.include_reasoning:
                        reasoning_ended = True
                    elif request._grammar_from_tool_parser:
                        # The Mistral grammar already includes an optional `think?`
                        # rule that handles both reasoning and non-reasoning outputs.
                        reasoning_ended = True
                    elif has_reasoning:
                        reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])
                    else:
                        reasoning_ended = None

                    generator = self.engine_client.generate(
                        engine_input,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        data_parallel_rank=data_parallel_rank,
                        reasoning_ended=reasoning_ended,
                        reasoning_parser_kwargs={"chat_template_kwargs": chat_template_kwargs}
                        if has_reasoning
                        else None,
                    )

                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )
