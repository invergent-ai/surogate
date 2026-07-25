"""Pin the vLLM API surface that surogate's inference server depends on.

surogate re-implements parts of vLLM's OpenAI server (a token-in chat endpoint,
custom routers, app-state wiring). Those depend on private-ish internals that
move between vLLM releases — the 0.25.1 upgrade renamed `openai_serving_render`
to `online_renderer`, moved `RequestLogger`, deleted `OpenAIServing`, and changed
the trailing kwargs of both chat generators. Each of those failed only when a
server was actually launched.

These tests turn every one of those into an import-time failure instead.
"""

import inspect

import pytest

vllm = pytest.importorskip("vllm")


class TestMovedSymbols:
    """Symbols surogate imports that have moved at least once."""

    def test_api_utils_helpers(self):
        from vllm.entrypoints.serve.utils.api_utils import (  # noqa: F401
            get_max_tokens,
            load_aware_call,
            validate_json_request,
            with_cancellation,
        )

    def test_request_logger(self):
        from vllm.entrypoints.serve.utils.request_logger import RequestLogger  # noqa: F401

    def test_engine_protocol_symbols(self):
        from vllm.entrypoints.openai.engine.protocol import (  # noqa: F401
            ErrorResponse,
            GenerationError,
            RequestResponseMetadata,
        )

    def test_serving_base_class(self):
        from vllm.entrypoints.generate.base.serving import BaseServing, GenerateBaseServing

        assert issubclass(GenerateBaseServing, BaseServing)
        assert hasattr(BaseServing, "create_error_response")

    def test_metrics_loggers_submodule(self):
        import vllm.v1.metrics.loggers as loggers

        assert hasattr(loggers, "PrometheusStatLogger")


class TestAppStateContract:
    """Attribute names surogate reads off `app.state`."""

    def test_init_app_state_sets_expected_names(self):
        from vllm.entrypoints.openai import api_server

        src = inspect.getsource(api_server.init_app_state)
        # Read by server.base() and by the re-class step.
        assert "state.serving_tokenization" in src
        assert "state.openai_serving_models" in src
        assert "state.online_renderer" in src

    def test_init_app_state_builds_the_chat_handler(self):
        """The re-class approach requires init_app_state to construct chat itself."""
        from vllm.entrypoints.openai import api_server

        src = inspect.getsource(api_server.init_app_state)
        assert "init_generate_state" in src, (
            "init_app_state no longer builds openai_serving_chat; "
            "custom_init_app_state can no longer re-class it"
        )

        from vllm.entrypoints.generate.api_router import init_generate_state

        assert "state.openai_serving_chat" in inspect.getsource(init_generate_state)

    def test_custom_init_app_state_signature_matches_upstream(self):
        from vllm.entrypoints.openai import api_server

        from surogate.grpo.inference.vllm.server import custom_init_app_state

        upstream = list(inspect.signature(api_server.init_app_state).parameters)
        ours = list(inspect.signature(custom_init_app_state).parameters)
        assert ours[: len(upstream)] == upstream


class TestReclassSafety:
    """`serving_chat.__class__ = OpenAIServingChatWithTokens` must stay valid."""

    def test_subclass_adds_no_state(self):
        from surogate.grpo.inference.vllm.serving_chat_with_tokens import OpenAIServingChatWithTokens

        assert "__init__" not in OpenAIServingChatWithTokens.__dict__, (
            "re-classing skips __init__; the subclass must not need one"
        )
        assert not getattr(OpenAIServingChatWithTokens, "__slots__", None)

    def test_direct_subclass_of_upstream_chat(self):
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        from surogate.grpo.inference.vllm.serving_chat_with_tokens import OpenAIServingChatWithTokens

        assert OpenAIServingChatWithTokens.__bases__ == (OpenAIServingChat,)


class TestTokenEndpointInternals:
    """Internals the token-in endpoint calls on OpenAIServingChat."""

    METHODS = [
        "render_chat_request",
        "_base_request_id",
        "_maybe_get_adapters",
        "_get_data_parallel_rank",
        "_extract_prompt_components",
        "_extract_prompt_len",
        "_log_inputs",
        "_get_trace_headers",
        "beam_search",
        "chat_completion_stream_generator",
        "chat_completion_full_generator",
        "create_error_response",
        "_effective_chat_template_kwargs",
        "_with_kv_transfer_rejection_cleanup",
    ]

    @pytest.mark.parametrize("name", METHODS)
    def test_method_exists(self, name):
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        assert hasattr(OpenAIServingChat, name)

    def test_module_level_helper(self):
        from vllm.entrypoints.openai.chat_completion.serving import _get_mm_token_counts  # noqa: F401

    def test_stream_generator_takes_chat_template_kwargs(self):
        """0.25.1 replaced the trailing reasoning-parser arg with a kwargs dict.

        Passing the old positional argument silently handed a parser object to a
        parameter expecting a dict, so this must be asserted by name.
        """
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        params = inspect.signature(OpenAIServingChat.chat_completion_stream_generator).parameters
        assert "chat_template_kwargs" in params
        assert "mm_token_counts" in params
        assert "reasoning_parser" not in params

    def test_full_generator_takes_parser(self):
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        params = inspect.signature(OpenAIServingChat.chat_completion_full_generator).parameters
        assert "parser" in params
        assert "mm_token_counts" in params

    def test_get_max_tokens_takes_truncate_prompt_tokens(self):
        from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

        assert "truncate_prompt_tokens" in inspect.signature(get_max_tokens).parameters

    def test_engine_generate_takes_reasoning_kwargs(self):
        from vllm.engine.protocol import EngineClient

        params = inspect.signature(EngineClient.generate).parameters
        assert "reasoning_ended" in params
        assert "reasoning_parser_kwargs" in params


class TestPatchedTargets:
    """Targets of surogate's monkeypatches must still exist."""

    def test_build_app_and_worker_proc(self):
        from vllm.entrypoints.openai import api_server
        from vllm.v1 import utils as v1_utils

        assert hasattr(api_server, "build_app")
        assert hasattr(api_server, "init_app_state")
        assert hasattr(v1_utils, "run_api_server_worker_proc")

    def test_build_app_signature(self):
        from vllm.entrypoints.openai import api_server

        from surogate.grpo.inference.vllm.server import custom_build_app

        upstream = list(inspect.signature(api_server.build_app).parameters)
        ours = list(inspect.signature(custom_build_app).parameters)
        assert ours == upstream
