"""Verifiers token bridge with checkpoint-compatible tool history.

Verifiers 0.3.0 renders an assistant-only bridge using the placeholder function
name ``f``. Some templates require an initial user; others put the function's
name in the tool-result role. Preserve that name without retokenizing any model
output. ContextVar keeps concurrent rollouts on the same client independent.
"""

from contextvars import ContextVar
from copy import deepcopy

from verifiers.clients.openai_chat_completions_token_client import OpenAIChatCompletionsTokenClient

_bridge_functions = ContextVar("surogate_bridge_functions", default=None)


class ToolChatCompletionsTokenClient(OpenAIChatCompletionsTokenClient):
    async def get_prompt_ids(self, state, prompt_messages, oai_tools, chat_template_kwargs=None):
        functions = {
            call["id"]: call["function"]["name"]
            for message in prompt_messages
            for call in message.get("tool_calls", []) or []
        }
        token = _bridge_functions.set(functions)
        try:
            return await super().get_prompt_ids(state, prompt_messages, oai_tools, chat_template_kwargs)
        finally:
            _bridge_functions.reset(token)

    async def tokenize(self, messages, tools, model, extra_kwargs=None, **kwargs):
        functions = _bridge_functions.get()
        if functions is not None and isinstance(messages, list) and messages and messages[0].get("role") == "assistant":
            messages = deepcopy(messages)
            if messages[0].get("tool_calls"):
                # Qwen3 renders empty reasoning differently for a final versus
                # historical assistant. A nonempty dummy span stabilizes both
                # templates; these dummy tokens are never sent to generation.
                messages[0]["reasoning_content"] = "."
            for call in messages[0].get("tool_calls", []) or []:
                if call["id"] in functions:
                    call["function"]["name"] = functions[call["id"]]
            if messages[0].get("tool_calls"):
                messages.insert(0, dict(role="user", content="."))
        return await super().tokenize(messages, tools, model, extra_kwargs=extra_kwargs, **kwargs)


def install_tool_token_client():
    """Install in the orchestrator and each spawned environment worker."""
    import verifiers.clients

    verifiers.clients.OpenAIChatCompletionsTokenClient = ToolChatCompletionsTokenClient
