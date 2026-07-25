from typing import Any, Literal, Required, TypedDict, Union

import openai.types.chat
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    Audio,
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call_union_param import ChatCompletionMessageToolCallUnionParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam


def monkey_patch_oai_iterable_types():
    """
    This monkey patch is necessary to avoid Pydantic validating fields using
    typing.Iterable (e.g. in multimodal or tool call messages) lazily which
    leads to tokenization errors, for more info see
    https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
    """

    class ModdedChatCompletionDeveloperMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[Literal["developer"]]
        name: str

    class ModdedChatCompletionSystemMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[Literal["system"]]
        name: str

    class ModdedChatCompletionUserMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[str | list[ChatCompletionContentPartParam]]
        role: Required[Literal["user"]]
        name: str

    class ModdedChatCompletionAssistantMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        role: Required[Literal["assistant"]]
        audio: Audio | None
        content: str | list[ContentArrayOfContentPart] | None
        function_call: FunctionCall | None
        name: str
        refusal: str | None
        tool_calls: list[ChatCompletionMessageToolCallUnionParam]

    class ModdedChatCompletionToolMessageParam(TypedDict, total=False):
        """Same as openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam, but replacing typing.Iterable with list to not mess up Pydantic."""

        content: Required[str | list[ChatCompletionContentPartTextParam]]
        role: Required[Literal["tool"]]
        tool_call_id: Required[str]

    # Patch OAI types
    openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam = (
        ModdedChatCompletionDeveloperMessageParam
    )
    openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam = (
        ModdedChatCompletionSystemMessageParam
    )
    openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam = (
        ModdedChatCompletionUserMessageParam
    )
    openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam = (
        ModdedChatCompletionAssistantMessageParam
    )
    openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam = (
        ModdedChatCompletionToolMessageParam
    )

    openai.types.chat.chat_completion_message_param.ChatCompletionMessageParam = Union[
        ChatCompletionDeveloperMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
        ModdedChatCompletionAssistantMessageParam,
        ModdedChatCompletionToolMessageParam,
        ChatCompletionFunctionMessageParam,
    ]


def monkey_patch_chat_completion_logprobs():
    """
    At large batch sizes and context, constructing OAI's Pydantic model
    ChatCompletion with logprobs is causing heavy CPU overhead (~200ms per
    object at 32K context, which translates to >10min overhead at 4K batch
    size). This function monkey-patches the OAI type and verifiers'
    post-processing utils to avoid validating the complex logprobs field.
    """

    class ChoiceAny(Choice):
        """Same as openai.types.chat.chat_completion.Choice, but without type validation for logprobs field."""

        logprobs: Any | None = None

    class ModdedChatCompletion(ChatCompletion):
        """Same as openai.types.chat.chat_completion.ChatCompletion, but using ChoiceAny instead of Choice."""

        choices: list[ChoiceAny]  # type: ignore

    # Patch OAI types
    openai.types.chat.chat_completion.Choice = ChoiceAny
    openai.types.chat.chat_completion.ChatCompletion = ModdedChatCompletion


# Key under which the orchestrator injects a per-rollout depth cap into
# RolloutInput.info. It travels with the rollout, so this works identically for
# in-process environments and for environments running in a separate env server
# (no shared mutable state, no RPC to mutate env.max_turns).
ROLLOUT_DEPTH_CAP_KEY = "_surogate_max_turns"


def monkey_patch_multiturn_env_depth_cap():
    """Let MultiTurnEnv honor a per-rollout depth cap from `info`.

    Adaptive rollout-depth budgeting needs to shorten rollouts per step without
    rebuilding environments. Rather than mutate `env.max_turns` (which is not
    reachable when the env runs in a separate server process, and would race
    in-flight rollouts), the orchestrator writes the cap into each example's
    `info` and this patch enforces it.

    Patches the existing `max_turns_reached` stop condition rather than adding a
    new one: `Environment.__post_init__` snapshots *bound* stop-condition methods
    at construction time, so a method added to the class afterwards would never
    be registered. Patching the class before any env is constructed is picked up
    normally. The `@vf.stop` marker attributes are carried over so the replacement
    is still discovered as a stop condition.
    """
    import verifiers as vf

    original = vf.MultiTurnEnv.max_turns_reached
    if getattr(original, "_surogate_depth_cap", False):
        return

    async def max_turns_reached(self, state) -> bool:
        info = state.get("info")
        if isinstance(info, dict):
            cap = info.get(ROLLOUT_DEPTH_CAP_KEY)
            if isinstance(cap, int) and cap > 0 and len(state["trajectory"]) >= cap:
                return True
        return await original(self, state)

    for attr in ("stop", "stop_priority"):
        if hasattr(original, attr):
            setattr(max_turns_reached, attr, getattr(original, attr))
    max_turns_reached.__name__ = getattr(original, "__name__", "max_turns_reached")
    max_turns_reached.__doc__ = original.__doc__
    max_turns_reached._surogate_depth_cap = True

    vf.MultiTurnEnv.max_turns_reached = max_turns_reached
