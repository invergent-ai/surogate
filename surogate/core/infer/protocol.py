import base64
import io
import os
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
from PIL import Image
from pydantic import BaseModel, Field, field_validator

@dataclass
class RequestConfig:
    """NOTE: The following behavior is inconsistent with the OpenAI API.
    Default values for OpenAI:
        temperature = 1.
        top_k = -1
        top_p = 1.
        repetition_penalty = 1.
    """
    max_tokens: Optional[int] = None  # None: max_model_len - num_tokens
    # None: use deploy_args
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    num_beams: int = 1
    stop: Optional[List[str]] = field(default_factory=list)

    seed: Optional[int] = None
    stream: bool = False
    logprobs: bool = False
    top_logprobs: Optional[int] = None

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    length_penalty: float = 1.
    # Return token_ids additionally (non-stream)
    return_details: bool = False

    def __post_init__(self):
        if self.stop is None:
            self.stop = []

@dataclass
class UsageInfo:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Function:
    name: str
    arguments: Optional[str]

    def __post_init__(self):
        if not isinstance(self.arguments, str):
            self.arguments = json.dumps(self.arguments)
        self.name = self.name.strip()
        self.arguments = self.arguments.strip()

@dataclass
class ChatCompletionMessageToolCall:
    function: Function
    type: str = 'function'
    id: str = field(default_factory=lambda: f'toolcall-{random_uuid()}')

@dataclass
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[Dict[str, Any]], int, float]
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    reasoning_content: Optional[str] = None


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None
    token_ids: Optional[List[int]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseChoice':
        self = deepcopy(self)
        assert not self.message.tool_calls, f'message: {self.message}'
        return CompletionResponseChoice(self.index, self.message.content, self.finish_reason, self.logprobs)

@dataclass
class CompletionResponseChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

@dataclass
class CompletionResponse:
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion'
    created: int = field(default_factory=lambda: int(time.time()))

@dataclass
class DeltaMessage:
    role: Literal['system', 'user', 'assistant', None] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    reasoning_content: Optional[str] = None


@dataclass
class ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseStreamChoice':
        self = deepcopy(self)
        assert not self.delta.tool_calls
        return CompletionResponseStreamChoice(self.index, self.delta.content, self.finish_reason, self.logprobs)


@dataclass
class CompletionResponseStreamChoice:
    index: int
    text: str
    finish_reason: Literal['stop', 'length', None]
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


@dataclass
class ChatCompletionStreamResponse:
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionStreamResponse':
        self = deepcopy(self)
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionStreamResponse(self.model, choices, self.usage, id_, created=self.created)

@dataclass
class CompletionStreamResponse:
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion.chunk'
    created: int = field(default_factory=lambda: int(time.time()))

@dataclass
class ChatCompletionResponse:
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    object: str = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))
    prompt_token_ids: Optional[List[int]] = None
    images_size: Optional[List[Tuple[int, int]]] = None

    def to_cmpl_response(self) -> 'CompletionResponse':
        self = deepcopy(self)
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        return CompletionResponse(self.model, choices, self.usage, id_, created=self.created)

def random_uuid() -> str:
    return str(uuid.uuid4().hex)