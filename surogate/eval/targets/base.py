# surogate/eval/targets/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TargetType(Enum):
    """Types of evaluation targets."""
    LLM = "llm"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    CLIP = "clip"
    CUSTOM = "custom"


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    OLLAMA = "ollama"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class TargetRequest:
    """Standardized request format to any target."""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.prompt:
            result['prompt'] = self.prompt
        if self.messages:
            result['messages'] = self.messages
        if self.inputs:
            result['inputs'] = self.inputs
        if self.parameters:
            result['parameters'] = self.parameters
        return result


@dataclass
class TargetResponse:
    """Response from a target."""
    content: str
    raw_response: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class BaseTarget(ABC):
    """Abstract base class for all targets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize target.

        Args:
            config: Target configuration
        """
        self.config = config
        self.name = config.get('name', 'unnamed')
        self.target_type = TargetType(config.get('type'))
        self._validate_config()

    def _validate_config(self):
        """Validate target configuration. Override in subclasses."""
        pass

    @abstractmethod
    def send_request(self, request: TargetRequest) -> TargetResponse:
        """
        Send request to target.

        Args:
            request: Request to send

        Returns:
            Response from target
        """
        raise NotImplementedError("Subclasses must implement send_request")

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if target is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        raise NotImplementedError("Subclasses must implement health_check")

    def cleanup(self):
        """Cleanup any resources used by target."""
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, type={self.target_type.value})"