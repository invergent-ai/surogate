# surogate/eval/targets/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class TargetType(Enum):
    """Types of evaluation targets."""
    LLM = "llm"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    CLIP = "clip"
    RAG = "rag"
    AGENT = "agent"
    CHATBOT = "chatbot"
    MCP = "mcp"
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
class TargetResponse:
    """Standardized response format from any target."""
    content: str
    raw_response: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class TargetRequest:
    """Standardized request format to any target."""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    inputs: Optional[Dict[str, Any]] = None
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


class BaseTarget(ABC):
    """Abstract base class for all evaluation targets."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize target.

        Args:
            config: Target configuration dict
        """
        self.config = config
        self.target_type = TargetType(config.get('type'))
        self.name = config.get('name', 'unnamed')
        self._validate_config()

    @abstractmethod
    def _validate_config(self):
        """Validate target-specific configuration."""
        pass

    @abstractmethod
    def send_request(self, request: TargetRequest) -> TargetResponse:
        """
        Send request to target and get response.

        Args:
            request: Standardized request

        Returns:
            Standardized response
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if target is healthy/accessible."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def cleanup(self):
        """Cleanup resources. Override if needed."""
        pass