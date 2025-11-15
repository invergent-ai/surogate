# surogate/eval/targets/__init__.py
from .base import BaseTarget, TargetType, ModelProvider, TargetRequest, TargetResponse
from .model import APIModelTarget, LocalModelTarget, EmbeddingTarget, RerankerTarget, CLIPTarget
from .factory import TargetFactory

__all__ = [
    'BaseTarget',
    'TargetType',
    'ModelProvider',
    'TargetRequest',
    'TargetResponse',
    'APIModelTarget',
    'LocalModelTarget',
    'EmbeddingTarget',
    'RerankerTarget',
    'CLIPTarget',
    'TargetFactory'
]