# surogate/eval/metrics/__init__.py
"""
Metrics module for Surogate evaluation.

Provides various metrics for evaluating LLM outputs:
- G-Eval family (basic, conversational, multimodal, arena, DAG)
- Multi-turn conversation metrics
- Safety metrics (toxicity, bias, harm)
- Performance metrics (latency, throughput, token generation speed)
- Non-LLM metrics (embeddings, classification)
"""

from .base import (
    BaseMetric,
    LLMJudgeMetric,
    MetricResult,
    BatchMetricResult,
    MetricType
)
from .registry import MetricRegistry, register_metric

# Import all metric implementations to trigger registration
from .g_eval import (
    GEvalMetric,
    ConversationalGEvalMetric,
    MultimodalGEvalMetric,
    ArenaGEvalMetric
)
from .dag_eval import DAGMetric, ConversationalDAGMetric
from .conversation import (
    ConversationCoherenceMetric,
    ContextRetentionMetric,
    TurnAnalysisMetric
)
from .safety import (
    ToxicityMetric,
    BiasMetric,
    HarmMetric
)
from .performance import (
    LatencyMetric,
    ThroughputMetric,
    TokenGenerationSpeedMetric,
    PerformanceAggregator
)
from .embeddings import (
    EmbeddingSimilarityMetric,
    ClassificationMetric
)

__all__ = [
    # Base classes
    'BaseMetric',
    'LLMJudgeMetric',
    'MetricResult',
    'BatchMetricResult',
    'MetricType',
    'MetricRegistry',
    'register_metric',

    # G-Eval family
    'GEvalMetric',
    'ConversationalGEvalMetric',
    'MultimodalGEvalMetric',
    'ArenaGEvalMetric',
    'DAGMetric',
    'ConversationalDAGMetric',

    # Conversation metrics
    'ConversationCoherenceMetric',
    'ContextRetentionMetric',
    'TurnAnalysisMetric',

    # Safety metrics
    'ToxicityMetric',
    'BiasMetric',
    'HarmMetric',

    # Performance metrics
    'LatencyMetric',
    'ThroughputMetric',
    'TokenGenerationSpeedMetric',
    'PerformanceAggregator',

    # Non-LLM metrics
    'EmbeddingSimilarityMetric',
    'ClassificationMetric',
]