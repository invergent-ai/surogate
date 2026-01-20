"""
DSL Model Registry

Maps HuggingFace architectures to Python DSL model implementations.
"""

from typing import Optional

# Maps HF architecture -> Python DSL model class name
PYTHON_DSL_MODEL_REGISTRY = {
    "Qwen3ForCausalLM": "Qwen3Model",
    "qwen3": "Qwen3Model",
    "LlamaForCausalLM": "LlamaModel",
    "llama": "LlamaModel",
}


def has_python_dsl_model(architecture: str) -> bool:
    """Check if a Python DSL model is available for the architecture."""
    if architecture in PYTHON_DSL_MODEL_REGISTRY:
        # Also verify the model is actually registered
        try:
            from surogate.dsl.decorators import _model_registry
            model_name = PYTHON_DSL_MODEL_REGISTRY[architecture]
            # Import lib models to ensure they're registered
            from surogate.dsl.lib import models  # noqa: F401
            return model_name in _model_registry
        except ImportError:
            return False
    return False


def get_python_dsl_model_name(architecture: str) -> Optional[str]:
    """Get the Python DSL model class name for an architecture."""
    return PYTHON_DSL_MODEL_REGISTRY.get(architecture)
