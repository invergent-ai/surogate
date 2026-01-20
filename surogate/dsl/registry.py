"""
DSL Model Registry

Maps HuggingFace architectures to DSL implementations (Python DSL or .module files).
"""

from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# =============================================================================
# Lark DSL Registry (.module files)
# =============================================================================

DSL_MODEL_REGISTRY = {
    # HF architecture -> DSL module path (relative to repo root)
    "Qwen3ForCausalLM": "std/models/qwen3.module"
}


def resolve_dsl_module_path(architecture: str) -> Path:
    if architecture not in DSL_MODEL_REGISTRY:
        raise ValueError(f"Unsupported architecture for DSL IR: {architecture}")
    path = REPO_ROOT / DSL_MODEL_REGISTRY[architecture]
    if not path.exists():
        raise FileNotFoundError(f"DSL module not found: {path}")
    return path


# =============================================================================
# Python DSL Registry
# =============================================================================

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
            # Import stdlib models to ensure they're registered
            from surogate.dsl.stdlib import models  # noqa: F401
            return model_name in _model_registry
        except ImportError:
            return False
    return False


def get_python_dsl_model_name(architecture: str) -> Optional[str]:
    """Get the Python DSL model class name for an architecture."""
    return PYTHON_DSL_MODEL_REGISTRY.get(architecture)
