"""
DSL IR Builder

Builds IR JSON for models from Python DSL classes.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_hf_config(model_dir: str) -> Dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in model dir: {model_dir}")
    return json.loads(config_path.read_text())


def resolve_architecture(config_json: Dict[str, Any]) -> str:
    archs = config_json.get("architectures", [])
    if archs:
        return archs[0]
    model_type = config_json.get("model_type")
    if model_type:
        return model_type
    raise ValueError("Could not resolve architecture from config.json")


def build_dsl_ir_from_python(
    architecture: str,
    config_json: Dict[str, Any],
    extra_config: Dict[str, Any] | None = None,
) -> str:
    """Build IR JSON using Python DSL models.

    Args:
        architecture: HuggingFace architecture name.
        config_json: The HuggingFace config.json contents.
        extra_config: Additional config overrides (e.g., ep_size from training config).

    Raises:
        RuntimeError: If DSL compilation fails, with detailed error message.
    """
    # Import here to avoid circular imports and ensure models are registered
    from surogate.dsl import models  # noqa: F401 - registers models
    from surogate.dsl.py_compiler import compile_model_for_hf

    ir_json = compile_model_for_hf(architecture, config_json, extra_config=extra_config)

    # Check if compilation succeeded
    result = json.loads(ir_json)
    if not result.get("success", False):
        errors = result.get("errors", [])
        if errors:
            # Format error messages for display
            error_msgs = []
            for err in errors:
                msg = err.get("message", str(err))
                if err.get("hint"):
                    msg += f"\n  Hint: {err['hint']}"
                error_msgs.append(msg)
            raise RuntimeError(
                f"DSL compilation failed for {architecture}:\n" +
                "\n".join(f"  - {msg}" for msg in error_msgs)
            )
        else:
            raise RuntimeError(f"DSL compilation failed for {architecture} (no error details)")

    return ir_json


def build_dsl_ir_for_model(
    model_dir: str,
    extra_config: Dict[str, Any] | None = None,
) -> str:
    """
    Build DSL IR JSON for a model.

    Args:
        model_dir: Path to the HuggingFace model directory
        extra_config: Additional config overrides (e.g., ep_size from training config)

    Returns:
        JSON string with the compiled IR
    """
    config_json = load_hf_config(model_dir)
    architecture = resolve_architecture(config_json)
    return build_dsl_ir_from_python(architecture, config_json, extra_config=extra_config)
