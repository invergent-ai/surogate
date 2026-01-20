"""
DSL IR Builder

Builds IR JSON for models from either Python DSL classes or .module files.
Python DSL is preferred when available.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from surogate.dsl.registry import resolve_dsl_module_path, has_python_dsl_model


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


# =============================================================================
# Python DSL (preferred)
# =============================================================================


def build_dsl_ir_from_python(architecture: str, config_json: Dict[str, Any]) -> str:
    """Build IR JSON using Python DSL models."""
    # Import here to avoid circular imports and ensure models are registered
    from surogate.dsl.stdlib import models  # noqa: F401 - registers models
    from surogate.dsl.py_compiler import compile_model_for_hf

    return compile_model_for_hf(architecture, config_json)


# =============================================================================
# Lark DSL (fallback for .module files)
# =============================================================================


def parse_hf_param_mapping(module_path: Path, architecture: str) -> Tuple[Dict[str, str], Optional[str]]:
    from surogate.dsl.parser import ModuleDSLParser
    from surogate.dsl.ast_nodes import ModelNode

    parser = ModuleDSLParser()
    program = parser.parse(module_path.read_text(), str(module_path))

    fallback = None
    for decl in program.models:
        if decl.hf_config:
            if decl.hf_config.architecture == architecture:
                return decl.hf_config.param_mapping, decl.name
            model_type = decl.hf_config.extras.get("model_type") if decl.hf_config.extras else None
            if model_type and model_type == architecture:
                return decl.hf_config.param_mapping, decl.name
        fallback = decl
    if fallback and fallback.hf_config:
        return fallback.hf_config.param_mapping, fallback.name
    raise ValueError(f"No hf_config mapping found in DSL module: {module_path}")


def build_param_overrides(mapping: Dict[str, str], config_json: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for dsl_param, hf_key in mapping.items():
        if hf_key in config_json and config_json[hf_key] is not None:
            params[dsl_param] = config_json[hf_key]
    return params


def compile_dsl_ir(module_path: Path, params: Dict[str, Any], module_name: Optional[str]) -> str:
    from surogate.dsl.compiler import Compiler, CompilerOptions
    from surogate.cli.dsl_compile import module_ir_to_dict

    options = CompilerOptions()
    std_path = Path(__file__).resolve().parents[2] / "std"
    options.search_paths = [str(std_path), str(module_path.parent)]

    compiler = Compiler(options)
    result = compiler.compile_file(str(module_path), params)
    if not result.success:
        errors = "\n".join(str(e) for e in result.errors)
        raise RuntimeError(f"DSL compile failed:\n{errors}")

    modules = result.modules
    if module_name:
        modules = [m for m in modules if m.name == module_name]
        if not modules:
            raise RuntimeError(f"Module not found in DSL IR: {module_name}")

    ir_dict = {
        "source_file": str(module_path),
        "success": True,
        "modules": [module_ir_to_dict(m) for m in modules],
    }
    return json.dumps(ir_dict)


def build_dsl_ir_from_lark(architecture: str, config_json: Dict[str, Any]) -> str:
    """Build IR JSON using Lark-based .module files."""
    module_path = resolve_dsl_module_path(architecture)
    mapping, module_name = parse_hf_param_mapping(module_path, architecture)
    params = build_param_overrides(mapping, config_json)
    return compile_dsl_ir(module_path, params, module_name)


# =============================================================================
# Main Entry Point
# =============================================================================


def build_dsl_ir_for_model(model_dir: str, use_python_dsl: bool = True) -> str:
    """
    Build DSL IR JSON for a model.

    Args:
        model_dir: Path to the HuggingFace model directory
        use_python_dsl: If True, prefer Python DSL; if False, use Lark .module files

    Returns:
        JSON string with the compiled IR
    """
    config_json = load_hf_config(model_dir)
    architecture = resolve_architecture(config_json)

    # Try Python DSL first if requested
    if use_python_dsl:
        try:
            # Check if Python DSL model is available for this architecture
            if has_python_dsl_model(architecture):
                return build_dsl_ir_from_python(architecture, config_json)
        except Exception as e:
            # Fall back to Lark DSL if Python DSL fails
            import warnings
            warnings.warn(f"Python DSL failed for {architecture}, falling back to Lark: {e}")

    # Use Lark-based .module files
    return build_dsl_ir_from_lark(architecture, config_json)
