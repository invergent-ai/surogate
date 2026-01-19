from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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
