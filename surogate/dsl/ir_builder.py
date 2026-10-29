"""
DSL IR Builder

Builds IR JSON for models from Python DSL classes.
"""

import json
from pathlib import Path
from typing import Any


def load_hf_config(model_dir: str) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in model dir: {model_dir}")
    return json.loads(config_path.read_text())


def resolve_architecture(config_json: dict[str, Any]) -> str:
    archs = config_json.get("architectures", [])
    if archs:
        return archs[0]
    model_type = config_json.get("model_type")
    if model_type:
        return model_type
    raise ValueError("Could not resolve architecture from config.json")


def build_dsl_ir_from_python(
    architecture: str,
    config_json: dict[str, Any],
    extra_config: dict[str, Any] | None = None,
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
                f"DSL compilation failed for {architecture}:\n" + "\n".join(f"  - {msg}" for msg in error_msgs)
            )
        else:
            raise RuntimeError(f"DSL compilation failed for {architecture} (no error details)")

    # EP sanity check: when ep_size > 1, the compiled graph must contain
    # ep_dispatch ops. Otherwise runtime will initialize EP comms but execute
    # a non-EP MoE path, which is unstable (expert gradients are not reduced
    # across EP ranks in that mode).
    ep_size = 1
    if extra_config:
        try:
            ep_size = int(extra_config.get("ep_size", 1) or 1)
        except Exception:
            ep_size = 1
    if ep_size > 1:
        has_ep_dispatch = False
        for module in result.get("modules", []):
            ops = module.get("forward", {}).get("operations", [])
            if any(op.get("kernel_type") == "ep_dispatch" for op in ops):
                has_ep_dispatch = True
                break
        if not has_ep_dispatch:
            raise RuntimeError(
                f"DSL compilation for {architecture} with ep_size={ep_size} "
                "produced no ep_dispatch ops. EP is not wired in this graph."
            )

    # Keep checkpoint-specific fields (for example nested rotary settings and
    # activation names) available when exporting a fully fine-tuned model.
    for module in result.get("modules", []):
        module.setdefault("hf_config", {})["source_config"] = config_json
    return json.dumps(result)


def build_dsl_ir_for_model(
    model_dir: str,
    extra_config: dict[str, Any] | None = None,
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
    ir_json = build_dsl_ir_from_python(architecture, config_json, extra_config=extra_config)
    names = _checkpoint_tensor_names(model_dir)
    if names is None:
        return ir_json
    result = json.loads(ir_json)
    if not _stack_per_expert_checkpoint(result, names):
        return ir_json
    return json.dumps(result)


def _checkpoint_tensor_names(model_dir: str) -> set[str] | None:
    """Tensor names of a local safetensors checkpoint (sharded index or single file); None if there is none."""
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        return set(json.loads(index.read_text()).get("weight_map", {}))
    single = root / "model.safetensors"
    if single.is_file():
        with single.open("rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            return {k for k in json.loads(f.read(header_len)) if k != "__metadata__"}
    return None


#: Fused routed-expert tensors (the layout of HF's own Qwen3.5 / Qwen3.6 MoE checkpoints) and the per-expert tensors
#: an exporter may write instead: (fused suffix, per-expert pattern suffix, fuse_gate_up).
_PER_EXPERT_LAYOUTS = (
    ("experts.gate_up_proj", "experts.{expert}.gate_proj.weight", True),
    ("experts.down_proj", "experts.{expert}.down_proj.weight", False),
)


def _stack_per_expert_checkpoint(ir: dict[str, Any], names: set[str]) -> bool:
    """Map fused routed experts to per-expert tensors when the checkpoint stores them one by one.

    The DSL maps Qwen3.5 / Qwen3.6 MoE experts as one fused tensor per layer (experts.gate_up_proj, gate first, and
    experts.down_proj). An llm-compressor export such as our NVFP4 release writes one tensor per expert and projection
    (experts.{e}.gate_proj.weight_packed, ...), so those mappings become stack_experts, gate and up fused in the
    runtime's [up | gate] order. Returns whether anything changed.
    """
    changed = False
    for module in ir.get("modules", []):
        mapping = module.get("hf_mapping") or {}
        for key, spec in list(mapping.items()):
            if isinstance(spec, str):
                source = spec
            elif isinstance(spec, dict) and spec.get("type") == "direct":
                source = spec.get("source") or ""
            else:
                continue
            for fused, per_expert, fuse_gate_up in _PER_EXPERT_LAYOUTS:
                if not source.endswith(fused):
                    continue
                if source.replace("{layer}", "0") in names:
                    break
                pattern = source[: -len(fused)] + per_expert
                probe = pattern.replace("{layer}", "0").replace("{expert}", "0")
                if probe not in names and f"{probe}_packed" not in names:
                    break
                mapping[key] = {"type": "stack_experts", "pattern": pattern}
                if fuse_gate_up:
                    mapping[key]["fuse_gate_up"] = True
                changed = True
                break
    return changed
