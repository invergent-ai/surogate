"""
Python DSL Compiler

Compiles Python DSL model/block/module classes to IR JSON format compatible
with the C++ runtime.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .decorators import _model_registry, _block_registry, _module_registry, _primitive_registry
from .specs import (
    ModelSpec,
    BlockSpec,
    ModuleSpec,
    ParamSpec,
    ParamKind,
    ForwardSpec,
    HFConfigSpec,
    HFMappingSpec,
    HFTransformSpec,
    ActivationSlotSpec,
    ActivationLayoutSpec,
    ActivationScope,
    ActivationMemoryHint,
)
from .graph_builder import GraphBuilder, GraphNode, GraphRef
from .hf import FuseMapping, SplitMapping, TransformMapping, StackExpertsMapping, TiedToMapping
from .dim import Dim, DimExpr, ConcreteDimValue, dim_to_ir
from .errors import (
    DSLError,
    DSLSyntaxError,
    DSLTypeError,
    DSLShapeError,
    DSLUndefinedError,
    ErrorCode,
    WarningCode,
    WarningCollector,
)

if TYPE_CHECKING:
    from .tensor_type import TensorAnnotation


# =============================================================================
# IR Dataclasses
# =============================================================================


@dataclass
class TensorRef:
    """Tensor reference in the IR."""
    shape: List[Any]
    dtype: Optional[str] = None
    is_param: bool = False
    is_input: bool = False
    is_output: bool = False
    quantizable: bool = True
    offload_group: int | str = -1


@dataclass
class OpIR:
    """Operation in the IR graph."""
    id: Optional[int] = None
    name: Optional[str] = None
    kernel_type: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphIR:
    """Computation graph IR."""
    name: Optional[str] = None
    inputs: Dict[str, TensorRef] = field(default_factory=dict)
    outputs: Dict[str, TensorRef] = field(default_factory=dict)
    params: Dict[str, TensorRef] = field(default_factory=dict)
    intermediates: Dict[str, TensorRef] = field(default_factory=dict)
    nodes: List[OpIR] = field(default_factory=list)
    save_list: List[str] = field(default_factory=list)
    recompute_list: List[str] = field(default_factory=list)


@dataclass
class ActivationSlotIR:
    """Activation slot in the IR.

    This represents a pre-allocated tensor buffer used during forward/backward.
    The C++ runtime uses this to:
    - Generate TensorSlot enum entries
    - Build shape inference tables
    - Create save/restore mappings for backward pass
    """
    name: str
    scope: str  # "block", "global", "gradient", "global_gradient"
    shape: List[Any]  # Shape expression with symbolic dims
    dtype: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    memory_hint: str = "persistent"  # "persistent", "save", "recompute", "temporary", "shared"
    shares_with: Optional[str] = None
    save_for_backward: bool = False
    recompute_in_backward: bool = False
    recompute_from: List[str] = field(default_factory=list)
    recompute_op: Optional[str] = None
    recompute_attrs: Dict[str, Any] = field(default_factory=dict)
    recompute_policy: str = "always"
    recompute_group: Optional[str] = None
    recompute_outputs: List[str] = field(default_factory=list)
    lora_targets: List[str] = field(default_factory=list)
    share_policy: str = "when_recomputed"  # "per_layer", "when_recomputed", "always_share", "fft_share", "lora_share"
    gradient_of: Optional[str] = None
    alias_of: Optional[str] = None
    slot_index: int = -1  # Index in the activation struct
    condition: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ActivationLayoutIR:
    """Complete activation layout in the IR.

    This aggregates all activation slots for a block or model and provides:
    - Ordered list of slots for C++ struct generation
    - Name → slot index mapping
    - Alias resolution table
    """
    name: str
    slots: List[ActivationSlotIR] = field(default_factory=list)
    gradient_slots: List[ActivationSlotIR] = field(default_factory=list)
    alias_map: Dict[str, str] = field(default_factory=dict)  # alias -> canonical name
    extends: Optional[str] = None


@dataclass
class ModuleIR:
    """Module IR output."""
    name: str
    kind: str  # "model", "block", "module"
    extends: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    # hf_config is the nested structure: {architecture, param_mapping, model_type}
    hf_config: Dict[str, Any] = field(default_factory=dict)
    hf_weight_mapping: Dict[str, Any] = field(default_factory=dict)
    hf_export_mapping: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, TensorRef] = field(default_factory=dict)
    forward_graph: Optional[GraphIR] = None
    backward_graph: Optional[GraphIR] = None
    save_tensors: List[str] = field(default_factory=list)
    recompute_tensors: List[str] = field(default_factory=list)
    is_model: bool = False
    is_block: bool = False
    # Activation layout for this module (block activations or global model activations)
    activation_layout: Optional[ActivationLayoutIR] = None


# =============================================================================
# Compiler Implementation
# =============================================================================


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _dsl_error_to_dict(err: DSLError) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": err.code.value, "message": err.message}
    if err.location is not None:
        payload["location"] = str(err.location)
    if err.hint:
        payload["hint"] = err.hint
    return payload


def _dsl_warning_to_dict(w) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": w.code.value, "message": w.message}
    if w.location is not None:
        payload["location"] = str(w.location)
    return payload


def _warn(warnings: WarningCollector | None, code: WarningCode, message: str) -> None:
    if warnings is None:
        return
    warnings.warn(code, message)


def _parse_shape_dim(dim: Any) -> Any:
    """Convert a shape dimension to IR format."""
    if isinstance(dim, int):
        return dim
    if isinstance(dim, ConcreteDimValue):
        return dim.value
    if isinstance(dim, (Dim, DimExpr)):
        return dim.to_expr_string()
    return str(dim)


def _build_dim_map(instance: Any) -> Dict[str, str]:
    """Build a mapping from attribute names to their Dim expression strings.

    This maps annotation strings like "C", "D", "QKV" to config parameter expressions
    like "d_model", "head_size", "(num_query_heads + 2 * num_kv_heads) * head_size".

    Also handles integer attributes used as dimension values in block definitions
    (e.g., self.P = 1552 for projection_size).

    Args:
        instance: An instance of a block/module class with Dim attributes.

    Returns:
        Dict mapping attribute name to its expression string.
    """
    dim_map: Dict[str, str] = {}
    for attr_name in dir(instance):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(instance, attr_name)
            if isinstance(attr, (Dim, DimExpr, ConcreteDimValue)):
                dim_map[attr_name] = _parse_shape_dim(attr)
            elif isinstance(attr, int) and attr >= 0:
                # Also capture integer dimension values (e.g., self.P = projection_size)
                # Use short uppercase names as heuristic for dimension aliases
                if len(attr_name) <= 10 and attr_name[0].isupper():
                    dim_map[attr_name] = str(attr)
        except Exception:
            pass
    return dim_map


def _tensor_annotation_to_ref(
    ann: TensorAnnotation,
    is_param: bool = False,
    is_input: bool = False,
    is_output: bool = False,
) -> TensorRef:
    """Convert a TensorAnnotation to a TensorRef for IR."""
    shape = [_parse_shape_dim(d) for d in ann.dims] if ann.dims else []
    return TensorRef(
        shape=shape,
        dtype=ann.dtype,
        is_param=is_param,
        is_input=is_input,
        is_output=is_output,
    )


def _substitute_dim_names(expr: str, dim_map: Dict[str, str]) -> str:
    """Substitute dimension attribute names in an expression with their config names.

    For example, "D // 2" with dim_map {"D": "head_size"} becomes "head_size // 2".
    """
    import re
    # Sort dim_map keys by length (longest first) to avoid partial substitutions
    sorted_names = sorted(dim_map.keys(), key=len, reverse=True)
    result = expr
    for name in sorted_names:
        # Use word boundaries to avoid partial matches (e.g., "C" shouldn't match in "MUp")
        pattern = r'\b' + re.escape(name) + r'\b'
        result = re.sub(pattern, dim_map[name], result)
    return result


def _param_spec_to_ref(
    spec: ParamSpec,
    config: Dict[str, Any],
    dim_map: Optional[Dict[str, str]] = None,
) -> TensorRef:
    """Convert a ParamSpec to a TensorRef.

    Args:
        spec: The ParamSpec to convert.
        config: Config dictionary with values for dimension names.
        dim_map: Optional mapping from annotation attribute names (like "C", "D")
                 to their Dim expression strings (like "d_model", "head_size").
                 If provided, annotation strings will be resolved through this map.
    """
    shape = []

    def resolve_dim(dim: Any) -> Any:
        """Resolve a dimension through dim_map and config."""
        parsed = _parse_shape_dim(dim)
        if isinstance(parsed, str):
            # First try to resolve through dim_map (annotation name -> Dim expression)
            if dim_map:
                if parsed in dim_map:
                    # Direct match (e.g., "C" -> "d_model")
                    parsed = dim_map[parsed]
                else:
                    # Expression with dimension names (e.g., "D // 2" -> "head_size // 2")
                    parsed = _substitute_dim_names(parsed, dim_map)
            # Then check if it's directly in config
            if parsed in config:
                return config[parsed]
            # If resolved to a numeric string (e.g., "1552" from dim_map integer),
            # convert to integer
            if isinstance(parsed, str) and parsed.isdigit():
                return int(parsed)
        return parsed

    # Handle ARRAY params (e.g., blocks: Array["n_layers", "BlockType"])
    if spec.kind == ParamKind.ARRAY and spec.array_size:
        shape.append(resolve_dim(spec.array_size))
    elif spec.shape:
        # Regular tensor params
        for dim in spec.shape:
            shape.append(resolve_dim(dim))

    return TensorRef(
        shape=shape,
        dtype=spec.dtype,
        is_param=True,
        quantizable=spec.quantizable,
        offload_group=spec.offload_group,
    )


def _is_symbolic_dim_expr(dim: str) -> bool:
    """Return True if *dim* looks like a valid symbolic Dim expression.

    Symbolic dim expressions come from ``_build_dim_map`` and are intentionally
    left as strings for the C++ runtime to resolve.  Examples::

        "C", "D", "E", "Hq", "MaxSeq", "Hq * D", "D // 2", "2 * M"

    Config parameter names that *should* have been resolved look like::

        "d_model", "head_size", "num_experts", "unknown_dim"

    Heuristic: a dim is symbolic if it starts with an uppercase letter or
    contains arithmetic operators.
    """
    if not dim:
        return False
    if any(op in dim for op in ("*", "+", "//", " - ")):
        return True
    if dim[0].isupper():
        return True
    return False


def _validate_param_shapes(
    params: Dict[str, TensorRef],
    *,
    warnings: WarningCollector | None = None,
) -> None:
    """Validate compiled parameter shapes.

    Checks:
    1. No parameter has a zero dimension (catches missing config exports like head_dim=0).
    2. Warn on unresolved string dimensions that look like config keys (lowercase/snake_case)
       but were not resolved to concrete values.  Symbolic Dim expressions (uppercase, arithmetic)
       are expected and silently passed through.
    """
    for name, ref in params.items():
        for i, dim in enumerate(ref.shape):
            if dim == 0:
                raise DSLShapeError(
                    f"param '{name}' has zero dimension at axis {i}",
                    hint="Ensure the dimension is exported to the IR config "
                         "(check @hf_config param_mapping and __init__).",
                )
            if isinstance(dim, str) and not dim.isdigit() and not _is_symbolic_dim_expr(dim):
                _warn(
                    warnings,
                    WarningCode.W006,
                    f"param '{name}' has unresolved string dimension '{dim}' at axis {i}",
                )


def _validate_graph_activation_slots(
    graph: GraphIR,
    layout: ActivationLayoutIR,
    *,
    warnings: WarningCollector | None = None,
) -> None:
    """Validate that graph intermediates have matching activation slots.

    Cross-references every graph intermediate/output name against the model's
    ActivationSlotIR names and aliases. This catches naming mismatches between
    the hybrid stacked blocks inlining and the activation layout declarations.
    """
    known: set[str] = {s.name for s in layout.slots}
    known |= set(layout.alias_map.keys())
    # Also include gradient slot names
    known |= {s.name for s in layout.gradient_slots}

    for name in graph.outputs:
        # Strip block prefix for matching (e.g., "blocks[0].mlp_down" -> "mlp_down")
        base_name = name
        dot = name.rfind(".")
        if dot >= 0:
            base_name = name[dot + 1:]

        if name not in known and base_name not in known:
            _warn(
                warnings,
                WarningCode.W007,
                f"graph output '{name}' has no matching activation slot",
            )


def _serialize_hf_spec(spec: Any) -> Any:
    """Serialize an HF weight mapping spec to JSON-compatible dict."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, FuseMapping):
        if len(spec.sources) < 2:
            raise DSLError(
                ErrorCode.E013,
                f"fuse() requires at least 2 sources, got {len(spec.sources)}",
            )
        payload = {
            "type": "fuse",
            "sources": list(spec.sources),
        }
        if spec.dim != 0:
            payload["dim"] = spec.dim
        return payload
    if isinstance(spec, SplitMapping):
        if not spec.ranges:
            raise DSLError(ErrorCode.E013, "split() requires ranges or parts")
        # Validate explicit ranges (ignore the parts sentinel (-1, parts))
        if spec.ranges and not (len(spec.ranges) == 1 and spec.ranges[0][0] == -1):
            for start, end in spec.ranges:
                if start < 0 or end <= start:
                    raise DSLError(
                        ErrorCode.E013,
                        f"invalid split() range [{start}, {end}] (expected 0 <= start < end)",
                    )
        payload = {
            "type": "split",
            "source": spec.source,
        }
        if spec.ranges:
            payload["ranges"] = list(spec.ranges)
        if spec.dim != 0:
            payload["dim"] = spec.dim
        return payload
    if isinstance(spec, TransformMapping):
        if not spec.fn:
            raise DSLError(ErrorCode.E010, "transform() requires a non-empty fn")
        payload = {
            "type": "transform",
            "source": spec.source,
        }
        if spec.fn:
            payload["fn"] = spec.fn
        return payload
    if isinstance(spec, TiedToMapping):
        if not spec.target:
            raise DSLError(ErrorCode.E010, "tied_to() requires a non-empty target")
        return {"type": "tied_to", "target": spec.target}
    if isinstance(spec, HFTransformSpec):
        payload = {"type": spec.kind}
        if spec.sources:
            payload["sources"] = spec.sources
        if spec.dim != 0:
            payload["dim"] = spec.dim
        if spec.fn:
            payload["fn"] = spec.fn
        return payload
    if isinstance(spec, StackExpertsMapping):
        if "{expert}" not in spec.pattern:
            raise DSLError(
                ErrorCode.E010,
                "stack_experts() pattern must include '{expert}' placeholder",
            )
        payload = {
            "type": "stack_experts",
            "pattern": spec.pattern,
        }
        if spec.num_experts > 0:
            payload["num_experts"] = spec.num_experts
        if spec.fuse_gate_up:
            payload["fuse_gate_up"] = spec.fuse_gate_up
        return payload
    raise DSLError(
        ErrorCode.E010,
        f"invalid weight mapping spec type: {type(spec).__name__}",
        hint="Expected a string HF path or an hf mapping helper like fuse()/split()/transform()/tied_to()/stack_experts().",
    )


def _compile_activation_slot(
    slot: ActivationSlotSpec,
    config: Dict[str, Any],
    dim_map: Optional[Dict[str, str]] = None,
    slot_index: int = -1,
) -> ActivationSlotIR:
    """Compile an ActivationSlotSpec to ActivationSlotIR.

    Args:
        slot: The activation slot specification.
        config: Config dictionary for dimension resolution.
        dim_map: Optional mapping from annotation names to Dim expressions.
        slot_index: Index of this slot in the activation struct.
    """
    # Resolve shape dimensions
    shape = []
    for dim in slot.shape:
        parsed = _parse_shape_dim(dim)
        if isinstance(parsed, str):
            # Try to resolve through dim_map first
            if dim_map and parsed in dim_map:
                parsed = dim_map[parsed]
            elif dim_map:
                parsed = _substitute_dim_names(parsed, dim_map)
            # Convert numeric strings to integers
            if isinstance(parsed, str) and parsed.isdigit():
                parsed = int(parsed)
        shape.append(parsed)

    return ActivationSlotIR(
        name=slot.name,
        scope=slot.scope.value,
        shape=shape,
        dtype=slot.dtype,
        aliases=list(slot.aliases),
        memory_hint=slot.memory_hint.value,
        shares_with=slot.shares_with,
        save_for_backward=slot.save_for_backward,
        recompute_in_backward=slot.recompute_in_backward,
        recompute_from=list(slot.recompute_from),
        recompute_op=slot.recompute_op,
        recompute_attrs=dict(slot.recompute_attrs),
        recompute_policy=slot.recompute_policy,
        recompute_group=slot.recompute_group,
        recompute_outputs=list(slot.recompute_outputs),
        lora_targets=list(slot.lora_targets),
        share_policy=slot.share_policy.value,
        gradient_of=slot.gradient_of,
        alias_of=slot.alias_of,
        slot_index=slot_index,
        condition=slot.condition_expr,
        description=slot.description,
    )


def _compile_activation_layout(
    layout: ActivationLayoutSpec,
    config: Dict[str, Any],
    dim_map: Optional[Dict[str, str]] = None,
) -> ActivationLayoutIR:
    """Compile an ActivationLayoutSpec to ActivationLayoutIR.

    Args:
        layout: The activation layout specification.
        config: Config dictionary for dimension resolution.
        dim_map: Optional mapping from annotation names to Dim expressions.
    """
    # Compile forward activation slots
    slots = []
    for i, slot in enumerate(layout.slots):
        slots.append(_compile_activation_slot(slot, config, dim_map, slot_index=i))

    # Compile gradient slots
    gradient_slots = []
    for i, slot in enumerate(layout.gradient_slots):
        gradient_slots.append(_compile_activation_slot(slot, config, dim_map, slot_index=i))

    # Build alias map
    alias_map = layout.build_alias_map()

    return ActivationLayoutIR(
        name=layout.name,
        slots=slots,
        gradient_slots=gradient_slots,
        alias_map=alias_map,
        extends=layout.extends,
    )


def _infer_output_names_from_graph(
    builder: GraphBuilder,
    num_outputs: int,
) -> Optional[List[str]]:
    """Infer output tensor names from graph by finding terminal tensors.

    For transformer blocks, we know the typical pattern:
    - First output: final MLP output (e.g., "mlp_down", "out", "mlp_output")
    - Second output: residual after attention (e.g., "res_att", "residual_att")

    We first try pattern matching, then fall back to graph structure analysis.
    """
    if not builder.nodes:
        return None

    # Collect all produced tensors and consumed tensors
    produced: List[str] = []  # Ordered list of produced tensors
    consumed: set = set()

    for node in builder.nodes:
        if isinstance(node, GraphNode):
            for inp in node.inputs:
                consumed.add(inp)
            for out in node.outputs:
                produced.append(out)

    # Find terminal tensors (produced but not consumed)
    terminals = set(t for t in produced if t not in consumed)

    # For 2-output blocks (common transformer pattern), look for known names
    if num_outputs == 2:
        # Common names for first output (final MLP/block output)
        first_candidates = ["mlp_down", "out", "mlp_output", "block_out", "ffn_out"]
        # Common names for second output (residual after attention)
        second_candidates = ["res_att", "residual_att", "residual_attention", "res_attn"]

        first_out = None
        second_out = None

        # Find first output (MLP output)
        for name in first_candidates:
            if name in terminals:
                first_out = name
                break

        # Find second output (residual)
        for name in second_candidates:
            if name in terminals:
                second_out = name
                break

        if first_out and second_out:
            return [first_out, second_out]

    # Fallback: find the last operation that produces unused outputs
    # and use those as the block outputs
    if num_outputs == 1:
        # Single output: look for the last terminal tensor
        for t in reversed(produced):
            if t in terminals:
                return [t]

    # For multiple outputs, try to find the last N terminals in produced order
    # But filter to only include actual terminal tensors
    terminal_list = [t for t in produced if t in terminals]
    if len(terminal_list) >= num_outputs:
        # Take last N, but this may not preserve the return order
        # At least we're getting actual tensor names
        return terminal_list[-num_outputs:]

    return None


def _compile_graph_builder(
    builder: GraphBuilder,
    spec: ForwardSpec,
    config: Dict[str, Any],
    params: Dict[str, ParamSpec],
    *,
    dim_map: Optional[Dict[str, str]] = None,
    warnings: WarningCollector | None = None,
) -> GraphIR:
    """Compile a GraphBuilder to GraphIR."""
    graph = GraphIR()

    # Add inputs from the forward spec
    for i, io_spec in enumerate(spec.inputs):
        ann = io_spec.tensor_type
        graph.inputs[io_spec.name] = _tensor_annotation_to_ref(ann, is_input=True)

    # Add outputs - use actual tensor names from graph if available
    # The returned outputs from forward() are stored in builder._returned_outputs
    returned_outputs = getattr(builder, '_returned_outputs', None)

    # If _returned_outputs wasn't captured, try to infer from graph structure
    if not returned_outputs and len(spec.outputs) > 0:
        inferred = _infer_output_names_from_graph(builder, len(spec.outputs))
        if inferred:
            returned_outputs = inferred

    for i, io_spec in enumerate(spec.outputs):
        ann = io_spec.tensor_type
        # Use the actual tensor name from the graph operations if available
        if returned_outputs and i < len(returned_outputs):
            out_name = returned_outputs[i]
        else:
            out_name = io_spec.name if io_spec.name else f"out_{i}"
        graph.outputs[out_name] = _tensor_annotation_to_ref(ann, is_output=True)

    # Add params (tensor weights only; arrays are expanded separately)
    for name, param_spec in params.items():
        if param_spec.kind != ParamKind.TENSOR:
            continue
        if param_spec.condition:
            try:
                mock = type("ConfigView", (), {})()
                for key, value in config.items():
                    setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass
        graph.params[name] = _param_spec_to_ref(param_spec, config, dim_map)

    # Convert nodes
    for i, node in enumerate(builder.nodes):
        if isinstance(node, GraphNode):
            # Check for _kernel_type in attrs (used by call() for module invocations)
            kernel_type = node.attrs.pop("_kernel_type", None) or node.op
            op = OpIR(
                id=i,
                name=node.op,
                kernel_type=kernel_type,
                inputs=node.inputs,
                outputs=node.outputs,
                attrs=node.attrs,
            )
            graph.nodes.append(op)

            # Track intermediates
            for out in node.outputs:
                if out not in graph.inputs and out not in graph.params:
                    graph.intermediates[out] = TensorRef(
                        shape=[],  # Shape inference would go here
                        is_param=False,
                        is_input=False,
                        is_output=out in [o for o in graph.outputs.keys()],
                    )

    # Save/recompute lists
    graph.save_list = _dedupe_preserve_order(list(spec.save) + list(builder._save_list))
    graph.recompute_list = _dedupe_preserve_order(list(spec.recompute) + list(builder._recompute_list))

    # Warn on save entries that don't correspond to any tensor in the graph.
    available = set(graph.inputs) | set(graph.params) | set(graph.intermediates) | set(graph.outputs)
    for name in graph.save_list:
        if name.startswith("@param:") or name.startswith("saved."):
            continue
        if name not in available:
            _warn(warnings, WarningCode.W004, f"unused saved tensor '{name}' (not present in graph)")

    return graph


def _init_instance_from_config(instance: Any, cls: type, config: Dict[str, Any]) -> None:
    """Initialize instance with config keys accepted by __init__.

    Raises ValueError if the __init__ fails with a ValueError (config validation error).
    Other exceptions are silently ignored for backward compatibility.
    """
    import inspect

    if not hasattr(cls, "__init__"):
        return
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return

    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in config:
            kwargs[name] = config[name]
    try:
        cls.__init__(instance, **kwargs)
    except ValueError:
        # Re-raise ValueError as these typically indicate config validation errors
        # (e.g., hybrid_override_pattern length doesn't match n_layers)
        raise
    except Exception:
        pass


def _expand_hybrid_hf_mappings(ir: "ModuleIR", block_types: list, block_mappings: dict) -> None:
    """Expand template-based HF mappings for hybrid models with physical layer indices.

    In hybrid models, all blocks use ``blocks[N]`` with physical layer indices.
    This function generates per-param HF mapping entries by looking up the block
    type for each physical layer and applying the corresponding HF mapping template.
    """

    def _replace_layer(spec, phys: int):
        """Replace {layer} in a serialized HF mapping spec."""
        s = str(phys)
        if isinstance(spec, str):
            return spec.replace("{layer}", s)
        if isinstance(spec, dict):
            out = {}
            for k, v in spec.items():
                if isinstance(v, str):
                    out[k] = v.replace("{layer}", s)
                elif isinstance(v, list):
                    out[k] = [x.replace("{layer}", s) if isinstance(x, str) else x for x in v]
                else:
                    out[k] = v
            return out
        return spec

    if not ir.forward_graph:
        return

    for param_name in ir.forward_graph.params:
        dot = param_name.find(".")
        if dot < 0:
            continue
        prefix_part = param_name[:dot]
        field = param_name[dot + 1:]
        # Match blocks[N]
        if not prefix_part.startswith("blocks["):
            continue
        close = prefix_part.find("]")
        if close < 0:
            continue
        try:
            phys_layer = int(prefix_part[7:close])
        except ValueError:
            continue
        if phys_layer < 0 or phys_layer >= len(block_types):
            continue
        if field not in block_mappings:
            continue
        serialized = _serialize_hf_spec(block_mappings[field])
        ir.hf_weight_mapping[param_name] = _replace_layer(serialized, phys_layer)


def _inline_stacked_blocks(
    graph: GraphIR,
    model_spec: ModelSpec,
    config: Dict[str, Any],
    *,
    warnings: WarningCollector | None = None,
) -> GraphIR:
    """Inline StackedBlocks and HybridStackedBlocks calls into per-layer block graphs."""
    # Quick check: if no StackedBlocks or HybridStackedBlocks present, return as-is
    if not any(node.name in ("StackedBlocks", "HybridStackedBlocks") for node in graph.nodes):
        return graph

    # Resolve block param spec from model
    def _resolve_block_spec(blocks_param: str) -> BlockSpec:
        param_spec = model_spec.params.get(blocks_param)
        if not param_spec or param_spec.kind != ParamKind.ARRAY or not param_spec.element_type:
            raise DSLTypeError(
                f"StackedBlocks expects array param '{blocks_param}' with element_type",
                hint="Declare blocks as Param(Array[\"n_layers\", \"YourBlock\"]) and pass blocks=\"blocks\" in g.call().",
            )
        block_spec = get_block_spec(param_spec.element_type)
        if block_spec is None:
            raise DSLUndefinedError(param_spec.element_type)
        return block_spec

    # Cache compiled block graphs by block name
    block_cache: Dict[str, ModuleIR] = {}

    def _get_block_ir(block_spec: BlockSpec) -> ModuleIR:
        cached = block_cache.get(block_spec.name)
        if cached is not None:
            return cached
        ir = compile_block_spec(block_spec, config)
        block_cache[block_spec.name] = ir
        return ir

    new_nodes: List[OpIR] = []
    new_params: Dict[str, TensorRef] = dict(graph.params)
    op_id = 0

    for node in graph.nodes:
        if node.name not in ("StackedBlocks", "HybridStackedBlocks"):
            op_id += 1
            new_nodes.append(OpIR(
                id=op_id,
                name=node.name,
                kernel_type=node.kernel_type,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                attrs=dict(node.attrs),
            ))
            continue

        n_layers = node.attrs.get("n_layers") or config.get("n_layers")
        if n_layers is None:
            raise DSLError(
                ErrorCode.E012,
                f"{node.name} missing n_layers",
                hint="Pass n_layers=... in g.call(...), or provide n_layers in the model config.",
            )

        if node.name == "HybridStackedBlocks":
            # HybridStackedBlocks: different block types per layer
            block_types = node.attrs.get("block_types")
            if not block_types:
                raise DSLError(
                    ErrorCode.E012,
                    "HybridStackedBlocks missing block_types",
                    hint="Pass block_types=[...] in g.call('HybridStackedBlocks', ...).",
                )
            if len(block_types) != int(n_layers):
                raise DSLError(
                    ErrorCode.E012,
                    f"HybridStackedBlocks block_types length ({len(block_types)}) != n_layers ({n_layers})",
                )

            # Map block type names to their param arrays
            block_type_to_param = {
                "mamba": node.attrs.get("mamba_blocks", "mamba_blocks"),
                "attention": node.attrs.get("attn_blocks", "attn_blocks"),
                "mlp": node.attrs.get("mlp_blocks", "mlp_blocks"),
                "moe": node.attrs.get("moe_blocks", "moe_blocks"),
            }

            # Track indices per block type
            block_type_indices: Dict[str, int] = {
                "mamba": 0,
                "attention": 0,
                "mlp": 0,
                "moe": 0,
            }

            # Resolve all block specs upfront
            block_specs: Dict[str, BlockSpec] = {}
            block_irs: Dict[str, ModuleIR] = {}
            for block_type, param_name in block_type_to_param.items():
                # Only resolve if this block type is actually used
                if block_type in block_types:
                    param_spec = model_spec.params.get(param_name)
                    if param_spec and param_spec.kind == ParamKind.ARRAY and param_spec.element_type:
                        spec = get_block_spec(param_spec.element_type)
                        if spec:
                            block_specs[block_type] = spec
                            block_irs[block_type] = _get_block_ir(spec)

            cur_inputs = list(node.inputs)

            for layer_idx in range(int(n_layers)):
                block_type = block_types[layer_idx]
                if block_type not in block_specs:
                    raise DSLError(
                        ErrorCode.E012,
                        f"No block spec found for block type '{block_type}' at layer {layer_idx}",
                        hint=f"Make sure {block_type_to_param.get(block_type, block_type + '_blocks')} param is defined.",
                    )

                block_spec = block_specs[block_type]
                block_ir = block_irs[block_type]
                block_graph = block_ir.forward_graph
                if not block_graph or not block_graph.nodes:
                    raise DSLError(
                        ErrorCode.E012,
                        f"Block graph missing for {block_spec.name}",
                        hint="Ensure the block defines an @forward method.",
                    )

                block_inputs = list(block_graph.inputs.keys())
                block_outputs = list(block_graph.outputs.keys())
                block_params = list(block_graph.params.keys())

                # Get the index within this block type's array
                block_idx = block_type_indices[block_type]
                block_type_indices[block_type] += 1

                blocks_param = block_type_to_param[block_type]

                # Outputs for this layer
                if layer_idx == int(n_layers) - 1:
                    layer_outputs = list(node.outputs)
                else:
                    # Residual outputs need blocks[N].* prefix so the C++ residual
                    # manager intercepts them.  Non-residual outputs (hidden state)
                    # use a neutral name to avoid the block-activation resolver.
                    layer_outputs = [
                        f"blocks[{layer_idx}].{name}"
                        if name in ("res_in", "res_ffn", "res_att")
                        else f"layer{layer_idx}.{name}"
                        for name in block_outputs
                    ]

                # Build name mapping — use physical layer index with uniform "blocks[N]"
                # prefix so the C++ parse_block_param (which expects "blocks[N]") works.
                prefix = f"blocks[{layer_idx}]."
                mapping: Dict[str, str] = {}

                # Map block inputs to current layer inputs (handle mismatched counts)
                # Some blocks take 2 inputs (x, residual), some take 3 (x, residual, position_ids)
                for i, b_in in enumerate(block_inputs):
                    if i < len(cur_inputs):
                        mapping[b_in] = cur_inputs[i]
                    else:
                        # For position_ids or other inputs not in cur_inputs, use original
                        if i < len(node.inputs):
                            mapping[b_in] = node.inputs[i]

                for b_out, c_out in zip(block_outputs, layer_outputs):
                    mapping[b_out] = c_out
                for p in block_params:
                    mapping[p] = f"{prefix}{p}"
                    if mapping[p] not in new_params:
                        new_params[mapping[p]] = block_graph.params[p]

                # Inline block nodes
                for bnode in block_graph.nodes:
                    op_id += 1
                    mapped_inputs = [mapping.get(i, f"{prefix}{i}") for i in bnode.inputs]
                    mapped_outputs = [mapping.get(o, f"{prefix}{o}") for o in bnode.outputs]
                    new_nodes.append(OpIR(
                        id=op_id,
                        name=bnode.name,
                        kernel_type=bnode.kernel_type,
                        inputs=mapped_inputs,
                        outputs=mapped_outputs,
                        attrs=dict(bnode.attrs),
                    ))

                # Next layer inputs (x, residual from outputs; keep position_ids)
                cur_inputs = list(layer_outputs[:2])  # First 2 outputs are typically (out, residual)
                if len(node.inputs) > 2:
                    cur_inputs.append(node.inputs[2])  # position_ids stays constant

        else:
            # Standard StackedBlocks: same block type for all layers
            blocks_param = node.attrs.get("blocks", "blocks")
            block_spec = _resolve_block_spec(blocks_param)
            block_ir = _get_block_ir(block_spec)
            if not block_ir.forward_graph or not block_ir.forward_graph.nodes:
                raise DSLError(
                    ErrorCode.E012,
                    f"Block graph missing for {block_spec.name}",
                    hint="Ensure the block defines an @forward method using 'with graph() as g:' and returns GraphRef(s).",
                )

            block_graph = block_ir.forward_graph
            block_inputs = list(block_graph.inputs.keys())
            block_outputs = list(block_graph.outputs.keys())
            block_params = list(block_graph.params.keys())

            # StackedBlocks inputs are (x, residual, position_ids)
            cur_inputs = list(node.inputs)
            if len(cur_inputs) < len(block_inputs):
                raise DSLTypeError(
                    f"StackedBlocks input mismatch: expected at least {len(block_inputs)}, got {len(cur_inputs)}",
                    hint="Make sure g.call('StackedBlocks', ...) inputs match the block forward signature.",
                )
            extra_inputs = []
            if len(cur_inputs) > len(block_inputs):
                extra_inputs = cur_inputs[len(block_inputs):]
                cur_inputs = cur_inputs[:len(block_inputs)]
            position_ids_input = cur_inputs[-1]
            deepstack_layers = int(node.attrs.get("deepstack_layers", 0)) if node.attrs else 0

            for layer_idx in range(int(n_layers)):
                # Outputs for this layer
                if layer_idx == int(n_layers) - 1:
                    layer_outputs = list(node.outputs)
                else:
                    layer_outputs = [f"{blocks_param}[{layer_idx}].{name}" for name in block_outputs]

                # Build name mapping
                prefix = f"{blocks_param}[{layer_idx}]."
                mapping: Dict[str, str] = {}
                for b_in, c_in in zip(block_inputs, cur_inputs):
                    mapping[b_in] = c_in
                for b_out, c_out in zip(block_outputs, layer_outputs):
                    mapping[b_out] = c_out
                for p in block_params:
                    mapping[p] = f"{prefix}{p}"
                    if mapping[p] not in new_params:
                        new_params[mapping[p]] = block_graph.params[p]

                # Inline block nodes
                for bnode in block_graph.nodes:
                    op_id += 1
                    mapped_inputs = [mapping.get(i, f"{prefix}{i}") for i in bnode.inputs]
                    mapped_outputs = [mapping.get(o, f"{prefix}{o}") for o in bnode.outputs]
                    new_nodes.append(OpIR(
                        id=op_id,
                        name=bnode.name,
                        kernel_type=bnode.kernel_type,
                        inputs=mapped_inputs,
                        outputs=mapped_outputs,
                        attrs=dict(bnode.attrs),
                    ))

                # Inject deepstack visual embeddings after the layer output (first N layers only)
                if deepstack_layers > 0 and extra_inputs:
                    visual_mask = extra_inputs[0]
                    deepstack_inputs = extra_inputs[1:]
                    if layer_idx < deepstack_layers and layer_idx < len(deepstack_inputs):
                        op_id += 1
                        new_nodes.append(OpIR(
                            id=op_id,
                            name="deepstack_inject",
                            kernel_type="deepstack_inject",
                            inputs=[layer_outputs[0], visual_mask, deepstack_inputs[layer_idx]],
                            outputs=[layer_outputs[0]],
                            attrs={},
                        ))

                # Next layer inputs
                cur_inputs = list(layer_outputs[:len(block_inputs) - 1])
                cur_inputs.append(position_ids_input)  # position_ids stays constant

    # Rebuild intermediates
    new_intermediates: Dict[str, TensorRef] = {}
    for op in new_nodes:
        for out in op.outputs:
            if out in graph.inputs or out in new_params or out in graph.outputs:
                continue
            if out not in new_intermediates:
                new_intermediates[out] = TensorRef(shape=[], is_param=False, is_input=False, is_output=False)

    graph.nodes = new_nodes
    graph.params = new_params
    graph.intermediates = new_intermediates
    return graph


def _evaluate_forward_for_graph(
    model_class: type,
    config: Dict[str, Any],
) -> Optional[GraphBuilder]:
    """
    Instantiate the model and call forward to capture the graph.

    This creates a mock instance with config values as attributes,
    then calls the forward method to build the graph.
    """
    # Create instance with config
    try:
        instance = model_class(**config)
    except Exception:
        # Try creating with minimal args
        try:
            instance = object.__new__(model_class)
            for key, value in config.items():
                setattr(instance, key, value)
        except Exception:
            return None

    # Find forward method
    forward_fn = None
    for name in dir(model_class):
        attr = getattr(model_class, name, None)
        if attr is not None and hasattr(attr, "_forward_spec"):
            forward_fn = attr
            break

    if forward_fn is None:
        return None

    # Call forward with mock inputs to capture graph
    # The forward function uses graph() context which builds the graph
    # We need to extract the GraphBuilder after execution

    # For now, we'll rely on the _forward_spec.graph_fn being set
    # The actual graph construction happens when forward is called
    # with proper inputs. For IR generation, we'll extract from the spec directly.

    return None


def _compile_merged_activation_layout(
    spec: ModelSpec,
    config: Dict[str, Any],
    dim_map: Optional[Dict[str, str]] = None,
) -> Optional[ActivationLayoutIR]:
    """Compile and merge model's global activation slots with block activation slots.

    For models with stacked blocks, the activation layout needs both:
    1. Global slots declared at model level (e.g., token_ids, xF, loss)
    2. Block-scoped slots declared in the block class (e.g., ln1, qkv, mlp_down)

    The C++ runtime uses this merged layout to resolve tensor references like
    "blocks[0].mlp_down" to the correct TensorSlot.

    Args:
        spec: The model specification.
        config: Config dictionary for dimension resolution.
        dim_map: Optional mapping from annotation names to Dim expressions.

    Returns:
        Merged ActivationLayoutIR containing both global and block-scoped slots,
        or None if no activation slots are declared.
    """
    slots: List[ActivationSlotIR] = []
    gradient_slots: List[ActivationSlotIR] = []
    alias_map: Dict[str, str] = {}

    # 1. Compile model's global activation slots
    if spec.activations:
        model_layout = _compile_activation_layout(spec.activations, config, dim_map)
        slots.extend(model_layout.slots)
        gradient_slots.extend(model_layout.gradient_slots)
        alias_map.update(model_layout.alias_map)

    # 2. Find block spec and compile its activation slots
    # Look for the 'blocks' param (or similar array param that references a block type)
    block_spec = None
    for param_name, param_spec in spec.params.items():
        if param_spec.kind == ParamKind.ARRAY and param_spec.element_type:
            # This is an array of blocks - get the block spec
            block_spec = get_block_spec(param_spec.element_type)
            if block_spec is not None:
                break

    if block_spec and block_spec.activations:
        # Build dim_map for the block if we have its python_class
        block_dim_map: Dict[str, str] = {}
        if block_spec.python_class:
            try:
                block_instance = object.__new__(block_spec.python_class)
                for key, value in config.items():
                    setattr(block_instance, key, value)
                _init_instance_from_config(block_instance, block_spec.python_class, config)
                block_dim_map = _build_dim_map(block_instance)
            except Exception:
                pass

        # Compile block activation layout
        block_layout = _compile_activation_layout(
            block_spec.activations, config, block_dim_map or dim_map
        )

        # Add block slots with scope="block"
        # The slot_index continues from where model slots ended
        block_slot_start = len(slots)
        for i, slot in enumerate(block_layout.slots):
            # Ensure block slots have scope="block"
            if slot.scope in ("block", ""):
                slot.scope = "block"
            slot.slot_index = block_slot_start + i
            slots.append(slot)

        # Add block gradient slots
        block_grad_start = len(gradient_slots)
        for i, slot in enumerate(block_layout.gradient_slots):
            if slot.scope in ("gradient", ""):
                slot.scope = "gradient"
            slot.slot_index = block_grad_start + i
            gradient_slots.append(slot)

        # Merge alias maps
        alias_map.update(block_layout.alias_map)

    # If no slots at all, return None
    if not slots and not gradient_slots:
        return None

    # Determine layout name
    layout_name = spec.activations.name if spec.activations else f"{spec.name}Activations"

    return ActivationLayoutIR(
        name=layout_name,
        slots=slots,
        gradient_slots=gradient_slots,
        alias_map=alias_map,
    )


def _capture_forward_graph(
    forward_fn: Any,
    instance: Any,
    inputs: List[IOSpec],
) -> Optional[GraphBuilder]:
    """
    Capture the graph from a forward function by temporarily patching the graph context.

    The forward function creates its own graph() context inside, so we need to
    intercept that context to capture the nodes.
    """
    from .graph_builder import GraphBuilder, _graph_stack
    import surogate.dsl.graph_builder as graph_module

    captured_builder: Optional[GraphBuilder] = None

    # Create a patched graph context manager that captures the builder
    from contextlib import contextmanager

    @contextmanager
    def patched_graph():
        nonlocal captured_builder
        builder = GraphBuilder()
        _graph_stack.append(builder)
        try:
            yield builder
        finally:
            _graph_stack.pop()
            captured_builder = builder

    # The forward function imports `graph` from graph_builder at import time.
    # We need to patch it in multiple places:
    # 1. The graph_module.graph (for direct imports)
    # 2. Any module that has already imported 'graph'

    # Get the module containing the forward function
    import sys
    original_graph = graph_module.graph

    # Find all modules that might have imported graph
    modules_to_patch = []
    for name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, 'graph') and getattr(mod, 'graph', None) is original_graph:
            modules_to_patch.append((mod, 'graph', original_graph))

    # Patch all occurrences
    graph_module.graph = patched_graph
    for mod, attr, _ in modules_to_patch:
        setattr(mod, attr, patched_graph)

    try:
        # Prepare mock inputs
        mock_inputs = [io.name for io in inputs]

        # Call forward and capture the return value (GraphRef or tuple of GraphRefs)
        returned_outputs = None
        forward_exception = None
        try:
            returned_outputs = forward_fn(instance, *mock_inputs)
        except Exception as e:
            forward_exception = e
            # Continue - we may still have captured graph nodes even if return failed

        # Store the returned output tensor names on the builder for later use
        if captured_builder is not None:
            from .graph_builder import GraphRef
            if returned_outputs is not None:
                if isinstance(returned_outputs, GraphRef):
                    captured_builder._returned_outputs = [returned_outputs.name]
                elif isinstance(returned_outputs, tuple):
                    captured_builder._returned_outputs = [
                        ref.name if isinstance(ref, GraphRef) else str(ref)
                        for ref in returned_outputs
                    ]
                else:
                    captured_builder._returned_outputs = []
            elif forward_exception is not None:
                # If forward raised an exception but we captured nodes,
                # try to infer outputs from the graph builder
                # This helps when the exception happens during return value processing
                # but the graph was already built
                pass  # _compile_graph_builder will try to infer

        return captured_builder
    finally:
        # Restore originals
        graph_module.graph = original_graph
        for mod, attr, orig in modules_to_patch:
            setattr(mod, attr, orig)


def compile_model_spec(
    spec: ModelSpec,
    config: Dict[str, Any],
    *,
    warnings: WarningCollector | None = None,
) -> ModuleIR:
    """Compile a ModelSpec to ModuleIR."""
    if spec.name in _primitive_registry:
        _warn(warnings, WarningCode.W001, f"model '{spec.name}' shadows a registered primitive of the same name")

    if spec.forward is None or spec.forward.graph_fn is None:
        raise DSLError(
            ErrorCode.E012,
            f"model '{spec.name}' is missing an @forward method",
            hint="Define a forward(self, ...) method decorated with @forward that builds the graph using 'with graph() as g:'.",
        )

    ir = ModuleIR(
        name=spec.name,
        kind="model",
        is_model=True,
        config=config,
    )

    # HF config mapping
    if spec.hf_config:
        ir.hf_config = {
            "architecture": spec.hf_config.architecture,
            "param_mapping": spec.hf_config.param_mapping,
        }
        if spec.hf_config.model_type:
            ir.hf_config["model_type"] = spec.hf_config.model_type

    # HF weight mapping (from class attribute _hf_block_mappings_ and @param decorators)
    if spec.hf_mapping:
        for name, mapping in spec.hf_mapping.mappings.items():
            ir.hf_weight_mapping[name] = _serialize_hf_spec(mapping)

    # Also check for _hf_block_mappings_ class attribute
    if spec.python_class and hasattr(spec.python_class, "_hf_block_mappings_"):
        block_mappings = spec.python_class._hf_block_mappings_
        for name, mapping in block_mappings.items():
            ir.hf_weight_mapping[name] = _serialize_hf_spec(mapping)

    # Check @param(hf_mapping=...) decorators
    for name, param_spec in spec.params.items():
        if param_spec.hf_path:
            ir.hf_weight_mapping[name] = param_spec.hf_path

    # HF export mapping
    if spec.hf_export:
        for name, mapping in spec.hf_export.mappings.items():
            ir.hf_export_mapping[name] = _serialize_hf_spec(mapping)

    # Create instance first so we can build dim_map for param resolution
    instance = None
    dim_map: Dict[str, str] = {}
    init_error: Exception | None = None
    if spec.python_class:
        try:
            instance = object.__new__(spec.python_class)
            for key, value in config.items():
                setattr(instance, key, value)
            _init_instance_from_config(instance, spec.python_class, config)
            dim_map = _build_dim_map(instance)

            # Capture any computed integer attributes from the instance that might be
            # needed for shape resolution (e.g., n_mamba_blocks, n_attn_blocks for hybrid models)
            # These are attributes that exist on the instance but not in config.
            # Also capture important string attributes like hybrid_pattern for runtime config.
            _important_string_attrs = {"hybrid_pattern", "mlp_activation", "activation"}
            _important_float_attrs = {"routed_scaling_factor"}
            for attr_name in dir(instance):
                if attr_name.startswith("_"):
                    continue
                # Always re-capture important string/float attrs from the instance —
                # __init__ may translate/transform the raw HF value (e.g.
                # hybrid_pattern: Nemotron '*'/'-' → standard 'A'/'P').
                _force_recapture = _important_string_attrs | _important_float_attrs
                if attr_name in config and attr_name not in _force_recapture:
                    continue
                try:
                    attr_val = getattr(instance, attr_name)
                    # Capture integers that look like dimension values
                    if isinstance(attr_val, int) and attr_val >= 0:
                        config[attr_name] = attr_val
                    # Also capture important string attributes for runtime config
                    elif isinstance(attr_val, str) and attr_name in _important_string_attrs:
                        config[attr_name] = attr_val
                    # Capture important float attributes for runtime config
                    elif isinstance(attr_val, float) and attr_name in _important_float_attrs:
                        config[attr_name] = attr_val
                except Exception:
                    pass
        except ValueError as e:
            # Critical configuration errors (e.g., hybrid_override_pattern length mismatch)
            # should be propagated rather than silently ignored
            init_error = e
        except Exception:
            pass

    # If there was a critical init error, raise it now with context
    if init_error is not None:
        raise DSLError(
            ErrorCode.E012,
            f"Model initialization failed: {init_error}",
            hint="Check that the model config is consistent (e.g., hybrid_override_pattern length matches num_hidden_layers).",
        )

    # Params - use dim_map to resolve annotation strings to Dim expressions
    for name, param_spec in spec.params.items():
        # Check condition
        if param_spec.condition:
            try:
                mock = instance if instance else object.__new__(spec.python_class)
                if not instance:
                    for key, value in config.items():
                        setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config, dim_map)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and instance:
            # Capture the graph by patching graph()
            builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)
            if builder is None:
                raise DSLSyntaxError(
                    f"could not capture forward graph for model '{spec.name}'",
                    hint="Ensure @forward uses 'with graph() as g:' and returns GraphRef(s).",
                )

            graph = _compile_graph_builder(builder, spec.forward, config, spec.params, dim_map=dim_map, warnings=warnings)
            # Expand StackedBlocks into per-layer block graphs
            graph = _inline_stacked_blocks(graph, spec, config, warnings=warnings)
            ir.forward_graph = graph

            # For hybrid models, expand block-level HF mappings with physical layer
            # indices so the C++ weight loader resolves typed block indices correctly.
            if (instance and hasattr(instance, "block_types") and
                    spec.python_class and hasattr(spec.python_class, "_hf_block_mappings_")):
                _expand_hybrid_hf_mappings(
                    ir, instance.block_types, spec.python_class._hf_block_mappings_)

    # Save/recompute
    if spec.forward:
        ir.save_tensors = _dedupe_preserve_order(list(spec.forward.save))
        ir.recompute_tensors = _dedupe_preserve_order(list(spec.forward.recompute))

    # Compile activation layout - merge model's global activations with block activations
    ir.activation_layout = _compile_merged_activation_layout(spec, config, dim_map)

    # --- Validation passes ---
    # 1. Validate param shapes (zero dims and unresolved string dims)
    if ir.forward_graph:
        _validate_param_shapes(ir.forward_graph.params, warnings=warnings)

    # 2. Validate graph outputs against activation slots
    if ir.forward_graph and ir.activation_layout:
        _validate_graph_activation_slots(ir.forward_graph, ir.activation_layout, warnings=warnings)

    return ir


def compile_block_spec(
    spec: BlockSpec,
    config: Dict[str, Any],
    *,
    warnings: WarningCollector | None = None,
) -> ModuleIR:
    """Compile a BlockSpec to ModuleIR."""
    if spec.name in _primitive_registry:
        _warn(warnings, WarningCode.W001, f"block '{spec.name}' shadows a registered primitive of the same name")

    if spec.extends:
        base = get_block_spec(spec.extends)
        if base is None:
            raise DSLUndefinedError(
                spec.extends,
                hint=f"block '{spec.name}' extends '{spec.extends}', but '{spec.extends}' is not registered",
            )

    ir = ModuleIR(
        name=spec.name,
        kind="block",
        is_block=True,
        extends=spec.extends,
        config=config,
    )

    # Create instance first so we can build dim_map for param resolution
    instance = None
    dim_map: Dict[str, str] = {}
    if spec.python_class:
        try:
            instance = object.__new__(spec.python_class)
            for key, value in config.items():
                setattr(instance, key, value)
            _init_instance_from_config(instance, spec.python_class, config)
            dim_map = _build_dim_map(instance)
        except Exception:
            pass

    # Params - use dim_map to resolve annotation strings to Dim expressions
    for name, param_spec in spec.params.items():
        # Check condition
        if param_spec.condition:
            try:
                mock = instance if instance else object.__new__(spec.python_class)
                if not instance:
                    for key, value in config.items():
                        setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config, dim_map)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and instance:
            builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)
            if builder is None:
                raise DSLSyntaxError(
                    f"could not capture forward graph for block '{spec.name}'",
                    hint="Ensure @forward uses 'with graph() as g:' and returns GraphRef(s).",
                )
            ir.forward_graph = _compile_graph_builder(builder, spec.forward, config, spec.params, dim_map=dim_map, warnings=warnings)

    # Compile activation layout if present
    if spec.activations:
        ir.activation_layout = _compile_activation_layout(spec.activations, config, dim_map)

    # Validate param shapes (zero dims and unresolved string dims)
    if ir.forward_graph:
        _validate_param_shapes(ir.forward_graph.params, warnings=warnings)

    return ir


def compile_module_spec(
    spec: ModuleSpec,
    config: Dict[str, Any],
    *,
    warnings: WarningCollector | None = None,
) -> ModuleIR:
    """Compile a ModuleSpec to ModuleIR."""
    if spec.name in _primitive_registry:
        _warn(warnings, WarningCode.W001, f"module '{spec.name}' shadows a registered primitive of the same name")

    if spec.extends:
        base = get_module_spec(spec.extends)
        if base is None:
            raise DSLUndefinedError(
                spec.extends,
                hint=f"module '{spec.name}' extends '{spec.extends}', but '{spec.extends}' is not registered",
            )

    ir = ModuleIR(
        name=spec.name,
        kind="module",
        extends=spec.extends,
        config=config,
    )

    # Create instance first so we can build dim_map for param resolution
    instance = None
    dim_map: Dict[str, str] = {}
    if spec.python_class:
        try:
            instance = object.__new__(spec.python_class)
            for key, value in config.items():
                setattr(instance, key, value)
            _init_instance_from_config(instance, spec.python_class, config)
            dim_map = _build_dim_map(instance)
        except Exception:
            pass

    # Params - use dim_map to resolve annotation strings to Dim expressions
    for name, param_spec in spec.params.items():
        if param_spec.condition:
            try:
                mock = instance if instance else object.__new__(spec.python_class)
                if not instance:
                    for key, value in config.items():
                        setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config, dim_map)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and instance:
            builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)
            if builder is None:
                # Abstract modules may intentionally omit a forward graph.
                if getattr(spec, "is_abstract", False):
                    ir.forward_graph = GraphIR()
                else:
                    raise DSLSyntaxError(
                        f"could not capture forward graph for module '{spec.name}'",
                        hint="Ensure @forward uses 'with graph() as g:' and returns GraphRef(s).",
                    )
            else:
                ir.forward_graph = _compile_graph_builder(builder, spec.forward, config, spec.params, dim_map=dim_map, warnings=warnings)

    # Validate param shapes (zero dims and unresolved string dims)
    if ir.forward_graph:
        _validate_param_shapes(ir.forward_graph.params, warnings=warnings)

    return ir


# =============================================================================
# Serialization
# =============================================================================


class _OffloadGroupResolver:
    """Resolves string offload_group names to stable integer IDs.

    String group names (e.g., ``"moe_experts"``) from ``Param(offload_group=...)``
    are mapped to monotonically increasing integer IDs starting from 0.
    Integer group IDs are passed through unchanged. The resolver is reset
    per compilation unit.
    """

    def __init__(self) -> None:
        self._name_to_id: Dict[str, int] = {}
        self._next_id: int = 0

    def resolve(self, group: int | str) -> int:
        if isinstance(group, int):
            return group
        if group not in self._name_to_id:
            self._name_to_id[group] = self._next_id
            self._next_id += 1
        return self._name_to_id[group]

    def reset(self) -> None:
        self._name_to_id.clear()
        self._next_id = 0


# Shared resolver instance (reset before each compilation)
_offload_resolver = _OffloadGroupResolver()


def _tensor_ref_to_dict(ref: TensorRef) -> Dict[str, Any]:
    """Convert TensorRef to JSON-serializable dict."""
    result = {
        "shape": ref.shape,
        "dtype": ref.dtype,
        "is_param": ref.is_param,
        "is_input": ref.is_input,
        "is_output": ref.is_output,
    }
    if ref.is_param:
        if not ref.quantizable:
            result["quantizable"] = False
        if ref.offload_group != -1:
            result["offload_group"] = _offload_resolver.resolve(ref.offload_group)
    return result


def _graph_ir_to_dict(graph: GraphIR) -> Dict[str, Any]:
    """Convert GraphIR to JSON-serializable dict."""
    return {
        "name": graph.name,
        "num_ops": len(graph.nodes),
        "inputs": {name: _tensor_ref_to_dict(ref) for name, ref in graph.inputs.items()},
        "outputs": {name: _tensor_ref_to_dict(ref) for name, ref in graph.outputs.items()},
        "params": {name: _tensor_ref_to_dict(ref) for name, ref in graph.params.items()},
        "intermediates": {name: _tensor_ref_to_dict(ref) for name, ref in graph.intermediates.items()},
        "save": graph.save_list,
        "recompute": graph.recompute_list,
        "operations": [
            {
                "id": str(op.id) if op.id is not None else None,
                "name": op.name,
                "kernel_type": op.kernel_type,
                "inputs": op.inputs,
                "outputs": op.outputs,
                "attrs": op.attrs,
            }
            for op in graph.nodes
        ],
    }


def _activation_slot_ir_to_dict(slot: ActivationSlotIR) -> Dict[str, Any]:
    """Convert ActivationSlotIR to JSON-serializable dict."""
    result = {
        "name": slot.name,
        "scope": slot.scope,
        "shape": slot.shape,
        "slot_index": slot.slot_index,
    }
    # Only include optional fields if they have non-default values
    if slot.dtype:
        result["dtype"] = slot.dtype
    if slot.aliases:
        result["aliases"] = slot.aliases
    if slot.memory_hint != "persistent":
        result["memory_hint"] = slot.memory_hint
    if slot.shares_with:
        result["shares_with"] = slot.shares_with
    if slot.save_for_backward:
        result["save_for_backward"] = True
    if slot.recompute_in_backward:
        result["recompute_in_backward"] = True
    if slot.recompute_from:
        result["recompute_from"] = list(slot.recompute_from)
    if slot.recompute_op:
        result["recompute_op"] = slot.recompute_op
    if slot.recompute_attrs:
        result["recompute_attrs"] = slot.recompute_attrs
    if slot.recompute_policy and slot.recompute_policy != "always":
        result["recompute_policy"] = slot.recompute_policy
    if slot.recompute_group:
        result["recompute_group"] = slot.recompute_group
    if slot.recompute_outputs:
        result["recompute_outputs"] = list(slot.recompute_outputs)
    if slot.lora_targets:
        result["lora_targets"] = list(slot.lora_targets)
    if slot.share_policy and slot.share_policy != "when_recomputed":
        result["share_policy"] = slot.share_policy
    if slot.gradient_of:
        result["gradient_of"] = slot.gradient_of
    if slot.alias_of:
        result["alias_of"] = slot.alias_of
    if slot.condition:
        result["condition"] = slot.condition
    if slot.description:
        result["description"] = slot.description
    return result


def _activation_layout_ir_to_dict(layout: ActivationLayoutIR) -> Dict[str, Any]:
    """Convert ActivationLayoutIR to JSON-serializable dict."""
    result = {
        "name": layout.name,
        "slots": [_activation_slot_ir_to_dict(s) for s in layout.slots],
    }
    if layout.gradient_slots:
        result["gradient_slots"] = [_activation_slot_ir_to_dict(s) for s in layout.gradient_slots]
    if layout.alias_map:
        result["alias_map"] = layout.alias_map
    if layout.extends:
        result["extends"] = layout.extends
    return result


def _module_ir_to_dict(ir: ModuleIR) -> Dict[str, Any]:
    """Convert ModuleIR to JSON-serializable dict."""
    result = {
        "name": ir.name,
        "kind": ir.kind,
        "extends": ir.extends,
        "config": ir.config,
        "hf_config": ir.hf_config,
        "hf_mapping": ir.hf_weight_mapping,
        "params": {name: _tensor_ref_to_dict(ref) for name, ref in ir.params.items()},
    }

    if ir.hf_export_mapping:
        result["hf_export"] = ir.hf_export_mapping

    if ir.forward_graph:
        result["forward"] = _graph_ir_to_dict(ir.forward_graph)

    if ir.backward_graph:
        result["backward"] = _graph_ir_to_dict(ir.backward_graph)

    if ir.save_tensors:
        result["save"] = ir.save_tensors

    if ir.recompute_tensors:
        result["recompute"] = ir.recompute_tensors

    if ir.activation_layout:
        result["activation_layout"] = _activation_layout_ir_to_dict(ir.activation_layout)

    return result


# =============================================================================
# Public API
# =============================================================================


def get_model_spec(name: str) -> Optional[ModelSpec]:
    """Get a registered model spec by name."""
    return _model_registry.get(name)


def get_block_spec(name: str) -> Optional[BlockSpec]:
    """Get a registered block spec by name."""
    return _block_registry.get(name)


def get_module_spec(name: str) -> Optional[ModuleSpec]:
    """Get a registered module spec by name."""
    return _module_registry.get(name)


def compile_model(
    model_class_or_name: type | str,
    config: Dict[str, Any],
    *,
    raise_on_error: bool = False,
    warnings: WarningCollector | None = None,
) -> str:
    """
    Compile a Python DSL model to JSON IR.

    Args:
        model_class_or_name: Either a decorated model class or its name
        config: Configuration parameters (e.g., from HuggingFace config.json)

    Returns:
        JSON string in the format expected by the C++ runtime
    """
    diag = warnings or WarningCollector()
    source_file = "python:<unknown>"

    try:
        # Get the spec
        if isinstance(model_class_or_name, str):
            spec = get_model_spec(model_class_or_name)
            if spec is None:
                raise DSLUndefinedError(model_class_or_name)
        else:
            if not hasattr(model_class_or_name, "_dsl_spec"):
                raise DSLError(
                    ErrorCode.E008,
                    f"class '{model_class_or_name.__name__}' is not a DSL model",
                    hint="Decorate the class with @model (and ensure it is imported so it registers).",
                )
            spec = model_class_or_name._dsl_spec
            if not isinstance(spec, ModelSpec):
                raise DSLError(
                    ErrorCode.E008,
                    f"class '{model_class_or_name.__name__}' is not a DSL model",
                    hint="Decorate the class with @model.",
                )
        source_file = f"python:{spec.name}"

        # Reset offload group resolver for fresh compilation
        _offload_resolver.reset()

        # Compile
        ir = compile_model_spec(spec, config, warnings=diag)

        # Serialize
        result: Dict[str, Any] = {
            "source_file": source_file,
            "success": True,
            "modules": [_module_ir_to_dict(ir)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)

    except DSLError as e:
        if raise_on_error:
            raise
        result = {
            "source_file": source_file,
            "success": False,
            "modules": [],
            "errors": [_dsl_error_to_dict(e)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)
    except Exception as e:
        wrapped = DSLSyntaxError(f"unhandled compiler error: {e}")
        if raise_on_error:
            raise wrapped from e
        result = {
            "source_file": source_file,
            "success": False,
            "modules": [],
            "errors": [_dsl_error_to_dict(wrapped)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)


def compile_model_for_hf(
    architecture: str,
    hf_config: Dict[str, Any],
    *,
    extra_config: Dict[str, Any] | None = None,
    raise_on_error: bool = False,
    warnings: WarningCollector | None = None,
) -> str:
    """
    Compile a model matching the HuggingFace architecture.

    Args:
        architecture: HuggingFace architecture name (e.g., "Qwen3ForCausalLM")
        hf_config: The HuggingFace config.json contents
        extra_config: Additional config overrides from training config (e.g., ep_size).
                      These are merged into the config dict after HF config extraction,
                      allowing training-time parameters to influence graph compilation.

    Returns:
        JSON string in the format expected by the C++ runtime
    """
    diag = warnings or WarningCollector()

    # Find model spec by architecture
    spec = None
    model_name = None

    for name, model_spec in _model_registry.items():
        if model_spec.hf_config:
            if model_spec.hf_config.architecture == architecture:
                spec = model_spec
                model_name = name
                break
            if model_spec.hf_config.model_type == architecture:
                spec = model_spec
                model_name = name
                break

    if spec is None:
        err = DSLUndefinedError(
            architecture,
            hint="Ensure the model is registered and has @hf_config(architecture=...).",
        )
        if raise_on_error:
            raise err
        result = {
            "source_file": f"python:<hf:{architecture}>",
            "success": False,
            "modules": [],
            "errors": [_dsl_error_to_dict(err)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)

    # Build config from HF config using the mapping
    def _get_hf_value(config_dict: Dict[str, Any], key: str) -> Any | None:
        if key in config_dict:
            return config_dict[key]
        if "." not in key:
            return None
        cur: Any = config_dict
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    config = {}
    if spec.hf_config:
        for dsl_param, hf_key in spec.hf_config.param_mapping.items():
            value = _get_hf_value(hf_config, hf_key)
            if value is not None:
                config[dsl_param] = value

    # Merge training-time overrides (e.g., ep_size) into the config.
    # These are not part of the HF model config but affect graph compilation.
    if extra_config:
        config.update(extra_config)

    try:
        # Compile
        ir = compile_model_spec(spec, config, warnings=diag)

        # Serialize
        result: Dict[str, Any] = {
            "source_file": f"python:{spec.name}",
            "success": True,
            "modules": [_module_ir_to_dict(ir)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)
    except DSLError as e:
        if raise_on_error:
            raise
        result = {
            "source_file": f"python:{spec.name}",
            "success": False,
            "modules": [],
            "errors": [_dsl_error_to_dict(e)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)
    except Exception as e:
        wrapped = DSLSyntaxError(f"unhandled compiler error: {e}")
        if raise_on_error:
            raise wrapped from e
        result = {
            "source_file": f"python:{spec.name}",
            "success": False,
            "modules": [],
            "errors": [_dsl_error_to_dict(wrapped)],
        }
        if diag.warnings:
            result["warnings"] = [_dsl_warning_to_dict(w) for w in diag.warnings]
        return json.dumps(result)


def get_hf_param_mapping(architecture: str) -> tuple[Dict[str, str], str]:
    """
    Get the HuggingFace parameter mapping for an architecture.

    Args:
        architecture: HuggingFace architecture name

    Returns:
        Tuple of (param_mapping dict, model_name)
    """
    for name, spec in _model_registry.items():
        if spec.hf_config:
            if spec.hf_config.architecture == architecture:
                return spec.hf_config.param_mapping, name
            if spec.hf_config.model_type == architecture:
                return spec.hf_config.param_mapping, name

    raise DSLUndefinedError(architecture)


def list_registered_models() -> List[str]:
    """List all registered model names."""
    return list(_model_registry.keys())


def list_registered_blocks() -> List[str]:
    """List all registered block names."""
    return list(_block_registry.keys())


def list_registered_modules() -> List[str]:
    """List all registered module names."""
    return list(_module_registry.keys())
