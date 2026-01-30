"""
Python DSL Compiler

Compiles Python DSL model/block/module classes to IR JSON format compatible
with the C++ runtime.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .decorators import _model_registry, _block_registry, _module_registry
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
from .hf import FuseMapping, SplitMapping, TransformMapping, StackExpertsMapping
from .dim import Dim, DimExpr, ConcreteDimValue, dim_to_ir

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
    gradient_of: Optional[str] = None
    slot_index: int = -1  # Index in the activation struct
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
    )


def _serialize_hf_spec(spec: Any) -> Any:
    """Serialize an HF weight mapping spec to JSON-compatible dict."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, FuseMapping):
        payload = {
            "type": "fuse",
            "sources": list(spec.sources),
        }
        if spec.dim != 0:
            payload["dim"] = spec.dim
        return payload
    if isinstance(spec, SplitMapping):
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
        payload = {
            "type": "transform",
            "source": spec.source,
        }
        if spec.fn:
            payload["fn"] = spec.fn
        return payload
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
        payload = {
            "type": "stack_experts",
            "pattern": spec.pattern,
        }
        if spec.num_experts > 0:
            payload["num_experts"] = spec.num_experts
        if spec.fuse_gate_up:
            payload["fuse_gate_up"] = spec.fuse_gate_up
        return payload
    return str(spec)


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
        gradient_of=slot.gradient_of,
        slot_index=slot_index,
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


def _compile_graph_builder(
    builder: GraphBuilder,
    spec: ForwardSpec,
    config: Dict[str, Any],
    params: Dict[str, ParamSpec],
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
        graph.params[name] = _param_spec_to_ref(param_spec, config)

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
    graph.save_list = builder._save_list
    graph.recompute_list = builder._recompute_list

    return graph


def _init_instance_from_config(instance: Any, cls: type, config: Dict[str, Any]) -> None:
    """Initialize instance with config keys accepted by __init__."""
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
    except Exception:
        pass


def _inline_stacked_blocks(
    graph: GraphIR,
    model_spec: ModelSpec,
    config: Dict[str, Any],
) -> GraphIR:
    """Inline StackedBlocks calls into per-layer block graphs."""
    # Quick check: if no StackedBlocks present, return as-is
    if not any(node.name == "StackedBlocks" for node in graph.nodes):
        return graph

    # Resolve block param spec from model
    def _resolve_block_spec(blocks_param: str) -> BlockSpec:
        param_spec = model_spec.params.get(blocks_param)
        if not param_spec or param_spec.kind != ParamKind.ARRAY or not param_spec.element_type:
            raise ValueError(f"StackedBlocks expects array param '{blocks_param}' with element_type")
        block_spec = get_block_spec(param_spec.element_type)
        if block_spec is None:
            raise ValueError(f"Block not found: {param_spec.element_type}")
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
        if node.name != "StackedBlocks":
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

        blocks_param = node.attrs.get("blocks", "blocks")
        n_layers = node.attrs.get("n_layers") or config.get("n_layers")
        if n_layers is None:
            raise ValueError("StackedBlocks missing n_layers")

        block_spec = _resolve_block_spec(blocks_param)
        block_ir = _get_block_ir(block_spec)
        if not block_ir.forward_graph or not block_ir.forward_graph.nodes:
            raise ValueError(f"Block graph missing for {block_spec.name}")

        block_graph = block_ir.forward_graph
        block_inputs = list(block_graph.inputs.keys())
        block_outputs = list(block_graph.outputs.keys())
        block_params = list(block_graph.params.keys())

        # StackedBlocks inputs are (x, residual, position_ids)
        cur_inputs = list(node.inputs)
        if len(cur_inputs) != len(block_inputs):
            raise ValueError(
                f"StackedBlocks input mismatch: expected {len(block_inputs)}, got {len(cur_inputs)}"
            )

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

            # Next layer inputs
            cur_inputs = list(layer_outputs[:len(block_inputs) - 1])
            cur_inputs.append(node.inputs[-1])  # position_ids stays constant

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
        try:
            returned_outputs = forward_fn(instance, *mock_inputs)
        except Exception:
            pass

        # Store the returned output tensor names on the builder for later use
        if captured_builder is not None and returned_outputs is not None:
            from .graph_builder import GraphRef
            if isinstance(returned_outputs, GraphRef):
                captured_builder._returned_outputs = [returned_outputs.name]
            elif isinstance(returned_outputs, tuple):
                captured_builder._returned_outputs = [
                    ref.name if isinstance(ref, GraphRef) else str(ref)
                    for ref in returned_outputs
                ]
            else:
                captured_builder._returned_outputs = []

        return captured_builder
    finally:
        # Restore originals
        graph_module.graph = original_graph
        for mod, attr, orig in modules_to_patch:
            setattr(mod, attr, orig)


def compile_model_spec(
    spec: ModelSpec,
    config: Dict[str, Any],
) -> ModuleIR:
    """Compile a ModelSpec to ModuleIR."""
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
            try:
                # Capture the graph by patching graph()
                builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)

                if builder:
                    graph = _compile_graph_builder(builder, spec.forward, config, spec.params)
                    # Expand StackedBlocks into per-layer block graphs
                    graph = _inline_stacked_blocks(graph, spec, config)
                    ir.forward_graph = graph
                else:
                    ir.forward_graph = GraphIR()

            except Exception as e:
                # If graph construction fails, create empty graph
                ir.forward_graph = GraphIR()

    # Save/recompute
    if spec.forward:
        ir.save_tensors = spec.forward.save
        ir.recompute_tensors = spec.forward.recompute

    # Compile activation layout - merge model's global activations with block activations
    ir.activation_layout = _compile_merged_activation_layout(spec, config, dim_map)

    return ir


def compile_block_spec(
    spec: BlockSpec,
    config: Dict[str, Any],
) -> ModuleIR:
    """Compile a BlockSpec to ModuleIR."""
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
            try:
                # Capture the graph by patching graph()
                builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)

                if builder:
                    ir.forward_graph = _compile_graph_builder(builder, spec.forward, config, spec.params)
                else:
                    ir.forward_graph = GraphIR()

            except Exception:
                ir.forward_graph = GraphIR()

    # Compile activation layout if present
    if spec.activations:
        ir.activation_layout = _compile_activation_layout(spec.activations, config, dim_map)

    return ir


def compile_module_spec(
    spec: ModuleSpec,
    config: Dict[str, Any],
) -> ModuleIR:
    """Compile a ModuleSpec to ModuleIR."""
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
            try:
                # Capture the graph by patching graph()
                builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)

                if builder:
                    ir.forward_graph = _compile_graph_builder(builder, spec.forward, config, spec.params)
                else:
                    ir.forward_graph = GraphIR()

            except Exception:
                ir.forward_graph = GraphIR()

    return ir


# =============================================================================
# Serialization
# =============================================================================


def _tensor_ref_to_dict(ref: TensorRef) -> Dict[str, Any]:
    """Convert TensorRef to JSON-serializable dict."""
    return {
        "shape": ref.shape,
        "dtype": ref.dtype,
        "is_param": ref.is_param,
        "is_input": ref.is_input,
        "is_output": ref.is_output,
    }


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
    if slot.gradient_of:
        result["gradient_of"] = slot.gradient_of
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
) -> str:
    """
    Compile a Python DSL model to JSON IR.

    Args:
        model_class_or_name: Either a decorated model class or its name
        config: Configuration parameters (e.g., from HuggingFace config.json)

    Returns:
        JSON string in the format expected by the C++ runtime
    """
    # Get the spec
    if isinstance(model_class_or_name, str):
        spec = get_model_spec(model_class_or_name)
        if spec is None:
            raise ValueError(f"Model not found: {model_class_or_name}")
    else:
        if not hasattr(model_class_or_name, "_dsl_spec"):
            raise ValueError(f"Class {model_class_or_name.__name__} is not a DSL model")
        spec = model_class_or_name._dsl_spec
        if not isinstance(spec, ModelSpec):
            raise ValueError(f"Class {model_class_or_name.__name__} is not a model")

    # Compile
    ir = compile_model_spec(spec, config)

    # Serialize
    result = {
        "source_file": f"python:{spec.name}",
        "success": True,
        "modules": [_module_ir_to_dict(ir)],
    }

    return json.dumps(result)


def compile_model_for_hf(
    architecture: str,
    hf_config: Dict[str, Any],
) -> str:
    """
    Compile a model matching the HuggingFace architecture.

    Args:
        architecture: HuggingFace architecture name (e.g., "Qwen3ForCausalLM")
        hf_config: The HuggingFace config.json contents

    Returns:
        JSON string in the format expected by the C++ runtime
    """
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
        raise ValueError(f"No Python DSL model found for architecture: {architecture}")

    # Build config from HF config using the mapping
    config = {}
    if spec.hf_config:
        for dsl_param, hf_key in spec.hf_config.param_mapping.items():
            if hf_key in hf_config and hf_config[hf_key] is not None:
                config[dsl_param] = hf_config[hf_key]

    # Compile
    ir = compile_model_spec(spec, config)

    # Serialize
    result = {
        "source_file": f"python:{spec.name}",
        "success": True,
        "modules": [_module_ir_to_dict(ir)],
    }

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

    raise ValueError(f"No Python DSL model found for architecture: {architecture}")


def list_registered_models() -> List[str]:
    """List all registered model names."""
    return list(_model_registry.keys())


def list_registered_blocks() -> List[str]:
    """List all registered block names."""
    return list(_block_registry.keys())


def list_registered_modules() -> List[str]:
    """List all registered module names."""
    return list(_module_registry.keys())
