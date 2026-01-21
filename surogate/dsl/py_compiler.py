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
)
from .graph_builder import GraphBuilder, GraphNode, GraphRef
from .hf import FuseMapping, SplitMapping, TransformMapping

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


# =============================================================================
# Compiler Implementation
# =============================================================================


def _parse_shape_dim(dim: Any) -> Any:
    """Convert a shape dimension to IR format."""
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str):
        return dim
    return str(dim)


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


def _param_spec_to_ref(spec: ParamSpec, config: Dict[str, Any]) -> TensorRef:
    """Convert a ParamSpec to a TensorRef."""
    shape = []

    # Handle ARRAY params (e.g., blocks: Array["n_layers", "BlockType"])
    if spec.kind == ParamKind.ARRAY and spec.array_size:
        # Array size is the first dimension
        size_dim = spec.array_size
        if isinstance(size_dim, str):
            # Resolve symbolic dim from config
            if size_dim in config:
                shape.append(config[size_dim])
            else:
                shape.append(size_dim)
        else:
            shape.append(size_dim)
    elif spec.shape:
        # Regular tensor params
        for dim in spec.shape:
            if isinstance(dim, str):
                # Resolve symbolic dims from config
                if dim in config:
                    shape.append(config[dim])
                else:
                    shape.append(dim)
            else:
                shape.append(dim)

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
    return str(spec)


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

    # Params
    for name, param_spec in spec.params.items():
        # Check condition
        if param_spec.condition:
            # Create mock instance to evaluate condition
            try:
                mock = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and spec.python_class:
            try:
                # Create instance
                instance = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(instance, key, value)

                # Initialize derived values (ignore extra config keys)
                _init_instance_from_config(instance, spec.python_class, config)

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

    # Params
    for name, param_spec in spec.params.items():
        # Check condition
        if param_spec.condition:
            try:
                mock = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and spec.python_class:
            try:
                instance = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(instance, key, value)

                _init_instance_from_config(instance, spec.python_class, config)

                # Capture the graph by patching graph()
                builder = _capture_forward_graph(forward_fn, instance, spec.forward.inputs)

                if builder:
                    ir.forward_graph = _compile_graph_builder(builder, spec.forward, config, spec.params)
                else:
                    ir.forward_graph = GraphIR()

            except Exception:
                ir.forward_graph = GraphIR()

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

    # Params
    for name, param_spec in spec.params.items():
        if param_spec.condition:
            try:
                mock = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(mock, key, value)
                if not param_spec.condition(mock):
                    continue
            except Exception:
                pass

        ir.params[name] = _param_spec_to_ref(param_spec, config)

    # Forward graph
    if spec.forward:
        forward_fn = spec.forward.graph_fn
        if forward_fn and spec.python_class:
            try:
                instance = object.__new__(spec.python_class)
                for key, value in config.items():
                    setattr(instance, key, value)

                _init_instance_from_config(instance, spec.python_class, config)

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
