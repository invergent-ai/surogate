"""
Lowering from Python DSL Specs to IR

Converts ModuleSpec, BlockSpec, ModelSpec, and PrimitiveSpec objects
into the existing GraphIR/ModuleIR format for execution.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

from .specs import (
    ModuleSpec,
    BlockSpec,
    ModelSpec,
    PrimitiveSpec,
    ParamSpec,
    ParamKind,
    ForwardSpec,
    BackwardSpec,
    IOSpec,
    HFConfigSpec,
    HFMappingSpec,
)
from .graph_builder import GraphBuilder, GraphNode, GraphRef
from .tensor_type import TensorAnnotation
from .ir import (
    ModuleIR,
    GraphIR,
    OpNode,
    TensorRef,
    KernelType,
    CompilationContext,
    PrimitiveSpec as IRPrimitiveSpec,
)
from .types import (
    Dtype,
    Shape,
    TensorTypeSpec,
    SymbolicDim,
    ConcreteDim,
    MemoryMode,
)


def _tensor_annotation_to_ref(
    name: str,
    ann: TensorAnnotation,
    is_param: bool = False,
    is_input: bool = False,
    is_output: bool = False,
) -> TensorRef:
    """Convert a TensorAnnotation to a TensorRef."""
    dtype = Dtype.from_string(ann.dtype) if ann.dtype else Dtype.BF16
    type_spec = ann.to_type_spec()
    return TensorRef(
        name=name,
        dtype=dtype,
        shape=type_spec.shape,
        is_param=is_param,
        is_input=is_input,
        is_output=is_output,
    )


def _kernel_type_from_op(op_name: str) -> KernelType:
    """Map operation name to KernelType enum."""
    op_map = {
        "matmul": KernelType.MATMUL,
        "matmul_bias": KernelType.MATMUL_BIAS,
        "batched_matmul": KernelType.BATCHED_MATMUL,
        "rmsnorm": KernelType.RMSNORM,
        "fused_residual_rmsnorm": KernelType.FUSED_RESIDUAL_RMSNORM,
        "layernorm": KernelType.LAYERNORM,
        "swiglu": KernelType.SWIGLU,
        "silu": KernelType.SILU,
        "relu": KernelType.RELU,
        "relu2": KernelType.RELU2,
        "gelu": KernelType.GELU,
        "softmax": KernelType.SOFTMAX,
        "flash_attention": KernelType.FLASH_ATTENTION,
        "flash_attention_qkv": KernelType.FLASH_ATTENTION,
        "rope": KernelType.ROPE,
        "qkv_qk_norm_rope": KernelType.QK_NORM,
        "view": KernelType.VIEW,
        "transpose": KernelType.TRANSPOSE,
        "permute": KernelType.PERMUTE,
        "contiguous": KernelType.CONTIGUOUS,
        "split": KernelType.SPLIT,
        "concat": KernelType.CONCAT,
        "copy": KernelType.COPY,
        "add": KernelType.ADD,
        "mul": KernelType.MUL,
        "scale": KernelType.SCALE,
        "bias_add": KernelType.ADD,
        "embedding": KernelType.EMBEDDING,
        "zeros": KernelType.ZEROS,
        "ones": KernelType.ONES,
        "fill": KernelType.FILL,
        "moe_softmax": KernelType.SOFTMAX,
        "moe_sigmoid": KernelType.CUSTOM,
        "moe_topk": KernelType.MOE_ROUTER,
        "moe_permute": KernelType.MOE_PERMUTE,
        "moe_unpermute": KernelType.MOE_UNPERMUTE,
        "moe_grouped_gemm": KernelType.GROUPED_GEMM,
        "silu_mul": KernelType.SILU,
    }
    return op_map.get(op_name, KernelType.CUSTOM)


class SpecLowerer:
    """Lowers Python DSL specs to IR."""

    def __init__(self):
        self.ctx = CompilationContext()
        self._node_counter = 0

    def _new_node_id(self) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"

    def lower_module(self, spec: ModuleSpec) -> ModuleIR:
        """Lower a ModuleSpec to ModuleIR."""
        ir = ModuleIR(
            name=spec.name,
            config=dict(spec.constructor_params),
            is_block=False,
            is_model=False,
            extends=spec.extends,
        )

        # Lower parameters
        for param_name, param_spec in spec.params.items():
            tensor_ref = self._lower_param(param_name, param_spec)
            if tensor_ref:
                ir.params.append(tensor_ref)

        # Lower forward graph
        if spec.forward:
            ir.forward_graph = self._lower_forward(spec.name, spec.forward)

        # Lower backward graph
        if spec.backward:
            ir.backward_graph = self._lower_backward(spec.name, spec.backward)

        return ir

    def lower_block(self, spec: BlockSpec) -> ModuleIR:
        """Lower a BlockSpec to ModuleIR."""
        ir = ModuleIR(
            name=spec.name,
            config=dict(spec.constructor_params),
            is_block=True,
            is_model=False,
            extends=spec.extends,
        )

        # Lower parameters
        for param_name, param_spec in spec.params.items():
            tensor_ref = self._lower_param(param_name, param_spec)
            if tensor_ref:
                ir.params.append(tensor_ref)

        # Lower forward graph
        if spec.forward:
            ir.forward_graph = self._lower_forward(spec.name, spec.forward)

        # Lower backward graph
        if spec.backward:
            ir.backward_graph = self._lower_backward(spec.name, spec.backward)

        return ir

    def lower_model(self, spec: ModelSpec) -> ModuleIR:
        """Lower a ModelSpec to ModuleIR."""
        ir = ModuleIR(
            name=spec.name,
            config=dict(spec.constructor_params),
            is_block=False,
            is_model=True,
        )

        # Lower parameters
        for param_name, param_spec in spec.params.items():
            tensor_ref = self._lower_param(param_name, param_spec)
            if tensor_ref:
                ir.params.append(tensor_ref)

        # Lower forward graph
        if spec.forward:
            ir.forward_graph = self._lower_forward(spec.name, spec.forward)

        # Lower backward graph
        if spec.backward:
            ir.backward_graph = self._lower_backward(spec.name, spec.backward)

        # HuggingFace mappings
        if spec.hf_config:
            ir.hf_config_mapping = self._lower_hf_config(spec.hf_config)
        if spec.hf_mapping:
            ir.hf_weight_mapping = self._lower_hf_mapping(spec.hf_mapping)
        if spec.hf_export:
            ir.hf_export_mapping = self._lower_hf_mapping(spec.hf_export)

        return ir

    def _lower_param(self, name: str, spec: ParamSpec) -> TensorRef | None:
        """Lower a ParamSpec to TensorRef."""
        if spec.kind == ParamKind.TENSOR:
            if spec.shape is None:
                return None

            # Build shape
            dims = []
            for d in spec.shape:
                if isinstance(d, int):
                    dims.append(ConcreteDim(d))
                else:
                    dims.append(SymbolicDim(str(d)))

            dtype = Dtype.from_string(spec.dtype) if spec.dtype else Dtype.BF16

            return TensorRef(
                name=name,
                dtype=dtype,
                shape=Shape(dims),
                is_param=True,
            )

        elif spec.kind == ParamKind.ARRAY:
            # Arrays are handled specially - return a marker
            return TensorRef(
                name=name,
                dtype=Dtype.AUTO,
                shape=None,
                is_param=True,
            )

        elif spec.kind == ParamKind.TIED:
            # Tied params reference another param
            return TensorRef(
                name=name,
                dtype=Dtype.AUTO,
                shape=None,
                is_param=True,
            )

        return None

    def _lower_forward(self, module_name: str, spec: ForwardSpec) -> GraphIR:
        """Lower a ForwardSpec to GraphIR."""
        graph = GraphIR(name=f"{module_name}_forward")

        # Add inputs
        for io_spec in spec.inputs:
            ref = _tensor_annotation_to_ref(
                io_spec.name,
                io_spec.tensor_type,
                is_input=True,
            )
            graph.inputs[io_spec.name] = ref

        # Add outputs
        for io_spec in spec.outputs:
            ref = _tensor_annotation_to_ref(
                io_spec.name,
                io_spec.tensor_type,
                is_output=True,
            )
            graph.outputs[io_spec.name] = ref

        # The graph_fn contains the computation logic
        # We can't execute it here, but we can store metadata
        # In practice, the graph is built when the forward method is called
        # with a GraphBuilder context

        # Store save/recompute lists
        graph.save_list = spec.save
        graph.recompute_list = spec.recompute

        return graph

    def _lower_backward(self, module_name: str, spec: BackwardSpec) -> GraphIR:
        """Lower a BackwardSpec to GraphIR."""
        graph = GraphIR(name=f"{module_name}_backward")

        # Add gradient inputs
        for io_spec in spec.gradient_inputs:
            ref = _tensor_annotation_to_ref(
                io_spec.name,
                io_spec.tensor_type,
                is_input=True,
            )
            graph.inputs[io_spec.name] = ref

        # Add gradient outputs
        for io_spec in spec.gradient_outputs:
            ref = _tensor_annotation_to_ref(
                io_spec.name,
                io_spec.tensor_type,
                is_output=True,
            )
            graph.outputs[io_spec.name] = ref

        return graph

    def _lower_hf_config(self, spec: HFConfigSpec) -> dict[str, Any]:
        """Lower HFConfigSpec to dict."""
        return {
            "architecture": spec.architecture,
            "model_type": spec.model_type,
            "config_class": spec.config_class,
            "param_mapping": spec.param_mapping,
            **spec.extras,
        }

    def _lower_hf_mapping(self, spec: HFMappingSpec) -> dict[str, Any]:
        """Lower HFMappingSpec to dict."""
        from .hf import mapping_to_dict, is_hf_mapping_spec

        result = {}
        for internal_name, external_spec in spec.mappings.items():
            if is_hf_mapping_spec(external_spec):
                result[internal_name] = mapping_to_dict(external_spec)
            else:
                result[internal_name] = {"kind": "direct", "path": str(external_spec)}
        return result


def lower_graph_builder(builder: GraphBuilder, name: str) -> GraphIR:
    """Convert a GraphBuilder's nodes to GraphIR."""
    graph = GraphIR(name=name)
    node_counter = 0

    for node in builder.nodes:
        if isinstance(node, GraphNode):
            node_counter += 1
            op_node = OpNode(
                id=f"node_{node_counter}",
                kernel_type=_kernel_type_from_op(node.op),
                name=node.op,
                inputs=node.inputs,
                outputs=node.outputs,
                attrs=node.attrs,
            )

            # Handle annotations
            if "memory" in node.annotations:
                mode = node.annotations["memory"]
                if mode == "save":
                    op_node.memory_mode = MemoryMode.SAVE
                elif mode == "recompute":
                    op_node.memory_mode = MemoryMode.RECOMPUTE

            graph.nodes.append(op_node)

            # Track intermediates
            for out_name in node.outputs:
                if out_name not in graph.inputs and out_name not in graph.outputs:
                    graph.intermediates[out_name] = TensorRef(name=out_name)

    # Set save/recompute lists
    graph.save_list = builder._save_list
    graph.recompute_list = builder._recompute_list

    return graph


def lower_primitive(spec: PrimitiveSpec) -> IRPrimitiveSpec:
    """Lower a PrimitiveSpec to IR PrimitiveSpec."""
    from .ir import PrimitiveSpec as IRPrimSpec

    # Build input types
    input_types = {}
    if spec.forward_in and spec.forward_in.named_tensors:
        for name, ann in spec.forward_in.named_tensors.items():
            input_types[name] = ann.to_type_spec()

    # Build output types
    output_types = {}
    if spec.forward_out and spec.forward_out.named_tensors:
        for name, ann in spec.forward_out.named_tensors.items():
            output_types[name] = ann.to_type_spec()

    # Determine kernel type
    kernel_type = _kernel_type_from_op(spec.name)

    # Build default attrs from params
    default_attrs = {}
    for param_name, (type_hint, default) in spec.params.items():
        if default is not None:
            default_attrs[param_name] = default

    return IRPrimSpec(
        name=spec.name,
        kernel_type=kernel_type,
        input_types=input_types,
        output_types=output_types,
        default_attrs=default_attrs,
        save=spec.save,
        forward_impl=spec.forward_impl,
        backward_impl=spec.backward_impl,
    )


# Convenience functions
def lower_module_spec(spec: ModuleSpec) -> ModuleIR:
    """Lower a ModuleSpec to ModuleIR."""
    lowerer = SpecLowerer()
    return lowerer.lower_module(spec)


def lower_block_spec(spec: BlockSpec) -> ModuleIR:
    """Lower a BlockSpec to ModuleIR."""
    lowerer = SpecLowerer()
    return lowerer.lower_block(spec)


def lower_model_spec(spec: ModelSpec) -> ModuleIR:
    """Lower a ModelSpec to ModuleIR."""
    lowerer = SpecLowerer()
    return lowerer.lower_model(spec)
