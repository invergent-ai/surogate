"""
Graph Builder for Python DSL

Provides a context manager and fluent API for building computation graphs.

Example:
    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=["B * T", "C"])
            y_flat = g.matmul(x_flat, self.weight, transpose="NT")
            y = g.view(y_flat, shape=["B", "T", "C"])
            return y
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Sequence, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .ir import GraphIR


class TransposeMode(str, Enum):
    """Transpose mode for matmul operations."""
    NN = "NN"
    NT = "NT"
    TN = "TN"
    TT = "TT"


@dataclass
class GraphNode:
    """A node in the computation graph."""
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalBranch:
    """A conditional branch in the graph."""
    condition: str  # Expression string
    true_nodes: list[GraphNode]
    false_nodes: list[GraphNode] | None = None


@dataclass
class GraphRef:
    """Reference to a tensor in the graph.

    This is returned by graph operations and can be used as input to other ops.
    """
    name: str
    builder: GraphBuilder

    def __repr__(self) -> str:
        return f"GraphRef({self.name!r})"


class GraphBuilder:
    """Builder for constructing computation graphs.

    Use within a graph() context manager to build dataflow graphs.
    """

    def __init__(self):
        self.nodes: list[GraphNode | ConditionalBranch] = []
        self._name_counter: int = 0
        self._inputs: list[str] = []
        self._outputs: list[str] = []
        self._save_list: list[str] = []
        self._recompute_list: list[str] = []
        self._condition_stack: list[list[GraphNode]] = []

    def _fresh_name(self, prefix: str = "t") -> str:
        """Generate a unique tensor name."""
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def _resolve_input(self, inp: str | GraphRef) -> str:
        """Resolve an input to a tensor name."""
        if isinstance(inp, GraphRef):
            return inp.name
        return inp

    def _add_node(self, node: GraphNode) -> None:
        """Add a node to the current scope (handles conditionals)."""
        if self._condition_stack:
            self._condition_stack[-1].append(node)
        else:
            self.nodes.append(node)

    def _make_output(self, name: str) -> GraphRef:
        """Create a GraphRef for an output tensor."""
        return GraphRef(name=name, builder=self)

    def _make_outputs(self, names: list[str]) -> tuple[GraphRef, ...]:
        """Create GraphRefs for multiple output tensors."""
        return tuple(GraphRef(name=n, builder=self) for n in names)

    # =========================================================================
    # Input/Output Registration
    # =========================================================================

    def input(self, name: str) -> GraphRef:
        """Register an input tensor."""
        self._inputs.append(name)
        return GraphRef(name=name, builder=self)

    def output(self, ref: str | GraphRef, name: str | None = None) -> None:
        """Register an output tensor."""
        tensor_name = self._resolve_input(ref)
        self._outputs.append(name or tensor_name)

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    def matmul(
        self,
        a: str | GraphRef,
        b: str | GraphRef,
        *,
        transpose: str | TransposeMode = "NN",
        accumulate: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
        out_name: str | None = None,
    ) -> GraphRef:
        """Matrix multiplication: C = alpha * op(A) @ op(B) + beta * C"""
        out = out_name if out_name else self._fresh_name("mm")
        self._add_node(GraphNode(
            op="matmul",
            inputs=[self._resolve_input(a), self._resolve_input(b)],
            outputs=[out],
            attrs={
                "transpose": str(transpose),
                "accumulate": accumulate,
                "alpha": alpha,
                "beta": beta,
            },
        ))
        return self._make_output(out)

    def batched_matmul(
        self,
        a: str | GraphRef,
        b: str | GraphRef,
        *,
        transpose: str | TransposeMode = "NN",
    ) -> GraphRef:
        """Batched matrix multiplication."""
        out = self._fresh_name("bmm")
        self._add_node(GraphNode(
            op="batched_matmul",
            inputs=[self._resolve_input(a), self._resolve_input(b)],
            outputs=[out],
            attrs={"transpose": str(transpose)},
        ))
        return self._make_output(out)

    # =========================================================================
    # Normalization
    # =========================================================================

    def rmsnorm(
        self,
        x: str | GraphRef,
        weight: str | GraphRef,
        *,
        eps: float = 1e-6,
    ) -> tuple[GraphRef, GraphRef]:
        """RMS normalization. Returns (y, rstd)."""
        y = self._fresh_name("rms")
        rstd = self._fresh_name("rstd")
        self._add_node(GraphNode(
            op="rmsnorm",
            inputs=[self._resolve_input(x), self._resolve_input(weight)],
            outputs=[y, rstd],
            attrs={"eps": eps},
        ))
        return self._make_outputs([y, rstd])

    def fused_residual_rmsnorm(
        self,
        residual: str | GraphRef,
        x: str | GraphRef,
        weight: str | GraphRef,
        *,
        eps: float = 1e-6,
        res_out_name: str | None = None,
        y_name: str | None = None,
        rstd_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """Fused residual add + RMS norm. Returns (residual_out, y, rstd)."""
        res_out = res_out_name if res_out_name else self._fresh_name("res")
        y = y_name if y_name else self._fresh_name("rms")
        rstd = rstd_name if rstd_name else self._fresh_name("rstd")
        self._add_node(GraphNode(
            op="fused_residual_rmsnorm",
            inputs=[
                self._resolve_input(residual),
                self._resolve_input(x),
                self._resolve_input(weight),
            ],
            outputs=[res_out, y, rstd],
            attrs={"eps": eps},
        ))
        return self._make_outputs([res_out, y, rstd])

    def layernorm(
        self,
        x: str | GraphRef,
        weight: str | GraphRef,
        bias: str | GraphRef | None = None,
        *,
        eps: float = 1e-5,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """Layer normalization. Returns (y, mean, rstd)."""
        y = self._fresh_name("ln")
        mean = self._fresh_name("mean")
        rstd = self._fresh_name("rstd")
        inputs = [self._resolve_input(x), self._resolve_input(weight)]
        if bias is not None:
            inputs.append(self._resolve_input(bias))
        self._add_node(GraphNode(
            op="layernorm",
            inputs=inputs,
            outputs=[y, mean, rstd],
            attrs={"eps": eps},
        ))
        return self._make_outputs([y, mean, rstd])

    # =========================================================================
    # Activations
    # =========================================================================

    def swiglu(self, x: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """SwiGLU activation: silu(gate) * up"""
        out = out_name if out_name else self._fresh_name("swiglu")
        self._add_node(GraphNode(
            op="swiglu",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def silu(self, x: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """SiLU (Swish) activation."""
        out = out_name if out_name else self._fresh_name("silu")
        self._add_node(GraphNode(
            op="silu",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def relu(self, x: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """ReLU activation."""
        out = out_name if out_name else self._fresh_name("relu")
        self._add_node(GraphNode(
            op="relu",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def relu2(self, x: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """ReLU squared activation."""
        out = out_name if out_name else self._fresh_name("relu2")
        self._add_node(GraphNode(
            op="relu2",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def gelu(self, x: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """GELU activation."""
        out = out_name if out_name else self._fresh_name("gelu")
        self._add_node(GraphNode(
            op="gelu",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def softmax(self, x: str | GraphRef, *, dim: int = -1) -> GraphRef:
        """Softmax activation."""
        out = self._fresh_name("softmax")
        self._add_node(GraphNode(
            op="softmax",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"dim": dim},
        ))
        return self._make_output(out)

    def silu_mul(self, gate: str | GraphRef, up: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """SiLU(gate) * up activation."""
        out = out_name if out_name else self._fresh_name("silu_mul")
        self._add_node(GraphNode(
            op="silu_mul",
            inputs=[self._resolve_input(gate), self._resolve_input(up)],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Attention
    # =========================================================================

    def flash_attention(
        self,
        qkv: str | GraphRef,
        *,
        causal: bool = True,
        softmax_scale: float | None = None,
        window_size: int | None = None,
        out_name: str | None = None,
        lse_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """FlashAttention. Returns (out, lse)."""
        out = out_name if out_name else self._fresh_name("attn")
        lse = lse_name if lse_name else self._fresh_name("lse")
        attrs = {"causal": causal}
        if softmax_scale is not None:
            attrs["softmax_scale"] = softmax_scale
        if window_size is not None:
            attrs["window_size"] = window_size
        self._add_node(GraphNode(
            op="flash_attention",
            inputs=[self._resolve_input(qkv)],
            outputs=[out, lse],
            attrs=attrs,
        ))
        return self._make_outputs([out, lse])

    def flash_attention_qkv(
        self,
        q: str | GraphRef,
        k: str | GraphRef,
        v: str | GraphRef,
        *,
        causal: bool = True,
        softmax_scale: float | None = None,
        out_name: str | None = None,
        lse_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """FlashAttention with separate Q, K, V. Returns (out, lse)."""
        out = out_name if out_name else self._fresh_name("attn")
        lse = lse_name if lse_name else self._fresh_name("lse")
        attrs = {"causal": causal}
        if softmax_scale is not None:
            attrs["softmax_scale"] = softmax_scale
        self._add_node(GraphNode(
            op="flash_attention_qkv",
            inputs=[
                self._resolve_input(q),
                self._resolve_input(k),
                self._resolve_input(v),
            ],
            outputs=[out, lse],
            attrs=attrs,
        ))
        return self._make_outputs([out, lse])

    def rope(
        self,
        qkv: str | GraphRef,
        freqs: str | GraphRef,
        position_ids: str | GraphRef,
        *,
        rotary_dim: int | str | None = None,
        out_name: str | None = None,
    ) -> GraphRef:
        """Apply rotary position embedding."""
        out = out_name if out_name else self._fresh_name("rope")
        attrs = {}
        if rotary_dim is not None:
            attrs["rotary_dim"] = rotary_dim
        self._add_node(GraphNode(
            op="rope",
            inputs=[
                self._resolve_input(qkv),
                self._resolve_input(freqs),
                self._resolve_input(position_ids),
            ],
            outputs=[out],
            attrs=attrs,
        ))
        return self._make_output(out)

    def qkv_qk_norm_rope(
        self,
        qkv: str | GraphRef,
        q_norm_weight: str | GraphRef,
        k_norm_weight: str | GraphRef,
        freqs: str | GraphRef,
        position_ids: str | GraphRef,
        *,
        eps: float = 1e-6,
        out_name: str | None = None,
        q_rstd_name: str | None = None,
        k_rstd_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """Fused QK norm + RoPE. Returns (qkv_out, q_rstd, k_rstd)."""
        qkv_out = out_name if out_name else self._fresh_name("qkv_rope")
        q_rstd = q_rstd_name if q_rstd_name else self._fresh_name("q_rstd")
        k_rstd = k_rstd_name if k_rstd_name else self._fresh_name("k_rstd")
        self._add_node(GraphNode(
            op="qkv_qk_norm_rope",
            inputs=[
                self._resolve_input(qkv),
                self._resolve_input(q_norm_weight),
                self._resolve_input(k_norm_weight),
                self._resolve_input(freqs),
                self._resolve_input(position_ids),
            ],
            outputs=[qkv_out, q_rstd, k_rstd],
            attrs={"eps": eps},
        ))
        return self._make_outputs([qkv_out, q_rstd, k_rstd])

    # =========================================================================
    # Tensor Manipulation
    # =========================================================================

    def view(
        self,
        x: str | GraphRef,
        *,
        shape: Sequence[str | int],
        out_name: str | None = None,
    ) -> GraphRef:
        """Reshape tensor."""
        out = out_name if out_name else self._fresh_name("view")
        self._add_node(GraphNode(
            op="view",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"shape": list(shape)},
        ))
        return self._make_output(out)

    def transpose(
        self,
        x: str | GraphRef,
        *,
        dim0: int = 0,
        dim1: int = 1,
    ) -> GraphRef:
        """Transpose two dimensions."""
        out = self._fresh_name("transpose")
        self._add_node(GraphNode(
            op="transpose",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"dim0": dim0, "dim1": dim1},
        ))
        return self._make_output(out)

    def permute(self, x: str | GraphRef, *, dims: Sequence[int]) -> GraphRef:
        """Permute dimensions."""
        out = self._fresh_name("permute")
        self._add_node(GraphNode(
            op="permute",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"dims": list(dims)},
        ))
        return self._make_output(out)

    def contiguous(self, x: str | GraphRef) -> GraphRef:
        """Make tensor contiguous."""
        out = self._fresh_name("contiguous")
        self._add_node(GraphNode(
            op="contiguous",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    def split(
        self,
        x: str | GraphRef,
        *,
        split_size: int | Sequence[int],
        dim: int = 0,
    ) -> tuple[GraphRef, ...]:
        """Split tensor along dimension."""
        if isinstance(split_size, int):
            # Will determine number of outputs at runtime
            num_outputs = 2  # Default assumption
        else:
            num_outputs = len(split_size)

        outputs = [self._fresh_name("split") for _ in range(num_outputs)]
        self._add_node(GraphNode(
            op="split",
            inputs=[self._resolve_input(x)],
            outputs=outputs,
            attrs={"split_size": split_size, "dim": dim},
        ))
        return self._make_outputs(outputs)

    def concat(
        self,
        *tensors: str | GraphRef,
        dim: int = 0,
    ) -> GraphRef:
        """Concatenate tensors along dimension."""
        out = self._fresh_name("concat")
        self._add_node(GraphNode(
            op="concat",
            inputs=[self._resolve_input(t) for t in tensors],
            outputs=[out],
            attrs={"dim": dim},
        ))
        return self._make_output(out)

    def copy(self, x: str | GraphRef) -> GraphRef:
        """Copy tensor."""
        out = self._fresh_name("copy")
        self._add_node(GraphNode(
            op="copy",
            inputs=[self._resolve_input(x)],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Elementwise Operations
    # =========================================================================

    def add(self, a: str | GraphRef, b: str | GraphRef) -> GraphRef:
        """Element-wise addition."""
        out = self._fresh_name("add")
        self._add_node(GraphNode(
            op="add",
            inputs=[self._resolve_input(a), self._resolve_input(b)],
            outputs=[out],
        ))
        return self._make_output(out)

    def mul(self, a: str | GraphRef, b: str | GraphRef) -> GraphRef:
        """Element-wise multiplication."""
        out = self._fresh_name("mul")
        self._add_node(GraphNode(
            op="mul",
            inputs=[self._resolve_input(a), self._resolve_input(b)],
            outputs=[out],
        ))
        return self._make_output(out)

    def scale(self, x: str | GraphRef, *, factor: float) -> GraphRef:
        """Scale tensor by constant."""
        out = self._fresh_name("scale")
        self._add_node(GraphNode(
            op="scale",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"factor": factor},
        ))
        return self._make_output(out)

    def bias_add(self, x: str | GraphRef, bias: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """Add bias to tensor."""
        out = out_name if out_name else self._fresh_name("bias")
        self._add_node(GraphNode(
            op="bias_add",
            inputs=[self._resolve_input(x), self._resolve_input(bias)],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Embedding
    # =========================================================================

    def embedding(
        self,
        indices: str | GraphRef,
        weight: str | GraphRef,
    ) -> GraphRef:
        """Embedding lookup."""
        out = self._fresh_name("embed")
        self._add_node(GraphNode(
            op="embedding",
            inputs=[self._resolve_input(indices), self._resolve_input(weight)],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Initialization
    # =========================================================================

    def zeros(
        self,
        *,
        shape: Sequence[str | int],
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create zero-filled tensor."""
        out = self._fresh_name("zeros")
        self._add_node(GraphNode(
            op="zeros",
            inputs=[],
            outputs=[out],
            attrs={"shape": list(shape), "dtype": dtype},
        ))
        return self._make_output(out)

    def ones(
        self,
        *,
        shape: Sequence[str | int],
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create one-filled tensor."""
        out = self._fresh_name("ones")
        self._add_node(GraphNode(
            op="ones",
            inputs=[],
            outputs=[out],
            attrs={"shape": list(shape), "dtype": dtype},
        ))
        return self._make_output(out)

    def fill(
        self,
        *,
        shape: Sequence[str | int],
        value: float,
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create tensor filled with value."""
        out = self._fresh_name("fill")
        self._add_node(GraphNode(
            op="fill",
            inputs=[],
            outputs=[out],
            attrs={"shape": list(shape), "value": value, "dtype": dtype},
        ))
        return self._make_output(out)

    # =========================================================================
    # MoE Operations
    # =========================================================================

    def moe_softmax(self, logits: str | GraphRef) -> GraphRef:
        """MoE router softmax."""
        out = self._fresh_name("moe_probs")
        self._add_node(GraphNode(
            op="moe_softmax",
            inputs=[self._resolve_input(logits)],
            outputs=[out],
        ))
        return self._make_output(out)

    def moe_sigmoid(self, logits: str | GraphRef) -> GraphRef:
        """MoE router sigmoid."""
        out = self._fresh_name("moe_probs")
        self._add_node(GraphNode(
            op="moe_sigmoid",
            inputs=[self._resolve_input(logits)],
            outputs=[out],
        ))
        return self._make_output(out)

    def moe_topk(
        self,
        probs: str | GraphRef,
        *,
        top_k: int,
        normalize: bool = True,
    ) -> tuple[GraphRef, GraphRef]:
        """MoE top-k selection. Returns (weights, indices)."""
        weights = self._fresh_name("moe_weights")
        indices = self._fresh_name("moe_indices")
        self._add_node(GraphNode(
            op="moe_topk",
            inputs=[self._resolve_input(probs)],
            outputs=[weights, indices],
            attrs={"top_k": top_k, "normalize": normalize},
        ))
        return self._make_outputs([weights, indices])

    def moe_permute(
        self,
        x: str | GraphRef,
        indices: str | GraphRef,
        *,
        top_k: int,
    ) -> GraphRef:
        """MoE input permutation."""
        out = self._fresh_name("moe_permuted")
        self._add_node(GraphNode(
            op="moe_permute",
            inputs=[self._resolve_input(x), self._resolve_input(indices)],
            outputs=[out],
            attrs={"top_k": top_k},
        ))
        return self._make_output(out)

    def moe_unpermute(
        self,
        x: str | GraphRef,
        weights: str | GraphRef,
        indices: str | GraphRef,
        *,
        top_k: int,
    ) -> GraphRef:
        """MoE output unpermutation and combination."""
        out = self._fresh_name("moe_combined")
        self._add_node(GraphNode(
            op="moe_unpermute",
            inputs=[
                self._resolve_input(x),
                self._resolve_input(weights),
                self._resolve_input(indices),
            ],
            outputs=[out],
            attrs={"top_k": top_k},
        ))
        return self._make_output(out)

    def moe_grouped_gemm(
        self,
        x: str | GraphRef,
        weights: str | GraphRef,
        offsets: str | GraphRef,
    ) -> GraphRef:
        """MoE grouped GEMM."""
        out = self._fresh_name("moe_gemm")
        self._add_node(GraphNode(
            op="moe_grouped_gemm",
            inputs=[
                self._resolve_input(x),
                self._resolve_input(weights),
                self._resolve_input(offsets),
            ],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Custom Operations
    # =========================================================================

    def custom(
        self,
        op_name: str,
        *inputs: str | GraphRef,
        num_outputs: int = 1,
        **attrs: Any,
    ) -> GraphRef | tuple[GraphRef, ...]:
        """Call a custom/user-defined operation."""
        outputs = [self._fresh_name(op_name) for _ in range(num_outputs)]
        self._add_node(GraphNode(
            op=op_name,
            inputs=[self._resolve_input(i) for i in inputs],
            outputs=outputs,
            attrs=attrs,
        ))
        if num_outputs == 1:
            return self._make_output(outputs[0])
        return self._make_outputs(outputs)

    def call(
        self,
        module_name: str,
        *inputs: str | GraphRef,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> GraphRef | tuple[GraphRef, ...]:
        """Call a submodule.

        This generates an operation with:
        - op (name): the module name directly (e.g., "StackedBlocks")
        - kernel_type set to "custom" (handled in attrs via _kernel_type)
        """
        outputs = [self._fresh_name(module_name) for _ in range(num_outputs)]
        # Set _kernel_type to "custom" for module calls
        attrs = dict(kwargs)
        attrs["_kernel_type"] = "custom"
        self._add_node(GraphNode(
            op=module_name,  # Use module name directly, not "call:module_name"
            inputs=[self._resolve_input(i) for i in inputs],
            outputs=outputs,
            attrs=attrs,
        ))
        if num_outputs == 1:
            return self._make_output(outputs[0])
        return self._make_outputs(outputs)

    # =========================================================================
    # Memory Directives
    # =========================================================================

    def save(self, *refs: str | GraphRef) -> None:
        """Mark tensors to save for backward pass."""
        for ref in refs:
            self._save_list.append(self._resolve_input(ref))

    def mark_recompute(self, *refs: str | GraphRef) -> None:
        """Mark tensors to recompute in backward pass."""
        for ref in refs:
            self._recompute_list.append(self._resolve_input(ref))

    # =========================================================================
    # Annotation Helpers
    # =========================================================================

    def annotate(self, ref: GraphRef, **annotations: Any) -> GraphRef:
        """Add annotations to the last operation that produced this ref."""
        # Find the node that produced this tensor
        for node in reversed(self.nodes):
            if isinstance(node, GraphNode) and ref.name in node.outputs:
                node.annotations.update(annotations)
                break
        return ref

    # =========================================================================
    # Saved Tensor Access
    # =========================================================================

    def saved(self, name: str) -> GraphRef:
        """Access a tensor saved from forward pass (for backward)."""
        return GraphRef(name=f"saved.{name}", builder=self)


# Global stack for nested graph contexts
_graph_stack: list[GraphBuilder] = []


@contextmanager
def graph():
    """Context manager for building computation graphs.

    Example:
        @forward
        def forward(self, x):
            with graph() as g:
                y = g.matmul(x, self.weight, transpose="NT")
                return y
    """
    builder = GraphBuilder()
    _graph_stack.append(builder)
    try:
        yield builder
    finally:
        _graph_stack.pop()


def current_graph() -> GraphBuilder | None:
    """Get the current graph builder, if any."""
    return _graph_stack[-1] if _graph_stack else None
