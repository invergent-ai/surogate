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

from .dim import Dim, DimExpr, ConcreteDimValue

if TYPE_CHECKING:
    from .ir import GraphIR

# Type alias for shape dimensions
ShapeDim = str | int | Dim | DimExpr | ConcreteDimValue


def _resolve_shape(shape: Sequence[ShapeDim]) -> list[str | int]:
    """Convert shape dimensions to IR-compatible format."""
    result: list[str | int] = []
    for dim in shape:
        if isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, ConcreteDimValue):
            result.append(dim.value)
        elif isinstance(dim, (Dim, DimExpr)):
            result.append(dim.to_expr_string())
        else:
            result.append(str(dim))
    return result


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

    def matmul_bias(
        self,
        a: str | GraphRef,
        b: str | GraphRef,
        bias: str | GraphRef,
        *,
        transpose: str | TransposeMode = "NN",
        accumulate: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
        out_name: str | None = None,
    ) -> GraphRef:
        """Matrix multiplication with fused bias: C = alpha * op(A) @ op(B) + bias (+ beta * C)."""
        out = out_name if out_name else self._fresh_name("mm")
        self._add_node(GraphNode(
            op="matmul_bias",
            inputs=[self._resolve_input(a), self._resolve_input(b), self._resolve_input(bias)],
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
    # Fused Operations
    # =========================================================================

    def matmul_swiglu(
        self,
        a: str | GraphRef,
        b: str | GraphRef,
        *,
        transpose: str | TransposeMode = "NT",
        out_name: str | None = None,
        up_out_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """Fused matmul + SwiGLU activation.

        Computes: swiglu(A @ B^T) where B contains fused [up, gate] weights.
        Returns (activation_output, matmul_output_for_backward).

        The matmul output (up_out) is saved for backward pass computation.
        """
        out = out_name if out_name else self._fresh_name("mlp_act")
        up_out = up_out_name if up_out_name else self._fresh_name("mlp_up")
        self._add_node(GraphNode(
            op="matmul_swiglu",
            inputs=[self._resolve_input(a), self._resolve_input(b)],
            outputs=[out, up_out],
            attrs={"transpose": str(transpose)},
        ))
        return self._make_outputs([out, up_out])

    def fused_lm_head_loss(
        self,
        xF_flat: str | GraphRef,
        weight: str | GraphRef,
        targets: str | GraphRef,
        *,
        compute_accuracy: bool = False,
        out_name: str | None = None,
    ) -> GraphRef:
        """Fused LM head matmul + cross-entropy loss."""
        out = out_name if out_name else self._fresh_name("loss")
        self._add_node(GraphNode(
            op="fused_lm_head_loss",
            inputs=[
                self._resolve_input(xF_flat),
                self._resolve_input(weight),
                self._resolve_input(targets),
            ],
            outputs=[out],
            attrs={"compute_accuracy": compute_accuracy},
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
        sinks: str | GraphRef | None = None,
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
        inputs = [self._resolve_input(qkv)]
        if sinks is not None:
            inputs.append(self._resolve_input(sinks))
        self._add_node(GraphNode(
            op="flash_attention",
            inputs=inputs,
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

    def mrope(
        self,
        qkv: str | GraphRef,
        freqs: str | GraphRef,
        position_ids: str | GraphRef,
        *,
        rotary_dim: int | str | None = None,
        mrope_section: list[int] | tuple[int, int, int] | None = None,
        out_name: str | None = None,
    ) -> GraphRef:
        """Apply multimodal rotary position embedding (MRoPE)."""
        out = out_name if out_name else self._fresh_name("mrope")
        attrs = {}
        if rotary_dim is not None:
            attrs["rotary_dim"] = rotary_dim
        if mrope_section is not None:
            attrs["mrope_section"] = list(mrope_section)
        self._add_node(GraphNode(
            op="mrope",
            inputs=[
                self._resolve_input(qkv),
                self._resolve_input(freqs),
                self._resolve_input(position_ids),
            ],
            outputs=[out],
            attrs=attrs,
        ))
        return self._make_output(out)

    def qkv_qk_norm(
        self,
        qkv: str | GraphRef,
        q_norm_weight: str | GraphRef,
        k_norm_weight: str | GraphRef,
        *,
        eps: float = 1e-6,
        out_name: str | None = None,
        q_rstd_name: str | None = None,
        k_rstd_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """QK norm (no RoPE). Returns (qkv_out, q_rstd, k_rstd)."""
        qkv_out = out_name if out_name else self._fresh_name("qkv_norm")
        q_rstd = q_rstd_name if q_rstd_name else self._fresh_name("q_rstd")
        k_rstd = k_rstd_name if k_rstd_name else self._fresh_name("k_rstd")
        self._add_node(GraphNode(
            op="qkv_qk_norm",
            inputs=[
                self._resolve_input(qkv),
                self._resolve_input(q_norm_weight),
                self._resolve_input(k_norm_weight),
            ],
            outputs=[qkv_out, q_rstd, k_rstd],
            attrs={"eps": eps},
        ))
        return self._make_outputs([qkv_out, q_rstd, k_rstd])

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
        shape: Sequence[ShapeDim],
        out_name: str | None = None,
    ) -> GraphRef:
        """Reshape tensor."""
        out = out_name if out_name else self._fresh_name("view")
        self._add_node(GraphNode(
            op="view",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"shape": _resolve_shape(shape)},
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

    def add(self, a: str | GraphRef, b: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """Element-wise addition."""
        out = out_name or self._fresh_name("add")
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

    def mask_scatter(
        self,
        x: str | GraphRef,
        mask: str | GraphRef,
        src: str | GraphRef,
        *,
        out_name: str | None = None,
    ) -> GraphRef:
        """Replace rows in x at masked positions with src (visual embeddings)."""
        out = out_name if out_name else self._fresh_name("mask_scatter")
        self._add_node(GraphNode(
            op="mask_scatter",
            inputs=[self._resolve_input(x), self._resolve_input(mask), self._resolve_input(src)],
            outputs=[out],
        ))
        return self._make_output(out)

    def deepstack_inject(
        self,
        x: str | GraphRef,
        mask: str | GraphRef,
        src: str | GraphRef,
        *,
        out_name: str | None = None,
    ) -> GraphRef:
        """Add src to x at masked positions (deepstack visual embeddings)."""
        out = out_name if out_name else self._fresh_name("deepstack")
        self._add_node(GraphNode(
            op="deepstack_inject",
            inputs=[self._resolve_input(x), self._resolve_input(mask), self._resolve_input(src)],
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
        shape: Sequence[ShapeDim],
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create zero-filled tensor."""
        out = self._fresh_name("zeros")
        self._add_node(GraphNode(
            op="zeros",
            inputs=[],
            outputs=[out],
            attrs={"shape": _resolve_shape(shape), "dtype": dtype},
        ))
        return self._make_output(out)

    def ones(
        self,
        *,
        shape: Sequence[ShapeDim],
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create one-filled tensor."""
        out = self._fresh_name("ones")
        self._add_node(GraphNode(
            op="ones",
            inputs=[],
            outputs=[out],
            attrs={"shape": _resolve_shape(shape), "dtype": dtype},
        ))
        return self._make_output(out)

    def fill(
        self,
        *,
        shape: Sequence[ShapeDim],
        value: float,
        dtype: str = "bf16",
    ) -> GraphRef:
        """Create tensor filled with value."""
        out = self._fresh_name("fill")
        self._add_node(GraphNode(
            op="fill",
            inputs=[],
            outputs=[out],
            attrs={"shape": _resolve_shape(shape), "value": value, "dtype": dtype},
        ))
        return self._make_output(out)

    # =========================================================================
    # MoE Operations
    # =========================================================================

    def moe_softmax(self, logits: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """MoE router softmax."""
        out = out_name or self._fresh_name("moe_probs")
        self._add_node(GraphNode(
            op="moe_softmax",
            inputs=[self._resolve_input(logits)],
            outputs=[out],
        ))
        return self._make_output(out)

    def moe_sigmoid(self, logits: str | GraphRef, *, out_name: str | None = None) -> GraphRef:
        """MoE router sigmoid."""
        out = out_name or self._fresh_name("moe_probs")
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
        scaling_factor: float = 1.0,
        rounding_scale: float | None = None,
        sort_by_index: bool = False,
        softmax: bool | None = None,
        correction_bias: str | GraphRef | None = None,
        weights_name: str | None = None,
        indices_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """MoE top-k selection. Returns (weights, indices).

        Args:
            correction_bias: Optional per-expert bias for expert selection (e.g.,
                e_score_correction_bias). When provided, expert selection uses
                (score + bias) but routing weights use original unbiased scores.
        """
        weights = weights_name or self._fresh_name("moe_weights")
        indices = indices_name or self._fresh_name("moe_indices")
        attrs: dict = {"top_k": top_k, "normalize": normalize}
        if scaling_factor != 1.0:
            attrs["scaling_factor"] = scaling_factor
        if rounding_scale is not None and rounding_scale != 0.0:
            attrs["topk_rounding_scale"] = rounding_scale
        if sort_by_index:
            attrs["topk_sort_by_index"] = True
        if softmax is not None:
            attrs["softmax"] = softmax
        inputs = [self._resolve_input(probs)]
        if correction_bias is not None:
            inputs.append(self._resolve_input(correction_bias))
        self._add_node(GraphNode(
            op="moe_topk",
            inputs=inputs,
            outputs=[weights, indices],
            attrs=attrs,
        ))
        return self._make_outputs([weights, indices])

    def moe_permute(
        self,
        x: str | GraphRef,
        indices: str | GraphRef,
        *,
        top_k: int,
        out_name: str | None = None,
        scatter_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """MoE input permutation. Returns (permuted_input, scatter_indices)."""
        permuted = out_name or self._fresh_name("moe_permuted")
        scatter_indices = scatter_name or self._fresh_name("moe_scatter_indices")
        self._add_node(GraphNode(
            op="moe_permute",
            inputs=[self._resolve_input(x), self._resolve_input(indices)],
            outputs=[permuted, scatter_indices],
            attrs={"top_k": top_k},
        ))
        return self._make_outputs([permuted, scatter_indices])

    def moe_unpermute(
        self,
        x: str | GraphRef,
        weights: str | GraphRef,
        indices: str | GraphRef,
        *,
        top_k: int,
        out_name: str | None = None,
    ) -> GraphRef:
        """MoE output unpermutation and combination."""
        out = out_name or self._fresh_name("moe_combined")
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

    def moe_grouped_gemm_gate_up(
        self,
        x: str | GraphRef,
        weights: str,
        scatter_indices: str | GraphRef,
        gate_up_interleaved: bool | None = None,
        out_name: str | None = None,
    ) -> GraphRef:
        """MoE grouped GEMM for gate+up projection.

        Args:
            x: Permuted input tensor (total_tokens, hidden_size)
            weights: Parameter name for expert weights (num_experts, 2*intermediate, hidden_size)
            scatter_indices: Scatter indices from moe_permute (total_tokens,)

        Returns:
            Output tensor (total_tokens, 2*intermediate_size)
        """
        out = out_name or self._fresh_name("moe_gate_up")
        attrs = {}
        if gate_up_interleaved is not None:
            attrs["gate_up_interleaved"] = gate_up_interleaved

        self._add_node(GraphNode(
            op="moe_grouped_gemm_gate_up",
            inputs=[
                self._resolve_input(x),
                weights,  # Parameter name, not resolved
                self._resolve_input(scatter_indices),
            ],
            outputs=[out],
            attrs=attrs,
        ))
        return self._make_output(out)

    def moe_grouped_gemm_down(
        self,
        x: str | GraphRef,
        weights: str,
        scatter_indices: str | GraphRef,
        out_name: str | None = None,
    ) -> GraphRef:
        """MoE grouped GEMM for down projection.

        Args:
            x: Activated tensor after SwiGLU (total_tokens, intermediate_size)
            weights: Parameter name for expert weights (num_experts, hidden_size, intermediate_size)
            scatter_indices: Scatter indices from moe_permute (total_tokens,)

        Returns:
            Output tensor (total_tokens, hidden_size)
        """
        out = out_name or self._fresh_name("moe_down")
        self._add_node(GraphNode(
            op="moe_grouped_gemm_down",
            inputs=[
                self._resolve_input(x),
                weights,  # Parameter name, not resolved
                self._resolve_input(scatter_indices),
            ],
            outputs=[out],
        ))
        return self._make_output(out)

    def gpt_oss_moe_act(
        self,
        x: str | GraphRef,
        *,
        alpha: float = 1.702,
        limit: float = 7.0,
        out_name: str | None = None,
    ) -> GraphRef:
        """GPT-OSS gated activation (interleaved gate/up)."""
        out = out_name or self._fresh_name("gpt_oss_act")
        self._add_node(GraphNode(
            op="gpt_oss_moe_act",
            inputs=[self._resolve_input(x)],
            outputs=[out],
            attrs={"alpha": alpha, "limit": limit},
        ))
        return self._make_output(out)

    def moe_expert_bias_add(
        self,
        x: str | GraphRef,
        bias: str | GraphRef,
        *,
        out_name: str | None = None,
    ) -> GraphRef:
        """Add per-expert bias to permuted MoE activations."""
        out = out_name or self._fresh_name("moe_bias")
        self._add_node(GraphNode(
            op="moe_expert_bias_add",
            inputs=[self._resolve_input(x), self._resolve_input(bias)],
            outputs=[out],
        ))
        return self._make_output(out)

    # =========================================================================
    # Mamba2 / SSM Operations
    # =========================================================================

    def mamba_conv1d(
        self,
        x: str | GraphRef,
        weight: str | GraphRef,
        bias: str | GraphRef | None = None,
        *,
        activation: str = "silu",
        out_name: str | None = None,
    ) -> GraphRef:
        """Causal 1D convolution for Mamba.

        Args:
            x: Input tensor [B, D_conv, T]
            weight: Conv weight [D_conv, kernel_size]
            bias: Optional conv bias [D_conv]
            activation: Activation function ("silu" or "swish")

        Returns:
            Convolved output [B, D_conv, T]
        """
        out = out_name or self._fresh_name("mamba_conv")
        inputs = [self._resolve_input(x), self._resolve_input(weight)]
        if bias is not None:
            inputs.append(self._resolve_input(bias))
        self._add_node(GraphNode(
            op="mamba_conv1d",
            inputs=inputs,
            outputs=[out],
            attrs={"activation": activation},
        ))
        return self._make_output(out)

    def mamba_ssm_scan(
        self,
        hidden_states: str | GraphRef,
        dt: str | GraphRef,
        A_log: str | GraphRef,
        B: str | GraphRef,
        C: str | GraphRef,
        D: str | GraphRef,
        *,
        dt_bias: str | GraphRef | None = None,
        dt_softplus: bool = True,
        dt_min: float = 0.0,
        dt_max: float = 1e9,
        chunk_size: int = 256,
        num_heads: int = 0,
        head_dim: int = 0,
        ssm_state_size: int = 0,
        n_groups: int = 0,
        out_name: str | None = None,
        state_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef]:
        """Mamba2 State Space Model scan.

        Args:
            hidden_states: Input [B, I, T]
            dt: Time step [B, I, T]
            A_log: Log of state decay [H]
            B: Input-to-state [B, G, N, T]
            C: State-to-output [B, G, N, T]
            D: Skip connection [H]
            dt_bias: Time step bias [H]
            dt_softplus: Apply softplus to dt
            dt_min, dt_max: Time step clipping
            chunk_size: Chunk size for scan
            num_heads: Number of SSM heads
            head_dim: Dimension per head
            ssm_state_size: SSM state dimension (N/dstate)
            n_groups: Number of groups for B/C

        Returns:
            (output, final_state): SSM output and final state
        """
        out = out_name or self._fresh_name("ssm_out")
        state = state_name or self._fresh_name("ssm_state")
        inputs = [
            self._resolve_input(hidden_states),
            self._resolve_input(dt),
            self._resolve_input(A_log),
            self._resolve_input(B),
            self._resolve_input(C),
            self._resolve_input(D),
        ]
        if dt_bias is not None:
            inputs.append(self._resolve_input(dt_bias))
        self._add_node(GraphNode(
            op="mamba_ssm_scan",
            inputs=inputs,
            outputs=[out, state],
            attrs={
                "dt_softplus": dt_softplus,
                "dt_min": dt_min,
                "dt_max": dt_max,
                "chunk_size": chunk_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "ssm_state_size": ssm_state_size,
                "n_groups": n_groups,
                "intermediate_size": num_heads * head_dim,
            },
        ))
        return self._make_outputs([out, state])

    def mamba_gated_rmsnorm(
        self,
        x: str | GraphRef,
        gate: str | GraphRef,
        weight: str | GraphRef,
        *,
        eps: float = 1e-5,
        n_groups: int = 1,
        norm_before_gate: bool = False,
        out_name: str | None = None,
    ) -> GraphRef:
        """Gated RMSNorm for Mamba2.

        Args:
            x: Input tensor
            gate: Gate tensor
            weight: RMSNorm weight
            eps: Epsilon for numerical stability
            n_groups: Number of groups for normalization (1 = full dim)
            norm_before_gate: If True, normalize before gating

        Returns:
            Gated normalized output
        """
        out = out_name or self._fresh_name("gated_norm")
        self._add_node(GraphNode(
            op="mamba_gated_rmsnorm",
            inputs=[
                self._resolve_input(x),
                self._resolve_input(gate),
                self._resolve_input(weight),
            ],
            outputs=[out],
            attrs={
                "eps": eps,
                "n_groups": n_groups,
                "norm_before_gate": norm_before_gate,
            },
        ))
        return self._make_output(out)

    def mamba_split_proj(
        self,
        projected: str | GraphRef,
        *,
        intermediate_size: int,
        conv_dim: int,
        num_heads: int,
        head_dim: int,
        gate_name: str | None = None,
        conv_input_name: str | None = None,
        dt_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """Split Mamba2 input projection into gate, conv_input, dt.

        Args:
            projected: Input projection output [B, T, P]
            intermediate_size: Gate size
            conv_dim: Conv input size
            num_heads: Number of heads (dt size in projection)
            head_dim: Dimension per head (dt expanded from num_heads to num_heads * head_dim)

        Returns:
            (gate, conv_input, dt)
        """
        gate = gate_name or self._fresh_name("gate")
        conv_input = conv_input_name or self._fresh_name("conv_input")
        dt = dt_name or self._fresh_name("dt")
        self._add_node(GraphNode(
            op="mamba_split_proj",
            inputs=[self._resolve_input(projected)],
            outputs=[gate, conv_input, dt],
            attrs={
                "intermediate_size": intermediate_size,
                "conv_dim": conv_dim,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
        ))
        return self._make_outputs([gate, conv_input, dt])

    def mamba_split_conv_out(
        self,
        conv_out: str | GraphRef,
        *,
        intermediate_size: int,
        groups_state_size: int,
        n_groups: int = 0,
        ssm_state_size: int = 0,
        hidden_name: str | None = None,
        B_name: str | None = None,
        C_name: str | None = None,
    ) -> tuple[GraphRef, GraphRef, GraphRef]:
        """Split conv output into hidden_states, B, C.

        Args:
            conv_out: Conv output [B, conv_dim, T]
            intermediate_size: Hidden states size
            groups_state_size: n_groups * ssm_state_size
            n_groups: Number of SSM groups
            ssm_state_size: SSM state dimension

        Returns:
            (hidden_states, B, C)
        """
        hidden = hidden_name or self._fresh_name("hidden")
        B_out = B_name or self._fresh_name("ssm_B")
        C_out = C_name or self._fresh_name("ssm_C")
        self._add_node(GraphNode(
            op="mamba_split_conv_out",
            inputs=[self._resolve_input(conv_out)],
            outputs=[hidden, B_out, C_out],
            attrs={
                "intermediate_size": intermediate_size,
                "groups_state_size": groups_state_size,
                "n_groups": n_groups,
                "ssm_state_size": ssm_state_size,
            },
        ))
        return self._make_outputs([hidden, B_out, C_out])

    def mamba_combine_scan(
        self,
        projected_states: str | GraphRef,
        conv_weight: str | GraphRef,
        conv_bias: str | GraphRef | None,
        dt_bias: str | GraphRef,
        A_log: str | GraphRef,
        D: str | GraphRef,
        norm_weight: str | GraphRef,
        out_proj_weight: str | GraphRef,
        out_proj_bias: str | GraphRef | None,
        *,
        chunk_size: int = 256,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        intermediate_size: int,
        ssm_state_size: int,
        eps: float = 1e-5,
        activation: str = "silu",
        dt_min: float = 0.0,
        dt_max: float = 1e9,
        out_name: str | None = None,
    ) -> GraphRef:
        """Fused Mamba2 forward: conv + SSM scan + gated norm + out proj."""
        out = out_name or self._fresh_name("mamba_out")
        inputs = [
            self._resolve_input(projected_states),
            self._resolve_input(conv_weight),
        ]
        if conv_bias is not None:
            inputs.append(self._resolve_input(conv_bias))
        else:
            inputs.append("")  # Placeholder for optional bias
        inputs.extend([
            self._resolve_input(dt_bias),
            self._resolve_input(A_log),
            self._resolve_input(D),
            self._resolve_input(norm_weight),
            self._resolve_input(out_proj_weight),
        ])
        if out_proj_bias is not None:
            inputs.append(self._resolve_input(out_proj_bias))

        self._add_node(GraphNode(
            op="mamba_combine_scan",
            inputs=inputs,
            outputs=[out],
            attrs={
                "chunk_size": chunk_size,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "n_groups": n_groups,
                "intermediate_size": intermediate_size,
                "ssm_state_size": ssm_state_size,
                "eps": eps,
                "activation": activation,
                "dt_min": dt_min,
                "dt_max": dt_max,
            },
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
