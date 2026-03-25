"""
Subgraph Partitioner for Kernel Fusion (Inference Forward Pass)

Analyzes a compiled DSL IR graph and identifies fusion opportunities:
1. Fusible subgraphs: contiguous sequences of lightweight ops between anchors
2. Epilogue fusions: compute ops that could absorb into the preceding anchor
3. Prologue fusions: compute ops that could absorb into the following anchor
4. Router pipeline fusions: matmul → softmax/sigmoid → topk

Usage:
    from surogate.compiler.partitioner import analyze_model_ir
    report = analyze_model_ir(ir_json, verbose=True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Op Classification
# =============================================================================


class OpCategory(Enum):
    """How an operation behaves with respect to fusion."""
    ANCHOR = auto()        # Heavy kernel, not fusible (matmul, attention, MoE GEMM)
    POINTWISE = auto()     # Elementwise op, prime fusion target
    REDUCTION = auto()     # Reduction op (norm, softmax), fusible with adjacent pointwise
    METADATA = auto()      # Zero-cost shape manipulation (view, transpose)
    ROUTING = auto()       # MoE routing (permute/unpermute), complex scatter/gather
    COMMUNICATION = auto() # Distributed communication (ep_dispatch, ep_combine)
    INIT = auto()          # Tensor initialization (zeros, ones, fill)
    FUSED_ANCHOR = auto()  # Already a fused kernel (fused_residual_rmsnorm, qkv_qk_norm_rope)


# Maps kernel_type string -> category
OP_CLASSIFICATION: Dict[str, OpCategory] = {
    # Anchors: heavy compute kernels
    "matmul":                   OpCategory.ANCHOR,
    "matmul_bias":              OpCategory.ANCHOR,
    "matmul_swiglu":            OpCategory.FUSED_ANCHOR,
    "batched_matmul":           OpCategory.ANCHOR,
    "flash_attention":          OpCategory.ANCHOR,
    "fused_lm_head_loss":       OpCategory.FUSED_ANCHOR,
    "embedding":                OpCategory.ANCHOR,

    # Already-fused anchors
    "fused_residual_rmsnorm":   OpCategory.FUSED_ANCHOR,
    "qkv_qk_norm_rope":        OpCategory.FUSED_ANCHOR,
    "qkv_qk_norm":             OpCategory.FUSED_ANCHOR,
    "mamba_combine_scan":       OpCategory.FUSED_ANCHOR,
    "mamba_conv1d":             OpCategory.ANCHOR,
    "mamba_selective_scan":     OpCategory.ANCHOR,

    # MoE anchors (grouped GEMM)
    "moe_grouped_gemm":        OpCategory.ANCHOR,
    "moe_grouped_gemm_gate_up": OpCategory.ANCHOR,
    "moe_grouped_gemm_down":   OpCategory.ANCHOR,

    # Pointwise: prime fusion targets
    "swiglu":       OpCategory.POINTWISE,
    "silu":         OpCategory.POINTWISE,
    "silu_mul":     OpCategory.POINTWISE,
    "sigmoid":      OpCategory.POINTWISE,
    "relu":         OpCategory.POINTWISE,
    "relu2":        OpCategory.POINTWISE,
    "gelu":         OpCategory.POINTWISE,
    "geglu":        OpCategory.POINTWISE,
    "add":          OpCategory.POINTWISE,
    "mul":          OpCategory.POINTWISE,
    "scale":        OpCategory.POINTWISE,
    "add3":         OpCategory.POINTWISE,
    "bias_add":     OpCategory.POINTWISE,

    # Reductions: fusible with adjacent pointwise
    "rmsnorm":      OpCategory.REDUCTION,
    "layernorm":    OpCategory.REDUCTION,
    "softmax":      OpCategory.REDUCTION,
    "moe_softmax":  OpCategory.REDUCTION,
    "moe_sigmoid":  OpCategory.REDUCTION,
    "reduce_sum":   OpCategory.REDUCTION,
    "reduce_mean":  OpCategory.REDUCTION,
    "reduce_max":   OpCategory.REDUCTION,

    # Metadata: zero-cost shape manipulation
    "view":         OpCategory.METADATA,
    "transpose":    OpCategory.METADATA,
    "permute":      OpCategory.METADATA,
    "contiguous":   OpCategory.METADATA,
    "copy":         OpCategory.METADATA,
    "split":        OpCategory.METADATA,
    "concat":       OpCategory.METADATA,

    # Routing: complex scatter/gather
    "moe_topk":     OpCategory.ROUTING,
    "moe_permute":  OpCategory.ROUTING,
    "moe_unpermute": OpCategory.ROUTING,

    # Communication
    "ep_dispatch":  OpCategory.COMMUNICATION,
    "ep_combine":   OpCategory.COMMUNICATION,

    # Init
    "zeros":        OpCategory.INIT,
    "ones":         OpCategory.INIT,
    "fill":         OpCategory.INIT,
    "fill_normal":  OpCategory.INIT,

    # Misc
    "rope":         OpCategory.REDUCTION,
    "mrope":        OpCategory.REDUCTION,
    "repeat_interleave_heads": OpCategory.METADATA,
    "mask_scatter":  OpCategory.POINTWISE,
    "deepstack_inject": OpCategory.POINTWISE,

    # Linear attention (Qwen3.5)
    "chunk_gated_delta_rule": OpCategory.ANCHOR,
    "qwen3_5_decay":          OpCategory.ANCHOR,
    "gdn_fused_proj":         OpCategory.FUSED_ANCHOR,
}


def classify_op(kernel_type: str) -> OpCategory:
    """Classify an operation by its kernel type."""
    return OP_CLASSIFICATION.get(kernel_type, OpCategory.ANCHOR)


def is_anchor(cat: OpCategory) -> bool:
    """Returns True if this category breaks a fusible subgraph."""
    return cat in (OpCategory.ANCHOR, OpCategory.FUSED_ANCHOR, OpCategory.ROUTING, OpCategory.COMMUNICATION)


def is_fusible(cat: OpCategory) -> bool:
    """Returns True if this op can participate in a fused kernel."""
    return cat in (OpCategory.POINTWISE, OpCategory.REDUCTION, OpCategory.METADATA)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OpInfo:
    """Parsed operation from the IR."""
    idx: int
    name: str
    kernel_type: str
    category: OpCategory
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    layer_idx: Optional[int] = None

    @property
    def is_metadata_only(self) -> bool:
        return self.category == OpCategory.METADATA

    @property
    def is_pointwise(self) -> bool:
        return self.category == OpCategory.POINTWISE

    @property
    def is_reduction(self) -> bool:
        return self.category == OpCategory.REDUCTION


@dataclass
class FusibleSubgraph:
    """A contiguous sequence of ops that could be fused into a single kernel."""
    ops: List[OpInfo]
    preceding_anchor: Optional[OpInfo] = None
    following_anchor: Optional[OpInfo] = None

    @property
    def num_ops(self) -> int:
        return len(self.ops)

    @property
    def num_compute_ops(self) -> int:
        return sum(1 for op in self.ops if not op.is_metadata_only)

    @property
    def has_compute(self) -> bool:
        return self.num_compute_ops > 0

    @property
    def compute_kernel_types(self) -> List[str]:
        return [op.kernel_type for op in self.ops if not op.is_metadata_only]

    def summary(self) -> str:
        pre = self.preceding_anchor.kernel_type if self.preceding_anchor else "START"
        post = self.following_anchor.kernel_type if self.following_anchor else "END"
        types = " → ".join(op.kernel_type for op in self.ops)
        return f"{pre} | [{types}] | {post}"


@dataclass
class EpilogueFusion:
    """A compute op that could be absorbed into the preceding anchor as an epilogue."""
    anchor: OpInfo
    epilogue_ops: List[OpInfo]       # compute ops (may include intervening views)
    all_ops: List[OpInfo]            # all ops between anchor and next anchor/routing
    pattern_name: str                # human-readable name
    benefit: str                     # "high", "medium", "low"
    description: str


@dataclass
class PrologueFusion:
    """A compute op that could be absorbed into the following anchor as a prologue."""
    anchor: OpInfo
    prologue_ops: List[OpInfo]
    all_ops: List[OpInfo]
    pattern_name: str
    benefit: str
    description: str


# =============================================================================
# Partitioner
# =============================================================================


def _parse_ops(graph_dict: Dict[str, Any]) -> List[OpInfo]:
    """Parse operations from a graph IR dict."""
    ops = []
    for i, op_dict in enumerate(graph_dict.get("operations", [])):
        kt = op_dict.get("kernel_type", op_dict.get("name", "unknown"))
        cat = classify_op(kt)
        ops.append(OpInfo(
            idx=i,
            name=op_dict.get("name", ""),
            kernel_type=kt,
            category=cat,
            inputs=op_dict.get("inputs", []),
            outputs=op_dict.get("outputs", []),
            attrs=op_dict.get("attrs", {}),
        ))
    return ops


def _detect_layer_idx(op: OpInfo) -> Optional[int]:
    """Try to extract layer index from op names/attrs."""
    if "layer_idx" in op.attrs:
        return op.attrs["layer_idx"]
    for name in op.outputs + op.inputs:
        if "blocks[" in name:
            try:
                return int(name.split("blocks[")[1].split("]")[0])
            except (ValueError, IndexError):
                pass
    return None


def partition_graph(ops: List[OpInfo]) -> List[FusibleSubgraph]:
    """Partition ops into fusible subgraphs between anchor ops."""
    subgraphs: List[FusibleSubgraph] = []
    current_fusible: List[OpInfo] = []
    last_anchor: Optional[OpInfo] = None

    for op in ops:
        if is_anchor(op.category):
            if current_fusible:
                subgraphs.append(FusibleSubgraph(
                    ops=list(current_fusible),
                    preceding_anchor=last_anchor,
                    following_anchor=op,
                ))
                current_fusible = []
            last_anchor = op
        elif is_fusible(op.category):
            current_fusible.append(op)
        else:
            if current_fusible:
                subgraphs.append(FusibleSubgraph(
                    ops=list(current_fusible),
                    preceding_anchor=last_anchor,
                    following_anchor=op,
                ))
                current_fusible = []
            last_anchor = op

    if current_fusible:
        subgraphs.append(FusibleSubgraph(
            ops=list(current_fusible),
            preceding_anchor=last_anchor,
            following_anchor=None,
        ))

    return subgraphs


# =============================================================================
# Epilogue/Prologue Pattern Detection
# =============================================================================

# Known epilogue patterns: (preceding_anchor_type, [compute_ops_between]) -> pattern info
EPILOGUE_PATTERNS = {
    # matmul → (view →) swiglu: could use matmul_swiglu fusion
    ("matmul", ("swiglu",)):
        ("matmul_swiglu_epilogue", "high",
         "SwiGLU activation can be fused into matmul epilogue (matmul_swiglu already exists)"),

    # matmul → (view →) silu: could fuse activation into GEMM epilogue
    ("matmul", ("silu",)):
        ("matmul_silu_epilogue", "high",
         "SiLU activation can be fused into matmul epilogue via CUTLASS/Triton"),

    # matmul → (view →) relu2: could fuse activation into GEMM epilogue
    ("matmul", ("relu2",)):
        ("matmul_relu2_epilogue", "high",
         "ReLU² activation can be fused into matmul epilogue via CUTLASS/Triton"),

    # matmul → (view →) gelu: could fuse activation into GEMM epilogue
    ("matmul", ("gelu",)):
        ("matmul_gelu_epilogue", "high",
         "GELU activation can be fused into matmul epilogue via CUTLASS/Triton"),

    # matmul → (view →) rope: could fuse RoPE into QKV matmul epilogue
    ("matmul", ("rope",)):
        ("matmul_rope_epilogue", "medium",
         "RoPE could be fused into QKV projection epilogue (needs custom kernel)"),

    # moe_grouped_gemm_gate_up → swiglu: expert activation epilogue
    ("moe_grouped_gemm_gate_up", ("swiglu",)):
        ("moe_gemm_swiglu_epilogue", "medium",
         "SwiGLU could be fused into MoE grouped GEMM epilogue"),

    # moe_grouped_gemm → relu2: expert activation epilogue
    ("moe_grouped_gemm", ("relu2",)):
        ("moe_gemm_relu2_epilogue", "medium",
         "ReLU² could be fused into MoE grouped GEMM epilogue"),

    # moe_grouped_gemm → silu: expert activation epilogue
    ("moe_grouped_gemm", ("silu",)):
        ("moe_gemm_silu_epilogue", "medium",
         "SiLU could be fused into MoE grouped GEMM epilogue"),
}

# Known prologue patterns: ([compute_ops], following_anchor_type) -> pattern info
PROLOGUE_PATTERNS = {
    # moe_softmax → moe_topk: router pipeline
    (("moe_softmax",), "moe_topk"):
        ("router_softmax_topk", "medium",
         "Softmax + top-k selection can be fused into single router kernel"),

    # moe_sigmoid → moe_topk: router pipeline
    (("moe_sigmoid",), "moe_topk"):
        ("router_sigmoid_topk", "medium",
         "Sigmoid + top-k selection can be fused into single router kernel"),
}

# Shared-expert pattern: silu(gate) * up between matmuls
MULTI_OP_PATTERNS = {
    ("silu", "mul"):
        ("shared_expert_swiglu", "high",
         "silu(gate) * up is a SwiGLU pattern — fuse into single pointwise kernel"),

    ("relu2",):
        ("shared_expert_relu2", "low",
         "Standalone relu2 activation between matmuls"),
}


def detect_epilogue_fusions(ops: List[OpInfo]) -> List[EpilogueFusion]:
    """Detect ops that could be absorbed as epilogues into preceding anchors."""
    fusions: List[EpilogueFusion] = []

    i = 0
    while i < len(ops):
        op = ops[i]
        if not is_anchor(op.category):
            i += 1
            continue

        # Collect non-anchor ops after this anchor
        j = i + 1
        trailing: List[OpInfo] = []
        while j < len(ops) and is_fusible(ops[j].category):
            trailing.append(ops[j])
            j += 1

        if not trailing:
            i += 1
            continue

        # Extract compute ops (skip metadata)
        compute_ops = [t for t in trailing if not t.is_metadata_only]
        compute_types = tuple(t.kernel_type for t in compute_ops)

        # Check against known epilogue patterns
        key = (op.kernel_type, compute_types)
        if key in EPILOGUE_PATTERNS:
            name, benefit, desc = EPILOGUE_PATTERNS[key]
            fusions.append(EpilogueFusion(
                anchor=op,
                epilogue_ops=compute_ops,
                all_ops=trailing,
                pattern_name=name,
                benefit=benefit,
                description=desc,
            ))

        # Also check multi-op patterns among the compute ops
        if len(compute_types) >= 2:
            for pattern_types, (name, benefit, desc) in MULTI_OP_PATTERNS.items():
                if compute_types == pattern_types:
                    fusions.append(EpilogueFusion(
                        anchor=op,
                        epilogue_ops=compute_ops,
                        all_ops=trailing,
                        pattern_name=name,
                        benefit=benefit,
                        description=desc,
                    ))

        i += 1

    return fusions


def detect_prologue_fusions(ops: List[OpInfo]) -> List[PrologueFusion]:
    """Detect ops that could be absorbed as prologues into following anchors."""
    fusions: List[PrologueFusion] = []

    i = 0
    while i < len(ops):
        op = ops[i]
        if not is_fusible(op.category) or op.is_metadata_only:
            i += 1
            continue

        # Collect contiguous fusible ops starting from this compute op
        j = i
        leading: List[OpInfo] = []
        while j < len(ops) and is_fusible(ops[j].category):
            leading.append(ops[j])
            j += 1

        # Check if followed by an anchor
        if j < len(ops) and is_anchor(ops[j].category):
            following = ops[j]
            compute_ops = [l for l in leading if not l.is_metadata_only]
            compute_types = tuple(l.kernel_type for l in compute_ops)

            key = (compute_types, following.kernel_type)
            if key in PROLOGUE_PATTERNS:
                name, benefit, desc = PROLOGUE_PATTERNS[key]
                fusions.append(PrologueFusion(
                    anchor=following,
                    prologue_ops=compute_ops,
                    all_ops=leading,
                    pattern_name=name,
                    benefit=benefit,
                    description=desc,
                ))

        i = j if j > i else i + 1

    return fusions


# =============================================================================
# Full Analysis Report
# =============================================================================


@dataclass
class FusionReport:
    """Complete fusion analysis for an inference forward graph."""

    model_name: str
    total_ops: int
    ops_by_category: Dict[str, int]
    anchor_ops: int
    fusible_ops: int
    subgraphs: List[FusibleSubgraph]
    epilogue_fusions: List[EpilogueFusion]
    prologue_fusions: List[PrologueFusion]
    all_ops: List[OpInfo]
    cuda_graph_report: Optional[CudaGraphReport] = None

    layer_count: int = 0
    ops_per_layer: int = 0

    @property
    def subgraphs_with_compute(self) -> List[FusibleSubgraph]:
        return [sg for sg in self.subgraphs if sg.has_compute]

    @property
    def total_kernel_launches(self) -> int:
        """Non-metadata ops = actual kernel launches."""
        return sum(1 for op in self.all_ops
                   if op.category not in (OpCategory.METADATA, OpCategory.INIT))

    @property
    def epilogue_high(self) -> List[EpilogueFusion]:
        return [f for f in self.epilogue_fusions if f.benefit == "high"]

    @property
    def epilogue_medium(self) -> List[EpilogueFusion]:
        return [f for f in self.epilogue_fusions if f.benefit == "medium"]

    def print(self, verbose: bool = False):
        """Print the fusion analysis report."""
        print(f"\n{'='*76}")
        print(f"  INFERENCE FORWARD FUSION ANALYSIS: {self.model_name}")
        print(f"{'='*76}\n")

        print(f"  Total operations in graph:  {self.total_ops}")
        print(f"  Actual kernel launches:     {self.total_kernel_launches}")
        if self.layer_count > 0:
            print(f"  Layers:                     {self.layer_count}")
            print(f"  Ops per layer:              ~{self.ops_per_layer}")
        print()

        # ── CUDA Graph Analysis (most important section) ──
        cg = self.cuda_graph_report
        if cg:
            print(f"  ─── CUDA GRAPH SEGMENTATION (decode mode) ───\n")

            if cg.is_single_graph_possible:
                print(f"    FULL CUDA GRAPH POSSIBLE: Yes")
                print(f"    No graph-breaking ops detected. Entire decode step")
                print(f"    can be captured as a single CUDA graph.\n")
            else:
                print(f"    FULL CUDA GRAPH POSSIBLE: No")
                print(f"    Graph-breaking ops prevent single-graph capture.\n")
                print(f"    Breaking ops:")
                for op_type, count in sorted(cg.graph_breaking_ops.items(), key=lambda x: -x[1]):
                    print(f"      {op_type:<30s} {count:>4d} instances")
                print()

            print(f"    Total segments across all layers:  {cg.total_segments}")
            print(f"      Graphable (captured):            {cg.total_graphable_segments}")
            print(f"      Eager (not captured):            {cg.total_eager_segments}")
            print(f"    Segments per layer:                {cg.avg_segments_per_layer:.1f}")
            print()

            if cg.segments_per_layer and verbose:
                # Show one layer's structure
                layer0 = cg.segments_per_layer[0]
                print(f"    Layer 0 segment structure:")
                for i, seg in enumerate(layer0):
                    mode = "EAGER" if seg.eager else "GRAPH"
                    types = ", ".join(op.kernel_type for op in seg.ops[:5])
                    if len(seg.ops) > 5:
                        types += f", ... ({len(seg.ops)} total)"
                    print(f"      [{mode:>5s}] {seg.num_ops:>3d} ops: {types}")
                print()

            # Compare to vLLM approach
            print(f"    ─── COMPARISON ───")
            print(f"    vLLM FULL mode:     1 CUDA graph for entire decode step")
            print(f"    vLLM PIECEWISE:     {cg.total_graphable_segments} captured + {cg.total_eager_segments} eager segments")
            print(f"    Surogate current:   {cg.total_graphable_segments} captured + {cg.total_eager_segments} eager segments")
            print()
            if not cg.is_single_graph_possible:
                overhead_us = cg.total_segments * 8  # ~8us per segment transition
                print(f"    Estimated segment overhead: ~{overhead_us} us/step ({cg.total_segments} transitions x ~8us)")
                print(f"    To match vLLM FULL mode: need graph-safe decode attention kernel")
            print()

        print("  Operations by category:")
        for cat_name, count in sorted(self.ops_by_category.items(), key=lambda x: -x[1]):
            pct = count / self.total_ops * 100 if self.total_ops > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"    {cat_name:<20s} {count:>4d}  ({pct:5.1f}%)  {bar}")
        print()

        # ── Subgraph analysis ──
        compute_sgs = self.subgraphs_with_compute
        if compute_sgs:
            patterns: Dict[str, int] = {}
            for sg in compute_sgs:
                key = sg.summary()
                patterns[key] = patterns.get(key, 0) + 1

            print(f"  ─── FUSIBLE SUBGRAPHS (chains between anchors) ───")
            print(f"  Total: {len(compute_sgs)} subgraphs with compute\n")

            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
                print(f"    {count:>3d}x  {pattern}")
            print()

        # ── Epilogue fusions ──
        if self.epilogue_fusions:
            print(f"  ─── EPILOGUE FUSIONS (absorb op into preceding anchor) ───\n")

            # Group by pattern
            by_pattern: Dict[str, List[EpilogueFusion]] = {}
            for ef in self.epilogue_fusions:
                by_pattern.setdefault(ef.pattern_name, []).append(ef)

            for pattern_name, fusions in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
                f0 = fusions[0]
                per_layer = len(fusions) // max(self.layer_count, 1)
                print(f"    [{f0.benefit.upper():>6s}]  {pattern_name}")
                print(f"             {len(fusions)}x total ({per_layer}/layer)")
                print(f"             {f0.description}")
                if verbose:
                    types = " → ".join(op.kernel_type for op in f0.all_ops)
                    print(f"             Pattern: {f0.anchor.kernel_type} → {types}")
                print()

            # Kernel savings from epilogue fusions
            saved = sum(len(ef.epilogue_ops) for ef in self.epilogue_fusions)
            print(f"    Kernel launches saved by epilogue fusion: {saved}")
            print()

        # ── Prologue fusions ──
        if self.prologue_fusions:
            print(f"  ─── PROLOGUE FUSIONS (absorb op into following anchor) ───\n")

            by_pattern: Dict[str, List[PrologueFusion]] = {}
            for pf in self.prologue_fusions:
                by_pattern.setdefault(pf.pattern_name, []).append(pf)

            for pattern_name, fusions in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
                f0 = fusions[0]
                per_layer = len(fusions) // max(self.layer_count, 1)
                print(f"    [{f0.benefit.upper():>6s}]  {pattern_name}")
                print(f"             {len(fusions)}x total ({per_layer}/layer)")
                print(f"             {f0.description}")
                print()

            saved = sum(len(pf.prologue_ops) for pf in self.prologue_fusions)
            print(f"    Kernel launches saved by prologue fusion: {saved}")
            print()

        # ── Summary ──
        total_saved = (
            sum(len(ef.epilogue_ops) for ef in self.epilogue_fusions)
            + sum(len(pf.prologue_ops) for pf in self.prologue_fusions)
        )
        total_launches = self.total_kernel_launches
        remaining = total_launches - total_saved

        print(f"  ─── SUMMARY ───\n")
        print(f"    Current kernel launches:  {total_launches}")
        print(f"    Epilogue fusion saves:    {sum(len(ef.epilogue_ops) for ef in self.epilogue_fusions)}")
        print(f"    Prologue fusion saves:    {sum(len(pf.prologue_ops) for pf in self.prologue_fusions)}")
        print(f"    Remaining after fusion:   {remaining}")
        if total_launches > 0:
            pct = total_saved / total_launches * 100
            print(f"    Reduction:                {total_saved} launches ({pct:.1f}%)")
        print()

        # Per-layer view
        if self.layer_count > 0:
            per_layer_current = total_launches // self.layer_count
            per_layer_saved = total_saved // self.layer_count
            per_layer_remaining = per_layer_current - per_layer_saved
            print(f"    Per layer: {per_layer_current} → {per_layer_remaining} kernel launches")
            print()

        # Verdict
        high_count = len(self.epilogue_high)
        med_count = len(self.epilogue_medium) + len(self.prologue_fusions)

        print(f"{'='*76}")
        if high_count > 0:
            print(f"  VERDICT: {high_count} HIGH-value epilogue fusions available.")
            print(f"           These are the primary targets for a codegen backend.")
            per_layer = high_count // max(self.layer_count, 1)
            print(f"           Impact: ~{per_layer} kernel launches saved per layer.")
        elif med_count > 0:
            print(f"  VERDICT: {med_count} MEDIUM-value fusions available.")
        else:
            print(f"  VERDICT: Graph is already well-optimized for inference.")
            print(f"           No high-value fusion opportunities in the forward pass.")
        print(f"{'='*76}\n")


# =============================================================================
# CUDA Graph Segment Analysis
# =============================================================================

# Ops that break CUDA graph capture for decode (T=1)
GRAPH_BREAKING_OPS_DECODE = {
    "flash_attention",          # Dynamic cu_seqlens for doc masking
    # Note: ChunkGatedDeltaRule and Qwen3_5Decay are graph-safe in decode
}

# Ops that break CUDA graph capture in all modes
GRAPH_BREAKING_OPS_ALL = {
    "flash_attention",
    "chunk_gated_delta_rule",
    "qwen3_5_decay",
}


@dataclass
class CudaGraphSegment:
    """A contiguous segment of ops, either graphable or eager."""
    start_idx: int
    end_idx: int
    eager: bool             # True = must run eagerly (graph-breaking op)
    ops: List[OpInfo] = field(default_factory=list)

    @property
    def num_ops(self) -> int:
        return len(self.ops)

    @property
    def breaking_op(self) -> Optional[str]:
        if self.eager and self.ops:
            return self.ops[0].kernel_type
        return None


@dataclass
class CudaGraphReport:
    """Analysis of CUDA graph segmentation for decode inference."""
    segments_per_layer: List[List[CudaGraphSegment]]
    total_segments: int
    total_graphable_segments: int
    total_eager_segments: int
    graph_breaking_ops: Dict[str, int]  # op_type -> count

    @property
    def avg_segments_per_layer(self) -> float:
        if not self.segments_per_layer:
            return 0
        return self.total_segments / len(self.segments_per_layer)

    @property
    def is_single_graph_possible(self) -> bool:
        return self.total_eager_segments == 0


def compute_cuda_graph_segments(
    ops: List[OpInfo],
    layer_count: int,
    mode: str = "decode",
) -> CudaGraphReport:
    """
    Compute CUDA graph segments for the forward pass.

    Simulates the C++ compute_layer_segments() logic.
    """
    breaking_ops = GRAPH_BREAKING_OPS_DECODE if mode == "decode" else GRAPH_BREAKING_OPS_ALL

    # Detect layer boundaries
    layer_ops: Dict[int, List[OpInfo]] = {}
    global_ops: List[OpInfo] = []  # ops not assigned to a layer

    for op in ops:
        if op.layer_idx is not None:
            layer_ops.setdefault(op.layer_idx, []).append(op)
        else:
            global_ops.append(op)

    all_segments: List[List[CudaGraphSegment]] = []
    total_segs = 0
    total_graphable = 0
    total_eager = 0
    breaking_counts: Dict[str, int] = {}

    # For each layer, compute segments
    for layer_idx in sorted(layer_ops.keys()):
        lops = layer_ops[layer_idx]
        segments: List[CudaGraphSegment] = []
        current_graphable: List[OpInfo] = []

        for op in lops:
            is_breaking = op.kernel_type in breaking_ops

            if is_breaking:
                # Flush graphable segment
                if current_graphable:
                    segments.append(CudaGraphSegment(
                        start_idx=current_graphable[0].idx,
                        end_idx=current_graphable[-1].idx + 1,
                        eager=False,
                        ops=list(current_graphable),
                    ))
                    current_graphable = []

                # Add eager segment for this op
                segments.append(CudaGraphSegment(
                    start_idx=op.idx,
                    end_idx=op.idx + 1,
                    eager=True,
                    ops=[op],
                ))
                breaking_counts[op.kernel_type] = breaking_counts.get(op.kernel_type, 0) + 1
            else:
                current_graphable.append(op)

        # Flush trailing
        if current_graphable:
            segments.append(CudaGraphSegment(
                start_idx=current_graphable[0].idx,
                end_idx=current_graphable[-1].idx + 1,
                eager=False,
                ops=list(current_graphable),
            ))

        all_segments.append(segments)
        total_segs += len(segments)
        total_graphable += sum(1 for s in segments if not s.eager)
        total_eager += sum(1 for s in segments if s.eager)

    return CudaGraphReport(
        segments_per_layer=all_segments,
        total_segments=total_segs,
        total_graphable_segments=total_graphable,
        total_eager_segments=total_eager,
        graph_breaking_ops=breaking_counts,
    )


# =============================================================================
# Entry Points
# =============================================================================


def analyze_graph(graph_dict: Dict[str, Any], model_name: str = "unknown", mode: str = "decode") -> FusionReport:
    """Analyze a forward graph IR dict."""
    ops = _parse_ops(graph_dict)
    for op in ops:
        op.layer_idx = _detect_layer_idx(op)

    by_cat: Dict[str, int] = {}
    for op in ops:
        cat_name = op.category.name.lower()
        by_cat[cat_name] = by_cat.get(cat_name, 0) + 1

    anchor_count = sum(1 for op in ops if is_anchor(op.category))
    fusible_count = sum(1 for op in ops if is_fusible(op.category))

    subgraphs = partition_graph(ops)
    epilogues = detect_epilogue_fusions(ops)
    prologues = detect_prologue_fusions(ops)

    layer_indices = {op.layer_idx for op in ops if op.layer_idx is not None}
    layer_count = len(layer_indices)
    ops_per_layer = len(ops) // layer_count if layer_count > 0 else len(ops)

    # CUDA graph segment analysis
    cg_report = compute_cuda_graph_segments(ops, layer_count, mode=mode) if layer_count > 0 else None

    return FusionReport(
        model_name=model_name,
        total_ops=len(ops),
        ops_by_category=by_cat,
        anchor_ops=anchor_count,
        fusible_ops=fusible_count,
        subgraphs=subgraphs,
        epilogue_fusions=epilogues,
        prologue_fusions=prologues,
        all_ops=ops,
        cuda_graph_report=cg_report,
        layer_count=layer_count,
        ops_per_layer=ops_per_layer,
    )


def analyze_model_ir(ir_json: str | dict, verbose: bool = False) -> FusionReport:
    """
    Analyze a full model IR JSON.

    Args:
        ir_json: JSON string or dict from compile_model_for_hf().
        verbose: If True, print extra detail.
    """
    data = json.loads(ir_json) if isinstance(ir_json, str) else ir_json

    if not data.get("success", False):
        errors = data.get("errors", [])
        msg = "; ".join(e.get("message", str(e)) for e in errors)
        raise RuntimeError(f"IR compilation failed: {msg}")

    modules = data.get("modules", [])
    if not modules:
        raise RuntimeError("No modules in compiled IR")

    module = modules[0]
    model_name = module.get("name", "unknown")
    forward = module.get("forward")
    if not forward:
        raise RuntimeError(f"No forward graph in module '{model_name}'")

    report = analyze_graph(forward, model_name=model_name)
    report.print(verbose=verbose)
    return report
