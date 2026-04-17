"""Module wrappers for Qwen3.5 gated delta rule primitives."""

from __future__ import annotations

from ..decorators import forward, module
from ..graph_builder import graph
from ..tensor_type import Tensor


@module
class ChunkGatedDeltaRule:
    """Chunked gated delta rule wrapper."""

    def __init__(
        self,
        chunk_size: int = 64,
        scale: float = 0.0,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        self.chunk_size = chunk_size
        self.scale = scale
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    @forward
    def forward(
        self,
        query: Tensor[B, T, H, K],
        key: Tensor[B, T, H, K],
        value: Tensor[B, T, H, V],
        g: Tensor[B, T, H],
        beta: Tensor[B, T, H],
        initial_state: Tensor[B, H, K, V] | None = None,
    ) -> tuple[Tensor[B, T, H, V], Tensor[B, H, K, V] | None]:
        with graph() as gb:
            inputs = [query, key, value, g, beta]
            if initial_state is not None:
                inputs.append(initial_state)
            out, final_state = gb.custom(
                "chunk_gated_delta_rule",
                *inputs,
                num_outputs=2,
                scale=self.scale,
                chunk_size=self.chunk_size,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            )
            return out, final_state
