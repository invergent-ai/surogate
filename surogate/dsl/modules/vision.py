"""The vision tower, as the declaration describes it.

Every Qwen3.5 checkpoint ships one, and the text stack has always been able to
take its output — `Qwen3_5ConditionalModel.forward` scatters `visual_embeds` into
the token embeddings by mask. What was missing was the tower that produces them,
so a multimodal checkpoint could be served but not trained end to end.

It is a plain ViT and needs no new kernels: a patch projection, a learned position
table, `layers` pre-norm blocks of bidirectional attention and a GELU MLP, then a
merger that pools each `merge x merge` patch group and projects into the text
width. The two places it differs from the text stack are worth stating because
both are easy to get silently wrong:

* the norms are **LayerNorm**, with a bias — not the RMSNorm the text stack uses
  everywhere — which is why the checkpoint carries `norm1.bias` at all;
* the attention is **bidirectional**. A vision transformer has no causal
  structure, and running it causally would still produce plausible-looking
  embeddings, so this is a correctness property no shape check can catch.

The geometry is per-model, not per-family: the 0.8B tower is 12 layers of 768, the
2B and 4B are 24 of 1024, Flash-Next and the 35B are 27 of 1152. It arrives from
each checkpoint's own `vision_config`.
"""

from __future__ import annotations

from typing import Any

from ..dim import Dim
from ..nn import Module, Proxy, Tracer


class VisionTower(Module):
    """Patches in, text-width embeddings out.

    Called with a packed patch tensor ``(P, patch_rows)`` and returns
    ``(P // merge_unit, d_model)`` — one embedding per merged patch group, which
    is what the text stack scatters into its token embeddings.
    """

    _hf_mapping_defaults_ = {
        "patch_embed_weight": "{prefix}.patch_embed.proj.weight",
        "patch_embed_bias": "{prefix}.patch_embed.proj.bias",
        "position_embedding": "{prefix}.pos_embed.weight",
        "merger_norm_weight": "{prefix}.merger.norm.weight",
        "merger_norm_bias": "{prefix}.merger.norm.bias",
        "merger_fc1_weight": "{prefix}.merger.linear_fc1.weight",
        "merger_fc1_bias": "{prefix}.merger.linear_fc1.bias",
        "merger_fc2_weight": "{prefix}.merger.linear_fc2.weight",
        "merger_fc2_bias": "{prefix}.merger.linear_fc2.bias",
    }

    def __init__(
        self,
        d_model: int,
        hidden: int,
        layers: int,
        intermediate: int,
        heads: int,
        patch_rows: int,
        position_embeddings: int,
        merge: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(
                f"vision hidden {hidden} must divide evenly across {heads} heads"
            )
        self.d_model = d_model
        self.layers = layers
        self.eps = eps

        # Short capitalised names so the shape resolver can use them as dims.
        self.C = Dim("C")
        self.VH = hidden
        self.VI = intermediate
        self.VHeads = heads
        self.VHeadDim = hidden // heads
        self.VQkv = 3 * hidden
        self.VPatch = patch_rows
        self.VPos = position_embeddings
        self.VMerge = merge
        self.VMergeUnit = merge * merge
        self.VMerged = hidden * merge * merge

    def layer_mapping(self, prefix: str, layer: int) -> dict[str, str]:
        """Checkpoint paths for one encoder block, so a model can splice these
        into its `_hf_block_mappings_` without restating the layout."""

        block = f"{prefix}.blocks.{layer}"
        return {
            f"vision_blocks_{layer}_norm1_weight": f"{block}.norm1.weight",
            f"vision_blocks_{layer}_norm1_bias": f"{block}.norm1.bias",
            f"vision_blocks_{layer}_qkv_weight": f"{block}.attn.qkv.weight",
            f"vision_blocks_{layer}_qkv_bias": f"{block}.attn.qkv.bias",
            f"vision_blocks_{layer}_out_weight": f"{block}.attn.proj.weight",
            f"vision_blocks_{layer}_out_bias": f"{block}.attn.proj.bias",
            f"vision_blocks_{layer}_norm2_weight": f"{block}.norm2.weight",
            f"vision_blocks_{layer}_norm2_bias": f"{block}.norm2.bias",
            f"vision_blocks_{layer}_fc1_weight": f"{block}.mlp.linear_fc1.weight",
            f"vision_blocks_{layer}_fc1_bias": f"{block}.mlp.linear_fc1.bias",
            f"vision_blocks_{layer}_fc2_weight": f"{block}.mlp.linear_fc2.weight",
            f"vision_blocks_{layer}_fc2_bias": f"{block}.mlp.linear_fc2.bias",
        }

    def _encoder_block(self, tracer: Tracer, g: Any, x: Any, layer: int) -> Any:
        """One pre-norm ViT block. LayerNorm, not RMSNorm; attention is
        bidirectional."""

        name = f"blocks_{layer}_"
        for suffix, shape in (
            ("norm1_weight", (self.VH,)),
            ("norm1_bias", (self.VH,)),
            ("norm2_weight", (self.VH,)),
            ("norm2_bias", (self.VH,)),
            ("qkv_bias", (self.VQkv,)),
            ("out_bias", (self.VH,)),
            ("fc1_bias", (self.VI,)),
            ("fc2_bias", (self.VH,)),
        ):
            tracer.register_param(name + suffix, shape, quantizable=False)
        tracer.register_param(name + "qkv_weight", (self.VQkv, self.VH))
        tracer.register_param(name + "out_weight", (self.VH, self.VH))
        tracer.register_param(name + "fc1_weight", (self.VI, self.VH))
        tracer.register_param(name + "fc2_weight", (self.VH, self.VI))

        normed, _, _ = g.layernorm(
            x, tracer.prefixed(name + "norm1_weight"), tracer.prefixed(name + "norm1_bias"),
            eps=self.eps,
        )
        qkv = g.matmul_bias(
            normed,
            tracer.prefixed(name + "qkv_weight"),
            tracer.prefixed(name + "qkv_bias"),
            transpose="NT",
            out_name=tracer.prefixed(name + "qkv"),
        )
        qkv = g.view(qkv, shape=["P", 3, self.VHeads, self.VHeadDim],
                     out_name=tracer.prefixed(name + "qkv_heads"))
        # A vision transformer attends over the whole image: causal=False is a
        # correctness property, not a tuning knob.
        attended, _ = g.flash_attention(qkv, causal=False)
        attended = g.view(attended, shape=["P", self.VH],
                          out_name=tracer.prefixed(name + "attn_flat"))
        projected = g.matmul_bias(
            attended,
            tracer.prefixed(name + "out_weight"),
            tracer.prefixed(name + "out_bias"),
            transpose="NT",
            out_name=tracer.prefixed(name + "attn_out"),
        )
        x = g.add(x, projected, out_name=tracer.prefixed(name + "res_attn"))

        normed2, _, _ = g.layernorm(
            x, tracer.prefixed(name + "norm2_weight"), tracer.prefixed(name + "norm2_bias"),
            eps=self.eps,
        )
        hidden = g.matmul_bias(
            normed2,
            tracer.prefixed(name + "fc1_weight"),
            tracer.prefixed(name + "fc1_bias"),
            transpose="NT",
            out_name=tracer.prefixed(name + "fc1"),
        )
        hidden = g.gelu(hidden, out_name=tracer.prefixed(name + "act"))
        hidden = g.matmul_bias(
            hidden,
            tracer.prefixed(name + "fc2_weight"),
            tracer.prefixed(name + "fc2_bias"),
            transpose="NT",
            out_name=tracer.prefixed(name + "fc2"),
        )
        return g.add(x, hidden, out_name=tracer.prefixed(name + "res_mlp"))

    def _trace(self, tracer: Tracer, *args: Proxy, **kwargs: Any) -> Proxy:
        g = tracer.graph
        (patches,) = args

        tracer.register_param("patch_embed_weight", (self.VH, self.VPatch))
        tracer.register_param("patch_embed_bias", (self.VH,), quantizable=False)
        tracer.register_param("position_embedding", (self.VPos, self.VH), quantizable=False)
        tracer.register_param("merger_norm_weight", (self.VH,), quantizable=False)
        tracer.register_param("merger_norm_bias", (self.VH,), quantizable=False)
        tracer.register_param("merger_fc1_weight", (self.VMerged, self.VMerged))
        tracer.register_param("merger_fc1_bias", (self.VMerged,), quantizable=False)
        tracer.register_param("merger_fc2_weight", ("C", self.VMerged))
        tracer.register_param("merger_fc2_bias", ("C",), quantizable=False)

        out_slot = tracer.register_activation(
            "embeds",
            ("P // merge_unit", "C"),
            share_policy="when_recomputed",
            description="Vision embeddings in the text width",
        )

        x = g.matmul_bias(
            patches.ref,
            tracer.prefixed("patch_embed_weight"),
            tracer.prefixed("patch_embed_bias"),
            transpose="NT",
            out_name=tracer.prefixed("patch_embed"),
        )
        # The position table is gathered per patch by the caller's index; adding it
        # here keeps the tower self-contained for the shapes the artifact stores.
        x = g.add(x, tracer.prefixed("position_embedding"), out_name=tracer.prefixed("positioned"))

        for layer in range(self.layers):
            x = self._encoder_block(tracer, g, x, layer)

        merged, _, _ = g.layernorm(
            x, tracer.prefixed("merger_norm_weight"), tracer.prefixed("merger_norm_bias"),
            eps=self.eps,
        )
        merged = g.view(merged, shape=["P // merge_unit", self.VMerged],
                        out_name=tracer.prefixed("merged"))
        merged = g.matmul_bias(
            merged,
            tracer.prefixed("merger_fc1_weight"),
            tracer.prefixed("merger_fc1_bias"),
            transpose="NT",
            out_name=tracer.prefixed("merger_fc1"),
        )
        merged = g.gelu(merged, out_name=tracer.prefixed("merger_act"))
        out = g.matmul_bias(
            merged,
            tracer.prefixed("merger_fc2_weight"),
            tracer.prefixed("merger_fc2_bias"),
            transpose="NT",
            out_name=out_slot,
        )
        return Proxy(out_slot, out)
