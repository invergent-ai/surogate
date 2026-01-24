"""LLaMA Model."""

from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, param, forward, hf_config
from ..graph_builder import graph
from ..hf import fuse


@model
@hf_config(
    architecture="LlamaForCausalLM",
    model_type="llama",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    eps="rms_norm_eps",
)
class LlamaModel:
    """LLaMA model."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        d_ff: int = 11008,
        max_seq: int = 4096,
        head_size: int = 128,
        eps: float = 1e-6,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps

        self.D = head_size if head_size > 0 else d_model // num_query_heads

    @param(hf_mapping="model.embed_tokens.weight")
    def embedding(self) -> Tensor["vocab_size", "d_model"]:
        ...

    @param
    def blocks(self) -> Array["n_layers", "LlamaBlock"]:
        ...

    @param(hf_mapping="model.norm.weight")
    def final_norm(self) -> Tensor["d_model"]:
        ...

    @param(hf_mapping="lm_head.weight")
    def lm_head(self) -> Tensor["vocab_size", "d_model"]:
        ...

    _hf_block_mappings_ = {
        "ln1_weight": "model.layers.{layer}.input_layernorm.weight",
        "ln2_weight": "model.layers.{layer}.post_attention_layernorm.weight",
        "qkv_weight": fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0,
        ),
        "out_weight": "model.layers.{layer}.self_attn.o_proj.weight",
        # swiglu expects [up, gate] concatenation
        "mlp_up_weight": fuse(
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight",
            dim=0,
        ),
        "mlp_down_weight": "model.layers.{layer}.mlp.down_proj.weight",
    }

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor["T", "int32"],
    ) -> Tensor["B", "T", "vocab_size"]:
        with graph() as g:
            x0 = g.embedding(token_ids, "embedding")
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            xN, residualN = g.call(
                "StackedBlocks",
                x0,
                residual0,
                position_ids,
                num_outputs=2,
                blocks="blocks",
                n_layers=self.n_layers,
            )

            # Final norm - use explicit names for outputs to map to C++ activation slots
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
                y_name="xF"  # Map to LNFinal slot in C++
            )

            # LM head
            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            logits_flat = g.matmul(xF_flat, "lm_head", transpose="NT", out_name="logits_flat")
            logits = g.view(logits_flat, shape=["B", "T", "vocab_size"], out_name="logits")

            return logits
