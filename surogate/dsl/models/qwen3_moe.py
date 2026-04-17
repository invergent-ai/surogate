"""Qwen3 MoE Model."""

from __future__ import annotations

from .. import nn
from ..blocks.qwen3_moe import Qwen3MoEBlock
from ..hf import build_attn_mappings, build_moe_mappings, build_norm_mappings
from ..modules.attention import Qwen3Attention
from ..nn import STANDARD_MODEL_NAME_REMAP
from ..specs import ActivationScope


@nn.hf_config(
    architecture="Qwen3MoeForCausalLM",
    model_type="qwen3_moe",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="moe_intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
    num_experts="num_experts",
    num_experts_per_tok="num_experts_per_tok",
    shared_expert_intermediate="shared_expert_intermediate_size",
)
class Qwen3MoEModel(nn.Model):
    """Qwen3 Mixture of Experts model."""

    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        **build_norm_mappings(),
        **build_attn_mappings(attn_module=Qwen3Attention),
        **build_moe_mappings(include_shared=True),
        # Model-level weight mappings (not per-block)
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 2048,
        n_layers: int = 28,
        num_query_heads: int = 16,
        num_kv_heads: int = 4,
        d_ff: int = 1408,
        max_seq: int = 40960,
        head_size: int = 128,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        use_shared_expert: bool = False,
        shared_expert_intermediate: int = 0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_shared_expert = use_shared_expert

        self.D = head_size if head_size > 0 else d_model // num_query_heads

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.BlockStack(
            n_layers,
            Qwen3MoEBlock,
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            d_ff=d_ff,
            max_seq=max_seq,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            eps=eps,
            use_qkv_bias=use_qkv_bias,
            use_qk_norm=use_qk_norm,
            norm_topk_prob=norm_topk_prob,
            use_shared_expert=use_shared_expert,
            shared_expert_intermediate=shared_expert_intermediate,
            ep_size=ep_size,
        )
        self.final_norm = nn.RMSNorm(d_model, eps=eps)
        self.lm_head = nn.LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        # IO slots
        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

        # Global intermediate slots (must match old DSL for C++ runtime)
        _h = ("B", "T", "d_model")
        self._register_activation("residual0", _h, scope=G)
        self._register_activation("x0", _h, aliases=["encoded"], scope=G)
        self._register_activation("xN", _h, scope=G)
        self._register_activation("residualN", _h, scope=G)
        self._register_activation("residual_final", _h, scope=G)
        self._register_activation("xF", _h, aliases=["ln_final"], scope=G)
        self._register_activation("xF_flat", ("B * T", "d_model"), scope=G)
        self._register_activation("ln_final_rstd", ("B", "T"), dtype="fp32", save=True, scope=G)
        self._register_activation("loss", ("B * T",), dtype="fp32", aliases=["losses"], scope=G)

        x = self.embedding(token_ids)
        residual = self._zeros(["B", "T", "d_model"])
        x, residual = self.blocks(x, residual, position_ids)
        residual, x = self.final_norm(residual, x)
        loss = self.lm_head(x, targets)
        return loss
