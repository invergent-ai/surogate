"""LLaMA Model."""

from __future__ import annotations

from .. import nn
from ..modules import Embedding, LMHead, RMSNorm
from ..blocks.llama import LlamaBlock
from ..hf import build_dense_block_mappings
from ..blocks.common import STANDARD_MODEL_NAME_REMAP
from ..specs import ActivationScope


@nn.hf_config(
    architecture="LlamaForCausalLM",
    model_type="llama",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
)
class LlamaModel(nn.Model):
    """LLaMA model with GQA attention and SwiGLU MLP."""

    _name_remap_ = STANDARD_MODEL_NAME_REMAP
    _hf_block_mappings_ = {
        **build_dense_block_mappings(),
        "embedding": "model.embed_tokens.weight",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    }

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

        self.D = head_size if head_size > 0 else d_model // num_query_heads

        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.BlockStack(
            n_layers,
            LlamaBlock,
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            d_ff=d_ff,
            max_seq=max_seq,
            eps=eps,
        )
        self.final_norm = RMSNorm(d_model, eps=eps)
        self.lm_head = LMHead(vocab_size, d_model)

    def forward(self, token_ids, position_ids, targets):
        G = ActivationScope.GLOBAL

        self._register_activation("token_ids", ("B", "T"), dtype="int32", scope=G)
        self._register_activation("position_ids", ("T",), dtype="int32", scope=G)
        self._register_activation("targets", ("B", "T"), dtype="int32", scope=G, aliases=["labels"])
        self._register_activation("freq_cis", ("max_seq", "D", 2), dtype="fp32", scope=G, aliases=["rope_freqs"])

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
