from typing import Optional, Tuple

import torch
from flash_attn import flash_attn_func
from peft import PeftModelForCausalLM
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
)
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from surogate.core.model.kernels.rms_layernorm import fast_rms_layernorm
from surogate.core.model.kernels.rope_embedding import fast_rope_embedding
from surogate.core.model.models.fast_llama import FastLlamaModel, LlamaRotaryEmbedding, \
    LlamaLinearScalingRotaryEmbedding, LlamaDecoderLayer_fast_forward, LlamaModel_fast_forward, \
    PeftModel_fast_forward, fix_prepare_inputs_for_generation
from surogate.core.model.patcher import patch_linear_scaling


def Qwen3Attention_fast_forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask: Optional[BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention

    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(
        bsz, q_len, n_heads, head_dim
    )  # .transpose(1, 2) # we will transpose after normalisation
    K = K.view(
        bsz, q_len, n_kv_heads, head_dim
    )  # .transpose(1, 2) # we will transpose after normalisation
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    # Qwen3 has QKNorm. This seems to be the only difference from Qwen2.
    # Note that using fast_layernorm_compiled causes issues as the dimensions don't match up.
    # I tried to add a compiled version of the new norm but the numbers don't match up with Transformers
    # TODO: Check on the differences here.
    Q = fast_rms_layernorm(self.q_norm, Q)
    K = fast_rms_layernorm(self.k_norm, K)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings and kv_seq_len <= position_embeddings[0].shape[0]:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len=kv_seq_len)
        device_index = Q.device.index

        if position_ids is None:
            # Useful for LongRoPE
            cos, sin = rotary_emb.get_cached(kv_seq_len, device_index)
        else:
            cos, sin = rotary_emb.get_cached(kv_seq_len, device_index)

    Q, K = fast_rope_embedding(Q, K, cos, sin)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        sw = kv_seq_len
        window = (-1, -1) if (kv_seq_len <= sw) else (sw, sw)
        A = flash_attn_func(Q, K, V, causal=True, window_size=window)
    else:
        # Grouped query attention
        # if n_groups != 1:
        K = K[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        V = V[:, :, None, :, :].expand(bsz, n_kv_heads, n_groups, kv_seq_len, head_dim)
        K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        # pass
        # Must be contiguous or else results are False!
        # https://github.com/pytorch/pytorch/issues/112577
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        # when qlen==vlen and attn_mask is None, we should use causal attention
        Q_len = Q.shape[-2]
        K_len = K.shape[-2]
        if attention_mask is None and Q_len == K_len:
            is_causal = True
        else:
            is_causal = False

        A = scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, is_causal=is_causal
        )
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()

    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


class FastQwen3Model(FastLlamaModel):
    @staticmethod
    def pre_patch():
        init_name, function = patch_linear_scaling(
            model_name="Qwen3",
            rope_module=LlamaRotaryEmbedding,
            scaled_rope_module=LlamaLinearScalingRotaryEmbedding,
            attention_module=Qwen3Attention,
        )
        if init_name is not None:
            exec(function, globals())
            Qwen3Attention.__init__ = eval(init_name)

        Qwen3Attention.forward = Qwen3Attention_fast_forward
        Qwen3DecoderLayer.forward = LlamaDecoderLayer_fast_forward
        Qwen3Model.forward = LlamaModel_fast_forward
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(Qwen3ForCausalLM)

        import transformers.models.qwen3.modeling_qwen3
        transformers.models.qwen3.modeling_qwen3.Qwen3RotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        return

    @staticmethod
    def from_pretrained(
            model_name,
            model_patcher=None,
            **kwargs,
    ):
        return FastLlamaModel.from_pretrained(
            model_name=model_name,
            model_patcher=FastQwen3Model,
            **kwargs,
        )
