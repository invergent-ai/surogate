import inspect
import math
import re
from typing import Optional, Tuple, Union, List

import torch
from flash_attn import flash_attn_func
from peft import PeftModelForCausalLM
from torch.nn.functional import scaled_dot_product_attention
from transformers import AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaModel, LlamaDecoderLayer
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from surogate.core.model.hf_config import HfConfigFactory
from surogate.core.model.kernels.rms_layernorm import fast_rms_layernorm
from surogate.core.model.kernels.rope_embedding import fast_rope_embedding
from surogate.core.model.kernels.utils import DEVICE_COUNT, dtype_from_config, _get_dtype
from surogate.utils.dist import get_current_device, get_device_count
from surogate.utils.logger import get_logger

logger = get_logger()

KV_CACHE_INCREMENT = 512  # KV Cache update size
SDPA_HAS_GQA = "enable_gqa" in scaled_dot_product_attention.__doc__


def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V


def original_apply_o(self, X):
    O = self.o_proj(X)
    return O


class LlamaRotaryEmbedding(torch.nn.Module):
    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            config=None,  # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        device_count = get_device_count()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            try:
                base = config.rope_theta
            except:
                base = getattr(config, "rope_parameters", {})
                base = base["rope_theta"]
            dim = getattr(config, "head_dim", None)
            if dim is None:
                dim = int((config.hidden_size // config.num_attention_heads))
            max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None] * device_count
        self.multi_gpu_sin_cached = [None] * device_count

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(device_count):
            self._set_cos_sin_cache(
                seq_len=self.current_rope_size,
                device=torch.device(device_idx),
                dtype=torch.get_default_dtype(),
            )

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )
        self.sin_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
                self.base
                ** (
                        torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float()
                        / self.dim
                )
        )
        t = torch.arange(
            self.current_rope_size, device="cpu", dtype=torch.int64
        ).float()

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )

    def get_cached(self, seq_len=None, device_index=None):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[
            device_index
        ]

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size:
            return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(get_device_count()):
            self._set_cos_sin_cache(
                self.current_rope_size, device=torch.device(device_idx), dtype=x.dtype
            )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    # Fixes https://github.com/huggingface/transformers/pull/28837
    # https://github.com/microsoft/DeepSpeed/issues/4932
    # The precision of RoPE buffers is not correct, so we cast to int64.
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            config=None,  # [TODO] Hack to pass in config - need to remove later
    ):
        self.scaling_factor = scaling_factor
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            config=config,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.current_rope_size = seq_len
        inv_freq = 1.0 / (
                self.base
                ** (
                        torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float()
                        / self.dim
                )
        )
        t = torch.arange(
            self.current_rope_size, device="cpu", dtype=torch.int64
        ).float()
        t = t / self.scaling_factor

        freqs = torch.outer(t, inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
        self,
        hidden_states: torch.Tensor,
        causal_mask=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        causal_mask: Optional[BlockDiagonalCausalMask] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    assert output_attentions is False
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
            logger.warning_once(
                f"Input IDs of shape {shape} with length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:, : self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, : self.max_seq_length, :]

    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)  # .long()
    else:
        position_ids = None

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))

    # Validate input_ids are within vocabulary bounds
    if input_ids is not None:
        vocab_size = self.embed_tokens.weight.shape[0]
        if input_ids.max() >= vocab_size or input_ids.min() < 0:
            invalid_mask = (input_ids >= vocab_size) | (input_ids < 0)
            logger.warning_once(
                f"Input IDs contain out-of-bounds indices. "
                f"Min: {input_ids.min().item()}, Max: {input_ids.max().item()}, "
                f"Vocab size: {vocab_size}. Clamping invalid indices."
            )
            input_ids = input_ids.clamp(0, vocab_size - 1)

    # Embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    inputs_embeds = inputs_embeds.to(_get_dtype(dtype_from_config(self.config)))

    IS_GRANITE = self.config.model_type.startswith("granite")

    train_embed_tokens = self.embed_tokens.weight.requires_grad

    # Fix up attention mask by setting elements to 0
    # Specifically for DPO
    if (
            getattr(self, "_has_no_labels", False) is True
            and (attention_mask is not None)
            and (past_key_values is None)
            and (not train_embed_tokens)
            and self.training
    ):
        # Careful for inference the attention_mask is size (1, kv_seq_len)
        # Whilst the input_embeds is size (1, 1, 4096)
        inputs_requires_grad = inputs_embeds.requires_grad
        if not inputs_embeds.is_leaf:
            inputs_embeds = inputs_embeds.detach()
            inputs_requires_grad = True
        elif inputs_requires_grad:
            inputs_embeds.requires_grad_(False)
        attention_mask = attention_mask[:, : self.max_seq_length]  # Must resize!
        inputs_embeds *= attention_mask.unsqueeze(0).transpose(0, 1).transpose(1, 2)
        if inputs_requires_grad:
            inputs_embeds.requires_grad_(True)

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        # if 0 in attention_mask:
        #     padding_mask = attention_mask
        # else:
        padding_mask = None

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=getattr(self.config, "sliding_window", None),
        )
        # Must NOT convert to bool - weirdly this causes stuff to error out!
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(torch.bool)

    hidden_states = inputs_embeds
    if IS_GRANITE:  # granite has embedding multiplier
        hidden_states = self.config.embedding_multiplier * hidden_states

    if past_key_values is None and self.training:
        use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # Check checkpointing method
    gradient_checkpointing = False

    if self.gradient_checkpointing and self.training and not use_cache:
        gradient_checkpointing = True

    if (hasattr(self, "rotary_emb") or not hasattr(self.layers[0].self_attn, "rotary_emb")) or IS_GRANITE:
        # Transformers main has made it mandatory to pass position_embeddings
        # https://github.com/huggingface/transformers/pull/34858
        # Also, transformers 4.45.0 supports granite but with the attention refactor (it always had the refactor)
        # unsloth's check for granite too has "version >= 4.45.0 (rightly so)".
        # so let granite always use the attention refactor implementation.

        self.rotary_emb.extend_rope_embedding(
            hidden_states, self.config.max_position_embeddings
        )
        position_embeddings = self.rotary_emb.get_cached(
            self.config.max_position_embeddings, hidden_states.device.index
        )
    else:
        position_embeddings = None

    # Go through every layer!
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past_key_value = past_key_values[idx] if past_key_values is not None else None

        mask = causal_mask

        if gradient_checkpointing and not isinstance(
                decoder_layer, GradientCheckpointingLayer
        ):

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(
                        *inputs,
                        past_key_value,
                        output_attentions,
                        padding_mask=padding_mask,
                        position_embeddings=position_embeddings,
                    )

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                mask,
                attention_mask,
                position_ids,
                use_reentrant=True,
                preserve_rng_state=False,
            )
            hidden_states = layer_outputs[0]

        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask=mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # Final layernorm
    if use_cache:
        hidden_states = fast_rms_layernorm_inference(self.norm, hidden_states)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


@torch._disable_dynamo
def PeftModel_fast_forward(
        self,
        input_ids=None,
        causal_mask=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        num_logits_to_keep=0,
        logits_to_keep=0,
        **kwargs,
):
    is_classification = "Classification" in str(type(self.base_model.model))
    if is_classification:
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
    else:
        return self.base_model(
            input_ids=input_ids,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            num_logits_to_keep=num_logits_to_keep,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


def fix_prepare_inputs_for_generation(module):
    # Fix prepare_inputs_for_generation
    if hasattr(module, "prepare_inputs_for_generation"):
        module.prepare_inputs_for_generation = _fast_prepare_inputs_for_generation


# Fix new HF's inference code
def _fast_prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
):
    past_key_values = kwargs.get("past_key_values", None)
    if past_key_values is not None:
        # Check for uninitialized DynamicCache
        if len(past_key_values) == 0:
            past_key_values = None
            kwargs["past_key_values"] = None
        # New since 4.56
        elif (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
        ):
            past_key_values = None
            kwargs["past_key_values"] = None
        else:
            bs, cache_length = input_ids.shape
            input_ids = input_ids[:, [-1]]

            # Get to the base model
            base_model = self
            if hasattr(base_model, "base_model_prefix"):
                base_model = getattr(base_model, base_model.base_model_prefix)

            if hasattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position"
            ):

                def needs_device_kw(fn) -> bool:
                    try:
                        sig = inspect.signature(inspect.unwrap(fn))
                        return "device" in sig.parameters
                    except:
                        return False

                kwargs = {
                    "sequence_length": 1,
                    "target_length": cache_length,
                    "dtype": self.dtype,
                    "cache_position": torch.arange(
                        cache_length, cache_length + 1, device=input_ids.device
                    ),
                    "batch_size": bs,
                    "config": self.config,
                    "past_key_values": past_key_values,
                }
                try:
                    if needs_device_kw(
                            base_model._prepare_4d_causal_attention_mask_with_cache_position
                    ):
                        kwargs["device"] = input_ids.device
                except:
                    print(
                        f"Could not inspect signature of {base_model._prepare_4d_causal_attention_mask_with_cache_position}"
                    )

                attention_mask = (
                    base_model._prepare_4d_causal_attention_mask_with_cache_position(
                        attention_mask,
                        **kwargs,
                    )
                )
            else:
                attention_mask = attention_mask[:, [-1]]

    if "cache_position" in kwargs:
        kwargs["position_ids"] = kwargs["cache_position"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        **kwargs,
    }


def fast_rms_layernorm_inference(self, X, XX=None, XX2=None, variance=None):
    old_dtype = X.dtype
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim=True)
    else:
        XX.copy_(X)
        torch.mean(torch.square(XX, out=XX2), -1, keepdim=True, out=variance)
    variance += self.variance_epsilon
    XX *= variance.rsqrt_()

    if XX is None:
        X = XX.to(old_dtype)
    else:
        X.copy_(XX)

    X *= self.weight
    return X


# Patches for Llama-3 LlamaExtendedRotaryEmbedding
def patch_llama_rope_scaling(
        model_name="llama",
        rope_module=None,
        scaled_rope_module=None,
        extended_rope_module=None,
        attention_module=None,
        longrope_module=None,
):
    assert (
            rope_module is not None
            and scaled_rope_module is not None
            and extended_rope_module is not None
    )
    assert attention_module is not None

    rope_name = rope_module.__name__
    scaled_rope_name = scaled_rope_module.__name__
    model_filepath = f"transformers.models.{model_name}.modeling_{model_name}"
    exec_code = (
        f"import torch.nn as nn\n"
        f"from typing import Union, Optional, List, Any, Callable, Tuple\n"
        f"from {model_filepath} import logger, "
        f"{model_name.title()}Attention, {model_name.title()}Config"
    )

    try:
        function = inspect.getsource(attention_module.__init__)
    except:
        # Most likely already patched!
        return None, None
    where = function.find("def")
    function = function.split("\n")
    function = "\n".join(x[where:] for x in function)
    init_name = f"{model_name.title()}Attention__init__"
    function = function.replace("def __init__", f"def {init_name}")
    function = function.replace(
        "super().__init__()",
        f"super({model_name.title()}Attention, self).__init__()",
    )
    fix_rope_function = """
    if getattr(self.config, "rope_scaling", None) is None:
        self.rotary_emb = {rope_function}(
            dim = self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type1 = self.config.rope_scaling.get("type", None)
        scaling_type2 = self.config.rope_scaling.get("rope_type", None)
        scaling_type = scaling_type1 if scaling_type1 is not None else scaling_type2
        scaling_factor = self.config.rope_scaling.get("factor")

        if scaling_type == "linear":
            self.rotary_emb = {scaled_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "llama3":
            self.rotary_emb = {extended_rope_function}(
                dim = self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        elif scaling_type == "longrope":
            self.rotary_emb = {longrope_rope_function}(
                dim = self.head_dim,
                max_position_embeddings = self.max_position_embeddings,
                original_max_position_embeddings = self.config.original_max_position_embeddings,
                base = self.rope_theta,
                short_factor = self.config.rope_scaling['short_factor'],
                long_factor  = self.config.rope_scaling['long_factor' ],
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {{scaling_type}}")
    pass
    """

    fix_rope_function = fix_rope_function.format(
        rope_function=rope_module.__name__,
        scaled_rope_function=scaled_rope_module.__name__,
        extended_rope_function=extended_rope_module.__name__,
        longrope_rope_function=(
            longrope_module if longrope_module is not None else rope_module
        ).__name__,
    )
    rotary_emb = re.findall(
        r"self\.rotary\_emb \= .+?\)",
        function,
        flags=re.DOTALL | re.MULTILINE,
    )
    if len(rotary_emb) == 0:
        return None, function
    rotary_emb = rotary_emb[0]
    function = function.replace(rotary_emb, fix_rope_function, 1)
    function = exec_code + "\n\n" + function
    return init_name, function


# See https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py#L736
# For Llama 3.1
class LlamaExtendedRotaryEmbedding(torch.nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            config=None,  # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            dim = int((config.hidden_size // config.num_attention_heads))
            max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(4 * 8192, self.max_position_embeddings)
        self.multi_gpu_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_sin_cached = [None] * DEVICE_COUNT

        # Normal Llama-3 RoPE
        inv_freq = 1.0 / (
                self.base
                ** (
                        torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float()
                        / self.dim
                )
        )
        inv_freq = self.apply_scaling(inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                seq_len=self.current_rope_size,
                device=torch.device(device_idx),
                dtype=torch.get_default_dtype(),
            )

        # dummy so that patch_utils doesn't fail for now
        self.cos_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )
        self.sin_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        t = torch.arange(
            self.current_rope_size, device=self.inv_freq.device, dtype=torch.int64
        ).float()

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype, device=device, non_blocking=True)
        sin = emb.sin().to(dtype=dtype, device=device, non_blocking=True)
        self.multi_gpu_cos_cached[device.index] = cos
        self.multi_gpu_sin_cached[device.index] = sin
        return cos, sin

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        device_index = x.device.index
        return (
            self.multi_gpu_cos_cached[device_index][:seq_len],
            self.multi_gpu_sin_cached[device_index][:seq_len],
        )

    def get_cached(self, seq_len=None, device_index=None):
        if device_index is None:
            device_index = get_current_device()
        return self.multi_gpu_cos_cached[device_index], self.multi_gpu_sin_cached[
            device_index
        ]

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size:
            return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                self.current_rope_size, device=torch.device(device_idx), dtype=x.dtype
            )

    # From https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py#L41
    def apply_scaling(self, freqs: torch.Tensor):
        # Values obtained from grid search
        scale_factor = 8
        low_freq_factor = 1
        high_freq_factor = 4
        old_context_len = 8192  # original llama3 length

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


class LongRopeRotaryEmbedding(torch.nn.Module):
    # For Phi 3.5 128K https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/modeling_phi3.py
    def __init__(
            self,
            dim=None,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            base=10000,
            short_factor=None,
            long_factor=None,
            device=None,
            config=None,  # [TODO] Hack to pass in config - need to remove later
    ):
        super().__init__()
        assert short_factor is not None
        assert long_factor is not None
        assert type(original_max_position_embeddings) is int

        if config is not None:
            # [TODO] Hack to pass in config - need to remove later
            base = config.rope_theta
            dim = int((config.hidden_size // config.num_attention_heads))
            max_position_embeddings = config.max_position_embeddings

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        # Dynamic RoPE we first set it to a max of 4 * 8192 tokens then we iteratively grow this
        self.current_rope_size = min(
            original_max_position_embeddings, self.max_position_embeddings
        )
        self.multi_gpu_short_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_short_sin_cached = [None] * DEVICE_COUNT
        self.multi_gpu_long_cos_cached = [None] * DEVICE_COUNT
        self.multi_gpu_long_sin_cached = [None] * DEVICE_COUNT

        # Long RoPE similar to RoPE except short sequences have 1 cos / sin
        # and long sequences have another cos / sin
        inv_freq_shape = (
                torch.arange(0, self.dim, 2, dtype=torch.int64, device="cpu").float()
                / self.dim
        )
        short_factor = torch.tensor(short_factor, device="cpu", dtype=torch.float32)
        long_factor = torch.tensor(long_factor, device="cpu", dtype=torch.float32)
        short_inv_freq = 1.0 / (short_factor * self.base ** inv_freq_shape)
        long_inv_freq = 1.0 / (long_factor * self.base ** inv_freq_shape)

        # Phi-3 Scale factor
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
        self.scaling_factor = scaling_factor

        # Short and long inv_freq
        self.register_buffer("short_inv_freq", short_inv_freq, persistent=False)
        self.register_buffer("long_inv_freq", long_inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        # Initialize short sequences cache for all devices
        dtype = torch.bfloat16
        t = torch.arange(
            original_max_position_embeddings,
            device=self.short_inv_freq.device,
            dtype=torch.int64,
        ).float()
        freqs = torch.outer(t, self.short_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        for device_idx in range(DEVICE_COUNT):
            device_obj = torch.device(device_idx)
            cos_cached = (emb.cos() * self.scaling_factor).to(
                dtype=dtype, device=device_obj, non_blocking=True
            )
            sin_cached = (emb.sin() * self.scaling_factor).to(
                dtype=dtype, device=device_obj, non_blocking=True
            )
            self.multi_gpu_short_cos_cached[device_idx] = cos_cached
            self.multi_gpu_short_sin_cached[device_idx] = sin_cached

        # dummy so that patch_utils doesn't fail for now
        self.short_cos_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )
        self.short_sin_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )
        self.long_cos_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )
        self.long_sin_cached = torch.empty(
            1, device=get_current_device(), dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # Note: on the original Llama codebase, these tensors are created on the target device (and not on CPU) and
        # in FP32. They are applied (multiplied) in FP32 as well.
        self.current_rope_size = seq_len

        t = torch.arange(
            self.current_rope_size, device=self.long_inv_freq.device, dtype=torch.int64
        ).float()
        # Long sequences
        freqs = torch.outer(t, self.long_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = (emb.cos() * self.scaling_factor).to(
            dtype=dtype, device=device, non_blocking=True
        )
        sin_cached = (emb.sin() * self.scaling_factor).to(
            dtype=dtype, device=device, non_blocking=True
        )
        self.multi_gpu_long_cos_cached[device.index] = cos_cached
        self.multi_gpu_long_sin_cached[device.index] = sin_cached
        return cos_cached, sin_cached

    def forward(self, x, position_ids=None, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len is not None and seq_len > self.current_rope_size:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        device_index = x.device.index

        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return (
                self.multi_gpu_short_cos_cached[device_index][:seq_len],
                self.multi_gpu_short_sin_cached[device_index][:seq_len],
            )
        else:
            return (
                self.multi_gpu_long_cos_cached[device_index][:seq_len],
                self.multi_gpu_long_sin_cached[device_index][:seq_len],
            )

    def get_cached(self, seq_len=None, device_index=None):
        if device_index is None:
            device_index = get_current_device()
        if seq_len is not None and seq_len < self.original_max_position_embeddings:
            return self.multi_gpu_short_cos_cached[
                device_index
            ], self.multi_gpu_short_sin_cached[device_index]
        return self.multi_gpu_long_cos_cached[
            device_index
        ], self.multi_gpu_long_sin_cached[device_index]

    def extend_rope_embedding(self, x, seq_len):
        if seq_len <= self.current_rope_size:
            return
        # Iteratively grow by increments of 8192
        self.current_rope_size = ((seq_len // 8192) + ((seq_len % 8192) != 0)) * 8192
        for device_idx in range(DEVICE_COUNT):
            self._set_cos_sin_cache(
                self.current_rope_size, device=torch.device(device_idx), dtype=x.dtype
            )


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
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
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings and kv_seq_len <= position_embeddings[0].shape[0]:
        cos, sin = position_embeddings
    else:
        # Extend RoPE dynamically to fit in VRA
        rotary_emb = self.rotary_emb
        rotary_emb.extend_rope_embedding(V, seq_len=kv_seq_len)

        # if position_ids is None:
        #     # Useful for LongRoPE
        #     cos, sin = rotary_emb.get_cached(kv_seq_len, device = Q.device)
        # else:
        #     cos, sin = rotary_emb.get_cached(seq_len = kv_seq_len, device = Q.device)
        cos, sin = rotary_emb.get_cached(kv_seq_len, Q.device.index)

    # Q, K = (
    #     fast_rope_embedding(Q, K, cos, sin)
    #     if position_ids is None
    #     else inplace_rope_embedding(Q, K, cos, sin, position_ids)
    # )
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
        A = flash_attn_func(Q, K, V, causal=True)
    else:
        # when qlen==vlen and attn_mask is None, we should use causal attention
        Q_len = Q.shape[-2]
        K_len = K.shape[-2]
        if attention_mask is None and Q_len == K_len:
            is_causal = True
        else:
            is_causal = False
        # Grouped query attention
        if SDPA_HAS_GQA:
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attention_mask,
                is_causal=is_causal,
                enable_gqa=n_groups != 1,
            )
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2)  # .contiguous()
        else:
            if n_groups != 1:
                K = K[:, :, None, :, :].expand(
                    bsz, n_kv_heads, n_groups, kv_seq_len, head_dim
                )
                V = V[:, :, None, :, :].expand(
                    bsz, n_kv_heads, n_groups, kv_seq_len, head_dim
                )
                K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
                V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
            pass
            # Must be contiguous or else results are False!
            # https://github.com/pytorch/pytorch/issues/112577
            Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
            # Needs (batch_size, n_heads, seq_len, head_dim)
            # is_casual and attention_mask must not be both set!
            A = scaled_dot_product_attention(
                Q, K, V, attn_mask=attention_mask, is_causal=is_causal
            )
            # Go back to (batch_size, seq_len, n_heads, head_dim)
            A = A.transpose(1, 2).contiguous()
        pass
    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


class FastLlamaModel:
    @staticmethod
    def pre_patch():
        init_name, function = patch_llama_rope_scaling(
            model_name="llama",
            rope_module=LlamaRotaryEmbedding,
            scaled_rope_module=LlamaLinearScalingRotaryEmbedding,
            extended_rope_module=LlamaExtendedRotaryEmbedding,
            attention_module=LlamaAttention,
            longrope_module=LongRopeRotaryEmbedding,
        )
        if init_name is not None:
            exec(function, globals())
            LlamaAttention.__init__ = eval(init_name)

        LlamaAttention.forward = LlamaAttention_fast_forward
        LlamaDecoderLayer.forward = LlamaDecoderLayer_fast_forward
        LlamaModel.forward = LlamaModel_fast_forward
        PeftModelForCausalLM.forward = PeftModel_fast_forward
        fix_prepare_inputs_for_generation(LlamaForCausalLM)

        # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
        # Inference can now be CUDAGraphed, but we shall retain the old rotary embeddings.
        # https://github.com/huggingface/transformers/pull/27931
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
        import transformers.models.llama.modeling_llama
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = (
            LlamaRotaryEmbedding
        )
        transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = (
            LlamaLinearScalingRotaryEmbedding
        )

    @staticmethod
    def from_pretrained(
            model_name,
            model_config,
            model_patcher=None,
            **kwargs,
    ):
        model_patcher = model_patcher or FastLlamaModel
        model_patcher.pre_patch()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **kwargs,
        )

        max_seq_length = HfConfigFactory.get_max_model_len(model_config)
        model.max_seq_length = max_seq_length
        m = model
        while hasattr(m, "model"):
            m.max_seq_length = max_seq_length
            m = m.model
        m.max_seq_length = max_seq_length
        # Save to modules as well
        for module in model.modules():
            module.max_seq_length = max_seq_length

        # Downcast RoPE embedding to correct data type
        correct_dtype = kwargs['dtype']
        for name, module in model.named_modules():
            if (name.endswith("rotary_emb") or hasattr(module, "cos_cached")):
                if hasattr(module, "cos_cached") and \
                        (module.cos_cached.dtype != correct_dtype):
                    module.cos_cached = module.cos_cached.to(correct_dtype)
                    module.sin_cached = module.sin_cached.to(correct_dtype)
                elif hasattr(module, "short_cos_cached") and \
                        (module.short_cos_cached.dtype != correct_dtype):
                    module.short_cos_cached = module.short_cos_cached.to(correct_dtype)
                    module.short_sin_cached = module.short_sin_cached.to(correct_dtype)

        is_torch_embedding = False
        try:
            old_input_embedding = model.get_input_embeddings()
            is_torch_embedding = type(old_input_embedding) is torch.nn.Embedding
            old_input_embedding = old_input_embedding.weight
        except:
            return model

        try:
            old_output_embedding = model.get_output_embeddings().weight
        except:
            old_output_embedding = torch.zeros(0)

        is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
                  or (model.config.tie_word_embeddings)

        if is_torch_embedding:
            model.set_input_embeddings(
                torch.nn.Embedding.from_pretrained(
                    old_input_embedding,
                    padding_idx=getattr(model.config, "pad_token_id", None),
                )
            )

        if old_output_embedding.numel() != 0:
            requires_grad = old_output_embedding.requires_grad
            lm_head = torch.nn.Linear(1, 1, bias=None)
            del lm_head.weight

            lm_head.weight = old_output_embedding if not is_tied else old_input_embedding
            lm_head.in_features = lm_head.weight.shape[1]
            lm_head.out_features = lm_head.weight.shape[0]

            lm_head.weight.requires_grad_(requires_grad)
            model.set_output_embeddings(lm_head)
            if hasattr(model, "lm_head"): model.lm_head = lm_head

        if is_tied: model.tie_weights()

        for layer in model.model.layers:
            set_of_devices = set(x.device for x in layer.parameters())
            if len(set_of_devices) > 1:
                raise ValueError(f"All parameters of {layer} should be on the same device")
            device = set_of_devices.pop()
            layer._per_layer_device_index = device.index

        # Patch up QKV / O and MLP
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.apply_qkv = original_apply_qkv
            layer.self_attn.apply_o = original_apply_o

        # For transformers > 4.47.1, we need to add rotary_emb to all attention layers
        if hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = rotary_emb

                # Set weight[padding_idx] = 0
                with torch.no_grad():
                    for name, module in model.named_modules():
                        if type(module) is torch.nn.Embedding:
                            if (
                                    getattr(module, "weight", None) is not None
                                    and getattr(module, "padding_idx", None) is not None
                            ):
                                if module.padding_idx < module.weight.shape[0]:
                                    module.weight[module.padding_idx] = 0

        return model
