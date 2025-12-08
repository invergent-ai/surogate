import importlib
import inspect
import types
from typing import Union, Generator, Tuple, Type

import torch
from peft import PeftModelForCausalLM
from torch import nn

from surogate.core.config.sft_config import SFTConfig
from surogate.core.model.patcher import detab_code
from surogate.train.lora_fast.fast_kernels import apply_lora_qkv, apply_lora_o, apply_lora_mlp_swiglu, \
    apply_lora_mlp_geglu
from surogate.utils.logger import get_logger

logger = get_logger()

SUPPORTED_ACTIVATIONS = ["silu", "gelu"]

APPLY_FN_MAPPING = {
    "silu": apply_lora_mlp_swiglu,
    "gelu": apply_lora_mlp_geglu,
}

ORIGINAL_O_CODE = """
    attn_output = self.o_proj(attn_output)
""".lstrip("\n")

PATCHED_O_CODE = """
    attn_output = self.apply_o(attn_output)
""".lstrip("\n")

QKV_PATCHES = [
    (
        """
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
    ),
    (
        """
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
        """
    query_states, key_states, value_states = self.apply_qkv(hidden_states)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)
""".lstrip("\n"),
    ),
]


def apply_lora_fast_pre_load(config: SFTConfig):
    """
    This method patches the inferred attention class forward pass with optimized LoRA implementations.
    It modifies the attention class to use optimized QKV and output projections.
    The original implementation is preserved and can be restored if needed.
    """
    attention_cls = config.model_template.attention_cls
    # Check if already patched
    if hasattr(attention_cls, "_original_forward"):
        logger.debug(f"{attention_cls.__name__} already patched")
        return

    self_attn_forward = inspect.getsource(attention_cls.forward)
    attention_cls._original_forward = self_attn_forward
    self_attn_forward, _ = detab_code(self_attn_forward)

    assert any(qkv_options[0] in self_attn_forward for qkv_options in QKV_PATCHES), (
        "Original QKV code not found"
    )
    assert ORIGINAL_O_CODE in self_attn_forward, "Original O code not found"

    for qkv_orig, qkv_patched in QKV_PATCHES:
        if qkv_orig in self_attn_forward:
            self_attn_forward = self_attn_forward.replace(
                qkv_orig,
                qkv_patched,
            )
            break

    self_attn_forward = self_attn_forward.replace(ORIGINAL_O_CODE, PATCHED_O_CODE)
    self_attn_forward = self_attn_forward.replace(
        "def forward(",
        "def lora_fast_attn_forward(",
        1,
    )

    # Load necessary imports
    module_name = attention_cls.__module__
    module = importlib.import_module(module_name)

    items_to_import = []
    for item in dir(module):
        if item in self_attn_forward:
            items_to_import.append(item)

    exec(
        f"from {module_name} import ({', '.join(items_to_import)})",
        globals(),
    )
    exec(self_attn_forward, globals())

    logger.info(f"Patched attention class with LoRA optimizations: {attention_cls.__name__}")
    attention_cls.forward = lora_fast_attn_forward


def apply_lora_fast_post_load(config: SFTConfig, model):
    if hasattr(model, "active_adapters"):
        assert len(model.active_adapters) == 1, (
            "LoraFast does not support LoRA Triton kernels for multiple adapters"
        )
        active_adapter = model.active_adapters[0]
    else:
        active_adapter = model.active_adapter

    lora_config = model.model.peft_config[active_adapter]

    can_patch = config.lora_dropout == 0 and lora_config.bias == "none"
    logger.warning_if(
        "LoraFast optimization requires lora_dropout to be 0 and bias to be 'none'. Skipping LoraFast patching.",
        not can_patch)

    # Choose activation based on model type
    activation = None
    text_config = (
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config
    )
    if hasattr(text_config, "hidden_act"):
        activation = text_config.hidden_act
    elif hasattr(text_config, "hidden_activation"):
        activation = text_config.hidden_activation

    # map activation to supported activation
    if "gelu" in activation:
        # gemma3 uses gelu_pytorch_tanh
        activation = "gelu"
        logger.warning_if(f"Activation {activation} is not supported for LoraFast. Skipping LoraFast patching.",
                          activation not in SUPPORTED_ACTIVATIONS)

    layers = get_layers(model)

    # Patch each layer
    for layer in layers:
        # Add QKV, O fallback implementations to start
        # These will be overwritten later (if some conditions apply)
        for self_attn in find_self_attn_in_layer(layer):
            self_attn.apply_qkv = types.MethodType(original_apply_qkv, self_attn)
            self_attn.apply_o = types.MethodType(original_apply_o, self_attn)

            # Query, key, value patching
            layer_modules = [
                getattr(self_attn, linear_proj)
                for linear_proj in ["q_proj", "k_proj", "v_proj"]
            ]
            can_patch_qkv = all(
                hasattr(module, "lora_A")
                and len(getattr(module, "lora_magnitude_vector", []) or []) == 0
                for module in layer_modules
            )
            if can_patch_qkv:
                # Add optimized implementation
                self_attn.apply_qkv = types.MethodType(apply_lora_qkv, self_attn)
            else:
                logger.warning_once(
                    "Cannot patch some attention QKV projections - requires LoRA adapters and no lora_magnitude_vector (DoRA)"
                )

            # Output patching
            layer_modules = [
                getattr(self_attn, linear_proj) for linear_proj in ["o_proj"]
            ]
            can_patch_o = all(
                hasattr(module, "lora_A")
                and len(getattr(module, "lora_magnitude_vector", []) or []) == 0
                for module in layer_modules
            )
            if can_patch_o:
                self_attn.apply_o = types.MethodType(apply_lora_o, self_attn)
            else:
                logger.warning_once("Cannot patch some attention output projection - requires LoRA adapters and no lora_magnitude_vector (DoRA)")

        for gate_proj, up_proj, down_proj, mlp in find_mlp_in_layer(layer):
            # MLP patching
            can_patch_mlp = all(
                hasattr(proj, "lora_A")
                and len(getattr(proj, "lora_magnitude_vector", []) or []) == 0
                for proj in (gate_proj, up_proj, down_proj)
            )
            if can_patch_mlp:
                apply_fn = APPLY_FN_MAPPING[activation]
                layer.mlp.forward = types.MethodType(apply_fn, mlp)
            else:
                logger.warning_once(
                    "Cannot patch MLP layers - requires LoRA adapters and no lora_magnitude_vector (DoRA)"
                )

    return model


def get_layers(model: PeftModelForCausalLM) -> list[nn.Module]:
    """
    Get the layers of the model. Handles text-only and multimodal models.

    Args:
        model: A PEFT model.

    Returns:
        A list of layers.
    """
    pretrained_model = model.model

    # check for multimodal models first
    if hasattr(pretrained_model, "language_model"):
        return pretrained_model.language_model.layers
    if hasattr(pretrained_model, "model"):
        return pretrained_model.model.layers

    raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported yet. Please create an Issue."
    )


def find_self_attn_in_layer(
        layer: nn.Module,
) -> Generator[Tuple[nn.Module], None, None]:
    # general case of most models
    if hasattr(layer, "self_attn"):
        if all(
                hasattr(layer.self_attn, proj)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            yield layer.self_attn

def find_mlp_in_layer(
        layer: nn.Module,
) -> Generator[Tuple[nn.Module, nn.Module, nn.Module, nn.Module], None, None]:
    # general case of most models
    if hasattr(layer, "mlp"):
        if all(
                hasattr(layer.mlp, proj) for proj in ["gate_proj", "up_proj", "down_proj"]
        ):
            yield layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj, layer.mlp
    # llama4 linearized experts
    if hasattr(layer, "feedforward") and hasattr(layer.feedforward, "shared_expert"):
        mlp = layer.feedforward.shared_expert
        yield mlp.gate_proj, mlp.up_proj, mlp.down_proj, mlp
    if hasattr(layer, "feedforward") and hasattr(layer.feedforward, "experts"):
        if all(
                hasattr(layer.feedforward.experts, proj)
                for proj in ["gate_projs", "up_projs", "down_projs"]
        ):
            for gate_proj, up_proj, down_proj in zip(
                    layer.feedforward.experts.gate_projs,
                    layer.feedforward.experts.up_projs,
                    layer.feedforward.experts.down_projs,
                    strict=False,
            ):
                yield (
                    gate_proj,
                    up_proj,
                    down_proj,
                    FakeMLP(gate_proj, up_proj, down_proj),
                )


def original_apply_qkv(self: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Original implementation of QKV projection without optimizations.

    Args:
        self: The attention module instance.
        hidden_states: Input tensor of shape [batch_size, seq_len, hidden_dim].

    Returns:
        A tuple `(query_states, key_states, value_states)` containing the projected
            states for query, key, and value.
    """
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    return query_states, key_states, value_states

def original_apply_o(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Original implementation of output projection without optimizations.

    Args:
        self: The attention module instance.
        hidden_states: Input tensor of shape `[`batch_size, seq_len, hidden_dim]`.

    Returns:
        The output projection result.
    """
    attn_output = self.o_proj(hidden_states)

    return attn_output


class FakeMLP(nn.Module):
    """
    placeholder MLP for triton patching
    """

    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear

    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj