import types

from peft import PeftModel
from transformers import PreTrainedModel

from surogate.core.model.kernels.fast_lora import apply_lora_qkv, apply_lora_o, apply_lora_mlp_swiglu
from surogate.utils.logger import get_logger

logger = get_logger()

def patch_peft_model(model: PeftModel) -> PeftModel:
    # Do patching
    n_mlp = 0
    n_qkv = 0
    n_o = 0
    model_type = model.model_info.native_model_type

    active_adapter = (
        model.active_adapters[0]
        if hasattr(model, "active_adapters")
        else model.active_adapter
    )

    # Get dropout and bias
    lora_dropout = model.peft_config[active_adapter].lora_dropout
    bias = model.peft_config[active_adapter].bias

    if lora_dropout > 0 or bias != "none":
        return model

    for idx, layer in enumerate(model.model.model.layers):
        # MLP patching
        mlp_module = layer.mlp
        gate_proj = mlp_module.gate_proj
        up_proj = mlp_module.up_proj
        down_proj = mlp_module.down_proj

        if (
                hasattr(gate_proj, "lora_A")
                and hasattr(up_proj, "lora_A")
                and hasattr(down_proj, "lora_A")
                and (getattr(gate_proj, "base_layer", gate_proj).bias is None)
                and (getattr(up_proj, "base_layer", up_proj).bias is None)
                and (getattr(down_proj, "base_layer", down_proj).bias is None)
                and (
                len(getattr(gate_proj, "lora_magnitude_vector", []) or [])
                == 0
        )
                and (
                len(getattr(up_proj, "lora_magnitude_vector", []) or [])
                == 0
        )
                and (
                len(getattr(down_proj, "lora_magnitude_vector", []) or [])
                == 0
        )
        ):
            # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            if hasattr(mlp_module, "_unsloth_forward"):
                # then we've patched the mlp to use TiledMLP
                mlp_module._unsloth_forward = types.MethodType(
                    apply_lora_mlp_swiglu, mlp_module
                )
            else:
                mlp_module.forward = types.MethodType(
                    apply_lora_mlp_swiglu, mlp_module
                )
            n_mlp += 1
        else:
            logger.warning_once(
                "Not an error, but Unsloth cannot patch MLP layers with our manual autograd engine since either LoRA adapters")

        # QKV attention patching
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        if (
            hasattr(q_proj, "lora_A")
            and hasattr(k_proj, "lora_A")
            and hasattr(v_proj, "lora_A")
            and (getattr(q_proj, "base_layer", q_proj).bias is None)
            and (getattr(k_proj, "base_layer", k_proj).bias is None)
            and (getattr(v_proj, "base_layer", v_proj).bias is None)
            and (len(getattr(q_proj, "lora_magnitude_vector", []) or []) == 0)
            and (len(getattr(k_proj, "lora_magnitude_vector", []) or []) == 0)
            and (len(getattr(v_proj, "lora_magnitude_vector", []) or []) == 0)
        ):
            layer.self_attn.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            if model_type == "qwen2":
                n_qkv += 1
            else:
                logger.warning_once(
                    "Cannot patch Attention layers with Unsloth's manual autograd engine since either LoRA adapters\n"
                    "are not enabled or a bias term (like in Qwen) is used.")

        # O attention patching
        o_proj = layer.self_attn.o_proj
        if (
                hasattr(o_proj, "lora_A")
                and (getattr(o_proj, "base_layer", o_proj).bias is None)
                and (len(getattr(o_proj, "lora_magnitude_vector", []) or []) == 0)
        ):
            layer.self_attn.apply_o = apply_lora_o
            n_o += 1
        else:
            logger.warning_once(
                "Not an error, but Unsloth cannot patch O projection layer with our manual autograd engine since either LoRA adapters\n"
                "are not enabled or a bias term (like in Qwen) is used."
            )

    logger.debug(
        f"Patched {len(model.model.model.layers)} layers with "
        f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers.",
    )

    return model


