from contextlib import nullcontext

from peft import PeftModel
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import GradientCheckpointingLayer
from trl.models.activation_offloading import OffloadActivations, NoOpManager, get_act_offloading_ctx_manager

from surogate.utils.logger import get_logger

logger = get_logger()

class ActivationCheckpointingMixin:
    def _apply_activation_checkpointing(self, model):
        if self.config.activation_checkpointing:
            if isinstance(model, PeftModel):
                self.activation_offload_context = self.get_lora_act_offloading_ctx_manager(model)
            else:
                self.activation_offload_context = get_act_offloading_ctx_manager(model, use_streams=True)

            auto_wrap_policy = ModuleWrapPolicy(set((GradientCheckpointingLayer,)))
            apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)

    def training_step(self, *args, **kwargs):
        ac_context = self.activation_offload_context if self.config.activation_checkpointing else nullcontext()
        with ac_context:
            return super().training_step(*args, **kwargs)

    def get_lora_act_offloading_ctx_manager(
            self,
            model: nn.Module,
            use_pin_memory: bool = True,
            use_streams: bool = True,
            min_offload_size: int = 1024,
            max_fwd_stash_size: int = 5,
            warn_if_no_head: bool = True,
    ) -> OffloadActivations:
        activations_handling_ctx = OffloadActivations(
            use_pin_memory=use_pin_memory,
            use_streams=use_streams,
            min_offload_size=min_offload_size,
            max_fwd_stash_size=max_fwd_stash_size,
        )

        # Below is our hack to disable offloading the last output Linear in every
        # step, as the cost for offloading the activation and then soon after bringing
        # it back is expensive.
        output_head_detected = False
        noop_ctx = NoOpManager()

        # Try to get the actual model if it's wrapped
        unwrapped_model = model
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module
        # check for PEFT models
        if hasattr(unwrapped_model, "base_model") and hasattr(unwrapped_model, "peft_config"):
            unwrapped_model = unwrapped_model.base_model

        # Check for different types of output heads
        if hasattr(unwrapped_model, "output"):
            if isinstance(unwrapped_model.output, nn.Module):
                unwrapped_model.output.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                unwrapped_model.output.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True
            elif hasattr(unwrapped_model.output, "linear") and isinstance(
                    unwrapped_model.output.linear, nn.Module
            ):
                unwrapped_model.output.linear.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                unwrapped_model.output.linear.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True
        # Check for HuggingFace model output heads
        elif hasattr(unwrapped_model, "lm_head"):
            unwrapped_model.lm_head.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            unwrapped_model.lm_head.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True
        # Check for decoder-based models
        elif hasattr(unwrapped_model, "decoder"):
            decoder = unwrapped_model.decoder
            if hasattr(decoder, "output"):
                decoder.output.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
                decoder.output.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True
            # Some models have lm_head in the decoder
            elif hasattr(decoder, "lm_head"):
                decoder.lm_head.register_forward_pre_hook(
                    lambda *args: noop_ctx.__enter__()
                )
                decoder.lm_head.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
                output_head_detected = True
        # Check for transformer models with final layer norm
        elif hasattr(unwrapped_model, "final_layer_norm") or hasattr(unwrapped_model, "ln_f"):
            final_norm = (
                    getattr(unwrapped_model, "final_layer_norm", None) or unwrapped_model.ln_f
            )
            final_norm.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            final_norm.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True
        # Check for models with head module
        elif hasattr(unwrapped_model, "head") and isinstance(
            unwrapped_model.head, nn.Module
        ):
            unwrapped_model.head.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            unwrapped_model.head.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True

        if not output_head_detected and warn_if_no_head:
            logger.warning(
                "During activation offloading, no output head was detected. If your model has an output head, it will be "
                "offloaded. This usually greatly slows training, given the large vocabulary size. To change this "
                "behavior, set your output head as model.output and make it an nn.Module. You can disable this warning by "
                "passing `warn_if_no_head=False`."
            )

        for name, module in unwrapped_model.named_modules():
            # Disable offloading for any Liger modules
            if "liger" in name.lower():
                module.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
                module.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )
            # disable offloading for any submodules to fix LoRA training
            if name.endswith("._checkpoint_wrapped_module"):
                for _, sub_module in module.named_modules():
                    sub_module.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
                    sub_module.register_forward_hook(
                        lambda *args: noop_ctx.__exit__(), always_call=True
                    )

        return activations_handling_ctx

