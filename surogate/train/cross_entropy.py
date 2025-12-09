from functools import partial

from surogate.core.config.sft_config import SFTConfig
from surogate.core.model.utils import get_causal_lm_model_cls_prefix
from surogate.utils.debug import debug_breakpoint
from surogate.utils.logger import get_logger

logger = get_logger()

_CCE_INSTALL_MESSAGE = (
    "Please install Axolotl's fork of cut_cross_entropy with transformers support using "
    '`pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@8a1a0ec"`'
)

def apply_cross_entropy_patch(config: SFTConfig):
    logger.info("Applying Cut Cross-Entropy (CCE) patch")

    from cut_cross_entropy.transformers.patch import AXOLOTL_CCE_FORK
    if not AXOLOTL_CCE_FORK:
        raise ImportError(
            "Axolotl's fork of cut_cross_entropy is not installed. "
            + _CCE_INSTALL_MESSAGE
        )
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)

    _patch_llama_like(config.model_info.native_model_type)

    from cut_cross_entropy.transformers.patch import cce_patch
    cce_patch(config.model_info.native_model_type)


def _patch_llama_like(model_type: str):
    from cut_cross_entropy.transformers.patch import PATCH_FNS

    def patch_generic(maybe_model, patch_options, model_type: str):
        import cut_cross_entropy.transformers.llama
        from cut_cross_entropy.transformers.llama import cce_forward
        debug_breakpoint()
        try:
            # Dynamically import the module and CausalLM class
            module_path = f"transformers.models.{model_type}.modeling_{model_type}"
            model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
            module = __import__(
                module_path, fromlist=[f"{model_cls_prefix}ForCausalLM"]
            )
            model_cls = getattr(module, f"{model_cls_prefix}ForCausalLM")
            cut_cross_entropy.transformers.llama._PATCH_OPTS = patch_options
            model_cls.forward = cce_forward
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Could not import ForCausalLM class for model_type: {model_type}. "
                f"Error: {str(e)}"
            ) from e

    if model_type not in PATCH_FNS:
        PATCH_FNS[model_type] = partial(patch_generic, model_type=model_type)
