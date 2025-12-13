from typing import TypeVar, Optional, Union, List

from transformers import PretrainedConfig

from surogate.core.model.hf_config import HfConfigFactory
from surogate.utils.logger import get_logger

logger = get_logger()

_T = TypeVar('_T')


class AttnImpl:
    attn_impl_keys = ['_attn_implementation', 'attn_implementation', 'llm_attn_implementation']
    use_flash_attn_keys = ['_flash_attn_2_enabled', 'use_flash_attn', '_use_flash_attention_2']

    @staticmethod
    def to_use_flash_attn(attn_impl: Optional[str], auto_value: _T = None) -> Union[bool, _T]:
        if attn_impl is None:
            return auto_value
        return attn_impl in {'flash_attn', 'flash_attention_2'}

    @staticmethod
    def update_attn_impl(
            config: PretrainedConfig,
            attn_impl: Optional[str],
            attn_impl_keys: Optional[List[str]] = None
    ) -> None:
        if attn_impl is None:
            return
        use_flash_attn = AttnImpl.to_use_flash_attn(attn_impl)
        if use_flash_attn:
            attn_impl = 'flash_attention_2'
        if isinstance(attn_impl_keys, str):
            attn_impl_keys = [attn_impl_keys]
        attn_impl_keys = attn_impl_keys or AttnImpl.attn_impl_keys
        for key in attn_impl_keys:
            HfConfigFactory.set_config_attr(config, key, attn_impl, include_vit=True, ensure_set=False)
        for key in AttnImpl.use_flash_attn_keys:
            HfConfigFactory.set_config_attr(config, key, use_flash_attn, include_vit=True, ensure_set=False)
