from dataclasses import dataclass, field
from typing import List, Union, Optional

MODEL_ARCH_MAPPING = {}

class LLMArchitecture:
    qwen = 'qwen'
    llama = 'llama'
    nemotron_nano = 'nemotron_nano'

class MLLMArchitecture:
    qwen2_vl = 'qwen2_vl'
    qwen3_vl = 'qwen3_vl'

class ModelArchitecture(LLMArchitecture, MLLMArchitecture):
    pass

@dataclass
class LLMComponents:
    arch_name: str = None

    embedding: str = None
    module_list: str = None
    lm_head: str = None

    q_proj: str = None
    k_proj: str = None
    v_proj: str = None
    o_proj: str = None
    attention: str = None

    mlp: str = None
    down_proj: str = None

    qkv_proj: str = None
    qk_proj: str = None
    qa_proj: str = None
    qb_proj: str = None
    kv_proj: str = None
    kva_proj: str = None
    kvb_proj: str = None

@dataclass
class MLLMComponents(LLMComponents):
    language_model: Union[str, List[str]] = field(default_factory=list)
    aligner: Union[str, List[str]] = field(default_factory=list)
    vision_tower: Union[str, List[str]] = field(default_factory=list)
    generator: Union[str, List[str]] = field(default_factory=list)

    def __post_init__(self):
        for key in ['language_model', 'aligner', 'vision_tower', 'generator']:
            v = getattr(self, key)
            if isinstance(v, str):
                setattr(self, key, [v])
            if v is None:
                setattr(self, key, [])


def register_model_architecture(model_arch: LLMComponents) -> None:
    arch_name = model_arch.arch_name
    if arch_name in MODEL_ARCH_MAPPING:
        raise ValueError(f'The `{arch_name}` has already been registered in the MODEL_ARCH_MAPPING.')
    MODEL_ARCH_MAPPING[arch_name] = model_arch

def get_model_architecture(arch_name: Optional[ModelArchitecture]) -> Optional[MLLMComponents]:
    return MODEL_ARCH_MAPPING.get(arch_name)
