from transformers import PretrainedConfig, AutoTokenizer, AutoConfig

from surogate.utils.dict import DictDefault
import addict


def load_model_config(cfg: DictDefault) -> PretrainedConfig | addict.Dict:
    try:
        model_config = AutoConfig.from_pretrained(cfg.get('model'), trust_remote_code=True)
    except ValueError as error:
        raise error

    return model_config

