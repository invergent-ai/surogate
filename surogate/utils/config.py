import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers.utils import is_torch_bf16_gpu_available

from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.schema.datasets import TextDataset, InstructionDataset, ConversationDataset, BaseDataset
from surogate.utils.schema.enums import SurogateDatasetType

logger = get_logger()


def load_config(config: str | Path) -> DictDefault:
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as file:
            cfg: DictDefault = DictDefault(yaml.safe_load(file))
        cfg.config_path = config

    try:
        device_props = torch.cuda.get_device_properties("cuda")
        gpu_version = "sm_" + str(device_props.major) + str(device_props.minor)
    except:
        gpu_version = None

    cfg = validate_config(
        cfg,
        capabilities={
            "bf16": is_torch_bf16_gpu_available(),
            "n_gpu": int(os.environ.get("WORLD_SIZE", 1)),
            "compute_capability": gpu_version,
        },
        env_capabilities={
            "torch_version": str(torch.__version__).split("+", maxsplit=1)[0]
        },
    )

    cfg_to_log = {
        k: v for k, v in cfg.items() if v is not None
    }

    logger.debug(
        "config:\n%s",
        json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
    )

    return cfg


def validate_config(
        cfg: DictDefault,
        capabilities: Optional[dict] = None,
        env_capabilities: Optional[dict] = None,
) -> DictDefault:
    # Convert datasets to proper format
    if cfg.get("datasets"):
        for idx, ds_cfg in enumerate(cfg["datasets"]):
            if isinstance(ds_cfg, BaseDataset):
                continue

            ds_cfg_dict = DictDefault(**dict(ds_cfg))
            if ds_cfg_dict.get('type') not in SurogateDatasetType.__dict__.keys():
                raise ValueError(f"Dataset type {ds_cfg_dict.get('type')} is not supported.")

            if ds_cfg_dict.get('type') == SurogateDatasetType.text:
                cfg["datasets"][idx] = TextDataset(**ds_cfg_dict)
            elif ds_cfg_dict.get('type') == SurogateDatasetType.instruction:
                cfg["datasets"][idx] = InstructionDataset(**ds_cfg_dict)
            elif ds_cfg_dict.get('type') == SurogateDatasetType.conversation:
                cfg["datasets"][idx] = ConversationDataset(**ds_cfg_dict)
            else:
                raise ValueError(f"Dataset type {ds_cfg_dict.get('type')} is not supported.")


    return DictDefault(**cfg)

