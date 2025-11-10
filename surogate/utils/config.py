import json
from pathlib import Path

import yaml
from swift import get_logger

from surogate.utils.dict import DictDefault

logger = get_logger()

def load_config(config: str | Path) -> DictDefault:
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as file:
            cfg: DictDefault = DictDefault(yaml.safe_load(file))
        cfg.config_path = config

    cfg_to_log = {
        k: v for k, v in cfg.items() if v is not None
    }

    logger.debug(
        "config:\n%s",
        json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
    )

    return cfg