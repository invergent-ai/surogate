import yaml

from pathlib import Path
from surogate.utils.dict import DictDefault


def load_cfg(config: str | Path, **kwargs) -> DictDefault:
    """
    Load configuration from a file.

    Args:
        config: Path to the configuration file.
        kwargs: Additional keyword arguments to override config file values.
    Returns:
        `DictDefault` mapping configuration keys to values.
    """
    if isinstance(config, (str, Path)):
        with open(config, encoding="utf-8") as file:
            cfg: DictDefault = DictDefault(yaml.safe_load(file))
        cfg.config_path = config

    # If there are any options passed in the cli, if it is something that seems valid
    # from the yaml, then overwrite the value
    cfg_keys = cfg.keys()
    for key, value in kwargs.items():
        if key in cfg_keys:
            if isinstance(cfg[key], bool):
                cfg[key] = bool(value)
            else:
                cfg[key] = value

    return cfg