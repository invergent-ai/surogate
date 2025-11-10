from pathlib import Path

from surogate.utils.config import load_config


class SurogateServe:
    def __init__(self, config: str | Path):
        self.config = load_config(config)
