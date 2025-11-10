import json
import sys

from swift import get_logger

from surogate.pretrain.config import load_cfg

logger = get_logger()

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: pretrain <config.yaml>")
        sys.exit(1)

    config_path = argv[0]
    cfg = load_cfg(config_path)

    cfg_to_log = {
        k: v for k, v in cfg.items() if v is not None
    }

    logger.debug(
        "config:\n%s",
        json.dumps(cfg_to_log, indent=2, default=str, sort_keys=True),
    )

    import pyllmq

    gpus = pyllmq.get_num_gpus()
    print(gpus)