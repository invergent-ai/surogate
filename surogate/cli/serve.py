import sys

from swift import get_logger

logger = get_logger()

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: serve <config.yaml>")
        sys.exit(1)

    config_path = argv[0]
