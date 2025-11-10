import sys

from swift import get_logger

from surogate.serve.serve import SurogateServe

logger = get_logger()

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: serve <config.yaml>")
        sys.exit(1)

    config_path = argv[0]
    SurogateServe(config_path).run()