import sys

from swift import get_logger

from surogate.eval.eval import SurogateEval

logger = get_logger()

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: eval <config.yaml>")
        sys.exit(1)

    config_path = argv[0]
    SurogateEval(config_path).run()