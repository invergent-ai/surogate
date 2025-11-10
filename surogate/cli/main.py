import importlib.util
import subprocess
import sys
from typing import Dict, Optional

from swift.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pretrain': 'surogate.cli.pretrain',
    'sft': 'surogate.cli.sft',
    'align': 'surogate.cli.align',
    'eval': 'surogate.cli.eval',
    'quantize': 'surogate.cli.quantize',
    'serve': 'surogate.cli.serve',
}
ROUTE_DESCRIPTIONS: Dict[str, str] = {
    'pretrain': 'Pretrain a LLM model',
    'sft': 'Fine-tune a LLM model',
    'align': 'Align a LLM model with Reinforcement Learning',
    'eval': 'Evaluate a LLM model',
    'quantize': 'Quantize a LLM model',
    'serve': 'Serve a LLM model',
}

def cli_main(route_mapping: Optional[Dict[str, str]] = None, is_megatron: bool = False) -> None:
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Please provide an action name to run. Available actions are:\n")
        for method in route_mapping.keys():
            description = ROUTE_DESCRIPTIONS[method]
            print(f"{method}: {description}")
        sys.exit(1)
    method_name = argv[0].replace('_', '-')
    argv = argv[1:]
    file_path = importlib.util.find_spec(route_mapping[method_name]).origin
    python_cmd = sys.executable
    args = [python_cmd, file_path, *argv]
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == '__main__':
    cli_main()