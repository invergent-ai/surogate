import sys
import argparse

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "env",
        type=str,
        help="The environment id to init",
    )
    
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory (default: ./environments)",
    )
    
    return parser

import verifiers.scripts.init as vf_init

vf_init.README_TEMPLATE = """\
# {env_id_dash}

> Replace the placeholders below, then remove this callout.

### Overview
- **Environment ID**: `{env_id_dash}`
- **Short description**: <one-sentence description>
- **Tags**: <comma-separated tags>

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: <single-turn | multi-turn | tool use>
- **Output format expectations (optional)**: <e.g., plain text, XML tags, JSON schema>
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation:

```bash
surogate vf-eval \
  -m <model_id_or_path> \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{{"key": "value"}}'  # env-specific args as JSON \
  {env_id_dash} 
```

Notes:
- Use `-a` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

"""

if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])
    
    from verifiers.scripts.init import init_environment
    init_environment(
        args.env,
        args.path,
        rewrite_readme=False,
        multi_file=False,
        openenv=False,
    )