# Golden data generators for DSL ops

This folder contains scripts that generate tiny, deterministic golden cases for DSL compiled ops.
Run them with the repo virtual environment:

```
.venv/bin/python tests/ops/generate_goldens.py --list
.venv/bin/python tests/ops/generate_goldens.py --op matmul_swiglu
.venv/bin/python tests/ops/generate_goldens.py --all
```

Output files are JSON under `tests/ops/goldens/`.
Each JSON has:
- `op`: op name
- `case`: case id
- `inputs`, `outputs`, optional `grads`
- `attrs` and `meta`

Add new generators in `tests/ops/generate_goldens.py` and update `OP_GENERATORS`.
