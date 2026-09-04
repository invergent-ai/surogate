"""Put the serve tree on the import path.

The Python reference implementations are imported as `tools.reference.*`, which is how they
are invoked from the command line (`python -m tools.reference.qwen3_5`). That spelling only
resolves with `surogate/serve` on the path, and a test run started from the repository root
does not have it.
"""

import pathlib
import sys

# Appended, not prepended: this directory holds `convert`, `artifact`, `gguf` and other
# names a test may legitimately import from elsewhere, and putting it first would shadow them.
_SERVE = pathlib.Path(__file__).resolve().parents[4] / "surogate" / "serve"
if str(_SERVE) not in sys.path:
    sys.path.append(str(_SERVE))
