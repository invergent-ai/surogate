"""Python wrapper for the serve engine (_surogate_serve extension).

The extension is a separate module from _surogate: the serve engine is
arch-gated (sm_120-class GPUs today) and its kernel payload is large, so it
never rides along with the training binding. `make serve-build` produces and
copies it next to the training extensions; in a source checkout the loader
also finds it in csrc/build-serve.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional


def _load_extension():
    try:
        return importlib.import_module("surogate._surogate_serve")
    except ImportError:
        pass
    try:
        return importlib.import_module("_surogate_serve")
    except ImportError:
        pass
    repo = Path(__file__).resolve().parent.parent.parent
    for candidate in sorted((repo / "csrc" / "build-serve").glob("_surogate_serve*.so")):
        spec = importlib.util.spec_from_file_location("_surogate_serve", candidate)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        "_surogate_serve is not built. Run `make serve-build` (requires an "
        "sm_120-class toolchain; see csrc/src/serve/PATCHES.md #26 for the "
        "architecture ladder)."
    )


class Engine:
    """Blocking generation against a .sinfer artifact.

    >>> engine = Engine("model.sinfer")
    >>> result = engine.generate("Hello!", max_new=64, greedy=True)
    >>> print(result["content"])

    Quant profile and checkpoint knobs are the SUROGATE_SERVE_* environment
    variables (set before construction): SUROGATE_SERVE_PREFILL_QUANT=fp4,
    SUROGATE_SERVE_DEFER_REWRITE_CHECKPOINT=1.
    """

    def __init__(
        self,
        artifact: str | Path,
        *,
        device: int = 0,
        max_context: int = 4096,
        kv_capacity: Optional[int] = None,
        prefill_chunk: int = 2048,
        max_concurrency: int = 1,
        use_cuda_graph: bool = True,
    ) -> None:
        ext = _load_extension()
        self._engine = ext.Engine(
            str(artifact),
            device=device,
            max_context=max_context,
            kv_capacity=kv_capacity,
            prefill_chunk=prefill_chunk,
            max_concurrency=max_concurrency,
            use_cuda_graph=use_cuda_graph,
        )

    def count_tokens(self, prompt: str, *, enable_thinking: bool = True) -> int:
        return self._engine.count_tokens(prompt, enable_thinking=enable_thinking)

    def generate(
        self,
        prompt: str,
        *,
        enable_thinking: bool = True,
        max_new: int = 512,
        greedy: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ) -> dict:
        return self._engine.generate(
            prompt,
            enable_thinking=enable_thinking,
            max_new=max_new,
            greedy=greedy,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop or [],
            on_delta=on_delta,
        )

    def memory_summary(self) -> dict:
        return self._engine.memory_summary()
