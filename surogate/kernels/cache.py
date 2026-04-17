# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Kernel cache: hash-based disk cache for compiled Triton/CuTe cubins.
#
# Cache key: {kernel_name}_{src_hash}_sm{sm}
# Cache dir: ~/.cache/surogate/kernels/ (overridable via SUROGATE_KERNEL_CACHE env)
#
# Usage:
#   cache = KernelCache()
#   manifests = cache.get_or_compile("gdr", src_files, dims, sm, compile_fn)

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "surogate" / "kernels"


def _hash_files(*paths: str | Path) -> str:
    """SHA-256 hash of one or more files' contents, truncated to 12 hex chars."""
    h = hashlib.sha256()
    for p in sorted(str(p) for p in paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]


def _hash_dict(d: dict[str, Any]) -> str:
    """SHA-256 hash of a dict's JSON representation, truncated to 8 hex chars."""
    h = hashlib.sha256()
    h.update(json.dumps(d, sort_keys=True).encode())
    return h.hexdigest()[:8]


class KernelCache:
    """Hash-based disk cache for compiled Triton cubins.

    The cache key combines:
    - Kernel source file hash (detects code changes)
    - SM version (different GPU architectures need different cubins)
    - Shape dimensions (H, K, V — different shapes need different cubins)

    When a cache hit occurs, the existing manifest paths are returned.
    On cache miss, the compile function is called and results are saved.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir or os.environ.get("SUROGATE_KERNEL_CACHE") or _DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compile(
        self,
        name: str,
        src_files: list[str | Path],
        dims: dict[str, int],
        sm: int,
        compile_fn: Callable[[str], dict[str, str]],
    ) -> dict[str, str]:
        """Get cached manifests or compile and cache.

        Args:
            name: Kernel family name (e.g. "gated_delta_rule").
            src_files: Python source files to hash (detects code changes).
            dims: Shape dimensions dict (e.g. {"H": 32, "K": 128, "V": 128}).
            sm: Target SM version (e.g. 120).
            compile_fn: Called with output_dir on cache miss. Must return
                dict mapping kernel names to manifest paths.

        Returns:
            Dict mapping kernel names to absolute manifest file paths.
        """
        src_hash = _hash_files(*src_files)
        dim_hash = _hash_dict(dims)
        cache_key = f"{name}_{src_hash}_sm{sm}_{dim_hash}"
        cache_entry = self.cache_dir / cache_key

        # Check cache
        index_file = cache_entry / "index.json"
        if index_file.exists():
            try:
                index = json.loads(index_file.read_text())
                manifests = index["manifests"]
                # Verify all manifest files still exist
                if all(Path(p).exists() for p in manifests.values()):
                    logger.info("Kernel cache hit: %s (%d kernels)", cache_key, len(manifests))
                    return manifests
                logger.warning("Cache entry %s has missing files, recompiling", cache_key)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt cache entry %s, recompiling", cache_key)

        # Cache miss — compile
        logger.info("Kernel cache miss: %s — compiling...", cache_key)
        cache_entry.mkdir(parents=True, exist_ok=True)

        manifests = compile_fn(str(cache_entry))

        # Make paths absolute
        manifests = {k: str(Path(v).resolve()) for k, v in manifests.items()}

        # Write index
        index_file.write_text(
            json.dumps(
                {
                    "name": name,
                    "src_hash": src_hash,
                    "sm": sm,
                    "dims": dims,
                    "manifests": manifests,
                },
                indent=2,
            )
        )

        logger.info("Compiled and cached %d kernels to %s", len(manifests), cache_entry)
        return manifests

    def clear(self):
        """Remove all cached kernels."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared kernel cache at %s", self.cache_dir)
