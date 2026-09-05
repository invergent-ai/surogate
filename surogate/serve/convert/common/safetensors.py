"""Lazy reads from explicitly selected indexed or single-file safetensors sources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shard: str
    shape: tuple[int, ...]
    dtype: str



#: Interchangeable module spellings for one tensor, applied in both directions. These are
#: naming conventions different exporters chose for the same module, not architecture
#: knowledge, so the list stays short and general rather than growing per model.
_SEGMENT_ALIASES: tuple[tuple[str, str], ...] = (
    ("block_sparse_moe.gate", "mlp.gate"),
    ("block_sparse_moe.experts", "mlp.experts"),
)


def name_spellings(name: str):
    """Every spelling one tensor may be stored under, most specific first.

    Three format conventions, applied to every checkpoint rather than listed per model: the
    VL-style `model.language_model.*` nesting folds to `model.*`; a trailing `.weight` is
    optional, because a stacked-expert tensor is a bare Parameter in some exports and a Module
    weight in others; and the aliases above name the same module two ways.
    """
    seeds = [name]
    prefix = "model.language_model."
    if name.startswith(prefix):
        seeds.append("model." + name[len(prefix):])
    elif name.startswith("model."):
        # and the other direction: a flat checkpoint answering a nested request
        seeds.append(prefix + name[len("model."):])
    for seed in list(seeds):
        for left, right in _SEGMENT_ALIASES:
            if left in seed:
                seeds.append(seed.replace(left, right))
            elif right in seed:
                seeds.append(seed.replace(right, left))
    for seed in list(seeds):
        seeds.append(seed[: -len(".weight")] if seed.endswith(".weight") else seed + ".weight")
    seen: set[str] = set()
    for candidate in seeds:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


class ShardReader:
    """Resolve a safetensors source while keeping at most one file handle open."""

    def __init__(
        self,
        model_dir: str | Path,
        index_filename: str = "model.safetensors.index.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / index_filename).read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._canonicalize_names()
        self._reset_handle()

    @classmethod
    def for_directory(cls, model_dir: str | Path) -> ShardReader:
        """The reader a model directory needs, sharded or not.

        A checkpoint small enough to fit one file ships no
        `model.safetensors.index.json`, and asking for one is a FileNotFoundError
        rather than a useful message. Both layouts are ordinary, so both are read
        here: the index when it exists, the single file when it does not.
        """
        root = Path(model_dir)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            return cls(root)
        singles = sorted(root.glob("*.safetensors"))
        if len(singles) == 1:
            return cls.from_file(singles[0])
        if not singles:
            raise FileNotFoundError(f"no safetensors weights in {root}")
        raise FileNotFoundError(
            f"{root} holds {len(singles)} safetensors files and no index naming them"
        )

    @classmethod
    def from_index(cls, index_path: str | Path) -> ShardReader:
        path = Path(index_path)
        return cls(path.parent, path.name)

    @classmethod
    def from_file(cls, tensor_path: str | Path) -> ShardReader:
        path = Path(tensor_path)
        reader = cls.__new__(cls)
        reader.model_dir = path.parent
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            reader.weight_map = {name: path.name for name in handle.keys()}
        reader._canonicalize_names()
        reader._reset_handle()
        return reader

    def _canonicalize_names(self) -> None:
        """Fold the VL-style `model.language_model.*` nesting to `model.*`.

        surogate vendor patch (PATCHES.md #16): the family recipes address
        sources in the flat dialect (`model.layers...`, matching GGUF-bridged
        checkpoints); official Qwen3.5 releases nest the text tower under
        `model.language_model.`. When both spellings exist for one tensor the
        checkpoint is ambiguous and left untouched.
        """
        prefix = "model.language_model."
        folded = {}
        for name, shard in self.weight_map.items():
            if name.startswith(prefix):
                flat = "model." + name[len(prefix):]
                if flat in self.weight_map:
                    return  # ambiguous; serve the names as stored
                folded[flat] = (name, shard)
        if not folded:
            return
        self._aliases: dict[str, str] = {}
        for flat, (stored, shard) in folded.items():
            del self.weight_map[stored]
            self.weight_map[flat] = shard
            self._aliases[flat] = stored

    def _stored_name(self, name: str) -> str:
        return getattr(self, "_aliases", {}).get(name, name)

    def _resolve(self, name: str) -> str:
        """The stored name for a requested one, across the spellings of one tensor.

        These are format conventions, not architecture knowledge, so they are applied to every
        checkpoint rather than listed per model:

        - the VL-style `model.language_model.*` nesting folds to `model.*`;
        - a trailing `.weight` is optional, because a stacked-expert tensor is a bare Parameter
          in some exports (`mlp.experts.down_proj`) and a Module weight in others;
        - a few module names are spelled differently by different families for the same thing,
          the MoE router above all (`block_sparse_moe.gate` and `mlp.gate`).
        """
        for candidate in self._spellings(name):
            if candidate in self.weight_map:
                return candidate
        return name

    def _spellings(self, name: str):
        return name_spellings(name)

    def _reset_handle(self) -> None:
        self._current_shard: str | None = None
        self._context = None
        self._handle = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def has(self, name: str) -> bool:
        return self._resolve(name) in self.weight_map

    def _open_shard(self, shard: str):
        if shard == self._current_shard:
            return self._handle
        self.close()
        self._context = safe_open(
            str(self.model_dir / shard),
            framework="pt",
            device="cpu",
        )
        self._handle = self._context.__enter__()
        self._current_shard = shard
        return self._handle

    def get(self, name: str) -> torch.Tensor:
        name = self._resolve(name)
        shard = self.weight_map[name]
        handle = self._open_shard(shard)
        return handle.get_tensor(self._stored_name(name))

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        self.close()
        by_shard: dict[str, list[str]] = {}
        requested: dict[str, str] = {}
        for name in names:
            resolved = self._resolve(name)
            requested[resolved] = name
            shard = self.weight_map[resolved]
            by_shard.setdefault(shard, []).append(resolved)

        result: dict[str, TensorMetadata] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(
                str(self.model_dir / shard),
                framework="pt",
                device="cpu",
            ) as handle:
                for name in shard_names:
                    tensor_slice = handle.get_slice(self._stored_name(name))
                    original = requested.get(name, name)
                    result[original] = TensorMetadata(
                        name=original,
                        shard=shard,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=str(tensor_slice.get_dtype()),
                    )
        return result

    def close(self) -> None:
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._current_shard = None
        self._context = None
        self._handle = None

    def __enter__(self) -> ShardReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
