"""Hashed n-gram memory: the trainer side of the engine's single-stream ``ngram_ple`` prologue.

The "transplant" arm of ``rl/ngram-memory-2026-10-18/DESIGN.md`` (Appendix A3): a frozen, already trained
n-gram table (Swift-1.5's ``per_layer_token_embd.weight``, IQ4_NL [160, 320,001,536]) feeds the residual
stream entering one decoder layer (layer 1, 0-based) through new trained projections. Per token t with
residual r (the residual entering the memory's layer)::

    rows  : 16 hashed bigram/trigram rows of t (the engine's rows_kernel, bit for bit)
    e     = concat_h IQ4_NL-decode(table[row_h])                    [E = heads * head_dim = 2560]
    k, v  = W_k e, W_v e                                            [C]   (W_k, W_v: [C, E])
    z     = < RMSNorm(k) * g_k , RMSNorm(r) * g_q > / sqrt(C)       (eps 1e-6, no +1 on the gains)
    g     = sigmoid(sign(z) * sqrt(max(|z|, 1e-6)))
    r    <- r + g * v

The dilated convolution of the engine op is frozen at zero (its branch is silu(0) = 0). W_v starts at zero,
so step 0 is the base model bit for bit; W_k ~ N(0, init_std); the gains start at 1. Trained: W_k, W_v,
g_k, g_q (adapter tensors of the LoRA manager; their own learning rate, no weight decay). The contract
with the engine (names, shapes, rounding points): ``rl/ngram-transplant-2026-10-18/INTERFACE.md``.

Option (SFT config ``ngram_memory``)::

    ngram_memory:
      table: /path/table.iq4nl          # raw IQ4_NL rows (the GGUF tensor payload)
      manifest: /path/manifest.json     # GGUF manifest (gguf_manifest.py) with the qwen4exp.ple.* metadata;
                                        # default: manifest.json next to the table
      layer: 1                          # default: the manifest's ple layer
      lr: 1.0e-4                        # default: the run's learning_rate
      only: true                        # memory-only PEFT (no LoRA adapters); default true
      init_std: 0.02
      table_residency: host             # host (pinned, mapped; default) or device
      zero_token_ids: [248056, 248057]  # memory output forced to 0 (the engine's image/video pad rule)

With no ``ngram_memory`` the compiled IR is unchanged. With it the model's IR config carries
``ngram_memory_layer`` and ``ngram_memory_spec`` (the JSON :meth:`NgramMemoryConfig.spec` returns).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The engine's limits (csrc/src/serve/api/ops/ngram_ple.h).
MAX_HEADS = 16
MAX_NGRAM = 3
#: Qwen3.5/3.6 image and video pad tokens.
DEFAULT_ZERO_TOKEN_IDS = (248056, 248057)
#: IQ4_NL code values (kvalues_iq4nl).
IQ4NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


@dataclass(frozen=True)
class TableMeta:
    """The hash constants and table geometry of one n-gram table."""

    multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    ngram_size: int
    heads_per_ngram: int
    head_dim: int
    row_count: int
    eos_token_id: int
    layer: int = 1
    conv_kernel: int = 4
    image_token_id: int | None = None
    table_offset: int = 0

    @property
    def heads(self) -> int:
        return len(self.head_offsets)

    @property
    def embed_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def row_bytes(self) -> int:
        return self.head_dim // 32 * 18

    def validate(self) -> None:
        if self.ngram_size < 2 or self.ngram_size > MAX_NGRAM:
            raise ValueError(f"ngram_memory: ngram_size must be 2 or 3, got {self.ngram_size}")
        if self.heads <= 0 or self.heads > MAX_HEADS or self.heads != (self.ngram_size - 1) * self.heads_per_ngram:
            raise ValueError(f"ngram_memory: {self.heads} heads for ngram_size {self.ngram_size} and "
                             f"heads_per_ngram {self.heads_per_ngram}")
        if len(self.head_vocab_sizes) != self.heads or len(self.multipliers) < self.ngram_size:
            raise ValueError("ngram_memory: head_vocab_sizes / multipliers do not match the heads")
        if self.head_dim <= 0 or self.head_dim % 32:
            raise ValueError(f"ngram_memory: head_dim must be a multiple of 32, got {self.head_dim}")
        if not 0 < self.row_count < 2**31:
            raise ValueError(f"ngram_memory: row_count {self.row_count} out of range")
        for h, (off, size) in enumerate(zip(self.head_offsets, self.head_vocab_sizes)):
            if off < 0 or size <= 0 or off + size > self.row_count:
                raise ValueError(f"ngram_memory: head {h} rows [{off}, {off + size}) exceed the table")
        for m in self.multipliers:
            if not 0 <= m < 2**64:
                raise ValueError("ngram_memory: multipliers must be uint64")

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "TableMeta":
        """From a GGUF manifest (``rl/ngram-transplant-2026-10-18/gguf_manifest.py``: ``meta`` with the
        ``qwen4exp.ple.*`` keys, ``tensors[0]`` = ``per_layer_token_embd.weight`` [head_dim, rows]) or a
        plain dict with this class's field names."""
        if "meta" not in manifest:
            fields = {k: manifest[k] for k in cls.__dataclass_fields__ if k in manifest}
            for k in ("multipliers", "head_offsets", "head_vocab_sizes"):
                fields[k] = tuple(int(x) for x in fields[k])
            return cls(**fields)
        meta = manifest["meta"]
        prefix = "qwen4exp.ple."
        tensor = next(t for t in manifest["tensors"] if t["name"] == "per_layer_token_embd.weight")
        if tensor.get("type", "IQ4_NL") != "IQ4_NL":
            raise ValueError(f"ngram_memory: the table must be IQ4_NL, got {tensor.get('type')}")
        head_dim, rows = (int(x) for x in tensor["dims"])
        layers = meta.get(prefix + "layers", [1])
        if len(layers) != 1:
            raise ValueError(f"ngram_memory: one memory layer is supported, the manifest has {layers}")
        return cls(
            multipliers=tuple(int(x) for x in meta[prefix + "layer_multipliers"]),
            head_offsets=tuple(int(x) for x in meta[prefix + "head_offsets"]),
            head_vocab_sizes=tuple(int(x) for x in meta[prefix + "head_vocab_sizes"]),
            ngram_size=int(meta[prefix + "ngram_size"]),
            heads_per_ngram=int(meta[prefix + "heads_per_ngram"]),
            head_dim=head_dim,
            row_count=rows,
            eos_token_id=int(meta[prefix + "eos_token_id"]),
            layer=int(layers[0]),
            conv_kernel=int(meta.get(prefix + "conv_kernel", 4)),
            image_token_id=int(meta[prefix + "image_token_id"]) if prefix + "image_token_id" in meta else None,
        )


@dataclass(frozen=True)
class NgramMemoryConfig:
    table: str
    meta: TableMeta
    layer: int
    lr: float | None = None
    only: bool = True
    init_std: float = 0.02
    table_residency: str = "host"
    zero_token_ids: tuple[int, ...] = field(default=DEFAULT_ZERO_TOKEN_IDS)
    seed: int = 0
    manifest: str | None = None
    #: The run's learning rate (set by the trainer): the memory trains at ``lr / base_lr`` times it.
    base_lr: float | None = None

    @classmethod
    def parse(cls, value: Any) -> "NgramMemoryConfig | None":
        """``None`` / ``{}`` / ``False`` -> None (no memory). A dict as documented in the module docstring, or
        an already resolved dict from :meth:`to_dict` (``meta`` inline)."""
        if not value:
            return None
        if isinstance(value, NgramMemoryConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"ngram_memory must be a mapping, got {type(value).__name__}")
        allowed = {"table", "manifest", "meta", "layer", "lr", "only", "init_std", "table_residency",
                   "zero_token_ids", "seed", "base_lr"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"ngram_memory: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)})")
        if "table" not in value:
            raise ValueError("ngram_memory: 'table' (the IQ4_NL table file) is required")
        table = os.path.abspath(str(value["table"]))  # the export and the engine read it from elsewhere
        if isinstance(value.get("meta"), dict):
            meta = TableMeta.from_manifest(value["meta"])
            manifest = value.get("manifest")
        else:
            manifest = os.path.abspath(str(value.get("manifest") or Path(table).with_name("manifest.json")))
            if not os.path.exists(manifest):
                raise ValueError(f"ngram_memory: manifest {manifest} not found")
            meta = TableMeta.from_manifest(json.loads(Path(manifest).read_text()))
        meta.validate()
        layer = int(value.get("layer", meta.layer))
        residency = str(value.get("table_residency", "host"))
        if residency not in ("host", "device"):
            raise ValueError(f"ngram_memory: table_residency must be 'host' or 'device', got {residency!r}")
        zero = tuple(int(x) for x in value.get("zero_token_ids", DEFAULT_ZERO_TOKEN_IDS))
        if len(zero) > 4:
            raise ValueError("ngram_memory: at most 4 zero_token_ids")
        lr = value.get("lr")
        cfg = cls(table=table, meta=meta, layer=layer, lr=float(lr) if lr is not None else None,
                  only=bool(value.get("only", True)), init_std=float(value.get("init_std", 0.02)),
                  table_residency=residency, zero_token_ids=zero, seed=int(value.get("seed", 0)),
                  manifest=manifest,
                  base_lr=float(value["base_lr"]) if value.get("base_lr") is not None else None)
        if layer < 0:
            raise ValueError("ngram_memory: layer must be >= 0")
        if cfg.lr is not None and not cfg.lr > 0:
            raise ValueError("ngram_memory: lr must be positive")
        if not cfg.init_std > 0:
            raise ValueError("ngram_memory: init_std must be positive")
        return cfg

    def check_table_file(self) -> None:
        size = os.path.getsize(self.table)
        need = self.meta.table_offset + self.meta.row_count * self.meta.row_bytes
        if size < need:
            raise ValueError(f"ngram_memory: table {self.table} holds {size} bytes, the manifest needs {need}")

    def to_dict(self) -> dict[str, Any]:
        """A resolved form that :meth:`parse` reads back (the manifest inlined)."""
        m = self.meta
        return {"table": self.table, "manifest": self.manifest, "layer": self.layer, "lr": self.lr,
                "only": self.only, "init_std": self.init_std, "table_residency": self.table_residency,
                "zero_token_ids": list(self.zero_token_ids), "seed": self.seed, "base_lr": self.base_lr,
                "meta": {"multipliers": list(m.multipliers), "head_offsets": list(m.head_offsets),
                         "head_vocab_sizes": list(m.head_vocab_sizes), "ngram_size": m.ngram_size,
                         "heads_per_ngram": m.heads_per_ngram, "head_dim": m.head_dim, "row_count": m.row_count,
                         "eos_token_id": m.eos_token_id, "layer": m.layer, "conv_kernel": m.conv_kernel,
                         "image_token_id": m.image_token_id, "table_offset": m.table_offset}}

    def lr_scale(self) -> float:
        """The memory's learning rate relative to the run's (1 without ``lr``)."""
        if self.lr is None:
            return 1.0
        if not self.base_lr:
            raise ValueError("ngram_memory: lr needs the run's learning_rate (base_lr)")
        return self.lr / self.base_lr

    def spec(self) -> dict[str, Any]:
        """The JSON the C++ runtime parses (runtime/ngram/ngram_memory.cpp, ngram_memory::Spec::parse)."""
        m = self.meta
        lr_scale = self.lr_scale()
        return {"layer": self.layer, "ngram_size": m.ngram_size, "heads": m.heads,
                "heads_per_ngram": m.heads_per_ngram, "head_dim": m.head_dim, "row_count": m.row_count,
                "row_bytes": m.row_bytes, "multipliers": list(m.multipliers), "head_offsets": list(m.head_offsets),
                "head_vocab_sizes": list(m.head_vocab_sizes), "eos_token_id": m.eos_token_id,
                "zero_token_ids": list(self.zero_token_ids), "table": self.table, "table_offset": m.table_offset,
                "table_residency": self.table_residency, "eps": 1e-6, "init_std": self.init_std, "seed": self.seed,
                "lr_scale": lr_scale, "embed_dim": m.embed_dim, "conv_kernel": m.conv_kernel,
                "conv_dilation": m.ngram_size}


def setup_ngram_memory(model, ngram_memory, n_layers: int) -> NgramMemoryConfig | None:
    """Validate the option for a model of ``n_layers`` layers and record the two IR config attributes the
    runtime reads (``ngram_memory_layer``, ``ngram_memory_spec``). Off -- no option -- records nothing."""
    config = NgramMemoryConfig.parse(ngram_memory)
    if config is None:
        return None
    if config.layer >= n_layers:
        raise ValueError(f"ngram_memory: layer {config.layer} is not a layer of this {n_layers}-layer model")
    model.ngram_memory_layer = config.layer
    model.ngram_memory_spec = json.dumps(config.spec(), sort_keys=True)
    return config


# ----------------------------------------------------------------------------------------------------------
# NumPy references (tests, export checks). Mirror the engine kernels; no torch needed.


def ngram_rows_reference(ids, positions, meta: TableMeta, carry=None):
    """rows [N, heads] (int64) for token ids [B, T] with per-token positions [B, T] (a conversation starts at
    position 0) and an optional carry [B, 2] (the two tokens before the chunk, -1 = none), with the trainer's
    rule: predecessor k is inside the conversation iff position >= k; a missing or EOS predecessor cuts the
    context (it and every older predecessor read as EOS)."""
    import numpy as np

    ids = np.asarray(ids, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    B, T = ids.shape
    eos = meta.eos_token_id
    full = np.full((B, T + 2), -1, dtype=np.int64)
    if carry is not None:
        full[:, :2] = np.asarray(carry, dtype=np.int64)
    full[:, 2:] = ids
    ctx = [ids]
    cut = np.zeros((B, T), dtype=bool)
    for k in range(1, meta.ngram_size):
        tok = full[:, 2 - k: 2 - k + T]
        tok = np.where(positions >= k, tok, -1)
        tok = np.where(cut, -1, tok)
        cut = cut | (tok < 0) | (tok == eos)
        ctx.append(np.where(cut, eos, tok))
    mult = [np.uint64(m) for m in meta.multipliers]
    per = meta.heads // (meta.ngram_size - 1)
    with np.errstate(over="ignore"):
        mixed2 = (ctx[0].astype(np.uint64) * mult[0]) ^ (ctx[1].astype(np.uint64) * mult[1])
        mixed3 = mixed2 ^ (ctx[2].astype(np.uint64) * mult[2]) if meta.ngram_size > 2 else mixed2
    rows = np.empty((B, T, meta.heads), dtype=np.int64)
    for h in range(meta.heads):
        mixed = mixed2 if h < per else mixed3
        rows[:, :, h] = (mixed % np.uint64(meta.head_vocab_sizes[h])).astype(np.int64) + meta.head_offsets[h]
    return rows.reshape(B * T, meta.heads)


def iq4nl_decode_rows(raw, head_dim: int):
    """raw uint8 [n, head_dim/32*18] IQ4_NL rows -> float32 [n, head_dim] (exact fp32 products)."""
    import numpy as np

    kv = np.array(IQ4NL_VALUES, dtype=np.float32)
    blk = np.asarray(raw, dtype=np.uint8).reshape(-1, head_dim // 32, 18)
    scale = blk[:, :, :2].copy().view(np.float16).astype(np.float32)
    qs = blk[:, :, 2:]
    v = np.concatenate([kv[qs & 15], kv[qs >> 4]], axis=2) * scale
    return v.reshape(-1, head_dim)


def iq4nl_encode_rows(values, seed: int = 0):
    """float [n, head_dim] -> IQ4_NL uint8 rows (a simple nearest-code quantizer for synthetic test tables)."""
    import numpy as np

    x = np.asarray(values, dtype=np.float32)
    n, d = x.shape
    blocks = x.reshape(n, d // 32, 32)
    amax = np.abs(blocks).max(axis=2, keepdims=True)
    scale = (amax / 127.0).astype(np.float16).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1e-8), scale)
    kv = np.array(IQ4NL_VALUES, dtype=np.float32)
    codes = np.abs(blocks[..., None] / scale[..., None] - kv).argmin(axis=-1).astype(np.uint8)
    out = np.zeros((n, d // 32, 18), dtype=np.uint8)
    out[:, :, :2] = scale.astype(np.float16).reshape(n, d // 32, 1).view(np.uint8).reshape(n, d // 32, 2)
    out[:, :, 2:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return out.reshape(n, d // 32 * 18)
