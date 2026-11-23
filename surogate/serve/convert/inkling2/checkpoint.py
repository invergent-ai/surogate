"""The Inkling-2 (``inkling2_loop``) checkpoint as the converter reads it: config, quantisation
declaration, tensor headers, and the dimensions every object's shape is derived from.

Nothing here hard-codes Inkling-2-Mini's sizes: they are read from ``config.json`` and checked
against the tensors, so a smaller checkpoint of the same family (the unit test's synthetic one)
converts through the same code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

MAIN_PREFIX = "model.llm."
VISION_PREFIX = "model.visual."
MTP_FILE = "mtp.safetensors"


@dataclass(frozen=True)
class DraftGeometry:
    layers: int
    local_layers: tuple[int, ...]
    hidden: int
    query_heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    rope_theta: float
    sliding_window: int
    block_size: int
    markov_rank: int
    markov_head_type: str
    target_visits: tuple[int, ...]
    rms_eps: float
    max_position: int


@dataclass(frozen=True)
class Geometry:
    hidden: int
    layers: int
    vocab_padded: int
    vocab: int
    query_heads: int
    kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    conv_kernel: int
    local_layers: tuple[int, ...]
    core_start: int
    core_end: int
    passes: int
    gate_rank: int
    experts: int
    experts_per_token: int
    intermediate: int
    expert_format: tuple[str, ...]  # per layer: "nvfp4" | "mxfp8"
    text: dict
    loop: dict
    draft: DraftGeometry | None

    @property
    def global_layers(self) -> tuple[int, ...]:
        return tuple(l for l in range(self.layers) if l not in self.local_layers)

    def is_global(self, layer: int) -> bool:
        return layer not in self.local_layers

    def is_core(self, layer: int) -> bool:
        return self.core_start <= layer < self.core_end

    @property
    def q_rows(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def kv_rows(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qkvr_slices(self) -> tuple[tuple[str, int], ...]:
        """The fused projection's rows in the reference's order (vLLM ``qkvr``)."""
        return (("wq_du", self.q_rows), ("wr_du", self.query_heads * self.d_rel),
                ("gate_attn", self.q_rows), ("attn_sink_proj", self.query_heads),
                ("wk_dv", self.kv_rows), ("wv_dv", self.kv_rows))

    @property
    def visits(self) -> tuple[tuple[int, int], ...]:
        """(layer, pass) per visit in execution order (vLLM ``layer_visit_ids``)."""
        out = [(l, 0) for l in range(self.core_start)]
        for p in range(self.passes):
            out += [(l, p) for l in range(self.core_start, self.core_end)]
        out += [(l, 0) for l in range(self.core_end, self.layers)]
        return tuple(out)

    def layer_visits(self, layer: int) -> list[int]:
        return [v for v, (l, _) in enumerate(self.visits) if l == layer]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


class Checkpoint:
    """Read-only access to the checkpoint's tensors (main file(s) + ``mtp.safetensors``)."""

    def __init__(self, model_dir: str | Path) -> None:
        self.dir = Path(model_dir)
        self.config = _load_json(self.dir / "config.json")
        if self.config.get("model_type") != "inkling2_loop":
            raise ValueError(f"{self.dir}: model_type {self.config.get('model_type')!r}, "
                             "this converter serves inkling2_loop")
        self.quant = _load_json(self.dir / "hf_quant_config.json")
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = _load_json(index)["weight_map"]
        else:
            single = self.dir / "model-00001-of-00001.safetensors"
            with safe_open(single, "pt") as handle:
                weight_map = {name: single.name for name in handle.keys()}
        self.files: dict[str, str] = dict(weight_map)
        mtp = self.dir / MTP_FILE
        self.has_mtp = mtp.exists()
        self._handles: dict[str, object] = {}
        self.headers: dict[str, tuple[str, tuple[int, ...]]] = {}
        for file in sorted(set(self.files.values())):
            self._read_header(file)
        self.mtp_headers: dict[str, tuple[str, tuple[int, ...]]] = {}
        if self.has_mtp:
            self.mtp_headers = self._header_of(MTP_FILE)
        self.geometry = self._geometry()

    # -- raw access -----------------------------------------------------------------------

    def _header_of(self, file: str) -> dict[str, tuple[str, tuple[int, ...]]]:
        raw = (self.dir / file).open("rb")
        try:
            size = int.from_bytes(raw.read(8), "little")
            header = json.loads(raw.read(size))
        finally:
            raw.close()
        return {name: (entry["dtype"], tuple(entry["shape"]))
                for name, entry in header.items() if name != "__metadata__"}

    def _read_header(self, file: str) -> None:
        for name, value in self._header_of(file).items():
            if self.files.get(name) == file:
                self.headers[name] = value

    def _handle(self, file: str):
        if file not in self._handles:
            self._handles[file] = safe_open(self.dir / file, "pt")
        return self._handles[file]

    def get(self, name: str) -> torch.Tensor:
        """A main-checkpoint tensor by its full name (``model.llm.*``)."""
        return self._handle(self.files[name]).get_tensor(name)

    def get_mtp(self, name: str) -> torch.Tensor:
        return self._handle(MTP_FILE).get_tensor(name)

    def text_names(self) -> list[str]:
        return sorted(n for n in self.headers if n.startswith(MAIN_PREFIX))

    def vision_names(self) -> list[str]:
        return sorted(n for n in self.headers if n.startswith(VISION_PREFIX))

    def source_files(self) -> list[str]:
        files = sorted(set(self.files.values()))
        return files + ([MTP_FILE] if self.has_mtp else [])

    def sha256(self, file: str, known: dict[str, str] | None = None) -> str:
        if known and file in known:
            return known[file]
        digest = hashlib.sha256()
        with (self.dir / file).open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # -- dimensions -----------------------------------------------------------------------

    def _expert_formats(self, layers: int) -> tuple[str, ...]:
        by_tensor: dict[str, str] = {}
        for group in self.quant["model"]["parameter_groups"]:
            compute = group.get("compute_dtype")
            for tensor in group["tensors"]:
                by_tensor[tensor] = compute
        formats = []
        for layer in range(layers):
            name = f"{MAIN_PREFIX}layers.{layer}.mlp.experts.w13_weight"
            declared = by_tensor.get(name)
            dtype = self.headers[name][0]
            if declared == "nvfp4" and dtype == "U8":
                formats.append("nvfp4")
            elif declared == "mxfp8" and dtype == "F8_E4M3":
                formats.append("mxfp8")
            else:
                raise ValueError(f"{name}: hf_quant_config declares {declared!r}, the file stores "
                                 f"{dtype}; expected nvfp4/U8 or mxfp8/F8_E4M3")
            down = by_tensor.get(f"{MAIN_PREFIX}layers.{layer}.mlp.experts.w2_weight")
            if down != declared:
                raise ValueError(f"layer {layer}: w13 is {declared} but w2 is {down}")
        return tuple(formats)

    def _geometry(self) -> Geometry:
        t = self.config["text_config"]
        for a, b in (("head_dim", "swa_head_dim"), ("num_attention_heads", "swa_num_attention_heads"),
                     ("num_key_value_heads", "swa_num_key_value_heads")):
            if t.get(b, t[a]) != t[a]:
                raise ValueError(f"text_config.{a} != {b}: local and global layers must share the "
                                 "head geometry in this artifact")
        if t.get("n_shared_experts", 0):
            raise ValueError("Inkling2 experts have no shared expert")
        loop = t["loop_config"]
        if loop.get("is_varloop"):
            raise ValueError("is_varloop checkpoints are not served (fixed pass count only)")
        layers = int(t["num_hidden_layers"])
        draft = None
        m = self.config.get("mtp_config")
        if m and self.has_mtp:
            draft = DraftGeometry(
                layers=int(m["num_hidden_layers"]), local_layers=tuple(m["local_layer_ids"]),
                hidden=int(m["hidden_size"]), query_heads=int(m["num_attention_heads"]),
                kv_heads=int(m["num_key_value_heads"]), head_dim=int(m["head_dim"]),
                intermediate=int(m["intermediate_size"]), rope_theta=float(m["rope_theta"]),
                sliding_window=int(m["sliding_window_size"]), block_size=int(m["block_size"]),
                markov_rank=int(m["markov_rank"]), markov_head_type=str(m["markov_head_type"]),
                target_visits=tuple(int(v) for v in m["target_layer_ids"]),
                rms_eps=float(m["rms_norm_eps"]), max_position=int(m["max_position_embeddings"]))
        g = Geometry(
            hidden=int(t["hidden_size"]), layers=layers, vocab_padded=int(t["vocab_size"]),
            vocab=int(t["unpadded_vocab_size"]), query_heads=int(t["num_attention_heads"]),
            kv_heads=int(t["num_key_value_heads"]), head_dim=int(t["head_dim"]), d_rel=int(t["d_rel"]),
            rel_extent=int(t["rel_extent"]), sliding_window=int(t["sliding_window_size"]),
            conv_kernel=int(t["sconv_kernel_size"]), local_layers=tuple(int(x) for x in t["local_layer_ids"]),
            core_start=int(loop["core_start"]), core_end=int(loop["core_end"]),
            passes=int(loop["num_passes"]), gate_rank=int(loop["gate_rank"]),
            experts=int(t["n_routed_experts"]), experts_per_token=int(t["num_experts_per_tok"]),
            intermediate=int(t["intermediate_size"]), expert_format=self._expert_formats(layers),
            text=t, loop=loop, draft=draft)
        if draft is not None and draft.target_visits[-1] != len(g.visits):
            raise ValueError("mtp_config.target_layer_ids must end with the visit count (final norm)")
        return g


__all__ = ["Checkpoint", "DraftGeometry", "Geometry", "MAIN_PREFIX", "MTP_FILE", "VISION_PREFIX"]
