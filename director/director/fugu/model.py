"""The Fugu selection router: a frozen backbone featurizer + a lightweight head.

The featurizer turns a query (and, for multi-turn, the running transcript) into a
decision-token hidden state ``h ∈ R^d``; the head projects ``h`` to ``L`` logits,
one per worker. The router is *selection-only* (no roles) and *decision-only* (the
backbone is never used to generate text).

The featurizer is an abstraction so the heavy HF backbone can be swapped for a tiny
deterministic stub in offline tests. ``trainable_vector`` / ``load_vector`` expose the
full trainable set (head + SVF scales) as one flat vector for sep-CMA-ES.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from .svf import apply_svf, count_trainable

# SVF targets: the seven linear projections of one decoder layer (layer 26, the
# second-to-last of Qwen3-0.6B's 28 layers), following the reference recipe
# ("seven layer-26 projections"). The reference also lists embed_tokens and lm_head,
# but for *selection-only* Fugu reading the hidden state at position -2 those are
# omitted on purpose: lm_head is applied after that hidden state (a no-op on routing),
# and SVD of the ~152k-vocab embedding would materialize a multi-hundred-MB U factor.
# Seven 1024-dim singular-value vectors => 7168 SVF offsets, well under ~20K.
DEFAULT_SVF_TARGETS = [
    r"layers\.26\.self_attn\.(q|k|v|o)_proj$",
    r"layers\.26\.mlp\.(gate|up|down)_proj$",
]


def window_token_ids(ids: list[int], window: int, head_tokens: int, strategy: str) -> list[int]:
    """Fit a token sequence into ``window`` tokens for long agentic transcripts.

    Routing needs the *goal* (task/system prefix) and the *recent state* (latest turns),
    not the stale middle. Strategies:
      - "head_tail": keep the first ``head_tokens`` + the most recent ``window-head_tokens``
        tokens (default — preserves goal AND recent context).
      - "tail": keep only the most recent ``window`` tokens.
      - "full": keep everything up to ``window`` (tail fallback beyond it).
    """
    if len(ids) <= window:
        return ids
    if strategy == "head_tail":
        h = max(0, min(head_tokens, window))
        t = window - h
        return ids[:h] + ids[-t:] if t > 0 else ids[:h]
    # "tail" and "full"-over-window both keep the most recent window
    return ids[-window:]


class Featurizer(nn.Module, ABC):
    """Maps a batch of texts to decision-token hidden states ``(B, d)``."""

    d: int

    @abstractmethod
    def features(self, texts: list[str]) -> torch.Tensor: ...


class HFFeaturizer(Featurizer):
    """Frozen HuggingFace backbone (e.g. Qwen3-0.6B) with SVF, returning the hidden
    state at the decision token position."""

    def __init__(
        self,
        backbone_name: str,
        svf_targets: list[str] | None = None,
        hidden_position: str = "penultimate",
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        context_window: int = 32768,
        context_strategy: str = "head_tail",
        head_tokens: int = 512,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.backbone_name = backbone_name
        self.svf_targets = svf_targets if svf_targets is not None else DEFAULT_SVF_TARGETS
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # sdpa = memory-efficient (fused) attention: O(seq) memory instead of eager's O(seq^2)
        # score matrix, which OOMs the GPU on long agentic transcripts (terminal rollouts).
        self.backbone = AutoModel.from_pretrained(backbone_name, dtype=dtype, attn_implementation="sdpa")
        self.backbone.eval()
        self.svf_modules = apply_svf(self.backbone, self.svf_targets)
        self.d = self.backbone.config.hidden_size
        self.hidden_position = hidden_position
        # Long-context routing: cap the window to the backbone's positional capacity.
        max_pos = getattr(self.backbone.config, "max_position_embeddings", context_window)
        self.context_window = min(context_window, max_pos)
        self.context_strategy = context_strategy
        self.head_tokens = head_tokens
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def _encode(self, texts: list[str]) -> dict:
        # Tokenize raw "role: content" text verbatim (no chat template). Window each
        # sequence to keep the goal + recent state (NOT first-N, which drops recent
        # turns), then right-pad the batch.
        pad_id = self.tokenizer.pad_token_id
        # Append a trailing EOS so the decision token (penultimate, -2) is the LAST CONTENT
        # token, matching Fugu Fig. 2 ("...<Head Input><EOS>"). Qwen's tokenizer does not
        # add EOS itself, so without this -2 would read the second-to-last content token
        # (off-by-one). EOS is after the decision token, so causal attention leaves its
        # hidden state unchanged. Window first, then append, so EOS is always final.
        eos_id = self.tokenizer.eos_token_id
        seqs = []
        for t in texts:
            ids = window_token_ids(
                self.tokenizer(t, add_special_tokens=True)["input_ids"],
                self.context_window, self.head_tokens, self.context_strategy,
            )
            if eos_id is not None:
                ids = ids + [eos_id]
            seqs.append(ids)
        maxlen = max((len(s) for s in seqs), default=1)
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            input_ids.append(s + [pad_id] * pad)
            attn.append([1] * len(s) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, device=self._device),
            "attention_mask": torch.tensor(attn, device=self._device),
        }

    def _decision_index(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # Right-padded: real tokens occupy the front, so index by length.
        lengths = attention_mask.sum(dim=1)
        if self.hidden_position == "first":
            return torch.zeros_like(lengths)
        if self.hidden_position == "penultimate":
            return (lengths - 2).clamp(min=0)
        return (lengths - 1).clamp(min=0)  # "last" real token

    def features(self, texts: list[str]) -> torch.Tensor:
        enc = self._encode(texts)
        # SVF scales are trainable, so do NOT wrap the whole forward in no_grad;
        # the frozen backbone params simply carry no grad.
        out = self.backbone(**enc)
        h = out.last_hidden_state  # (B, T, d)
        idx = self._decision_index(enc["attention_mask"])  # (B,)
        batch = torch.arange(h.shape[0], device=h.device)
        return h[batch, idx]  # (B, d)


class SelectionRouter(nn.Module):
    def __init__(self, featurizer: Featurizer, num_workers: int):
        super().__init__()
        self.featurizer = featurizer
        self.num_workers = num_workers
        # Bias-free routing head (reference: f_theta: R^d -> R^L, no bias).
        self.head = nn.Linear(featurizer.d, num_workers, bias=False)
        # capture trainable params in a stable order (featurizer SVF scales, then head)
        self._trainable: list[nn.Parameter] = [
            p for p in self.parameters() if p.requires_grad
        ]

    @classmethod
    def from_pretrained(
        cls,
        backbone_name: str,
        num_workers: int,
        svf_targets: list[str] | None = None,
        hidden_position: str = "penultimate",
        device: str | None = None,
        context_window: int = 32768,
        context_strategy: str = "head_tail",
        head_tokens: int = 512,
    ) -> SelectionRouter:
        feat = HFFeaturizer(
            backbone_name,
            svf_targets=svf_targets,
            hidden_position=hidden_position,
            device=device,
            context_window=context_window,
            context_strategy=context_strategy,
            head_tokens=head_tokens,
        )
        router = cls(feat, num_workers)
        return router.to(feat._device)

    def logits(self, texts: list[str]) -> torch.Tensor:
        return self.head(self.featurizer.features(texts))

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.logits(texts)

    # ---- flat-vector interface for sep-CMA-ES -----------------------------
    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self._trainable)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self._trainable

    @torch.no_grad()
    def trainable_vector(self) -> torch.Tensor:
        if not self._trainable:
            return torch.zeros(0)
        return torch.cat([p.detach().reshape(-1).cpu() for p in self._trainable])

    @torch.no_grad()
    def load_vector(self, vec: torch.Tensor) -> None:
        offset = 0
        for p in self._trainable:
            n = p.numel()
            chunk = vec[offset : offset + n].to(p.device, p.dtype).reshape(p.shape)
            p.copy_(chunk)
            offset += n
        if offset != vec.numel():
            raise ValueError(f"vector size {vec.numel()} != trainable count {offset}")

    def summary(self) -> str:
        return (
            f"SelectionRouter(workers={self.num_workers}, d={self.featurizer.d}, "
            f"trainable={count_trainable(self)})"
        )


def save_router(router: SelectionRouter, path: str, worker_ids: list[str] | None = None) -> None:
    """Persist only the tiny trainable vector + the config to rebuild the backbone.

    The frozen backbone is never written (it is re-downloaded/loaded on demand).
    """
    feat = router.featurizer
    if not isinstance(feat, HFFeaturizer):
        raise TypeError("save_router only supports HFFeaturizer-backed routers")
    ckpt = {
        "backbone_name": feat.backbone_name,
        "svf_targets": feat.svf_targets,
        "hidden_position": feat.hidden_position,
        "context_window": feat.context_window,
        "context_strategy": feat.context_strategy,
        "head_tokens": feat.head_tokens,
        "num_workers": router.num_workers,
        "worker_ids": worker_ids or getattr(router, "worker_ids", None),
        "vector": router.trainable_vector(),
    }
    torch.save(ckpt, path)


def load_router(path: str, device: str | None = None) -> SelectionRouter:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    router = SelectionRouter.from_pretrained(
        ckpt["backbone_name"],
        ckpt["num_workers"],
        svf_targets=ckpt["svf_targets"],
        hidden_position=ckpt["hidden_position"],
        context_window=ckpt.get("context_window", 32768),
        context_strategy=ckpt.get("context_strategy", "head_tail"),
        head_tokens=ckpt.get("head_tokens", 512),
        device=device,
    )
    router.load_vector(ckpt["vector"])
    if ckpt.get("worker_ids"):
        router.worker_ids = ckpt["worker_ids"]
    return router
