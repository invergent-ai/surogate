"""SFT stage: train the router to match the soft worker distribution by KL.

Minimizes ``KL(p_i ‖ softmax(router.logits(q_i)))`` over the labeled questions,
updating only the trainable set (head + SVF scales). Tiny and fast — minutes on one
GPU or CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..shared.transcript import raw_query
from .labels import SoftLabel
from .model import SelectionRouter


@dataclass
class SFTStats:
    losses: list[float] = field(default_factory=list)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def _dropout_mask(targets: torch.Tensor, keep_prob: float) -> torch.Tensor:
    """(B, L) boolean keep-mask for pool dropout: each worker kept ~Bernoulli(keep_prob),
    but every row keeps >=2 workers and >=1 with positive target reward, so the
    renormalized soft target stays well-defined. Dropping the best worker is intentional
    — that is exactly the fallback signal we want the router to learn."""
    B, L = targets.shape
    keep = torch.rand(B, L, device=targets.device) < keep_prob
    for b in range(B):
        row = keep[b]
        if int(row.sum()) < 2:
            row.zero_()
            row[targets[b].topk(min(2, L)).indices] = True
        elif float((targets[b] * row).sum()) <= 0.0:
            row[int(targets[b].argmax())] = True  # ensure a positive worker survives
    return keep


def train_sft(
    router: SelectionRouter,
    labels: list[SoftLabel],
    *,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 16,
    micro_batch: int = 8,
    weight_decay: float = 0.0,
    device: str | None = None,
    shuffle: bool = True,
    log_every: int = 0,
    pool_dropout: float = 0.0,
) -> SFTStats:
    if not labels:
        raise ValueError("no labels to train on")
    if not 0.0 <= pool_dropout < 1.0:
        raise ValueError("pool_dropout must be in [0, 1)")
    expected = router.num_workers
    for lab in labels:
        if len(lab.p) != expected:
            raise ValueError(
                f"label {lab.task_id} has {len(lab.p)} workers, router expects {expected}; "
                "labels must be generated against the same ordered worker pool"
            )

    device = device or next(router.parameters()).device.type
    opt = torch.optim.AdamW(
        router.trainable_parameters(), lr=lr, weight_decay=weight_decay
    )
    targets = torch.tensor([lab.p for lab in labels], dtype=torch.float32, device=device)
    # Route on the raw "role: content" surface form (must match eval/inference).
    prompts = [raw_query(lab.prompt) for lab in labels]
    n = len(labels)
    stats = SFTStats()

    router.train()
    t0 = time.time()
    if log_every:
        print(f"[sft] training {n} labels × {epochs} epochs (batch {batch_size}, lr {lr:g}) on {device}", flush=True)
    for epoch in range(epochs):
        order = torch.randperm(n) if shuffle else torch.arange(n)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            bsz = len(idx)
            opt.zero_grad()
            batch_loss = 0.0
            # Gradient accumulation: process the (effective) batch in memory-safe micro-batches and
            # accumulate grads, scaled so the sum equals the full-batch mean, then step once. Lets us
            # match a large effective batch (e.g. 64) without the activation memory of one big forward.
            for ms in range(0, bsz, micro_batch):
                midx = idx[ms : ms + micro_batch]
                m_prompts = [prompts[i] for i in midx.tolist()]
                m_p = targets[midx]
                logits = router.logits(m_prompts)
                if pool_dropout > 0.0:
                    # Mask a random worker subset per example and match the target
                    # renormalized over the survivors → teaches fallback routing. Cross-entropy
                    # form (KL up to the constant H(p)) avoids 0*log0 NaNs on dropped entries.
                    keep = _dropout_mask(m_p, 1.0 - pool_dropout)
                    logq = F.log_softmax(logits.masked_fill(~keep, -1e9), dim=-1)
                    p = m_p * keep
                    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                    loss = -(p * logq).sum(dim=-1).mean()
                else:
                    logq = F.log_softmax(logits, dim=-1)
                    loss = F.kl_div(logq, m_p, reduction="batchmean")
                scale = len(midx) / bsz  # weight each micro-mean into the full-batch mean
                (loss * scale).backward()
                batch_loss += float(loss.detach()) * scale
            opt.step()
            epoch_loss += batch_loss
            nb += 1
        avg = epoch_loss / max(nb, 1)
        stats.losses.append(avg)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            el = time.time() - t0
            eta = el / (epoch + 1) * (epochs - epoch - 1)
            print(f"[sft] epoch {epoch + 1:4d}/{epochs}  kl={avg:.5f}  "
                  f"({el:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
    router.eval()
    return stats
