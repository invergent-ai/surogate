"""Bin packing for tokenize-time ``sample_packing: bin``: whole documents, almost no padding.

``sample_packing: true`` concatenates every document into one stream and cuts it every
``sequence_len`` tokens, so a document that straddles a cut is split across two windows. The
windows are shuffled independently, so the second half trains in another step as a document of
its own, without the prompt it answers.

``sample_packing: bin`` keeps every document whole. Documents are assigned online with
Next-k-Fit (Johnson 1974): each goes into the oldest of up to ``k`` open windows that still has
room, and when none has, the fullest open window is written out and a new one opened. A larger
``k`` trades a little planning time for less padding; ``k = 1`` is plain Next-Fit. The window's
unused tail is padding without targets. Documents longer than ``sequence_len`` are truncated,
exactly as a padded shard truncates them.

Each document keeps position ids ``0..len-1``, so the engine's document masking
(``compute_doc_masking``) isolates it; the padding continues the last document's positions, as in
the other shard layouts, so it adds no document of its own. Every position id stays below
``sequence_len`` (concatenated windows can carry a split document's original, larger positions).

The planner is pure Python; only :func:`materialize_window` needs numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Documents per window at most. The attention backward (FlashAttention varlen) allocates scratch
#: per document, and a captured CUDA-graph step starts with room for 64 documents per row
#: (``SUROGATE_OUTER_CAPTURE_MAX_DOCS``); the same bound as row packing's ``MAX_ROWS_PER_WINDOW``.
MAX_DOCS_PER_WINDOW = 64


def next_k_fit(
    lengths: Sequence[int],
    seq_len: int,
    k: int,
    max_docs: int = MAX_DOCS_PER_WINDOW,
) -> list[list[int]]:
    """Assign documents (by index) to ``seq_len`` windows with Next-k-Fit.

    Documents are taken in order. Each goes into the oldest open window with room for it; if
    none has room, a new window is opened, first writing out the fullest open window when ``k``
    are already open. A window is written out as soon as it is exactly full or holds
    ``max_docs`` documents. Lengths above ``seq_len`` count as ``seq_len`` (the writer truncates
    them). Returns the windows in the order they were written out; every document of nonzero
    length is in exactly one window, and zero-length documents are dropped. Deterministic.
    """
    if seq_len < 1 or k < 1 or max_docs < 1:
        raise ValueError("seq_len, k and max_docs must be >= 1")

    # Open windows, oldest first: [room left, document indices].
    open_windows: list[list] = []
    out: list[list[int]] = []
    for i, n in enumerate(lengths):
        n = min(int(n), seq_len)
        if n <= 0:
            continue
        j = next((b for b, w in enumerate(open_windows) if w[0] >= n), None)
        if j is None:
            if len(open_windows) == k:
                fullest = min(range(k), key=lambda b: open_windows[b][0])
                out.append(open_windows.pop(fullest)[1])
            j = len(open_windows)
            open_windows.append([seq_len, []])
        w = open_windows[j]
        w[0] -= n
        w[1].append(i)
        if w[0] == 0 or len(w[1]) >= max_docs:
            out.append(open_windows.pop(j)[1])
    out.extend(w[1] for w in open_windows)
    return out


def padding_share(lengths: Sequence[int], windows: Sequence[Sequence[int]], seq_len: int) -> float:
    """Share of the windows' positions that hold no document token."""
    if not windows:
        return 0.0
    used = sum(min(int(lengths[i]), seq_len) for w in windows for i in w)
    return 1.0 - used / (len(windows) * seq_len)


def materialize_window(tokens, masks, docs: Sequence[int], seq_len: int, pad_token_id: int):
    """Lay one planned window out as ``(tokens, position_ids, mask)`` int32 arrays of ``seq_len``.

    ``tokens[i]`` / ``masks[i]`` are document ``i``'s tokens and its input-aligned loss mask
    (``mask[t]`` trains the prediction of ``tokens[t + 1]``; the last bit is 0). A truncated
    document's last mask bit is cleared too: the token it would predict was cut off, and the
    shard's next position belongs to another window.
    """
    import numpy as np  # here, so the planner above imports without numpy

    out_tokens = np.full(seq_len, pad_token_id, dtype=np.int32)
    out_mask = np.zeros(seq_len, dtype=np.int32)
    out_pos = np.empty(seq_len, dtype=np.int32)
    o = 0
    for i in docs:
        n = min(len(tokens[i]), seq_len)
        if o + n > seq_len:
            raise ValueError(f"window holds more than sequence_len {seq_len} tokens")
        out_tokens[o : o + n] = tokens[i][:n]
        out_mask[o : o + n] = masks[i][:n]
        out_mask[o + n - 1] = 0
        out_pos[o : o + n] = np.arange(n, dtype=np.int32)
        o += n
    # Padding continues the last document's positions: one document, not one per pad token.
    last = int(out_pos[o - 1]) if o else -1
    out_pos[o:] = np.arange(last + 1, last + 1 + seq_len - o, dtype=np.int32)
    return out_tokens, out_pos, out_mask
