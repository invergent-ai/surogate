"""Teacher top-K logprob capture for offline knowledge distillation.

Reads each tokenized `train-NNN.bin` shard directly (tokens + per-doc
position_ids), runs the teacher over non-overlapping `sequence_len` windows —
exactly the chunks the native DataLoader serves — and writes a
`train-NNN.bin.kd` sidecar (see `surogate.distill.sidecar` for the format).

Sidecar row w*S+i stores the teacher's top-k next-token logprobs from output
position i of window w, i.e. the distribution predicting tokens[w*S+i+1].
This includes i == S-1 (predicting the first token of the next window): the
DataLoader's targets for a chunk at chunk_pos are tokens[chunk_pos+1 ..
chunk_pos+S], so the last row of every window is consumed, not discarded.
Rows past the last full window are zero-filled.

Two teacher backends:
- Local HF model (default): teacher loaded via transformers, whole windows per
  forward. Packed documents are isolated by transformers' own packed-sequence
  support, which reads the shard's `position_ids` resets (requires
  `use_cache=False`).
- OpenAI-compatible API (`distillation.teacher_api_base`): one vLLM
  `prompt_logprobs` completions request per document (split at position-id
  resets, one-token lookahead), which gives exact context isolation with no
  flash-attn requirement. Requires a vLLM-compatible server started with
  `--max-logprobs >= top_k`; aggregators like OpenRouter cannot be used.

torch/transformers are imported lazily inside functions; the API backend never
imports them.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from surogate.core.config.sft_config import SFTConfig
from surogate.distill.packed import teacher_logits, verify_isolation
from surogate.distill.sidecar import (
    SidecarWriter,
    read_token_shard_header,
    sidecar_path_for,
    validate_sidecar,
)
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


def _read_shard(path: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Memory-map a token shard; returns (tokens, position_ids or None, n_tokens)."""
    header = read_token_shard_header(path)
    n_tokens = header.n_tokens
    tokens = np.memmap(path, dtype="<i4", mode="r", offset=header.tokens_offset, shape=(n_tokens,))
    position_ids = None
    if header.version >= 3:
        position_ids = np.memmap(
            path, dtype="<i4", mode="r", offset=header.position_ids_offset, shape=(n_tokens,)
        )
    return tokens, position_ids, n_tokens


def _load_teacher(teacher_model: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading teacher model '{teacher_model}' (sdpa, bf16) on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model


def _topk_logprobs(logits, top_k: int):
    """Top-k of log_softmax(logits) computed as topk(logits) - logsumexp(logits).

    topk indices over raw logits equal those over log_softmax (monotone shift);
    the fp32 normalizer is computed one window at a time to avoid materializing
    a full fp32 copy of the (bsz, S, V) logits. This relies on `logits` arriving
    as [B, S, V], one row per window - capture used to flatten to [1, B*S] for
    flash-attention's padding-free path, which silently collapsed this loop to a
    single iteration over the whole batch.
    """
    import torch

    values, ids = logits.topk(top_k, dim=-1)
    lse = torch.stack([torch.logsumexp(row.to(torch.float32), dim=-1) for row in logits])
    logprobs = values.to(torch.float32) - lse.unsqueeze(-1)
    return logprobs, ids


def capture_shard(
    model,
    shard_path: str,
    sidecar_path: str,
    top_k: int,
    sequence_len: int,
    teacher_batch_size: int,
    teacher_vocab_size: int,
    tokenize_hash: str,
    device: str,
) -> None:
    tokens, position_ids, n_tokens = _read_shard(shard_path)
    student_vocab = read_token_shard_header(shard_path).vocab_size
    max_token_id = int(tokens.max()) if n_tokens > 0 else -1
    if max_token_id >= teacher_vocab_size:
        raise ValueError(
            f"Token shard '{shard_path}' contains token id {max_token_id} but the teacher vocab "
            f"size is {teacher_vocab_size}. Student and teacher must share a tokenizer. For "
            "cross-tokenizer distillation, first transplant the teacher's tokenizer onto the "
            "student: `surogate transplant-tokenizer --student <student> --teacher <teacher> "
            "--output <dir>`, then re-tokenize and re-capture against the transplanted model."
        )

    n_windows = n_tokens // sequence_len
    if n_windows == 0:
        raise ValueError(
            f"Token shard '{shard_path}' has only {n_tokens} tokens (< sequence_len={sequence_len}); "
            "nothing to capture."
        )

    tmp_path = sidecar_path + ".tmp"
    try:
        _capture_shard_into(
            model, tmp_path, tokens, position_ids, n_tokens, n_windows,
            top_k=top_k, sequence_len=sequence_len,
            teacher_batch_size=teacher_batch_size,
            teacher_vocab_size=teacher_vocab_size,
            student_vocab=student_vocab, tokenize_hash=tokenize_hash,
            device=device, shard_path=shard_path,
        )
    except BaseException:
        # SidecarWriter preallocates the full file up front, so an abort part way
        # through would otherwise strand a full-size .tmp that nothing ever cleans
        # up -- and the isolation check makes aborting an expected outcome, not
        # just a crash path.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.replace(tmp_path, sidecar_path)
    tail = n_tokens - n_windows * sequence_len
    logger.info(
        f"Wrote {sidecar_path}: {n_windows * sequence_len} rows captured, {tail} tail rows zero-filled."
    )


def _capture_shard_into(
    model,
    tmp_path: str,
    tokens,
    position_ids,
    n_tokens: int,
    n_windows: int,
    *,
    top_k: int,
    sequence_len: int,
    teacher_batch_size: int,
    teacher_vocab_size: int,
    student_vocab: int,
    tokenize_hash: str,
    device: str,
    shard_path: str,
) -> None:
    import torch
    from tqdm import tqdm

    with SidecarWriter(
        tmp_path,
        n_tokens=n_tokens,
        k=top_k,
        vocab_size=teacher_vocab_size,
        tokenize_hash=tokenize_hash,
    ) as writer:
        with torch.inference_mode():
            for w0 in tqdm(
                range(0, n_windows, teacher_batch_size),
                desc=f"Capturing {os.path.basename(shard_path)}",
                unit="batch",
            ):
                w1 = min(w0 + teacher_batch_size, n_windows)
                span = slice(w0 * sequence_len, w1 * sequence_len)
                bsz = w1 - w0
                # One row per window. transformers derives the packed-document
                # boundaries from position_ids itself and builds the
                # block-diagonal causal mask, so each packed document is isolated
                # without a hand-rolled mask -- and sliding-window teachers keep
                # their sliding layers, which a 4D attention_mask would override.
                #
                # use_cache=False is load-bearing: with a cache allocated,
                # _preprocess_mask_arguments skips the packed detection entirely
                # and the teacher attends straight across document boundaries.
                batch_ids = torch.from_numpy(
                    np.ascontiguousarray(tokens[span], dtype=np.int64)
                ).view(bsz, sequence_len).to(device)
                batch_pos = None
                if position_ids is not None:
                    batch_pos = torch.from_numpy(
                        np.ascontiguousarray(position_ids[span], dtype=np.int64)
                    ).view(bsz, sequence_len).to(device)
                # Pre-v3 shards carry no position_ids: transformers defaults to
                # arange per row, i.e. one document per window, plain causal.
                logits = teacher_logits(model, batch_ids, batch_pos)
                if w0 == 0 and batch_pos is not None:
                    # Once per shard, against the batch just computed: losing
                    # isolation is invisible in the output, so prove it held
                    # rather than write a plausible-looking sidecar built from
                    # cross-document context. Searches every row of the batch for
                    # a usable probe; costs one forward over <= 32 tokens.
                    verify_isolation(model, batch_ids, batch_pos, logits)
                logprobs, ids = _topk_logprobs(logits, top_k)
                ids = ids.reshape(-1, top_k).cpu().numpy()
                logprobs = logprobs.reshape(-1, top_k).cpu().numpy()
                # Teacher entries outside the student vocab (padded teacher
                # lm_head rows) carry probability the student cannot express:
                # mask them so the native q-renormalization drops their mass
                # (missing-probability = zero semantics).
                oov = ids >= student_vocab
                if oov.any():
                    ids = np.where(oov, 0, ids)
                    logprobs = np.where(oov, np.float32(-np.inf), logprobs)
                writer.write_rows(
                    ids.astype(np.uint32),
                    logprobs.astype(np.float16),
                )


def _api_guidance(api_base: str, top_k: int) -> str:
    return (
        f"The server at {api_base} did not return prompt_logprobs. distill-capture requires a "
        f"vLLM-compatible inference server started with --max-logprobs >= {top_k}, serving a "
        "teacher that SHARES the student's tokenizer. OpenAI-compatible aggregators/providers "
        "(e.g. OpenRouter) do not expose per-position prompt logprobs or token-id prompts and "
        "cannot be used for logit capture."
    )


def _make_api_client(dist):
    import httpx

    api_key = os.environ.get(dist.teacher_api_key_var or "", "") or "EMPTY"
    return httpx.Client(
        base_url=dist.teacher_api_base,
        timeout=dist.teacher_api_timeout,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _doc_spans(position_ids: np.ndarray | None, start: int, end: int) -> list[tuple[int, int]]:
    """Absolute [a, b) document spans within window [start, end).

    pos[i] == 0 marks a document start; the window start always begins a span
    (a doc continuing across the window boundary loses its earlier context,
    matching the local backend's window-at-a-time processing).
    """
    if position_ids is None:
        return [(start, end)]
    starts = [start] + [i for i in range(start + 1, end) if position_ids[i] == 0]
    return list(zip(starts, starts[1:] + [end]))


def _post_prompt_logprobs(client, model: str, prompt: list[int], top_k: int, retry_backoff: float = 1.0):
    """POST an OpenAI completions request with vLLM's prompt_logprobs extension.

    Returns choices[0].prompt_logprobs (list aligned with the prompt; entry 0 is
    null, entry i is {token_id_str: {"logprob": float, "rank": int, ...}}).
    Retries transient failures (transport errors, 429, 5xx) 3 times with backoff.
    """
    import httpx

    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": top_k,
    }
    attempts = 3
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post("/completions", json=body)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"transient HTTP {response.status_code}", request=response.request, response=response
                )
            if 400 <= response.status_code < 500:
                text = response.text.lower()
                if "prompt_logprobs" in text or (
                    "prompt" in text and any(t in text for t in ("unknown", "unexpected", "invalid", "unsupported"))
                ):
                    raise RuntimeError(
                        f"{_api_guidance(str(client.base_url), top_k)} "
                        f"(HTTP {response.status_code}: {response.text[:300]})"
                    )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0].get("prompt_logprobs")
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status != 429 and status < 500:
                raise
            last_error = e
            if attempt < attempts - 1:
                time.sleep(retry_backoff * 2**attempt)
    raise RuntimeError(
        f"Teacher API request failed after {attempts} attempts ({client.base_url}): {last_error}"
    )


def _row_from_entry(
    entry: dict, top_k: int, api_base: str, student_vocab: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k (ids, logprobs) from one prompt_logprobs entry, sorted by logprob descending.

    vLLM may return top_k + 1 candidates when the actual prompt token falls
    outside the top-k; truncation keeps the k most probable. Candidates outside
    the student vocab (padded teacher lm_head rows) are masked to logprob -inf
    so the native q-renormalization drops their mass.
    """
    ranked = sorted(((float(v["logprob"]), int(tid)) for tid, v in entry.items()), reverse=True)
    if len(ranked) < top_k:
        raise RuntimeError(
            f"The server at {api_base} returned only {len(ranked)} prompt_logprobs candidates for a "
            f"position but distillation.top_k={top_k}; restart the vLLM server with "
            f"--max-logprobs >= {top_k}."
        )
    ranked = ranked[:top_k]
    ids = np.array([tid for _, tid in ranked], dtype=np.int64)
    logprobs = np.array([lp for lp, _ in ranked], dtype=np.float32)
    if student_vocab is not None:
        oov = ids >= student_vocab
        if oov.any():
            ids = np.where(oov, 0, ids)
            logprobs = np.where(oov, np.float32(-np.inf), logprobs)
    return ids.astype(np.uint32), logprobs.astype(np.float16)


def capture_shard_api(
    client,
    teacher_model: str,
    shard_path: str,
    sidecar_path: str,
    top_k: int,
    sequence_len: int,
    concurrency: int,
    tokenize_hash: str,
    retry_backoff: float = 1.0,
) -> None:
    from tqdm import tqdm

    shard_header = read_token_shard_header(shard_path)
    tokens, position_ids, n_tokens = _read_shard(shard_path)

    n_windows = n_tokens // sequence_len
    if n_windows == 0:
        raise ValueError(
            f"Token shard '{shard_path}' has only {n_tokens} tokens (< sequence_len={sequence_len}); "
            "nothing to capture."
        )

    api_base = str(client.base_url)
    first_response_checked = False

    def fetch(span: tuple[int, int]):
        a, b = span
        prompt = np.asarray(tokens[a : min(b + 1, n_tokens)]).tolist()
        return _post_prompt_logprobs(client, teacher_model, prompt, top_k, retry_backoff=retry_backoff)

    tmp_path = sidecar_path + ".tmp"
    # Header vocab_size: the teacher config is not available in API mode; the
    # shared-tokenizer requirement makes the shard's vocab the same vocab.
    with SidecarWriter(
        tmp_path,
        n_tokens=n_tokens,
        k=top_k,
        vocab_size=shard_header.vocab_size,
        tokenize_hash=tokenize_hash,
    ) as writer:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            windows_per_flush = max(1, concurrency)
            for w0 in tqdm(
                range(0, n_windows, windows_per_flush),
                desc=f"Capturing {os.path.basename(shard_path)} (API)",
                unit="chunk",
            ):
                start = w0 * sequence_len
                end = min(w0 + windows_per_flush, n_windows) * sequence_len
                spans = [
                    span
                    for w in range(w0, min(w0 + windows_per_flush, n_windows))
                    for span in _doc_spans(position_ids, w * sequence_len, (w + 1) * sequence_len)
                ]
                buf_ids = np.zeros((end - start, top_k), dtype=np.uint32)
                buf_logprobs = np.zeros((end - start, top_k), dtype=np.float16)
                for (a, b), prompt_logprobs in zip(spans, pool.map(fetch, spans)):
                    if not first_response_checked:
                        entries = [e for e in (prompt_logprobs or []) if e]
                        if not entries:
                            raise RuntimeError(_api_guidance(api_base, top_k))
                        if len(entries[0]) < top_k:
                            raise RuntimeError(
                                f"The server at {api_base} returned only {len(entries[0])} "
                                f"prompt_logprobs candidates but distillation.top_k={top_k}; restart "
                                f"the vLLM server with --max-logprobs >= {top_k}."
                            )
                        first_response_checked = True
                    if not prompt_logprobs:
                        raise RuntimeError(_api_guidance(api_base, top_k))
                    # Entry i (i >= 1) predicts prompt position i -> sidecar row
                    # a + i - 1. The one-token lookahead (prompt runs to b) makes
                    # the doc's last row b-1 predict tokens[b]; when b == n_tokens
                    # there is no lookahead and the final row stays zero-filled.
                    for i, entry in enumerate(prompt_logprobs):
                        if i == 0 or not entry:
                            continue
                        row = a + i - 1
                        buf_ids[row - start], buf_logprobs[row - start] = _row_from_entry(
                            entry, top_k, api_base, student_vocab=shard_header.vocab_size
                        )
                writer.write_rows(buf_ids, buf_logprobs)
    os.replace(tmp_path, sidecar_path)
    tail = n_tokens - n_windows * sequence_len
    logger.info(
        f"Wrote {sidecar_path}: {n_windows * sequence_len} rows captured via API, "
        f"{tail} tail rows zero-filled."
    )


def run_capture(
    config: SFTConfig,
    shard_paths: list[str],
    tokenize_hash: str,
    device: str = "cuda:0",
) -> None:
    dist = config.distillation
    logger.warning(
        "distill-capture assumes the student and teacher share a tokenizer: the sidecar stores "
        "teacher-vocab token ids against the student's token stream. Cross-tokenizer "
        "distillation is NOT supported."
    )

    kd_dir = dist.kd_dir
    if kd_dir:
        Path(kd_dir).mkdir(parents=True, exist_ok=True)
        if any(Path(kd_dir).resolve() != Path(p).parent.resolve() for p in shard_paths):
            logger.info(
                f"distillation.kd_dir={kd_dir} differs from the token shard directory; the trainer "
                "will validate these sidecars and symlink them next to the token shards at "
                "training start."
            )

    pending: list[tuple[str, str]] = []
    for shard in shard_paths:
        sidecar = sidecar_path_for(shard, kd_dir)
        try:
            validate_sidecar(sidecar, shard, expect_k=dist.top_k, expect_hash=tokenize_hash)
            logger.info(f"Skipping {shard}: valid sidecar exists at {sidecar}.")
            continue
        except ValueError as e:
            if os.path.exists(sidecar):
                logger.info(f"Recapturing {shard}: {e}")
        pending.append((shard, sidecar))

    if not pending:
        logger.info("All shards already have valid sidecars; nothing to capture.")
        return

    if dist.teacher_api_base:
        logger.info(
            f"Using API teacher '{dist.teacher_model}' at {dist.teacher_api_base} "
            f"(concurrency={dist.teacher_api_concurrency}, timeout={dist.teacher_api_timeout}s)."
        )
        client = _make_api_client(dist)
        try:
            for shard, sidecar in pending:
                capture_shard_api(
                    client,
                    dist.teacher_model,
                    shard,
                    sidecar,
                    top_k=dist.top_k,
                    sequence_len=config.sequence_len,
                    concurrency=dist.teacher_api_concurrency,
                    tokenize_hash=tokenize_hash,
                )
        finally:
            client.close()
    else:
        model = _load_teacher(dist.teacher_model, device)
        teacher_vocab_size = int(model.config.vocab_size)
        if dist.top_k > teacher_vocab_size:
            raise ValueError(
                f"distillation.top_k={dist.top_k} exceeds the teacher vocab size {teacher_vocab_size}."
            )

        for shard, sidecar in pending:
            capture_shard(
                model,
                shard,
                sidecar,
                top_k=dist.top_k,
                sequence_len=config.sequence_len,
                teacher_batch_size=dist.teacher_batch_size,
                teacher_vocab_size=teacher_vocab_size,
                tokenize_hash=tokenize_hash,
                device=device,
            )

    logger.info(f"Capture complete: {len(pending)} sidecar(s) written.")


def distill_capture_main(config: SFTConfig, args: DictDefault) -> None:
    if config.distillation is None or not config.distillation.teacher_model:
        raise ValueError(
            "distill-capture requires a `distillation:` block with `teacher_model` set in the config "
            "(with teacher_api_base, teacher_model is the served model name)."
        )

    if args.get("hub_token"):
        # Covers the teacher download and any gated dataset/tokenizer fetches.
        os.environ.setdefault("HF_TOKEN", args["hub_token"])

    if args.get("api_base"):
        config.distillation.teacher_api_base = args["api_base"]
    if config.distillation.teacher_api_base and args.get("device"):
        logger.warning(
            "--device is ignored in API mode "
            "(the teacher runs on the server; per-document requests give exact context isolation)."
        )

    # Tokenize first if shards are missing/stale — same flow sft.py runs before
    # globbing (TokenizeDatasets.run() is hash-cached and calls __post_init__).
    from surogate.train.tokenize import TokenizeDatasets, read_tokenize_hash

    TokenizeDatasets(config, args).run()

    output_path = Path(config.output_dir)
    train_files = sorted([str(p) for p in output_path.glob("train-*.bin")])
    if not train_files:
        raise RuntimeError(
            f"No training files found matching '{config.output_dir}/train-*.bin'. "
            "Tokenization produced no usable rows (all dataset rows were dropped)."
        )

    tokenize_hash = read_tokenize_hash(config.output_dir)
    if not tokenize_hash:
        raise RuntimeError(
            f"No {config.output_dir}/.tokenize_hash found; run `surogate tokenize` (or re-run "
            "distill-capture) after a successful tokenization."
        )

    run_capture(
        config,
        train_files,
        tokenize_hash,
        device=args.get("device") or "cuda:0",
    )
