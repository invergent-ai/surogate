# Knowledge Distillation

Surogate supports **offline top-K logit distillation** on the SFT path: a larger *teacher* model is run once over your tokenized training data, its top-K next-token log-probabilities are stored on disk, and the *student* then trains against a combined cross-entropy + KL loss that pulls its output distribution towards the teacher's.

Because the teacher runs offline (once, before training), the training loop itself pays no teacher-inference cost — the teacher signal is streamed from disk by the native DataLoader alongside the token shards.

**Hard requirement: the teacher and student must share a tokenizer.** The sidecar stores teacher-vocab token ids against the student's token stream. Distilling within a model family (e.g. Qwen3-1.7B → Qwen3-0.6B) satisfies this. For teachers with a different tokenizer, see [Cross-tokenizer distillation](#cross-tokenizer-distillation) — a vocabulary transplant turns it into a same-tokenizer setup.

## When to use it

- **Compressing a model**: transfer a large model's behavior into a smaller, cheaper one from the same family.
- **Better soft targets**: the teacher's full next-token distribution carries more signal per token than the one-hot label, which typically improves small-model quality on the same data.
- **Pure distillation on unlabeled text**: with `kd_weight: 1.0` and `ce_weight: 0.0`, the labels only select which positions are trained; the training signal is entirely the teacher's.

## Two-step workflow

Both steps read the **same config file** — one with a `distillation:` block:

```bash
# Step 1: run the teacher over the tokenized shards, write .kd sidecars
surogate distill-capture config.yaml

# Step 2: train the student against the sidecars
surogate sft config.yaml
```

`distill-capture` tokenizes the datasets first if the shards are missing or stale (the same hash-cached flow `surogate sft` uses), then writes one `train-NNN.bin.kd` sidecar next to each `{output_dir}/train-NNN.bin` token shard. Shards that already have a valid sidecar are skipped, so an interrupted capture can simply be re-run.

### Capture CLI flags

```bash
surogate distill-capture config.yaml [--api-base URL] [--device cuda:0] [--allow-cross-doc-attention] [--hub_token TOKEN]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--api-base` | none | OpenAI-compatible base URL of a served teacher (e.g. `http://localhost:8000/v1`); overrides `distillation.teacher_api_base` and switches capture to the API backend (below). |
| `--device` | `cuda:0` | Device to run the local teacher model on (e.g. `cuda:1`). Ignored with a warning in API mode. |
| `--allow-cross-doc-attention` | off | Local backend only: allow the `sdpa` fallback when flash-attention-2 is unavailable. Packed documents will then attend across document boundaries during capture (an approximation). Ignored with a warning in API mode. |
| `--hub_token` | none | Hugging Face token for private model access |

By default the teacher is loaded locally with `transformers` in bf16. The local backend **requires flash-attention-2** for per-document attention isolation of packed samples (the shard's per-document `position_ids` resets drive varlen attention); without flash-attn it errors out unless you pass `--allow-cross-doc-attention`.

### Remote teacher (vLLM API)

Setting `distillation.teacher_api_base` (or passing `--api-base`) switches capture to an **OpenAI-compatible vLLM server** instead of loading the teacher locally — useful when the teacher does not fit next to your capture GPU. In this mode `distillation.teacher_model` is the **served model name**.

```bash
# serve the teacher (must expose at least top_k prompt logprobs)
vllm serve Qwen/Qwen3-32B --max-logprobs 64

# capture against it
surogate distill-capture config.yaml --api-base http://localhost:8000/v1
```

How it works: capture sends **token-id prompts** through vLLM's `prompt_logprobs` completions extension — one request per packed document (split at the shard's position-id resets, with a one-token lookahead), which gives **exact per-document context isolation with no flash-attn requirement**. Up to `teacher_api_concurrency` requests are in flight at once (default 8), each with a `teacher_api_timeout` of 1200 s; transient failures (transport errors, HTTP 429/5xx) are retried 3 times with backoff. The API key is read from the environment variable named by `teacher_api_key_var` (default `VLLM_API_KEY`); if unset, `EMPTY` is sent, which local unauthenticated vLLM servers accept.

Two hard requirements:

- The server **must be started with `--max-logprobs >= distillation.top_k`**, or capture aborts with: `The server at <url> returned only <N> prompt_logprobs candidates but distillation.top_k=<K>; restart the vLLM server with --max-logprobs >= <K>.`
- It must be a **vLLM-compatible server, not an aggregator**. OpenAI-compatible aggregators/providers (e.g. OpenRouter) do not expose per-position prompt logprobs or token-id prompts; capture fails with: `The server at <url> did not return prompt_logprobs. distill-capture requires a vLLM-compatible inference server started with --max-logprobs >= <K>, serving a teacher that SHARES the student's tokenizer.`

## Configuration reference

```yaml
model: Qwen/Qwen3-0.6B          # the student

distillation:
  teacher_model: Qwen/Qwen3-1.7B
  top_k: 64
  temperature: 1.0
  kd_weight: 0.5
```

The `distillation:` block:

| Option | Type | Default | Used by | Description |
|--------|------|---------|---------|-------------|
| `teacher_model` | str | `null` | capture | Teacher model id or path. Required by `distill-capture`; ignored at train time. |
| `top_k` | int | `32` | both | Number of teacher logprobs stored per token. Must be in `[1, 1024]` and not exceed the teacher vocab size. Training validates that the sidecars were captured with exactly this value. |
| `temperature` | float | `1.0` | train | Distillation temperature τ. Must be > 0. |
| `kd_weight` | float | `0.5` | train | Weight of the KD (KL) term. Must be ≥ 0. |
| `ce_weight` | float | `1 - kd_weight` | train | Weight of the CE term. Must be ≥ 0. When `kd_weight > 1` the default would be negative, so `ce_weight` must then be set explicitly. |
| `teacher_batch_size` | int | `4` | capture | Local backend only: windows of `sequence_len` tokens per teacher forward pass during capture. Must be ≥ 1. Raise it until the teacher fills its GPU. |
| `kd_dir` | str | `null` | both | Sidecar directory override (default: alongside the token shards). The trainer validates the sidecars there and bridges them to the native DataLoader with `<shard>.kd` symlinks automatically. |
| `teacher_api_base` | str | `null` | capture | OpenAI-compatible base URL of a served teacher (e.g. `http://localhost:8000/v1`). When set, capture queries the API instead of loading `teacher_model` locally — see [Remote teacher](#remote-teacher-vllm-api). |
| `teacher_api_key_var` | str | `"VLLM_API_KEY"` | capture | Name of the environment variable holding the API key. Empty/unset resolves to `EMPTY` (accepted by local vLLM servers). |
| `teacher_api_concurrency` | int | `8` | capture | Concurrent in-flight capture requests in API mode. Must be ≥ 1. |
| `teacher_api_timeout` | int | `1200` | capture | Per-request timeout in seconds in API mode. Must be ≥ 1. |

Enabling distillation also changes a few top-level behaviors automatically (logged at startup):

- `use_cuda_graphs` is forced to `false` — KD micro-steps run eagerly in v1.
- `lmhead_drop_ignored_rows` is forced to `false` — the compact LM-head row path does not support KD in v1.

## The loss

Per valid token *t* (i.e. `target != -100`), with student logits *z*:

```
L_t = ce_weight · CE_t  +  kd_weight · τ² · KL(q ‖ p_τ)
```

where

- `CE_t = -log softmax(z)[target_t]` — the standard cross-entropy, at temperature 1;
- `p_τ = softmax(z / τ)` — the student distribution at temperature τ;
- `q` — the teacher's stored top-K logprobs `lp`, renormalized over the K entries at temperature τ: `q_k = exp(lp_k/τ) / Σ_j exp(lp_j/τ)`.

Probability mass outside the teacher's top-K is dropped and the top-K renormalized to sum to 1 (the "zero" missing-mass policy, DistillKit's default). The `τ²` factor keeps gradient magnitudes comparable across temperatures. The loss is a mean over valid tokens; masked positions (`target == -100`, e.g. prompt tokens in completion-only SFT) contribute nothing to either term.

The KD term is fused into the LM-head cross-entropy backward kernel, so there is no extra logit materialization or separate KD pass.

## Choosing hyperparameters

- **`top_k` (32–128)**: how much of the teacher distribution is preserved. `32` already captures the bulk of the mass for a confident teacher; `64` is a good default trade-off; go to `128` for high-entropy targets or when using higher temperatures (a flatter teacher puts more mass outside a small top-K, and renormalization discards it). Storage scales linearly with K (see below).
- **`temperature` (1–2)**: `1.0` matches the teacher distribution as-is; `1.5`–`2.0` softens both distributions, giving the "dark knowledge" in near-miss tokens more gradient weight. Values much above 2 interact poorly with top-K truncation because the renormalization drops an increasingly large tail.
- **`kd_weight` / `ce_weight`**: the default `kd_weight: 0.5` (with `ce_weight` resolving to `0.5`) balances label fitting and teacher matching. Use `kd_weight: 1.0`, `ce_weight: 0.0` for pure distillation. If you set `kd_weight > 1`, you must set `ce_weight` explicitly.

## Storage cost

A sidecar stores, per token, K uint32 ids + K fp16 logprobs = **6·K bytes per token** (plus a fixed 1 KiB header per shard):

| `top_k` | Bytes/token | 100M tokens | 1B tokens |
|---------|-------------|-------------|-----------|
| 32 | 192 B | ~19 GB | ~192 GB |
| 64 | 384 B | ~38 GB | ~384 GB |
| 128 | 768 B | ~77 GB | ~768 GB |

## The `kd_loss` metric

During KD training the step log line carries an extra `kd_loss` field:

- `loss` is the plain (unweighted) cross-entropy mean per valid token, exactly as in normal SFT.
- `kd_loss` is the mean **τ²-scaled KL divergence** `τ² · KL(teacher_topk ‖ student)` per valid token, accumulated over the micro-steps of the optimizer step. It is reported **rank-0 local** (like the GRPO metrics).

The actual optimized objective is `ce_weight · loss + kd_weight · kd_loss`. A healthy run shows `kd_loss` decreasing as the student's distribution moves towards the teacher's. Note that for MoE models the step log line carries the MoE metrics instead; `kd_loss` remains available in the structured step metrics.

Evaluation is unaffected by distillation: **eval loss is CE-only** (the eval loader never reads sidecars), so eval numbers stay comparable with non-KD runs.

## The `.kd` sidecar format

One sidecar per token shard, named `train-NNN.bin.kd` next to `train-NNN.bin`:

```
[Header: 1024 bytes]
  bytes 0..7   : magic "KD.LOGP\n"
  int32 fields (little-endian, int32 index = byte offset / 4):
    [2] version = 1
    [3] k                (top-K per token, 1..1024)
    [4] n_tokens         (must equal the token shard's n_tokens)
    [5] vocab_size       (teacher vocab)
    [6] logprob_dtype    (0 = fp16; only fp16 in v1)
    [7] reserved = 0
  bytes 64..79 : tokenize hash (16 ASCII hex chars from .tokenize_hash,
                 zero-padded) — validated by the Python layer before
                 training (C++ ignores it)
[ids      : n_tokens × k uint32]   at byte 1024
[logprobs : n_tokens × k fp16]     at byte 1024 + 4·k·n_tokens
```

**Alignment**: row *i* holds the teacher's next-token distribution at input position *i* — the distribution predicting `tokens[i+1]`, i.e. aligned with `targets[i]`. The DataLoader reads sidecar rows `[chunk_pos, chunk_pos + sequence_len)` for a chunk starting at `chunk_pos`, with no extra shift.

Capture runs the teacher over non-overlapping `sequence_len` windows — exactly the chunks the DataLoader serves. The last row of each window is captured too (it predicts the first token of the next window and is consumed by training, since a chunk's targets are `tokens[chunk_pos+1 .. chunk_pos+S]`). Only rows past the last full window (the shard tail) are zero-filled; training never reads past the last full chunk. Masked positions are captured normally and skipped at train time via `target == -100`.

Teacher top-K entries whose token id falls **outside the student's vocabulary** (padded teacher LM-head rows) are masked to `-inf` logprob at capture time, so the training-side renormalization drops their mass (missing-probability = zero). The trainer logs a warning when the sidecar's teacher vocab exceeds the student vocab. In API mode the header's `vocab_size` is the shard's (student) vocab — identical to the teacher's under the shared-tokenizer requirement.

The reader/writer lives in `surogate/distill/sidecar.py` (pure numpy) if you need to inspect or post-process sidecars.

## Cross-tokenizer distillation

The `.kd` sidecar stores **teacher token ids**, and training uses them to index the **student's LM head** — which is why the tokenizers must match. To distill from a teacher with a *different* tokenizer, transplant the teacher's tokenizer onto the student first with `surogate transplant-tokenizer`: the student's embedding and LM-head rows are rebuilt for the teacher's vocabulary (shared tokens copied exactly, new tokens approximated), after which the standard same-tokenizer pipeline runs unchanged.

The default approximation is **OMP with `--k 64`** (arXiv:2506.06607): each new token's teacher-space embedding is expressed as a sparse combination of at most k shared tokens, and the same coefficients are applied to the student's embeddings of those tokens. This is the recipe Arcee used to build **SuperNova-Medius** (Llama-3.1-405B logits distilled into Qwen2.5-14B: transplant → offline top-K capture → KD → transplant back).

**Dependency**: transplantation wraps Arcee's `mergekit-tokensurgeon` as an **external subprocess** — mergekit is LGPL-3.0 and is deliberately *not* a surogate dependency (never imported or vendored). Install it with `pip install mergekit` so `mergekit-tokensurgeon` is on `PATH` (or run in a separate environment via `uvx --from mergekit ...` to avoid its dependency pins).

The workflow:

```bash
# 1) Transplant the teacher's tokenizer onto the student
surogate transplant-tokenizer --student Qwen/Qwen3-8B --teacher deepseek-ai/DeepSeek-V3 \
    --output ./qwen3-8b-dsv3-vocab --device cuda

# 2) Point the KD config's `model:` at ./qwen3-8b-dsv3-vocab
#    (distillation.teacher_model stays the real teacher)

# 3-5) Run the standard pipeline against it — now same-tokenizer by construction
surogate tokenize config.yaml          # optional; distill-capture tokenizes if needed
surogate distill-capture config.yaml
surogate sft config.yaml

# 6) Optional: restore the student's native tokenizer, then run a short
#    healing SFT on native-template data
surogate transplant-tokenizer --restore ./qwen3-8b-dsv3-vocab/transplant_manifest.json \
    --student ./output/merged --output ./qwen3-8b-distilled-native
```

Notes:

- The transplanted model carries the teacher's tokenizer *and chat template*, so tokenization, loss masking, and capture are consistent end-to-end; the tokenize-hash machinery guards drift as usual.
- The forward transplant needs no separate healing stage — the KD run itself supplies dense gradients to exactly the rows that were approximated. After the *reverse* transplant (`--restore`), a short SFT on native-template data is recommended before release (chat-template semantics shift back with the tokenizer).
- A `transplant_manifest.json` is written next to the transplanted model; `--restore` consumes it (the original student acts as the tokenizer donor, reusing the recorded method and k).
- **Tied-embedding students**: small models with `tie_word_embeddings=true` (e.g. Qwen3-0.6B/1.7B) share one matrix between input embeddings and LM head, so the transplant solves a compromise between the two approximations. They heal, but expect a few hundred million KD tokens before quality recovers; prefer untied students where possible.
- `--method` accepts mergekit's other approximation methods (`common_interpolation`, `subword`, `mean`, ...), but `omp` is the recommended default.

## Limitations (v1)

- **Shared tokenizer only.** The teacher's token ids are stored against the student's token stream. Capture validates that every token id in the shards is below the teacher's vocab size, but that is a necessary check, not a sufficient one — using teachers from a different tokenizer family produces garbage supervision. Escape hatch: transplant the teacher's tokenizer onto the student first — see [Cross-tokenizer distillation](#cross-tokenizer-distillation).
- **Offline only.** No online/on-the-fly teacher during training; the teacher signal is frozen at capture time (capture itself may query a served teacher over the [vLLM API](#remote-teacher-vllm-api), but the result is still a fixed sidecar).
- **CUDA graphs are auto-disabled** (`use_cuda_graphs: false` is forced); KD micro-steps run eagerly.
- **Eval loss is CE-only.** The KD term is train-time only.
- **Single-node only.** Combining `distillation:` with a `distributed:` block raises an error, and `parallelism: dispatch_pp` is rejected.
- **LM-head row compaction is auto-disabled.** `lmhead_drop_ignored_rows` is forced off with a warning.
- **No on-the-fly vision training.** `train_vision` is not supported with KD (KD needs tokenized shards with sidecars).
- Sidecar logprobs are fp16 only (`logprob_dtype = 0`).

## Troubleshooting

All sidecars are validated before training starts (magic, version, `n_tokens`, `top_k`, file size, and the embedded tokenize hash against `{output_dir}/.tokenize_hash`). The errors are actionable:

**Missing sidecar** — `KD sidecar '<shard>.kd' is missing for token shard '<shard>'. Run 'surogate distill-capture <config.yaml>' to capture teacher logprobs.`
You ran `surogate sft` with a `distillation:` block before capturing. Run the capture step. (When `kd_dir` is set, the trainer looks there and symlinks `<shard>.kd` automatically — but if a *different* real sidecar already sits at `<shard>.kd`, training aborts with `distillation.kd_dir points at <sidecar> but a different sidecar already exists at <shard>.kd; remove one of them.`)

**Tokenize-hash mismatch** — `KD sidecar '<path>' embeds tokenize hash '<old>' but the current token shards have hash '<new>'. The token shards were re-tokenized — re-run 'surogate distill-capture'.`
Anything that changes tokenization (datasets, `sequence_len`, packing, template) re-tokenizes the shards and invalidates existing sidecars; the alignment between teacher rows and tokens would silently break, so training refuses to start. Re-run the capture (it recaptures only invalid sidecars). A `n_tokens` mismatch between sidecar and shard is caught the same way.

**top_k mismatch** — `KD sidecar '<path>' was captured with top_k=<K> but the config requests top_k=<K'>. Re-run 'surogate distill-capture' with the desired distillation.top_k.`
`distillation.top_k` is baked into the sidecars at capture time. Either set the config back to the captured value or recapture.

**Truncated sidecar** — `KD sidecar '<path>' is truncated or oversized: ... The capture likely did not finish — re-run 'surogate distill-capture'.`
Captures write to a `.tmp` file and rename atomically on completion, so this normally means a stray or hand-copied file; re-running the capture fixes it.

**Missing `.tokenize_hash`** — `distillation is enabled but no <output_dir>/.tokenize_hash was found...`
The output dir has shards but no tokenization hash (e.g. partially copied). Re-run tokenization and the capture.

**Capture: flash-attention-2 not available** — `distill-capture requires flash-attention-2 for per-document attention isolation...`
Install flash-attn, or accept the cross-document-attention approximation with `--allow-cross-doc-attention` (packed documents will then see each other during teacher capture).

**Capture: token id out of teacher vocab** — `Token shard '<path>' contains token id <X> but the teacher vocab size is <Y>. Student and teacher must share a tokenizer. For cross-tokenizer distillation, first transplant the teacher's tokenizer onto the student ... then re-tokenize and re-capture against the transplanted model.`
Your teacher does not share the student's tokenizer. Pick a teacher from the same family, or follow the error's advice — see [Cross-tokenizer distillation](#cross-tokenizer-distillation).

**Capture (API): server rejects or omits `prompt_logprobs`** — see [Remote teacher](#remote-teacher-vllm-api): the endpoint must be a vLLM-compatible server started with `--max-logprobs >= top_k`; aggregators such as OpenRouter cannot be used.

## Example

A complete runnable config lives at [`examples/distillation/qwen3-kd.yaml`](https://github.com/invergent-ai/surogate/blob/main/examples/distillation/qwen3-kd.yaml) (Qwen3-1.7B teacher → Qwen3-0.6B student).

## See also

- [Configuration](configuration.md)
- [Datasets](datasets.md)
- [Metrics](metrics.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
