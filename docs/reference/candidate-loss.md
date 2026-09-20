# Candidate-only supervised loss

The offline-sidecar path supports hard-label classification over a supplied
candidate set. The backward-compatible default is `candidate_objective: cross_entropy`.
For each supervised position with target token `y` and candidate token IDs `S`,
its loss is:

```text
L = logsumexp(logits[S]) - logits[y]
```

Excluded vocabulary tokens have exactly zero loss gradient. Ignored target
positions (`-100`) contribute neither loss nor gradient. This mode does not
use a teacher, teacher probabilities, generated reasoning, or reinforcement
learning. Disabling candidate mode preserves standard SFT loss selection. The
separate data-parallel LoRA normalization correction described below also applies
to ordinary token-normalized SFT.

```yaml
distillation:
  candidate_only: true
  candidate_objective: cross_entropy  # default; also brier or rps
  top_k: 8          # padded candidate-set width, not a model-selected top-k
  temperature: 1.0
  kd_weight: 1.0
  ce_weight: 0.0
eval_steps: 0
```

The existing `.kd` sidecar format transports the candidate IDs, aligned to the
shifted target positions in each token shard. `-1` pads shorter candidate sets.
The sidecar logprob field is unused; zeros are sufficient. Candidate IDs must be
unique, in the vocabulary, include the gold token, and contain at least two
valid entries. Validation happens before dispatching GPU workers. Use every
supplied candidate, not a top-k approximation from model logits.

The native `step_with_kd(..., candidate_only=True)` API requires the same
temperature and weights shown above. The flag is stamped on both forward and
backward execution requests. Forward loss, valid counts and candidate accuracy
are computed over the restricted set; backward writes sparse-in-vocabulary
gradients into the dense LM-head gradient buffer.


## Brier and ordered RPS

Set `candidate_objective: brier` or `candidate_objective: rps` to use a proper
score with the same gold-token one-hot target. These new objectives require
`lora: true` and `recipe: fp8_hybrid`, enforced by configuration and the native
training binding. Expert parallelism is rejected for these two objectives before
collective dispatch; legacy CE expert-parallel behavior remains unchanged.
They accept 2–255 actual allowed candidates per supervised
row, including when a sidecar has more padding slots. Legacy candidate CE
continues to allow up to 1024 slots/candidates.

For candidate-normalized probabilities `p`, target indicator `t`, and actual
allowed count `C`:

```text
Brier = sum_i (p_i - t_i)^2
RPS   = sum_{k=1}^{C-1} (sum_{i<=k} p_i - sum_{i<=k} t_i)^2 / (C - 1)
```

Brier is the multiclass sum convention: for two labels it is twice scalar
binary Brier. RPS treats the order of non-padding sidecar IDs as the semantic
ordinal order. Interspersed `-1` slots add no boundary and do not change the
`C-1` denominator. Never sort these IDs by token ID or teacher probability.
Use RPS only for homogeneous ordinal batches/runs; a global objective flag is
not a per-row primitive selector and is inappropriate for unordered Choice
labels. This is an explicit proper-score reconstruction, not a claim about
TypeSafe's unpublished RLCD implementation.

All candidate probabilities, CDFs, objective sums and derivative intermediates
are FP32. The final dense LM-head gradient is cast to the existing logit dtype.
Excluded vocabulary derivatives are exactly zero. The kernels emit one raw
objective and gradient per supervised row. Existing global valid-token
normalization applies once after accumulation/all-reduce; there is no extra
microbatch/world-size/candidate-count division beyond RPS's ordinal denominator.
Softcap derivatives and aliased logit/gradient buffers use the existing path.

The `kd_logprobs` array remains a transport placeholder and is ignored for
**all three** candidate objectives, even though ordinary KD interprets it as
teacher data. These flags do not support soft labels, KL, a teacher model,
EMA, policy rollouts, or a combination of CE and proper scores. Keep
`temperature: 1`, `kd_weight: 1`, `ce_weight: 0`, and `eval_steps: 0`.

## Data-parallel LoRA normalization correction

Token-normalized LoRA gradients are averaged across data-parallel ranks, while
valid-token counts are summed globally. The corrected runtime compensates by
`world_size / global_valid_count`, giving the intended global token mean. The
same scale reaches reported norms, clipping and the optimizer's gradient input;
it applies to standard SFT and candidate CE/Brier/RPS together. One-GPU behavior
and non-token-normalized custom-loss paths are unchanged.

Earlier runtimes divided the averaged gradient by the global count again,
leaving gradients/norms smaller by `world_size`. This does **not** imply the
learning rate or Adam updates were smaller by that factor: moment normalization
can cancel uniform gradient scaling, while clipping, epsilon and precision may
change behavior. It does not explain any observed accuracy gap by itself.
Existing checkpoints retain their measured results. New objective comparisons
need a fresh SFT/CE control using the same corrected runtime.

The correction is limited to the proven non-EP data-parallel path. Existing CE
expert-parallel behavior is unchanged; it mixes full-world dense reductions
with DP-subgroup expert reductions and needs separate validation.

Current implementation limits:

- Reuses the sidecar loader and eager KD scheduling path; CUDA graphs and
  ignored-row LM-head compaction are disabled. It still computes the full
  vocabulary projection, so this is not a reduced-head compute optimization.
- Uses normal token-count normalization and supports gradient accumulation and
  local multi-GPU data parallelism. Existing KD restrictions on multi-node and
  dispatch-PP execution still apply.
- Built-in `validate()` remains full-vocabulary CE, so this configuration requires
  `eval_steps: 0`. Evaluate candidate metrics separately with the same prompt
  and allowed-token mapping. Do not label ordinary validation CE as a candidate proper loss.
- `get_kd_loss()` exposes the selected candidate objective in this mode but retains its existing
  rank-zero-local metric contract. Standard training `loss` is the reduced
  candidate objective; the two can differ for different per-rank examples.
- Probabilities need separate calibration evaluation. Hard-label training does
  not guarantee calibration or correct decisions.

Tests cover FP32/BF16 loss and gradients, small/chunked vocabularies, masked rows,
variable candidate counts, aliased buffers, softcap derivatives, invalid candidate
sets, and LoRA gradient accumulation. The standalone Jev experiment bridge also
checks candidate alignment through the actual shuffled native loader.
