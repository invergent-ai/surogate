# Candidate-only supervised loss

The offline-sidecar path supports hard-label classification over a supplied
candidate set. For each supervised position with target token `y` and candidate
token IDs `S`, the loss is:

```text
L = logsumexp(logits[S]) - logits[y]
```

Excluded vocabulary tokens have exactly zero loss gradient. Ignored target
positions (`-100`) contribute neither loss nor gradient. This mode does not
use a teacher, teacher probabilities, generated reasoning, or reinforcement
learning. The standard SFT loss and gradients are unchanged when it is disabled.

```yaml
distillation:
  candidate_only: true
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

Current implementation limits:

- Reuses the sidecar loader and eager KD scheduling path; CUDA graphs and
  ignored-row LM-head compaction are disabled. It still computes the full
  vocabulary projection, so this is not a reduced-head compute optimization.
- Uses normal token-count normalization and supports gradient accumulation and
  local multi-GPU data parallelism. Existing KD restrictions on multi-node and
  dispatch-PP execution still apply.
- Built-in `validate()` remains full-vocabulary CE, so this configuration requires
  `eval_steps: 0`. Evaluate candidate metrics separately with the same prompt
  and allowed-token mapping. Do not label ordinary validation CE as candidate CE.
- `get_kd_loss()` exposes candidate CE in this mode but retains its existing
  rank-zero-local metric contract. Standard training `loss` is the reduced
  candidate objective; the two can differ for different per-rank examples.
- Probabilities need separate calibration evaluation. Hard-label training does
  not guarantee calibration or correct decisions.

Tests cover FP32/BF16 loss and gradients, small/chunked vocabularies, masked rows,
variable candidate counts, aliased buffers, softcap derivatives, invalid candidate
sets, and LoRA gradient accumulation. The standalone Jev experiment bridge also
checks candidate alignment through the actual shuffled native loader.
