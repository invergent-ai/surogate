# Native GRPO Step Design

## Goal

Replace the Python-mediated GRPO forward/loss/backward sequence with a native trainer API that keeps trainer logprobs and per-token gradient generation on the CUDA stream. The first implementation must remove the `forward_for_grpo()` host logprob barrier from the production GRPO training path while preserving the old APIs for compatibility and parity tests.

## Current Path

The Python trainer currently calls `forward_for_grpo()`, receives a NumPy logprob array, computes GRPO per-token gradients with `surogate.grpo.loss.compute_grpo_per_token_grads()`, then calls `backward_grpo()` with a NumPy gradient array. The C++ runtime allocates temporary device buffers with `cudaMalloc`, copies from pageable host arrays, synchronizes to return logprobs, and synchronizes again before freeing custom dloss buffers.

## Proposed Path

Add `step_grpo_native()` to the Python binding and C++ trainer. The method accepts the existing input IDs, targets, position IDs, temperatures, and GRPO loss-side arrays: inference logprobs, advantages, loss mask, sample range starts/ends, optional teacher logprobs, and scalar loss parameters. The runtime copies these arrays through pinned staging buffers, runs the training forward, computes custom dloss in a CUDA kernel from `rs.Losses`, then runs backward on the same stream.

## API

Production Python code should call:

```python
trainer.step_grpo_native(
    input_ids,
    targets,
    inference_logprobs,
    advantages,
    loss_mask,
    sample_starts,
    sample_ends,
    position_ids=None,
    temperatures=None,
    teacher_logprobs=None,
    loss_scale=1.0,
    ipo_mask_low=0.2,
    ipo_mask_high=0.2,
    adv_tau=1.0,
    teacher_tau=0.0,
    kl_tau=1e-3,
)
```

The old `forward_for_grpo()` and `backward_grpo()` methods remain available.

## Native Loss Semantics

The native kernel must match `surogate/grpo/loss.py` for custom dloss:

```text
trainer_logprob = shifted(-rs.Losses)
log_ratio = trainer_logprob - inference_logprob
ratio = exp(log_ratio)
trainer_prob = exp(trainer_logprob)
inference_prob = exp(inference_logprob)
masked = (trainer_prob - inference_prob > ipo_mask_high)
      or (trainer_prob - inference_prob < -ipo_mask_low)
keep = loss_mask and not masked
scaled_adv = adv_tau * advantage
if teacher_logprobs: scaled_adv += teacher_tau * (teacher_logprob - trainer_logprob)
dloss = keep * scaled_adv * ratio - loss_mask * 2 * kl_tau * log_ratio
dloss /= loss_scale
```

The kernel writes dloss in the unshifted target layout expected by the existing LM-head backward. Token 0 receives zero. Token `t - 1` receives the gradient computed for logical token `t`, matching the Python left-shift of `per_token_grads_flat`.

## Metrics

The first production implementation may return no detailed metrics from C++. Python should keep existing metric code only for a debug/parity path. Reduced native metrics can be added later without changing the core API.

## Testing

Add CPU parity coverage for the shift and GRPO dloss formula. Add a binding smoke test where feasible. The first runtime verification target is successful compilation plus old Python loss tests continuing to pass.
