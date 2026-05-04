# Native GRPO Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native GRPO training step that computes per-token custom dloss in C++/CUDA and removes the production `forward_for_grpo()` host logprob round trip.

**Architecture:** Keep the existing compatibility APIs, add a new `step_grpo_native()` binding, and route `surogate/grpo/trainer.py` through it. Use allocator-backed scratch buffers in `DslRunState` for device loss inputs and custom dloss. Add a CUDA helper that derives shifted trainer logprobs from `rs.Losses` and writes the unshifted custom dloss buffer consumed by existing backward execution.

**Tech Stack:** C++17, CUDA kernels, nanobind, NumPy, existing `TensorAllocator`, existing DSL executor request path.

---

### Task 1: Add CPU Parity Test For Native Formula

**Files:**
- Create: `tests/grpo/test_native_formula.py`

- [ ] Add a pure Python reference helper for the native shifted formula.
- [ ] Compare it against `compute_grpo_per_token_grads()` plus the existing left shift for teacher/no-teacher and masked-token cases.
- [ ] Run `pytest tests/grpo/test_native_formula.py -q`.
- [ ] Commit with `test: add native grpo formula parity coverage`.

### Task 2: Add Runtime API Types And Scratch Buffers

**Files:**
- Modify: `csrc/src/runtime/dsl/dsl_run_state.h`
- Modify: `csrc/src/runtime/dsl/dsl_run_state.cpp`
- Modify: `csrc/src/runtime/dsl/dsl_model.h`

- [ ] Add `GrpoNativeScratch` tensors for inference logprobs, advantages, loss mask bytes, teacher logprobs, sample starts, sample ends, custom dloss, and optional inverse temperatures.
- [ ] Allocate scratch during run-state allocation using max `B*T` and a conservative max sample count of `B*T`.
- [ ] Add `DslModel::step_grpo_native(...)` declaration with scalar loss config.
- [ ] Compile the touched headers with the project build target.
- [ ] Commit with `feat: add native grpo scratch state`.

### Task 3: Add Native GRPO CUDA DLoss Helper

**Files:**
- Modify: `csrc/src/kernels/kernels.h`
- Modify: `csrc/src/kernels/kernels.cpp`
- Add or modify CUDA implementation file if GRPO kernels are separated locally.

- [ ] Implement `compute_grpo_custom_dloss(...)`.
- [ ] Match the design formula and shift: logical token `t` reads trainer logprob from `losses[t - 1]` and writes `custom_dloss[t - 1]`; token 0 and masked/non-range slots write zero.
- [ ] Accept `sample_starts` and `sample_ends` so packed sample boundaries can prevent cross-sample shifting.
- [ ] Build to catch signature and launch errors.
- [ ] Commit with `feat: add native grpo dloss kernel`.

### Task 4: Wire C++ Trainer And Binding API

**Files:**
- Modify: `csrc/src/runtime/dsl/dsl_model_execution.cpp`
- Modify: `csrc/src/binding/py_train.h`
- Modify: `csrc/src/binding/py_train.cpp`
- Modify: `csrc/src/binding/binding.cpp`

- [ ] Add `DslModel::step_grpo_native(...)`: upload loss arrays through scratch, run forward, launch native dloss kernel, run backward with scratch custom dloss, clear doc masking, record backward event.
- [ ] Add `MultiGPUPyTrainer::step_grpo_native(...)` that slices multi-GPU rows like existing GRPO methods.
- [ ] Expose nanobind method with ndarray arguments and scalar loss config.
- [ ] Keep old `forward_for_grpo()` and `backward_grpo()` unchanged.
- [ ] Build the extension target.
- [ ] Commit with `feat: expose native grpo step`.

### Task 5: Route Python GRPO Trainer Through Native API

**Files:**
- Modify: `surogate/grpo/trainer.py`

- [ ] Replace production `forward_for_grpo()` / `compute_grpo_per_token_grads()` / `backward_grpo()` sequence with `step_grpo_native()`.
- [ ] Preserve old path behind a local debug flag only if needed for parity.
- [ ] Keep metrics behavior explicit: either compute debug metrics on old path or mark detailed metrics unavailable on native path.
- [ ] Run GRPO-related Python tests.
- [ ] Commit with `feat: use native grpo training step`.

### Task 6: Verification And Cleanup

**Files:**
- Modify only files needed by failing checks.

- [ ] Run targeted unit tests.
- [ ] Run compile/build target used by this repo for C++ extension changes.
- [ ] Run a minimal import smoke test for the binding.
- [ ] Inspect `git diff` for accidental unrelated changes.
- [ ] Commit any final fixes with `fix: stabilize native grpo step`.
