# Dispatch-PP — Stage-Level Dispatch Rewrite (next dedicated pass)

**Status:** planned. Grounded in a full read of `study/RoundPipe/roundpipe/` (run.py, device.py,
transfer.py, scheduler.py, roundpipe.py), not inference. The current `dispatch_pp_train_step_multigpu`
(microbatch-diagonal wavefront) must be **replaced** by this model — it cannot be extended into it.

## Why (what RoundPipe actually does — verified in the source)

1. **Whole model pinned in CPU** (`memory.py:pin_module_alloc`).
2. **Dispatch unit = a whole stage** (`roundpipe.py:444`: `for stage in fwd_plan: get_next_device().launch_forward(stage)`),
   round-robin to per-device controller threads (`device.py:DeviceManager.controller` + `launch_forward`).
3. **A stage is held resident across ALL its microbatches** (`run.py:run_forward`): `upload_layers` at
   `batch_idx==0` (line 334), reuse `context.gpu_fwd_layers` for microbatches 1..M-1, `download_layer`
   (free) at `batch_idx==num_microbatches-1` (line 412). ⇒ **each stage streams once per step, reused by
   all M microbatches** = the amortization.
4. **N stages run on N devices concurrently**, pipelined by the tracker semaphores
   (`forward_wait_for`/`forward_notify`, `is_idle` backpressure) ⇒ the N× parallelism (N parallel PCIe
   links streaming N different stages at once).
5. **Stages are small** — the planner (`scheduler.py:ModelExecutePlan.auto`) sizes them so **2× the
   largest stage ≤ VRAM budget**. That is why a 235B LoRA fits on a single 24 GB 4090, and why a 27B
   fits comfortably on 4×32 GB. (`get_min_gpu_memory()*0.6`, then `/2` per stage.)
6. `num_microbatch = num_devices + 1` (`run_config.py:163`).

The combined effect: full model streamed **once per step** over **N parallel links** =
`full_model / (N · PCIe_BW)` per step — N× better than the committed wavefront, which re-streams
(`full_model · M / (N·BW)` = `full_model/BW` for M=N, i.e. only the parallel-links half ⇒ ~2.2×).

## What's already committed (foundations this pass builds on)

- `fb3879be` async per-GPU dispatch primitive (`dispatch_async`/`wait_gpu`, per-GPU done counters).
- `b249fae3` microbatch machinery (grad-accumulated; `backward_stage(micro_step, total_micro)`,
  `dispatch_pp_zero_grads`, `GradAccumSteps=M` normalization).
- `deffd4cb` microbatch-diagonal wavefront (2.2× on 27B) — **to be replaced** by stage-level dispatch.
- Non-block weight offload for LoRA frozen base (`dsl_weight_manager.cpp:268-` `freeze_base` branch):
  embedding/lm_head masters → pinned CPU, freeing ~13 GB so a resident stage fits. (Verified correct on
  0.8B; **uncommitted** at time of writing — re-land it.)

## The rewrite (4 pieces)

### 1. DispatchPlanner — small stages
Port `ModelExecutePlan.auto` (see `study/RoundPipe/DISPATCH_PP_REFERENCE.md §2`). v1 can be simpler than
the asymmetric cost search: pick `stage_blocks` so `2 · stage_blocks · block_bytes(bf16) + activations ≤
vram_budget`. For the 27B: ~4-block stages → 16 stages on 4 GPUs. Replaces the current
`nst = gpus` even partition in `trainer.py:run_training_loop`. Forward stages ascending, backward
descending; `num_stages > gpus` is the norm.

### 2. Stage-resident gather (DslWeightManager)
Make `kNumPrefetchBuffers` **configurable** (array→vector), set to ~2× the small-stage block count (hold
current stage + prefetch next). Add a range API mirroring `upload_layers`/`download_layer`:
`gather_stage(lo, hi)` (gather + hold), `release_stage(lo, hi)` (free). With slots ≥ stage size the
existing `gather_block` version-cache already keeps a stage resident across its microbatches; the new
part is the explicit hold/release lifetime and the configurable capacity. Keep the non-block offload.

### 3. Stage-level dispatch (replace the wavefront)
Per device, a **stage job** = gather the stage (resident) → run all M microbatches (forward or backward)
→ free. Mirrors `device.py:launch_forward` + `controller` looping `run_forward` over the run_context list.
Dispatch stages round-robin (`dispatch_async`), `is_idle`/`wait_gpu` backpressure (stage k waits for GPU
k mod N to free from stage k−N). A device runs its stages (k, k+N, …) sequentially, one resident at a
time → peak ≈ 2× one small stage.

### 4. Cross-stage per-microbatch pipeline (the hard part)
Stage k+1's microbatch m needs stage k's microbatch m boundary, while stage k continues m+1. This is the
tracker (`scheduler.py:ModelTracker`, `forward_wait_for`/`notify`) + shared per-microbatch boundary state
(`batch.flatten_states`). Needs per-(stage,microbatch) events, not the coarse `wait_gpu`. surogate's AOT
backward graph removes RoundPipe's autograd-tag machinery (`DISPATCH_PP_REFERENCE.md §7`) — encode the
order with explicit CUDA events + the boundary handoff already used by the wavefront, made per-microbatch.

### Then: async 1-step-stale optimizer (`optim_stream.py`) — move the apply off the critical path
(`mDispatchPpPendingGrads` already stashes stale grads).

## Verification plan
- 0.8B LoRA: loss parity vs sequential at each step.
- 27B LoRA (4×32 GB): confirm it **fits** with small stages (the memory objection was a big-stage bug),
  then measure tok/s vs the ~2.9k target (RoundPipe 32B@4-GPU). Expect the stage-resident + pipeline to
  go from ~2.2× (wavefront) toward ~N× over the M=1 baseline.

## Hard-won corrections (do not repeat)
- **Trust the RoundPipe benchmarks.** 235B on 24 GB ⇒ 27B on 32 GB fits with margin. "Doesn't fit" was a
  big-stage (`num_stages=gpus`, 16-block) bug, not a wall.
- **Stages must be small** (planner), and a GPU holds **one stage at a time** (streams through its
  stages), not its whole `num_layers/N` share. The microbatch-diagonal wavefront forces the whole-share
  residency and is the wrong schedule.
