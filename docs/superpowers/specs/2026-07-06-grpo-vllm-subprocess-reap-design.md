# GRPO shutdown: reap orphaned vLLM subprocesses

## Problem

When a split-mode GRPO run shuts down (normal completion or error), the vLLM
subprocess tree is torn down in the `finally` block of `grpo_split`
(`surogate/grpo/split.py`). The vLLM subprocess deliberately `os.setsid()`s
(split.py:65) so its engine workers share its process group and the parent can
kill the tree with `os.killpg()`. But some workers survive teardown, linger
orphaned, and strand the run.

**Impact.**
- **Platform (dstack):** the runner waits for the job process group to empty →
  the run hangs in `running` forever, holding the pod + GPU slot (Bug 04).
- **Direct terminal (densemax2):** `surogate grpo` returns, but leftover vLLM
  `multiprocessing`/`EngineCore` workers remain and must be `kill -9`'d by hand —
  the recurring "orphaned EngineCore" chore.

## Root cause (confirmed live, 2026-07-06)

Captured the live process tree of an `orch-mini` run mid-flight and again after
exit. The survivor is a vLLM `multiprocessing.spawn` worker — a spawn child of a
vLLM `EngineCore` (`VLLM_WORKER_MULTIPROC_METHOD=spawn`). Two facts about it
defeat every group- or tree-based teardown:

1. **It reparents to PID 1 before teardown.** Its parent (an intermediate spawn
   process under the main process) exits mid-run, so the worker reparents to PID
   1 and *leaves the main process's live descendant tree*. A one-shot
   `psutil.Process().children(recursive=True)` taken in the `finally` block no
   longer finds it.
2. **It is not in the vLLM session group.** It sits in the *launching shell's*
   process group, not the `setsid`'d vLLM session, so `killpg(vllm_pid)` never
   reaches it.

Concretely, from the live capture: main (`surogate`) is **not** a process-group
leader (its pgid is the shell's), the vLLM subprocess is the one `setsid`'d
session leader, and the orphan is a grandchild that starts in the shell's group
and reparents to PID 1 when its parent dies.

This also rules out three tempting fixes:
- **`killpg(vllm_pid)` alone** — misses workers outside the vLLM session group.
- **One-shot descendant snapshot at teardown** — misses workers that already
  reparented to PID 1.
- **Process-group scan** — the orphan *is* in the main process's group, but that
  group is the *shell's* group; sweeping it would kill the user's shell and
  siblings. Unsafe, and skipped entirely whenever main isn't the group leader
  (the normal non-interactive / dstack case).

## Design

Stop the orphans from escaping in the first place, then reap with one walk.

At startup, before spawning anything, the pipeline marks itself a **child
subreaper** (`PR_SET_CHILD_SUBREAPER` via `prctl(2)`, best-effort through
`ctypes`). After that, any orphaned descendant reparents to *this process*
instead of PID 1 — so it stays inside our subprocess tree even when its parent
dies mid-run. This is the same mechanism `tini` / `systemd` / Docker init use.

```python
_PR_SET_CHILD_SUBREAPER = 36  # from <linux/prctl.h>, not in the stdlib

def _set_child_subreaper():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        logger.warning(...)        # best-effort: log + continue if unavailable
```

The `finally` block then needs only a single walk:

1. `_terminate_vllm_trees_in_parallel(watched_vllms)` — the **graceful** path:
   `killpg` gives each vLLM session a chance to release its GPU cleanly and
   reaps the leader's zombie (so reparenting of its children completes).
2. join trainer / watchdog threads.
3. `survivors = psutil.Process().children(recursive=True)` — one snapshot; the
   subreaper guarantees the reparented escapees are in it.
4. `_reap_survivors(survivors)` — SIGTERM-then-SIGKILL by PID anything still
   alive.

### Why this design

- **Prevents the escape instead of chasing it.** The reparent-to-PID-1 case that
  defeated a one-shot snapshot no longer happens: orphans reparent to us, so one
  teardown walk finds them. No run-long polling thread, no accumulation.
- **Catches the wrong-group case.** Reaping by PID needs neither the vLLM session
  group nor the shell's group.
- **Safe by construction.** We only ever reap our own descendants. The launching
  shell is main's *parent*, never a descendant, so it can never be reaped —
  unlike a group scan.
- **Graceful first, forceful second.** `killpg` still lets vLLM shut down cleanly
  and free the GPU; the reap is the safety net for escapees.
- **Degrades cleanly.** If `prctl` is unavailable (non-Linux, restricted
  sandbox) the call logs and continues, leaving the graceful `killpg` teardown;
  the platform and densemax2 are both Linux, so the subreaper is always active
  there.
- **Covers both paths.** The engine shutdown is identical under dstack and in a
  bare terminal, so this fixes the platform hang and the terminal chore at once.

### Rejected alternatives

- **`killpg(vllm_pid)` only / unconditional group SIGKILL** — misses workers
  outside the vLLM session group. Failed live.
- **One-shot descendant snapshot in `finally` (no subreaper)** — misses workers
  that reparented to PID 1 before teardown. Failed live. The subreaper is exactly
  what makes the one-shot walk sufficient.
- **Run-long descendant-monitor thread** (poll `children(recursive=True)` every
  second, union into a registry, reap the union): works and was verified live,
  but chases the escape with a polling thread + unbounded accumulation where the
  subreaper prevents the escape outright. Replaced by the subreaper.
- **Process-group scan** — the orphan's group is the shell's; sweeping it risks
  the user's shell, and it is skipped when main isn't the group leader. Unsafe
  and ineffective.
- **Ops monitor safety-net** (force-terminate the pod if it doesn't exit within N
  seconds): defense-in-depth in a different repo. Deferred; add if the engine fix
  proves insufficient (tracked in the bug doc).

## Testing

- **Unit** (`tests/grpo/test_reap_process_group.py`):
  - `_set_child_subreaper` makes an orphaned **grandchild reparent to the
    subreaper** (not PID 1) so a recursive `children()` walk finds it — the exact
    orphan shape, reproduced in a forked subprocess without vLLM/GPUs (the prctl
    never touches the test runner).
  - `_reap_survivors` kills SIGTERM-ignoring workers; no-op when nothing
    survives. `_terminate_vllm_tree` reaps an in-group worker; no-op for an
    unstarted process.
- **Live — terminal (done):** applied the `split.py` change to the densemax2
  checkout (`/home/monica/work/surogate`, editable-installed → live), ran the
  `orch-mini` GRPO example, and confirmed **zero** leftover
  `multiprocessing`/`EngineCore` processes on exit — the reap force-kills the one
  escaped worker and nothing lingers. Verified independently twice.
- **Live — Studio / platform (deferred, not run):** the platform runs GRPO in a
  pod that pulls a *baked* image (`surogate:latest-cu128`, wheel-installed, no
  per-run code override), so the editable-checkout fix is invisible there.
  Confirming it would need a one-off overlay image pushed to the org registry and
  local ops pointed at it. Skipped as ceremony for a *confirmatory* check: the
  engine fix is the same `grpo_split` path proven in the terminal, and removing
  exactly these orphans is already known to finalize the run. Platform
  confirmation comes from the first real GRPO run after the fixed image ships.

## Scope

- `surogate/grpo/split.py` (`_set_child_subreaper` + its call at `grpo_split`
  startup, `_reap_survivors`, and the single-walk reap in the `finally`) plus
  unit tests.
- One repo, one focused change. Ops safety-net out of scope (deferred).
