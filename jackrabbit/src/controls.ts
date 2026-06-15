// Control a launched run: read its PID sidecar (<feed>.pid) and stop it.
// Only runs jackrabbit launched have a PID file, so control is offered only
// for those.

import fs from "node:fs";

export function readRunPid(feedPath: string): number | null {
  try {
    const n = Number(fs.readFileSync(`${feedPath}.pid`, "utf8").trim());
    return Number.isInteger(n) && n > 0 ? n : null;
  } catch {
    return null;
  }
}

export function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0); // signal 0 = liveness probe
    return true;
  } catch (e) {
    // ESRCH = no such process; EPERM = exists but not ours (treat as alive)
    return (e as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** True if this feed has a known, still-running launched process. */
export function runControllable(feedPath: string): boolean {
  const pid = readRunPid(feedPath);
  return pid !== null && pidAlive(pid);
}

/** Signal a launched run's whole process group (it was spawned `detached`, so
 *  children — vLLM, workers — get the signal too). Falls back to the single pid. */
export function signalRun(feedPath: string, signal: NodeJS.Signals): boolean {
  const pid = readRunPid(feedPath);
  if (pid === null) return false;
  try {
    process.kill(-pid, signal); // negative pid = process group
    return true;
  } catch {
    try {
      process.kill(pid, signal); // fall back to the single process
      return true;
    } catch {
      return false;
    }
  }
}

/** Stop a run (SIGTERM the group). */
export function stopRun(feedPath: string): boolean {
  return signalRun(feedPath, "SIGTERM");
}

/** Pause a run (SIGSTOP). Note: freezing a multi-GPU job holds VRAM and can
 *  upset NCCL — best-effort, intended for single-GPU / quick holds. */
export function pauseRun(feedPath: string): boolean {
  return signalRun(feedPath, "SIGSTOP");
}

/** Resume a paused run (SIGCONT). */
export function resumeRun(feedPath: string): boolean {
  return signalRun(feedPath, "SIGCONT");
}
