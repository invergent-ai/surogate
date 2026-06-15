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

/** Stop a launched run. We spawned it `detached`, so it leads its own process
 *  group — signal the whole group so children (vLLM, workers) die too. */
export function stopRun(feedPath: string, signal: NodeJS.Signals = "SIGTERM"): boolean {
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
