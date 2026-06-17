// Control a launched run via its PID sidecar (<feed>.pid). To guard against a
// stale .pid whose number the OS has recycled, the sidecar records the process
// start-time at launch and we re-check it before signalling — a recycled PID has
// a different start-time, so it can never be mistaken for our run.

import fs from "node:fs";

export interface RunPid {
  pid: number;
  start: string | null; // /proc/<pid>/stat field 22 (starttime), null off Linux
}

/** Read /proc/<pid>/stat field 22 (starttime, clock ticks since boot). Linux
 *  only; null elsewhere or if the process is gone. */
export function procStartTime(pid: number): string | null {
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, "utf8");
    // comm (field 2) can contain spaces/parens — start parsing after the final ')'.
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    return fields[19] ?? null; // field 22 = index 19 once comm is dropped
  } catch {
    return null;
  }
}

export function readRunPid(feedPath: string): RunPid | null {
  try {
    const raw = fs.readFileSync(`${feedPath}.pid`, "utf8").trim();
    if (raw.startsWith("{")) {
      const o = JSON.parse(raw) as { pid?: unknown; start?: unknown };
      const pid = Number(o.pid);
      return Number.isInteger(pid) && pid > 0 ? { pid, start: typeof o.start === "string" ? o.start : null } : null;
    }
    const pid = Number(raw); // legacy bare-integer pid file
    return Number.isInteger(pid) && pid > 0 ? { pid, start: null } : null;
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

/** True if the live process is the one we launched, not a recycled PID. Prefers
 *  the recorded start-time; falls back to a cmdline check for legacy pid files. */
function isOurRun(p: RunPid): boolean {
  if (!pidAlive(p.pid)) return false;
  if (p.start !== null) return procStartTime(p.pid) === p.start;
  try {
    return fs.readFileSync(`/proc/${p.pid}/cmdline`, "utf8").includes("surogate");
  } catch {
    return process.platform !== "linux"; // can't verify off Linux — trust liveness
  }
}

/** True if this feed has a known, still-running launched process we can confirm
 *  is ours. */
export function runControllable(feedPath: string): boolean {
  const p = readRunPid(feedPath);
  return p !== null && isOurRun(p);
}

/** Signal a launched run's whole process group (it was spawned `detached`, so
 *  children — vLLM, workers — get the signal too). Falls back to the single pid. */
export function signalRun(feedPath: string, signal: NodeJS.Signals): boolean {
  const p = readRunPid(feedPath);
  if (p === null || !isOurRun(p)) return false; // never signal a stale/recycled PID
  try {
    process.kill(-p.pid, signal); // negative pid = process group
    return true;
  } catch {
    try {
      process.kill(p.pid, signal); // fall back to the single process
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
