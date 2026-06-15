// Run isolation: each launched training writes to its OWN metrics feed so
// multiple concurrent runs (SFT, GRPO, …) never interleave into one file.
// The dashboard can list these feeds and follow whichever run you pick.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const RUNS_DIR = path.join(os.homedir(), ".surogate-watch", "runs");

export function ensureRunsDir(): string {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  return RUNS_DIR;
}

function slug(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "run";
}

/** A fresh, unique feed path for a new run (sortable timestamp + label). */
export function newRunFeedPath(label: string, nowMs: number): string {
  ensureRunsDir();
  const d = new Date(nowMs);
  const pad = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  return path.join(RUNS_DIR, `${stamp}-${slug(label)}.jsonl`);
}

export interface RunInfo {
  path: string;
  name: string;
  mtimeMs: number;
  sizeBytes: number;
  ageMs: number; // since last write, vs nowMs passed in
}

/** List known run feeds (the runs dir + any extra paths like the default feed),
 *  newest first. */
export function listRuns(extraPaths: string[], nowMs: number): RunInfo[] {
  const seen = new Set<string>();
  const out: RunInfo[] = [];
  const add = (p: string) => {
    const abs = path.resolve(p);
    if (seen.has(abs)) return;
    seen.add(abs);
    try {
      const st = fs.statSync(abs);
      if (!st.isFile()) return;
      out.push({
        path: abs,
        name: path.basename(abs),
        mtimeMs: st.mtimeMs,
        sizeBytes: st.size,
        ageMs: Math.max(0, nowMs - st.mtimeMs),
      });
    } catch {
      /* missing */
    }
  };
  try {
    for (const f of fs.readdirSync(RUNS_DIR)) if (f.endsWith(".jsonl")) add(path.join(RUNS_DIR, f));
  } catch {
    /* no runs dir yet */
  }
  for (const p of extraPaths) add(p);
  out.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return out;
}

/** A run is "active" if its feed was written within the last ~20s. */
export function isActive(r: RunInfo): boolean {
  return r.ageMs < 20_000;
}
