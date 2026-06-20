// Run isolation: each launched training gets its OWN directory under
// ~/.surogate-watch/runs/<stamp>-<label>/ holding everything that run produced:
//
//   metrics.jsonl        the live feed the dashboard tails
//   metrics.jsonl.pid    {pid,start} sidecar for pause/resume/stop
//   config.yaml          the exact config it was launched with
//   config.yaml.log      surogate stdout+stderr
//   meta.json            model / dataset / recipe / gpus / maxSteps / startedAt
//   output/              checkpoints (output_dir)
//
// Keeping each run self-contained means concurrent runs never interleave and the
// Runs tab can show a tidy, comparable history.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseLine } from "./records.ts";

export const RUNS_DIR = path.join(os.homedir(), ".surogate-watch", "runs");

export function ensureRunsDir(): string {
  fs.mkdirSync(RUNS_DIR, { recursive: true });
  return RUNS_DIR;
}

function slug(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "run";
}

function stampOf(nowMs: number): string {
  const d = new Date(nowMs);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

/** Create a fresh run directory and return the path of its metrics feed. */
export function newRunFeedPath(label: string, nowMs: number): string {
  ensureRunsDir();
  const dir = path.join(RUNS_DIR, `${stampOf(nowMs)}-${slug(label)}`);
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, "metrics.jsonl");
}

export interface RunArtifacts {
  dir: string;
  configPath: string;
  metaPath: string;
  logPath: string;
  outputDir: string;
}

/** The set of files that live alongside a run's metrics feed. */
export function runArtifacts(feedPath: string): RunArtifacts {
  const dir = path.dirname(feedPath);
  return {
    dir,
    configPath: path.join(dir, "config.yaml"),
    metaPath: path.join(dir, "meta.json"),
    logPath: path.join(dir, "config.yaml.log"),
    outputDir: path.join(dir, "output"),
  };
}

export interface RunMeta {
  mode: string; // sft | grpo | ruler
  model?: string;
  dataset?: string;
  recipe?: string;
  gpus?: number[];
  maxSteps?: number;
  startedAt: number; // ms epoch
  label: string;
  remote?: { kind: "ssh" | "dstack" | "modal"; host: string; session: string; dir: string }; // set for remote runs
}

export function writeRunMeta(feedPath: string, meta: RunMeta): void {
  try {
    fs.writeFileSync(runArtifacts(feedPath).metaPath, JSON.stringify(meta, null, 2));
  } catch {
    /* best effort */
  }
}

export function readRunMeta(feedPath: string): RunMeta | null {
  try {
    return JSON.parse(fs.readFileSync(runArtifacts(feedPath).metaPath, "utf8")) as RunMeta;
  } catch {
    return null;
  }
}

export interface Checkpoint {
  name: string;
  step: number;
}

/** Saved checkpoints in a run's output dir (surogate writes `checkpoint-<step>/`),
 *  newest first. */
export function listCheckpoints(outputDir: string): Checkpoint[] {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(outputDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: Checkpoint[] = [];
  for (const e of entries) {
    const m = /^checkpoint-(\d+)$/.exec(e.name);
    if (e.isDirectory() && m) out.push({ name: e.name, step: Number(m[1]) });
  }
  return out.sort((a, b) => b.step - a.step);
}

export interface FileEntry {
  name: string;
  bytes: number;
  kind: "file" | "checkpoint";
  step?: number; // for checkpoints
  path: string;
}

/** Recursive byte size of a file or directory (best-effort). Uses lstat and
 *  skips symlinks so a symlink cycle can't blow the stack. */
export function pathSize(p: string): number {
  try {
    const st = fs.lstatSync(p);
    if (st.isSymbolicLink()) return 0; // don't follow → no cycles
    if (st.isFile()) return st.size;
    if (!st.isDirectory()) return 0;
    let total = 0;
    for (const e of fs.readdirSync(p)) total += pathSize(path.join(p, e));
    return total;
  } catch {
    return 0;
  }
}

/** Everything a run produced: its config/log/feed/meta files + each checkpoint
 *  (with size + step), newest checkpoints first. */
export function runFileEntries(feedPath: string): FileEntry[] {
  const art = runArtifacts(feedPath);
  const out: FileEntry[] = [];
  const addFile = (name: string, p: string) => {
    try {
      const st = fs.statSync(p);
      if (st.isFile()) out.push({ name, bytes: st.size, kind: "file", path: p });
    } catch {
      /* missing */
    }
  };
  addFile("config.yaml", art.configPath);
  addFile("meta.json", art.metaPath);
  addFile("metrics.jsonl", feedPath);
  addFile("train.log", art.logPath);
  for (const cp of listCheckpoints(art.outputDir)) {
    const p = path.join(art.outputDir, cp.name);
    out.push({ name: cp.name, bytes: pathSize(p), kind: "checkpoint", step: cp.step, path: p });
  }
  return out;
}

/** Read the last `maxBytes` of a file as text (best-effort; always closes the fd). */
export function tailBytes(p: string, maxBytes: number): string {
  let fd: number | null = null;
  try {
    const st = fs.statSync(p);
    if (!st.size) return "";
    const len = Math.min(st.size, maxBytes);
    const buf = Buffer.alloc(len);
    fd = fs.openSync(p, "r");
    fs.readSync(fd, buf, 0, len, st.size - len);
    return buf.toString("utf8");
  } catch {
    return "";
  } finally {
    if (fd !== null)
      try {
        fs.closeSync(fd);
      } catch {
        /* ignore */
      }
  }
}

/** Cheaply read the last logged step + train loss from the tail of a feed. */
function tailMetrics(feedPath: string): { step: number; loss: number | null } | null {
  const lines = tailBytes(feedPath, 16384).split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const rec = parseLine(lines[i]!);
    if (rec && rec.kind === "step") return { step: rec.step, loss: rec.trainLoss };
  }
  return null;
}

export type RunStatus = "running" | "done" | "stopped" | "idle";

export interface RunInfo {
  path: string;
  name: string; // run folder name (or file name for legacy flat feeds)
  mtimeMs: number;
  sizeBytes: number;
  ageMs: number; // since last write, vs nowMs passed in
  meta: RunMeta | null;
  lastStep: number | null;
  lastLoss: number | null;
  checkpoints: Checkpoint[];
  status: RunStatus;
}

/** A run is "active" if its feed was written within the last ~20s. */
export function isActive(r: { ageMs: number }): boolean {
  return r.ageMs < 20_000;
}

/** Write a JSON summary of a run next to its feed; return the file path. */
export function exportRunSummary(r: RunInfo): string {
  const out = path.join(runArtifacts(r.path).dir, "summary.json");
  const data = {
    name: r.name,
    status: r.status,
    feed: r.path,
    meta: r.meta,
    lastStep: r.lastStep,
    lastLoss: r.lastLoss,
    checkpoints: r.checkpoints,
    sizeBytes: r.sizeBytes,
  };
  fs.writeFileSync(out, JSON.stringify(data, null, 2));
  return out;
}

function classify(active: boolean, lastStep: number | null, maxSteps?: number): RunStatus {
  if (active) return "running";
  if (lastStep === null) return "idle";
  if (maxSteps && lastStep >= maxSteps) return "done";
  return "stopped";
}

/** List known run feeds (run dirs under RUNS_DIR + any extra flat paths like the
 *  default feed), enriched with metadata, last metrics, and checkpoints. Newest
 *  first. */
// Per-run content cache (keyed by mtime+size) so unchanged runs skip the
// meta/tail/checkpoint re-read on every refresh.
const runCache = new Map<string, { mtimeMs: number; size: number; meta: RunMeta | null; lastStep: number | null; lastLoss: number | null; checkpoints: Checkpoint[] }>();

export function listRuns(extraPaths: string[], nowMs: number): RunInfo[] {
  const seen = new Set<string>();
  const out: RunInfo[] = [];
  const add = (feed: string, displayName?: string) => {
    const abs = path.resolve(feed);
    if (seen.has(abs)) return;
    seen.add(abs);
    let st: fs.Stats;
    try {
      st = fs.statSync(abs);
      if (!st.isFile()) return;
    } catch {
      return; // missing
    }
    const ageMs = Math.max(0, nowMs - st.mtimeMs);
    // Re-reading meta/tail/checkpoints is the expensive part and only depends on
    // file content — cache it by (mtime,size) so finished runs aren't re-read on
    // every 2s refresh. ageMs/status still recompute (they depend on `nowMs`).
    let c = runCache.get(abs);
    if (!c || c.mtimeMs !== st.mtimeMs || c.size !== st.size) {
      const tail = tailMetrics(abs);
      c = {
        mtimeMs: st.mtimeMs,
        size: st.size,
        meta: readRunMeta(abs),
        lastStep: tail?.step ?? null,
        lastLoss: tail?.loss ?? null,
        checkpoints: listCheckpoints(runArtifacts(abs).outputDir),
      };
      runCache.set(abs, c);
    }
    out.push({
      path: abs,
      name: displayName ?? path.basename(path.dirname(abs)),
      mtimeMs: st.mtimeMs,
      sizeBytes: st.size,
      ageMs,
      meta: c.meta,
      lastStep: c.lastStep,
      lastLoss: c.lastLoss,
      checkpoints: c.checkpoints,
      status: classify(isActive({ ageMs }), c.lastStep, c.meta?.maxSteps),
    });
  };

  try {
    for (const e of fs.readdirSync(RUNS_DIR, { withFileTypes: true })) {
      if (e.isDirectory()) add(path.join(RUNS_DIR, e.name, "metrics.jsonl"), e.name);
      else if (e.isFile() && e.name.endsWith(".jsonl")) add(path.join(RUNS_DIR, e.name), e.name); // legacy flat feeds
    }
  } catch {
    /* no runs dir yet */
  }
  for (const p of extraPaths) add(p, path.basename(p));
  out.sort((a, b) => b.mtimeMs - a.mtimeMs);
  return out;
}
