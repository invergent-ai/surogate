// Pull a finished run's outputs (trained model, checkpoints, logs) back to the
// local machine — under <run-folder>/output. Dispatches by where the run ran:
//   • local  → already on disk (the run folder), nothing to fetch
//   • ssh    → scp -r the remote output dir back
//   • modal  → `modal volume get` the run's Volume
//   • dstack → not downloadable (dstack tears the instance down — push to the Hub)
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { parseSshTarget, scpBaseArgs } from "./ssh.ts";
import { type RunMeta, readRunMeta, runArtifacts } from "./runs.ts";

export interface FetchResult {
  ok: boolean;
  dest?: string;
  reason?: string;
}

function run(cmd: string, args: string[], onLine?: (l: string) => void): Promise<{ ok: boolean; code: number | null }> {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let buf = "";
    const pump = (c: Buffer) => {
      if (!onLine) return;
      buf += c.toString();
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        onLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    };
    child.stdout?.on("data", pump);
    child.stderr?.on("data", pump);
    child.on("error", (e) => {
      onLine?.(`error: ${e.message}`);
      resolve({ ok: false, code: null });
    });
    child.on("close", (code) => resolve({ ok: code === 0, code }));
  });
}

/** Where artifacts will land for a given run feed. */
export function artifactDest(feedPath: string): string {
  return path.join(runArtifacts(feedPath).dir, "output");
}

/** Fetch a run's artifacts. `feedPath` is the run's metrics.jsonl. onLine streams
 *  progress (scp / modal CLI output) for a live log. */
export async function fetchArtifacts(feedPath: string, onLine?: (l: string) => void): Promise<FetchResult> {
  const meta: RunMeta | null = readRunMeta(feedPath);
  const dest = artifactDest(feedPath);
  fs.mkdirSync(dest, { recursive: true });
  const r = meta?.remote;

  if (!r) {
    // local run — outputs are already in the run folder
    return { ok: true, dest };
  }

  if (r.kind === "ssh") {
    const t = parseSshTarget(r.host);
    onLine?.(`scp ${t.host}:${r.dir}/output → ${dest}`);
    const res = await run("scp", ["-r", ...scpBaseArgs(t), `${t.host}:${r.dir}/output/.`, dest], onLine);
    return res.ok ? { ok: true, dest } : { ok: false, reason: "scp failed — see the fetch log" };
  }

  if (r.kind === "modal") {
    // surogate writes to /outputs/model on the Volume; logs at /train.log
    onLine?.(`modal volume get ${r.session} → ${dest}`);
    const m = await run("modal", ["volume", "get", "--force", r.session, "/model", dest], onLine);
    await run("modal", ["volume", "get", "--force", r.session, "/train.log", dest], onLine).catch(() => {});
    return m.ok ? { ok: true, dest } : { ok: false, reason: "`modal volume get` failed — see the fetch log" };
  }

  return {
    ok: false,
    reason: "dstack runs aren't downloadable (the instance is torn down) — configure a Hub push, or use SSH/Modal",
  };
}

/** Can this run's artifacts be fetched back? (false for dstack / unknown.) */
export function isFetchable(feedPath: string): boolean {
  const r = readRunMeta(feedPath)?.remote;
  return !r || r.kind === "ssh" || r.kind === "modal";
}
