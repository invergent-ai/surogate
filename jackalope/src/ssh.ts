// Remote compute over SSH — the substrate for "bring your own GPU server" (and,
// later, managed providers that hand you an SSH box). We shell out to the system
// `ssh`/`scp` so it honors ~/.ssh/config, the agent, keys, and ProxyJump.
//
// Model: run `surogate` on the remote inside a detached tmux session, and mirror
// the remote metrics.jsonl back into a LOCAL run folder with `tail -F`, so the
// rest of the dashboard (Feed, Runs, Monitor) works unchanged on a local file.

import { type ChildProcess, execFile, execFileSync, spawn } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import path from "node:path";

const execFileP = promisify(execFile);

import os from "node:os";
const HOSTS_FILE = path.join(os.homedir(), ".surogate-watch", "ssh-hosts");

/** Recently-used SSH targets (newest first), for the GPUs tab. */
export function recentSshHosts(): string[] {
  try {
    return fs
      .readFileSync(HOSTS_FILE, "utf8")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
      .slice(0, 6);
  } catch {
    return [];
  }
}

/** Remember an SSH host (most-recent first, deduped). */
export function rememberSshHost(host: string): void {
  const h = host.trim();
  if (!h) return;
  try {
    const list = [h, ...recentSshHosts().filter((x) => x !== h)].slice(0, 6);
    fs.mkdirSync(path.dirname(HOSTS_FILE), { recursive: true });
    fs.writeFileSync(HOSTS_FILE, list.join("\n") + "\n");
  } catch {
    /* best effort */
  }
}
import { type Gpu, NVIDIA_SMI_QUERY, parseGpuCsv } from "./launch.ts";
import { newRunFeedPath, runArtifacts, writeRunMeta } from "./runs.ts";

export interface SshTarget {
  host: string; // "user@host" or a ~/.ssh/config alias
  port?: number;
  identityFile?: string;
  workdir?: string; // remote base dir for run folders (default ~/.surogate-watch/remote)
}

/** Parse "user@host:port" / "host" / an alias into an SshTarget. */
export function parseSshTarget(s: string): SshTarget {
  const trimmed = s.trim();
  const portMatch = /^(.*?):(\d+)$/.exec(trimmed);
  if (portMatch) return { host: portMatch[1]!, port: Number(portMatch[2]) };
  return { host: trimmed };
}

/** Common ssh options: non-interactive (fail fast instead of hanging the TUI),
 *  short connect timeout, and keepalives so a dropped link is noticed in ~45s. */
export function sshBaseArgs(t: SshTarget): string[] {
  const args = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
  ];
  if (t.port) args.push("-p", String(t.port));
  if (t.identityFile) args.push("-i", t.identityFile);
  return args;
}

// scp uses -P (capital) for port and -i for the key — different from ssh.
export function scpBaseArgs(t: SshTarget): string[] {
  const args = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"];
  if (t.port) args.push("-P", String(t.port));
  if (t.identityFile) args.push("-i", t.identityFile);
  return args;
}

const remoteBase = (t: SshTarget) => t.workdir ?? "~/.surogate-watch/remote";

/** The detached tmux command that runs surogate on the remote. Pure so it can be
 *  unit-tested. */
export function remoteLaunchCommand(remoteDir: string, session: string, surogateBin: string): string {
  const feed = `${remoteDir}/metrics.jsonl`;
  // tmux keeps it alive + reattachable; redirect all streams so ssh returns.
  return (
    `mkdir -p ${remoteDir} && tmux new-session -d -s ${session} ` +
    `'cd ${remoteDir} && SUROGATE_METRICS_PATH=${feed} ${surogateBin} sft config.yaml > train.log 2>&1 < /dev/null'`
  );
}

/** Probe remote GPUs over SSH (non-blocking). Throws with a readable reason. */
export async function detectRemoteGpus(t: SshTarget): Promise<Gpu[]> {
  const { stdout } = await execFileP("ssh", [...sshBaseArgs(t), t.host, "nvidia-smi", ...NVIDIA_SMI_QUERY], {
    timeout: 15000,
  });
  return parseGpuCsv(stdout);
}

// Active mirrors, each with a stop() that flips its `stopped` flag AND kills the
// live tail — so stopping actually halts the reconnect loop (not just the current
// child). Killed on stop / app exit so no orphan `ssh tail` keeps reconnecting.
const mirrors = new Set<{ stop: () => void }>();
export function killMirrors(): void {
  for (const m of mirrors) m.stop();
  mirrors.clear();
}

/** Mirror a remote file into a local file via `tail -F`, reconnecting with
 *  backoff if the link drops. Appends to the local file (Feed tails it). */
function mirrorRemoteFile(t: SshTarget, remotePath: string, localPath: string): void {
  let stopped = false;
  let attempt = 0;
  let current: ChildProcess | null = null;
  const handle = {
    stop: () => {
      stopped = true;
      try {
        current?.kill();
      } catch {
        /* ignore */
      }
    },
  };
  mirrors.add(handle);
  const spawnTail = () => {
    if (stopped) return;
    const fd = fs.openSync(localPath, "a");
    const child = spawn("ssh", [...sshBaseArgs(t), t.host, "tail", "-n", "+1", "-F", remotePath], {
      stdio: ["ignore", fd, "ignore"],
    });
    current = child;
    child.on("exit", () => {
      try {
        fs.closeSync(fd);
      } catch {
        /* ignore */
      }
      if (stopped) {
        mirrors.delete(handle);
        return;
      }
      attempt += 1;
      // back off: 1s, 2s, then 5s thereafter — survive transient drops
      const delay = attempt === 1 ? 1000 : attempt === 2 ? 2000 : 5000;
      setTimeout(spawnTail, delay);
    });
    child.on("error", () => {});
  };
  spawnTail();
}

export interface RemoteLaunchResult {
  ok: true;
  feed: string;
  session: string;
  host: string;
}
export type RemoteLaunch = RemoteLaunchResult | { ok: false; reason: string };

/** Launch a run on the remote and start mirroring its feed + log locally.
 *  `configText` is the surogate YAML (already built by the caller). */
export async function launchRemoteRun(
  t: SshTarget,
  configText: string,
  label: string,
  surogateBin = "surogate",
): Promise<RemoteLaunch> {
  const feed = newRunFeedPath(`ssh-${label}`, Date.now());
  const art = runArtifacts(feed);
  const runId = path.basename(art.dir);
  const remoteDir = `${remoteBase(t)}/${runId}`;
  const session = `sur_${runId}`.replace(/[^a-zA-Z0-9_]/g, "_");

  try {
    fs.writeFileSync(art.configPath, configText);
    // copy the config up, then start it detached
    await execFileP("ssh", [...sshBaseArgs(t), t.host, `mkdir -p ${remoteDir}`], { timeout: 15000 });
    await execFileP("scp", [...scpBaseArgs(t), art.configPath, `${t.host}:${remoteDir}/config.yaml`], { timeout: 30000 });
    await execFileP("ssh", [...sshBaseArgs(t), t.host, remoteLaunchCommand(remoteDir, session, surogateBin)], {
      timeout: 20000,
    });
  } catch (e) {
    return { ok: false, reason: (e as Error).message.split("\n")[0] || "ssh/scp failed" };
  }

  writeRunMeta(feed, {
    mode: "sft",
    startedAt: Date.now(),
    label: `ssh-${label}`,
    remote: { kind: "ssh", host: t.host, session, dir: remoteDir },
  });
  // mirror metrics (drives the dashboard) + the run log (Logs tab)
  mirrorRemoteFile(t, `${remoteDir}/metrics.jsonl`, feed);
  mirrorRemoteFile(t, `${remoteDir}/train.log`, art.logPath);
  return { ok: true, feed, session, host: t.host };
}

/** Stop a remote run by killing its tmux session. */
export function stopRemoteSession(t: SshTarget, session: string): boolean {
  try {
    execFileSync("ssh", [...sshBaseArgs(t), t.host, `tmux kill-session -t ${session}`], { timeout: 15000 });
    return true;
  } catch {
    return false;
  }
}

// best-effort cleanup so closing the dashboard doesn't leave orphan tails
process.on("exit", killMirrors);
