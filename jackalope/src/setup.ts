// Setup/onboarding backend: detect whether surogate is installed (locally or on
// a remote), build the right install plan for each compute target, and run that
// install for the user while streaming its output. Grounded in surogate's three
// install modes (docs/getting-started/installation.md):
//   • script — curl -LsSf https://surogate.ai/install.sh | sh  (creates .venv)
//   • source — uv pip install -e .  (editable, for contributors)
//   • docker — the cloud image already ships surogate; only the dstack CLI is local
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripAnsi } from "./ansi.ts";
import { sshBaseArgs, type SshTarget } from "./ssh.ts";

const execFileP = promisify(execFile);

export type Compute = "local" | "ssh" | "dstack";
export type InstallMode = "script" | "source" | "dstack" | "modal";

export interface BinCheck {
  ok: boolean;
  version: string | null; // first line of `--help` output, best-effort
  reason?: string;
}

export interface InstallPlan {
  mode: InstallMode;
  title: string; // "script install"
  command: string; // the shell command shown to the user AND run for them
  cwd?: string; // directory to run it in (source build)
  prereqs: string[]; // human prerequisites the wizard can't do for you
  note: string; // one-line explanation of what it does
  postHint?: string; // a follow-up the user may still need (e.g. activate the venv)
}

function firstLine(s: string): string | null {
  return s.split("\n").map((x) => x.trim()).find(Boolean) ?? null;
}

// argparse prints usage to stdout and exits 0 for `--help`; some builds exit
// nonzero but still print "usage:" — both mean the binary exists.
function helpOk(out: string | undefined): boolean {
  return !!out && /usage:/i.test(out);
}

/** Probe a local surogate binary: `<bin> --help` succeeds when it's installed. */
export async function surogateAvailable(bin: string): Promise<BinCheck> {
  try {
    const { stdout } = await execFileP(bin, ["--help"], { timeout: 8000 });
    return { ok: true, version: firstLine(stdout) };
  } catch (e) {
    const err = e as { code?: string; stdout?: string };
    if (helpOk(err.stdout)) return { ok: true, version: firstLine(err.stdout!) };
    return { ok: false, version: null, reason: err.code === "ENOENT" ? "not on PATH" : (err.code ?? "not runnable") };
  }
}

/** Probe surogate on a remote host over SSH. */
export async function surogateAvailableRemote(t: SshTarget, bin: string): Promise<BinCheck> {
  try {
    const { stdout } = await execFileP("ssh", [...sshBaseArgs(t), t.host, bin, "--help"], { timeout: 15000 });
    return { ok: true, version: firstLine(stdout) };
  } catch (e) {
    const err = e as { stdout?: string };
    if (helpOk(err.stdout)) return { ok: true, version: firstLine(err.stdout!) };
    return { ok: false, version: null, reason: "not found on remote" };
  }
}

/** Does `repoRoot` look like a surogate source checkout (so `-e .` makes sense)? */
export function isSurogateSource(repoRoot: string): boolean {
  try {
    return (
      fs.existsSync(path.join(repoRoot, "pyproject.toml")) &&
      (fs.existsSync(path.join(repoRoot, "surogate")) || fs.existsSync(path.join(repoRoot, "src", "surogate")))
    );
  } catch {
    return false;
  }
}

const SCRIPT_PLAN: InstallPlan = {
  mode: "script",
  title: "script install",
  command: "curl -LsSf https://surogate.ai/install.sh | sh",
  prereqs: ["internet access", "a CUDA-capable NVIDIA GPU"],
  note: "creates a local .venv, detects your CUDA version, installs a matching wheel",
  postHint: "source .venv/bin/activate",
};

/** Local install options, recommended-first: a source checkout prefers `-e .`,
 *  everything else prefers the one-line script. */
export function localInstallPlans(repoRoot: string): InstallPlan[] {
  const source: InstallPlan = {
    mode: "source",
    title: "build from source",
    command: "uv pip install -e .",
    cwd: repoRoot,
    prereqs: ["uv", "CUDA toolkit + NCCL dev libraries", "a surogate source checkout"],
    note: "editable install from this surogate checkout (for contributors)",
  };
  return isSurogateSource(repoRoot) ? [source, SCRIPT_PLAN] : [SCRIPT_PLAN, source];
}

/** Install plan for a remote host (runs the script over SSH). */
export function remoteInstallPlan(): InstallPlan {
  return {
    ...SCRIPT_PLAN,
    title: "script install (remote)",
    note: "runs the install script on the remote host over SSH",
    postHint: "source .venv/bin/activate  (on the remote)",
  };
}

/** For cloud: surogate ships inside the Docker image, so the only thing missing
 *  locally is the dstack CLI that provisions the GPU. */
export function dstackInstallPlan(): InstallPlan {
  return {
    mode: "dstack",
    title: "install dstack",
    command: 'uv tool install "dstack[all]"',
    prereqs: ["a cloud account (RunPod / Lambda / Vast / AWS / GCP / Azure)", "configured dstack backends"],
    note: "surogate itself ships in the cloud image — you only need the dstack CLI here",
    postHint: "dstack server   (then add backends)",
  };
}

/** Modal runs surogate inside a Modal Sandbox on a rented GPU; locally you only
 *  need the `modal` client + an authenticated token. Pure-Python wheel, so the
 *  same command works on macOS / Windows / Linux. */
export function modalInstallPlan(): InstallPlan {
  return {
    mode: "modal",
    title: "install modal",
    command: "uv tool install modal",
    prereqs: ["a Modal account (modal.com)", "a workspace token"],
    note: "surogate runs in Modal's GPU container — you only need the modal client here",
    postHint: "modal token new   (or paste a token below to connect)",
  };
}

/** Shell command to install a Python CLI client, preferring uv (surogate's
 *  installer already provides it) and falling back to pip — same on mac/Win/Linux.
 *  Run via runShell (bash -lc), so the `command -v` test works. */
export function clientInstallCommand(tool: "modal" | "dstack"): string {
  const spec = tool === "dstack" ? '"dstack[all]"' : "modal";
  return `command -v uv >/dev/null 2>&1 && uv tool install ${spec} || python3 -m pip install --user ${spec}`;
}

/** Is the modal client installed locally? */
export async function modalAvailable(): Promise<boolean> {
  try {
    await execFileP("modal", ["--version"], { timeout: 8000 });
    return true;
  } catch {
    return false;
  }
}

/** Connect to Modal by writing the token to ~/.modal.toml via the client, with a
 *  live verification request. Returns { ok, reason? }. The secret is handed to
 *  the modal client (its own store) and never persisted by jackalope. */
export async function modalConnect(tokenId: string, tokenSecret: string): Promise<{ ok: boolean; reason?: string }> {
  try {
    await execFileP("modal", ["token", "set", "--token-id", tokenId, "--token-secret", tokenSecret, "--verify"], {
      timeout: 20000,
    });
    return { ok: true };
  } catch (e) {
    const err = e as { code?: string; stderr?: string; message?: string };
    if (err.code === "ENOENT") return { ok: false, reason: "modal client not installed — run: uv tool install modal" };
    return { ok: false, reason: (err.stderr || err.message || "token verification failed").split("\n")[0] };
  }
}

export interface RunHandle {
  done: Promise<{ ok: boolean; code: number | null }>;
  cancel: () => void;
}

// Spawn a process, streaming each whole stdout/stderr line to onLine. Carriage
// returns are dropped so progress bars don't smear the activity log.
function spawnStreaming(cmd: string, args: string[], onLine: (line: string) => void, cwd?: string): RunHandle {
  const child = spawn(cmd, args, { cwd });
  let buf = "";
  const pump = (chunk: Buffer) => {
    buf += chunk.toString();
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      onLine(stripAnsi(buf.slice(0, nl)));
      buf = buf.slice(nl + 1);
    }
  };
  child.stdout?.on("data", pump);
  child.stderr?.on("data", pump);
  const done = new Promise<{ ok: boolean; code: number | null }>((resolve) => {
    child.on("error", (err) => {
      onLine(`error: ${err.message}`);
      resolve({ ok: false, code: null });
    });
    child.on("close", (code) => {
      const tail = stripAnsi(buf).trim();
      if (tail) onLine(tail);
      resolve({ ok: code === 0, code });
    });
  });
  return { done, cancel: () => child.kill("SIGTERM") };
}

/** Run a shell command for the user, streaming output. A login shell resolves
 *  PATH-managed tools (uv, curl). */
export function runShell(command: string, onLine: (line: string) => void, cwd?: string): RunHandle {
  return spawnStreaming("bash", ["-lc", command], onLine, cwd && fs.existsSync(cwd) ? cwd : undefined);
}

/** Run a command on a remote host over SSH, streaming output. */
export function runShellRemote(t: SshTarget, command: string, onLine: (line: string) => void): RunHandle {
  return spawnStreaming("ssh", [...sshBaseArgs(t), t.host, command], onLine);
}

// Persisted onboarding state so a re-run can resume / skip what's done. The path
// is resolved at call-time (not module load) so it honors $HOME — handy in tests.
function stateFile(): string {
  return path.join(os.homedir(), ".surogate-watch", "onboarding.json");
}

export interface OnboardingState {
  completed: boolean;
  compute?: Compute;
  sshHost?: string;
  surogateOk?: boolean;
  ts: string;
}

export function loadOnboarding(): OnboardingState | null {
  try {
    return JSON.parse(fs.readFileSync(stateFile(), "utf8")) as OnboardingState;
  } catch {
    return null;
  }
}

export function saveOnboarding(s: OnboardingState): void {
  try {
    fs.mkdirSync(path.dirname(stateFile()), { recursive: true });
    fs.writeFileSync(stateFile(), JSON.stringify(s, null, 2));
  } catch {
    /* best effort */
  }
}
