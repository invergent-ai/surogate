// Modal compute: run surogate on a Modal GPU Sandbox, stream metrics back, and
// leave the trained model / checkpoints / logs on a Modal Volume (retrieved with
// `modal volume get`, see artifacts.ts).
//
// Modal Sandboxes are SDK-only (no CLI), so we spawn a tiny bundled Python driver
// via `uv run --with modal python` (auths from ~/.modal.toml, written by
// `modal token set` when the user connected). The driver runs surogate inside the
// sandbox, tails its metrics.jsonl to stdout, and we pipe the JSONL lines into a
// local run feed — exactly like the dstack path — so the whole dashboard works
// unchanged. On stop/exit the sandbox is terminated, which syncs the Volume.
import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pipeMetrics } from "./feed-pipe.ts";
import { newRunFeedPath, runArtifacts, writeRunMeta } from "./runs.ts";
import { modalAvailable } from "./setup.ts";

/** Is the modal client importable (installed + authed)? (alias of setup's check) */
export const modalReady = modalAvailable;

export interface ModalConfig {
  gpu: string; // "H100", "A100-80GB", "L4", …
  count: number;
  image: string; // a surogate-ready image (training needs surogate installed)
}

export const MODAL_DEFAULT_IMAGE = "ghcr.io/invergent-ai/surogate:latest-cu128";
// GPU types Modal offers (string passed straight to `gpu=`).
export const MODAL_GPUS = ["T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100", "H200", "B200"];

// The Python driver, written to a temp file and run with `uv run --with modal`.
// Kept tiny + dependency-free (only the modal SDK). Reads everything from env.
export const MODAL_DRIVER = String.raw`
import base64, os, signal, sys, time
import modal

gpu_t = os.environ.get("JK_GPU", "H100")
count = os.environ.get("JK_COUNT", "1")
mode = os.environ.get("JK_MODE", "sft")
image_uri = os.environ.get("JK_IMAGE", "ghcr.io/invergent-ai/surogate:latest-cu128")
volume = os.environ.get("JK_VOLUME", "jackalope-run")
config = base64.b64decode(os.environ["JK_CONFIG_B64"]).decode()
gpu = gpu_t if count in ("", "1") else gpu_t + ":" + count

app = modal.App.lookup("jackalope", create_if_missing=True)
vol = modal.Volume.from_name(volume, create_if_missing=True)
image = modal.Image.from_registry(
    image_uri, add_python="3.12",
    # The surogate image activates a venv (/root/.venv) that has no pip and sits
    # first on PATH, shadowing the standalone python add_python drops into
    # /usr/local — so Modal's client bootstrap (python -m pip ...) hits the venv
    # python and dies with "No module named pip". Prepend /usr/local/bin before
    # that step runs; surogate's console scripts keep their own venv shebang.
    setup_dockerfile_commands=["ENV PATH=/usr/local/bin:$PATH"],
).entrypoint([])  # the image's ENTRYPOINT is surogate; clear it so Modal's
# sandbox keep-alive sleep runs directly instead of as "surogate sleep". We
# launch training ourselves via sb.exec below.

print("JK: creating " + gpu + " sandbox (image " + image_uri + ")", file=sys.stderr, flush=True)
sb = modal.Sandbox.create(app=app, image=image, gpu=gpu, timeout=24 * 3600,
                          volumes={"/outputs": vol}, workdir="/root")
# Persist the sandbox id next to the run so it can be terminated later from a
# fresh process — the local driver dying must never strand a paid GPU sandbox.
sid = sb.object_id or ""
try:
    open("sandbox.id", "w").write(sid)
except Exception:
    pass
print("JK: sandbox " + (sid or "?"), file=sys.stderr, flush=True)

def _cleanup(*_):
    try:
        sb.terminate()
    finally:
        os._exit(1)
signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)

try:
    cfg_b64 = base64.b64encode(config.encode()).decode()
    sb.exec("bash", "-lc", "mkdir -p /root /outputs && echo " + cfg_b64 + " | base64 -d > /root/config.yaml").wait()
    # Run surogate writing metrics + the full log to the Volume, and stream BOTH
    # back to our stdout: metric JSONL lines drive the live chart, train.log lines
    # surface real progress and any crash (otherwise a fast failure looks silent).
    # A trailing "JK_EXIT <code>" reports the trainer's exit status.
    run = ("cd /root && touch /outputs/metrics.jsonl /outputs/train.log && "
           "(SUROGATE_METRICS_PATH=/outputs/metrics.jsonl surogate " + mode + " config.yaml > /outputs/train.log 2>&1; "
           "echo $? > /outputs/DONE) & "
           "tail -n +1 -F /outputs/metrics.jsonl & TM=$! ; "
           "tail -n +1 -F /outputs/train.log & TL=$! ; "
           "while [ ! -f /outputs/DONE ]; do sleep 2; done; sleep 2; "
           "kill $TM $TL 2>/dev/null; echo JK_EXIT $(cat /outputs/DONE)")
    p = sb.exec("bash", "-lc", run)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    p.wait()
finally:
    sb.terminate()
print("JK_DONE", flush=True)
`;

interface ModalHandle {
  child: ChildProcess;
  name: string;
  feed: string;
}
const streams = new Set<ModalHandle>();
process.on("exit", () => killModalStreams());
export function killModalStreams(): void {
  for (const s of streams) {
    try {
      s.child.kill("SIGTERM");
    } catch {
      /* ignore */
    }
  }
  streams.clear();
}

export type ModalLaunch = { ok: true; feed: string; name: string; volume: string } | { ok: false; reason: string };

/** Launch a surogate run on a Modal GPU sandbox; streams metrics into a local
 *  feed and leaves artifacts on the named Volume. */
export function launchModalRun(cfg: ModalConfig, configText: string, label: string): ModalLaunch {
  const feed = newRunFeedPath(`modal-${label}`, Date.now());
  const art = runArtifacts(feed);
  const runId = path.basename(art.dir);
  const volume = `jackalope-${runId}`.replace(/[^a-z0-9-]/gi, "-").toLowerCase().slice(0, 50);
  const driverPath = path.join(art.dir, "modal_driver.py");
  // surogate must write its outputs onto the mounted Volume so we can retrieve
  // them later — pin output_dir to /outputs/model (YAML last-key-wins).
  const onVolume = configText.replace(/\s*$/, "") + "\noutput_dir: /outputs/model\n";
  try {
    fs.writeFileSync(art.configPath, onVolume);
    fs.writeFileSync(driverPath, MODAL_DRIVER);
  } catch (e) {
    return { ok: false, reason: (e as Error).message };
  }
  const env = {
    ...process.env,
    JK_GPU: cfg.gpu,
    JK_COUNT: String(cfg.count),
    JK_MODE: "sft",
    JK_IMAGE: cfg.image || MODAL_DEFAULT_IMAGE,
    JK_VOLUME: volume,
    JK_CONFIG_B64: Buffer.from(onVolume).toString("base64"),
  };
  // `uv run --with modal` guarantees the modal SDK is importable regardless of how
  // the user installed the CLI; it auths from ~/.modal.toml.
  const child = spawn("uv", ["run", "--with", "modal", "python", driverPath], { cwd: art.dir, env, stdio: ["ignore", "pipe", "pipe"] });
  child.on("error", () => {});
  if (!child.pid) return { ok: false, reason: 'could not start the Modal driver — is uv installed? (the Modal client needs it)' };
  pipeMetrics(child, feed, art.logPath);
  streams.add({ child, name: volume, feed });
  writeRunMeta(feed, {
    mode: "sft",
    startedAt: Date.now(),
    label: `modal-${label}`,
    remote: { kind: "modal", host: `Modal · ${cfg.gpu}:${cfg.count}`, session: volume, dir: "(modal volume)" },
  });
  return { ok: true, feed, name: volume, volume };
}

/** Is the Modal driver for this run still running? (false once it exits — the run
 *  finished or the sandbox failed to start.) */
export function modalStreamAlive(name: string): boolean {
  return [...streams].some((s) => s.name === name && s.child.exitCode === null && !s.child.killed);
}

/** Stop a Modal run for the given feed. Terminates the cloud sandbox by its
 *  persisted id (works even after the local driver has exited — the launcher
 *  process dying must not strand a paid GPU), and stops local streaming. */
export function stopModalRun(feed: string): boolean {
  let stopped = false;
  // 1) terminate the sandbox itself, by id, from a fresh `uv run` — independent
  //    of whether this process still holds the driver handle.
  try {
    const sid = fs.readFileSync(path.join(runArtifacts(feed).dir, "sandbox.id"), "utf8").trim();
    if (sid) {
      const py = `import modal; modal.Sandbox.from_id(${JSON.stringify(sid)}).terminate()`;
      const killer = spawn("uv", ["run", "--with", "modal", "python", "-c", py], { stdio: "ignore", detached: true });
      killer.on("error", () => {});
      killer.unref();
      stopped = true;
    }
  } catch {
    /* no sandbox id yet (sandbox not created) — fall through to killing the driver */
  }
  // 2) stop the local streaming driver so the feed stops updating.
  for (const s of [...streams]) {
    if (s.feed === feed) {
      try {
        s.child.kill("SIGTERM");
        stopped = true;
      } catch {
        /* ignore */
      }
      streams.delete(s);
    }
  }
  return stopped;
}
