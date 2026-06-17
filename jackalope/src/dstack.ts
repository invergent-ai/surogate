// Cloud GPUs via dstack (https://dstack.ai) — one OSS CLI that provisions
// AWS/GCP/Azure + Lambda/Vast/RunPod/CUDO (and your own machines as SSH fleets),
// so this single backend covers most managed providers.
//
// Transport: dstack has no "download a file from the run" command, so we stream
// metrics over the task's STDOUT. The generated task writes the surogate config
// on the instance from a heredoc (no file-upload needed), runs training to a log,
// and `tail -F metrics.jsonl` to stdout. `dstack apply` runs attached and streams
// that stdout locally; we keep only valid JSONL lines and append them to a local
// run feed, so the rest of the dashboard works unchanged.

import { type ChildProcess, execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pipeMetrics } from "./feed-pipe.ts";
import { newRunFeedPath, runArtifacts, writeRunMeta } from "./runs.ts";

const execFileP = promisify(execFile);

// ── backend credentials ───────────────────────────────────────────────────────
// dstack reads backend creds from ~/.dstack/server/config.yml. We write the chosen
// backend there so `dstack apply` can provision it. The creds go to dstack's own
// config (never into jackalope's providers.json). The YAML shape is small and
// fixed, so we emit it directly rather than pull in a YAML dependency.
const DSTACK_CONFIG = (): string => path.join(os.homedir(), ".dstack", "server", "config.yml");

// The credential fields each backend needs — drives the UI form and the config
// writer. `file` fields take either a path on disk OR the contents pasted
// directly (JSON key / PEM) — whichever the user gives gets inlined; `secret`
// fields are masked in the UI. Grounded in dstack's backend schema.
export interface BackendField {
  key: string;
  label: string;
  secret?: boolean;
  file?: boolean;
}
export const DSTACK_BACKEND_FIELDS: Record<string, BackendField[]> = {
  runpod: [{ key: "api_key", label: "API key", secret: true }],
  lambda: [{ key: "api_key", label: "API key", secret: true }],
  vastai: [{ key: "api_key", label: "API key", secret: true }],
  aws: [
    { key: "access_key", label: "access key id" },
    { key: "secret_key", label: "secret access key", secret: true },
  ],
  gcp: [
    { key: "project_id", label: "project id" },
    { key: "data_file", label: "service-account JSON (path or paste)", file: true },
  ],
  azure: [
    { key: "tenant_id", label: "tenant id" },
    { key: "subscription_id", label: "subscription id" },
    { key: "client_id", label: "client id" },
    { key: "client_secret", label: "client secret", secret: true },
  ],
  nebius: [
    { key: "service_account_id", label: "service account id" },
    { key: "public_key_id", label: "public key id" },
    { key: "private_key_file", label: "private key (path or paste)", file: true },
  ],
  oci: [
    { key: "user", label: "user OCID" },
    { key: "tenancy", label: "tenancy OCID" },
    { key: "region", label: "region" },
    { key: "fingerprint", label: "fingerprint" },
    { key: "key_file", label: "private key (path or paste)", file: true },
  ],
};

// Indent a multi-line file's contents as a YAML block scalar body.
function block(content: string, indent: string): string[] {
  return content.replace(/\r/g, "").replace(/\s+$/, "").split("\n").map((l) => indent + l);
}

/** The `creds:` block (with any backend-level fields) for a backend. Throws with
 *  a readable reason if a required field is missing or a key file can't be read. */
function credsYaml(backend: string, f: Record<string, string>): string[] {
  const v = (k: string) => (f[k] ?? "").trim();
  // YAML-quote a scalar value (a JSON string is a valid YAML double-quoted scalar),
  // so secrets containing ':' '#' or other meta-chars can't corrupt the document.
  const q = (k: string) => JSON.stringify(v(k));
  const fields = DSTACK_BACKEND_FIELDS[backend];
  if (!fields) throw new Error(`unknown backend ${backend}`);
  for (const fld of fields) if (!v(fld.key)) throw new Error(`missing ${fld.label}`);
  // Accept EITHER a path to a key/JSON file OR the contents pasted directly: if
  // the value resolves to a real file, inline it; otherwise treat the value as
  // the literal key/JSON the user pasted (Nebius/OCI PEM keys are short and
  // commonly pasted rather than saved to disk first).
  const fileText = (k: string): string => {
    const raw = v(k);
    let p = raw;
    if (p === "~" || p.startsWith("~/")) p = path.join(os.homedir(), p.slice(1)); // expand ~
    try {
      if (fs.statSync(p).isFile()) return fs.readFileSync(p, "utf8");
    } catch {
      /* not a readable file — decide below whether it was meant as a path */
    }
    // A single-line value that clearly names a filesystem path (~, /, ./, ../) but
    // doesn't resolve is a mistyped path, not pasted key material — fail clearly
    // rather than writing the path string itself into the backend config. (Anchored
    // to the start so base64 key bodies, which contain '/', aren't mistaken for paths.)
    if (!raw.includes("\n") && /^(~|\.{0,2}\/)/.test(raw)) throw new Error(`could not read ${k}: ${raw}`);
    return raw;
  };
  switch (backend) {
    case "runpod":
    case "lambda":
    case "vastai":
      return [`  - type: ${backend}`, `    creds:`, `      type: api_key`, `      api_key: ${q("api_key")}`];
    case "aws":
      return [`  - type: aws`, `    creds:`, `      type: access_key`, `      access_key: ${q("access_key")}`, `      secret_key: ${q("secret_key")}`];
    case "gcp":
      return [`  - type: gcp`, `    project_id: ${q("project_id")}`, `    creds:`, `      type: service_account`, `      filename: ""`, `      data: |`, ...block(fileText("data_file"), "        ")];
    case "azure":
      return [`  - type: azure`, `    tenant_id: ${q("tenant_id")}`, `    subscription_id: ${q("subscription_id")}`, `    creds:`, `      type: client`, `      client_id: ${q("client_id")}`, `      client_secret: ${q("client_secret")}`];
    case "nebius":
      return [`  - type: nebius`, `    creds:`, `      type: service_account`, `      service_account_id: ${q("service_account_id")}`, `      public_key_id: ${q("public_key_id")}`, `      private_key_content: |`, ...block(fileText("private_key_file"), "        ")];
    case "oci":
      return [`  - type: oci`, `    region: ${q("region")}`, `    creds:`, `      type: client`, `      user: ${q("user")}`, `      tenancy: ${q("tenancy")}`, `      fingerprint: ${q("fingerprint")}`, `      key_content: |`, ...block(fileText("key_file"), "        ")];
    default:
      throw new Error(`unsupported backend ${backend}`);
  }
}

/** Write a single active backend into dstack's server config so `dstack apply`
 *  can use it. Returns { ok, reason? }. */
export function configureDstackBackend(backend: string, fields: Record<string, string>): { ok: boolean; reason?: string } {
  try {
    const yaml = ["projects:", "- name: main", "  backends:", ...credsYaml(backend, fields), ""].join("\n");
    const cfg = DSTACK_CONFIG();
    fs.mkdirSync(path.dirname(cfg), { recursive: true });
    // This sets a single active backend; back up any existing config first so a
    // user's hand-managed multi-backend setup isn't lost irrecoverably.
    if (fs.existsSync(cfg)) {
      try {
        fs.copyFileSync(cfg, cfg + ".bak");
      } catch {
        /* best effort */
      }
    }
    fs.writeFileSync(cfg, yaml, { mode: 0o600 });
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: `${backend}: ${(e as Error).message}` };
  }
}

export interface DstackConfig {
  gpu: string; // GPU type, e.g. "H100", "A100", "L4"
  count: number; // number of GPUs
  image: string; // a surogate-ready Docker image (training needs surogate installed)
  backend?: string; // pin a backend (runpod/lambda/vast/aws/gcp/azure) — empty = cheapest
  region?: string;
}

/** Is the dstack CLI installed + on PATH? */
export async function dstackAvailable(): Promise<boolean> {
  try {
    await execFileP("dstack", ["--version"], { timeout: 8000 });
    return true;
  } catch {
    return false;
  }
}

/** Generate a dstack task config. The surogate config is embedded via a heredoc
 *  so nothing has to be uploaded; metrics stream out on stdout. Pure → testable. */
export function dstackTaskYaml(name: string, cfg: DstackConfig, configText: string, surogateBin = "surogate"): string {
  const indented = configText.replace(/\n/g, "\n    ").replace(/\s+$/, "");
  const lines = [
    "type: task",
    `name: ${name}`,
    `image: ${cfg.image}`,
  ];
  if (cfg.backend) lines.push(`backends: [${cfg.backend}]`);
  if (cfg.region) lines.push(`regions: [${cfg.region}]`);
  // one literal block = one shell session: write the config, run training in the
  // background to a log, and tail metrics to stdout (dstack streams it back).
  lines.push(
    "commands:",
    "  - |",
    "    cat > config.yaml <<'JACKALOPE_YAML'",
    `    ${indented}`,
    "    JACKALOPE_YAML",
    `    SUROGATE_METRICS_PATH=metrics.jsonl ${surogateBin} sft config.yaml > train.log 2>&1 &`,
    "    sleep 3",
    "    tail -n +1 -F metrics.jsonl",
    "resources:",
    `  gpu: "${cfg.gpu}:${cfg.count}"`,
  );
  return lines.join("\n") + "\n";
}

// Active dstack streams, so we can stop the cloud run + kill the local apply.
interface DstackHandle {
  child: ChildProcess;
  name: string;
}
const streams = new Set<DstackHandle>();
export function killDstackStreams(): void {
  for (const s of streams) {
    try {
      s.child.kill();
    } catch {
      /* ignore */
    }
  }
  streams.clear();
}

export type DstackLaunch = { ok: true; feed: string; name: string } | { ok: false; reason: string };

/** Provision + run on cloud GPUs via `dstack apply`, streaming metrics back. */
export function launchDstackRun(cfg: DstackConfig, configText: string, label: string, surogateBin = "surogate"): DstackLaunch {
  const feed = newRunFeedPath(`dstack-${label}`, Date.now());
  const art = runArtifacts(feed);
  const name = `sur-${path.basename(art.dir)}`.replace(/[^a-z0-9-]/gi, "-").toLowerCase().slice(0, 40);
  const taskFile = path.join(art.dir, "task.dstack.yml");
  try {
    fs.writeFileSync(art.configPath, configText);
    fs.writeFileSync(taskFile, dstackTaskYaml(name, cfg, configText, surogateBin));
  } catch (e) {
    return { ok: false, reason: (e as Error).message };
  }
  const child = spawn("dstack", ["apply", "-y", "--name", name, "-f", taskFile], {
    cwd: art.dir,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.on("error", () => {});
  if (!child.pid) return { ok: false, reason: 'could not run "dstack" — is the CLI installed? (pip install dstack)' };
  pipeMetrics(child, feed, art.logPath);
  streams.add({ child, name });
  writeRunMeta(feed, {
    mode: "sft",
    startedAt: Date.now(),
    label: `dstack-${label}`,
    remote: {
      kind: "dstack",
      host: `dstack · ${cfg.backend || "cheapest"} · ${cfg.gpu}:${cfg.count}`,
      session: name,
      dir: "(cloud)",
    },
  });
  return { ok: true, feed, name };
}

/** Stop a cloud run by name (and kill the local apply stream). */
export function stopDstackRun(name: string): boolean {
  for (const s of [...streams]) {
    if (s.name === name) {
      try {
        s.child.kill();
      } catch {
        /* ignore */
      }
      streams.delete(s);
    }
  }
  try {
    spawn("dstack", ["stop", "-y", name], { stdio: "ignore", detached: true }).unref();
    return true;
  } catch {
    return false;
  }
}

/** Is the `dstack apply` stream for this run still running? (false once it exits,
 *  i.e. the cloud run finished or failed to provision.) */
export function dstackStreamAlive(name: string): boolean {
  return [...streams].some((s) => s.name === name && s.child.exitCode === null && !s.child.killed);
}

process.on("exit", killDstackStreams);
