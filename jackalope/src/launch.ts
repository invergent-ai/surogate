// Launch helpers: GPU discovery (nvidia-smi), config-build, spawn, and example
// train-file discovery. No Ink here — pure logic, unit-testable.

import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { procStartTime } from "./controls.ts";
import { newRunFeedPath, type RunInfo, runArtifacts, writeRunMeta } from "./runs.ts";

/** Outcome of a launch attempt: a live PID, or a reason we can show the user. */
export type SpawnResult = { ok: true; pid: number } | { ok: false; reason: string };

export interface Gpu {
  id: number;
  name: string;
  memMB: number | null;
  memUsedMB: number | null;
  util: number | null;
  busy: boolean; // already running something (util>5% or >1.5GB used)
  sm: number | null; // compute capability ×10 (e.g. 8.9 → 89, 9.0 → 90, 12.0 → 120)
}

export const RECIPES = ["bf16", "fp8-hybrid", "nvfp4", "nvfp4_quartet"] as const;
export type Recipe = (typeof RECIPES)[number];

export const DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

export const OPTIMIZERS = ["adamw_8bit", "adamw", "normuon"] as const;
export const LR_SCHEDULERS = ["linear", "cosine", "constant", "wsd"] as const;

export interface LaunchFields {
  // data + precision
  model: string;
  datasetPath: string;
  datasetType: string;
  recipe: Recipe;
  // training
  maxSteps: string;
  learningRate: string;
  perDeviceBatch: string;
  gradAccum: string;
  sequenceLen: string;
  evalSteps: string;
  warmupRatio: string;
  gradientCheckpointing: boolean;
  // optimizer
  optimizer: string;
  lrScheduler: string;
  weightDecay: string;
  maxGradNorm: string;
  // LoRA
  lora: boolean;
  loraRank: string;
  loraAlpha: string;
  // checkpoints + output
  saveSteps: string;
  saveTotalLimit: string;
  resumeFromCheckpoint: boolean;
  mergeAdapter: boolean;
  outputDir: string;
}

export const DEFAULT_FIELDS: LaunchFields = {
  // model + dataset start empty — the user picks them in Models/Datasets (or types
  // a path in Parameters). No surprise preselection.
  model: "",
  datasetPath: "",
  datasetType: "auto",
  recipe: "fp8-hybrid",
  maxSteps: "200",
  learningRate: "2e-4",
  perDeviceBatch: "2",
  gradAccum: "4",
  sequenceLen: "2048",
  evalSteps: "25",
  warmupRatio: "0.15",
  gradientCheckpointing: true,
  optimizer: "adamw_8bit",
  lrScheduler: "linear",
  weightDecay: "0",
  maxGradNorm: "1.0",
  lora: true,
  loraRank: "16",
  loraAlpha: "32",
  saveSteps: "50",
  saveTotalLimit: "5",
  resumeFromCheckpoint: false,
  mergeAdapter: false,
  outputDir: "./watch-out",
};

/** Enumerate GPUs via nvidia-smi (with live util/memory so we can flag busy
 *  devices); falls back to a single generic entry. */
// nvidia-smi query shared by local + remote (SSH) detection. compute_cap lets us
// tell which precision recipes the GPU can run (fp8 = SM89+, nvfp4 = SM100+).
export const NVIDIA_SMI_QUERY = [
  "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap",
  "--format=csv,noheader,nounits",
];

const numOrNull = (s: string | undefined): number | null =>
  s && s !== "[Not Supported]" && s !== "[N/A]" && Number.isFinite(Number(s)) ? Number(s) : null;

/** Parse `nvidia-smi --query-gpu=...,--format=csv,noheader,nounits` output. */
export function parseGpuCsv(out: string): Gpu[] {
  const gpus: Gpu[] = [];
  for (const line of out.trim().split("\n")) {
    if (!line.trim()) continue;
    const [idx, name, mem, memUsed, util, cc] = line.split(",").map((s) => s.trim());
    if (idx === undefined || idx === "" || !Number.isFinite(Number(idx))) continue;
    const memUsedMB = numOrNull(memUsed);
    const u = numOrNull(util);
    const busy = (u !== null && u > 5) || (memUsedMB !== null && memUsedMB > 1500);
    const capNum = numOrNull(cc); // "8.9" → 8.9
    const sm = capNum !== null ? Math.round(capNum * 10) : null; // 8.9 → 89, 12.0 → 120
    gpus.push({ id: Number(idx), name: name ?? `GPU ${idx}`, memMB: numOrNull(mem), memUsedMB, util: u, busy, sm });
  }
  return gpus;
}

/** Local GPUs via nvidia-smi. Returns [] when there's no NVIDIA driver (e.g. a
 *  Mac/Windows laptop) — the caller then offers remote compute. */
// nvidia-smi is a blocking spawn (~100-300ms); cache it briefly so re-mounting a
// page or flipping tabs doesn't re-run it and stutter the UI. The static device
// list barely changes; live util comes from the metrics feed, not this.
let gpuCache: { at: number; gpus: Gpu[] } | null = null;
export function discoverGpus(): Gpu[] {
  if (gpuCache && Date.now() - gpuCache.at < 2500) return gpuCache.gpus;
  let gpus: Gpu[] = [];
  try {
    gpus = parseGpuCsv(execFileSync("nvidia-smi", NVIDIA_SMI_QUERY, { encoding: "utf8", timeout: 5000 }));
  } catch {
    gpus = []; // no NVIDIA driver here
  }
  gpuCache = { at: Date.now(), gpus };
  return gpus;
}

// GPUs the user earmarked for training in the GPUs tab; Launch pre-selects these.
const GPU_SEL_FILE = path.join(os.homedir(), ".surogate-watch", "gpu-selection.json");

export function loadGpuSelection(): number[] {
  try {
    const v = JSON.parse(fs.readFileSync(GPU_SEL_FILE, "utf8"));
    return Array.isArray(v) ? v.filter((x): x is number => typeof x === "number") : [];
  } catch {
    return [];
  }
}

export function saveGpuSelection(ids: number[]): void {
  try {
    fs.mkdirSync(path.dirname(GPU_SEL_FILE), { recursive: true });
    fs.writeFileSync(GPU_SEL_FILE, JSON.stringify([...ids].sort((a, b) => a - b)));
  } catch {
    /* best effort */
  }
}

const DEFAULT_METRICS_PATH = "/tmp/surogate_metrics.jsonl";

/** Resolve the surogate metrics feed path the same way the framework does
 *  (config wins in surogate; here we only see the env + default). */
export function resolveFeedPath(explicit?: string): string {
  return explicit || process.env["SUROGATE_METRICS_PATH"] || DEFAULT_METRICS_PATH;
}

/** Pick which feed to watch when none is given on the command line. Honors an
 *  explicit arg / $SUROGATE_METRICS_PATH; otherwise picks the most-recently
 *  written feed among the default path and the per-run feeds, so launching
 *  `jackalope` just finds an active training. Falls back to the default (the
 *  tailer waits for it to appear), covering "run jackalope, then start training". */
export function discoverFeedPath(explicit?: string): string {
  if (explicit) return explicit;
  if (process.env["SUROGATE_METRICS_PATH"]) return process.env["SUROGATE_METRICS_PATH"]!;
  const runsDir = path.join(os.homedir(), ".surogate-watch", "runs");
  const candidates = [DEFAULT_METRICS_PATH];
  try {
    for (const e of fs.readdirSync(runsDir, { withFileTypes: true })) {
      if (e.isDirectory()) candidates.push(path.join(runsDir, e.name, "metrics.jsonl"));
      else if (e.name.endsWith(".jsonl")) candidates.push(path.join(runsDir, e.name)); // legacy flat feeds
    }
  } catch {
    /* no runs dir yet */
  }
  let best: string | null = null;
  let bestMtime = -1;
  for (const c of candidates) {
    try {
      const st = fs.statSync(c);
      if (st.isFile() && st.size > 0 && st.mtimeMs > bestMtime) {
        bestMtime = st.mtimeMs;
        best = c;
      }
    } catch {
      /* not present */
    }
  }
  return best ?? DEFAULT_METRICS_PATH;
}

/** Build the `surogate sft` invocation string (for preview). */
export function buildCommand(gpuIds: number[], configPath: string, surogateBin = "surogate"): string {
  const prefix = gpuIds.length ? `CUDA_VISIBLE_DEVICES=${gpuIds.join(",")} ` : "";
  return `${prefix}${surogateBin} sft ${configPath}`;
}

/** Append override lines after a base YAML body. Appended keys win (YAML
 *  last-key wins). Values must be simple scalars (no unquoted colons/spaces). */
function appendYamlOverlay(base: string, lines: string[]): string {
  return base.replace(/\s*$/, "") + "\n" + lines.join("\n") + "\n";
}

/** Take an existing SFT YAML and append a monitoring overlay (gpus + the surogate
 *  metrics feed) so the dashboard can follow it; the user's file is otherwise run
 *  exactly as written. */
export function overlayExistingSft(yamlPath: string, nGpus: number, metricsPath: string): string {
  let base = "";
  try {
    base = fs.readFileSync(yamlPath, "utf8");
  } catch {
    /* missing file → empty base; surogate will error clearly */
  }
  return appendYamlOverlay(base, [
    "# --- jackalope monitoring overlay (overrides the above) ---",
    `gpus: ${Math.max(1, nGpus)}`,
    "report_to: [surogate]",
    `surogate_metrics_path: "${metricsPath}"`,
  ]);
}

/** Resume a previous run: take its saved config and append an overlay that turns
 *  on resume_from_checkpoint and points the feed at a fresh path. The config's
 *  output_dir already points at the run's checkpoints, so surogate continues from
 *  the latest one. */
export function buildResumeYaml(baseConfig: string, newMetricsPath: string): string {
  return appendYamlOverlay(baseConfig, [
    "# --- jackalope resume overlay ---",
    "resume_from_checkpoint: true",
    "report_to: [surogate]",
    `surogate_metrics_path: "${newMetricsPath}"`,
  ]);
}

/** Build a YAML SFT config from the launch fields (matches surogate's schema). */
export function buildConfigYaml(f: LaunchFields, nGpus: number, metricsPath?: string): string {
  const lines = [
    `model: "${f.model}"`,
    `output_dir: "${f.outputDir}"`,
    `gpus: ${Math.max(1, nGpus)}`,
    `recipe: ${f.recipe}`,
    `per_device_train_batch_size: ${f.perDeviceBatch}`,
    `gradient_accumulation_steps: ${f.gradAccum}`,
    `sequence_len: ${f.sequenceLen}`,
    `sample_packing: true`,
    `max_steps: ${f.maxSteps}`,
    `eval_steps: ${f.evalSteps}`,
    `learning_rate: ${f.learningRate}`,
    `warmup_ratio: ${f.warmupRatio}`,
    `optimizer: ${f.optimizer}`,
    `lr_scheduler_type: ${f.lrScheduler}`,
    `weight_decay: ${f.weightDecay}`,
    `max_grad_norm: ${f.maxGradNorm}`,
    `recompute: ${f.gradientCheckpointing ? "true" : "false"}`,
    `save_steps: ${f.saveSteps}`,
    `save_total_limit: ${f.saveTotalLimit}`,
    `resume_from_checkpoint: ${f.resumeFromCheckpoint ? "true" : "false"}`,
    `lora: ${f.lora ? "true" : "false"}`,
  ];
  if (f.lora) {
    lines.push(`lora_rank: ${f.loraRank}`, `lora_alpha: ${f.loraAlpha}`);
    lines.push("lora_target_modules:");
    for (const m of DEFAULT_TARGET_MODULES) lines.push(`  - ${m}`);
  }
  if (f.mergeAdapter) lines.push("merge_adapter: true");
  lines.push(
    "# live monitoring",
    "logging_steps: 1",
    "log_gpu_util: 5",
    "report_to: [surogate]",
  );
  if (metricsPath) lines.push(`surogate_metrics_path: "${metricsPath}"`);
  lines.push("datasets:", `  - path: "${f.datasetPath}"`, `    type: ${f.datasetType}`);
  return lines.join("\n") + "\n";
}

// ---------------- GRPO (split-GPU RL) ----------------

export interface GrpoConfigs {
  train: string;
  infer: string;
  orch: string;
  judge?: string; // RULER only
}

// The default reward environment. `markdown-table-qa` ships in the surogate repo
// (environments/markdown-table-qa) as a single-file verifiers env whose only dep
// is `verifiers` (already in the venv). It has no "/" in its id, so surogate never
// tries to `prime env install` it — it must be importable, which we guarantee by
// pointing env.path at the repo dir (grpo_orch.py inserts it onto sys.path).
const RL_ENV_ID = "markdown-table-qa";
// Disjoint vLLM ports: the student rollout server and the RULER judge must not
// collide (surogate validates judge.port != rollout.port), and the orchestrator's
// client/judge base_urls must point at the matching port (no auto-reconcile).
const STUDENT_PORT = 8007;
const JUDGE_PORT = 8001;

/** Locate the shipped verifiers environment so the orchestrator can import it via
 *  sys.path (env.path) with no hub install. Returns null if the repo isn't here. */
function findRlEnvPath(repoRoot: string): string | null {
  const d = path.join(repoRoot, "environments", RL_ENV_ID);
  return fs.existsSync(path.join(d, `${RL_ENV_ID.replace(/-/g, "_")}.py`)) ? d : null;
}

const MANAGED_HEADER = "# generated by jackalope — regenerated on each launch; edit a copy if you need custom configs\n";

function grpoTrainYaml(): string {
  return `${MANAGED_HEADER}model: Qwen/Qwen3-0.6B
output_dir: ./outputs
per_device_train_batch_size: 1
sequence_len: 2048
max_steps: 20
logging_steps: 1
learning_rate: 1e-4
lr_scheduler_type: constant
warmup_steps: 0
max_grad_norm: 1.0
weight_decay: 0.01
optimizer: adamw
recipe: bf16
lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
save_steps: 100
report_to: [surogate]
`;
}

function grpoInferYaml(port: number): string {
  return `${MANAGED_HEADER}model: Qwen/Qwen3-0.6B
enable_lora: true
max_lora_rank: 32
max_model_len: 2048
gpu_memory_utilization: 0.3
port: ${port}
`;
}

// The reward environment block, with env.path so it imports without a hub install.
function envBlock(envPath: string | null): string {
  return envPath ? `env:\n  - id: ${RL_ENV_ID}\n    path: ${envPath}\n` : `env:\n  - id: ${RL_ENV_ID}\n`;
}

function grpoOrchYaml(envPath: string | null): string {
  return `${MANAGED_HEADER}model:
  name: Qwen/Qwen3-0.6B
  lora_adapter: default
  lora_rank: 16
  lora_alpha: 32
${envBlock(envPath)}batch_size: 128
rollouts_per_example: 16
seq_len: 2048
max_steps: 20
use_token_client: false
sampling:
  max_tokens: 128
client:
  base_url:
    - http://localhost:${STUDENT_PORT}/v1
dump_metrics: true
`;
}

// RULER: a frozen judge vLLM (Qwen3-1.7B at :8001) scores groups; mode=replace
// uses the judge as the sole reward (needs rollouts_per_example >= 2).
function rulerOrchYaml(envPath: string | null): string {
  return `${MANAGED_HEADER}model:
  name: Qwen/Qwen3-0.6B
  lora_adapter: default
${envBlock(envPath)}client:
  base_url:
    - http://localhost:${STUDENT_PORT}/v1
batch_size: 16
rollouts_per_example: 4
seq_len: 2048
max_steps: 20
use_token_client: false
sampling:
  max_tokens: 256
verification:
  enabled: true
ruler:
  enabled: true
  mode: replace
  judge_model: "Qwen/Qwen3-1.7B"
  judge:
    base_url:
      - http://localhost:${JUDGE_PORT}/v1
    api_key_var: RULER_JUDGE_API_KEY
    timeout: 90.0
  weight: 1.0
  max_concurrent_judges: 8
  sampling:
    temperature: 0.0
    max_completion_tokens: 4096
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
  swallow_exceptions: true
dump_metrics: true
`;
}

function judgeYaml(port: number): string {
  return `${MANAGED_HEADER}model: Qwen/Qwen3-1.7B
enable_lora: false
max_model_len: 4096
gpu_memory_utilization: 0.3
port: ${port}
`;
}

function writeManaged(p: string, content: string): void {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, content);
}

/** Generate self-contained, runnable RL configs into ~/.surogate-watch/configs/
 *  <mode>/. Regenerated each call so port/env-path fixes always propagate (these
 *  are jackalope-managed scaffolds — markdown-table-qa reward env, disjoint vLLM
 *  ports, judge for RULER). The trainer/vLLM/judge GPU counts are injected later,
 *  per-launch, by spawnGrpo (they depend on the user's GPU selection). */
export function ensureRlConfigs(
  mode: "grpo" | "ruler",
  repoRoot: string,
  configRoot = path.join(os.homedir(), ".surogate-watch", "configs"),
): GrpoConfigs {
  const dir = path.join(configRoot, mode);
  const out: GrpoConfigs = {
    train: path.join(dir, "train.yaml"),
    infer: path.join(dir, "infer.yaml"),
    orch: path.join(dir, "orch.yaml"),
    ...(mode === "ruler" ? { judge: path.join(dir, "judge.yaml") } : {}),
  };
  const envPath = findRlEnvPath(repoRoot);
  writeManaged(out.train, grpoTrainYaml());
  writeManaged(out.infer, grpoInferYaml(STUDENT_PORT));
  writeManaged(out.orch, mode === "ruler" ? rulerOrchYaml(envPath) : grpoOrchYaml(envPath));
  if (out.judge) writeManaged(out.judge, judgeYaml(JUDGE_PORT));
  return out;
}

// ---------------- GRPO preflight: the vLLM stack ----------------
// GRPO/RULER run a vLLM rollout server + (for RULER) a judge vLLM, which the base
// surogate install may omit. Missing it surfaces as a cryptic `ModuleNotFoundError:
// No module named 'vllm'|'msgspec'|'uvloop'` deep in the run log, so we detect it
// up front and offer a one-key install into surogate's own venv.
const GRPO_STACK_PKGS = ["vllm", "msgspec", "uvloop"];

/** Resolve the Python interpreter the `surogate` console-script runs under, so we
 *  probe/install in the SAME venv surogate uses (not the system python). */
function resolveSurogatePython(surogateBin: string): string | null {
  let scriptPath = surogateBin;
  if (!surogateBin.includes("/")) {
    // resolve on PATH ourselves rather than shelling out (no flag interpolated
    // into bash → no injection from a hostile --surogate-bin value).
    const hit = (process.env.PATH ?? "")
      .split(path.delimiter)
      .map((d) => path.join(d, surogateBin))
      .find((p) => {
        try {
          return fs.statSync(p).isFile();
        } catch {
          return false;
        }
      });
    if (!hit) return null;
    scriptPath = hit;
  }
  // the console-script's shebang points straight at its venv interpreter
  try {
    const first = fs.readFileSync(scriptPath, "utf8").split("\n", 1)[0] ?? "";
    const m = first.match(/^#!\s*(\S+)(?:\s+(\S+))?/);
    if (m) {
      // `#!/usr/bin/env python` → the real interpreter is the 2nd token; only an
      // absolute path is usable directly (a bare name falls through to the sibling).
      const interp = path.basename(m[1]!) === "env" && m[2] ? m[2] : m[1]!;
      if (interp.includes("/") && fs.existsSync(interp)) return interp;
    }
  } catch {
    /* not a readable text script (e.g. a compiled binary) — fall back below */
  }
  const sib = path.join(path.dirname(scriptPath), "python");
  return fs.existsSync(sib) ? sib : null;
}

/** Is the vLLM stack importable in surogate's venv? Cheap metadata check (does not
 *  import vllm) run off the event loop. Resolves true if we can't resolve the
 *  interpreter (don't block on an unknown setup — e.g. a Docker/remote surogate). */
export function grpoStackAvailable(surogateBin = "surogate"): Promise<boolean> {
  const py = resolveSurogatePython(surogateBin);
  if (!py) return Promise.resolve(true); // unknown interpreter → assume present rather than nag
  const probe = `import importlib.metadata as m; [m.version(x) for x in ${JSON.stringify(GRPO_STACK_PKGS)}]`;
  return new Promise((resolve) => {
    const child = spawn(py, ["-c", probe], { stdio: "ignore" });
    const timer = setTimeout(() => {
      child.kill();
      resolve(false);
    }, 8000);
    child.on("error", () => {
      clearTimeout(timer);
      resolve(false);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      resolve(code === 0);
    });
  });
}

/** Shell command to install the vLLM stack into surogate's venv (uv preferred,
 *  pip fallback). Run via runShell (bash -lc). */
export function grpoStackInstallCommand(surogateBin = "surogate"): string {
  const py = resolveSurogatePython(surogateBin) ?? "python3";
  const pkgs = GRPO_STACK_PKGS.join(" ");
  return `command -v uv >/dev/null 2>&1 && uv pip install --python "${py}" ${pkgs} || "${py}" -m pip install ${pkgs}`;
}

export function buildGrpoCommand(
  trainerGpus: number[],
  vllmGpus: number[],
  c: GrpoConfigs,
  surogateBin = "surogate",
  judgeGpus: number[] = [],
): string {
  let cmd =
    `${surogateBin} grpo --train ${c.train} --infer ${c.infer} --orch ${c.orch} ` +
    `--trainer-gpus ${trainerGpus.join(",")} --vllm-gpus ${vllmGpus.join(",")}`;
  if (c.judge && judgeGpus.length) cmd += ` --judge-infer ${c.judge} --judge-gpus ${judgeGpus.join(",")}`;
  return cmd;
}

/** Plain key→value overrides appended to the GRPO train / orch configs (YAML
 *  last-key wins). Built by the UI from the GRPO parameter form. */
export interface GrpoOverlay {
  train?: Record<string, string>;
  orch?: Record<string, string>;
}

/** Copy a base config and append override key/values, written to disk. */
function writeOverlay(basePath: string, outPath: string, extra: string[]): string {
  let body = "";
  try {
    body = fs.readFileSync(basePath, "utf8");
  } catch {
    /* missing base → just the overlay */
  }
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, appendYamlOverlay(body, extra));
  return outPath;
}

/** Spawn `surogate grpo …`. Overlays the train (and, if edited, orch) config so
 *  the run reports to the surogate feed and reflects any parameter edits. */
export function spawnGrpo(
  c: GrpoConfigs,
  trainerGpus: number[],
  vllmGpus: number[],
  metricsPath: string,
  surogateBin = "surogate",
  judgeGpus: number[] = [],
  overlay: GrpoOverlay = {},
): SpawnResult {
  const tag = path.basename(metricsPath);
  // surogate validates trainer-GPU count == train.gpus and vLLM-GPU count == tp*dp,
  // so inject the per-role counts from the user's GPU selection (no auto-reconcile).
  const trainExtra = [
    "# --- jackalope overlay ---",
    ...Object.entries(overlay.train ?? {}).map(([k, v]) => `${k}: ${v}`),
    `gpus: ${trainerGpus.length}`,
    "report_to: [surogate]",
    `surogate_metrics_path: "${metricsPath}"`,
  ];
  const trainOverlay = writeOverlay(c.train, path.join(RUNS_OVERLAY_DIR(), `grpo-train-${tag}.yaml`), trainExtra);

  // vLLM tensor-parallelism = number of inference GPUs (judge likewise).
  const inferOverlay = writeOverlay(c.infer, path.join(RUNS_OVERLAY_DIR(), `grpo-infer-${tag}.yaml`), [
    "# --- jackalope overlay ---",
    `tp: ${Math.max(1, vllmGpus.length)}`,
  ]);

  let orchPath = c.orch;
  if (overlay.orch && Object.keys(overlay.orch).length) {
    const orchExtra = ["# --- jackalope overlay ---", ...Object.entries(overlay.orch).map(([k, v]) => `${k}: ${v}`)];
    orchPath = writeOverlay(c.orch, path.join(RUNS_OVERLAY_DIR(), `grpo-orch-${tag}.yaml`), orchExtra);
  }

  const env = { ...process.env, SUROGATE_METRICS_PATH: metricsPath };
  const argv = [
    "grpo",
    "--train",
    trainOverlay,
    "--infer",
    inferOverlay,
    "--orch",
    orchPath,
    "--trainer-gpus",
    trainerGpus.join(","),
    "--vllm-gpus",
    vllmGpus.join(","),
  ];
  if (c.judge && judgeGpus.length) {
    const judgeOverlay = writeOverlay(c.judge, path.join(RUNS_OVERLAY_DIR(), `grpo-judge-${tag}.yaml`), [
      "# --- jackalope overlay ---",
      `tp: ${Math.max(1, judgeGpus.length)}`,
    ]);
    argv.push("--judge-infer", judgeOverlay, "--judge-gpus", judgeGpus.join(","));
  }
  return spawnDetached(surogateBin, argv, env, `${metricsPath}.log`, metricsPath);
}

/** Spawn a detached training process, recording its PID next to the feed. A
 *  missing/bad binary emits an async 'error' event; without a listener Node
 *  throws it as an uncaughtException and crashes the dashboard, so we swallow it
 *  — child.pid stays undefined and we surface the failure in the result. */
function spawnDetached(bin: string, argv: string[], env: NodeJS.ProcessEnv, logPath: string, metricsPath: string): SpawnResult {
  const log = fs.openSync(logPath, "w");
  const child = spawn(bin, argv, { env, stdio: ["ignore", log, log], detached: true });
  child.on("error", () => {});
  child.unref();
  if (!child.pid) return { ok: false, reason: `could not launch "${bin}" — is it on PATH? (pass --surogate-bin)` };
  writePidFile(metricsPath, child.pid);
  return { ok: true, pid: child.pid };
}

/** Record a launched run's PID (+ start-time, to detect PID recycling) next to
 *  its feed so the dashboard can control it. */
export function writePidFile(metricsPath: string, pid: number): void {
  try {
    fs.writeFileSync(`${metricsPath}.pid`, JSON.stringify({ pid, start: procStartTime(pid) }));
  } catch {
    /* ignore */
  }
}

function RUNS_OVERLAY_DIR(): string {
  return path.join(os.homedir(), ".surogate-watch", "configs");
}

// ---------------- VRAM estimate & GPU headroom ----------------

const RECIPE_BYTES: Record<string, number> = { bf16: 2, "fp8-hybrid": 1, nvfp4: 0.5, nvfp4_quartet: 0.5 };

/** Parse the parameter count (in billions) from a model id like "Qwen3-0.6B". */
export function paramsBFromModel(model: string): number | null {
  const m = /(\d+(?:\.\d+)?)\s*B/i.exec(model);
  if (!m) return null;
  // MoE "A3B" active-params hint (…-A3B) → prefer the active count when present
  const active = /A(\d+(?:\.\d+)?)B/i.exec(model);
  return active ? Number(active[1]) : Number(m[1]);
}

/** Rough VRAM estimate (GB) for a LoRA/FFT SFT run. Deliberately approximate —
 *  shown as "~est" to guide GPU choice, not as a guarantee. */
export function estimateRunVramGB(f: LaunchFields): number | null {
  const params = paramsBFromModel(f.model);
  if (params === null) return null;
  const bpp = RECIPE_BYTES[f.recipe] ?? 2;
  const weights = params * bpp;
  const bsz = Number(f.perDeviceBatch) || 1;
  const seq = Number(f.sequenceLen) || 2048;
  const act = params * 0.25 * ((bsz * seq) / 2048);
  const optimizer = f.lora ? params * 0.05 : params * 12; // LoRA: tiny; FFT: adam fp32-ish
  return Math.round((weights + act + optimizer + 1.0) * 10) / 10; // +1GB context
}

export function gpuFreeGB(g: Gpu): number | null {
  if (g.memMB === null) return null;
  const used = g.memUsedMB ?? 0;
  return Math.max(0, (g.memMB - used) / 1024);
}

export type FitVerdict = "fits" | "tight" | "risk" | "unknown";

export function fitOnGpu(estGB: number | null, freeGB: number | null): FitVerdict {
  if (estGB === null || freeGB === null) return "unknown";
  if (freeGB >= estGB * 1.3) return "fits";
  if (freeGB >= estGB) return "tight";
  return "risk";
}

/** Spawn `surogate sft <config>` detached, wiring GPUs + metrics feed. Returns the PID. */
export function spawnTraining(
  configPath: string,
  gpuIds: number[],
  metricsPath: string,
  surogateBin = "surogate",
): SpawnResult {
  const env = { ...process.env };
  if (gpuIds.length) env["CUDA_VISIBLE_DEVICES"] = gpuIds.join(",");
  env["SUROGATE_METRICS_PATH"] = metricsPath;
  return spawnDetached(surogateBin, ["sft", configPath], env, `${configPath}.log`, metricsPath);
}

/** Continue a finished/stopped run from its latest checkpoint: re-run its saved
 *  config (which already points output_dir at the checkpoints) with
 *  resume_from_checkpoint on, into a fresh run folder so it gets its own feed. */
export function relaunchFromCheckpoint(
  run: RunInfo,
  surogateBin = "surogate",
): { ok: true; feed: string; pid: number } | { ok: false; reason: string } {
  if (run.checkpoints.length === 0) return { ok: false, reason: "this run has no checkpoints to resume from" };
  if (run.meta && run.meta.mode !== "sft") return { ok: false, reason: "resume currently supports SFT runs only" };
  let base: string;
  try {
    base = fs.readFileSync(runArtifacts(run.path).configPath, "utf8");
  } catch {
    return { ok: false, reason: "couldn't read the run's config.yaml" };
  }
  const label = `resume-${run.meta?.label ?? "run"}`;
  const feed = newRunFeedPath(label, Date.now());
  const art = runArtifacts(feed);
  fs.writeFileSync(art.configPath, buildResumeYaml(base, feed));
  writeRunMeta(feed, {
    mode: "sft",
    model: run.meta?.model,
    dataset: run.meta?.dataset,
    recipe: run.meta?.recipe,
    gpus: run.meta?.gpus,
    maxSteps: run.meta?.maxSteps,
    startedAt: Date.now(),
    label,
  });
  const r = spawnTraining(art.configPath, run.meta?.gpus ?? [], feed, surogateBin);
  return r.ok ? { ok: true, feed, pid: r.pid } : { ok: false, reason: r.reason };
}

/** Find example train files (examples/sft/**\/*.yaml) to use as a base. */
export function listExampleConfigs(repoRoot: string, limit = 60): string[] {
  const base = path.join(repoRoot, "examples", "sft");
  const found: string[] = [];
  const walk = (dir: string) => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (found.length >= limit) return;
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.isFile() && (e.name.endsWith(".yaml") || e.name.endsWith(".yml"))) found.push(p);
    }
  };
  walk(base);
  return found.sort();
}
