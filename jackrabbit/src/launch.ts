// Launch helpers: GPU discovery (nvidia-smi), config-build, spawn, and example
// train-file discovery. No Ink here — pure logic, unit-testable.

import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface Gpu {
  id: number;
  name: string;
  memMB: number | null;
  memUsedMB: number | null;
  util: number | null;
  busy: boolean; // already running something (util>5% or >1.5GB used)
}

export const RECIPES = ["bf16", "fp8-hybrid", "nvfp4", "nvfp4_quartet"] as const;
export type Recipe = (typeof RECIPES)[number];

export const DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

export interface LaunchFields {
  model: string;
  recipe: Recipe;
  lora: boolean;
  loraRank: string;
  loraAlpha: string;
  perDeviceBatch: string;
  gradAccum: string;
  sequenceLen: string;
  maxSteps: string;
  evalSteps: string;
  learningRate: string;
  warmupRatio: string;
  outputDir: string;
  datasetPath: string;
  datasetType: string;
}

export const DEFAULT_FIELDS: LaunchFields = {
  model: "Qwen/Qwen3-0.6B",
  recipe: "fp8-hybrid",
  lora: true,
  loraRank: "16",
  loraAlpha: "32",
  perDeviceBatch: "2",
  gradAccum: "4",
  sequenceLen: "2048",
  maxSteps: "200",
  evalSteps: "25",
  learningRate: "2e-4",
  warmupRatio: "0.15",
  outputDir: "./watch-out",
  datasetPath: "OpenLLM-Ro/ro_gsm8k",
  datasetType: "auto",
};

/** Enumerate GPUs via nvidia-smi (with live util/memory so we can flag busy
 *  devices); falls back to a single generic entry. */
export function discoverGpus(): Gpu[] {
  try {
    const out = execFileSync(
      "nvidia-smi",
      ["--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
      { encoding: "utf8", timeout: 5000 },
    );
    const gpus: Gpu[] = [];
    for (const line of out.trim().split("\n")) {
      const [idx, name, mem, memUsed, util] = line.split(",").map((s) => s.trim());
      if (idx === undefined) continue;
      const memUsedMB = memUsed ? Number(memUsed) : null;
      const u = util ? Number(util) : null;
      const busy = (u !== null && u > 5) || (memUsedMB !== null && memUsedMB > 1500);
      gpus.push({
        id: Number(idx),
        name: name ?? `GPU ${idx}`,
        memMB: mem ? Number(mem) : null,
        memUsedMB,
        util: u,
        busy,
      });
    }
    if (gpus.length) return gpus;
  } catch {
    /* nvidia-smi unavailable */
  }
  return [{ id: 0, name: "GPU 0", memMB: null, memUsedMB: null, util: null, busy: false }];
}

/** Resolve the surogate metrics feed path the same way the framework does. */
export function resolveFeedPath(explicit?: string): string {
  return explicit || process.env.SUROGATE_METRICS_PATH || "/tmp/surogate_metrics.jsonl";
}

/** Build the `surogate sft` invocation string (for preview). */
export function buildCommand(gpuIds: number[], configPath: string, surogateBin = "surogate"): string {
  const prefix = gpuIds.length ? `CUDA_VISIBLE_DEVICES=${gpuIds.join(",")} ` : "";
  return `${prefix}${surogateBin} sft ${configPath}`;
}

/** Build a YAML SFT config from the launch fields (matches surogate's schema). */
export function buildConfigYaml(f: LaunchFields, nGpus: number, metricsPath?: string): string {
  const lines = [
    `model: ${f.model}`,
    `output_dir: ${f.outputDir}`,
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
    `lora: ${f.lora ? "true" : "false"}`,
  ];
  if (f.lora) {
    lines.push(`lora_rank: ${f.loraRank}`, `lora_alpha: ${f.loraAlpha}`);
    lines.push("lora_target_modules:");
    for (const m of DEFAULT_TARGET_MODULES) lines.push(`  - ${m}`);
  }
  lines.push(
    "# live monitoring",
    "logging_steps: 1",
    "log_gpu_util: 5",
    "report_to: [surogate]",
  );
  if (metricsPath) lines.push(`surogate_metrics_path: ${metricsPath}`);
  lines.push("datasets:", `  - path: "${f.datasetPath}"`, `    type: ${f.datasetType}`);
  return lines.join("\n") + "\n";
}

// ---------------- GRPO (split-GPU RL) ----------------

export interface GrpoConfigs {
  train: string;
  infer: string;
  orch: string;
}

/** Default GRPO example config paths shipped with surogate. */
export function exampleGrpoConfigs(repoRoot: string): GrpoConfigs {
  const d = path.join(repoRoot, "examples", "grpo");
  return { train: path.join(d, "train.yaml"), infer: path.join(d, "infer.yaml"), orch: path.join(d, "orch.yaml") };
}

export function grpoConfigsExist(c: GrpoConfigs): boolean {
  return fs.existsSync(c.train) && fs.existsSync(c.infer) && fs.existsSync(c.orch);
}

export function buildGrpoCommand(
  trainerGpus: number[],
  vllmGpus: number[],
  c: GrpoConfigs,
  surogateBin = "surogate",
): string {
  return (
    `${surogateBin} grpo --train ${c.train} --infer ${c.infer} --orch ${c.orch} ` +
    `--trainer-gpus ${trainerGpus.join(",")} --vllm-gpus ${vllmGpus.join(",")}`
  );
}

/** Spawn `surogate grpo …`. Writes a train-config overlay that points the
 *  metrics feed at `metricsPath` so the dashboard can follow this run. */
export function spawnGrpo(
  c: GrpoConfigs,
  trainerGpus: number[],
  vllmGpus: number[],
  metricsPath: string,
  surogateBin = "surogate",
): number {
  // overlay: copy the train config and ensure it reports to the surogate feed
  const overlayPath = path.join(RUNS_OVERLAY_DIR(), `grpo-train-${path.basename(metricsPath)}.yaml`);
  let trainBody = "";
  try {
    trainBody = fs.readFileSync(c.train, "utf8");
  } catch {
    /* ignore */
  }
  trainBody += `\nreport_to: [surogate]\nsurogate_metrics_path: ${metricsPath}\n`;
  fs.mkdirSync(path.dirname(overlayPath), { recursive: true });
  fs.writeFileSync(overlayPath, trainBody);

  const env = { ...process.env, SUROGATE_METRICS_PATH: metricsPath };
  const log = fs.openSync(`${metricsPath}.log`, "w");
  const child = spawn(
    surogateBin,
    [
      "grpo",
      "--train",
      overlayPath,
      "--infer",
      c.infer,
      "--orch",
      c.orch,
      "--trainer-gpus",
      trainerGpus.join(","),
      "--vllm-gpus",
      vllmGpus.join(","),
    ],
    { env, stdio: ["ignore", log, log], detached: true },
  );
  child.unref();
  if (child.pid) writePidFile(metricsPath, child.pid);
  return child.pid ?? -1;
}

/** Record a launched run's PID next to its feed so the dashboard can control it. */
export function writePidFile(metricsPath: string, pid: number): void {
  try {
    fs.writeFileSync(`${metricsPath}.pid`, String(pid));
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
): number {
  const env = { ...process.env };
  if (gpuIds.length) env["CUDA_VISIBLE_DEVICES"] = gpuIds.join(",");
  env["SUROGATE_METRICS_PATH"] = metricsPath;
  const logFd = fs.openSync(`${configPath}.log`, "w");
  const child = spawn(surogateBin, ["sft", configPath], {
    env,
    stdio: ["ignore", logFd, logFd],
    detached: true,
  });
  child.unref();
  if (child.pid) writePidFile(metricsPath, child.pid);
  return child.pid ?? -1;
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
