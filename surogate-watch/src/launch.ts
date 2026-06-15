// Launch helpers: GPU discovery (nvidia-smi), config-build, spawn, and example
// train-file discovery. No Ink here — pure logic, unit-testable.

import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export interface Gpu {
  id: number;
  name: string;
  memMB: number | null;
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

/** Enumerate GPUs via nvidia-smi; falls back to a single generic entry. */
export function discoverGpus(): Gpu[] {
  try {
    const out = execFileSync(
      "nvidia-smi",
      ["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
      { encoding: "utf8", timeout: 5000 },
    );
    const gpus: Gpu[] = [];
    for (const line of out.trim().split("\n")) {
      const [idx, name, mem] = line.split(",").map((s) => s.trim());
      if (idx === undefined) continue;
      gpus.push({ id: Number(idx), name: name ?? `GPU ${idx}`, memMB: mem ? Number(mem) : null });
    }
    if (gpus.length) return gpus;
  } catch {
    /* nvidia-smi unavailable */
  }
  return [{ id: 0, name: "GPU 0", memMB: null }];
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
export function buildConfigYaml(f: LaunchFields, nGpus: number): string {
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
    "datasets:",
    `  - path: "${f.datasetPath}"`,
    `    type: ${f.datasetType}`,
  );
  return lines.join("\n") + "\n";
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
