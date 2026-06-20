// Field schemas for the SFT + GRPO parameter editors (FieldEditor). Keys match
// LaunchFields / GrpoFields; the editor reads/writes a flat values map.

import { LR_SCHEDULERS, OPTIMIZERS } from "../launch.ts";
import type { FieldDef } from "./FieldEditor.tsx";

const whenLora: FieldDef["show"] = (v) => !!v["lora"];

// One-line "why" for each enum option (surogate-grounded), shown under the
// focused selector so the choice is self-explanatory.
const RECIPE_DESC: Record<string, string> = {
  bf16: "max accuracy, most VRAM — the safe default",
  "fp8-hybrid": "~1.5–2× faster GEMMs, keeps a bf16 master",
  nvfp4: "4-bit weights (Blackwell SM100+), lowest VRAM",
  nvfp4_quartet: "4-bit + Four-over-Six — lower quant error",
};
const OPTIMIZER_DESC: Record<string, string> = {
  adamw_8bit: "8-bit Adam states — ~4× less optimizer memory (default)",
  adamw: "full fp32 Adam — most memory, plain baseline",
  normuon: "orthogonalized 2D updates — often better, ~15% slower",
};
const SCHEDULER_DESC: Record<string, string> = {
  linear: "decay to 0 after warmup — simple, reliable",
  cosine: "smooth cosine decay — common for SFT",
  constant: "flat LR after warmup — for short / resumed runs",
  wsd: "warmup-stable-decay — great for known step budgets",
};
const DATASET_TYPE_DESC: Record<string, string> = {
  auto: "infer format from the columns (recommended)",
  instruction: "single-turn prompt/response pairs",
  conversation: "multi-turn chat (messages/roles)",
};

export function sftSchema(recipeOptions: readonly string[]): FieldDef[] {
  return [
    { group: "data", key: "model", label: "model", kind: "text" },
    { group: "data", key: "datasetPath", label: "dataset", kind: "text" },
    { group: "data", key: "datasetType", label: "dataset type", kind: "enum", options: ["auto", "instruction", "conversation"], desc: DATASET_TYPE_DESC },
    { group: "data", key: "recipe", label: "precision", kind: "enum", options: recipeOptions, desc: RECIPE_DESC },
    { group: "training", key: "maxSteps", label: "max steps", kind: "num", help: "-1 = derive from epochs × dataset size" },
    { group: "training", key: "learningRate", label: "learning rate", kind: "text", help: "LoRA SFT likes ~1e-4 to 2e-4; full FT lower" },
    { group: "training", key: "perDeviceBatch", label: "batch / device", kind: "num", help: "raise for throughput if VRAM allows" },
    { group: "training", key: "gradAccum", label: "grad accumulation", kind: "num", help: "effective batch = batch × accum × GPUs" },
    { group: "training", key: "sequenceLen", label: "sequence length", kind: "num", help: "≥8K? enable recompute / long context" },
    { group: "training", key: "evalSteps", label: "eval steps", kind: "num", help: "how often eval ◆ markers appear" },
    { group: "training", key: "warmupRatio", label: "warmup ratio", kind: "text", help: "0.1–0.15 stabilizes the first steps" },
    { group: "training", key: "gradientCheckpointing", label: "recompute (grad ckpt)", kind: "bool", help: "trade compute for memory — key OOM lever" },
    { group: "optimizer", key: "optimizer", label: "optimizer", kind: "enum", options: OPTIMIZERS, desc: OPTIMIZER_DESC },
    { group: "optimizer", key: "lrScheduler", label: "lr scheduler", kind: "enum", options: LR_SCHEDULERS, desc: SCHEDULER_DESC },
    { group: "optimizer", key: "weightDecay", label: "weight decay", kind: "text", help: "0 is fine for most LoRA SFT" },
    { group: "optimizer", key: "maxGradNorm", label: "max grad norm", kind: "text", help: "clip gradients; 1.0 safe, 0 disables" },
    { group: "lora", key: "lora", label: "LoRA adapter", kind: "bool", help: "train small adapters instead of full weights" },
    { group: "lora", key: "loraRank", label: "LoRA rank", kind: "num", show: whenLora, help: "16 is a solid default; raise for harder tasks" },
    { group: "lora", key: "loraAlpha", label: "LoRA alpha", kind: "num", show: whenLora, help: "scaling; commonly 2× the rank" },
    { group: "checkpoints", key: "saveSteps", label: "save every (steps)", kind: "num", help: "checkpoint cadence" },
    { group: "checkpoints", key: "saveTotalLimit", label: "keep last N", kind: "num", help: "older checkpoints are pruned" },
    { group: "checkpoints", key: "resumeFromCheckpoint", label: "resume from ckpt", kind: "bool", help: "continue if a checkpoint exists in output dir" },
    { group: "checkpoints", key: "mergeAdapter", label: "merge adapter", kind: "bool", show: whenLora, help: "bake LoRA into the base model after training" },
    { group: "output", key: "outputDir", label: "output dir", kind: "text", help: "where checkpoints are written" },
  ];
}

// GRPO/RULER editable overlay. These all map to TOP-LEVEL scalars in the example
// train.yaml / orch.yaml, so appending them as an overlay safely overrides (YAML
// last-key wins) without disturbing the rest of the user's config.
export interface GrpoFields {
  learningRate: string; // train
  maxSteps: string; // train + orch (must match)
  saveSteps: string; // train
  perDeviceBatch: string; // train
  sequenceLen: string; // train + orch
  recipe: string; // train
  rolloutsPerExample: string; // orch — group size (G)
  batchSize: string; // orch
}

export const GRPO_DEFAULTS: GrpoFields = {
  learningRate: "1e-4",
  maxSteps: "20",
  saveSteps: "100",
  perDeviceBatch: "1",
  sequenceLen: "2048",
  recipe: "bf16",
  rolloutsPerExample: "16",
  batchSize: "128",
};

// dstack cloud config (collected before launching a remote cloud run).
export interface DstackFields {
  gpu: string;
  count: string; // stored as string (FieldEditor edits strings)
  image: string;
  backend: string; // "cheapest" | provider name
  region: string;
}
export const DSTACK_DEFAULTS: DstackFields = {
  gpu: "H100",
  count: "1",
  image: "",
  backend: "cheapest",
  region: "",
};
const GPU_DESC: Record<string, string> = {
  H100: "Hopper · bf16 + fp8-hybrid · the fast default (no NVFP4)",
  A100: "Ampere · bf16 + QLoRA only (no fp8) · widely available",
  L40S: "Ada · bf16 + fp8-hybrid · 48GB · great value",
  L4: "Ada · bf16 + fp8-hybrid · 24GB · small models / LoRA",
  A10G: "Ampere · bf16 + QLoRA · 24GB · small models / LoRA",
  T4: "Turing · bf16 + QLoRA · 16GB · tiny models only",
};
const BACKEND_DESC: Record<string, string> = {
  cheapest: "let dstack pick the cheapest available provider",
  runpod: "RunPod · community + secure cloud",
  lambda: "Lambda · per-minute, no spot",
  vastai: "Vast.ai · marketplace, cheapest spot",
  aws: "AWS EC2",
  gcp: "Google Cloud",
  azure: "Microsoft Azure",
};
export function dstackSchema(): FieldDef[] {
  return [
    { group: "cloud", key: "gpu", label: "GPU type", kind: "enum", options: ["H100", "A100", "L40S", "L4", "A10G", "T4"], desc: GPU_DESC },
    { group: "cloud", key: "count", label: "GPU count", kind: "num", help: "how many GPUs to provision" },
    { group: "cloud", key: "image", label: "Docker image", kind: "text", help: "a surogate-ready image — training needs surogate installed in it" },
    { group: "cloud", key: "backend", label: "provider", kind: "enum", options: ["cheapest", "runpod", "lambda", "vastai", "aws", "gcp", "azure"], desc: BACKEND_DESC },
    { group: "cloud", key: "region", label: "region", kind: "text", help: "optional — e.g. us-east-1 (leave blank for any)" },
  ];
}

export function grpoSchema(recipeOptions: readonly string[]): FieldDef[] {
  return [
    { group: "trainer", key: "learningRate", label: "learning rate", kind: "text" },
    { group: "trainer", key: "maxSteps", label: "max steps", kind: "num" },
    { group: "trainer", key: "saveSteps", label: "save every (steps)", kind: "num" },
    { group: "trainer", key: "perDeviceBatch", label: "batch / device", kind: "num" },
    { group: "trainer", key: "sequenceLen", label: "sequence length", kind: "num" },
    { group: "trainer", key: "recipe", label: "precision", kind: "enum", options: recipeOptions, desc: RECIPE_DESC },
    { group: "RL (orchestrator)", key: "rolloutsPerExample", label: "rollouts/example (group)", kind: "num", help: "GRPO group size G — completions scored per prompt" },
    { group: "RL (orchestrator)", key: "batchSize", label: "orch batch size", kind: "num", help: "prompts per orchestrator step" },
  ];
}
