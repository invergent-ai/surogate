// Typed records parsed from the surogate JSONL metrics feed.
//
// The feed (surogate/train/metrics_writer.py) appends three record shapes: a
// `config` header, default step lines (train/* and eval/* keys), and
// type:"gpu" lines. Parsing is forgiving: anything unusable -> null.
//
// Real-feed key facts (verified against a live qwen3 run on an RTX 5090):
//   - gradient norm arrives as `train/norm` (not `train/grad_norm`)
//   - epoch arrives as `train/epoch` (the reporter prefixes numeric step keys)
//   - GPU telemetry: `temperature`, `gpu_util` (%), `mem_util` (%), `power` (mW)

export interface ConfigRecord {
  kind: "config";
  ts: number;
  recipe: string | null;
  model: string | null;
  fields: Record<string, unknown>;
}

export interface StepRecord {
  kind: "step";
  step: number;
  ts: number;
  epoch: number | null;
  trainLoss: number | null;
  evalLoss: number | null;
  lr: number | null;
  gradNorm: number | null;
  tokensPerSecond: number | null;
  phase: string | null;
}

export interface GpuRecord {
  kind: "gpu";
  gpuId: number;
  step: number;
  ts: number;
  name: string | null;
  temp: number | null;
  power: number | null; // watts
  smUtil: number | null; // percent 0..100
  memUtil: number | null; // fraction 0..1
}

export type Record_ = ConfigRecord | StepRecord | GpuRecord;

function asNumber(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  return null;
}

function firstNumber(obj: Record<string, unknown>, ...keys: string[]): number | null {
  for (const k of keys) {
    if (k in obj) {
      const n = asNumber(obj[k]);
      if (n !== null) return n;
    }
  }
  return null;
}

function asFraction(v: number | null): number | null {
  // Accept a fraction (0..1) or a percent (0..100); normalize to 0..1.
  if (v === null) return null;
  return v > 1.0 ? v / 100.0 : v;
}

function asPower(v: number | null): number | null {
  // NVML reports power in milliwatts (e.g. 325152 -> 325 W); real GPUs never
  // draw >= 2000 W, so larger values are treated as milliwatts.
  if (v === null) return null;
  return v >= 2000 ? v / 1000.0 : v;
}

export function parseLine(line: string): Record_ | null {
  const s = line.trim();
  if (!s) return null;
  let obj: unknown;
  try {
    obj = JSON.parse(s);
  } catch {
    return null;
  }
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) return null;
  const o = obj as Record<string, unknown>;
  const type = o["type"];

  if (type === "config") {
    const fields: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) if (k !== "type" && k !== "ts") fields[k] = v;
    return {
      kind: "config",
      ts: asNumber(o["ts"]) ?? 0,
      recipe: typeof o["recipe"] === "string" ? (o["recipe"] as string) : null,
      model: typeof o["model"] === "string" ? (o["model"] as string) : null,
      fields,
    };
  }

  if (type === "gpu") {
    return {
      kind: "gpu",
      gpuId: asNumber(o["gpu_id"]) ?? 0,
      step: asNumber(o["step"]) ?? 0,
      ts: asNumber(o["ts"]) ?? 0,
      name: typeof o["name"] === "string" ? (o["name"] as string) : null,
      temp: firstNumber(o, "temperature", "temp"),
      power: asPower(firstNumber(o, "power")),
      smUtil: firstNumber(o, "gpu_utilization", "gpu_util", "sm_util"),
      memUtil: asFraction(firstNumber(o, "mem_utilization", "mem_util")),
    };
  }

  if (type === undefined && "step" in o) {
    const step = asNumber(o["step"]);
    if (step === null) return null;
    const phase = o["train/phase"] ?? o["phase"];
    return {
      kind: "step",
      step,
      ts: asNumber(o["ts"]) ?? 0,
      epoch: firstNumber(o, "train/epoch", "epoch"),
      trainLoss: asNumber(o["train/loss"]),
      evalLoss: asNumber(o["eval/loss"]),
      lr: asNumber(o["train/lr"]),
      gradNorm: firstNumber(o, "train/norm", "train/grad_norm"),
      tokensPerSecond: asNumber(o["train/tokens_per_second"]) ?? asNumber(o["eval/tokens_per_second"]),
      phase: typeof phase === "string" ? phase : null,
    };
  }

  return null;
}
