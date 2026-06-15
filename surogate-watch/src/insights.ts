// Real-time insights derived from WatchState: training health, device alerts,
// smart tips, and run facts. Pure logic — the rail just renders these.

import type { WatchState } from "./state.ts";
import { fmtCount, fmtEta, fmtFloat } from "./format.ts";

export type InsightColor = "green" | "warm" | "red" | "accent" | "eval" | "muted";
export interface Insight {
  icon: string;
  text: string;
  color: InsightColor;
}
export interface InsightGroups {
  health: Insight[];
  alerts: Insight[];
  tips: Insight[];
  facts: Insight[];
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function str(v: unknown): string | null {
  return typeof v === "string" && v ? v : null;
}

export function computeHealth(s: WatchState): Insight[] {
  const out: Insight[] = [];
  if (s.lossHistory.length >= 2) {
    const t = s.lossTrend(20);
    const rel = Math.abs(t) / (Math.abs(s.latestTrainLoss ?? 1) || 1);
    if (t < 0 && rel > 0.005) out.push({ icon: "✓", text: "loss ↓ steady", color: "green" });
    else if (rel <= 0.005) out.push({ icon: "≈", text: "loss plateau", color: "warm" });
    else out.push({ icon: "▲", text: "loss rising", color: "red" });
  }
  if (s.maxSteps && s.maxSteps > 0) {
    const pct = Math.min(100, Math.round((s.step / s.maxSteps) * 100));
    out.push({ icon: "◔", text: `${pct}% · ${s.step}/${s.maxSteps}`, color: "accent" });
  }
  const eta = s.etaSeconds();
  if (eta !== null) out.push({ icon: "◷", text: `ETA ${fmtEta(eta)}`, color: "accent" });
  if (s.tokensPerSecond !== null) out.push({ icon: "⚡", text: `${fmtCount(s.tokensPerSecond)} tok/s`, color: "accent" });
  if (s.latestEvalLoss !== null) out.push({ icon: "◆", text: `eval ${fmtFloat(s.latestEvalLoss)}`, color: "eval" });
  return out;
}

export function computeAlerts(s: WatchState): Insight[] {
  const out: Insight[] = [];
  for (const g of s.gpusSorted()) {
    if (g.temp !== null && g.temp >= 80) out.push({ icon: "⚠", text: `gpu${g.gpuId} ${Math.round(g.temp)}° hot`, color: "red" });
    else if (g.temp !== null && g.temp >= 70) out.push({ icon: "⚠", text: `gpu${g.gpuId} ${Math.round(g.temp)}° warm`, color: "warm" });
    if (g.memUtil !== null && g.memUtil >= 0.92) out.push({ icon: "⚠", text: `gpu${g.gpuId} mem ${Math.round(g.memUtil * 100)}%`, color: "red" });
    if (g.smUtil !== null && g.smUtil < 50 && s.step > 2)
      out.push({ icon: "⚠", text: `gpu${g.gpuId} util ${Math.round(g.smUtil)}% low`, color: "warm" });
  }
  if (out.length === 0 && s.hasGpus) out.push({ icon: "✓", text: "all devices nominal", color: "green" });
  return out;
}

export function computeTips(s: WatchState): Insight[] {
  const out: Insight[] = [];
  const recipe = s.recipe ?? "";
  if (recipe === "bf16") out.push({ icon: "💡", text: "fp8-hybrid ≈1.8× faster than bf16", color: "muted" });
  else if (recipe.startsWith("fp8") || recipe.startsWith("nvfp4")) out.push({ icon: "💡", text: `${recipe} fast path active`, color: "muted" });
  if (s.gradNorm !== null && s.gradNorm < 0.3) out.push({ icon: "💡", text: "grad-norm low → LR headroom", color: "muted" });
  else if (s.gradNorm !== null && s.gradNorm > 5) out.push({ icon: "💡", text: "grad-norm high → lower LR / clip", color: "muted" });
  const gpus = s.gpusSorted();
  if (gpus.length && gpus.every((g) => g.memUtil !== null && g.memUtil < 0.6))
    out.push({ icon: "💡", text: "memory headroom → larger batch", color: "muted" });
  if (out.length === 0) out.push({ icon: "💡", text: "report_to:[surogate] enables this view", color: "muted" });
  return out;
}

export function computeFacts(s: WatchState): Insight[] {
  const f = s.configFields;
  const out: Insight[] = [];
  if (s.model) out.push({ icon: "·", text: s.model, color: "muted" });
  if (s.recipe) out.push({ icon: "·", text: `recipe ${s.recipe}`, color: "muted" });
  const bsz = num(f["per_device_train_batch_size"]);
  const seq = num(f["sequence_len"]);
  if (bsz !== null) out.push({ icon: "·", text: `bsz ${bsz}${seq !== null ? ` · seq ${seq}` : ""}`, color: "muted" });
  const rank = num(f["lora_rank"]);
  if (s.lora && rank !== null) out.push({ icon: "·", text: `LoRA r${rank}`, color: "muted" });
  const out_dir = str(f["output_dir"]);
  if (out_dir) out.push({ icon: "·", text: out_dir, color: "muted" });
  if (s.hasGpus) out.push({ icon: "·", text: `${s.gpusSorted().length} GPU(s)`, color: "muted" });
  return out;
}

export function computeInsights(s: WatchState): InsightGroups {
  return {
    health: computeHealth(s),
    alerts: computeAlerts(s),
    tips: computeTips(s),
    facts: computeFacts(s),
  };
}
