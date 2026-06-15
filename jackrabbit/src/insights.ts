// Real-time insights derived from WatchState: training health, device alerts,
// smart tips, and run facts. Pure logic — the rail just renders these.

import type { WatchState } from "./state.ts";
import { fmtCount, fmtEta, fmtFloat } from "./format.ts";
import { TIPS, type TipTag } from "./tips.ts";

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

/** Context tags that are "active" for the current run — used to filter tips so
 *  the feed shows things relevant to what's actually happening. */
export function activeTags(s: WatchState): Set<TipTag> {
  const a = new Set<TipTag>();
  const recipe = s.recipe ?? "";
  if (recipe === "bf16") a.add("bf16");
  else if (recipe.startsWith("fp8")) a.add("fp8");
  else if (recipe.startsWith("nvfp4")) a.add("nvfp4");
  if (s.lora) a.add("lora");
  else if (s.lora === false) a.add("fft");
  if (Object.keys(s.configFields).some((k) => k.startsWith("qlora") && s.configFields[k])) a.add("qlora");
  if (/moe|a\d+b/i.test(s.model ?? "") || (num(s.configFields["ep_size"]) ?? 0) > 1) a.add("moe");
  if (Object.keys(s.configFields).some((k) => /kl|rollout|grpo|advantage|reward/i.test(k))) a.add("grpo");
  const gpus = s.gpusSorted();
  if (gpus.some((g) => g.memUtil !== null && g.memUtil >= 0.85)) a.add("memtight");
  if (gpus.length && gpus.every((g) => g.memUtil !== null && g.memUtil < 0.6)) a.add("memroom");
  if (gpus.some((g) => g.temp !== null && g.temp >= 78)) a.add("hot");
  if (s.step > 2 && gpus.some((g) => g.smUtil !== null && g.smUtil < 50)) a.add("lowutil");
  const trend = s.lossTrend(20);
  const rel = Math.abs(trend) / (Math.abs(s.latestTrainLoss ?? 1) || 1);
  if (s.lossHistory.length >= 5 && rel <= 0.005) a.add("plateau");
  if (s.lossHistory.length >= 5 && trend > 0 && rel > 0.01) a.add("diverge");
  if ((num(s.configFields["sequence_len"]) ?? 0) >= 8192) a.add("longseq");
  if (gpus.length >= 2) a.add("multigpu");
  a.add("throughput"); // broadly useful
  return a;
}

/** Eligible tips = general (untagged) + any whose tag matches the live run. */
export function eligibleTips(s: WatchState): string[] {
  const active = activeTags(s);
  return TIPS.filter((t) => !t.tags || t.tags.length === 0 || t.tags.some((tag) => active.has(tag))).map((t) => t.t);
}

const TIP_ROTATE_MS = 4500;

/** A rotating window of `count` tips, advancing over time (a live feed). */
export function computeTips(s: WatchState, nowMs: number, count = 2): Insight[] {
  const pool = eligibleTips(s);
  if (pool.length === 0) return [];
  const base = Math.floor(nowMs / TIP_ROTATE_MS);
  const out: Insight[] = [];
  for (let i = 0; i < Math.min(count, pool.length); i++) {
    out.push({ icon: "💡", text: pool[(base + i) % pool.length]!, color: "muted" });
  }
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

export function computeInsights(s: WatchState, nowMs: number): InsightGroups {
  return {
    health: computeHealth(s),
    alerts: computeAlerts(s),
    tips: computeTips(s, nowMs),
    facts: computeFacts(s),
  };
}
