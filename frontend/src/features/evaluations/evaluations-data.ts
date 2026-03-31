// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

// ── Types ─────────────────────────────────────────────────────

export type BenchmarkCategory =
  | "reasoning"
  | "language"
  | "knowledge"
  | "coding"
  | "safety"
  | "chat"
  | "instruction"
  | "custom";

export interface Benchmark {
  id: string;
  name: string;
  category: BenchmarkCategory;
  description: string;
  samples: number;
  metric: string;
  icon: string;
}

export interface Score {
  value: number;
  previous: number | null;
  delta: number | null;
}

export interface EvalSample {
  benchmark: string;
  id: string;
  input: string;
  expected: string;
  predicted: string;
  correct: boolean;
}

export interface EvalRun {
  id: string;
  name: string;
  status: "completed" | "running" | "failed" | "queued";
  model: string;
  modelLabel: string;
  compareModel: string | null;
  compareLabel: string | null;
  benchmarks: string[];
  startedAt: string;
  completedAt: string | null;
  duration: string | null;
  runner: string;
  compute: string;
  gpu: string;
  progress?: number;
  currentBenchmark?: string;
  scores: Record<string, Score>;
  samples: EvalSample[];
}

export interface LeaderboardEntry {
  model: string;
  label: string;
  gsm8k: number | null;
  mmlu: number | null;
  humaneval: number | null;
  mtbench: number | null;
  color: string;
}

// ── Category colors ───────────────────────────────────────────

export const CATEGORY_COLORS: Record<
  BenchmarkCategory,
  { bg: string; fg: string; border: string }
> = {
  reasoning: { bg: "#F59E0B12", fg: "#F59E0B", border: "#F59E0B30" },
  language: { bg: "#3B82F612", fg: "#3B82F6", border: "#3B82F630" },
  knowledge: { bg: "#8B5CF612", fg: "#8B5CF6", border: "#8B5CF630" },
  coding: { bg: "#22C55E12", fg: "#22C55E", border: "#22C55E30" },
  safety: { bg: "#EF444412", fg: "#EF4444", border: "#EF444430" },
  chat: { bg: "#06B6D412", fg: "#06B6D4", border: "#06B6D430" },
  instruction: { bg: "#EC489912", fg: "#EC4899", border: "#EC489930" },
  custom: { bg: "#F59E0B12", fg: "#F59E0B", border: "#F59E0B30" },
};

// ── Data ──────────────────────────────────────────────────────

export const BENCHMARKS: Benchmark[] = [
  { id: "gsm8k", name: "GSM8K", category: "reasoning", description: "Grade school math word problems testing multi-step arithmetic reasoning", samples: 1319, metric: "accuracy", icon: "\u2211" },
  { id: "arc-agi", name: "ARC-AGI", category: "reasoning", description: "Abstraction and Reasoning Corpus for measuring fluid intelligence and pattern recognition", samples: 400, metric: "accuracy", icon: "\u25C8" },
  { id: "hellaswag", name: "HellaSwag", category: "language", description: "Sentence completion requiring commonsense reasoning about physical situations", samples: 10042, metric: "accuracy", icon: "\u22A1" },
  { id: "mmlu", name: "MMLU", category: "knowledge", description: "Massive Multitask Language Understanding across 57 academic subjects", samples: 14042, metric: "accuracy", icon: "\u25A4" },
  { id: "humaneval", name: "HumanEval", category: "coding", description: "Python function completion measuring code generation correctness", samples: 164, metric: "pass@1", icon: "\u229E" },
  { id: "mbpp", name: "MBPP", category: "coding", description: "Mostly Basic Python Problems for code generation evaluation", samples: 500, metric: "pass@1", icon: "\u229E" },
  { id: "truthfulqa", name: "TruthfulQA", category: "safety", description: "Questions designed to test whether models generate truthful and informative answers", samples: 817, metric: "accuracy", icon: "\u25C7" },
  { id: "mt-bench", name: "MT-Bench", category: "chat", description: "Multi-turn conversation quality rated by GPT-4 judge across 8 categories", samples: 80, metric: "score/10", icon: "\u22A1" },
  { id: "ifeval", name: "IFEval", category: "instruction", description: "Instruction Following Evaluation measuring adherence to verifiable constraints", samples: 541, metric: "accuracy", icon: "\u26A1" },
  { id: "toxigen", name: "ToxiGen", category: "safety", description: "Toxicity detection and generation safety across 13 demographic groups", samples: 6541, metric: "safety%", icon: "\u25C8" },
  { id: "cx-quality", name: "CX Quality", category: "custom", description: "Custom eval: customer experience response quality, empathy, accuracy", samples: 200, metric: "score/5", icon: "\u2605" },
  { id: "sql-accuracy", name: "SQL Accuracy", category: "custom", description: "Custom eval: SQL generation correctness against gold queries", samples: 150, metric: "exec_match", icon: "\u2605" },
];

export const EVAL_RUNS: EvalRun[] = [
  {
    id: "eval-0018",
    name: "CX Fine-tune v4 \u2014 Full Suite",
    status: "completed",
    model: "llama-3.1-8b-cx",
    modelLabel: "Llama 3.1 8B CX v4",
    compareModel: "llama-3.1-8b-cx-v3",
    compareLabel: "CX v3 (previous)",
    benchmarks: ["gsm8k", "mmlu", "hellaswag", "truthfulqa", "mt-bench", "cx-quality", "ifeval", "toxigen"],
    startedAt: "2h ago",
    completedAt: "45m ago",
    duration: "1h 15m",
    runner: "A. Kov\u00E1cs",
    compute: "local",
    gpu: "2\u00D7 A100",
    scores: {
      "gsm8k": { value: 82.4, previous: 78.1, delta: +4.3 },
      "mmlu": { value: 68.2, previous: 67.8, delta: +0.4 },
      "hellaswag": { value: 79.1, previous: 78.5, delta: +0.6 },
      "truthfulqa": { value: 54.8, previous: 52.1, delta: +2.7 },
      "mt-bench": { value: 7.8, previous: 7.4, delta: +0.4 },
      "cx-quality": { value: 4.6, previous: 4.1, delta: +0.5 },
      "ifeval": { value: 71.2, previous: 68.9, delta: +2.3 },
      "toxigen": { value: 96.2, previous: 95.8, delta: +0.4 },
    },
    samples: [
      { benchmark: "gsm8k", id: "gsm-0042", input: "Janet\u2019s ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4 every morning...", expected: "9", predicted: "9", correct: true },
      { benchmark: "gsm8k", id: "gsm-0108", input: "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts total for 3 robes?", expected: "9", predicted: "9", correct: true },
      { benchmark: "gsm8k", id: "gsm-0247", input: "Weng earns $12 per hour babysitting. Yesterday she babysat 50 minutes...", expected: "10", predicted: "12", correct: false },
      { benchmark: "cx-quality", id: "cx-0012", input: "Customer asks about refund for a damaged item received 45 days ago", expected: "Empathetic, checks premium status, offers correct policy", predicted: "Applies standard 30-day policy without checking membership", correct: false },
      { benchmark: "cx-quality", id: "cx-0033", input: "Customer wants to upgrade plan mid-cycle", expected: "Clear proration explanation, confirms before charging", predicted: "Calculates proration, shows breakdown, asks for confirmation", correct: true },
    ],
  },
  {
    id: "eval-0017",
    name: "DeepSeek R1 \u2014 Code Benchmarks",
    status: "completed",
    model: "deepseek-r1-code",
    modelLabel: "DeepSeek R1",
    compareModel: null,
    compareLabel: null,
    benchmarks: ["humaneval", "mbpp", "gsm8k", "mmlu", "mt-bench"],
    startedAt: "6h ago",
    completedAt: "4h ago",
    duration: "2h 10m",
    runner: "M. Chen",
    compute: "local",
    gpu: "4\u00D7 A100",
    scores: {
      "humaneval": { value: 91.5, previous: null, delta: null },
      "mbpp": { value: 86.2, previous: null, delta: null },
      "gsm8k": { value: 94.1, previous: null, delta: null },
      "mmlu": { value: 82.4, previous: null, delta: null },
      "mt-bench": { value: 8.6, previous: null, delta: null },
    },
    samples: [],
  },
  {
    id: "eval-0016",
    name: "Qwen 2.5 72B \u2014 SQL & Reasoning",
    status: "completed",
    model: "qwen-2.5-72b",
    modelLabel: "Qwen 2.5 72B",
    compareModel: null,
    compareLabel: null,
    benchmarks: ["gsm8k", "mmlu", "sql-accuracy", "mt-bench", "arc-agi"],
    startedAt: "1d ago",
    completedAt: "1d ago",
    duration: "3h 40m",
    runner: "R. Silva",
    compute: "aws",
    gpu: "4\u00D7 H100",
    scores: {
      "gsm8k": { value: 91.8, previous: null, delta: null },
      "mmlu": { value: 84.6, previous: null, delta: null },
      "sql-accuracy": { value: 78.4, previous: null, delta: null },
      "mt-bench": { value: 8.4, previous: null, delta: null },
      "arc-agi": { value: 42.1, previous: null, delta: null },
    },
    samples: [],
  },
  {
    id: "eval-0015",
    name: "Guard 3B \u2014 Safety Suite",
    status: "completed",
    model: "guard-3b",
    modelLabel: "LlamaGuard 3B v2",
    compareModel: "guard-3b-v1",
    compareLabel: "Guard v1",
    benchmarks: ["toxigen", "truthfulqa"],
    startedAt: "2d ago",
    completedAt: "2d ago",
    duration: "28m",
    runner: "A. Kov\u00E1cs",
    compute: "local",
    gpu: "1\u00D7 A100",
    scores: {
      "toxigen": { value: 97.8, previous: 94.2, delta: +3.6 },
      "truthfulqa": { value: 61.2, previous: 55.8, delta: +5.4 },
    },
    samples: [],
  },
  {
    id: "eval-0014",
    name: "Mistral 7B vs Llama 8B \u2014 CX A/B",
    status: "completed",
    model: "mistral-7b-exp",
    modelLabel: "Mistral 7B",
    compareModel: "llama-3.1-8b-cx",
    compareLabel: "Llama 3.1 8B CX v4",
    benchmarks: ["gsm8k", "mmlu", "hellaswag", "cx-quality", "mt-bench", "ifeval"],
    startedAt: "5d ago",
    completedAt: "5d ago",
    duration: "1h 50m",
    runner: "A. Kov\u00E1cs",
    compute: "local",
    gpu: "2\u00D7 A100",
    scores: {
      "gsm8k": { value: 72.6, previous: 82.4, delta: -9.8 },
      "mmlu": { value: 64.1, previous: 68.2, delta: -4.1 },
      "hellaswag": { value: 81.2, previous: 79.1, delta: +2.1 },
      "cx-quality": { value: 3.4, previous: 4.6, delta: -1.2 },
      "mt-bench": { value: 7.2, previous: 7.8, delta: -0.6 },
      "ifeval": { value: 65.8, previous: 71.2, delta: -5.4 },
    },
    samples: [],
  },
  {
    id: "eval-0019",
    name: "CX v4 \u2014 Safety Regression Check",
    status: "running",
    model: "llama-3.1-8b-cx",
    modelLabel: "Llama 3.1 8B CX v4",
    compareModel: null,
    compareLabel: null,
    benchmarks: ["toxigen", "truthfulqa", "ifeval"],
    startedAt: "12m ago",
    completedAt: null,
    duration: null,
    runner: "A. Kov\u00E1cs",
    compute: "local",
    gpu: "1\u00D7 A100",
    progress: 42,
    currentBenchmark: "truthfulqa",
    scores: {
      "toxigen": { value: 96.5, previous: null, delta: null },
    },
    samples: [],
  },
];

export const LEADERBOARD: LeaderboardEntry[] = [
  { model: "deepseek-r1-code", label: "DeepSeek R1", gsm8k: 94.1, mmlu: 82.4, humaneval: 91.5, mtbench: 8.6, color: "#3B82F6" },
  { model: "qwen-2.5-72b", label: "Qwen 2.5 72B", gsm8k: 91.8, mmlu: 84.6, humaneval: null, mtbench: 8.4, color: "#8B5CF6" },
  { model: "llama-3.1-8b-cx", label: "Llama 3.1 8B CX v4", gsm8k: 82.4, mmlu: 68.2, humaneval: null, mtbench: 7.8, color: "#F59E0B" },
  { model: "mistral-7b-exp", label: "Mistral 7B", gsm8k: 72.6, mmlu: 64.1, humaneval: null, mtbench: 7.2, color: "#22C55E" },
  { model: "guard-3b", label: "LlamaGuard 3B", gsm8k: null, mmlu: null, humaneval: null, mtbench: null, color: "#EF4444" },
];
