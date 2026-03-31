// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Status } from "@/components/ui/status-dot";

// ── Types ───────────────────────────────────────────────────────

export interface ModelReplicas {
  current: number;
  desired: number;
}

export interface ModelGpu {
  type: string;
  count: number;
  utilization: number;
}

export interface ModelVram {
  used: string;
  total: string;
  pct: number;
}

export interface ModelConnectedAgent {
  name: string;
  status: string;
  rps: number;
}

export interface ModelServingConfig {
  maxModelLen: number;
  tensorParallelSize: number;
  maxBatchSize: number;
  gpuMemoryUtilization: number;
  swapSpace: string;
  quantization: string;
  dtype: string;
  enforceEager: boolean;
  enableChunkedPrefill: boolean;
  maxNumSeqs: number;
}

export interface ModelGenerationDefaults {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repetitionPenalty: number;
  stopSequences: string[];
}

export interface ModelFineTune {
  name: string;
  method: string;
  dataset: string;
  date: string;
  status: string;
  loss: string;
  hubRef: string;
}

export interface ModelMetricsHistory {
  tps: number[];
  latency: number[];
  gpu: number[];
  queue: number[];
}

export interface ModelEvent {
  time: string;
  text: string;
  type: string;
}

export interface Model {
  id: string;
  name: string;
  displayName: string;
  description: string;
  base: string;
  family: string;
  paramCount: string;
  type: string;
  quantization: string;
  contextWindow: number;
  status: string;
  engine: string;
  replicas: ModelReplicas;
  gpu: ModelGpu;
  vram: ModelVram;
  cpu: string;
  mem: string;
  memLimit: string;
  tps: number;
  p50: string;
  p95: string;
  p99: string;
  queueDepth: number;
  batchSize: string;
  tokensIn24h: string;
  tokensOut24h: string;
  requests24h: number;
  errorRate: string;
  uptime: string;
  lastDeployed: string;
  deployedBy: string;
  namespace: string;
  projectColor: string;
  endpoint: string;
  image: string;
  hubRef: string;
  connectedAgents: ModelConnectedAgent[];
  servingConfig: ModelServingConfig | null;
  generationDefaults: ModelGenerationDefaults | null;
  fineTunes: ModelFineTune[];
  metricsHistory: ModelMetricsHistory;
  events: ModelEvent[];
}

// ── Status mapping ──────────────────────────────────────────────

const STATUS_MAP: Record<string, Status> = {
  serving: "serving",
  running: "running",
  deploying: "deploying",
  error: "error",
  stopped: "stopped",
};

export function toStatus(raw: string): Status {
  return STATUS_MAP[raw] ?? "stopped";
}

// ── Type badge styles ───────────────────────────────────────────

export const TYPE_STYLES: Record<string, { bg: string; fg: string }> = {
  "Fine-tuned": { bg: "bg-amber-500/10", fg: "text-amber-500" },
  Base: { bg: "bg-blue-500/10", fg: "text-blue-500" },
};

// ── Event type colors ───────────────────────────────────────────

export const EVENT_COLORS: Record<string, string> = {
  deploy: "#3B82F6",
  scale: "#22C55E",
  warning: "#F59E0B",
  error: "#EF4444",
  config: "#8B5CF6",
  hub: "#22C55E",
};

// ── Demo data ───────────────────────────────────────────────────

export const MODELS: Model[] = [
  {
    id: "llama-3.1-8b-cx",
    name: "llama-3.1-8b-cx",
    displayName: "Llama 3.1 8B — CX Fine-tune",
    description: "Production fine-tune of Llama 3.1 8B for customer experience workflows. Trained on 12K curated support conversations with SFT + DPO. Optimized for low-latency, high-throughput support interactions.",
    base: "meta-llama/Llama-3.1-8B-Instruct",
    family: "Llama 3.1",
    paramCount: "8B",
    type: "Fine-tuned",
    quantization: "AWQ 4-bit",
    contextWindow: 8192,
    status: "serving",
    engine: "vLLM 0.6.4",
    replicas: { current: 2, desired: 2 },
    gpu: { type: "A100 80GB", count: 2, utilization: 74 },
    vram: { used: "26Gi", total: "32Gi", pct: 81 },
    cpu: "18%",
    mem: "4.2Gi",
    memLimit: "8Gi",
    tps: 1840,
    p50: "48ms",
    p95: "92ms",
    p99: "140ms",
    queueDepth: 3,
    batchSize: "avg 12",
    tokensIn24h: "2.1M",
    tokensOut24h: "1.5M",
    requests24h: 18420,
    errorRate: "0.2%",
    uptime: "14d 6h",
    lastDeployed: "2h ago",
    deployedBy: "A. Kovács",
    namespace: "prod-cx",
    projectColor: "#F59E0B",
    endpoint: "http://llm-serving.prod-cx:8000/v1",
    image: "registry.internal/models/llama-3.1-8b-cx:v4-awq",
    hubRef: "models/llama-3.1-8b-cx/v4",
    connectedAgents: [
      { name: "cx-support-v3", status: "running", rps: 124 },
      { name: "onboarding-bot", status: "running", rps: 5 },
    ],
    servingConfig: {
      maxModelLen: 8192,
      tensorParallelSize: 2,
      maxBatchSize: 32,
      gpuMemoryUtilization: 0.92,
      swapSpace: "4Gi",
      quantization: "awq",
      dtype: "float16",
      enforceEager: false,
      enableChunkedPrefill: true,
      maxNumSeqs: 64,
    },
    generationDefaults: {
      temperature: 0.3,
      topP: 0.9,
      topK: 40,
      maxTokens: 4096,
      repetitionPenalty: 1.05,
      stopSequences: ["</s>", "[END]"],
    },
    fineTunes: [
      { name: "llama-3.1-8b-cx-v4", method: "SFT+DPO", dataset: "cx-convos-v4", date: "3d ago", status: "active", loss: "0.312", hubRef: "models/llama-3.1-8b-cx/v4" },
      { name: "llama-3.1-8b-cx-v3", method: "SFT", dataset: "cx-convos-v3", date: "2w ago", status: "previous", loss: "0.445", hubRef: "models/llama-3.1-8b-cx/v3" },
      { name: "llama-3.1-8b-cx-v2", method: "SFT", dataset: "cx-convos-v2", date: "1mo ago", status: "archived", loss: "0.521", hubRef: "models/llama-3.1-8b-cx/v2" },
    ],
    metricsHistory: {
      tps: [1650, 1700, 1720, 1680, 1750, 1800, 1780, 1820, 1840, 1800, 1830, 1850, 1820, 1840, 1840],
      latency: [140, 135, 150, 130, 138, 128, 142, 132, 125, 136, 130, 122, 128, 135, 130],
      gpu: [68, 70, 72, 71, 74, 76, 73, 75, 78, 74, 76, 72, 74, 73, 74],
      queue: [2, 3, 4, 2, 3, 5, 3, 4, 6, 3, 4, 2, 3, 4, 3],
    },
    events: [
      { time: "2h", text: "Redeployed with v4 fine-tune weights", type: "deploy" },
      { time: "6h", text: "Auto-scaled from 1 to 2 replicas", type: "scale" },
      { time: "1d", text: "VRAM warning: 89% utilization spike", type: "warning" },
      { time: "3d", text: "v4 fine-tune weights published to Hub", type: "hub" },
      { time: "5d", text: "Quantized to AWQ 4-bit from FP16", type: "config" },
    ],
  },
  {
    id: "deepseek-r1-code",
    name: "deepseek-r1-code",
    displayName: "DeepSeek R1 — Code",
    description: "DeepSeek R1 reasoning model specialized for code generation, refactoring, and debugging. Running in FP16 with extended 128K context window for repository-aware code assistance.",
    base: "deepseek-ai/DeepSeek-R1",
    family: "DeepSeek R1",
    paramCount: "70B",
    type: "Base",
    quantization: "FP16",
    contextWindow: 131072,
    status: "serving",
    engine: "vLLM 0.6.4",
    replicas: { current: 1, desired: 1 },
    gpu: { type: "A100 80GB", count: 4, utilization: 82 },
    vram: { used: "108Gi", total: "128Gi", pct: 84 },
    cpu: "24%",
    mem: "12.4Gi",
    memLimit: "16Gi",
    tps: 920,
    p50: "380ms",
    p95: "720ms",
    p99: "890ms",
    queueDepth: 8,
    batchSize: "avg 4",
    tokensIn24h: "4.8M",
    tokensOut24h: "6.2M",
    requests24h: 4120,
    errorRate: "0.3%",
    uptime: "8d 12h",
    lastDeployed: "6h ago",
    deployedBy: "M. Chen",
    namespace: "prod-code",
    projectColor: "#3B82F6",
    endpoint: "http://llm-serving.prod-code:8000/v1",
    image: "registry.internal/models/deepseek-r1:latest-fp16",
    hubRef: "models/deepseek-r1-code/v1",
    connectedAgents: [
      { name: "code-assist-v2", status: "running", rps: 89 },
    ],
    servingConfig: {
      maxModelLen: 131072,
      tensorParallelSize: 4,
      maxBatchSize: 8,
      gpuMemoryUtilization: 0.95,
      swapSpace: "8Gi",
      quantization: "none",
      dtype: "float16",
      enforceEager: false,
      enableChunkedPrefill: true,
      maxNumSeqs: 16,
    },
    generationDefaults: {
      temperature: 0.1,
      topP: 0.95,
      topK: 50,
      maxTokens: 16384,
      repetitionPenalty: 1.0,
      stopSequences: ["</s>"],
    },
    fineTunes: [],
    metricsHistory: {
      tps: [840, 860, 880, 870, 900, 920, 910, 930, 940, 920, 910, 930, 920, 910, 920],
      latency: [890, 880, 920, 860, 880, 850, 870, 860, 840, 860, 850, 840, 870, 880, 850],
      gpu: [78, 80, 82, 79, 83, 85, 81, 84, 86, 82, 84, 80, 82, 81, 82],
      queue: [5, 6, 8, 4, 7, 9, 6, 8, 10, 7, 8, 5, 7, 8, 8],
    },
    events: [
      { time: "6h", text: "Upgraded context window to 128K", type: "config" },
      { time: "2d", text: "Enabled chunked prefill for long contexts", type: "config" },
      { time: "1w", text: "Initial deployment from Hub", type: "deploy" },
    ],
  },
  {
    id: "qwen-2.5-72b",
    name: "qwen-2.5-72b",
    displayName: "Qwen 2.5 72B",
    description: "Qwen 2.5 72B for general-purpose reasoning and data analysis. Currently used by Data Analyst Agent for SQL generation and chart interpretation. GPTQ 4-bit quantized.",
    base: "Qwen/Qwen2.5-72B-Instruct",
    family: "Qwen 2.5",
    paramCount: "72B",
    type: "Base",
    quantization: "GPTQ 4-bit",
    contextWindow: 32768,
    status: "serving",
    engine: "vLLM 0.6.4",
    replicas: { current: 1, desired: 1 },
    gpu: { type: "H100 80GB", count: 4, utilization: 45 },
    vram: { used: "112Gi", total: "160Gi", pct: 70 },
    cpu: "12%",
    mem: "8.1Gi",
    memLimit: "16Gi",
    tps: 1100,
    p50: "280ms",
    p95: "580ms",
    p99: "720ms",
    queueDepth: 1,
    batchSize: "avg 6",
    tokensIn24h: "320K",
    tokensOut24h: "480K",
    requests24h: 640,
    errorRate: "0.8%",
    uptime: "22d 4h",
    lastDeployed: "18m ago",
    deployedBy: "R. Silva",
    namespace: "staging-da",
    projectColor: "#8B5CF6",
    endpoint: "http://llm-serving.staging-da:8000/v1",
    image: "registry.internal/models/qwen-2.5-72b:gptq",
    hubRef: "models/qwen-2.5-72b/v1",
    connectedAgents: [
      { name: "data-analyst-v1", status: "deploying", rps: 12 },
    ],
    servingConfig: {
      maxModelLen: 32768,
      tensorParallelSize: 4,
      maxBatchSize: 16,
      gpuMemoryUtilization: 0.90,
      swapSpace: "4Gi",
      quantization: "gptq",
      dtype: "float16",
      enforceEager: false,
      enableChunkedPrefill: true,
      maxNumSeqs: 32,
    },
    generationDefaults: {
      temperature: 0.0,
      topP: 1.0,
      topK: -1,
      maxTokens: 8192,
      repetitionPenalty: 1.0,
      stopSequences: ["<|im_end|>"],
    },
    fineTunes: [],
    metricsHistory: {
      tps: [1000, 1020, 1050, 1040, 1080, 1100, 1090, 1110, 1120, 1100, 1090, 1110, 1100, 1090, 1100],
      latency: [720, 710, 740, 690, 710, 700, 720, 700, 680, 710, 700, 690, 710, 720, 710],
      gpu: [40, 42, 45, 43, 46, 48, 44, 47, 50, 45, 47, 43, 45, 44, 45],
      queue: [0, 1, 1, 0, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1],
    },
    events: [
      { time: "18m", text: "Redeployed for data-analyst-v1 RC2", type: "deploy" },
      { time: "3d", text: "Switched from AWQ to GPTQ quantization", type: "config" },
    ],
  },
  {
    id: "guard-3b",
    name: "guard-3b",
    displayName: "LlamaGuard 3B",
    description: "Safety classification model for content moderation. Designed to run alongside production agents for real-time output filtering. Currently in error state due to OOM.",
    base: "meta-llama/Llama-Guard-3-1B",
    family: "Llama Guard",
    paramCount: "3B",
    type: "Fine-tuned",
    quantization: "FP16",
    contextWindow: 2048,
    status: "error",
    engine: "vLLM 0.6.4",
    replicas: { current: 0, desired: 1 },
    gpu: { type: "A100 80GB", count: 1, utilization: 0 },
    vram: { used: "0Gi", total: "8Gi", pct: 0 },
    cpu: "0%",
    mem: "0Gi",
    memLimit: "4Gi",
    tps: 0,
    p50: "\u2014",
    p95: "\u2014",
    p99: "\u2014",
    queueDepth: 0,
    batchSize: "\u2014",
    tokensIn24h: "0",
    tokensOut24h: "0",
    requests24h: 0,
    errorRate: "\u2014",
    uptime: "\u2014",
    lastDeployed: "1h ago",
    deployedBy: "A. Kovács",
    namespace: "staging-da",
    projectColor: "#8B5CF6",
    endpoint: "\u2014",
    image: "registry.internal/models/guard-3b:v2-fp16",
    hubRef: "models/guard-3b/v2",
    connectedAgents: [
      { name: "safety-reviewer", status: "error", rps: 0 },
    ],
    servingConfig: {
      maxModelLen: 2048,
      tensorParallelSize: 1,
      maxBatchSize: 64,
      gpuMemoryUtilization: 0.90,
      swapSpace: "2Gi",
      quantization: "none",
      dtype: "float16",
      enforceEager: true,
      enableChunkedPrefill: false,
      maxNumSeqs: 128,
    },
    generationDefaults: {
      temperature: 0.0,
      topP: 1.0,
      topK: -1,
      maxTokens: 512,
      repetitionPenalty: 1.0,
      stopSequences: [],
    },
    fineTunes: [
      { name: "guard-3b-v2", method: "SFT", dataset: "safety-labels-v2", date: "2d ago", status: "active", loss: "0.312", hubRef: "models/guard-3b/v2" },
      { name: "guard-3b-v1", method: "SFT", dataset: "safety-labels-v1", date: "1w ago", status: "previous", loss: "0.489", hubRef: "models/guard-3b/v1" },
    ],
    metricsHistory: {
      tps: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      latency: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      gpu: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      queue: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    events: [
      { time: "1h", text: "OOM killed \u2014 needs 16Gi VRAM, allocated 8Gi", type: "error" },
      { time: "1h", text: "Restart attempt #3 failed", type: "error" },
      { time: "2h", text: "v2 fine-tune deployed (OOM fix attempt)", type: "deploy" },
      { time: "1d", text: "Initial deploy \u2014 OOM on first request", type: "error" },
    ],
  },
  {
    id: "llama-3.1-70b-base",
    name: "llama-3.1-70b-base",
    displayName: "Llama 3.1 70B Instruct",
    description: "General-purpose Llama 3.1 70B base model. Not currently serving \u2014 available in Hub for fine-tuning and experimentation.",
    base: "meta-llama/Llama-3.1-70B-Instruct",
    family: "Llama 3.1",
    paramCount: "70B",
    type: "Base",
    quantization: "\u2014",
    contextWindow: 131072,
    status: "stopped",
    engine: "\u2014",
    replicas: { current: 0, desired: 0 },
    gpu: { type: "\u2014", count: 0, utilization: 0 },
    vram: { used: "0Gi", total: "0Gi", pct: 0 },
    cpu: "0%",
    mem: "0Gi",
    memLimit: "\u2014",
    tps: 0,
    p50: "\u2014",
    p95: "\u2014",
    p99: "\u2014",
    queueDepth: 0,
    batchSize: "\u2014",
    tokensIn24h: "0",
    tokensOut24h: "0",
    requests24h: 0,
    errorRate: "\u2014",
    uptime: "\u2014",
    lastDeployed: "\u2014",
    deployedBy: "\u2014",
    namespace: "\u2014",
    projectColor: "#6B7280",
    endpoint: "\u2014",
    image: "\u2014",
    hubRef: "models/llama-3.1-70b-instruct/base",
    connectedAgents: [],
    servingConfig: null,
    generationDefaults: null,
    fineTunes: [],
    metricsHistory: {
      tps: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      latency: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      gpu: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      queue: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    events: [],
  },
  {
    id: "mistral-7b-exp",
    name: "mistral-7b-exp",
    displayName: "Mistral 7B \u2014 Experiment",
    description: "Experimental Mistral 7B deployment for A/B testing against Llama 3.1 8B in CX workflows. Currently paused pending eval results.",
    base: "mistralai/Mistral-7B-Instruct-v0.3",
    family: "Mistral",
    paramCount: "7B",
    type: "Base",
    quantization: "AWQ 4-bit",
    contextWindow: 32768,
    status: "stopped",
    engine: "vLLM 0.6.4",
    replicas: { current: 0, desired: 0 },
    gpu: { type: "A100 80GB", count: 1, utilization: 0 },
    vram: { used: "0Gi", total: "0Gi", pct: 0 },
    cpu: "0%",
    mem: "0Gi",
    memLimit: "4Gi",
    tps: 0,
    p50: "\u2014",
    p95: "\u2014",
    p99: "\u2014",
    queueDepth: 0,
    batchSize: "\u2014",
    tokensIn24h: "0",
    tokensOut24h: "0",
    requests24h: 0,
    errorRate: "\u2014",
    uptime: "\u2014",
    lastDeployed: "5d ago",
    deployedBy: "A. Kovács",
    namespace: "staging-da",
    projectColor: "#8B5CF6",
    endpoint: "\u2014",
    image: "registry.internal/models/mistral-7b:awq",
    hubRef: "models/mistral-7b-exp/v1",
    connectedAgents: [],
    servingConfig: null,
    generationDefaults: null,
    fineTunes: [],
    metricsHistory: {
      tps: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      latency: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      gpu: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      queue: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    events: [
      { time: "5d", text: "Stopped \u2014 pending GSM8K eval comparison", type: "config" },
      { time: "1w", text: "Deployed for A/B test vs llama-3.1-8b-cx", type: "deploy" },
    ],
  },
];
