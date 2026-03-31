// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export const TIME_RANGES = [
  { id: "5m", label: "5m" },
  { id: "15m", label: "15m" },
  { id: "1h", label: "1h" },
  { id: "6h", label: "6h" },
  { id: "24h", label: "24h" },
  { id: "7d", label: "7d" },
] as const;

export type TimeRangeId = (typeof TIME_RANGES)[number]["id"];

export interface AgentHealth {
  name: string;
  status: "healthy" | "degraded" | "down";
  rps: number;
  p50: number;
  p95: number;
  p99: number;
  errorRate: number;
  tokens24h: string;
  sparkRps: number[];
  sparkLatency: number[];
  sparkErrors: number[];
  color: string;
  namespace: string;
  replicas: string;
  cpu: number;
  mem: number;
}

export const AGENTS_HEALTH: AgentHealth[] = [
  {
    name: "cx-support-v3", status: "healthy", rps: 124, p50: 180, p95: 280, p99: 320, errorRate: 0.8, tokens24h: "2.1M",
    sparkRps: [98, 105, 112, 108, 115, 120, 118, 124, 130, 128, 126, 124, 122, 119, 124, 128, 130, 126, 122, 124],
    sparkLatency: [320, 310, 340, 290, 300, 280, 310, 290, 270, 300, 280, 260, 270, 290, 280, 300, 310, 290, 280, 270],
    sparkErrors: [1.2, 0.9, 1.1, 0.8, 0.7, 1.0, 0.6, 0.8, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.8, 0.7, 0.5, 0.8, 0.9, 0.8],
    color: "#F59E0B", namespace: "prod-cx", replicas: "3/3", cpu: 42, mem: 45,
  },
  {
    name: "code-assist-v2", status: "healthy", rps: 89, p50: 420, p95: 780, p99: 890, errorRate: 0.3, tokens24h: "11.0M",
    sparkRps: [72, 78, 82, 80, 85, 88, 84, 89, 92, 90, 87, 89, 91, 88, 89, 92, 90, 88, 86, 89],
    sparkLatency: [890, 870, 920, 850, 880, 840, 870, 860, 830, 860, 850, 840, 870, 880, 850, 860, 840, 870, 880, 860],
    sparkErrors: [0.5, 0.4, 0.3, 0.6, 0.3, 0.2, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 0.3, 0.4, 0.2, 0.3, 0.3, 0.3],
    color: "#3B82F6", namespace: "prod-code", replicas: "2/2", cpu: 67, mem: 40,
  },
  {
    name: "data-analyst-v1", status: "degraded", rps: 12, p50: 650, p95: 1100, p99: 1200, errorRate: 2.1, tokens24h: "800K",
    sparkRps: [8, 10, 9, 11, 10, 12, 11, 13, 12, 14, 12, 11, 12, 13, 12, 11, 10, 12, 13, 12],
    sparkLatency: [1200, 1150, 1300, 1100, 1200, 1180, 1250, 1200, 1150, 1180, 1200, 1220, 1190, 1200, 1180, 1200, 1250, 1190, 1210, 1200],
    sparkErrors: [3.0, 2.5, 2.8, 2.2, 2.0, 2.5, 2.1, 1.8, 2.2, 2.0, 1.9, 2.1, 2.3, 2.0, 2.1, 2.4, 2.2, 2.0, 1.9, 2.1],
    color: "#8B5CF6", namespace: "staging-da", replicas: "1/2", cpu: 23, mem: 22,
  },
  {
    name: "onboarding-bot", status: "healthy", rps: 5, p50: 140, p95: 190, p99: 210, errorRate: 0.1, tokens24h: "107K",
    sparkRps: [3, 4, 5, 4, 6, 5, 4, 5, 6, 5, 4, 5, 5, 4, 5, 6, 5, 4, 5, 5],
    sparkLatency: [210, 200, 220, 190, 200, 210, 195, 205, 210, 200, 195, 210, 205, 200, 210, 200, 190, 205, 210, 200],
    sparkErrors: [0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1],
    color: "#22C55E", namespace: "prod-cx", replicas: "1/1", cpu: 8, mem: 10,
  },
  {
    name: "safety-reviewer", status: "down", rps: 0, p50: 0, p95: 0, p99: 0, errorRate: 100, tokens24h: "0",
    sparkRps: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    sparkLatency: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    sparkErrors: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    color: "#EF4444", namespace: "staging-da", replicas: "0/1", cpu: 0, mem: 0,
  },
];

export interface RpsDataPoint {
  t: number;
  cx: number;
  code: number;
  da: number;
  ob: number;
}

export const COMBINED_RPS: RpsDataPoint[] = Array.from({ length: 30 }, (_, i) => ({
  t: i,
  cx: 98 + Math.floor(Math.random() * 30),
  code: 72 + Math.floor(Math.random() * 20),
  da: 8 + Math.floor(Math.random() * 6),
  ob: 3 + Math.floor(Math.random() * 4),
}));

export const LATENCY_BUCKETS = ["0-50", "50-100", "100-200", "200-500", "500-1s", "1-2s", "2-5s", "5s+"];
const HEATMAP_COLS = 24;

export const LATENCY_HEATMAP: number[][] = Array.from({ length: LATENCY_BUCKETS.length }, (_, r) =>
  Array.from({ length: HEATMAP_COLS }, () => {
    const base = r <= 2 ? 0.5 + Math.random() * 0.5 : r <= 4 ? 0.2 + Math.random() * 0.4 : r <= 5 ? Math.random() * 0.2 : Math.random() * 0.08;
    return Math.round(base * 100) / 100;
  }),
);

export type Severity = "critical" | "warning" | "info";

export interface Alert {
  id: string;
  severity: Severity;
  title: string;
  message: string;
  time: string;
  acknowledged: boolean;
}

export const ALERTS: Alert[] = [
  { id: "a-001", severity: "critical", title: "safety-reviewer OOM crash loop", message: "Pod restarting — 3 failures in 10m. Needs VRAM increase to 16Gi.", time: "12m ago", acknowledged: false },
  { id: "a-002", severity: "warning", title: "data-analyst-v1 high error rate", message: "Error rate at 2.1% — above 1.5% threshold. SQL generation failures.", time: "18m ago", acknowledged: false },
  { id: "a-003", severity: "warning", title: "GPU utilization spike on gpu-04", message: "GPU util reached 96% for 5m. deepseek-r1 queue depth growing.", time: "32m ago", acknowledged: true },
  { id: "a-004", severity: "info", title: "data-analyst-v1 auto-scaling", message: "Auto-scaler triggered by queue depth > 5. Deploying second replica.", time: "45m ago", acknowledged: true },
  { id: "a-005", severity: "info", title: "CX SFT Round 4 checkpoint", message: "Epoch 2/3 completed. Loss: 0.847. Checkpoint saved to Hub.", time: "1h ago", acknowledged: true },
];

export interface ErrorLogEntry {
  time: string;
  agent: string;
  type: string;
  message: string;
  traceId: string;
  status: number;
}

export const ERROR_LOG: ErrorLogEntry[] = [
  { time: "2m ago", agent: "data-analyst-v1", type: "SQL_ERROR", message: "GROUP BY clause mismatch on aggregation query", traceId: "tr-8a2f", status: 500 },
  { time: "5m ago", agent: "cx-support-v3", type: "TOOL_TIMEOUT", message: "order-lookup timed out after 5000ms — CRM unreachable", traceId: "tr-7b3e", status: 504 },
  { time: "8m ago", agent: "data-analyst-v1", type: "SQL_ERROR", message: "Invalid column reference 'revenue_ytd' — not in schema", traceId: "tr-6c4d", status: 500 },
  { time: "14m ago", agent: "cx-support-v3", type: "GUARDRAIL", message: "sentiment-guard blocked response — empathy score 0.42", traceId: "tr-5d5c", status: 200 },
  { time: "22m ago", agent: "code-assist-v2", type: "CTX_OVERFLOW", message: "Input exceeded 128K context window — truncated", traceId: "tr-4e6b", status: 200 },
  { time: "31m ago", agent: "data-analyst-v1", type: "SQL_ERROR", message: "Ambiguous column name 'date' in multi-table join", traceId: "tr-3f7a", status: 500 },
  { time: "45m ago", agent: "cx-support-v3", type: "RATE_LIMIT", message: "subscription-manager Stripe API rate limited", traceId: "tr-2g89", status: 429 },
];

export interface TokenUsageEntry {
  agent: string;
  input: number;
  output: number;
  total: number;
  color: string;
  pctOfTotal: number;
}

export const TOKEN_USAGE: TokenUsageEntry[] = [
  { agent: "code-assist-v2", input: 4800, output: 6200, total: 11000, color: "#3B82F6", pctOfTotal: 78.6 },
  { agent: "cx-support-v3", input: 1200, output: 890, total: 2090, color: "#F59E0B", pctOfTotal: 14.9 },
  { agent: "data-analyst-v1", input: 320, output: 480, total: 800, color: "#8B5CF6", pctOfTotal: 5.7 },
  { agent: "onboarding-bot", input: 45, output: 62, total: 107, color: "#22C55E", pctOfTotal: 0.8 },
];

export interface ModelHealth {
  name: string;
  status: "serving" | "error";
  gpu: number;
  vram: number;
  tps: number;
  p99: string;
}

export const MODEL_HEALTH: ModelHealth[] = [
  { name: "llama-3.1-8b-cx", status: "serving", gpu: 74, vram: 81, tps: 1840, p99: "140ms" },
  { name: "deepseek-r1-code", status: "serving", gpu: 82, vram: 84, tps: 920, p99: "890ms" },
  { name: "qwen-2.5-72b", status: "serving", gpu: 45, vram: 70, tps: 1100, p99: "720ms" },
  { name: "guard-3b", status: "error", gpu: 0, vram: 0, tps: 0, p99: "—" },
];

export interface LiveFeedEntry {
  time: string;
  type: "request" | "tool" | "error" | "guardrail";
  agent: string;
  detail: string;
  color: string;
}

export const LIVE_FEED: LiveFeedEntry[] = [
  { time: "now", type: "request", agent: "cx-support-v3", detail: "POST /chat — 180ms — 342 tok", color: "#22C55E" },
  { time: "1s", type: "request", agent: "code-assist-v2", detail: "POST /chat — 1.2s — 2,140 tok", color: "#3B82F6" },
  { time: "2s", type: "request", agent: "cx-support-v3", detail: "POST /chat — 220ms — 418 tok", color: "#22C55E" },
  { time: "3s", type: "tool", agent: "cx-support-v3", detail: "order-lookup invoked — 42ms", color: "#F59E0B" },
  { time: "3s", type: "request", agent: "onboarding-bot", detail: "POST /chat — 140ms — 290 tok", color: "#22C55E" },
  { time: "5s", type: "error", agent: "data-analyst-v1", detail: "SQL_ERROR — GROUP BY mismatch", color: "#EF4444" },
  { time: "6s", type: "request", agent: "code-assist-v2", detail: "POST /chat — 890ms — 1,820 tok", color: "#3B82F6" },
  { time: "7s", type: "request", agent: "cx-support-v3", detail: "POST /chat — 190ms — 510 tok", color: "#22C55E" },
  { time: "8s", type: "guardrail", agent: "cx-support-v3", detail: "sentiment-guard PASS — score 0.91", color: "#22C55E" },
  { time: "9s", type: "request", agent: "cx-support-v3", detail: "POST /chat — 160ms — 380 tok", color: "#22C55E" },
];

export interface EndpointEntry {
  path: string;
  method: string;
  agents: number;
  status: "up" | "degraded";
  avgMs: number;
}

export const ENDPOINTS: EndpointEntry[] = [
  { path: "/v1/chat", method: "POST", agents: 4, status: "up", avgMs: 284 },
  { path: "/v1/completions", method: "POST", agents: 2, status: "up", avgMs: 420 },
  { path: "/health", method: "GET", agents: 5, status: "degraded", avgMs: 8 },
  { path: "/metrics", method: "GET", agents: 5, status: "up", avgMs: 3 },
];

export const SEVERITY_COLORS: Record<Severity, string> = {
  critical: "#EF4444",
  warning: "#F59E0B",
  info: "#3B82F6",
};

export const RPS_CHART_LEGEND = [
  { label: "cx-support", key: "cx" as const, color: "#F59E0B" },
  { label: "code-assist", key: "code" as const, color: "#3B82F6" },
  { label: "data-analyst", key: "da" as const, color: "#8B5CF6" },
  { label: "onboarding", key: "ob" as const, color: "#22C55E" },
];
