// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Status } from "@/components/ui/status-dot";

// ── Types ───────────────────────────────────────────────────────

export interface AgentReplicas {
  current: number;
  desired: number;
}

export interface AgentResources {
  cpuReq: string;
  cpuLim: string;
  memReq: string;
  memLim: string;
}

export interface AgentSkill {
  name: string;
  version: string;
  type: "tool" | "rag" | "workflow" | "guardrail";
  status: string;
}

export interface AgentMcpServer {
  name: string;
  status: string;
  latency: string;
}

export interface AgentVersion {
  version: string;
  date: string;
  author: string;
  change: string;
  status: string;
  hash: string;
}

export interface AgentEnvVar {
  key: string;
  value: string;
}

export interface AgentMetricsHistory {
  rps: number[];
  latency: number[];
  errors: number[];
  tokens: number[];
}

export interface Agent {
  id: string;
  name: string;
  displayName: string;
  description: string;
  version: string;
  status: string;
  replicas: AgentReplicas;
  projectId: string;
  model: string;
  modelBase: string;
  createdBy: string;
  createdAt: string;
  lastDeployed: string;
  endpoint: string;
  cpu: string;
  mem: string;
  memLimit: string;
  gpu: string;
  rps: number;
  p50: string;
  p95: string;
  p99: string;
  errorRate: string;
  tokensIn24h: string;
  tokensOut24h: string;
  conversations24h: number;
  avgTurns: number;
  satisfaction: string;
  image: string;
  resources: AgentResources;
  skills: AgentSkill[];
  mcpServers: AgentMcpServer[];
  env: AgentEnvVar[];
  versions: AgentVersion[];
  metricsHistory: AgentMetricsHistory;
}

// ── Status mapping ──────────────────────────────────────────────

const STATUS_MAP: Record<string, Status> = {
  active: "running",
  connected: "serving",
  running: "running",
  serving: "serving",
  deploying: "deploying",
  error: "error",
  stopped: "stopped",
};

export function toStatus(raw: string): Status {
  return STATUS_MAP[raw] ?? "stopped";
}

// ── Skill type styles ───────────────────────────────────────────

export const SKILL_TYPE_STYLES: Record<string, { bg: string; fg: string }> = {
  tool: { bg: "bg-blue-500/10", fg: "text-blue-500" },
  rag: { bg: "bg-green-500/10", fg: "text-green-500" },
  workflow: { bg: "bg-amber-500/10", fg: "text-amber-500" },
  guardrail: { bg: "bg-red-500/10", fg: "text-red-500" },
};

// ── Demo data ───────────────────────────────────────────────────

export const AGENTS: Agent[] = [
  {
    id: "cx-support-v3",
    name: "cx-support-v3",
    displayName: "CX Support Agent",
    description: "Customer support agent handling subscription management, billing inquiries, and general support. Uses RAG over knowledge base and order-lookup skill.",
    version: "3.2.1",
    status: "running",
    replicas: { current: 3, desired: 3 },
    projectId: "cx-support-agent",
    model: "llama-3.1-8b-cx",
    modelBase: "Llama 3.1 8B",
    createdBy: "A. Kovács",
    createdAt: "2025-09-14",
    lastDeployed: "2h ago",
    endpoint: "https://agents.internal/cx-support-v3",
    cpu: "42%",
    mem: "1.8Gi",
    memLimit: "4Gi",
    gpu: "\u2014",
    rps: 124,
    p50: "180ms",
    p95: "280ms",
    p99: "320ms",
    errorRate: "0.8%",
    tokensIn24h: "1.2M",
    tokensOut24h: "890K",
    conversations24h: 1847,
    avgTurns: 6.2,
    satisfaction: "94%",
    image: "registry.internal/agents/cx-support:3.2.1",
    resources: { cpuReq: "500m", cpuLim: "2000m", memReq: "1Gi", memLim: "4Gi" },
    skills: [
      { name: "order-lookup", version: "2.1.0", type: "tool", status: "active" },
      { name: "subscription-manager", version: "1.4.2", type: "tool", status: "active" },
      { name: "kb-search", version: "3.0.1", type: "rag", status: "active" },
      { name: "escalation-router", version: "1.0.0", type: "workflow", status: "active" },
      { name: "sentiment-guard", version: "0.9.3", type: "guardrail", status: "active" },
    ],
    mcpServers: [
      { name: "crm-connector", status: "connected", latency: "45ms" },
      { name: "billing-api", status: "connected", latency: "120ms" },
    ],
    env: [
      { key: "MAX_TOKENS", value: "4096" },
      { key: "TEMPERATURE", value: "0.3" },
      { key: "SYSTEM_PROMPT_VERSION", value: "v14" },
      { key: "RAG_TOP_K", value: "5" },
      { key: "GUARDRAIL_THRESHOLD", value: "0.85" },
    ],
    versions: [
      { version: "3.2.1", date: "2h ago", author: "A. Kovács", change: "Tuned escalation threshold to 0.85", status: "active", hash: "a3f8c21" },
      { version: "3.2.0", date: "2d ago", author: "M. Chen", change: "Added sentiment-guard skill", status: "previous", hash: "b7e4d09" },
      { version: "3.1.4", date: "1w ago", author: "A. Kovács", change: "Updated system prompt v14", status: "previous", hash: "c1a9f33" },
      { version: "3.1.3", date: "2w ago", author: "R. Silva", change: "Switched to llama-3.1-8b-cx fine-tune", status: "previous", hash: "d5b2e77" },
      { version: "3.0.0", date: "1mo ago", author: "A. Kovács", change: "Major: multi-turn conversation rewrite", status: "archived", hash: "e8c1a44" },
    ],
    metricsHistory: {
      rps: [98, 105, 112, 108, 115, 120, 118, 124, 130, 128, 126, 124, 122, 119, 124],
      latency: [320, 310, 340, 290, 300, 280, 310, 290, 270, 300, 280, 260, 270, 290, 280],
      errors: [1.2, 0.9, 1.1, 0.8, 0.7, 1.0, 0.6, 0.8, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.8],
      tokens: [800, 850, 920, 880, 950, 1000, 1050, 980, 1100, 1150, 1080, 1200, 1180, 1220, 1200],
    },
  },
  {
    id: "code-assist-v2",
    name: "code-assist-v2",
    displayName: "Code Assistant",
    description: "Developer productivity agent for code generation, refactoring, debugging, and documentation. Supports multi-file context with repository-aware RAG.",
    version: "2.7.0",
    status: "running",
    replicas: { current: 2, desired: 2 },
    projectId: "code-assistant",
    model: "deepseek-r1-code",
    modelBase: "DeepSeek R1",
    createdBy: "M. Chen",
    createdAt: "2025-11-02",
    lastDeployed: "6h ago",
    endpoint: "https://agents.internal/code-assist-v2",
    cpu: "67%",
    mem: "3.2Gi",
    memLimit: "8Gi",
    gpu: "\u2014",
    rps: 89,
    p50: "420ms",
    p95: "780ms",
    p99: "890ms",
    errorRate: "0.3%",
    tokensIn24h: "4.8M",
    tokensOut24h: "6.2M",
    conversations24h: 412,
    avgTurns: 18.4,
    satisfaction: "97%",
    image: "registry.internal/agents/code-assist:2.7.0",
    resources: { cpuReq: "1000m", cpuLim: "4000m", memReq: "2Gi", memLim: "8Gi" },
    skills: [
      { name: "repo-indexer", version: "2.0.0", type: "rag", status: "active" },
      { name: "code-executor", version: "1.2.1", type: "tool", status: "active" },
      { name: "lsp-bridge", version: "0.8.0", type: "tool", status: "active" },
      { name: "doc-generator", version: "1.1.0", type: "tool", status: "active" },
    ],
    mcpServers: [
      { name: "github-connector", status: "connected", latency: "85ms" },
      { name: "jira-connector", status: "connected", latency: "110ms" },
    ],
    env: [
      { key: "MAX_TOKENS", value: "16384" },
      { key: "TEMPERATURE", value: "0.1" },
      { key: "CONTEXT_WINDOW", value: "128000" },
      { key: "CODE_EXEC_TIMEOUT", value: "30s" },
    ],
    versions: [
      { version: "2.7.0", date: "6h ago", author: "M. Chen", change: "Upgraded context window to 128k", status: "active", hash: "f2a8b11" },
      { version: "2.6.3", date: "3d ago", author: "M. Chen", change: "Fixed multi-file diff generation", status: "previous", hash: "g4c9d22" },
      { version: "2.6.0", date: "1w ago", author: "L. Park", change: "Added LSP bridge for type checking", status: "previous", hash: "h6e1f33" },
    ],
    metricsHistory: {
      rps: [72, 78, 82, 80, 85, 88, 84, 89, 92, 90, 87, 89, 91, 88, 89],
      latency: [890, 870, 920, 850, 880, 840, 870, 860, 830, 860, 850, 840, 870, 880, 850],
      errors: [0.5, 0.4, 0.3, 0.6, 0.3, 0.2, 0.4, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 0.3],
      tokens: [3200, 3400, 3600, 3500, 3800, 4000, 4200, 4100, 4400, 4600, 4500, 4800, 5000, 4900, 4800],
    },
  },
  {
    id: "data-analyst-v1",
    name: "data-analyst-v1",
    displayName: "Data Analyst Agent",
    description: "Natural language to SQL agent with chart generation. Connects to data warehouse and produces executive-ready visualizations.",
    version: "1.4.0-rc2",
    status: "deploying",
    replicas: { current: 1, desired: 2 },
    projectId: "data-analyst-agent",
    model: "qwen-2.5-72b",
    modelBase: "Qwen 2.5 72B",
    createdBy: "R. Silva",
    createdAt: "2026-01-18",
    lastDeployed: "18m ago",
    endpoint: "https://agents.internal/data-analyst-v1",
    cpu: "23%",
    mem: "0.9Gi",
    memLimit: "4Gi",
    gpu: "\u2014",
    rps: 12,
    p50: "650ms",
    p95: "1.1s",
    p99: "1.2s",
    errorRate: "2.1%",
    tokensIn24h: "320K",
    tokensOut24h: "480K",
    conversations24h: 64,
    avgTurns: 5.8,
    satisfaction: "88%",
    image: "registry.internal/agents/data-analyst:1.4.0-rc2",
    resources: { cpuReq: "500m", cpuLim: "2000m", memReq: "1Gi", memLim: "4Gi" },
    skills: [
      { name: "sql-generator", version: "1.3.0", type: "tool", status: "active" },
      { name: "chart-renderer", version: "0.9.1", type: "tool", status: "active" },
      { name: "schema-introspect", version: "1.0.0", type: "rag", status: "active" },
    ],
    mcpServers: [
      { name: "warehouse-connector", status: "connected", latency: "200ms" },
    ],
    env: [
      { key: "MAX_TOKENS", value: "8192" },
      { key: "TEMPERATURE", value: "0.0" },
      { key: "SQL_DIALECT", value: "postgresql" },
    ],
    versions: [
      { version: "1.4.0-rc2", date: "18m ago", author: "R. Silva", change: "RC2: fixed GROUP BY regression", status: "deploying", hash: "j8g3h55" },
      { version: "1.3.2", date: "4d ago", author: "R. Silva", change: "Improved chart type selection", status: "previous", hash: "k0i5j66" },
    ],
    metricsHistory: {
      rps: [8, 10, 9, 11, 10, 12, 11, 13, 12, 14, 12, 11, 12, 13, 12],
      latency: [1200, 1150, 1300, 1100, 1200, 1180, 1250, 1200, 1150, 1180, 1200, 1220, 1190, 1200, 1180],
      errors: [3.0, 2.5, 2.8, 2.2, 2.0, 2.5, 2.1, 1.8, 2.2, 2.0, 1.9, 2.1, 2.3, 2.0, 2.1],
      tokens: [200, 220, 250, 240, 280, 300, 290, 310, 320, 310, 300, 320, 330, 310, 320],
    },
  },
  {
    id: "onboarding-bot",
    name: "onboarding-bot",
    displayName: "Onboarding Bot",
    description: "Guides new employees through company onboarding. Provides policy information, IT setup assistance, and benefits enrollment.",
    version: "1.0.3",
    status: "running",
    replicas: { current: 1, desired: 1 },
    projectId: "cx-support-agent",
    model: "llama-3.1-8b-cx",
    modelBase: "Llama 3.1 8B",
    createdBy: "L. Park",
    createdAt: "2026-02-01",
    lastDeployed: "3d ago",
    endpoint: "https://agents.internal/onboarding-bot",
    cpu: "8%",
    mem: "0.4Gi",
    memLimit: "2Gi",
    gpu: "\u2014",
    rps: 5,
    p50: "140ms",
    p95: "190ms",
    p99: "210ms",
    errorRate: "0.1%",
    tokensIn24h: "45K",
    tokensOut24h: "62K",
    conversations24h: 18,
    avgTurns: 8.1,
    satisfaction: "96%",
    image: "registry.internal/agents/onboarding-bot:1.0.3",
    resources: { cpuReq: "250m", cpuLim: "1000m", memReq: "512Mi", memLim: "2Gi" },
    skills: [
      { name: "policy-search", version: "1.0.0", type: "rag", status: "active" },
      { name: "it-ticket-create", version: "1.1.0", type: "tool", status: "active" },
    ],
    mcpServers: [],
    env: [
      { key: "MAX_TOKENS", value: "4096" },
      { key: "TEMPERATURE", value: "0.4" },
    ],
    versions: [
      { version: "1.0.3", date: "3d ago", author: "L. Park", change: "Updated benefits FAQ for 2026", status: "active", hash: "m2k7l88" },
      { version: "1.0.2", date: "2w ago", author: "L. Park", change: "Added IT setup checklist skill", status: "previous", hash: "n4m9n99" },
    ],
    metricsHistory: {
      rps: [3, 4, 5, 4, 6, 5, 4, 5, 6, 5, 4, 5, 5, 4, 5],
      latency: [210, 200, 220, 190, 200, 210, 195, 205, 210, 200, 195, 210, 205, 200, 210],
      errors: [0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, 0.1, 0.1],
      tokens: [30, 35, 40, 38, 42, 45, 43, 48, 45, 42, 40, 45, 48, 44, 45],
    },
  },
  {
    id: "safety-reviewer",
    name: "safety-reviewer",
    displayName: "Safety Reviewer",
    description: "Content moderation and safety classification agent. Reviews outputs from other agents for policy compliance.",
    version: "0.9.1",
    status: "error",
    replicas: { current: 0, desired: 1 },
    projectId: "data-analyst-agent",
    model: "guard-3b",
    modelBase: "LlamaGuard 3B",
    createdBy: "A. Kovács",
    createdAt: "2026-02-20",
    lastDeployed: "1h ago",
    endpoint: "\u2014",
    cpu: "0%",
    mem: "0Gi",
    memLimit: "2Gi",
    gpu: "\u2014",
    rps: 0,
    p50: "\u2014",
    p95: "\u2014",
    p99: "\u2014",
    errorRate: "\u2014",
    tokensIn24h: "0",
    tokensOut24h: "0",
    conversations24h: 0,
    avgTurns: 0,
    satisfaction: "\u2014",
    image: "registry.internal/agents/safety-reviewer:0.9.1",
    resources: { cpuReq: "250m", cpuLim: "1000m", memReq: "512Mi", memLim: "2Gi" },
    skills: [
      { name: "toxicity-classifier", version: "1.0.0", type: "guardrail", status: "error" },
      { name: "pii-detector", version: "0.8.0", type: "guardrail", status: "error" },
    ],
    mcpServers: [],
    env: [
      { key: "MAX_TOKENS", value: "512" },
      { key: "THRESHOLD", value: "0.9" },
    ],
    versions: [
      { version: "0.9.1", date: "1h ago", author: "A. Kovács", change: "Attempted OOM fix \u2014 still failing", status: "error", hash: "p6o1q00" },
      { version: "0.9.0", date: "1d ago", author: "A. Kovács", change: "Initial deployment", status: "previous", hash: "q8r3s11" },
    ],
    metricsHistory: {
      rps: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      latency: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      errors: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      tokens: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
  },
];
