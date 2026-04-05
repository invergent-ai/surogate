// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export interface Preset {
  id: string;
  name: string;
  system: string;
  temp: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repPenalty: number;
}

export interface Session {
  id: string;
  name: string;
  model: string;
  time: string;
  turns: number;
}

export interface PlaygroundParams {
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repPenalty: number;
}

export const DEFAULT_COLOR = "#3B82F6";

export const PRESETS: Preset[] = [
  {
    id: "default",
    name: "Default",
    system: "You are a helpful assistant.",
    temp: 0.7,
    topP: 0.9,
    topK: 40,
    maxTokens: 2048,
    repPenalty: 1.0,
  },
  {
    id: "creative",
    name: "Creative",
    system:
      "You are a creative writing assistant. Be imaginative, vivid, and bold in your language.",
    temp: 1.0,
    topP: 0.95,
    topK: 80,
    maxTokens: 4096,
    repPenalty: 1.1,
  },
  {
    id: "precise",
    name: "Precise",
    system:
      "You are a precise, factual assistant. Only respond with verified information. Be concise.",
    temp: 0.1,
    topP: 0.5,
    topK: 10,
    maxTokens: 1024,
    repPenalty: 1.0,
  },
  {
    id: "code",
    name: "Code",
    system:
      "You are an expert programmer. Write clean, efficient, well-documented code. Explain your reasoning step by step.",
    temp: 0.2,
    topP: 0.9,
    topK: 40,
    maxTokens: 8192,
    repPenalty: 1.0,
  },
  {
    id: "cx-agent",
    name: "CX Agent",
    system:
      "You are a customer support agent for a SaaS company. Be empathetic, helpful, and solution-oriented. Always verify account details before making changes. Escalate if needed.",
    temp: 0.3,
    topP: 0.9,
    topK: 40,
    maxTokens: 4096,
    repPenalty: 1.05,
  },
];

export const DEMO_SESSIONS: Session[] = [
  {
    id: "s-001",
    name: "CX refund flow test",
    model: "llama-3.1-8b-cx",
    time: "14m ago",
    turns: 6,
  },
  {
    id: "s-002",
    name: "Code generation eval",
    model: "deepseek-r1-code",
    time: "1h ago",
    turns: 12,
  },
  {
    id: "s-003",
    name: "SQL query debugging",
    model: "qwen-2.5-72b",
    time: "3h ago",
    turns: 8,
  },
  {
    id: "s-004",
    name: "Creative writing test",
    model: "llama-3.1-8b-cx",
    time: "yesterday",
    turns: 4,
  },
];
