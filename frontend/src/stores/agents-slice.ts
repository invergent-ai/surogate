// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { Agent } from "@/features/agents/agents-data";
import * as agentsApi from "@/api/agents";
import type { AgentResponse, AgentCreateRequest, AgentUpdateRequest } from "@/api/agents";
import type { AppState } from "./app-store";

export function toAgent(a: AgentResponse): Agent {
  return {
    id: a.id,
    name: a.name,
    displayName: a.display_name,
    description: a.description,
    version: a.version,
    status: a.status,
    replicas: {
      current: a.replicas?.current ?? 0,
      desired: a.replicas?.desired ?? 0,
    },
    projectId: a.project_id,
    model: a.model_name,
    modelBase: "",
    createdBy: a.created_by_username,
    createdAt: a.created_at ?? "",
    lastDeployed: a.created_at ?? "",
    endpoint: a.endpoint,
    cpu: "—",
    mem: "—",
    memLimit: "—",
    gpu: "—",
    rps: 0,
    p50: "—",
    p95: "—",
    p99: "—",
    errorRate: "—",
    tokensIn24h: "0",
    tokensOut24h: "0",
    conversations24h: 0,
    avgTurns: 0,
    satisfaction: "—",
    image: a.image,
    resources: {
      cpuReq: a.resources?.cpuReq ?? "",
      cpuLim: a.resources?.cpuLim ?? "",
      memReq: a.resources?.memReq ?? "",
      memLim: a.resources?.memLim ?? "",
    },
    skills: [],
    mcpServers: [],
    env: Object.entries(a.env_vars ?? {}).map(([key, value]) => ({ key, value })),
    versions: [],
    metricsHistory: { rps: [], latency: [], errors: [], tokens: [] },
  };
}

export type AgentsSlice = {
  agents: Agent[];
  selectedAgent: Agent | null;
  agentsLoading: boolean;

  fetchAgents: (projectId?: string) => Promise<void>;
  fetchAgent: (agentId: string) => Promise<void>;
  createAgent: (projectId: string, body: AgentCreateRequest) => Promise<Agent | null>;
  updateAgent: (agentId: string, body: AgentUpdateRequest) => Promise<Agent | null>;
  deleteAgent: (agentId: string) => Promise<boolean>;
  startAgent: (agentId: string) => Promise<Agent | null>;
  stopAgent: (agentId: string) => Promise<Agent | null>;
  setSelectedAgent: (agent: Agent | null) => void;
};

export const createAgentsSlice: StateCreator<AppState, [], [], AgentsSlice> = (set, get) => ({
  agents: [],
  selectedAgent: null,
  agentsLoading: false,

  fetchAgents: async (projectId?: string) => {
    set({ agentsLoading: true, error: null });
    try {
      const res = await agentsApi.listAgents(projectId);
      set({ agents: res.agents.map(toAgent), agentsLoading: false });
    } catch (e) {
      set({ error: (e as Error).message, agentsLoading: false });
    }
  },

  fetchAgent: async (agentId: string) => {
    set({ agentsLoading: true, error: null });
    try {
      const raw = await agentsApi.getAgent(agentId);
      const agent = toAgent(raw);
      set({ selectedAgent: agent, agentsLoading: false });
    } catch (e) {
      set({ error: (e as Error).message, agentsLoading: false });
    }
  },

  createAgent: async (projectId: string, body: AgentCreateRequest) => {
    set({ error: null });
    try {
      const raw = await agentsApi.createAgent(projectId, body);
      const agent = toAgent(raw);
      set((s) => ({ agents: [agent, ...s.agents] }));
      return agent;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  updateAgent: async (agentId: string, body: AgentUpdateRequest) => {
    set({ error: null });
    try {
      const raw = await agentsApi.updateAgent(agentId, body);
      const agent = toAgent(raw);
      set((s) => ({
        agents: s.agents.map((a) => (a.id === agentId ? agent : a)),
        selectedAgent: s.selectedAgent?.id === agentId ? agent : s.selectedAgent,
      }));
      return agent;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  deleteAgent: async (agentId: string) => {
    set({ error: null });
    try {
      await agentsApi.deleteAgent(agentId);
      set((s) => ({
        agents: s.agents.filter((a) => a.id !== agentId),
        selectedAgent: s.selectedAgent?.id === agentId ? null : s.selectedAgent,
      }));
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  startAgent: async (agentId: string) => {
    set({ error: null });
    set((s) => ({
      agents: s.agents.map((a) => (a.id === agentId ? { ...a, status: "deploying" } : a)),
      selectedAgent:
        s.selectedAgent?.id === agentId
          ? { ...s.selectedAgent, status: "deploying" }
          : s.selectedAgent,
    }));
    try {
      const raw = await agentsApi.startAgent(agentId);
      const agent = toAgent(raw);
      set((s) => ({
        agents: s.agents.map((a) => (a.id === agentId ? agent : a)),
        selectedAgent: s.selectedAgent?.id === agentId ? agent : s.selectedAgent,
      }));
      return agent;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  stopAgent: async (agentId: string) => {
    set({ error: null });
    try {
      const raw = await agentsApi.stopAgent(agentId);
      const agent = toAgent(raw);
      set((s) => ({
        agents: s.agents.map((a) => (a.id === agentId ? agent : a)),
        selectedAgent: s.selectedAgent?.id === agentId ? agent : s.selectedAgent,
      }));
      return agent;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  setSelectedAgent: (agent: Agent | null) => set({ selectedAgent: agent }),
});
