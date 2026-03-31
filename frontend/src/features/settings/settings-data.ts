// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export interface Member {
  name: string;
  email: string;
  role: "Owner" | "Editor" | "Viewer";
  avatar: string;
  color: string;
  status: "online" | "offline";
  lastActive: string;
}

export interface Project {
  id: string;
  name: string;
  namespace: string;
  color: string;
  agents: number;
  models: number;
  status: string;
  members: Member[];
}

export interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  created: string;
  lastUsed: string;
  scopes: string[];
  status: "active" | "revoked";
}

export interface HubConfig {
  gitUrl: string;
  s3Endpoint: string;
  apiUrl: string;
  storageUsed: string;
  storageLimit: string;
  repositories: number;
  lfsEnabled: boolean;
}

export interface Integration {
  id: string;
  name: string;
  icon: string;
  status: "connected" | "disconnected";
  description: string;
  config: Record<string, string>;
}

export interface NotificationCategory {
  category: string;
  items: { label: string; email: boolean; inApp: boolean }[];
}

export const CURRENT_USER = {
  name: "Attila Kovacs",
  email: "a.kovacs@company.com",
  role: "Skill Engineer",
  avatar: "AK",
  defaultProject: "prod-cx",
  createdAt: "2025-06-15",
  lastLogin: "2026-03-28 09:12",
};

export const PROJECTS: Project[] = [
  {
    id: "prod-cx",
    name: "CX Support Agent",
    namespace: "prod-cx",
    color: "#F59E0B",
    agents: 2,
    models: 2,
    status: "active",
    members: [
      { name: "Attila Kovacs", email: "a.kovacs@company.com", role: "Owner", avatar: "AK", color: "#3B82F6", status: "online", lastActive: "now" },
      { name: "Ming Chen", email: "m.chen@company.com", role: "Editor", avatar: "MC", color: "#22C55E", status: "online", lastActive: "2m ago" },
      { name: "Rodrigo Silva", email: "r.silva@company.com", role: "Editor", avatar: "RS", color: "#F59E0B", status: "offline", lastActive: "3h ago" },
      { name: "Lisa Park", email: "l.park@company.com", role: "Viewer", avatar: "LP", color: "#8B5CF6", status: "online", lastActive: "15m ago" },
    ],
  },
  {
    id: "prod-code",
    name: "Code Assistant",
    namespace: "prod-code",
    color: "#3B82F6",
    agents: 1,
    models: 1,
    status: "active",
    members: [
      { name: "Ming Chen", email: "m.chen@company.com", role: "Owner", avatar: "MC", color: "#22C55E", status: "online", lastActive: "2m ago" },
      { name: "Lisa Park", email: "l.park@company.com", role: "Editor", avatar: "LP", color: "#8B5CF6", status: "online", lastActive: "15m ago" },
    ],
  },
  {
    id: "staging-da",
    name: "Data Analyst Agent",
    namespace: "staging-da",
    color: "#8B5CF6",
    agents: 1,
    models: 2,
    status: "active",
    members: [
      { name: "Rodrigo Silva", email: "r.silva@company.com", role: "Owner", avatar: "RS", color: "#F59E0B", status: "offline", lastActive: "3h ago" },
      { name: "Attila Kovacs", email: "a.kovacs@company.com", role: "Editor", avatar: "AK", color: "#3B82F6", status: "online", lastActive: "now" },
      { name: "James Wright", email: "j.wright@company.com", role: "Viewer", avatar: "JW", color: "#EF4444", status: "offline", lastActive: "1d ago" },
    ],
  },
];

export const API_KEYS: ApiKey[] = [
  { id: "key-001", name: "CI/CD Pipeline", prefix: "sk-...f8a2", created: "2026-01-10", lastUsed: "2h ago", scopes: ["read", "deploy"], status: "active" },
  { id: "key-002", name: "Monitoring Dashboard", prefix: "sk-...3b1c", created: "2026-02-20", lastUsed: "5m ago", scopes: ["read"], status: "active" },
  { id: "key-003", name: "Training Scripts", prefix: "sk-...9d4e", created: "2025-11-05", lastUsed: "1d ago", scopes: ["read", "write", "train"], status: "active" },
  { id: "key-004", name: "Old Integration", prefix: "sk-...1a7f", created: "2025-08-12", lastUsed: "3mo ago", scopes: ["read"], status: "revoked" },
];

export const HUB_CONFIG: HubConfig = {
  gitUrl: "git.studio.internal:3000",
  s3Endpoint: "s3.studio.internal:9000",
  apiUrl: "https://hub.studio.internal/api/v1",
  storageUsed: "2.4 TB",
  storageLimit: "10 TB",
  repositories: 10,
  lfsEnabled: true,
};

export const INTEGRATIONS: Integration[] = [
  { id: "int-hf", name: "Hugging Face", icon: "HF", status: "connected", description: "Import models and datasets from HF Hub", config: { token: "hf_...****", org: "company-ai" } },
  { id: "int-wandb", name: "Weights & Biases", icon: "WB", status: "connected", description: "Training metrics and experiment tracking", config: { apiKey: "wb_...****", project: "surogate-studio" } },
  { id: "int-slack", name: "Slack", icon: "SL", status: "disconnected", description: "Alerts and notifications to Slack channels", config: {} },
  { id: "int-github", name: "GitHub", icon: "GH", status: "connected", description: "Code sync and agent flow version control", config: { token: "ghp_...****", org: "company-ai" } },
];

export const NOTIFICATION_CATEGORIES: NotificationCategory[] = [
  {
    category: "Deployments",
    items: [
      { label: "Agent deployment succeeded", email: true, inApp: true },
      { label: "Agent deployment failed", email: true, inApp: true },
      { label: "Model serving started/stopped", email: false, inApp: true },
      { label: "Auto-scaling events", email: false, inApp: true },
    ],
  },
  {
    category: "Training",
    items: [
      { label: "Training run completed", email: true, inApp: true },
      { label: "Training run failed", email: true, inApp: true },
      { label: "Checkpoint saved", email: false, inApp: true },
      { label: "Evaluation completed", email: false, inApp: true },
    ],
  },
  {
    category: "Monitoring",
    items: [
      { label: "Critical alerts (agent down)", email: true, inApp: true },
      { label: "Warning alerts (high latency)", email: false, inApp: true },
      { label: "Error rate spike", email: true, inApp: true },
      { label: "Budget threshold reached", email: true, inApp: true },
    ],
  },
  {
    category: "Conversations",
    items: [
      { label: "Flagged conversation", email: false, inApp: true },
      { label: "Escalation triggered", email: true, inApp: true },
    ],
  },
];
