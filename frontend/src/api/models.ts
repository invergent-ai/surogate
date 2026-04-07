// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";
import type { Model } from "@/types/model";

// ── Request types ─────────────────────────────────────────────

export interface CreateModelRequest {
  name: string;
  display_name: string;
  base_model: string;
  project_id: string;
  family?: string;
  param_count?: string;
  model_type?: string;
  quantization?: string;
  context_window?: number;
  engine?: string;
  image?: string;
  hub_ref?: string;
  namespace?: string;
  source?: string;
  serving_config?: Record<string, unknown>;
  generation_defaults?: Record<string, unknown>;
}

export interface UpdateModelRequest {
  engine?: string;
  accelerators?: string;
  infra?: string | null;
  use_spot?: boolean;
  serving_config?: Record<string, unknown>;
  generation_defaults?: Record<string, unknown>;
}

export interface ScaleModelRequest {
  replicas: number;
}

// ── Response types ────────────────────────────────────────────

interface RawModelListResponse {
  models: RawModel[];
  total: number;
  status_counts: Record<string, number>;
}

export interface ModelListResponse {
  models: Model[];
  total: number;
  statusCounts: Record<string, number>;
}

// Raw backend shape (snake_case)
export interface RawModel {
  id: string;
  name: string;
  display_name: string;
  description: string;
  base: string;
  project_id: string;
  family: string;
  param_count: string;
  type: string;
  quantization: string;
  context_window: number;
  status: string;
  engine: string;
  replicas: { current: number; desired: number };
  gpu: { type: string; count: number; utilization: number };
  vram: { used: string; total: string; pct: number };
  cpu: string;
  mem: string;
  mem_limit: string;
  tps: number;
  p50: string;
  p95: string;
  p99: string;
  queue_depth: number;
  batch_size: string;
  tokens_in_24h: string;
  tokens_out_24h: string;
  requests_24h: number;
  error_rate: string;
  uptime: string;
  last_deployed: string;
  deployed_by: string;
  namespace: string;
  project_color: string;
  endpoint: string;
  image: string;
  hub_ref: string;
  infra: string | null;
  source: string | null;
  connected_agents: { name: string; status: string; rps: number }[];
  serving_config: Record<string, unknown> | null;
  generation_defaults: Record<string, unknown> | null;
  fine_tunes: {
    name: string;
    method: string;
    dataset: string;
    date: string;
    status: string;
    loss: string;
    hub_ref: string;
  }[];
  metrics_history: {
    tps: number[];
    latency: number[];
    gpu: number[];
    queue: number[];
  };
  events: { time: string; text: string; type: string }[];
}

// ── Transform ─────────────────────────────────────────────────

export function transformModel(r: RawModel): Model {
  return {
    id: r.id,
    name: r.name,
    displayName: r.display_name,
    description: r.description,
    base: r.base,
    projectId: r.project_id,
    family: r.family,
    paramCount: r.param_count,
    type: r.type,
    quantization: r.quantization,
    contextWindow: r.context_window,
    status: r.status,
    engine: r.engine,
    replicas: r.replicas,
    gpu: r.gpu,
    vram: r.vram,
    cpu: r.cpu,
    mem: r.mem,
    memLimit: r.mem_limit,
    tps: r.tps,
    p50: r.p50,
    p95: r.p95,
    p99: r.p99,
    queueDepth: r.queue_depth,
    batchSize: r.batch_size,
    tokensIn24h: r.tokens_in_24h,
    tokensOut24h: r.tokens_out_24h,
    requests24h: r.requests_24h,
    errorRate: r.error_rate,
    uptime: r.uptime,
    lastDeployed: r.last_deployed,
    deployedBy: r.deployed_by,
    namespace: r.namespace,
    projectColor: r.project_color,
    endpoint: r.endpoint,
    image: r.image,
    hubRef: r.hub_ref,
    infra: r.infra,
    source: r.source,
    connectedAgents: r.connected_agents,
    servingConfig: r.serving_config ?? null,
    generationDefaults: r.generation_defaults
      ? {
          temperature: (r.generation_defaults.temperature as number) ?? 0,
          topP: (r.generation_defaults.top_p as number) ?? 0,
          topK: (r.generation_defaults.top_k as number) ?? 0,
          maxTokens: (r.generation_defaults.max_tokens as number) ?? 0,
          repetitionPenalty: (r.generation_defaults.repetition_penalty as number) ?? 1,
          stopSequences: (r.generation_defaults.stop_sequences as string[]) ?? [],
        }
      : null,
    fineTunes: r.fine_tunes.map((ft) => ({
      name: ft.name,
      method: ft.method,
      dataset: ft.dataset,
      date: ft.date,
      status: ft.status,
      loss: ft.loss,
      hubRef: ft.hub_ref,
    })),
    metricsHistory: r.metrics_history,
    events: r.events,
  };
}

// ── API functions ─────────────────────────────────────────────

export async function listModels(params?: {
  project_id?: string;
  status?: string;
  search?: string;
  limit?: number;
}): Promise<ModelListResponse> {
  const qs = new URLSearchParams();
  if (params?.project_id) qs.set("project_id", params.project_id);
  if (params?.status) qs.set("status", params.status);
  if (params?.search) qs.set("search", params.search);
  if (params?.limit) qs.set("limit", String(params.limit));

  const url = `/api/models/${qs.toString() ? `?${qs}` : ""}`;
  const response = await authFetch(url);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to list models");
  }
  const raw = (await response.json()) as RawModelListResponse;
  return {
    models: raw.models.map(transformModel),
    total: raw.total,
    statusCounts: raw.status_counts,
  };
}

export async function getModel(modelId: string): Promise<Model> {
  const response = await authFetch(`/api/models/${modelId}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to get model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function createModel(req: CreateModelRequest): Promise<Model> {
  const response = await authFetch("/api/models/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to deploy model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function updateModel(
  modelId: string,
  req: UpdateModelRequest,
): Promise<Model> {
  const response = await authFetch(`/api/models/${modelId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to update model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function scaleModel(
  modelId: string,
  req: ScaleModelRequest,
): Promise<Model> {
  const response = await authFetch(`/api/models/${modelId}/scale`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to scale model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function startModel(modelId: string): Promise<Model> {
  const response = await authFetch(`/api/models/${modelId}/start`, {
    method: "POST",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to start model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function restartModel(modelId: string): Promise<Model> {
  const response = await authFetch(`/api/models/${modelId}/restart`, {
    method: "POST",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to restart model");
  }
  return transformModel((await response.json()) as RawModel);
}

export async function stopModel(modelId: string): Promise<void> {
  const response = await authFetch(`/api/models/${modelId}/stop`, {
    method: "POST",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to stop model");
  }
}

// ── Logs ─────────────────────────────────────────────────────

export interface ModelLogsResponse {
  model_id: string;
  target: string;
  lines: string[];
}

export async function getModelLogs(
  modelId: string,
  params?: { target?: string; replica_id?: number; tail?: number },
): Promise<ModelLogsResponse> {
  const qs = new URLSearchParams();
  if (params?.target) qs.set("target", params.target);
  if (params?.replica_id != null) qs.set("replica_id", String(params.replica_id));
  if (params?.tail != null) qs.set("tail", String(params.tail));
  const url = `/api/models/${modelId}/logs${qs.toString() ? `?${qs}` : ""}`;
  const response = await authFetch(url);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to get model logs");
  }
  return (await response.json()) as ModelLogsResponse;
}

export async function deleteModel(modelId: string): Promise<void> {
  const response = await authFetch(`/api/models/${modelId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to delete model");
  }
}
