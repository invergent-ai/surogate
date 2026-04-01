// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

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
