// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";

/** Live hardware metrics pushed via WebSocket for running workloads. */
export interface WorkloadMetrics {
  cpu_usage_percent?: number;
  memory_usage_bytes?: number;
  memory_working_set_bytes?: number;
  memory_total_bytes?: number;
  cpus_detected_num?: number;
  gpus_detected_num?: number;
  gpu_memory_total_bytes?: number;
  [key: string]: number | undefined;
}

/** A single timestamped metrics snapshot. */
export interface MetricsSnapshot {
  time: number;
  metrics: WorkloadMetrics;
}

/** Max samples kept per workload (~5 min at 5s interval). */
const HISTORY_MAX = 60;

export type MetricsSlice = {
  workloadMetrics: Record<string, WorkloadMetrics>;
  workloadMetricsHistory: Record<string, MetricsSnapshot[]>;
  pushMetrics: (entityId: string, metrics: WorkloadMetrics) => void;
};

export const createMetricsSlice: StateCreator<AppState, [], [], MetricsSlice> = (set) => ({
  workloadMetrics: {},
  workloadMetricsHistory: {},

  pushMetrics: (entityId, metrics) =>
    set((s) => {
      const prev = s.workloadMetricsHistory[entityId] ?? [];
      const snapshot: MetricsSnapshot = { time: Date.now(), metrics };
      const next =
        prev.length >= HISTORY_MAX
          ? [...prev.slice(-(HISTORY_MAX - 1)), snapshot]
          : [...prev, snapshot];
      return {
        workloadMetrics: { ...s.workloadMetrics, [entityId]: metrics },
        workloadMetricsHistory: { ...s.workloadMetricsHistory, [entityId]: next },
      };
    }),
});
