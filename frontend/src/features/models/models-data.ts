// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Status } from "@/components/ui/status-dot";

// Re-export model types from canonical location
export type {
  Model,
  ModelReplicas,
  ModelGpu,
  ModelVram,
  ModelConnectedAgent,
  ModelServingConfig,
  ModelGenerationDefaults,
  ModelFineTune,
  ModelMetricsHistory,
  ModelEvent,
} from "@/types/model";

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
