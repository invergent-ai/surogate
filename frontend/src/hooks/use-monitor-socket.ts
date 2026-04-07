// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
// Connects to the server's /ws/monitor WebSocket and dispatches
// incoming entity updates directly into the Zustand stores,
// replacing polling for status changes.

import { useEffect, useRef } from "react";
import { useAppStore } from "@/stores/app-store";
import { transformModel, type RawModel } from "@/api/models";
import type { LocalTask } from "@/api/tasks";
import { getAuthToken } from "@/features/auth";

interface TransitionMessage {
  type: "transition";
  entity_type: "model" | "job" | "task";
  entity_id: string;
  name: string;
  old_status: string;
  new_status: string;
  data: Record<string, unknown>;
}

interface MetricsMessage {
  type: "metrics";
  entity_type: "model" | "job";
  entity_id: string;
  metrics: Record<string, number>;
}

type WsMessage = TransitionMessage | MetricsMessage;

const RECONNECT_DELAY = 3_000;

export function useMonitorSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    let unmounted = false;

    function connect() {
      if (unmounted) return;

      const token = getAuthToken();
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${window.location.host}/ws/monitor${
        token ? `?token=${encodeURIComponent(token)}` : ""
      }`;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        let msg: WsMessage;
        try {
          msg = JSON.parse(ev.data as string) as WsMessage;
        } catch {
          return;
        }
        if (msg.type === "transition") {
          dispatch(msg);
        } else if (msg.type === "metrics") {
          dispatchMetrics(msg);
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (!unmounted) {
          reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      unmounted = true;
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);
}

// ── Dispatchers ────────────────────────────────────────────────

function dispatch(msg: TransitionMessage) {
  const state = useAppStore.getState();

  switch (msg.entity_type) {
    case "model":
      dispatchModel(state, msg);
      break;
    case "job":
      dispatchJob(msg);
      break;
    case "task":
      dispatchTask(state, msg);
      break;
  }
}

function dispatchModel(
  state: ReturnType<typeof useAppStore.getState>,
  msg: TransitionMessage,
) {
  const updated = transformModel(msg.data as unknown as RawModel);

  // Replace in the models list
  const models = state.models.map((m) =>
    m.id === updated.id ? updated : m,
  );
  const statusCounts = { ...state.modelsStatusCounts };
  if (msg.old_status !== msg.new_status) {
    statusCounts[msg.old_status] = Math.max(0, (statusCounts[msg.old_status] ?? 0) - 1);
    statusCounts[msg.new_status] = (statusCounts[msg.new_status] ?? 0) + 1;
  }

  // Clear any pending action for this model since its status just changed
  const pending = { ...state.modelPending };
  if (updated.id in pending) {
    delete pending[updated.id];
  }

  useAppStore.setState({
    models,
    modelsStatusCounts: statusCounts,
    modelPending: pending,
    // Keep selectedModel in sync
    selectedModel:
      state.selectedModel?.id === updated.id ? updated : state.selectedModel,
  });
}



function dispatchJob(msg: TransitionMessage) {
  // Jobs don't have a dedicated slice with an entity list yet.
  // For now, just trigger a refetch of compute data if anyone is polling.
  // This is a no-op placeholder — extend when a jobs slice exists.
  void msg;
}

function dispatchTask(
  state: ReturnType<typeof useAppStore.getState>,
  msg: TransitionMessage,
) {
  const updated = msg.data as unknown as LocalTask;

  const tasks = state.tasks.map((t) =>
    t.id === updated.id ? updated : t,
  );
  useAppStore.setState({ tasks });

  // Fire browser notification for completed/failed tasks
  if (
    msg.old_status === "running" &&
    (msg.new_status === "completed" || msg.new_status === "failed")
  ) {
    notifyTaskFinished(updated);
  }
}

function dispatchMetrics(msg: MetricsMessage) {
  useAppStore.getState().pushMetrics(msg.entity_id, msg.metrics);
}

function notifyTaskFinished(task: LocalTask) {
  if (!("Notification" in window)) return;
  if (Notification.permission === "granted") {
    const icon = task.status === "completed" ? "\u2705" : "\u274c";
    new Notification(`${icon} ${task.name}`, {
      body:
        task.status === "completed"
          ? "Task completed successfully"
          : `Task failed: ${task.error_message ?? "unknown error"}`,
    });
  }
}
