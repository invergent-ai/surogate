// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";

export interface LocalTask {
  id: string;
  name: string;
  task_type: string;
  status: string;
  pid?: number | null;
  exit_code?: number | null;
  error_message?: string | null;
  progress?: string | null;
  project_id: string;
  requested_by: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

export interface TaskLogsResponse {
  task_id: string;
  lines: string[];
}

export async function spawnTask(params: {
  task_type: string;
  name: string;
  project_id: string;
  params: Record<string, string>;
}): Promise<LocalTask> {
  const response = await authFetch("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to spawn task");
  }
  return (await response.json()) as LocalTask;
}

export async function listTasks(params?: {
  project_id?: string;
  task_type?: string;
  status?: string;
}): Promise<LocalTask[]> {
  const qs = new URLSearchParams();
  if (params?.project_id) qs.append("project_id", params.project_id);
  if (params?.task_type) qs.append("task_type", params.task_type);
  if (params?.status) qs.append("status", params.status);
  const response = await authFetch(`/api/tasks?${qs}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to list tasks");
  }
  return (await response.json()) as LocalTask[];
}

export async function getTask(taskId: string): Promise<LocalTask> {
  const response = await authFetch(`/api/tasks/${taskId}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to get task");
  }
  return (await response.json()) as LocalTask;
}

export async function cancelTask(taskId: string): Promise<void> {
  const response = await authFetch(`/api/tasks/${taskId}/cancel`, { method: "POST" });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to cancel task");
  }
}

export async function getTaskLogs(taskId: string, tail = 100): Promise<TaskLogsResponse> {
  const response = await authFetch(`/api/tasks/${taskId}/logs?tail=${tail}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to get task logs");
  }
  return (await response.json()) as TaskLogsResponse;
}
