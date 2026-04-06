// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useMemo, useEffect } from "react";
import { useAppStore } from "@/stores/app-store";
import type { LocalTask } from "@/api/tasks";
import type { Model } from "@/types/model";
import type { ExtendedWorkload } from "./detail-shared";

function taskToWorkloadItem(t: LocalTask): ExtendedWorkload {
  return {
    id: t.id,
    name: t.name,
    type: "task",
    method: t.task_type.replace("_", " "),
    status: t.status,
    priority: 3,
    gpu: "\u2014",
    gpuCount: 0,
    location: "local",
    node: t.pid ? `pid:${t.pid}` : "\u2014",
    eta: t.progress ?? "\u2014",
    startedAt: t.started_at ? new Date(t.started_at).toLocaleString() : null,
    requestedBy: t.requested_by ?? "\u2014",
    project: t.project_id ?? "\u2014",
    _task: t,
  };
}

function modelToWorkloadItem(m: Model): ExtendedWorkload {
  const gpuLabel = m.gpu.count > 0 ? `${m.gpu.count}\u00d7 ${m.gpu.type}` : "\u2014";
  return {
    id: m.id,
    name: m.displayName || m.name,
    type: "serving",
    method: m.engine !== "\u2014" ? m.engine : "\u2014",
    status: m.status,
    priority: 0,
    gpu: gpuLabel,
    gpuCount: m.gpu.count,
    location: "local",
    node: m.namespace !== "\u2014" ? m.namespace : "\u2014",
    eta: m.uptime !== "\u2014" ? `up ${m.uptime}` : "\u2014",
    startedAt: m.lastDeployed !== "\u2014" ? m.lastDeployed : null,
    requestedBy: m.deployedBy || "\u2014",
    project: m.projectId ?? "\u2014",
    _model: m,
  };
}

const ACTIVE_STATUSES = [
  "running", "queued", "submitted", "provisioning", "cancelling",
  "pending", "serving", "deploying",
];

/**
 * Fetches tasks + models from the store, converts them to a unified
 * ExtendedWorkload list filtered by the active project, sorted with
 * active items first, then by start time descending.
 */
export function useWorkloadItems() {
  const tasks = useAppStore((s) => s.tasks);
  const fetchTasks = useAppStore((s) => s.fetchTasks);
  const models = useAppStore((s) => s.models);
  const fetchModels = useAppStore((s) => s.fetchModels);
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const projects = useAppStore((s) => s.projects);
  const activeProject = projects.find((p) => p.id === activeProjectId);

  useEffect(() => {
    void fetchTasks();
    void fetchModels();
  }, [fetchTasks, fetchModels]);

  const items = useMemo<ExtendedWorkload[]>(() => {
    const taskItems = tasks.map(taskToWorkloadItem);
    const modelItems = models.map(modelToWorkloadItem);
    const all = [...taskItems, ...modelItems];

    const projectMatches = activeProject
      ? all.filter((w) => {
          if (!w.project) return true;
          const p = w.project.toLowerCase();
          return p === activeProject.id
            || p === activeProject.name.toLowerCase()
            || p === activeProject.namespace.toLowerCase();
        })
      : all;

    projectMatches.sort((a, b) => {
      const aActive = ACTIVE_STATUSES.includes(a.status) ? 1 : 0;
      const bActive = ACTIVE_STATUSES.includes(b.status) ? 1 : 0;
      if (aActive !== bActive) return bActive - aActive;
      const aTime = a.startedAt ? new Date(a.startedAt).getTime() : 0;
      const bTime = b.startedAt ? new Date(b.startedAt).getTime() : 0;
      return bTime - aTime;
    });

    return projectMatches;
  }, [tasks, models, activeProject]);

  return items;
}
