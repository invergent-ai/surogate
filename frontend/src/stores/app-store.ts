// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { create } from "zustand";
import type { Project } from "@/types/platform";
import type { LocalTask } from "@/api/tasks";
import * as tasksApi from "@/api/tasks";

const ACTIVE_STATUSES = new Set(["pending", "running"]);
const POLL_INTERVAL = 3_000;

type AppState = {
  /* projects */
  projects: Project[];
  activeProjectId: string | null;

  setProjects: (projects: Project[]) => void;
  setActiveProject: (id: string) => void;
  addProject: (project: Project) => void;

  /* tasks */
  tasks: LocalTask[];
  addTask: (task: LocalTask) => void;
  fetchTasks: () => Promise<void>;
  cancelTask: (taskId: string) => Promise<void>;
  startTaskPolling: () => void;
  stopTaskPolling: () => void;
  _pollTimer: ReturnType<typeof setInterval> | null;
};

export const useAppStore = create<AppState>((set, get) => ({
  projects: [],
  activeProjectId: null,

  setProjects: (projects) => set({ projects }),
  setActiveProject: (id) => set({ activeProjectId: id }),
  addProject: (project) =>
    set((s) => ({ projects: [...s.projects, project] })),

  /* tasks */
  tasks: [],
  _pollTimer: null,

  addTask: (task) =>
    set((s) => {
      const tasks = [task, ...s.tasks.filter((t) => t.id !== task.id)];
      // Start polling if this task is active and we're not already polling
      if (ACTIVE_STATUSES.has(task.status) && !s._pollTimer) {
        setTimeout(() => get().startTaskPolling(), 0);
      }
      return { tasks };
    }),

  fetchTasks: async () => {
    try {
      const tasks = await tasksApi.listTasks();
      const prev = get().tasks;
      // Detect tasks that just finished
      for (const task of tasks) {
        const old = prev.find((t) => t.id === task.id);
        if (old && ACTIVE_STATUSES.has(old.status) && !ACTIVE_STATUSES.has(task.status)) {
          _notifyTaskFinished(task);
        }
      }
      set({ tasks });
      // Stop polling if nothing is active
      if (!tasks.some((t) => ACTIVE_STATUSES.has(t.status))) {
        get().stopTaskPolling();
      }
    } catch {
      // silently ignore polling errors
    }
  },

  cancelTask: async (taskId) => {
    await tasksApi.cancelTask(taskId);
    set((s) => ({
      tasks: s.tasks.map((t) =>
        t.id === taskId ? { ...t, status: "cancelled" } : t,
      ),
    }));
  },

  startTaskPolling: () => {
    const s = get();
    if (s._pollTimer) return;
    const timer = setInterval(() => void get().fetchTasks(), POLL_INTERVAL);
    set({ _pollTimer: timer });
  },

  stopTaskPolling: () => {
    const s = get();
    if (s._pollTimer) {
      clearInterval(s._pollTimer);
      set({ _pollTimer: null });
    }
  },
}));

function _notifyTaskFinished(task: LocalTask) {
  if (!("Notification" in window)) return;
  if (Notification.permission === "granted") {
    const icon = task.status === "completed" ? "\u2705" : "\u274c";
    new Notification(`${icon} ${task.name}`, {
      body: task.status === "completed"
        ? "Task completed successfully"
        : `Task failed: ${task.error_message ?? "unknown error"}`,
    });
  }
}
