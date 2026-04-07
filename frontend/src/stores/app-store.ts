// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { create } from "zustand";
import { createProjectsSlice, type ProjectsSlice } from "./projects-slice";
import { createTasksSlice, type TasksSlice } from "./tasks-slice";
import { createHubSlice, type HubSlice } from "./hub-slice";
import { createSkillsSlice, type SkillsSlice } from "./skills-slice";
import { createComputeSlice, type ComputeSlice } from "./compute-slice";
import { createModelsSlice, type ModelsSlice } from "./models-slice";
import { createMetricsSlice, type MetricsSlice } from "./metrics-slice";

export type { WorkloadMetrics, MetricsSnapshot } from "./metrics-slice";

export type AppState = ProjectsSlice & TasksSlice & HubSlice & SkillsSlice & ComputeSlice & ModelsSlice & MetricsSlice & {
  loading: boolean;
  error: string | null;
};

export const useAppStore = create<AppState>((...a) => ({
  loading: false,
  error: null,
  ...createProjectsSlice(...a),
  ...createTasksSlice(...a),
  ...createHubSlice(...a),
  ...createSkillsSlice(...a),
  ...createComputeSlice(...a),
  ...createModelsSlice(...a),
  ...createMetricsSlice(...a),
}));
