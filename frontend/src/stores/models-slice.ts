// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { Model } from "@/types/model";
import type { DeployModelRequest, ScaleModelRequest, UpdateModelRequest } from "@/api/models";
import * as modelsApi from "@/api/models";

let pollTimer: ReturnType<typeof setInterval> | null = null;
let pollRefCount = 0;

export type ModelsSlice = {
  models: Model[];
  selectedModel: Model | null;
  modelsStatusCounts: Record<string, number>;
  modelsLoading: boolean;

  fetchModels: (params?: { status?: string; search?: string }) => Promise<void>;
  fetchModel: (modelId: string) => Promise<void>;
  deployModel: (req: DeployModelRequest) => Promise<Model | null>;
  updateModel: (modelId: string, req: UpdateModelRequest) => Promise<boolean>;
  scaleModel: (modelId: string, req: ScaleModelRequest) => Promise<boolean>;
  restartModel: (modelId: string) => Promise<boolean>;
  stopModel: (modelId: string) => Promise<boolean>;
  deleteModel: (modelId: string) => Promise<boolean>;
  startModelsPolling: () => () => void;
};

export const createModelsSlice: StateCreator<AppState, [], [], ModelsSlice> = (
  set,
  get,
) => ({
  models: [],
  selectedModel: null,
  modelsStatusCounts: {},
  modelsLoading: false,

  fetchModels: async (params) => {
    try {
      set({ modelsLoading: true });
      const res = await modelsApi.listModels(params);
      set({
        models: res.models,
        modelsStatusCounts: res.statusCounts,
        modelsLoading: false,
      });
      // Refresh selected model if it's still in the list
      const sel = get().selectedModel;
      if (sel) {
        const updated = res.models.find((m) => m.id === sel.id);
        if (updated) set({ selectedModel: updated });
      }
    } catch (e) {
      set({ modelsLoading: false, error: (e as Error).message });
    }
  },

  fetchModel: async (modelId) => {
    try {
      const model = await modelsApi.getModel(modelId);
      set({ selectedModel: model });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  deployModel: async (req) => {
    try {
      const model = await modelsApi.deployModel(req);
      await get().fetchModels();
      return model;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  updateModel: async (modelId, req) => {
    try {
      const model = await modelsApi.updateModel(modelId, req);
      set({ selectedModel: model });
      await get().fetchModels();
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  scaleModel: async (modelId, req) => {
    try {
      const model = await modelsApi.scaleModel(modelId, req);
      set({ selectedModel: model });
      await get().fetchModels();
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  restartModel: async (modelId) => {
    try {
      const model = await modelsApi.restartModel(modelId);
      set({ selectedModel: model });
      await get().fetchModels();
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  stopModel: async (modelId) => {
    try {
      await modelsApi.stopModel(modelId);
      await get().fetchModels();
      const sel = get().selectedModel;
      if (sel?.id === modelId) {
        await get().fetchModel(modelId);
      }
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  deleteModel: async (modelId) => {
    try {
      await modelsApi.deleteModel(modelId);
      set({ selectedModel: null });
      await get().fetchModels();
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  startModelsPolling: () => {
    pollRefCount++;
    if (!pollTimer) {
      get().fetchModels();
      pollTimer = setInterval(() => get().fetchModels(), 10_000);
    }
    return () => {
      pollRefCount--;
      if (pollRefCount <= 0 && pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
        pollRefCount = 0;
      }
    };
  },
});
