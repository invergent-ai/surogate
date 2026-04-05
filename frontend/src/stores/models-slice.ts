// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { Model } from "@/types/model";
import type { CreateModelRequest, ScaleModelRequest, UpdateModelRequest } from "@/api/models";
import * as modelsApi from "@/api/models";

export type ModelsSlice = {
  models: Model[];
  selectedModel: Model | null;
  modelsStatusCounts: Record<string, number>;
  modelsLoading: boolean;
  /** Tracks pending actions per model — maps modelId → status at time of action */
  modelPending: Record<string, string>;

  fetchModels: (params?: { status?: string; search?: string }) => Promise<void>;
  fetchModel: (modelId: string) => Promise<void>;
  createModel: (req: CreateModelRequest) => Promise<Model | null>;
  updateModel: (modelId: string, req: UpdateModelRequest) => Promise<boolean>;
  scaleModel: (modelId: string, req: ScaleModelRequest) => Promise<boolean>;
  startModel: (modelId: string) => Promise<boolean>;
  restartModel: (modelId: string) => Promise<boolean>;
  stopModel: (modelId: string) => Promise<boolean>;
  deleteModel: (modelId: string) => Promise<boolean>;
};

export const createModelsSlice: StateCreator<AppState, [], [], ModelsSlice> = (
  set,
  get,
) => ({
  models: [],
  selectedModel: null,
  modelsStatusCounts: {},
  modelsLoading: false,
  modelPending: {},

  fetchModels: async (params) => {
    try {
      set({ modelsLoading: true });
      const res = await modelsApi.listModels(params);
      // Clear pending for models whose status differs from the snapshot
      const prev = get();
      const pending = { ...prev.modelPending };
      for (const m of res.models) {
        if (m.id in pending && m.status !== pending[m.id]) {
          delete pending[m.id];
        }
      }
      set({
        models: res.models,
        modelsStatusCounts: res.statusCounts,
        modelsLoading: false,
        modelPending: pending,
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

  createModel: async (req) => {
    try {
      const model = await modelsApi.createModel(req);
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

  startModel: async (modelId) => {
    const cur = get().models.find((m) => m.id === modelId);
    set((s) => ({ modelPending: { ...s.modelPending, [modelId]: cur?.status ?? "" } }));
    try {
      const updated = await modelsApi.startModel(modelId);
      set((s) => {
        const pending = { ...s.modelPending };
        delete pending[modelId];
        return {
          selectedModel: s.selectedModel?.id === modelId ? updated : s.selectedModel,
          models: s.models.map((m) => (m.id === modelId ? updated : m)),
          modelPending: pending,
        };
      });
      return true;
    } catch (e) {
      set((s) => {
        const pending = { ...s.modelPending };
        delete pending[modelId];
        return { modelPending: pending, error: (e as Error).message };
      });
      return false;
    }
  },

  restartModel: async (modelId) => {
    const cur = get().models.find((m) => m.id === modelId);
    set((s) => ({ modelPending: { ...s.modelPending, [modelId]: cur?.status ?? "" } }));
    try {
      const updated = await modelsApi.restartModel(modelId);
      set((s) => {
        const pending = { ...s.modelPending };
        delete pending[modelId];
        return {
          selectedModel: s.selectedModel?.id === modelId ? updated : s.selectedModel,
          models: s.models.map((m) => (m.id === modelId ? updated : m)),
          modelPending: pending,
        };
      });
      return true;
    } catch (e) {
      set((s) => {
        const pending = { ...s.modelPending };
        delete pending[modelId];
        return { modelPending: pending, error: (e as Error).message };
      });
      return false;
    }
  },

  stopModel: async (modelId) => {
    const cur = get().models.find((m) => m.id === modelId);
    set((s) => ({ modelPending: { ...s.modelPending, [modelId]: cur?.status ?? "" } }));
    try {
      await modelsApi.stopModel(modelId);
      await get().fetchModels();
      const sel = get().selectedModel;
      if (sel?.id === modelId) {
        await get().fetchModel(modelId);
      }
      return true;
    } catch (e) {
      set((s) => {
        const pending = { ...s.modelPending };
        delete pending[modelId];
        return { modelPending: pending, error: (e as Error).message };
      });
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

});
