// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import { create } from "zustand";
import {
  PRESETS,
  type PlaygroundParams,
} from "../playground-data";

const PARAMS_KEY = "surogate_playground_params";
const SYSTEM_PROMPT_KEY = "surogate_playground_system_prompt";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function asFiniteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function loadPlaygroundParams(): PlaygroundParams {
  if (!canUseStorage()) {
    return {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      maxTokens: 2048,
      repPenalty: 1.0,
    };
  }
  try {
    const raw = localStorage.getItem(PARAMS_KEY);
    if (!raw) {
      return {
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        maxTokens: 2048,
        repPenalty: 1.0,
      };
    }
    const parsed = JSON.parse(raw) as Partial<PlaygroundParams>;
    return {
      temperature: asFiniteNumber(parsed.temperature, 0.7),
      topP: asFiniteNumber(parsed.topP, 0.9),
      topK: asFiniteNumber(parsed.topK, 40),
      maxTokens: asFiniteNumber(parsed.maxTokens, 2048),
      repPenalty: asFiniteNumber(parsed.repPenalty, 1.0),
    };
  } catch {
    return {
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      maxTokens: 2048,
      repPenalty: 1.0,
    };
  }
}

function savePlaygroundParams(params: PlaygroundParams): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(PARAMS_KEY, JSON.stringify(params));
  } catch {
    // ignore
  }
}

function loadSystemPrompt(): string {
  if (!canUseStorage()) return PRESETS[0].system;
  try {
    return localStorage.getItem(SYSTEM_PROMPT_KEY) ?? PRESETS[0].system;
  } catch {
    return PRESETS[0].system;
  }
}

function saveSystemPrompt(prompt: string): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(SYSTEM_PROMPT_KEY, prompt);
  } catch {
    // ignore
  }
}

type PlaygroundStore = {
  selectedModelId: string | null;
  compareModelId: string | null;
  showSessions: boolean;
  systemPrompt: string;
  activePreset: string | null;
  params: PlaygroundParams;
  runningByThreadId: Record<string, boolean>;
  toolStatus: string | null;
  contextUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cachedTokens: number;
  } | null;
  activeThreadId: string | null;

  setSelectedModelId: (id: string | null) => void;
  setCompareModelId: (id: string | null) => void;
  toggleSessions: () => void;
  setSystemPrompt: (prompt: string) => void;
  setParams: (params: PlaygroundParams) => void;
  applyPreset: (presetId: string) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  setToolStatus: (status: string | null) => void;
  setContextUsage: (usage: PlaygroundStore["contextUsage"]) => void;
  setActiveThreadId: (id: string | null) => void;
};

export const usePlaygroundStore = create<PlaygroundStore>((set) => ({
  selectedModelId: null,
  compareModelId: null,
  showSessions: false,
  systemPrompt: loadSystemPrompt(),
  activePreset: "default",
  params: loadPlaygroundParams(),
  runningByThreadId: {},
  toolStatus: null,
  contextUsage: null,
  activeThreadId: null,

  setSelectedModelId: (selectedModelId) => set({ selectedModelId }),
  setCompareModelId: (compareModelId) => set({ compareModelId }),
  toggleSessions: () => set((s) => ({ showSessions: !s.showSessions })),

  setSystemPrompt: (systemPrompt) =>
    set(() => {
      saveSystemPrompt(systemPrompt);
      return { systemPrompt, activePreset: null };
    }),

  setParams: (params) =>
    set(() => {
      savePlaygroundParams(params);
      return { params, activePreset: null };
    }),

  applyPreset: (presetId) =>
    set(() => {
      const preset = PRESETS.find((p) => p.id === presetId);
      if (!preset) return {};
      const params: PlaygroundParams = {
        temperature: preset.temp,
        topP: preset.topP,
        topK: preset.topK,
        maxTokens: preset.maxTokens,
        repPenalty: preset.repPenalty,
      };
      savePlaygroundParams(params);
      saveSystemPrompt(preset.system);
      return {
        params,
        systemPrompt: preset.system,
        activePreset: presetId,
      };
    }),

  setThreadRunning: (threadId, running) =>
    set((state) => {
      const next = { ...state.runningByThreadId };
      if (running) {
        next[threadId] = true;
      } else {
        delete next[threadId];
      }
      return { runningByThreadId: next };
    }),

  setToolStatus: (toolStatus) => set({ toolStatus }),
  setContextUsage: (contextUsage) => set({ contextUsage }),
  setActiveThreadId: (activeThreadId) =>
    set({ activeThreadId, contextUsage: null }),
}));
