// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { StateCreator } from "zustand";
import type { Skill } from "@/types/skill";
import * as skillsApi from "@/api/skills";
import type { SkillResponse, SkillCreateRequest, SkillUpdateRequest } from "@/api/skills";
import type { AppState } from "./app-store";

function toSkill(s: SkillResponse): Skill {
  return {
    id: s.id,
    name: s.name,
    displayName: s.display_name,
    description: s.description,
    content: s.content,
    version: s.version,
    status: s.status,
    author: s.author_username || s.author_id,
    updatedAt: s.updated_at ?? "",
    tags: s.tags,
    agent: "",
    versions: [],
    hubRef: s.hub_ref,
  };
}

export type SkillsSlice = {
  skills: Skill[];
  currentSkill: Skill | null;

  fetchSkills: (projectId?: string) => Promise<void>;
  fetchSkill: (skillId: string) => Promise<void>;
  createSkill: (projectId: string, body: SkillCreateRequest) => Promise<Skill | null>;
  updateSkill: (skillId: string, body: SkillUpdateRequest) => Promise<Skill | null>;
  deleteSkill: (skillId: string) => Promise<boolean>;
  publishSkill: (skillId: string, tag: string) => Promise<boolean>;
};

export const createSkillsSlice: StateCreator<AppState, [], [], SkillsSlice> = (set) => ({
  skills: [],
  currentSkill: null,

  fetchSkills: async (projectId?: string) => {
    set({ loading: true, error: null });
    try {
      const res = await skillsApi.listSkills(projectId);
      set({ skills: res.skills.map(toSkill), loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  fetchSkill: async (skillId: string) => {
    set({ loading: true, error: null, currentSkill: null });
    try {
      const skill = await skillsApi.getSkill(skillId);
      set({ currentSkill: toSkill(skill), loading: false });
    } catch (e) {
      set({ error: (e as Error).message, loading: false });
    }
  },

  createSkill: async (projectId: string, body: SkillCreateRequest) => {
    set({ error: null });
    try {
      const raw = await skillsApi.createSkill(projectId, body);
      const skill = toSkill(raw);
      set((s) => ({ skills: [...s.skills, skill] }));
      return skill;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  updateSkill: async (skillId: string, body: SkillUpdateRequest) => {
    set({ error: null });
    try {
      const raw = await skillsApi.updateSkill(skillId, body);
      const skill = toSkill(raw);
      set((s) => ({
        skills: s.skills.map((sk) => (sk.id === skillId ? skill : sk)),
        currentSkill: s.currentSkill?.id === skillId ? skill : s.currentSkill,
      }));
      return skill;
    } catch (e) {
      set({ error: (e as Error).message });
      return null;
    }
  },

  deleteSkill: async (skillId: string) => {
    set({ error: null });
    try {
      await skillsApi.deleteSkill(skillId);
      set((s) => ({
        skills: s.skills.filter((sk) => sk.id !== skillId),
        currentSkill: s.currentSkill?.id === skillId ? null : s.currentSkill,
      }));
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },

  publishSkill: async (skillId: string, tag: string) => {
    set({ error: null });
    try {
      await skillsApi.publishSkill(skillId, tag);
      // Update version in store to reflect the published tag
      set((s) => ({
        skills: s.skills.map((sk) => (sk.id === skillId ? { ...sk, version: tag } : sk)),
        currentSkill: s.currentSkill?.id === skillId ? { ...s.currentSkill, version: tag } : s.currentSkill,
      }));
      return true;
    } catch (e) {
      set({ error: (e as Error).message });
      return false;
    }
  },
});
