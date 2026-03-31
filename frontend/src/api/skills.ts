// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { authFetch } from "@/api/auth";

// ── Types ──────────────────────────────────────────────────────────

export interface SkillResponse {
  id: string;
  name: string;
  display_name: string;
  description: string;
  content: string;
  version: string;
  status: string;
  author_id: string;
  author_username: string;
  tags: string[];
  hub_ref: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface SkillListResponse {
  skills: SkillResponse[];
  total: number;
}

export interface SkillCreateRequest {
  name: string;
  display_name: string;
  description?: string;
  content?: string;
  tags?: string[];
}

export interface SkillUpdateRequest {
  display_name?: string;
  description?: string;
  content?: string;
  tags?: string[];
}

// ── API calls ──────────────────────────────────────────────────────

export async function listSkills(projectId?: string, status?: string, limit?: number): Promise<SkillListResponse> {
  const params = new URLSearchParams();
  if (projectId) params.append("project_id", projectId);
  if (status) params.append("status", status);
  if (limit != null) params.append("limit", String(limit));
  const response = await authFetch(`/api/skills/skills?${params}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to fetch skills");
  }
  return (await response.json()) as SkillListResponse;
}

export async function getSkill(skillId: string): Promise<SkillResponse> {
  const response = await authFetch(`/api/skills/skills/${skillId}`);
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to fetch skill");
  }
  return (await response.json()) as SkillResponse;
}

export async function createSkill(projectId: string, body: SkillCreateRequest): Promise<SkillResponse> {
  const params = new URLSearchParams({ project_id: projectId });
  const response = await authFetch(`/api/skills/skills?${params}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to create skill");
  }
  return (await response.json()) as SkillResponse;
}

export async function updateSkill(skillId: string, body: SkillUpdateRequest): Promise<SkillResponse> {
  const response = await authFetch(`/api/skills/skills/${skillId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to update skill");
  }
  return (await response.json()) as SkillResponse;
}

export interface SkillPublishResponse {
  tag: string;
  skill_id: string;
  repository: string;
}

export async function publishSkill(skillId: string, tag: string): Promise<SkillPublishResponse> {
  const response = await authFetch(`/api/skills/skills/${skillId}/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag }),
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to publish skill");
  }
  return (await response.json()) as SkillPublishResponse;
}

export async function deleteSkill(skillId: string): Promise<void> {
  const response = await authFetch(`/api/skills/skills/${skillId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const err = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(err?.detail ?? "Failed to delete skill");
  }
}
