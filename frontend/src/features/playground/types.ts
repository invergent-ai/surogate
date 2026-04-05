// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

export interface ThreadRecord {
  id: string;
  title: string;
  modelId?: string;
  pairId?: string;
  side?: "left" | "right";
  archived: boolean;
  createdAt: number;
}

export interface MessageRecord {
  id: string;
  threadId: string;
  role: import("@assistant-ui/react").ThreadMessage["role"];
  content: import("@assistant-ui/react").ThreadMessage["content"];
  attachments?: import("@assistant-ui/react").ThreadMessage["attachments"];
  metadata?: Record<string, unknown>;
  createdAt: number;
}

export type PlaygroundView =
  | { mode: "single"; threadId?: string; newThreadNonce?: string }
  | { mode: "compare"; pairId: string };
