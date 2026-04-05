// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import { authFetch } from "@/api/auth";

// ── Types ────────────────────────────────────────────────────

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OpenAIChatRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  top_k: number;
  repetition_penalty: number;
  image_base64?: string;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string;
  /** Qwen3 / OpenAI-compatible reasoning tokens (separate from content). */
  reasoning_content?: string;
}

export interface OpenAIChatChunkChoice {
  delta?: OpenAIChatDelta;
  finish_reason?: string | null;
}

export interface OpenAIChatChunk {
  choices?: OpenAIChatChunkChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  timings?: Record<string, number>;
}

// ── Helpers ──────────────────────────────────────────────────

export function parseErrorText(status: number, body: unknown): string {
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    typeof body.detail === "string"
  ) {
    return body.detail;
  }
  if (
    body &&
    typeof body === "object" &&
    "message" in body &&
    typeof body.message === "string"
  ) {
    return body.message;
  }
  return `Request failed (${status})`;
}

function parseSseEvent(rawEvent: string): string[] {
  const dataLines: string[] = [];
  for (const line of rawEvent.split(/\r?\n/)) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  return dataLines;
}

// ── Streaming ────────────────────────────────────────────────

export async function* streamChatCompletions(
  endpoint: string,
  payload: OpenAIChatRequest,
  signal: AbortSignal,
): AsyncGenerator<OpenAIChatChunk> {
  const response = await authFetch(`${endpoint}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(parseErrorText(response.status, body));
  }

  if (!response.body) {
    throw new Error("Stream response missing body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let separatorIndex = buffer.search(/\r?\n\r?\n/);
    while (separatorIndex >= 0) {
      const rawEvent = buffer.slice(0, separatorIndex);
      const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
      buffer = buffer.slice(separatorIndex + separatorLength);

      const dataLines = parseSseEvent(rawEvent);
      if (dataLines.length === 0) {
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }

      const dataText = dataLines.join("\n");
      if (dataText === "[DONE]") {
        return;
      }

      const parsed = JSON.parse(dataText) as
        | OpenAIChatChunk
        | {
            type?: string;
            content?: string;
            error?: { message?: string };
          };
      if ("error" in parsed && parsed.error) {
        throw new Error(parsed.error.message || "Stream error");
      }
      // Tool status events are custom SSE payloads, not OpenAI chunks
      if ("type" in parsed && parsed.type === "tool_status") {
        yield {
          _toolStatus: parsed.content ?? "",
        } as unknown as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }
      // Tool start/end events carry full input/output
      if (
        "type" in parsed &&
        (parsed.type === "tool_start" || parsed.type === "tool_end")
      ) {
        yield { _toolEvent: parsed } as unknown as OpenAIChatChunk;
        separatorIndex = buffer.search(/\r?\n\r?\n/);
        continue;
      }
      yield parsed as OpenAIChatChunk;

      // The proxy can batch many SSE events into a single network chunk.
      // Without this yield, the inner while-loop processes all buffered
      // events synchronously and React never gets a chance to re-render
      // until the entire stream ends. Yielding to the macrotask queue
      // between events lets React commit intermediate renders.
      separatorIndex = buffer.search(/\r?\n\r?\n/);
      if (separatorIndex >= 0) {
        await new Promise<void>((r) => setTimeout(r, 0));
      }
    }
  }
}
