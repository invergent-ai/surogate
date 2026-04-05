// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//

import type { ChatModelAdapter } from "@assistant-ui/react";
import type { MessageTiming, ToolCallMessagePart } from "@assistant-ui/core";
import { toast } from "sonner";
import { streamChatCompletions } from "./playground-api";
import { usePlaygroundStore } from "../stores/playground-store";
import { useAppStore } from "@/stores/app-store";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";

/** Server-side usage data from vLLM / TGI. */
interface ServerUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** Server-side timing data. */
interface ServerTimings {
  prompt_n: number;
  cache_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
}

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

// ── Helpers ──────────────────────────────────────────────────

function estimateTokenCount(text: string): number | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  return Math.max(1, Math.round(trimmed.length / 4));
}

function buildTiming(
  streamStartTime: number,
  totalChunks: number,
  firstTokenTime?: number,
  totalStreamTime?: number,
  tokenCount?: number,
  toolCallCount = 0,
  tokensPerSecondOverride?: number,
): MessageTiming {
  return {
    streamStartTime,
    firstTokenTime,
    totalStreamTime,
    tokenCount,
    tokensPerSecond:
      tokensPerSecondOverride ??
      (typeof totalStreamTime === "number" &&
      totalStreamTime > 0 &&
      typeof tokenCount === "number"
        ? tokenCount / (totalStreamTime / 1000)
        : undefined),
    totalChunks,
    toolCallCount,
  };
}

function collectTextParts(message: RunMessage): string[] {
  const textParts = message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text);

  if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
    for (const attachment of message.attachments ?? []) {
      for (const part of attachment.content ?? []) {
        if (part.type === "text") {
          textParts.push(part.text);
        }
      }
    }
  }

  return textParts;
}

function toOpenAIMessage(
  message: RunMessage,
): { role: "system" | "user" | "assistant"; content: string } | null {
  if (
    message.role !== "system" &&
    message.role !== "user" &&
    message.role !== "assistant"
  ) {
    return null;
  }

  const content = collectTextParts(message).join("\n");
  return { role: message.role, content };
}

function extractImageBase64(input: string): string | undefined {
  if (!input) return undefined;
  if (input.startsWith("data:")) {
    const commaIndex = input.indexOf(",");
    return commaIndex >= 0 ? input.slice(commaIndex + 1) : undefined;
  }
  return input;
}

function findLatestUserImageBase64(
  messages: RunMessages,
): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") continue;

    for (const part of message.content ?? []) {
      if (part.type === "image" && "image" in part) {
        const encoded = extractImageBase64(part.image);
        if (encoded) return encoded;
      }
    }

    if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
      for (const attachment of message.attachments ?? []) {
        for (const part of attachment.content ?? []) {
          if (part.type !== "image") continue;
          const encoded = extractImageBase64(
            (part as { image: string }).image,
          );
          if (encoded) return encoded;
        }
      }
    }
  }

  return undefined;
}

// ── Adapter ──────────────────────────────────────────────────

export function createPlaygroundAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      const store = usePlaygroundStore.getState();
      const { params, systemPrompt } = store;

      // Resolve the active model from the app store
      const allModels = useAppStore.getState().models;
      const model = allModels.find(
        (m) => m.id === store.selectedModelId,
      );

      if (!model?.endpoint) {
        toast.error("No model selected", {
          description: "Pick a serving model from the top bar.",
        });
        throw new Error("No model endpoint available");
      }

      // Convert messages to OpenAI format
      const outboundMessages = messages
        .map(toOpenAIMessage)
        .filter(
          (m): m is NonNullable<typeof m> => Boolean(m),
        );

      if (systemPrompt.trim()) {
        outboundMessages.unshift({
          role: "system",
          content: systemPrompt.trim(),
        });
      }

      const imageBase64 = findLatestUserImageBase64(messages);

      const threadKey = unstable_threadId || "__default";
      store.setThreadRunning(threadKey, true);

      const streamStartTime = Date.now();
      let firstTokenTime: number | undefined;
      let totalChunks = 0;
      type ContentPart = NonNullable<
        import("@assistant-ui/react").ChatModelRunResult["content"]
      >[number];
      let cumulativeText = "";
      let cumulativeReasoning = "";
      let reasoningStartAt: number | null = null;
      let reasoningDuration = 0;
      const toolCallParts: ToolCallMessagePart[] = [];
      let serverMetadata: {
        usage?: ServerUsage;
        timings?: ServerTimings;
      } | null = null;

      try {
        const stream = streamChatCompletions(
          model.endpoint,
          {
            model: model.name,
            messages: outboundMessages,
            stream: true,
            temperature: params.temperature,
            top_p: params.topP,
            max_tokens: params.maxTokens,
            top_k: params.topK,
            repetition_penalty: params.repPenalty,
            ...(imageBase64 ? { image_base64: imageBase64 } : {}),
          },
          abortSignal,
        );

        for await (const chunk of stream) {
          // Handle tool status events
          const toolStatusText = (
            chunk as unknown as { _toolStatus?: string }
          )._toolStatus;
          if (toolStatusText !== undefined) {
            store.setToolStatus(toolStatusText || null);
            continue;
          }

          // Handle tool start/end events
          const toolEvent = (
            chunk as unknown as { _toolEvent?: Record<string, unknown> }
          )._toolEvent;
          if (toolEvent !== undefined) {
            if (toolEvent.type === "tool_start") {
              const id =
                (toolEvent.tool_call_id as string) ||
                `${toolEvent.tool_name}_${Date.now()}`;
              const toolArgs = (toolEvent.arguments ?? {}) as ToolCallMessagePart["args"];
              toolCallParts.push({
                type: "tool-call" as const,
                toolCallId: id,
                toolName: toolEvent.tool_name as string,
                argsText: JSON.stringify(toolArgs),
                args: toolArgs,
              });
            } else if (toolEvent.type === "tool_end") {
              const id =
                (toolEvent.tool_call_id as string) ||
                toolCallParts[toolCallParts.length - 1]?.toolCallId ||
                "";
              const idx = toolCallParts.findIndex(
                (p) => p.toolCallId === id,
              );
              if (idx !== -1) {
                toolCallParts[idx] = {
                  ...toolCallParts[idx],
                  result: toolEvent.result as string,
                };
              }
            }
            const textParts = parseAssistantContent(cumulativeText);
            yield {
              content: [...toolCallParts, ...textParts],
              metadata: {
                timing: buildTiming(
                  streamStartTime,
                  totalChunks,
                  firstTokenTime,
                ),
                custom: { reasoningDuration },
              },
            };
            continue;
          }

          // Usage-only chunk (choices empty, usage populated)
          if (chunk.choices?.length === 0 && chunk.usage) {
            serverMetadata = {
              usage: chunk.usage,
              timings: chunk.timings as unknown as ServerTimings | undefined,
            };
            continue;
          }

          const choiceDelta = chunk.choices?.[0]?.delta;
          const contentDelta = choiceDelta?.content;
          const reasoningDelta = choiceDelta?.reasoning_content;

          if (!contentDelta && !reasoningDelta) continue;

          totalChunks += 1;

          if (firstTokenTime === undefined) {
            firstTokenTime = Date.now() - streamStartTime;
          }

          // Accumulate reasoning_content (Qwen3 / OpenAI-style)
          if (reasoningDelta) {
            cumulativeReasoning += reasoningDelta;
            if (!reasoningStartAt) {
              reasoningStartAt = Date.now();
            }
          }

          // Accumulate regular content
          if (contentDelta) {
            cumulativeText += contentDelta;
            // When switching from reasoning to content, record reasoning duration
            if (reasoningStartAt && !reasoningDuration) {
              reasoningDuration = Math.round(
                (Date.now() - reasoningStartAt) / 1000,
              );
            }
          }

          // Build content parts: use explicit reasoning_content if present,
          // otherwise fall back to <think> tag parsing in content
          let parts: ContentPart[];
          if (cumulativeReasoning) {
            // Server sends reasoning via separate field
            parts = [];
            parts.push({
              type: "reasoning" as const,
              text: cumulativeReasoning,
            });
            if (cumulativeText) {
              parts.push({
                type: "text" as const,
                text: cumulativeText,
              });
            }
          } else {
            // Fall back to <think> tag parsing for models that embed reasoning in content
            parts = parseAssistantContent(cumulativeText);

            if (
              parts.some((part) => part.type === "reasoning") &&
              !reasoningStartAt
            ) {
              reasoningStartAt = Date.now();
            }
            if (
              hasClosedThinkTag(cumulativeText) &&
              reasoningStartAt &&
              !reasoningDuration
            ) {
              reasoningDuration = Math.round(
                (Date.now() - reasoningStartAt) / 1000,
              );
            }
          }

          if (parts.length > 0 || toolCallParts.length > 0) {
            yield {
              content: [...toolCallParts, ...parts],
              metadata: {
                timing: buildTiming(
                  streamStartTime,
                  totalChunks,
                  firstTokenTime,
                ),
                custom: { reasoningDuration },
              },
            };
          }
        }

        // Final yield with server metadata
        const meta = serverMetadata;
        const finalTokenCount =
          meta?.usage?.completion_tokens ??
          estimateTokenCount(cumulativeText);
        const finalTokPerSec = meta?.timings?.predicted_per_second;
        const serverPromptEvalTime = meta?.timings?.prompt_ms;

        if (
          meta?.usage &&
          typeof meta.usage.prompt_tokens === "number" &&
          typeof meta.usage.completion_tokens === "number" &&
          typeof meta.usage.total_tokens === "number"
        ) {
          usePlaygroundStore.getState().setContextUsage({
            promptTokens: meta.usage.prompt_tokens,
            completionTokens: meta.usage.completion_tokens,
            totalTokens: meta.usage.total_tokens,
            cachedTokens: meta.timings?.cache_n ?? 0,
          });
        }

        const finalTiming = buildTiming(
          streamStartTime,
          totalChunks,
          serverPromptEvalTime ?? firstTokenTime,
          Date.now() - streamStartTime,
          finalTokenCount,
          toolCallParts.length,
          finalTokPerSec,
        );

        // Build final content parts using same dual-path logic
        const finalParts: ContentPart[] = cumulativeReasoning
          ? [
              { type: "reasoning" as const, text: cumulativeReasoning },
              ...(cumulativeText
                ? [{ type: "text" as const, text: cumulativeText }]
                : []),
            ]
          : parseAssistantContent(cumulativeText);

        yield {
          content: [
            ...toolCallParts,
            ...finalParts,
          ],
          metadata: {
            timing: finalTiming,
            custom: {
              reasoningDuration,
              serverTimings: meta?.timings ?? undefined,
              contextUsage: meta?.usage
                ? {
                    promptTokens: meta.usage.prompt_tokens,
                    completionTokens: meta.usage.completion_tokens,
                    totalTokens: meta.usage.total_tokens,
                    cachedTokens: meta.timings?.cache_n ?? 0,
                    modelId: store.selectedModelId,
                  }
                : undefined,
              timing: finalTiming,
            },
          },
        };
      } catch (err) {
        if (!abortSignal.aborted) {
          toast.error("Generation failed", {
            description:
              err instanceof Error ? err.message : "Unknown error",
          });
        }
        throw err;
      } finally {
        usePlaygroundStore.getState().setToolStatus(null);
        usePlaygroundStore.getState().setThreadRunning(threadKey, false);
      }
    },
  };
}
