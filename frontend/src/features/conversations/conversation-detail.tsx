// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { SENTIMENT_COLORS } from "./conversations-data";
import { ThreadTab } from "./thread-tab";
import { TrajectoryTab } from "./trajectory-tab";
import { MetadataTab } from "./metadata-tab";
import type { Conversation } from "./conversations-data";

// ── Conversation detail panel ─────────────────────────────────

export function ConversationDetail({
  convo,
  onExport,
}: {
  convo: Conversation;
  onExport: () => void;
}) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span
                className="inline-block w-2 h-2 rounded-full shrink-0"
                style={{
                  backgroundColor: SENTIMENT_COLORS[convo.sentiment],
                }}
              />
              <span className="text-sm font-bold text-foreground font-display tracking-tight">
                {convo.agent}
              </span>
              <span className="text-[11px] text-muted-foreground/30">
                &harr;
              </span>
              <span className="text-[12px] text-muted-foreground font-medium">
                {convo.user}
              </span>
              <code className="text-[9px] text-muted-foreground/30 bg-muted px-1.5 py-px rounded">
                {convo.id}
              </code>
              {convo.status === "active" && (
                <span className="inline-flex items-center gap-1 text-[9px] text-green-500 font-semibold">
                  <span className="w-[5px] h-[5px] rounded-full bg-green-500 animate-pulse" />
                  LIVE
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground/30">
              <span>{convo.turns} turns</span>
              <span>{convo.duration}</span>
              <span>avg {convo.latencyAvg}ms</span>
              <span>{convo.toolCalls} tool calls</span>
              <span>
                sentiment:{" "}
                <span
                  style={{ color: SENTIMENT_COLORS[convo.sentiment] }}
                >
                  {convo.sentiment} (
                  {(convo.sentimentScore * 100).toFixed(0)}%)
                </span>
              </span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-1.5 shrink-0">
            <Button variant="outline" size="xs">
              {convo.starred ? "\u2605" : "\u2606"}
            </Button>
            <Button
              variant={convo.flagged ? "destructive" : "outline"}
              size="xs"
            >
              &#x2691; Flag
            </Button>
            <Button size="xs" onClick={onExport}>
              &rarr; Dataset
            </Button>
            <Button variant="ghost" size="icon-xs">
              &#x22EF;
            </Button>
          </div>
        </div>

        {/* Tags */}
        <div className="flex gap-1 mb-2">
          {convo.tags.map((t) => (
            <span
              key={t}
              className="text-[8px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground/50"
            >
              {t}
            </span>
          ))}
          <button
            type="button"
            className="text-[8px] px-1.5 py-0.5 rounded border border-dashed border-border bg-transparent text-muted-foreground/30 cursor-pointer hover:text-muted-foreground/50"
          >
            + tag
          </button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs
        defaultValue="thread"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-5 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="thread">Thread</TabsTrigger>
            <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
            <TabsTrigger value="metadata">Metadata</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          <TabsContent value="thread" className="mt-0">
            <ThreadTab convo={convo} />
          </TabsContent>
          <TabsContent value="trajectory" className="mt-0">
            <TrajectoryTab convo={convo} />
          </TabsContent>
          <TabsContent value="metadata" className="mt-0">
            <MetadataTab convo={convo} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────

export function ConversationEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x2B21;</div>
        <div className="font-display text-sm">
          Select a conversation to inspect
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Browse, filter, and review agent conversations. Export to datasets for
          fine-tuning.
        </div>
      </div>
    </div>
  );
}
