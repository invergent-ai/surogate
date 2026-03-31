// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import type { Conversation } from "./conversations-data";

// ── Sentiment bar ─────────────────────────────────────────────

function SentimentBar({ score, width = 80 }: { score: number; width?: number }) {
  const color = score > 0.65 ? "#22C55E" : score > 0.4 ? "#F59E0B" : "#EF4444";
  return (
    <div className="flex items-center gap-1.5">
      <div
        className="h-1 rounded-sm overflow-hidden bg-muted"
        style={{ width }}
      >
        <div
          className="h-full rounded-sm transition-[width] duration-300"
          style={{ width: `${score * 100}%`, background: color }}
        />
      </div>
      <span className="text-[9px] font-medium" style={{ color }}>
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// ── Detail row ────────────────────────────────────────────────

function DetailRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1 border-b border-border/50">
      <span className="text-[10px] text-muted-foreground/40">{label}</span>
      <span className="text-[10px] text-foreground/70">{value}</span>
    </div>
  );
}

// ── Metadata tab ──────────────────────────────────────────────

export function MetadataTab({ convo }: { convo: Conversation }) {
  return (
    <div className="animate-in fade-in duration-150">
      <div className="grid grid-cols-2 gap-4">
        {/* Conversation Details */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Conversation Details
          </div>
          <DetailRow label="ID" value={convo.id} />
          <DetailRow label="Agent" value={convo.agent} />
          <DetailRow label="User" value={convo.user} />
          <DetailRow label="Status" value={convo.status} />
          <DetailRow label="Started" value={convo.timestamp} />
          <DetailRow label="Duration" value={convo.duration} />
          <DetailRow label="Turns" value={convo.turns} />
          <DetailRow label="Resolved" value={convo.resolved ? "Yes" : "No"} />
          <DetailRow label="Escalated" value={convo.escalated ? "Yes" : "No"} />
          <DetailRow label="Dataset" value={convo.dataset || "\u2014"} />
        </section>

        {/* Token & Performance */}
        <section className="bg-muted/40 border border-border rounded-lg p-3.5">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Token & Performance
          </div>
          <DetailRow
            label="Tokens In"
            value={convo.tokens.in.toLocaleString()}
          />
          <DetailRow
            label="Tokens Out"
            value={convo.tokens.out.toLocaleString()}
          />
          <DetailRow
            label="Total Tokens"
            value={(convo.tokens.in + convo.tokens.out).toLocaleString()}
          />
          <DetailRow label="Avg Latency" value={`${convo.latencyAvg}ms`} />
          <DetailRow label="Tool Calls" value={convo.toolCalls} />
          <DetailRow
            label="Sentiment"
            value={`${convo.sentiment} (${(convo.sentimentScore * 100).toFixed(0)}%)`}
          />

          <div className="mt-3">
            <div className="text-[10px] text-muted-foreground/50 mb-1 font-display">
              Sentiment
            </div>
            <SentimentBar score={convo.sentimentScore} width={200} />
          </div>
        </section>
      </div>

      {/* Skills & Tools Used */}
      <section className="bg-muted/40 border border-border rounded-lg p-3.5 mt-4">
        <div className="text-xs font-semibold text-foreground font-display mb-2.5">
          Skills & Tools Used
        </div>
        <div className="flex flex-wrap gap-1.5">
          {[
            ...new Set(
              convo.messages.flatMap((m) =>
                (m.tools || []).map((t) => t.name),
              ),
            ),
          ].map((skill) => {
            const calls = convo.messages.flatMap((m) =>
              (m.tools || []).filter((t) => t.name === skill),
            );
            const avgLat = Math.round(
              calls.reduce((s, c) => s + c.latency, 0) / calls.length,
            );
            return (
              <div
                key={skill}
                className="px-2.5 py-1.5 rounded-md bg-muted border border-border"
              >
                <div className="text-[11px] text-foreground font-medium mb-0.5">
                  &#x26A1; {skill}
                </div>
                <div className="text-[9px] text-muted-foreground/40">
                  {calls.length} calls &middot; avg {avgLat}ms
                </div>
              </div>
            );
          })}
          {convo.messages.every(
            (m) => !m.tools || m.tools.length === 0,
          ) && (
            <div className="text-[11px] text-muted-foreground/30">
              No tool calls in this conversation
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
