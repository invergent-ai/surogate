// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { ANNOTATION_STYLES } from "./conversations-data";
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

// ── Trajectory tab ────────────────────────────────────────────

export function TrajectoryTab({ convo }: { convo: Conversation }) {
  return (
    <div className="animate-in fade-in duration-150">
      <div className="text-xs font-semibold text-foreground font-display mb-3">
        Conversation Trajectory
      </div>

      {/* timeline */}
      <div className="relative pl-7">
        {convo.messages.map((msg, i) => {
          const hasTool = msg.tools && msg.tools.length > 0;
          const hasAnnotation = !!msg.annotation;
          const isUser = msg.role === "user";

          const dotColor = hasAnnotation
            ? ANNOTATION_STYLES[msg.annotation!.type]?.fg
            : hasTool
              ? "#22C55E"
              : isUser
                ? "var(--color-muted-foreground)"
                : "var(--color-border)";

          return (
            <div key={i} className="relative mb-1">
              {/* vertical line */}
              <div className="absolute -left-5 top-0 bottom-0 w-px bg-border" />
              {/* dot */}
              <div
                className="absolute -left-6 top-1.5 w-2 h-2 rounded-full border-2"
                style={{
                  background: dotColor,
                  borderColor: dotColor,
                }}
              />

              <div
                className="px-3 py-1.5 rounded-md border"
                style={{
                  background: hasAnnotation
                    ? ANNOTATION_STYLES[msg.annotation!.type]?.bg
                    : "transparent",
                  borderColor: hasAnnotation
                    ? ANNOTATION_STYLES[msg.annotation!.type]?.border
                    : "transparent",
                }}
              >
                {/* header */}
                <div className="flex items-center gap-1.5 text-[10px]">
                  <span
                    className="font-medium font-display"
                    style={{
                      color: isUser
                        ? "var(--color-muted-foreground)"
                        : convo.agentColor,
                    }}
                  >
                    {isUser ? "User" : "Agent"}
                  </span>
                  <span className="text-muted-foreground/30 text-[9px]">
                    {msg.timestamp}
                  </span>
                  <span className="text-muted-foreground/30 text-[9px]">
                    {msg.tokens} tok
                  </span>
                  {msg.latency && (
                    <span className="text-muted-foreground/30 text-[9px]">
                      {msg.latency}ms
                    </span>
                  )}
                </div>

                {/* tool calls */}
                {hasTool && (
                  <div className="flex gap-1 mt-0.5">
                    {msg.tools!.map((t, ti) => (
                      <span
                        key={ti}
                        className="text-[8px] px-1.5 py-px rounded font-medium border"
                        style={{
                          background: "#22C55E10",
                          color: "#22C55E",
                          borderColor: "#22C55E20",
                        }}
                      >
                        &#x26A1; {t.name}.{t.action} ({t.latency}ms)
                      </span>
                    ))}
                  </div>
                )}

                {/* annotation */}
                {hasAnnotation && (
                  <div
                    className="text-[9px] mt-0.5 leading-snug"
                    style={{
                      color: ANNOTATION_STYLES[msg.annotation!.type]?.fg,
                    }}
                  >
                    <span className="font-semibold">
                      {ANNOTATION_STYLES[msg.annotation!.type]?.label}:
                    </span>{" "}
                    {msg.annotation!.note}
                  </div>
                )}

                {/* content preview */}
                <div className="text-[10px] text-muted-foreground mt-0.5 truncate max-w-[500px]">
                  {msg.content.substring(0, 100)}
                  {msg.content.length > 100 ? "..." : ""}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* trajectory stats */}
      <div className="mt-5 grid grid-cols-3 gap-2.5">
        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Tool Calls
          </div>
          <div className="text-xl font-bold text-green-500">
            {convo.toolCalls}
          </div>
          <div className="text-[9px] text-muted-foreground/30 mt-0.5">
            {[
              ...new Set(
                convo.messages.flatMap((m) =>
                  (m.tools || []).map((t) => t.name),
                ),
              ),
            ].join(", ") || "none"}
          </div>
        </div>

        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Annotations
          </div>
          <div className="text-xl font-bold text-amber-500">
            {convo.messages.filter((m) => m.annotation).length}
          </div>
          <div className="text-[9px] text-muted-foreground/30 mt-0.5">
            {convo.messages
              .filter((m) => m.annotation)
              .map((m) => m.annotation!.type.replace("_", " "))
              .join(", ") || "none"}
          </div>
        </div>

        <div className="bg-muted/40 border border-border rounded-lg p-3">
          <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
            Sentiment Arc
          </div>
          <SentimentBar score={convo.sentimentScore} width={100} />
          <div className="text-[9px] text-muted-foreground/30 mt-1">
            {convo.resolved ? "\u2713 Resolved" : "\u2717 Unresolved"} &middot;{" "}
            {convo.escalated ? "Escalated" : "Not escalated"}
          </div>
        </div>
      </div>
    </div>
  );
}
