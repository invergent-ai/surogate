// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { ANNOTATION_STYLES } from "./conversations-data";
import type { Conversation } from "./conversations-data";

export function ThreadTab({ convo }: { convo: Conversation }) {
  return (
    <div className="animate-in fade-in duration-150 space-y-3">
      {convo.messages.map((msg, i) => {
        const isUser = msg.role === "user";
        return (
          <div
            key={i}
            className="flex gap-2.5"
            style={{ flexDirection: isUser ? "row-reverse" : "row" }}
          >
            {/* avatar */}
            <div
              className="w-7 h-7 rounded-md shrink-0 flex items-center justify-center text-[10px] font-semibold font-display border"
              style={{
                backgroundColor: isUser ? undefined : `${convo.agentColor}18`,
                borderColor: isUser ? undefined : `${convo.agentColor}30`,
                color: isUser ? undefined : convo.agentColor,
              }}
            >
              {isUser ? "U" : "A"}
            </div>

            {/* bubble */}
            <div
              className="max-w-[75%] border px-3.5 py-2.5 relative"
              style={{
                background: isUser
                  ? "var(--color-muted)"
                  : "var(--color-card)",
                borderColor: "var(--color-border)",
                borderRadius: isUser
                  ? "10px 10px 4px 10px"
                  : "10px 10px 10px 4px",
              }}
            >
              {/* tool calls */}
              {msg.tools && msg.tools.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {msg.tools.map((t, ti) => (
                    <span
                      key={ti}
                      className="text-[9px] px-2 py-0.5 rounded flex items-center gap-1 font-display border"
                      style={{
                        background:
                          t.status === "success" ? "#22C55E08" : "#EF444408",
                        borderColor:
                          t.status === "success" ? "#22C55E20" : "#EF444420",
                        color:
                          t.status === "success" ? "#22C55E" : "#EF4444",
                      }}
                    >
                      <span className="text-[8px]">&#x26A1;</span>
                      <span className="font-medium">{t.name}</span>
                      <span className="opacity-30">&rarr;</span>
                      <span>{t.action}</span>
                      <span className="opacity-30">&middot;</span>
                      <span className="opacity-50">{t.latency}ms</span>
                    </span>
                  ))}
                </div>
              )}

              {/* content */}
              <div className="text-[12px] text-foreground leading-relaxed whitespace-pre-wrap break-words">
                {msg.content}
              </div>

              {/* annotation */}
              {msg.annotation && (
                <div
                  className="mt-2 px-2.5 py-1.5 rounded"
                  style={{
                    background: ANNOTATION_STYLES[msg.annotation.type]?.bg,
                    border: `1px solid ${ANNOTATION_STYLES[msg.annotation.type]?.border}`,
                  }}
                >
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span
                      className="text-[7px] px-1 py-px rounded font-bold tracking-wide"
                      style={{
                        background: `${ANNOTATION_STYLES[msg.annotation.type]?.fg}20`,
                        color: ANNOTATION_STYLES[msg.annotation.type]?.fg,
                      }}
                    >
                      {ANNOTATION_STYLES[msg.annotation.type]?.label}
                    </span>
                  </div>
                  <div
                    className="text-[10px] leading-snug opacity-85"
                    style={{
                      color: ANNOTATION_STYLES[msg.annotation.type]?.fg,
                    }}
                  >
                    {msg.annotation.note}
                  </div>
                </div>
              )}

              {/* meta footer */}
              <div className="flex items-center justify-between mt-1.5 text-[9px] text-muted-foreground/30 group-hover:text-muted-foreground/50 transition-opacity">
                <span>{msg.timestamp}</span>
                <div className="flex gap-2">
                  <span>{msg.tokens} tok</span>
                  {msg.latency && <span>{msg.latency}ms</span>}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
