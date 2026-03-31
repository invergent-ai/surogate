// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { EVENT_COLORS } from "./models-data";
import type { Model } from "./models-data";

export function EventsTab({ model }: { model: Model }) {
  return (
    <div className="animate-in fade-in duration-150">
      <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Recent Events
          </span>
        </div>

        {model.events.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground/40 text-xs">
            No events recorded
          </div>
        ) : (
          <div className="py-1">
            {model.events.map((e, i) => (
              <div
                key={i}
                className="px-4 py-2.5 border-b border-border/50 last:border-b-0 flex items-start gap-3 hover:bg-muted/30 transition-colors cursor-pointer"
              >
                <span className="text-[9px] text-muted-foreground/40 w-7 text-right shrink-0 mt-0.5 font-display">
                  {e.time}
                </span>
                <div
                  className="w-1.5 h-1.5 rounded-full shrink-0 mt-1.5"
                  style={{
                    backgroundColor: EVENT_COLORS[e.type] ?? "#6B7585",
                    boxShadow:
                      e.type === "error"
                        ? `0 0 6px ${EVENT_COLORS.error}`
                        : "none",
                  }}
                />
                <span
                  className="text-[11px] leading-snug"
                  style={{
                    color:
                      e.type === "error" ? "#EF4444" : undefined,
                  }}
                >
                  <span className={e.type !== "error" ? "text-muted-foreground" : undefined}>
                    {e.text}
                  </span>
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
