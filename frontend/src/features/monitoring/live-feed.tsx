// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LIVE_FEED } from "./monitoring-data";

export function LiveFeed() {
  const [paused, setPaused] = useState(false);

  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-2.5 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-[5px] h-[5px] rounded-full bg-primary animate-pulse shrink-0" />
          <span className="text-xs font-semibold text-foreground font-display">Live Feed</span>
        </div>
        <Button
          variant={paused ? "outline" : "ghost"}
          size="xs"
          className="text-[9px]"
          onClick={() => setPaused(!paused)}
        >
          {paused ? "\u25B6 Resume" : "\u23F8 Pause"}
        </Button>
      </div>

      {!paused && (
        <div className="h-px bg-card overflow-hidden relative">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/25 to-transparent animate-[scan-line_2s_linear_infinite]" />
        </div>
      )}

      <ScrollArea className="max-h-[260px]">
        {LIVE_FEED.map((e, i) => (
          <div
            key={i}
            className="px-4 py-[5px] flex items-center gap-2 border-b border-background text-[10px] font-mono"
          >
            <span className="text-muted-foreground w-[22px] text-right shrink-0 text-[8px]">
              {e.time}
            </span>
            <span
              className="w-1 h-1 rounded-full shrink-0"
              style={{ background: e.color }}
            />
            <span className="text-foreground/60 w-[100px] shrink-0 truncate text-[9px]">
              {e.agent}
            </span>
            <span
              className="truncate text-[9px]"
              style={{ color: e.type === "error" ? "#EF4444" : "var(--muted-foreground)" }}
            >
              {e.detail}
            </span>
          </div>
        ))}
      </ScrollArea>
    </section>
  );
}
