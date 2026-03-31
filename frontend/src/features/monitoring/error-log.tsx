// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ERROR_LOG } from "./monitoring-data";

const FILTERS = ["All", "5xx", "Timeout", "Guardrail"] as const;

export function ErrorLog() {
  const [activeFilter, setActiveFilter] = useState<string>("All");

  const filtered = ERROR_LOG.filter((e) => {
    if (activeFilter === "All") return true;
    if (activeFilter === "5xx") return e.status >= 500;
    if (activeFilter === "Timeout") return e.type === "TOOL_TIMEOUT";
    if (activeFilter === "Guardrail") return e.type === "GUARDRAIL";
    return true;
  });

  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-destructive">&#x2715;</span>
          <span className="text-[13px] font-semibold text-foreground font-display">Error Log</span>
          <Badge variant="danger">{ERROR_LOG.length}</Badge>
        </div>
        <div className="flex gap-1">
          {FILTERS.map((f) => (
            <Button
              key={f}
              variant={activeFilter === f ? "secondary" : "ghost"}
              size="xs"
              className="text-[9px]"
              onClick={() => setActiveFilter(f)}
            >
              {f}
            </Button>
          ))}
        </div>
      </div>
      {filtered.map((e, i) => (
        <div
          key={i}
          className="px-4 py-2 border-b border-border/50 flex items-start gap-2.5 cursor-pointer hover:bg-muted/50 transition-colors"
        >
          <span className="text-[9px] text-muted-foreground w-[46px] text-right shrink-0 mt-0.5 font-display">
            {e.time}
          </span>
          <span
            className="text-[7px] px-[5px] py-[2px] rounded shrink-0 mt-px font-semibold"
            style={{
              background: e.status >= 500 ? "#EF444415" : e.status === 429 ? "#F59E0B15" : "#14B8A615",
              color: e.status >= 500 ? "#EF4444" : e.status === 429 ? "#F59E0B" : "#14B8A6",
            }}
          >
            {e.status}
          </span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-0.5">
              <span className="text-[10px] text-foreground/80 font-medium">{e.agent}</span>
              <Badge>{e.type}</Badge>
            </div>
            <div className="text-[10px] text-muted-foreground truncate">{e.message}</div>
          </div>
          <code className="text-[8px] text-muted-foreground shrink-0">{e.traceId}</code>
        </div>
      ))}
    </section>
  );
}
