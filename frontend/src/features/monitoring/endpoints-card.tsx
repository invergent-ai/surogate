// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot } from "@/components/ui/status-dot";
import { ENDPOINTS } from "./monitoring-data";

export function EndpointsCard() {
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-2.5 border-b border-border flex items-center gap-2">
        <span className="text-primary">&#x22A1;</span>
        <span className="text-xs font-semibold text-foreground font-display">Endpoints</span>
      </div>
      {ENDPOINTS.map((ep) => (
        <div
          key={ep.path}
          className="px-4 py-[7px] border-b border-border/50 flex items-center justify-between text-[10px]"
        >
          <div className="flex items-center gap-2">
            <StatusDot status={ep.status === "up" ? "serving" : "deploying"} />
            <code className="text-muted-foreground text-[9px]">{ep.method}</code>
            <code className="text-foreground/80 font-medium">{ep.path}</code>
          </div>
          <div className="flex items-center gap-2 text-[9px] text-muted-foreground">
            <span>{ep.agents} agents</span>
            <span className="text-muted-foreground/50">·</span>
            <span>{ep.avgMs}ms</span>
          </div>
        </div>
      ))}
    </section>
  );
}
