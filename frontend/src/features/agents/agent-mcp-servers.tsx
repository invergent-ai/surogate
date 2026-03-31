// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import type { AgentMcpServer } from "./agents-data";

interface AgentMcpServersProps {
  servers: AgentMcpServer[];
}

export function AgentMcpServers({ servers }: AgentMcpServersProps) {
  return (
    <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-violet-500">&#x229E;</span>
          <span className="text-[13px] font-semibold text-foreground font-display">
            MCP Servers
          </span>
          <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
            {servers.length}
          </span>
        </div>
        <Button variant="outline" size="xs">
          + Connect
        </Button>
      </div>

      {servers.length === 0 ? (
        <div className="py-6 text-center text-muted-foreground/40 text-[11px]">
          No MCP servers connected
        </div>
      ) : (
        servers.map((m) => (
          <div
            key={m.name}
            className="px-4 py-2.5 border-b border-border/50 last:border-b-0 flex items-center justify-between hover:bg-muted/30 transition-colors cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <StatusDot status="running" />
              <span className="text-xs font-medium text-foreground">
                {m.name}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {m.status}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[10px] text-muted-foreground/50">
                latency:{" "}
                <span className="text-muted-foreground">{m.latency}</span>
              </span>
              <Button variant="ghost" size="icon-xs">
                &#x22EF;
              </Button>
            </div>
          </div>
        ))
      )}
    </section>
  );
}
