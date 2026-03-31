// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { MCP_SERVERS, toStatus } from "./skills-data";
import type { McpServer } from "./skills-data";

// ── List item ───────────────────────────────────────────────────

function McpListItem({
  server,
  selected,
  onSelect,
}: {
  server: McpServer;
  selected: boolean;
  onSelect: () => void;
}) {
  const connected = server.status === "connected";
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-4 py-3.5 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-violet-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-7 h-7 rounded-md flex items-center justify-center text-xs border shrink-0",
              connected
                ? "bg-violet-500/10 border-violet-500/15 text-violet-500"
                : "bg-muted border-border text-muted-foreground/30",
            )}
          >
            &#x229E;
          </div>
          <div>
            <span className="text-xs font-semibold text-foreground font-display">
              {server.name}
            </span>
            <div className="text-[9px] text-muted-foreground mt-px">
              {server.transport} &middot; {server.toolCount} tools &middot;{" "}
              {server.latency}
            </div>
          </div>
        </div>
        <StatusDot status={toStatus(server.status)} />
      </div>
      <p className="text-[10px] text-muted-foreground leading-snug line-clamp-1 mb-1.5">
        {server.description}
      </p>
      <div className="flex items-center gap-1.5">
        {server.connectedAgents.slice(0, 3).map((a) => (
          <Badge key={a}>&#x2B21; {a}</Badge>
        ))}
        {server.connectedAgents.length === 0 && (
          <span className="text-[9px] text-muted-foreground/30">
            No agents connected
          </span>
        )}
      </div>
    </button>
  );
}

// ── Detail panel ────────────────────────────────────────────────

function McpDetail({
  server,
  onClose,
}: {
  server: McpServer;
  onClose: () => void;
}) {
  const connected = server.status === "connected";

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-lg bg-violet-500/10 border border-violet-500/15 flex items-center justify-center text-lg text-violet-500">
              &#x229E;
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-foreground font-display">
                  {server.name}
                </span>
                <StatusDot status={toStatus(server.status)} />
                <Badge variant={connected ? "active" : "default"}>
                  {server.status.toUpperCase()}
                </Badge>
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">
                {server.transport} &middot; Auth: {server.auth} &middot; Updated{" "}
                {server.updatedAt}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5">
            {connected ? (
              <Button variant="destructive" size="xs">
                Disconnect
              </Button>
            ) : (
              <Button size="xs">Connect</Button>
            )}
            <Button variant="ghost" size="icon-xs" onClick={onClose}>
              &#x2715;
            </Button>
          </div>
        </div>
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          {server.description}
        </p>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {/* Connection info */}
        <div className="bg-muted/40 border border-border rounded-md px-3.5 py-3">
          <div className="text-[9px] font-semibold text-muted-foreground/40 uppercase tracking-wide mb-2 font-display">
            Connection
          </div>
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: "URL", value: server.url, span: true },
              { label: "Transport", value: server.transport },
              { label: "Auth", value: server.auth },
              { label: "Latency", value: server.latency },
              { label: "Tools", value: String(server.toolCount) },
            ].map((f) => (
              <div
                key={f.label}
                className={f.span ? "col-span-2" : undefined}
              >
                <div className="text-[8px] text-muted-foreground/40 uppercase mb-px font-display">
                  {f.label}
                </div>
                <code className="text-[10px] text-foreground/70 break-all">
                  {f.value}
                </code>
              </div>
            ))}
          </div>
        </div>

        {/* Exposed tools */}
        <div>
          <div className="text-[11px] font-semibold text-foreground font-display mb-2 flex items-center gap-1.5">
            <span className="text-violet-500">&#x2699;</span> Exposed Tools
            <span className="text-[9px] text-muted-foreground font-normal">
              {server.toolCount}
            </span>
          </div>
          <div className="flex flex-wrap gap-1">
            {server.tools.map((t) => (
              <code
                key={t}
                className="text-[10px] px-2 py-1 rounded bg-muted/40 border border-border text-foreground/70"
              >
                {t}
              </code>
            ))}
          </div>
        </div>

        {/* Connected agents */}
        <div>
          <div className="text-[11px] font-semibold text-foreground font-display mb-2 flex items-center gap-1.5">
            <span className="text-amber-500">&#x2B21;</span> Connected Agents
          </div>
          {server.connectedAgents.length === 0 ? (
            <div className="py-3 text-center text-muted-foreground/30 text-[10px] bg-muted/40 rounded-md border border-border">
              No agents connected
            </div>
          ) : (
            server.connectedAgents.map((a) => (
              <div
                key={a}
                className="px-3 py-2 bg-muted/40 border border-border rounded-md mb-1 flex items-center gap-1.5"
              >
                <StatusDot status="running" />
                <span className="text-[11px] text-foreground">{a}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function McpEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x229E;</div>
        <div className="font-display text-sm">
          Select an MCP server to view details
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          MCP servers expose tools that agents can call. Connect servers to make
          their tools available to your agents.
        </div>
      </div>
    </div>
  );
}

// ── Main ────────────────────────────────────────────────────────

export function McpServersTab() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = selectedId
    ? MCP_SERVERS.find((m) => m.id === selectedId) ?? null
    : null;

  const connectedCount = MCP_SERVERS.filter(
    (m) => m.status === "connected",
  ).length;

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* List */}
      <div className="w-[420px] min-w-[420px] border-r border-border flex flex-col">
        <div className="px-4 py-3 border-b border-border flex justify-between items-center">
          <span className="text-[10px] text-muted-foreground">
            {connectedCount}/{MCP_SERVERS.length} connected
          </span>
          <Button size="xs">+ Connect Server</Button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {MCP_SERVERS.map((m) => (
            <McpListItem
              key={m.id}
              server={m}
              selected={selectedId === m.id}
              onSelect={() => setSelectedId(m.id)}
            />
          ))}
        </div>
      </div>

      {/* Detail / Empty */}
      {selected ? (
        <McpDetail server={selected} onClose={() => setSelectedId(null)} />
      ) : (
        <McpEmptyState />
      )}
    </div>
  );
}
