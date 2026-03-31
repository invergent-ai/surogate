// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { TOOLS, TOOL_CATEGORIES, CAT_STYLES, toStatus } from "./skills-data";
import type { Tool } from "./skills-data";

// ── Category sidebar ────────────────────────────────────────────

function CategorySidebar({
  active,
  onChange,
}: {
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div className="w-[180px] min-w-[180px] border-r border-border p-3">
      <div className="text-[9px] font-semibold text-muted-foreground/40 uppercase tracking-widest mb-2 px-1">
        Categories
      </div>
      {TOOL_CATEGORIES.map((c) => {
        const count =
          c.id === "all"
            ? TOOLS.length
            : TOOLS.filter((t) => t.category === c.id).length;
        const isActive = active === c.id;
        return (
          <button
            key={c.id}
            onClick={() => onChange(c.id)}
            className={cn(
              "flex items-center gap-2 w-full px-2 py-1.5 my-px rounded-md text-[11px] font-display transition-colors cursor-pointer",
              isActive
                ? "bg-muted text-amber-500 font-semibold"
                : "text-muted-foreground hover:bg-muted/50",
            )}
          >
            <span className="text-xs w-[18px] text-center">{c.icon}</span>
            <span className="flex-1 text-left">{c.label}</span>
            <span
              className={cn(
                "text-[9px]",
                isActive ? "text-amber-500/60" : "text-muted-foreground/30",
              )}
            >
              {count}
            </span>
          </button>
        );
      })}
      <div className="mt-4">
        <Button size="xs" className="w-full">
          + New Tool
        </Button>
      </div>
    </div>
  );
}

// ── Tool list item ──────────────────────────────────────────────

function ToolListItem({
  tool,
  selected,
  onSelect,
}: {
  tool: Tool;
  selected: boolean;
  onSelect: () => void;
}) {
  const cs = CAT_STYLES[tool.category] ?? CAT_STYLES.tool;
  const catIcon =
    TOOL_CATEGORIES.find((c) => c.id === tool.category)?.icon ?? "\u2699";

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-2.5 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-current"
          : "border-l-transparent hover:bg-muted/30",
      )}
      style={selected ? { borderLeftColor: `var(--color-${tool.category === "tool" ? "blue" : tool.category === "rag" ? "green" : tool.category === "workflow" ? "amber" : tool.category === "guardrail" ? "red" : "violet"}-500)` } : undefined}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-6 h-6 rounded flex items-center justify-center text-[10px] border",
              cs.bg,
              cs.fg,
              cs.border,
            )}
          >
            {catIcon}
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-semibold text-foreground font-display">
              {tool.displayName}
            </span>
            <Badge>v{tool.version}</Badge>
          </div>
        </div>
        <StatusDot status={toStatus(tool.status)} />
      </div>
      <div className="flex items-center gap-2 text-[9px] text-muted-foreground/50">
        <span
          className={cn(
            "text-[7px] px-1 py-px rounded uppercase font-semibold",
            cs.bg,
            cs.fg,
          )}
        >
          {tool.category}
        </span>
        <span>
          {tool.usedByAgents.length} agent
          {tool.usedByAgents.length !== 1 ? "s" : ""}
        </span>
        <span>&middot;</span>
        <span>
          {tool.metrics.calls24h > 0
            ? `${tool.metrics.calls24h.toLocaleString()}/24h`
            : "\u2014"}
        </span>
        <span>&middot;</span>
        <span>{tool.metrics.avgLatency}</span>
      </div>
    </button>
  );
}

// ── Tool detail panel ───────────────────────────────────────────

function ToolDetail({ tool, onClose }: { tool: Tool; onClose: () => void }) {
  const cs = CAT_STYLES[tool.category] ?? CAT_STYLES.tool;
  const catIcon =
    TOOL_CATEGORIES.find((c) => c.id === tool.category)?.icon ?? "\u2699";

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-border shrink-0">
        <div className="flex items-center justify-between mb-1.5">
          <div className="flex items-center gap-2.5">
            <div
              className={cn(
                "w-9 h-9 rounded-lg flex items-center justify-center text-base border",
                cs.bg,
                cs.fg,
                cs.border,
              )}
            >
              {catIcon}
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-bold text-foreground font-display">
                  {tool.displayName}
                </span>
                <StatusDot status={toStatus(tool.status)} />
              </div>
              <div className="text-[10px] text-muted-foreground mt-0.5">
                {tool.name} &middot; v{tool.version} &middot; {tool.author}
              </div>
            </div>
          </div>
          <Button variant="ghost" size="icon-xs" onClick={onClose}>
            &#x2715;
          </Button>
        </div>
        <p className="text-[10px] text-muted-foreground leading-relaxed">
          {tool.description}
        </p>
      </div>

      {/* Sub-tabs */}
      <Tabs defaultValue="overview" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-5 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="schema">I/O Schema</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
          </TabsList>
        </div>

        {/* Overview */}
        <TabsContent value="overview" className="flex-1 overflow-y-auto p-5">
          <div className="grid grid-cols-2 gap-2 mb-4">
            {[
              { label: "Calls / 24h", value: tool.metrics.calls24h > 0 ? tool.metrics.calls24h.toLocaleString() : "0" },
              { label: "Avg Latency", value: tool.metrics.avgLatency },
              { label: "Error Rate", value: tool.metrics.errorRate },
              { label: "P99", value: tool.metrics.p99 },
            ].map((m) => (
              <div
                key={m.label}
                className="bg-muted/40 border border-border rounded-md px-3 py-2.5"
              >
                <div className="text-[8px] text-muted-foreground/50 uppercase tracking-wide mb-0.5 font-display">
                  {m.label}
                </div>
                <div
                  className={cn(
                    "text-base font-bold",
                    m.value === "\u2014" ? "text-muted-foreground/30" : "text-foreground",
                  )}
                >
                  {m.value}
                </div>
              </div>
            ))}
          </div>

          <div className="text-[11px] font-semibold text-foreground font-display mb-2">
            Used by Agents
          </div>
          {tool.usedByAgents.map((a) => (
            <div
              key={a.name}
              className="px-3 py-2 bg-muted/40 border border-border rounded-md mb-1 flex items-center justify-between"
            >
              <div className="flex items-center gap-1.5">
                <StatusDot status={toStatus(a.status)} />
                <span className="text-[11px] text-foreground">{a.name}</span>
              </div>
              <span
                className={cn(
                  "text-[9px]",
                  a.status === "error"
                    ? "text-destructive"
                    : a.status === "deploying"
                      ? "text-amber-500"
                      : "text-green-500",
                )}
              >
                {a.status}
              </span>
            </div>
          ))}
        </TabsContent>

        {/* Schema */}
        <TabsContent value="schema" className="flex-1 overflow-y-auto p-5">
          <div className="text-[11px] font-semibold text-foreground font-display mb-2">
            Input Parameters
          </div>
          <div className="bg-muted/40 border border-border rounded-md overflow-hidden mb-4">
            {tool.inputSchema.map((p) => (
              <div
                key={p.name}
                className="px-3.5 py-2.5 border-b border-border/50 last:border-b-0"
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <code className="text-[11px] text-blue-500 font-medium">
                    {p.name}
                  </code>
                  <Badge>{p.type}</Badge>
                  {p.required && <Badge variant="danger">REQ</Badge>}
                </div>
                <div className="text-[9px] text-muted-foreground">{p.description}</div>
              </div>
            ))}
          </div>

          <div className="text-[11px] font-semibold text-foreground font-display mb-2">
            Output
          </div>
          <div className="bg-muted/40 border border-border rounded-md overflow-hidden">
            {tool.outputSchema.map((p) => (
              <div
                key={p.name}
                className="px-3.5 py-2.5 border-b border-border/50 last:border-b-0"
              >
                <div className="flex items-center gap-1.5 mb-0.5">
                  <code className="text-[11px] text-green-500 font-medium">
                    {p.name}
                  </code>
                  <Badge>{p.type}</Badge>
                </div>
                <div className="text-[9px] text-muted-foreground">{p.description}</div>
              </div>
            ))}
          </div>
        </TabsContent>

        {/* Config */}
        <TabsContent value="config" className="flex-1 overflow-y-auto p-5">
          {tool.config.length === 0 ? (
            <div className="py-6 text-center text-muted-foreground/40 text-[11px]">
              No configuration required
            </div>
          ) : (
            <div className="bg-muted/40 border border-border rounded-md overflow-hidden">
              {tool.config.map((c) => (
                <div
                  key={c.key}
                  className="px-3.5 py-2.5 border-b border-border/50 last:border-b-0 flex items-center justify-between"
                >
                  <div>
                    <code className="text-[11px] text-blue-500">{c.key}</code>
                    <span className="text-muted-foreground/30 mx-1.5">=</span>
                    <span className="text-[11px] text-foreground/70">{c.value}</span>
                  </div>
                  {c.secret && <Badge variant="danger">SECRET</Badge>}
                </div>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ToolEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x2699;</div>
        <div className="font-display text-sm">Select a tool to view details</div>
      </div>
    </div>
  );
}

// ── Main ────────────────────────────────────────────────────────

export function ToolsTab() {
  const [category, setCategory] = useState("all");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const filtered = TOOLS.filter((t) => {
    if (category !== "all" && t.category !== category) return false;
    if (
      search &&
      !t.displayName.toLowerCase().includes(search.toLowerCase()) &&
      !t.name.toLowerCase().includes(search.toLowerCase())
    )
      return false;
    return true;
  });

  const selected = selectedId
    ? TOOLS.find((t) => t.id === selectedId) ?? null
    : null;

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Categories */}
      <CategorySidebar
        active={category}
        onChange={(id) => {
          setCategory(id);
          setSelectedId(null);
        }}
      />

      {/* Tool list */}
      <div className="w-[360px] min-w-[360px] border-r border-border flex flex-col">
        <div className="px-3.5 py-2.5 border-b border-border flex items-center gap-2">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter tools..."
            className="h-7 text-[11px]"
          />
          <span className="text-[9px] text-muted-foreground/30 whitespace-nowrap">
            {filtered.length}
          </span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {filtered.map((t) => (
            <ToolListItem
              key={t.id}
              tool={t}
              selected={selectedId === t.id}
              onSelect={() => setSelectedId(t.id)}
            />
          ))}
        </div>
      </div>

      {/* Detail / Empty */}
      {selected ? (
        <ToolDetail tool={selected} onClose={() => setSelectedId(null)} />
      ) : (
        <ToolEmptyState />
      )}
    </div>
  );
}
