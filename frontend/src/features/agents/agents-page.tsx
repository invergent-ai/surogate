// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusDot } from "@/components/ui/status-dot";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { cn } from "@/utils/cn";
import { AGENTS, toStatus } from "./agents-data";
import { AgentDetail, AgentEmptyState } from "./agent-detail";
import type { Agent } from "./agents-data";

// ── Status filter buttons ──────────────────────────────────────

const STATUS_FILTERS = [
  { id: "all", label: "All" },
  { id: "running", label: "Running" },
  { id: "deploying", label: "Deploying" },
  { id: "error", label: "Error" },
] as const;

// ── Agent list item ────────────────────────────────────────────

function AgentListItem({
  agent,
  selected,
  onSelect,
}: {
  agent: Agent;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-amber-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-start gap-2.5">
        <div
          className="w-[34px] h-[34px] rounded-lg shrink-0 flex items-center justify-center text-[15px] border"
          style={{
            backgroundColor: `${agent.projectColor}12`,
            borderColor: `${agent.projectColor}25`,
            color: agent.projectColor,
          }}
        >
          &#x2B21;
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="text-xs font-semibold text-foreground font-display">
              {agent.name}
            </span>
            <Badge>v{agent.version}</Badge>
          </div>
          <div className="text-[10px] text-muted-foreground mb-1 truncate">
            {agent.displayName}
          </div>
          <div className="flex items-center gap-2.5 text-[10px]">
            <span className="flex items-center gap-1">
              <StatusDot status={toStatus(agent.status)} />
              <span
                className={cn(
                  agent.status === "error"
                    ? "text-destructive"
                    : agent.status === "deploying"
                      ? "text-amber-500"
                      : "text-green-500",
                )}
              >
                {agent.status}
              </span>
            </span>
            <span className="text-muted-foreground/30">&middot;</span>
            <span className="text-muted-foreground">
              {agent.replicas.current}/{agent.replicas.desired}
            </span>
            <span className="text-muted-foreground/30">&middot;</span>
            <span className="text-muted-foreground">{agent.rps} rps</span>
          </div>
        </div>
      </div>
    </button>
  );
}

// ── Deploy modal ───────────────────────────────────────────────

const DEPLOY_TEMPLATES = [
  { icon: "\u2B21", label: "Blank Agent", desc: "Start from scratch" },
  { icon: "\u2B21", label: "Chat Agent", desc: "Conversational template" },
  { icon: "\u26A1", label: "Tool Agent", desc: "With tool-use skills" },
  { icon: "\u25A4", label: "RAG Agent", desc: "Knowledge-grounded" },
  { icon: "\u25C8", label: "Safety Agent", desc: "Content moderation" },
  { icon: "\u29C9", label: "From Hub", desc: "Import from registry" },
];

function DeployDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Deploy New Agent</DialogTitle>
          <DialogDescription>
            Choose a template or upload a configuration
          </DialogDescription>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-2.5">
          {DEPLOY_TEMPLATES.map((t) => (
            <button
              key={t.label}
              className="p-3.5 rounded-lg border border-border bg-muted/40 hover:border-amber-500/30 hover:bg-muted/60 transition-colors cursor-pointer text-left flex items-start gap-2.5"
            >
              <span className="text-lg text-amber-500 shrink-0">{t.icon}</span>
              <div>
                <div className="text-xs font-semibold text-foreground font-display">
                  {t.label}
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5">
                  {t.desc}
                </div>
              </div>
            </button>
          ))}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Scale modal ────────────────────────────────────────────────

function ScaleDialog({
  open,
  onOpenChange,
  agent,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  agent: Agent;
}) {
  const [value, setValue] = useState(agent.replicas.desired);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Scale {agent.name}</DialogTitle>
          <DialogDescription>
            Current: {agent.replicas.current}/{agent.replicas.desired} replicas
          </DialogDescription>
        </DialogHeader>
        <div className="flex items-center gap-4 justify-center py-4">
          <Button
            variant="outline"
            size="icon"
            onClick={() => setValue(Math.max(0, value - 1))}
          >
            &minus;
          </Button>
          <div className="text-4xl font-bold text-foreground w-16 text-center">
            {value}
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={() => setValue(Math.min(10, value + 1))}
          >
            +
          </Button>
        </div>
        <div className="text-[10px] text-muted-foreground/50 text-center">
          Est. resources: CPU {value * 500}m&ndash;{value * 2000}m &middot; MEM{" "}
          {value}Gi&ndash;{value * 4}Gi
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => onOpenChange(false)}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ──────────────────────────────────────────────────

export function AgentsPage() {
  const [selectedId, setSelectedId] = useState<string | null>("cx-support-v3");
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSearch, setFilterSearch] = useState("");
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [showScaleModal, setShowScaleModal] = useState(false);

  const agent = selectedId
    ? AGENTS.find((a) => a.id === selectedId) ?? null
    : null;

  const filtered = AGENTS.filter((a) => {
    if (filterStatus !== "all" && a.status !== filterStatus) return false;
    if (
      filterSearch &&
      !a.name.toLowerCase().includes(filterSearch.toLowerCase()) &&
      !a.displayName.toLowerCase().includes(filterSearch.toLowerCase())
    )
      return false;
    return true;
  });

  const statusCounts = {
    all: AGENTS.length,
    running: AGENTS.filter((a) => a.status === "running").length,
    deploying: AGENTS.filter((a) => a.status === "deploying").length,
    error: AGENTS.filter((a) => a.status === "error").length,
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Agents"
        subtitle={
          <>
            {statusCounts.running} running &middot; {statusCounts.deploying}{" "}
            deploying &middot; {statusCounts.error} error
          </>
        }
        action={
          <Button size="sm" onClick={() => setShowDeployModal(true)}>
            + Deploy Agent
          </Button>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Agent list (left) */}
        <div className="w-[360px] min-w-[360px] border-r border-border flex flex-col">
          {/* Search + filters */}
          <div className="px-3.5 py-3 border-b border-border space-y-2.5">
            <Input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Filter agents..."
              className="h-8 text-xs"
            />
            <div className="flex gap-1">
              {STATUS_FILTERS.map((f) => {
                const count = statusCounts[f.id as keyof typeof statusCounts];
                const isActive = filterStatus === f.id;
                return (
                  <button
                    key={f.id}
                    onClick={() => setFilterStatus(f.id)}
                    className={cn(
                      "px-2 py-1 rounded text-[10px] font-medium font-display border transition-colors cursor-pointer",
                      isActive
                        ? f.id === "error"
                          ? "border-red-500/20 bg-red-500/10 text-red-500"
                          : "border-amber-500/20 bg-amber-500/10 text-amber-500"
                        : "border-transparent text-muted-foreground hover:bg-muted/50",
                    )}
                  >
                    {f.label}{" "}
                    <span className="opacity-60">{count}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto">
            {filtered.map((a) => (
              <AgentListItem
                key={a.id}
                agent={a}
                selected={selectedId === a.id}
                onSelect={() => setSelectedId(a.id)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                No agents match filters
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {agent ? (
          <AgentDetail
            agent={agent}
            onScale={() => setShowScaleModal(true)}
          />
        ) : (
          <AgentEmptyState />
        )}
      </div>

      {/* Modals */}
      <DeployDialog
        open={showDeployModal}
        onOpenChange={setShowDeployModal}
      />
      {agent && (
        <ScaleDialog
          open={showScaleModal}
          onOpenChange={setShowScaleModal}
          agent={agent}
        />
      )}
    </div>
  );
}
