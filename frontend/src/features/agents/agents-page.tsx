// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect, useState } from "react";
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
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { cn } from "@/utils/cn";
import { toStatus } from "./agents-data";
import { AgentDetail, AgentEmptyState } from "./agent-detail";
import type { Agent } from "./agents-data";
import { useAppStore } from "@/stores/app-store";

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
  const projectColor =
    useAppStore((s) => s.projects.find((p) => p.id === agent.projectId)?.color) ?? "#F59E0B";
  return (
    <button type="button"
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
            backgroundColor: `${projectColor}12`,
            borderColor: `${projectColor}25`,
            color: projectColor,
          }}
        >
          &#x2B21;
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="text-sm font-semibold text-foreground font-display">
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
  const updateAgent = useAppStore((s) => s.updateAgent);

  const handleApply = async () => {
    await updateAgent(agent.id, {
      replicas: { current: agent.replicas.current, desired: value },
    });
    onOpenChange(false);
  };

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
          <Button onClick={handleApply}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Create modal ───────────────────────────────────────────────

const SUROGATES_HARNESS = "surogates";

function CreateAgentDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const activeProjectId = useAppStore((s) => s.activeProjectId);
  const createAgent = useAppStore((s) => s.createAgent);
  const models = useAppStore((s) => s.models);
  const fetchModels = useAppStore((s) => s.fetchModels);
  const agents = useAppStore((s) => s.agents);

  useEffect(() => {
    if (open) fetchModels();
  }, [open, fetchModels]);

  const [displayName, setDisplayName] = useState("");
  const [slug, setSlug] = useState("");
  const [slugTouched, setSlugTouched] = useState(false);
  const [description, setDescription] = useState("");
  const [modelId, setModelId] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const slugify = (v: string) =>
    v.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const effectiveSlug = slugTouched ? slug : slugify(displayName);

  const projectAgentSlugs = new Set(
    agents.filter((a) => a.projectId === activeProjectId).map((a) => a.name),
  );
  const slugTaken = !!effectiveSlug && projectAgentSlugs.has(effectiveSlug);

  const handleClose = (v: boolean) => {
    if (!v) {
      setDisplayName("");
      setSlug("");
      setSlugTouched(false);
      setDescription("");
      setModelId("");
      setSubmitting(false);
    }
    onOpenChange(v);
  };

  const handleCreate = async () => {
    if (!activeProjectId || !effectiveSlug || !modelId || slugTaken) return;
    setSubmitting(true);
    const agent = await createAgent(activeProjectId, {
      name: effectiveSlug,
      harness: SUROGATES_HARNESS,
      display_name: displayName || effectiveSlug,
      description,
      model_id: modelId,
    });
    setSubmitting(false);
    if (agent) handleClose(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="min-w-150">
        <DialogHeader>
          <DialogTitle>Deploy New Agent</DialogTitle>
          <DialogDescription>Configure your Surogates agent</DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div>
            <label htmlFor="agent-display-name" className="text-xs font-medium text-muted-foreground mb-1 block">Display Name</label>
            <Input
              id="agent-display-name"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="My Agent"
              className="h-8 text-xs"
            />
          </div>
          <div>
            <label htmlFor="agent-slug" className="text-xs font-medium text-muted-foreground mb-1 block">
              Slug <span className="text-muted-foreground/60">(unique within project)</span>
            </label>
            <Input
              id="agent-slug"
              value={effectiveSlug}
              onChange={(e) => {
                setSlug(slugify(e.target.value));
                setSlugTouched(true);
              }}
              placeholder="my-agent"
              className={cn(
                "h-8 text-xs",
                slugTaken && "border-destructive focus-visible:ring-destructive",
              )}
            />
            {slugTaken && (
              <div className="text-[10px] text-destructive mt-1">
                Slug "{effectiveSlug}" is already used in this project.
              </div>
            )}
          </div>
          <div>
            <label htmlFor="agent-description" className="text-xs font-medium text-muted-foreground mb-1 block">Description</label>
            <Input
              id="agent-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this agent do?"
              className="h-8 text-xs"
            />
          </div>
          <div>
            <span className="text-xs font-medium text-muted-foreground mb-1 block">Model</span>
            <Select value={modelId} onValueChange={setModelId} disabled={models.length === 0}>
              <SelectTrigger size="sm" className="w-full text-xs">
                <SelectValue
                  placeholder={
                    models.length === 0 ? "No models available" : "Select a model..."
                  }
                />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleClose(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            disabled={!effectiveSlug || !modelId || slugTaken || submitting}
          >
            {submitting ? "Creating..." : "Create Agent"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ──────────────────────────────────────────────────

export function AgentsPage() {
  const agents = useAppStore((s) => s.agents);
  const agentsLoading = useAppStore((s) => s.agentsLoading);
  const fetchAgents = useAppStore((s) => s.fetchAgents);
  const selectedAgent = useAppStore((s) => s.selectedAgent);
  const setSelectedAgent = useAppStore((s) => s.setSelectedAgent);

  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSearch, setFilterSearch] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showScaleModal, setShowScaleModal] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const deleteAgent = useAppStore((s) => s.deleteAgent);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  const filtered = agents.filter((a) => {
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
    all: agents.length,
    running: agents.filter((a) => a.status === "running").length,
    deploying: agents.filter((a) => a.status === "deploying").length,
    error: agents.filter((a) => a.status === "error").length,
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
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
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
            />
            <div className="flex gap-1">
              {STATUS_FILTERS.map((f) => {
                const count = statusCounts[f.id as keyof typeof statusCounts];
                const isActive = filterStatus === f.id;
                return (
                  <button type="button"
                    key={f.id}
                    onClick={() => setFilterStatus(f.id)}
                    className={cn(
                      "px-2 py-1 rounded text-xs font-medium font-display border transition-colors cursor-pointer",
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
                selected={selectedAgent?.id === a.id}
                onSelect={() => setSelectedAgent(a)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                {agentsLoading ? "Loading agents..." : "No agents match filters"}
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {selectedAgent ? (
          <AgentDetail
            agent={selectedAgent}
            onScale={() => setShowScaleModal(true)}
            onDelete={() => setShowDeleteConfirm(true)}
          />
        ) : (
          <AgentEmptyState />
        )}
      </div>

      {/* Modals */}
      <CreateAgentDialog
        open={showCreateModal}
        onOpenChange={setShowCreateModal}
      />
      {selectedAgent && (
        <ScaleDialog
          open={showScaleModal}
          onOpenChange={setShowScaleModal}
          agent={selectedAgent}
        />
      )}
      {selectedAgent && (
        <ConfirmDialog
          open={showDeleteConfirm}
          title={`Delete ${selectedAgent.name}?`}
          description="This will permanently remove the agent and all its versions. This action cannot be undone."
          confirmLabel="Delete"
          variant="destructive"
          onConfirm={async () => {
            await deleteAgent(selectedAgent.id);
            setShowDeleteConfirm(false);
          }}
          onCancel={() => setShowDeleteConfirm(false)}
        />
      )}
    </div>
  );
}
