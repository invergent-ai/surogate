// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { TYPE_STYLES, toStatus } from "./models-data";
import { ModelDetail, ModelEmptyState } from "./model-detail";
import { useAppStore } from "@/stores/app-store";
import type { Model } from "./models-data";
import { ServeModelDialog } from "./serve-model-dialog";

// ── Status filter buttons ──────────────────────────────────────

const STATUS_FILTERS = [
  { id: "all", label: "All" },
  { id: "serving", label: "Serving" },
  { id: "error", label: "Error" },
  { id: "stopped", label: "Stopped" },
] as const;

// ── Model list item ────────────────────────────────────────────

function ModelListItem({
  model,
  selected,
  onSelect,
}: {
  model: Model;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-blue-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-start gap-2.5">
        <div
          className="w-8.5 h-8.5 rounded-lg shrink-0 flex items-center justify-center text-[15px] border"
          style={{
            backgroundColor: toStatus(model.status) === "serving" ? "#3B82F612" : undefined,
            borderColor: toStatus(model.status) === "serving" ? "#3B82F625" : undefined,
            color: toStatus(model.status) === "serving" ? "#3B82F6" : undefined,
          }}
        >
          &#x25C7;
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="text-xs font-semibold text-foreground font-display truncate">
              {model.name}
            </span>
            {TYPE_STYLES[model.type] && (
              <span
                className={cn(
                  "text-[8px] px-1.5 py-px rounded font-medium uppercase tracking-wide shrink-0",
                  TYPE_STYLES[model.type].bg,
                  TYPE_STYLES[model.type].fg,
                )}
              >
                {model.type}
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground mb-1 truncate">
            {model.family} &middot; {model.paramCount}
            {model.quantization !== "\u2014" ? ` \u00B7 ${model.quantization}` : ""}
          </div>
          <div className="flex items-center gap-2.5 text-[10px]">
            <span className="flex items-center gap-1">
              <StatusDot status={toStatus(model.status)} />
              <span
                className={cn(
                  toStatus(model.status) === "error"
                    ? "text-destructive"
                    : toStatus(model.status) === "serving"
                      ? "text-green-500"
                      : "text-muted-foreground",
                )}
              >
                {model.status}
              </span>
            </span>
            {toStatus(model.status) === "serving" && (
              <>
                <span className="text-muted-foreground/30">&middot;</span>
                <span className="text-muted-foreground">
                  {model.gpu.count}&times; GPU
                </span>
                <span className="text-muted-foreground/30">&middot;</span>
                <span className="text-foreground/70 font-medium">
                  {model.tps} tok/s
                </span>
              </>
            )}
          </div>
        </div>
      </div>
    </button>
  );
}

export function ModelsPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState("all");
  const [filterSearch, setFilterSearch] = useState("");
  const [showServeModal, setShowServeModal] = useState(false);

  const models = useAppStore((s) => s.models);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const modelsStatusCounts = useAppStore((s) => s.modelsStatusCounts);
  const fetchModels = useAppStore((s) => s.fetchModels);
  const fetchModel = useAppStore((s) => s.fetchModel);

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  const handleSelect = (id: string) => {
    setSelectedId(id);
    fetchModel(id);
  };

  const model = selectedModel && selectedModel.id === selectedId
    ? selectedModel
    : null;

  const filtered = models.filter((m: Model) => {
    if (filterStatus !== "all" && toStatus(m.status) !== filterStatus) return false;
    if (
      filterSearch &&
      !m.name.toLowerCase().includes(filterSearch.toLowerCase()) &&
      !m.displayName.toLowerCase().includes(filterSearch.toLowerCase())
    )
      return false;
    return true;
  });

  const sum = (...keys: string[]) =>
    keys.reduce((n, k) => n + (modelsStatusCounts[k] ?? 0), 0);

  const statusCounts = {
    all: models.length,
    serving: sum("running"),
    error: sum("failed"),
    stopped: sum("stopped", "cancelled", "completed"),
  };

  const totalGpus = models.reduce(
    (s: number, m: Model) => s + (toStatus(m.status) === "serving" ? m.gpu.count : 0),
    0,
  );
  const totalTps = models.reduce((s: number, m: Model) => s + m.tps, 0);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Models"
        subtitle={
          <>
            {statusCounts.serving} serving &middot; {totalGpus} GPUs allocated
            &middot; {totalTps.toLocaleString()} tok/s total
          </>
        }
        action={
          <div className="flex gap-2">
            <Button onClick={() => setShowServeModal(true)}>
              &#x25C7; Deploy Model
            </Button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Model list (left) */}
        <div className="w-95 min-w-95 border-r border-border flex flex-col">
          {/* Search + filters */}
          <div className="px-3.5 py-3 border-b border-border space-y-2.5">
            <Input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Filter models..."
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
                          : f.id === "serving"
                            ? "border-green-500/20 bg-green-500/10 text-green-500"
                            : "border-blue-500/20 bg-blue-500/10 text-blue-500"
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
            {filtered.map((m) => (
              <ModelListItem
                key={m.id}
                model={m}
                selected={selectedId === m.id}
                onSelect={() => handleSelect(m.id)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                No models match filters
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {model ? <ModelDetail model={model} /> : <ModelEmptyState />}
      </div>

      {/* Modals */}
      <ServeModelDialog
        open={showServeModal}
        onOpenChange={setShowServeModal}
      />
    </div>
  );
}
