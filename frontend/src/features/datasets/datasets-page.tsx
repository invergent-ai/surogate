// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { DATASETS, FORMAT_STYLES, toStatus } from "./datasets-data";
import { DatasetDetail, DatasetEmptyState } from "./dataset-detail";
import { NewDatasetDialog } from "./new-dataset-dialog";
import type { Dataset } from "./datasets-data";

// ── Format filter buttons ─────────────────────────────────────

const FORMAT_FILTERS = [
  { id: "all", label: "All" },
  { id: "SFT", label: "SFT" },
  { id: "DPO", label: "DPO" },
  { id: "GRPO", label: "GRPO" },
] as const;

// ── Dataset list item ─────────────────────────────────────────

function DatasetListItem({
  dataset,
  selected,
  onSelect,
}: {
  dataset: Dataset;
  selected: boolean;
  onSelect: () => void;
}) {
  const fs = FORMAT_STYLES[dataset.format] ?? FORMAT_STYLES.SFT;

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60"
          : "border-l-transparent hover:bg-muted/30",
      )}
      style={selected ? { borderLeftColor: dataset.color } : undefined}
    >
      <div className="flex items-start gap-2.5">
        <div
          className="w-[34px] h-[34px] rounded-lg shrink-0 flex items-center justify-center text-[15px] border"
          style={{
            backgroundColor: `${dataset.color}12`,
            borderColor: `${dataset.color}25`,
            color: dataset.color,
          }}
        >
          &#x25A4;
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="text-xs font-semibold text-foreground font-display truncate">
              {dataset.name}
            </span>
            <span className={cn("text-[8px] font-semibold px-1.5 py-px rounded border shrink-0", fs.bg, fs.fg, fs.border)}>
              {dataset.format}
            </span>
          </div>
          <div className="text-[10px] text-muted-foreground mb-1 truncate">
            {dataset.displayName}
          </div>
          <div className="flex items-center gap-2.5 text-[10px]">
            <span className="flex items-center gap-1">
              <StatusDot status={toStatus(dataset.status)} />
              <span
                className={cn(
                  dataset.status === "error"
                    ? "text-destructive"
                    : dataset.status === "building"
                      ? "text-amber-500"
                      : "text-green-500",
                )}
              >
                {dataset.status}
              </span>
            </span>
            <span className="text-muted-foreground/30">&middot;</span>
            <span className="text-muted-foreground">
              {dataset.samples.toLocaleString()} samples
            </span>
            <span className="text-muted-foreground/30">&middot;</span>
            <span className="text-muted-foreground">{dataset.size}</span>
            {dataset.published && (
              <>
                <span className="text-muted-foreground/30">&middot;</span>
                <span className="text-[8px] text-green-500">&#x2295; Hub</span>
              </>
            )}
          </div>
        </div>
      </div>
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────

export function DatasetsPage() {
  const [selectedId, setSelectedId] = useState<string | null>("cx-convos-v5");
  const [filterFormat, setFilterFormat] = useState("all");
  const [filterSearch, setFilterSearch] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);

  const dataset = selectedId
    ? DATASETS.find((d) => d.id === selectedId) ?? null
    : null;

  const filtered = DATASETS.filter((d) => {
    if (filterFormat !== "all" && d.format !== filterFormat) return false;
    if (
      filterSearch &&
      !d.name.toLowerCase().includes(filterSearch.toLowerCase()) &&
      !d.displayName.toLowerCase().includes(filterSearch.toLowerCase()) &&
      !d.tags.some((t) => t.includes(filterSearch.toLowerCase()))
    )
      return false;
    return true;
  });

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Datasets"
        subtitle={
          <>
            {DATASETS.length} datasets &middot;{" "}
            {DATASETS.reduce((s, d) => s + d.samples, 0).toLocaleString()} total samples &middot;{" "}
            {DATASETS.filter((d) => d.status === "building").length} building
          </>
        }
        action={
          <div className="flex gap-1.5">
            <Button variant="outline" size="sm">&#x2295; Import</Button>
            <Button size="sm" onClick={() => setShowCreateModal(true)}>
              &#x25A4; New Dataset
            </Button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Dataset list (left) */}
        <div className="w-[400px] min-w-[400px] border-r border-border flex flex-col">
          {/* Search + filters */}
          <div className="px-3.5 py-3 border-b border-border space-y-2.5">
            <Input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Filter datasets..."
              className="h-8 text-xs"
            />
            <div className="flex gap-1">
              {FORMAT_FILTERS.map((f) => {
                const isActive = filterFormat === f.id;
                const fStyles = FORMAT_STYLES[f.id];
                return (
                  <button
                    key={f.id}
                    onClick={() => setFilterFormat(f.id)}
                    className={cn(
                      "px-2 py-1 rounded text-[10px] font-medium font-display border transition-colors cursor-pointer",
                      isActive && fStyles
                        ? `${fStyles.border} ${fStyles.bg} ${fStyles.fg}`
                        : isActive
                          ? "border-amber-500/20 bg-amber-500/10 text-amber-500"
                          : "border-transparent text-muted-foreground hover:bg-muted/50",
                    )}
                  >
                    {f.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* List */}
          <div className="flex-1 overflow-y-auto">
            {filtered.map((d) => (
              <DatasetListItem
                key={d.id}
                dataset={d}
                selected={selectedId === d.id}
                onSelect={() => setSelectedId(d.id)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="py-8 text-center text-muted-foreground/30 text-xs">
                No datasets match filters
              </div>
            )}
          </div>
        </div>

        {/* Detail (right) */}
        {dataset ? (
          <DatasetDetail dataset={dataset} />
        ) : (
          <DatasetEmptyState />
        )}
      </div>

      {/* Modal */}
      <NewDatasetDialog open={showCreateModal} onOpenChange={setShowCreateModal} />
    </div>
  );
}
