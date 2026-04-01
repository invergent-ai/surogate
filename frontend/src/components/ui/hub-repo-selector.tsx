// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect, useRef } from "react";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import {
  ChevronRight,
  ChevronLeft,
  GitBranch,
  Tag,
  Search,
  Loader2,
  ChevronsUpDown,
} from "lucide-react";
import type { Repository, Ref } from "@/types/hub";

export interface HubRefSelection {
  repo: string;
  ref: string;
  refType: "branch" | "tag";
}

interface HubRepoSelectorProps {
  onSelect: (selection: HubRefSelection) => void;
  value?: HubRefSelection | null;
  className?: string;
  repoFilter?: (repo: Repository) => boolean;
  placeholder?: string;
}

export function HubRepoSelector({
  onSelect,
  value,
  className,
  repoFilter,
  placeholder = "Select repository & ref\u2026",
}: HubRepoSelectorProps) {
  const repositories = useAppStore((s) => s.repositories);
  const branches = useAppStore((s) => s.branches);
  const tags = useAppStore((s) => s.tags);
  const fetchRepositories = useAppStore((s) => s.fetchRepositories);
  const fetchBranches = useAppStore((s) => s.fetchBranches);
  const fetchTags = useAppStore((s) => s.fetchTags);

  const [open, setOpen] = useState(false);
  const [selectedRepo, setSelectedRepo] = useState<string | null>(
    value?.repo ?? null,
  );
  const [search, setSearch] = useState("");
  const [refsLoading, setRefsLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Fetch repos on mount
  useEffect(() => {
    fetchRepositories();
  }, [fetchRepositories]);

  // Fetch branches + tags when a repo is selected
  const loadRefs = (repo: string) => {
    setRefsLoading(true);
    Promise.all([fetchBranches(repo), fetchTags(repo)]).finally(
      () => setRefsLoading(false),
    );
  };

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
        setSearch("");
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Focus search when panel opens
  useEffect(() => {
    if (open) searchRef.current?.focus();
  }, [open, selectedRepo]);

  const filteredRepos = repositories.filter((r) => {
    if (repoFilter && !repoFilter(r)) return false;
    if (search && !r.id.toLowerCase().includes(search.toLowerCase()))
      return false;
    return true;
  });

  const repoBranches = selectedRepo ? (branches[selectedRepo] ?? []) : [];
  const repoTags = selectedRepo ? (tags[selectedRepo] ?? []) : [];

  const filteredBranches = search
    ? repoBranches.filter((b) =>
        b.id.toLowerCase().includes(search.toLowerCase()),
      )
    : repoBranches;
  const filteredTags = search
    ? repoTags.filter((t) =>
        t.id.toLowerCase().includes(search.toLowerCase()),
      )
    : repoTags;

  const handleRepoClick = (repo: Repository) => {
    setSelectedRepo(repo.id);
    setSearch("");
    loadRefs(repo.id);
  };

  const handleBack = () => {
    setSelectedRepo(null);
    setSearch("");
  };

  const handleRefSelect = (ref: Ref, refType: "branch" | "tag") => {
    onSelect({ repo: selectedRepo!, ref: ref.id, refType });
    setOpen(false);
    setSearch("");
  };

  // Trigger label
  const label = value
    ? `${value.repo} @ ${value.ref}`
    : placeholder;

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex w-full items-center justify-between h-8 rounded-lg border border-input bg-transparent px-2.5 text-xs transition-colors cursor-pointer",
          "hover:border-ring/50 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 outline-none",
          value ? "text-foreground font-mono" : "text-muted-foreground",
        )}
      >
        <span className="truncate">{label}</span>
        <ChevronsUpDown size={12} className="shrink-0 ml-2 text-muted-foreground/50" />
      </button>

      {/* Overlay panel */}
      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-lg border border-border bg-popover shadow-md overflow-hidden">
          {/* Header / breadcrumb */}
          <div className="flex items-center gap-1.5 border-b border-border bg-muted/30 px-2 h-7">
            {selectedRepo ? (
              <>
                <button
                  onClick={handleBack}
                  className="flex items-center gap-0.5 text-muted-foreground hover:text-foreground transition-colors cursor-pointer bg-transparent border-none p-0"
                >
                  <ChevronLeft size={14} />
                  <span className="text-[10px] font-display">Repos</span>
                </button>
                <span className="text-muted-foreground/40 text-[10px]">/</span>
                <span className="text-[11px] font-mono text-foreground truncate">
                  {selectedRepo}
                </span>
              </>
            ) : (
              <span className="text-[10px] text-muted-foreground/60 uppercase tracking-wide font-display">
                Repositories
              </span>
            )}
          </div>

          {/* Search */}
          <div className="relative border-b border-border">
            <Search
              size={12}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/50"
            />
            <input
              ref={searchRef}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={
                selectedRepo
                  ? "Filter branches & tags\u2026"
                  : "Search repositories\u2026"
              }
              className="w-full h-7 pl-7 pr-2 text-xs bg-transparent border-none outline-none placeholder:text-muted-foreground/40"
            />
          </div>

          {/* Sliding panels */}
          <div className="relative h-40 overflow-hidden">
            <div
              className={cn(
                "absolute inset-0 flex transition-transform duration-200 ease-out",
                selectedRepo ? "-translate-x-full" : "translate-x-0",
              )}
            >
              {/* Panel 1 — Repository list */}
              <div className="min-w-full h-full overflow-y-auto">
                {filteredRepos.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-xs text-muted-foreground/50">
                    No repositories found
                  </div>
                ) : (
                  filteredRepos.map((repo) => (
                    <button
                      key={repo.id}
                      onClick={() => handleRepoClick(repo)}
                      className={cn(
                        "flex w-full items-center justify-between px-3 py-1.5 text-left text-xs transition-colors cursor-pointer border-none bg-transparent",
                        "hover:bg-accent/60",
                        value?.repo === repo.id && "bg-accent/40",
                      )}
                    >
                      <span className="font-mono truncate">{repo.id}</span>
                      <ChevronRight
                        size={12}
                        className="text-muted-foreground/40 shrink-0 ml-2"
                      />
                    </button>
                  ))
                )}
              </div>

              {/* Panel 2 — Branches + Tags */}
              <div className="min-w-full h-full overflow-y-auto">
                {refsLoading ? (
                  <div className="flex items-center justify-center h-full">
                    <Loader2
                      size={14}
                      className="animate-spin text-muted-foreground"
                    />
                  </div>
                ) : (
                  <>
                    {filteredBranches.length > 0 && (
                      <div>
                        <div className="px-3 py-1 text-[9px] uppercase tracking-wide text-muted-foreground/50 font-display sticky top-0 bg-popover z-10">
                          Branches
                        </div>
                        {filteredBranches.map((b) => (
                          <button
                            key={b.id}
                            onClick={() => handleRefSelect(b, "branch")}
                            className={cn(
                              "flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors cursor-pointer border-none bg-transparent",
                              "hover:bg-accent/60",
                              value?.ref === b.id &&
                                value?.refType === "branch" &&
                                "bg-primary/10 text-primary",
                            )}
                          >
                            <GitBranch
                              size={12}
                              className="shrink-0 text-muted-foreground"
                            />
                            <span className="font-mono truncate">{b.id}</span>
                          </button>
                        ))}
                      </div>
                    )}

                    {filteredTags.length > 0 && (
                      <div>
                        <div className="px-3 py-1 text-[9px] uppercase tracking-wide text-muted-foreground/50 font-display sticky top-0 bg-popover z-10">
                          Tags
                        </div>
                        {filteredTags.map((t) => (
                          <button
                            key={t.id}
                            onClick={() => handleRefSelect(t, "tag")}
                            className={cn(
                              "flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors cursor-pointer border-none bg-transparent",
                              "hover:bg-accent/60",
                              value?.ref === t.id &&
                                value?.refType === "tag" &&
                                "bg-primary/10 text-primary",
                            )}
                          >
                            <Tag
                              size={12}
                              className="shrink-0 text-muted-foreground"
                            />
                            <span className="font-mono truncate">{t.id}</span>
                          </button>
                        ))}
                      </div>
                    )}

                    {filteredBranches.length === 0 &&
                      filteredTags.length === 0 && (
                        <div className="flex items-center justify-center h-full text-xs text-muted-foreground/50">
                          No branches or tags
                        </div>
                      )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
