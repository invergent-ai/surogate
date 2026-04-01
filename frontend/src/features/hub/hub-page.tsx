// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useMemo, useEffect } from "react";
import { useNavigate } from "@tanstack/react-router";
import { PageHeader } from "@/components/page-header";
import { Card } from "@/components/ui/card";
import { useAppStore } from "@/stores/app-store";
import { RepositoryType } from "@/types/hub";
import { TYPE_META } from "./hub-data";
import type { RepoType } from "./hub-data";
import { CreateRepoDialog } from "./create-repo-dialog";
import { ImportDialog } from "./import-dialog";
import { Database, GitBranch, Clock, Download, LayoutGrid, Loader2, Plus, RefreshCw } from "lucide-react";
import { spawnTask } from "@/api/tasks";
import { toast } from "sonner";

export function HubPage() {
  const navigate = useNavigate();
  const { repositories = [], loading, error, fetchRepositories, createRepository: storeCreate, activeProjectId, addTask } = useAppStore();
  const [filterSearch, setFilterSearch] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [showImport, setShowImport] = useState(false);

  useEffect(() => {
    void fetchRepositories();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const counts: Record<string, number> = useMemo(() => {
    const c: Record<string, number> = { all: repositories.length };
    for (const t of Object.values(RepositoryType)) c[t] = 0;
    for (const r of repositories) {
      const t = r.metadata?.type;
      if (t && t in c) c[t]++;
    }
    return c;
  }, [repositories]);

  const allTags = useMemo(() => {
    const set = new Set<string>();
    for (const r of repositories) {
      if (r.metadata?.tags) r.metadata.tags.split(",").forEach((t) => set.add(t.trim()));
    }
    return [...set].sort();
  }, [repositories]);

  const filtered = useMemo(() => {
    let items = repositories;
    if (filterType !== "all") items = items.filter((r) => r.metadata?.type === filterType);
    if (selectedTag) items = items.filter((r) => r.metadata?.tags?.split(",").map((t) => t.trim()).includes(selectedTag));
    if (filterSearch) {
      const q = filterSearch.toLowerCase();
      items = items.filter((r) =>
        r.id.toLowerCase().includes(q) ||
        (r.metadata?.description ?? "").toLowerCase().includes(q) ||
        (r.metadata?.tags ?? "").toLowerCase().includes(q)
      );
    }
    return items;
  }, [repositories, filterType, selectedTag, filterSearch]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Hub"
        subtitle={`${repositories.length} repositories`}
      />

      {/* filter bar */}
      <div className="px-7 py-2.5 border-b border-line bg-card shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1.5 bg-input border border-border rounded-md flex-1 max-w-[480px]">
            <span className="text-faint text-[13px]">⌕</span>
            <input
              value={filterSearch}
              onChange={(e) => setFilterSearch(e.target.value)}
              placeholder="Search repositories..."
              className="flex-1 bg-transparent border-none outline-none text-foreground font-mono"
            />
            {filterSearch && (
              <button
                type="button"
                onClick={() => setFilterSearch("")}
                className="bg-transparent border-none text-muted-foreground cursor-pointer"
              >
                ✕
              </button>
            )}
          </div>
          <span className="text-faint">{filtered.length} results</span>
          <button
            type="button"
            onClick={() => fetchRepositories()}
            disabled={loading}
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border text-muted-foreground cursor-pointer font-display hover:text-foreground transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
          <button
            type="button"
            onClick={() => setShowImport(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-border text-muted-foreground cursor-pointer font-display hover:text-foreground transition-colors"
          >
            <Download size={14} />
            Import
          </button>
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-success/30 bg-success/5 text-success cursor-pointer font-display hover:bg-success/10 transition-colors"
          >
            <Plus size={14} />
            New Repository
          </button>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap mt-2">
          {[
            { id: "all", icon: LayoutGrid, label: "All", count: counts.all, color: "#22C55E" },
            ...Object.entries(TYPE_META).map(([id, m]) => ({
              id, icon: m.icon, label: m.plural, count: counts[id] ?? 0, color: m.color,
            })),
          ].map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => { setFilterType(c.id); setSelectedTag(null); }}
              className="flex items-center gap-1.5 border px-2.5 py-1 rounded-[5px] cursor-pointer font-display transition-all duration-150"
              style={{
                borderColor: filterType === c.id ? `${c.color}33` : "var(--border)",
                background: filterType === c.id ? `${c.color}10` : "transparent",
                color: filterType === c.id ? c.color : "var(--subtle)",
                fontWeight: filterType === c.id ? 600 : 400,
              }}
            >
              <c.icon size={12} />
              {c.label}
              <span className="text-[9px] opacity-60">{c.count}</span>
            </button>
          ))}
          <div className="w-px h-[18px] bg-line mx-0.5" />
          {(selectedTag ? [selectedTag] : allTags.slice(0, 10)).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setSelectedTag(selectedTag === t ? null : t)}
              className="px-[7px] py-[3px] rounded cursor-pointer border font-display"
              style={{
                borderColor: selectedTag === t ? "#22C55E33" : "var(--border)",
                background: selectedTag === t ? "#22C55E10" : "transparent",
                color: selectedTag === t ? "#22C55E" : "var(--muted-foreground)",
              }}
            >
              {t}
            </button>
          ))}
          {selectedTag && (
            <button
              type="button"
              onClick={() => setSelectedTag(null)}
              className="text-success cursor-pointer font-display font-semibold bg-transparent border-none"
            >
              CLEAR
            </button>
          )}
          {!selectedTag && allTags.length > 10 && (
            <span className="text-faint">+{allTags.length - 10}</span>
          )}
        </div>
      </div>

      {/* import dialog */}
      <ImportDialog
        open={showImport}
        onClose={() => setShowImport(false)}
        onImport={async (params) => {
          setShowImport(false);
          const taskType = params.type === "model" ? "import_model" : "import_dataset";
          const repoId = params.repoId.replace("/", "-");
          try {
            const task = await spawnTask({
              task_type: taskType,
              name: `Import ${params.repoId}`,
              project_id: activeProjectId ?? "",
              params: {
                hf_repo_id: params.repoId,
                lakefs_repo_id: repoId,
                lakefs_branch: "main",
                ...(params.token ? { hf_token: params.token } : {}),
                ...(params.subset ? { hf_dataset_subset: params.subset } : {}),
                ...(params.ggufFile ? { gguf_file: params.ggufFile } : {}),
              },
            });
            addTask(task);
            toast.success(`Import started: ${params.repoId}`, {
              description: "Track progress in Compute \u2192 Workload Queue",
            });
          } catch (e) {
            toast.error("Failed to start import", {
              description: (e as Error).message,
            });
          }
        }}
      />

      {/* create dialog */}
      <CreateRepoDialog
        open={showCreate}
        error={error}
        onClose={() => setShowCreate(false)}
        onCreate={storeCreate}
      />

      {/* content */}
      <div className="flex-1 overflow-y-auto px-7 py-4 pb-10">
        {loading && (
          <div className="py-10 flex justify-center text-faint">
            <Loader2 className="animate-spin" size={20} />
          </div>
        )}

        {error && (
          <div className="py-10 text-center text-destructive">{error}</div>
        )}

        {!loading && !error && (
          <div className="flex flex-col gap-2">
            {filtered.map((r) => {
              const repoType = r.metadata?.type as RepoType | undefined;
              const meta = repoType ? TYPE_META[repoType] : null;
              const Icon = meta?.icon ?? Database;
              const color = meta?.color ?? "#6B7280";
              return (
              <div
                key={r.id}
                onClick={() => navigate({ to: `/studio/hub/${r.id}` })}
                className="cursor-pointer"
              >
                <Card className="px-4 py-3.5 transition-all duration-150 hover:border-border">
                  <div className="flex items-start gap-3">
                    <div
                      className="w-9 h-9 rounded-lg shrink-0 flex items-center justify-center text-base border"
                      style={{ background: `${color}10`, borderColor: `${color}22`, color }}
                    >
                      <Icon size={18} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 mb-0.5 flex-wrap">
                        <span className="font-bold text-foreground font-display">{r.id}</span>
                        {meta && (
                          <span
                            className="text-sm px-[5px] py-px rounded font-semibold uppercase border"
                            style={{ background: `${color}12`, color, borderColor: `${color}25` }}
                          >
                            {meta.label}
                          </span>
                        )}
                        {r.read_only && (
                          <span className="text-xs px-[5px] py-px rounded bg-[#EF444412] text-[#EF4444]">
                            read-only
                          </span>
                        )}
                      </div>
                      {r.metadata?.description && (
                        <p className="text-sm text-subtle leading-normal mb-1.5 truncate">{r.metadata.description}</p>
                      )}
                      <div className="flex items-center gap-2.5 text-sm text-faint">
                        <span className="flex items-center gap-1">
                          <GitBranch size={12} />
                          {r.default_branch}
                        </span>
                        <span>·</span>
                        <span className="flex items-center gap-1">
                          <Clock size={12} />
                          {new Date(r.creation_date * 1000).toLocaleDateString()}
                        </span>
                        <span className="flex-1" />
                        {r.metadata?.tags && (
                          <div className="flex gap-[3px]">
                            {(() => {
                              const tags = r.metadata.tags.split(",").map((t) => t.trim());
                              return (
                                <>
                                  {tags.slice(0, 4).map((t) => (
                                    <span key={t} className="text-sm px-1 py-px rounded bg-accent text-muted-foreground">{t}</span>
                                  ))}
                                  {tags.length > 4 && <span className="text-sm text-faint">+{tags.length - 4}</span>}
                                </>
                              );
                            })()}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
              );
            })}
            {filtered.length === 0 && !loading && (
              <div className="py-10 text-center text-faint">No repositories found</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
