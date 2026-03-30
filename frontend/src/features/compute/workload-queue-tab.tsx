// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import React, { useState, useMemo, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { useAppStore } from "@/stores/app-store";
import { WORKLOAD_QUEUE, WORKLOAD_COLORS, STATUS_COLORS, PROVIDER_COLORS } from "./compute-data";
import type { LocalTask } from "@/api/tasks";
import { TaskDetail, statusForDot } from "./task-detail";
import type { ExtendedWorkload } from "./task-detail";
import { Trash2 } from "lucide-react";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

const TASK_COLOR = "#06B6D4";

function taskToWorkloadItem(t: LocalTask): ExtendedWorkload {
  return {
    id: t.id,
    name: t.name,
    type: "task",
    method: t.task_type.replace("_", " "),
    status: t.status,
    priority: 3,
    gpu: "\u2014",
    gpuCount: 0,
    location: "local",
    node: t.pid ? `pid:${t.pid}` : "\u2014",
    eta: t.progress ?? "\u2014",
    startedAt: t.started_at ? new Date(t.started_at).toLocaleString() : null,
    requestedBy: t.requested_by ?? "\u2014",
    project: t.project_id ?? "\u2014",
    _task: t,
  };
}

const FILTERS = ["all", "training", "serving", "eval", "task"] as const;

const EXTENDED_WORKLOAD_COLORS: Record<string, string> = {
  ...WORKLOAD_COLORS,
  task: TASK_COLOR,
};

export function WorkloadQueueTab() {
  const [filter, setFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ExtendedWorkload | null>(null);
  const tasks = useAppStore((s) => s.tasks);
  const fetchTasks = useAppStore((s) => s.fetchTasks);
  const deleteTask = useAppStore((s) => s.deleteTask);

  useEffect(() => {
    void fetchTasks();
  }, []);

  const allItems = useMemo<ExtendedWorkload[]>(() => {
    const localTaskItems = tasks.map(taskToWorkloadItem);
    const items = [...WORKLOAD_QUEUE, ...localTaskItems];
    items.sort((a, b) => {
      const aActive = a.status === "running" || a.status === "pending" ? 1 : 0;
      const bActive = b.status === "running" || b.status === "pending" ? 1 : 0;
      if (aActive !== bActive) return bActive - aActive;
      const aTime = a.startedAt ? new Date(a.startedAt).getTime() : 0;
      const bTime = b.startedAt ? new Date(b.startedAt).getTime() : 0;
      return bTime - aTime;
    });
    return items;
  }, [tasks]);

  const filtered = useMemo(
    () => filter === "all" ? allItems : allItems.filter(w => w.type === filter),
    [filter, allItems],
  );

  return (
    <div className="animate-in fade-in duration-200">
      <div className="flex items-center gap-1 mb-3">
        {FILTERS.map(f => {
          const color = EXTENDED_WORKLOAD_COLORS[f] ?? "var(--primary)";
          const active = filter === f;
          const count = f === "all" ? allItems.length : allItems.filter(w => w.type === f).length;
          return (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className="px-2 py-1 rounded border text-[11px] font-display font-medium capitalize cursor-pointer transition-all"
              style={{
                borderColor: active ? color + "33" : "var(--border)",
                background: active ? color + "10" : "transparent",
                color: active ? color : "var(--faint)",
              }}
            >
              {f}
              <span className="ml-1 opacity-60">{count}</span>
            </button>
          );
        })}
      </div>

      <Card size="sm" className="overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-line">
              {["Pri", "Workload", "Type", "Status", "GPU", "Location", "Progress / ETA", "Requested By", ""].map(h => (
                <th key={h} className="px-3 py-2 text-left text-[10px] font-medium text-faint uppercase tracking-wider font-display">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(w => {
              const isTask = w.type === "task";
              const isSelected = selectedId === w.id;
              return (
                <React.Fragment key={w.id}>
                  <tr
                    onClick={() => setSelectedId(isSelected ? null : w.id)}
                    className="border-b border-line hover:bg-muted/50 transition-colors cursor-pointer"
                    style={{
                      opacity: w.status === "queued" || w.status === "pending" ? 0.7 : 1,
                      background: isSelected ? "var(--muted)" : undefined,
                    }}
                  >
                    <td className="px-3 py-2.5">
                      <span
                        className="w-5 h-5 rounded inline-flex items-center justify-center text-[11px] font-bold border"
                        style={{
                          background: w.priority === 0 ? "#22C55E12" : w.priority === 1 ? "#F59E0B12" : "var(--muted)",
                          color: w.priority === 0 ? "#22C55E" : w.priority === 1 ? "#F59E0B" : "var(--faint)",
                          borderColor: w.priority === 0 ? "#22C55E20" : w.priority === 1 ? "#F59E0B20" : "var(--border)",
                        }}
                      >
                        {w.priority}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <div className="text-sm text-foreground font-medium">{w.name}</div>
                      {w.method !== "\u2014" && <div className="text-[11px] text-faint">{w.method}</div>}
                    </td>
                    <td className="px-3 py-2.5">
                      <span
                        className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase border"
                        style={{
                          background: (EXTENDED_WORKLOAD_COLORS[w.type] ?? "#3A4154") + "12",
                          color: EXTENDED_WORKLOAD_COLORS[w.type] ?? "#3A4154",
                          borderColor: (EXTENDED_WORKLOAD_COLORS[w.type] ?? "#3A4154") + "20",
                        }}
                      >
                        {w.type}
                      </span>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="flex items-center gap-1 text-[11px]">
                        <StatusDot status={statusForDot(w.status)} />
                        <span style={{ color: STATUS_COLORS[w.status] ?? "var(--muted-foreground)" }}>{w.status}</span>
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-[11px] text-muted-foreground">{w.gpu}</td>
                    <td className="px-3 py-2.5">
                      <span className="text-[11px]" style={{ color: w.location === "aws" ? PROVIDER_COLORS.aws : "var(--primary)" }}>
                        {w.location === "aws" ? "\u2601 AWS" : "\u229e Local"}
                      </span>
                      {w.node !== "\u2014" && <div className="text-[10px] text-faint">{w.node}</div>}
                    </td>
                    <td className="px-3 py-2.5 text-[11px]" style={{ color: w.status === "queued" || w.status === "pending" ? "#F59E0B" : "var(--muted-foreground)" }}>
                      {isTask && w.eta !== "\u2014" ? (
                        <div className="text-[10px] text-muted-foreground">{w.eta}</div>
                      ) : (
                        w.eta
                      )}
                    </td>
                    <td className="px-3 py-2.5 text-[11px] text-faint">{w.requestedBy}</td>
                    <td className="px-3 py-2.5">
                      {isTask && (
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          className="text-muted-foreground hover:text-destructive"
                          onClick={(e) => { e.stopPropagation(); setDeleteTarget(w); }}
                          title="Delete task"
                        >
                          <Trash2 size={12} />
                        </Button>
                      )}
                    </td>
                  </tr>
                  {isSelected && (
                    <tr className="border-b border-line">
                      <TaskDetail item={w} onClose={() => setSelectedId(null)} />
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-faint text-sm">No workloads</td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>

      <ConfirmDialog
        open={deleteTarget !== null}
        title="Delete Task"
        description={
          deleteTarget && (deleteTarget.status === "running" || deleteTarget.status === "pending")
            ? <>This will kill the running process and delete <span className="font-semibold text-foreground">{deleteTarget?.name}</span>. Continue?</>
            : <>Delete <span className="font-semibold text-foreground">{deleteTarget?.name}</span>? This cannot be undone.</>
        }
        confirmLabel="Delete"
        confirmIcon={<Trash2 size={14} className="mr-1.5" />}
        onConfirm={async () => {
          if (!deleteTarget) return;
          await deleteTask(deleteTarget.id);
          setDeleteTarget(null);
        }}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
