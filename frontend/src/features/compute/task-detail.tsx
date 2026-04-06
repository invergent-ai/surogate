// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAppStore } from "@/stores/app-store";
import { getTaskLogs } from "@/api/tasks";
import { STATUS_COLORS } from "./compute-data";
import { X, Terminal, Clock, User, Folder, Cpu, MapPin, ChevronRight } from "lucide-react";
import { statusForDot, parseProgress, InfoRow, TASK_COLOR, EXTENDED_WORKLOAD_COLORS } from "./detail-shared";
import type { ExtendedWorkload } from "./detail-shared";

export function TaskDetail({ item, onClose }: { item: ExtendedWorkload; onClose: () => void }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const cancelTask = useAppStore((s) => s.cancelTask);
  const dot = statusForDot(item.status);
  const isActive = dot === "running" || dot === "deploying" || item.status === "pending";
  const task = item._task;

  useEffect(() => {
    let cancelled = false;
    const fetchLogs = async () => {
      setLogsLoading(true);
      try {
        const res = await getTaskLogs(item.id, 200);
        if (!cancelled) setLogs(res.lines);
      } catch {
        if (!cancelled) setLogs(["(could not fetch logs)"]);
      } finally {
        if (!cancelled) setLogsLoading(false);
      }
    };
    void fetchLogs();
    const interval = isActive ? setInterval(() => void fetchLogs(), 3000) : undefined;
    return () => { cancelled = true; clearInterval(interval); };
  }, [item.id, isActive]);

  const color = EXTENDED_WORKLOAD_COLORS[item.type] ?? "#6B7280";

  return (
    <td colSpan={9} className="p-0">
      <div className="bg-muted/30 border-t border-line animate-in fade-in duration-150">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <div className="flex items-center gap-2.5">
            <StatusDot status={statusForDot(item.status)} />
            <span className="text-sm font-bold text-foreground font-display">{item.name}</span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase border"
              style={{ background: color + "12", color, borderColor: color + "20" }}
            >
              {item.type}
            </span>
            <span className="text-[11px] font-medium" style={{ color: STATUS_COLORS[item.status] ?? "var(--muted-foreground)" }}>
              {item.status}
            </span>
            {item.method !== "\u2014" && <span className="text-[11px] text-faint">{"\u00b7"} {item.method}</span>}
          </div>
          <div className="flex items-center gap-2">
            {isActive && (
              <Button
                variant="outline"
                size="xs"
                className="text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={(e) => { e.stopPropagation(); void cancelTask(item.id); }}
              >
                Cancel
              </Button>
            )}
            <Button variant="ghost" size="icon-xs" onClick={(e) => { e.stopPropagation(); onClose(); }}>
              <X size={14} />
            </Button>
          </div>
        </div>

        <div className="flex">
          {/* Left: info */}
          <div className="w-100 shrink-0 px-4 py-3 border-r border-line">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
              <InfoRow icon={Folder} label="Project" value={item.project} />
              <InfoRow icon={User} label="Requested by" value={item.requestedBy} />
              <InfoRow icon={MapPin} label="Location" value={item.location === "aws" ? "AWS Cloud" : "Local"} />
              <InfoRow icon={Cpu} label="GPU" value={item.gpu} />
              {item.node !== "\u2014" && <InfoRow icon={ChevronRight} label="Node" value={item.node} />}
              {item.startedAt && <InfoRow icon={Clock} label="Started" value={item.startedAt} />}
              {task?.exit_code != null && <InfoRow icon={Terminal} label="Exit code" value={String(task.exit_code)} />}
              {task?.created_at && <InfoRow icon={Clock} label="Created" value={new Date(task.created_at).toLocaleString()} />}
              {task?.completed_at && <InfoRow icon={Clock} label="Completed" value={new Date(task.completed_at).toLocaleString()} />}
            </div>

            {task?.progress && (
              <div className="mt-3">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="text-faint">Progress</span>
                  <span className="text-muted-foreground font-mono">{task.progress}</span>
                </div>
                <ProgressBar value={parseProgress(task.progress)} color={TASK_COLOR} />
              </div>
            )}

            {task?.error_message && (
              <div className="mt-3 p-2 rounded-md bg-destructive/10 border border-destructive/20">
                <div className="text-[10px] text-destructive uppercase font-display font-semibold mb-0.5">Error</div>
                <div className="text-[11px] text-destructive/80 leading-relaxed">{task.error_message}</div>
              </div>
            )}
          </div>

          {/* Right: log viewer */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center gap-2 px-3 py-1.5 border-b border-line shrink-0">
              <Terminal size={11} className="text-faint" />
              <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Log Output</span>
              {isActive && <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />}
            </div>
            <ScrollArea className="h-70 bg-background">
              <div className="p-3">
                {logsLoading && logs.length === 0 ? (
                  <div className="text-[11px] text-faint animate-pulse">Loading logs...</div>
                ) : logs.length === 0 ? (
                  <div className="text-[11px] text-faint">No log output yet</div>
                ) : (
                  <pre className="text-[11px] leading-[1.6] text-muted-foreground font-mono whitespace-pre-wrap break-all">
                    {logs.map((line, i) => (
                      <div
                        key={i}
                        className={
                          line.startsWith("ERROR:") ? "text-destructive font-semibold" :
                          line.startsWith("PROGRESS:") ? "text-primary" :
                          line.startsWith("Warning:") ? "text-[#F59E0B]" :
                          ""
                        }
                      >
                        {line}
                      </div>
                    ))}
                  </pre>
                )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    </td>
  );
}
