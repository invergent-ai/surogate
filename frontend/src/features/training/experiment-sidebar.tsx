// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot, type Status } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { cn } from "@/utils/cn";
import { AGENT_FLOWS, EVALUATORS } from "./training-data";

const METHOD_COLORS: Record<string, { bg: string; fg: string; border: string }> = {
  SFT: { bg: "bg-green-500/[0.07]", fg: "text-green-500", border: "border-green-500/20" },
  DPO: { bg: "bg-violet-500/[0.07]", fg: "text-violet-500", border: "border-violet-500/20" },
  GRPO: { bg: "bg-blue-500/[0.07]", fg: "text-blue-500", border: "border-blue-500/20" },
  PPO: { bg: "bg-amber-500/[0.07]", fg: "text-amber-500", border: "border-amber-500/20" },
};

interface Experiment {
  id: string;
  name: string;
  baseModel: string;
  runs: Run[];
}

interface Run {
  id: string;
  name: string;
  method: string;
  status: string;
  color: string;
  progress?: number;
  bestLoss: number | null;
}

interface RlSelection {
  type: "flow" | "evaluator";
  id: string;
}

function toStatus(s: string): Status {
  if (s === "running" || s === "completed" || s === "error" || s === "stopped") return s;
  if (s === "queued") return "stopped";
  if (s === "active") return "running";
  return "stopped";
}

interface ExperimentSidebarProps {
  experiments: Experiment[];
  selectedExpId: string | null;
  selectedRunId: string | null;
  compareRunIds: string[];
  topTab: "sft" | "rl";
  selectedRlItem: RlSelection | null;
  onSelectExp: (id: string) => void;
  onSelectRun: (id: string) => void;
  onToggleCompare: (id: string) => void;
  onClearCompare: () => void;
  onViewComparison: () => void;
  onSelectRlItem: (item: RlSelection | null) => void;
}

export function ExperimentSidebar({
  experiments,
  selectedExpId,
  selectedRunId,
  compareRunIds,
  topTab,
  selectedRlItem,
  onSelectExp,
  onSelectRun,
  onToggleCompare,
  onClearCompare,
  onViewComparison,
  onSelectRlItem,
}: ExperimentSidebarProps) {
  const allRuns = experiments.flatMap(e => e.runs);
  const compareRuns = compareRunIds.map(id => allRuns.find(r => r.id === id)).filter(Boolean) as Run[];

  return (
    <div className="w-85 min-w-85 border-r border-border flex flex-col bg-card">
      <div className="flex-1 overflow-y-auto">
        {experiments.map(experiment => {
          const isExpSelected = selectedExpId === experiment.id;
          const runningCount = experiment.runs.filter(r => r.status === "running").length;

          return (
            <div key={experiment.id}>
              {/* Experiment header */}
              <button
                onClick={() => onSelectExp(experiment.id)}
                className={cn(
                  "w-full text-left px-3.5 py-2.5 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
                  isExpSelected
                    ? "bg-muted/60 border-l-primary"
                    : "border-l-transparent hover:bg-muted/30",
                )}
              >
                <div className="flex items-center gap-2 mb-0.5">
                  <span className={cn("text-[11px] transition-transform", isExpSelected ? "text-primary" : "text-muted-foreground")}>
                    {isExpSelected ? "\u25BE" : "\u25B8"}
                  </span>
                  <span className="text-xs font-semibold text-foreground font-display flex-1 truncate">
                    {experiment.name}
                  </span>
                  {runningCount > 0 && (
                    <span className="flex items-center gap-1 text-[8px] px-1.5 py-px rounded bg-green-500/[0.07] text-green-500 border border-green-500/15">
                      <span className="w-1 h-1 rounded-full bg-green-500 animate-pulse" />
                      {runningCount}
                    </span>
                  )}
                </div>
                <div className="text-[9px] text-muted-foreground/50 pl-[19px]">
                  {experiment.runs.length} runs &middot; {experiment.baseModel.split("/").pop()}
                </div>
              </button>

              {/* Runs inside experiment */}
              {isExpSelected && experiment.runs.map(r => {
                const isRunSelected = selectedRunId === r.id;
                const isCompared = compareRunIds.includes(r.id);
                const mc = METHOD_COLORS[r.method] || METHOD_COLORS.SFT;

                return (
                  <button
                    key={r.id}
                    onClick={() => onSelectRun(r.id)}
                    className={cn(
                      "w-full text-left pl-9 pr-3.5 py-2.5 border-b border-b-border/30 transition-colors cursor-pointer",
                      isRunSelected ? "bg-muted" : "hover:bg-muted/30",
                    )}
                  >
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <div className="w-[3px] h-[18px] rounded-sm shrink-0" style={{ background: r.color }} />
                      <span className={cn(
                        "text-[11px] font-medium flex-1 truncate",
                        isRunSelected ? "text-foreground" : "text-foreground/70",
                      )}>
                        {r.name}
                      </span>
                      <span className={cn(
                        "text-[8px] px-1.5 py-px rounded font-semibold border",
                        mc.bg, mc.fg, mc.border,
                      )}>
                        {r.method}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 pl-[9px] text-[9px]">
                      <StatusDot status={toStatus(r.status)} />
                      <span className={cn(
                        r.status === "running" ? "text-green-500" :
                        r.status === "completed" ? "text-blue-500" :
                        "text-muted-foreground"
                      )}>
                        {r.status}
                      </span>
                      {r.status === "running" && r.progress != null && (
                        <span className="text-muted-foreground/50">{r.progress}%</span>
                      )}
                      {r.bestLoss !== null && (
                        <span className="text-muted-foreground/30">
                          loss: <span className="text-muted-foreground/60">{r.bestLoss}</span>
                        </span>
                      )}
                      <span className="flex-1" />
                      {/* Compare toggle */}
                      <span
                        onClick={e => { e.stopPropagation(); onToggleCompare(r.id); }}
                        className={cn(
                          "w-3.5 h-3.5 rounded-sm border flex items-center justify-center text-[7px] cursor-pointer",
                          isCompared ? "border-current" : "border-border",
                        )}
                        style={isCompared ? { color: r.color, backgroundColor: r.color + "20" } : undefined}
                      >
                        {isCompared ? "\u2713" : ""}
                      </span>
                    </div>
                    {r.status === "running" && r.progress != null && (
                      <div className="mt-1 pl-[9px]">
                        <ProgressBar value={r.progress} color={r.color} />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* Compare bar */}
      {compareRunIds.length > 0 && (
        <div className="px-3.5 py-2 border-t border-border bg-muted/40">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-primary font-semibold font-display">
              Comparing {compareRunIds.length} runs
            </span>
            <button onClick={onClearCompare} className="bg-transparent border-none text-muted-foreground/40 cursor-pointer text-[9px] hover:text-foreground">
              Clear
            </button>
          </div>
          <div className="flex gap-1 mb-1.5">
            {compareRuns.map(r => (
              <span
                key={r.id}
                className="text-[8px] px-1.5 py-0.5 rounded-sm border"
                style={{ background: r.color + "15", color: r.color, borderColor: r.color + "30" }}
              >
                {r.name.substring(0, 20)}
              </span>
            ))}
          </div>
          <button
            onClick={onViewComparison}
            className="w-full py-1.5 rounded border border-primary/20 bg-primary/[0.07] text-primary text-[10px] cursor-pointer font-display font-semibold hover:bg-primary/15 transition-colors"
          >
            View Comparison &rarr;
          </button>
        </div>
      )}

      {/* RL: Agent Flows & Evaluators */}
      {topTab === "rl" && (
        <div className="border-t border-border shrink-0 max-h-65 overflow-y-auto">
          {/* Agent Flows */}
          <div className="px-3.5 pt-2 pb-1 flex justify-between items-center">
            <span className="text-[9px] font-semibold text-muted-foreground/40 uppercase tracking-widest font-display">
              Agent Flows
            </span>
            <button className="bg-transparent border-none text-blue-500 cursor-pointer text-[9px] font-display font-semibold">
              + New
            </button>
          </div>
          {AGENT_FLOWS.map(f => {
            const isSel = selectedRlItem?.type === "flow" && selectedRlItem.id === f.id;
            return (
              <button
                key={f.id}
                onClick={() => onSelectRlItem(isSel ? null : { type: "flow", id: f.id })}
                className={cn(
                  "w-full text-left px-3.5 py-1.5 border-l-2 transition-colors cursor-pointer",
                  isSel
                    ? "bg-muted/60 border-l-blue-500"
                    : "border-l-transparent hover:bg-muted/30",
                )}
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-blue-500">{f.icon}</span>
                  <span className="text-[10px] font-medium text-foreground font-display flex-1">{f.name}</span>
                  <span className="text-[7px] text-muted-foreground/40">
                    {f.type === "multi-agent" ? "multi" : "single"}
                  </span>
                </div>
              </button>
            );
          })}

          {/* Evaluators */}
          <div className="px-3.5 pt-2 pb-1 flex justify-between items-center mt-1">
            <span className="text-[9px] font-semibold text-muted-foreground/40 uppercase tracking-widest font-display">
              Evaluators
            </span>
            <button className="bg-transparent border-none text-green-500 cursor-pointer text-[9px] font-display font-semibold">
              + New
            </button>
          </div>
          {EVALUATORS.map(ev => {
            const isSel = selectedRlItem?.type === "evaluator" && selectedRlItem.id === ev.id;
            return (
              <button
                key={ev.id}
                onClick={() => onSelectRlItem(isSel ? null : { type: "evaluator", id: ev.id })}
                className={cn(
                  "w-full text-left px-3.5 py-1.5 border-l-2 transition-colors cursor-pointer",
                  isSel
                    ? "bg-muted/60 border-l-green-500"
                    : "border-l-transparent hover:bg-muted/30",
                )}
              >
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-green-500">{ev.icon}</span>
                  <span className="text-[10px] font-medium text-foreground font-display flex-1">{ev.name}</span>
                  <StatusDot status={ev.status === "active" ? "running" : "error"} />
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export { METHOD_COLORS, toStatus };
export type { RlSelection };
