// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot } from "@/components/ui/status-dot";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function RlItemDetail({ item, isFlow, onClose }: { item: any; isFlow: boolean; onClose: () => void }) {
  const accent = isFlow ? "#3B82F6" : "#22C55E";
  const accentTextClass = isFlow ? "text-blue-500" : "text-green-500";

  return (
    <div className="flex-1 flex flex-col overflow-hidden animate-in fade-in duration-150">
      {/* Header */}
      <div className="px-6 py-5 border-b border-border shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2.5">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center text-lg border"
              style={{
                backgroundColor: accent + "10",
                borderColor: accent + "22",
                color: accent,
              }}
            >
              {item.icon}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-foreground font-display">
                  {item.name}
                </span>
                <span
                  className="text-[8px] px-1.5 py-0.5 rounded font-semibold border"
                  style={{
                    background: accent + "12",
                    color: accent,
                    borderColor: accent + "25",
                  }}
                >
                  {isFlow ? "AGENT FLOW" : "EVALUATOR"}
                </span>
                <StatusDot status={item.status === "active" ? "running" : "error"} />
              </div>
              <div className="text-[10px] text-muted-foreground/60 mt-0.5">
                {isFlow
                  ? `${item.type} \u00B7 ${item.agents.join(", ")} \u00B7 ${item.stepsPerEpisode} steps/episode`
                  : `${item.rewardType} reward \u00B7 ${item.signals.length} signals \u00B7 ${item.usedByRuns} runs`}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5">
            <Button variant="outline" size="xs">Edit</Button>
            <Button variant="outline" size="xs" onClick={onClose}>
              {"\u2715"}
            </Button>
          </div>
        </div>
        <p className="text-[11px] text-muted-foreground leading-relaxed">{item.description}</p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {/* Signals (evaluator) or Agents (flow) */}
        {isFlow ? (
          <div className="mb-4">
            <div className="text-[11px] font-semibold text-foreground font-display mb-2">
              Agents (Trajectories)
            </div>
            <div className="flex gap-1.5">
              {item.agents.map((a: string) => (
                <div
                  key={a}
                  className="px-3.5 py-2 bg-muted/50 border border-border rounded-md flex items-center gap-1.5"
                >
                  <span className="text-[11px] text-blue-500">{"\u2B21"}</span>
                  <span className="text-[11px] text-foreground">{a}</span>
                </div>
              ))}
            </div>
            <div className="text-[9px] text-muted-foreground/40 mt-1.5">
              Each agent produces one Trajectory per Episode. Grouped by task_id:trajectory.name for advantage computation.
            </div>
          </div>
        ) : (
          <div className="mb-4">
            <div className="text-[11px] font-semibold text-foreground font-display mb-2">
              Signals
            </div>
            <div className="flex gap-1 flex-wrap">
              {item.signals.map((s: string) => (
                <Badge key={s} variant="active" className="bg-green-500/[0.07] text-green-500 border border-green-500/15">
                  {s}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Configuration */}
        <div className="mb-4">
          <div className="text-[11px] font-semibold text-foreground font-display mb-2">
            Configuration
          </div>
          <div className="bg-card rounded-md ring-1 ring-foreground/10 overflow-hidden">
            {Object.entries(item.config).map(([k, v]) => (
              <div key={k} className="px-3.5 py-[7px] border-b border-border/30 flex gap-3 text-[11px]">
                <span className={`${accentTextClass} min-w-40`}>{k}</span>
                <span className="text-muted-foreground/20">=</span>
                <span className="text-foreground/70">
                  {Array.isArray(v) ? v.join(", ") : String(v)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Source (flow) */}
        {isFlow && item.source && (
          <div className="mb-4">
            <div className="text-[11px] font-semibold text-foreground font-display mb-2">
              Source
            </div>
            <code className="text-[11px] px-3 py-2 bg-muted/50 border border-border rounded-md block text-foreground/70">
              {item.source}
            </code>
          </div>
        )}

        {/* Summary cards */}
        <div className={`grid gap-2 ${isFlow ? "grid-cols-3" : "grid-cols-2"}`}>
          {(isFlow
            ? [
                { label: "Type", value: item.type },
                { label: "Max Rollout Tokens", value: item.maxRolloutTokens.toLocaleString() },
                { label: "Used by Runs", value: item.usedByRuns },
              ]
            : [
                { label: "Reward Type", value: item.rewardType },
                { label: "Used by Runs", value: item.usedByRuns },
              ]
          ).map(f => (
            <div key={f.label} className="bg-card rounded-md ring-1 ring-foreground/10 px-3 py-2.5">
              <div className="text-[8px] text-muted-foreground/50 uppercase tracking-wide mb-0.5 font-display">
                {f.label}
              </div>
              <div className="text-sm font-bold text-foreground">{f.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
