// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { PIPELINE_TYPE_STYLES, toStatus } from "./datasets-data";
import type { Dataset } from "./datasets-data";

export function PipelineTab({ dataset }: { dataset: Dataset }) {
  if (dataset.pipeline.length === 0) {
    return (
      <div className="animate-in fade-in duration-150">
        <div className="py-10 text-center">
          <div className="text-xs text-muted-foreground/40 mb-3">
            No transformation pipeline configured
          </div>
          <Button size="sm">+ Build Pipeline</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-in fade-in duration-150">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Transformation Pipeline
          </span>
          <span className="text-[9px] px-1.5 py-px rounded bg-violet-500/10 text-violet-500 font-medium">
            NeMo Data Designer
          </span>
        </div>
        <div className="flex gap-1.5">
          <Button variant="outline" size="xs">Edit Pipeline</Button>
          <Button variant="outline" size="xs">Re-run</Button>
        </div>
      </div>

      {/* Pipeline visualization */}
      <div className="relative pl-8">
        {dataset.pipeline.map((step, i) => {
          const isLast = i === dataset.pipeline.length - 1;
          const ts = PIPELINE_TYPE_STYLES[step.type] ?? PIPELINE_TYPE_STYLES.transform;
          const statusColor = step.status === "completed" ? "text-green-500" : step.status === "running" ? "text-amber-500" : "text-muted-foreground";

          return (
            <div key={step.id} className={cn("relative", !isLast && "mb-1")}>
              {/* Connector line */}
              {!isLast && (
                <div
                  className={cn(
                    "absolute left-[-20px] top-7 w-px",
                    step.status === "completed" ? "bg-border" : "bg-border/50",
                  )}
                  style={{ height: "calc(100% - 8px)" }}
                />
              )}

              {/* Node */}
              <div
                className={cn(
                  "absolute left-[-28px] top-2 w-4 h-4 rounded flex items-center justify-center text-[8px] border-2",
                  step.status === "completed" && "bg-green-500/10 border-green-500 text-green-500",
                  step.status === "running" && "bg-amber-500/10 border-amber-500 text-amber-500 animate-pulse",
                  step.status === "pending" && "bg-muted border-muted-foreground/30 text-muted-foreground",
                )}
              >
                {step.status === "completed" ? "\u2713" : step.status === "running" ? "\u27F3" : "\u00B7"}
              </div>

              {/* Step card */}
              <div
                className={cn(
                  "bg-muted/40 border rounded-lg px-4 py-3",
                  step.status === "running" ? "border-amber-500/30" : "border-border",
                )}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className={cn("text-sm", ts.fg)}>{step.icon}</span>
                    <span className="text-xs font-semibold text-foreground font-display">{step.step}</span>
                    <span className={cn("text-[8px] font-medium px-1.5 py-px rounded border uppercase", ts.bg, ts.fg, `${ts.fg.replace("text-", "border-")}/20`)}>
                      {step.type}
                    </span>
                  </div>
                  <span className="flex items-center gap-1 text-[9px]">
                    <StatusDot status={toStatus(step.status)} />
                    <span className={statusColor}>{step.status}</span>
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground">{step.detail}</div>
                {step.status === "running" && (
                  <div className="h-[3px] bg-border rounded-sm overflow-hidden mt-2">
                    <div className="h-full w-[65%] rounded-sm bg-gradient-to-r from-amber-500 to-amber-400 animate-[progress-stripe_1s_linear_infinite] bg-[length:40px_40px]" style={{
                      backgroundImage: "linear-gradient(45deg, rgba(255,255,255,.08) 25%, transparent 25%, transparent 50%, rgba(255,255,255,.08) 50%, rgba(255,255,255,.08) 75%, transparent 75%)",
                    }} />
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
