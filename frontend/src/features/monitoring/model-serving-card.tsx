// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { MODEL_HEALTH } from "./monitoring-data";

export function ModelServingCard() {
  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span style={{ color: "#3B82F6" }}>&#x25C7;</span>
        <span className="text-xs font-semibold text-foreground font-display">Model Serving</span>
      </div>
      {MODEL_HEALTH.map((m) => (
        <div key={m.name} className="px-4 py-2 border-b border-border/50">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <StatusDot status={m.status} />
              <span className="text-[11px] text-foreground font-medium">{m.name}</span>
            </div>
            <span className="text-[10px] font-medium" style={{ color: m.tps > 0 ? undefined : "var(--muted-foreground)" }}>
              {m.tps > 0 ? `${m.tps} tok/s` : "—"}
            </span>
          </div>
          {m.status === "serving" ? (
            <div className="flex gap-2">
              <div className="flex-1 flex items-center gap-1.5">
                <span className="text-[9px] text-muted-foreground font-display w-7">GPU</span>
                <div className="flex-1">
                  <ProgressBar value={m.gpu} color={m.gpu > 80 ? "#EF4444" : "#22C55E"} />
                </div>
                <span className="text-[10px] w-7 text-right font-medium" style={{ color: m.gpu > 80 ? "#EF4444" : "var(--muted-foreground)" }}>
                  {m.gpu}%
                </span>
              </div>
              <div className="flex-1 flex items-center gap-1.5">
                <span className="text-[9px] text-muted-foreground font-display w-7">VRAM</span>
                <div className="flex-1">
                  <ProgressBar value={m.vram} color={m.vram > 80 ? "#EF4444" : "#8B5CF6"} />
                </div>
                <span className="text-[10px] w-7 text-right font-medium" style={{ color: m.vram > 80 ? "#EF4444" : "var(--muted-foreground)" }}>
                  {m.vram}%
                </span>
              </div>
            </div>
          ) : (
            <div className="text-[9px] text-destructive">OOM — needs VRAM increase</div>
          )}
        </div>
      ))}
    </section>
  );
}
