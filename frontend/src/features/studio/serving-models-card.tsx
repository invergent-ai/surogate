// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect } from "react";
import { Card } from "@/components/ui/card";
import { StatusDot } from "@/components/ui/status-dot";
import { cn } from "@/utils/cn";
import { useAppStore } from "@/stores/app-store";
import { toStatus } from "@/features/models/models-data";
import { useNavigate } from "@tanstack/react-router";

export function ServingModelsCard() {
  const models = useAppStore((s) => s.models);
  const fetchModels = useAppStore((s) => s.fetchModels);
  const navigate = useNavigate();

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  return (
    <Card>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <span style={{ color: "#3B82F6" }}>{"\u25C7"}</span>
        <span className="text-sm font-semibold text-foreground font-display">Serving Models</span>
        <span className="text-[11px] px-1.5 py-px rounded bg-muted text-faint">{models.length}</span>
      </div>
      {models.length === 0 && (
        <div className="px-4 py-6 text-center text-faint text-xs">No models deployed</div>
      )}
      {models.map((m) => (
        <div
          key={m.id}
          className="px-4 py-2.5 border-b border-input flex items-center gap-2.5 cursor-pointer hover:bg-muted/30 transition-colors"
          onClick={() => navigate({ to: "/studio/models", search: { modelId: m.id } })}
        >
          <StatusDot status={toStatus(m.status)} />
          <div className="flex-1 min-w-0">
            <div className="text-foreground font-medium truncate">{m.displayName || m.name}</div>
            <div className="text-[9px] text-faint mt-px">
              {m.family} {"\u00b7"} {m.paramCount}
              {m.gpu.count > 0 && <> {"\u00b7"} {m.gpu.count}{"\u00d7"} {m.gpu.type}</>}
            </div>
          </div>
          <div className="text-right">
            <div className={cn("font-medium", m.tps > 0 ? "text-foreground/80" : "text-faint")}>
              {m.tps > 0 ? m.tps : "\u2014"}
            </div>
            <div className="text-[8px] text-faint">tok/s</div>
          </div>
        </div>
      ))}
    </Card>
  );
}
