// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ALERTS, SEVERITY_COLORS } from "./monitoring-data";

export function AlertsPanel() {
  const [hoveredAlert, setHoveredAlert] = useState<number | null>(null);
  const activeAlerts = ALERTS.filter((a) => !a.acknowledged).length;

  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-destructive">&#x26A0;</span>
          <span className="text-[13px] font-semibold text-foreground font-display">Alerts</span>
          {activeAlerts > 0 && <Badge variant="danger">{activeAlerts}</Badge>}
        </div>
      </div>
      {ALERTS.map((a, i) => {
        const sevColor = SEVERITY_COLORS[a.severity];
        return (
          <div
            key={a.id}
            className="px-4 py-2.5 border-b border-border/50 cursor-pointer transition-all"
            style={{
              background: hoveredAlert === i ? "var(--muted)" : undefined,
              opacity: a.acknowledged ? 0.5 : 1,
              borderLeft: !a.acknowledged ? `2px solid ${sevColor}` : "2px solid transparent",
            }}
            onMouseEnter={() => setHoveredAlert(i)}
            onMouseLeave={() => setHoveredAlert(null)}
          >
            <div className="flex items-center gap-1.5 mb-1">
              <span
                className="text-[7px] px-[5px] py-[1px] rounded font-bold uppercase tracking-wider"
                style={{ background: sevColor + "15", color: sevColor }}
              >
                {a.severity}
              </span>
              <span className="text-[9px] text-muted-foreground font-display">{a.time}</span>
            </div>
            <div className="text-[11px] font-medium font-display mb-0.5" style={{ color: !a.acknowledged ? "var(--foreground)" : "var(--muted-foreground)" }}>
              {a.title}
            </div>
            <div className="text-[9px] text-muted-foreground leading-relaxed">{a.message}</div>
            {!a.acknowledged && (
              <Button variant="outline" size="xs" className="mt-1.5 text-[9px]">
                Acknowledge
              </Button>
            )}
          </div>
        );
      })}
    </section>
  );
}
