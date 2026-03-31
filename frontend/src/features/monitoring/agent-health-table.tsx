// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Sparkline } from "@/components/ui/sparkline";
import { AGENTS_HEALTH } from "./monitoring-data";

const STATUS_COLORS: Record<string, string> = {
  healthy: "#22C55E",
  degraded: "#F59E0B",
  down: "#EF4444",
};

const COLUMNS = ["Agent", "Status", "RPS", "P50", "P95", "P99", "Errors", "CPU", "MEM", "Trend"];

export function AgentHealthTable() {
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  return (
    <section className="rounded-lg border border-border bg-card overflow-hidden">
      <div className="px-4 py-3 border-b border-border flex items-center gap-2">
        <span className="text-primary">&#x2B21;</span>
        <span className="text-[13px] font-semibold text-foreground font-display">Agent Health</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              {COLUMNS.map((h) => (
                <th key={h} className="px-2.5 py-[7px] text-left text-[9px] font-medium text-muted-foreground uppercase tracking-widest font-display">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {AGENTS_HEALTH.map((a, i) => (
              <tr
                key={a.name}
                className="border-b border-border/50 cursor-pointer transition-colors"
                style={{ background: hoveredRow === i ? "var(--muted)" : undefined }}
                onMouseEnter={() => setHoveredRow(i)}
                onMouseLeave={() => setHoveredRow(null)}
              >
                <td className="px-2.5 py-2">
                  <div className="flex items-center gap-1.5">
                    <div className="w-[3px] h-5 rounded-sm shrink-0" style={{ background: a.color }} />
                    <div>
                      <div className="text-[11px] text-foreground font-medium">{a.name}</div>
                      <div className="text-[8px] text-muted-foreground">{a.namespace} · {a.replicas}</div>
                    </div>
                  </div>
                </td>
                <td className="px-2.5 py-2">
                  <span
                    className="inline-flex items-center gap-1 text-[9px] px-[7px] py-[2px] rounded font-semibold uppercase tracking-wide"
                    style={{
                      background: STATUS_COLORS[a.status] + "12",
                      color: STATUS_COLORS[a.status],
                    }}
                  >
                    <span
                      className="inline-block w-[5px] h-[5px] rounded-full shrink-0"
                      style={{
                        backgroundColor: STATUS_COLORS[a.status],
                        boxShadow: `0 0 6px ${STATUS_COLORS[a.status]}`,
                        animation: a.status === "down" ? "pulse 2s ease-in-out infinite" : undefined,
                      }}
                    />
                    {a.status}
                  </span>
                </td>
                <td className="px-2.5 py-2 text-[11px] text-foreground/80 font-semibold">{a.rps}</td>
                <td className="px-2.5 py-2 text-[10px] text-muted-foreground">{a.p50 || "—"}</td>
                <td className="px-2.5 py-2 text-[10px] text-muted-foreground">{a.p95 || "—"}</td>
                <td className="px-2.5 py-2 text-[10px]" style={{ color: a.p99 > 1000 ? "#F59E0B" : undefined, fontWeight: a.p99 > 1000 ? 600 : 400 }}>
                  {a.p99 || "—"}
                </td>
                <td className="px-2.5 py-2 text-[10px]" style={{ color: a.errorRate > 1.5 ? "#EF4444" : undefined, fontWeight: a.errorRate > 1.5 ? 600 : 400 }}>
                  {a.rps > 0 ? `${a.errorRate}%` : "—"}
                </td>
                <td className="px-2.5 py-2 w-[70px]">
                  <ProgressBar value={a.cpu} color="#3B82F6" />
                </td>
                <td className="px-2.5 py-2 w-[70px]">
                  <ProgressBar value={a.mem} color="#8B5CF6" />
                </td>
                <td className="px-2.5 py-2">
                  <Sparkline data={a.sparkRps} color={a.color} height={20} width={56} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
