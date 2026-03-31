// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Sparkline } from "@/components/ui/sparkline";
import { ProgressBar } from "@/components/ui/progress-bar";
import type { Model } from "./models-data";

export function PerformanceTab({ model }: { model: Model }) {
  const charts = [
    {
      label: "Throughput (tok/s)",
      data: model.metricsHistory.tps,
      color: "#22C55E",
      unit: "tok/s",
    },
    {
      label: "P99 Latency (ms)",
      data: model.metricsHistory.latency,
      color: "#3B82F6",
      unit: "ms",
    },
    {
      label: "GPU Utilization (%)",
      data: model.metricsHistory.gpu,
      color: "#8B5CF6",
      unit: "%",
    },
    {
      label: "Queue Depth",
      data: model.metricsHistory.queue,
      color: "#F59E0B",
      unit: "reqs",
    },
  ];

  const latencyBars = [
    { label: "P50", value: model.p50, pct: 40, color: "#22C55E" },
    { label: "P95", value: model.p95, pct: 70, color: "#F59E0B" },
    { label: "P99", value: model.p99, pct: 90, color: "#EF4444" },
  ];

  return (
    <div className="animate-in fade-in duration-150 space-y-4">
      {/* Sparkline charts */}
      <div className="grid grid-cols-2 gap-4">
        {charts.map((chart) => (
          <section
            key={chart.label}
            className="bg-muted/40 border border-border rounded-lg overflow-hidden"
          >
            <div className="px-4 py-3 border-b border-border flex items-center justify-between">
              <span className="text-xs font-semibold text-foreground font-display">
                {chart.label}
              </span>
              <span
                className="text-[11px] font-semibold"
                style={{ color: chart.color }}
              >
                {chart.data[chart.data.length - 1]}{" "}
                <span className="text-[9px] text-muted-foreground font-normal">
                  {chart.unit}
                </span>
              </span>
            </div>
            <div className="p-4 pb-3">
              <Sparkline
                data={chart.data}
                color={chart.color}
                height={80}
                width={420}
              />
              <div className="flex justify-between text-[9px] text-muted-foreground/40 mt-2 font-display">
                <span>15m ago</span>
                <span>now</span>
              </div>
            </div>
          </section>
        ))}
      </div>

      {/* Latency Distribution */}
      <section className="bg-muted/40 border border-border rounded-lg p-4">
        <div className="text-xs font-semibold text-foreground font-display mb-3.5">
          Latency Distribution
        </div>
        <div className="flex gap-8 items-end">
          {latencyBars.map((p) => (
            <div
              key={p.label}
              className="flex-1 flex flex-col items-center gap-1.5"
            >
              <div
                className="text-base font-bold"
                style={{ color: p.color }}
              >
                {p.value}
              </div>
              <ProgressBar value={p.pct} max={100} color={p.color} />
              <div className="text-[9px] text-muted-foreground font-display">
                {p.label}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
