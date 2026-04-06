// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Sparkline } from "@/components/ui/sparkline";
import { StatusDot } from "@/components/ui/status-dot";
import { toStatus } from "./models-data";
import type { Model } from "./models-data";

// ── Gauge ring (SVG donut) ─────────────────────────────────────

function GaugeRing({
  value,
  max = 100,
  color,
  size = 72,
  label,
  sublabel,
}: {
  value: number;
  max?: number;
  color: string;
  size?: number;
  label: string;
  sublabel?: string;
}) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          className="block"
          style={{ transform: "rotate(-90deg)" }}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            className="stroke-border"
            strokeWidth="5"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="5"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-[stroke-dashoffset] duration-600 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-foreground tracking-tight">
            {value}%
          </span>
        </div>
      </div>
      <div className="text-center">
        <div className="text-[9px] text-muted-foreground font-display">
          {label}
        </div>
        {sublabel && (
          <div className="text-[8px] text-muted-foreground/40">{sublabel}</div>
        )}
      </div>
    </div>
  );
}

// ── Overview tab ───────────────────────────────────────────────

export function OverviewTab({ model }: { model: Model }) {
  return (
    <div className="animate-in fade-in duration-150">
      {/* Key metrics */}
      <div className="grid grid-cols-4 gap-2.5 mb-5">
        {[
          {
            label: "Throughput",
            value: model.tps > 0 ? model.tps.toLocaleString() : "\u2014",
            unit: "tok/s",
            spark: model.metricsHistory.tps,
            color: "#22C55E",
          },
          {
            label: "P99 Latency",
            value: model.p99,
            unit: "",
            spark: model.metricsHistory.latency,
            color: "#3B82F6",
          },
          {
            label: "Queue Depth",
            value: model.queueDepth,
            unit: "reqs",
            spark: model.metricsHistory.queue,
            color: "#F59E0B",
          },
          {
            label: "Requests (24h)",
            value:
              model.requests24h > 0
                ? model.requests24h.toLocaleString()
                : "0",
            unit: "",
            spark: model.metricsHistory.tps,
            color: "#8B5CF6",
          },
        ].map((m) => (
          <div
            key={m.label}
            className="bg-muted/40 border border-border rounded-lg px-3.5 py-3 flex justify-between items-end"
          >
            <div>
              <div className="text-[9px] text-muted-foreground/50 uppercase tracking-wide mb-1 font-display">
                {m.label}
              </div>
              <div className="text-xl font-bold text-foreground tracking-tight">
                {m.value}{" "}
                {m.unit && (
                  <span className="text-[10px] text-muted-foreground font-normal">
                    {m.unit}
                  </span>
                )}
              </div>
            </div>
            <Sparkline data={m.spark} color={m.color} height={28} width={70} />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_300px] gap-4">
        {/* Left column */}
        <div className="flex flex-col gap-4">
          {/* GPU Resources */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-4">
              GPU Resources
            </div>
            {model.status === "running" ? (
              <div className="flex gap-8 items-start">
                <div className="flex gap-5 items-center">
                  <GaugeRing
                    value={model.gpu.utilization}
                    color="#22C55E"
                    label="GPU Util"
                    sublabel={`${model.gpu.count}\u00D7 ${model.gpu.type}`}
                  />
                  <GaugeRing
                    value={model.vram.pct}
                    color="#8B5CF6"
                    label="VRAM"
                    sublabel={`${model.vram.used} / ${model.vram.total}`}
                  />
                </div>
                <div className="flex-1">
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: "GPU Type", value: model.gpu.type },
                      { label: "GPU Count", value: model.gpu.count },
                      { label: "VRAM Used", value: model.vram.used },
                      { label: "VRAM Total", value: model.vram.total },
                      {
                        label: "Tensor Parallel",
                        value:
                          model.servingConfig?.tensorParallelSize ?? "\u2014",
                      },
                      { label: "Batch Size", value: model.batchSize },
                    ].map((f) => (
                      <div key={f.label}>
                        <div className="text-[9px] text-muted-foreground/40 uppercase tracking-wide mb-0.5 font-display">
                          {f.label}
                        </div>
                        <div className="text-[11px] text-foreground/70">
                          {f.value}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-4 text-center text-muted-foreground/40 text-[11px]">
                {toStatus(model.status) === "error"
                  ? "Model failed to start \u2014 check Events tab for details"
                  : "Model not currently serving \u2014 no GPU allocated"}
              </div>
            )}
          </section>

          {/* Deployment Info */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Deployment Info
            </div>
            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  label: "Replicas",
                  value: `${model.replicas.current} / ${model.replicas.desired}`,
                },
                { label: "Engine", value: model.engine },
                { label: "Quantization", value: model.quantization },
                {
                  label: "Context Window",
                  value: model.contextWindow.toLocaleString(),
                },
                { label: "Namespace", value: model.namespace },
                { label: "Uptime", value: model.uptime },
                { label: "Error Rate", value: model.errorRate },
                { label: "Last Deploy", value: model.lastDeployed },
                { label: "By", value: model.deployedBy },
              ].map((f) => (
                <div key={f.label}>
                  <div className="text-[9px] text-muted-foreground/40 uppercase tracking-wide mb-0.5 font-display">
                    {f.label}
                  </div>
                  <div className="text-[11px] text-foreground/70 truncate">
                    {f.value}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          {/* Connected Agents */}
          <section className="bg-muted/40 border border-border rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b border-border flex items-center gap-2">
              <span className="text-amber-500">&#x2B21;</span>
              <span className="text-xs font-semibold text-foreground font-display">
                Connected Agents
              </span>
              <span className="text-[9px] px-1.5 py-px bg-muted rounded text-muted-foreground">
                {model.connectedAgents.length}
              </span>
            </div>
            {model.connectedAgents.length === 0 ? (
              <div className="py-5 text-center text-muted-foreground/40 text-[11px]">
                No agents using this model
              </div>
            ) : (
              model.connectedAgents.map((a) => (
                <div
                  key={a.name}
                  className="px-4 py-2.5 border-b border-border/50 last:border-b-0 flex items-center justify-between hover:bg-muted/30 transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <StatusDot status={toStatus(a.status)} />
                    <span className="text-xs text-foreground font-medium">
                      {a.name}
                    </span>
                  </div>
                  <span className="text-[10px] text-muted-foreground">
                    {a.rps} rps
                  </span>
                </div>
              ))
            )}
          </section>

          {/* Tokens (24h) */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Tokens (24h)
            </div>
            <div className="flex justify-between items-center">
              <div className="text-center">
                <div className="text-lg font-bold text-green-500 tracking-tight">
                  {model.tokensIn24h}
                </div>
                <div className="text-[9px] text-muted-foreground/40 mt-0.5 font-display">
                  Input
                </div>
              </div>
              <div className="text-muted-foreground/20 text-base">
                &rarr;
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-500 tracking-tight">
                  {model.tokensOut24h}
                </div>
                <div className="text-[9px] text-muted-foreground/40 mt-0.5 font-display">
                  Output
                </div>
              </div>
            </div>
          </section>

          {/* Model Card */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-2.5">
              Model Card
            </div>
            <div className="flex flex-col gap-1.5">
              {[
                { label: "Family", value: model.family },
                { label: "Parameters", value: model.paramCount },
                {
                  label: "Base",
                  value: model.base.split("/").pop() ?? model.base,
                },
                { label: "Hub Ref", value: model.hubRef },
              ].map((f) => (
                <div
                  key={f.label}
                  className="flex justify-between text-[10px]"
                >
                  <span className="text-muted-foreground/40">{f.label}</span>
                  <span className="text-muted-foreground max-w-[160px] truncate text-right">
                    {f.value}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
