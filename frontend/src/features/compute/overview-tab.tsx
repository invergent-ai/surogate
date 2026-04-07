// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Card, CardContent } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import { useAppStore } from "@/stores/app-store";
import {
  CLOUD_ACCOUNTS,
  PROVIDER_COLORS,
} from "./compute-data";
import { useWorkloadItems } from "./use-workload-items";
import { statusForDot } from "./detail-shared";

export function OverviewTab() {
  const k8sNodes = useAppStore((s) => s.k8sNodes);
  const cloudInstances = useAppStore((s) => s.cloudInstances);
  const workloads = useWorkloadItems();

  const totalLocalGpu = k8sNodes.reduce((s, n) => s + n.accelerator_count, 0);
  const usedLocalGpu = totalLocalGpu - k8sNodes.reduce((s, n) => s + n.accelerator_available, 0);
  const readyNodes = k8sNodes.filter((n) => n.is_ready).length;
  const cloudOnly = cloudInstances.filter(c => c.provider !== "kubernetes");
  const activeInstances = cloudOnly.filter(c => c.status === "idle" || c.status === "busy");
  const cloudHourlyCost = cloudOnly.reduce((s, c) => s + c.cost_per_hour, 0);
  const monthlySpend = CLOUD_ACCOUNTS.reduce((s, a) => s + a.monthlySpend, 0);
  const monthlyBudget = CLOUD_ACCOUNTS.reduce((s, a) => s + a.monthlyBudget, 0);

  const running = workloads.filter(w => statusForDot(w.status) === "running").length;
  const queued = workloads.filter(w => w.status === "queued" || w.status === "pending").length;

  const kpis = [
    { label: "Local GPUs", value: `${usedLocalGpu}/${totalLocalGpu}`, sub: `${readyNodes} nodes` },
    { label: "Cloud Instances", value: activeInstances.length, sub: `${cloudOnly.length} total` },
    { label: "Queue Depth", value: queued, sub: `${running} running` },
    { label: "Cloud $/hr", value: `$${cloudHourlyCost.toFixed(0)}`, sub: "active instances" },
    { label: "Monthly Spend", value: `$${(monthlySpend / 1000).toFixed(1)}K`, sub: `of $${(monthlyBudget / 1000).toFixed(0)}K budget` },
    { label: "Spot Savings", value: "\u2014", sub: "" },
  ];

  return (
    <div className="space-y-5 animate-in fade-in duration-200">
      {/* KPI row */}
      <div className="grid grid-cols-6 gap-2.5">
        {kpis.map((m) => (
          <Card key={m.label} size="sm">
            <CardContent className="p-3">
              <div className="text-[10px] text-faint uppercase tracking-wider font-display mb-1">{m.label}</div>
              <div className="text-xl font-bold text-foreground tracking-tight">{m.value}</div>
              <div className="text-[11px] text-primary mt-0.5">{m.sub}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_380px] gap-4">
        {/* Cluster map */}
        <Card size="sm">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
            <span className="text-sm font-semibold text-foreground font-display">Cluster Map</span>
            <div className="flex gap-2.5 text-[11px]">
              {[
                { label: "Used", color: "var(--primary)" },
                { label: "Free", color: "#1A1F2E" },
              ].map(l => (
                <span key={l.label} className="flex items-center gap-1 text-faint">
                  <span className="w-2 h-2 rounded-sm" style={{ background: l.color }} />
                  {l.label}
                </span>
              ))}
            </div>
          </div>
          <div className="p-4 grid grid-cols-4 gap-2">
            {k8sNodes.map(node => {
              const gpuTotal = node.accelerator_count;
              const gpuFree = node.accelerator_available;
              const gpuUsed = gpuTotal - gpuFree;
              const cpuUtil = node.metrics?.cpu_utilization_percent ?? 0;
              const memTotalGb = node.memory_gb ?? 0;
              const memFreeGb = node.metrics?.free_memory_bytes
                ? node.metrics.free_memory_bytes / (1024 ** 3)
                : 0;
              const memUsedGb = memTotalGb - memFreeGb;

              return (
                <Card
                  key={node.name}
                  size="sm"
                  className="p-2.5 cursor-pointer transition-all hover:border-border"
                  style={{ opacity: node.is_ready ? 1 : 0.5 }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] text-muted-foreground font-medium">{node.name}</span>
                    <StatusDot status={node.is_ready ? "running" : "error"} />
                  </div>
                  <div className="text-[10px] text-faint mb-1.5">
                    {gpuTotal > 0 ? `${gpuTotal}\u00d7 ${node.accelerator_type ?? "GPU"}` : "CPU only"}
                  </div>
                  {gpuTotal > 0 && (
                    <div className="grid gap-0.5 mb-1.5" style={{ gridTemplateColumns: `repeat(${gpuTotal}, 1fr)` }}>
                      {Array.from({ length: gpuUsed }, (_, i) => (
                        <div key={`u${i}`} className="h-4 rounded" style={{ background: "var(--primary)", border: "1px solid var(--primary)" }} title="used" />
                      ))}
                      {Array.from({ length: gpuFree }, (_, i) => (
                        <div key={`f${i}`} className="h-4 rounded" style={{ background: "#1A1F2E", border: "1px solid var(--border)" }} title="free" />
                      ))}
                    </div>
                  )}
                  <ProgressBar value={cpuUtil} color="var(--primary)" />
                  <div className="text-[10px] text-faint mt-1">
                    {gpuTotal > 0 && `GPU ${gpuUsed}/${gpuTotal} · `}CPU {cpuUtil.toFixed(0)}% · {memUsedGb.toFixed(0)}/{memTotalGb.toFixed(0)}Gi
                  </div>
                </Card>
              );
            })}
          </div>
        </Card>

        {/* Right sidebar */}
        <div className="flex flex-col gap-4">
          {/* Cloud instances */}
          <Card size="sm">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
              <span className="text-sm font-semibold text-foreground font-display">Cloud Instances</span>
              <span className="text-[11px] px-1.5 py-px rounded bg-muted text-faint">{cloudOnly.length}</span>
            </div>
            {cloudOnly.length === 0 && (
              <div className="px-4 py-4 text-center text-[11px] text-faint">No active cloud instances</div>
            )}
            {cloudOnly.map(inst => {
              const isRunning = inst.status === "idle" || inst.status === "busy";
              return (
                <div key={inst.id} className="px-4 py-2.5 border-b border-line last:border-0">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1.5">
                      <StatusDot status={isRunning ? "running" : "deploying"} />
                      <span className="text-sm text-foreground font-medium">{inst.workload || inst.name}</span>
                    </div>
                    <span className="text-sm font-semibold" style={{ color: "#F59E0B" }}>${inst.cost_per_hour}/hr</span>
                  </div>
                  <div className="flex gap-2 text-[11px] text-faint">
                    <span style={{ color: PROVIDER_COLORS[inst.provider] }}>{inst.provider.toUpperCase()}</span>
                    <span>{inst.region}</span>
                    {inst.gpu && <span>{inst.gpu}</span>}
                    {inst.spot_instance && <span className="text-success">spot</span>}
                  </div>
                </div>
              );
            })}
          </Card>

          {/* Budget gauge */}
          <Card size="sm" className="p-4">
            <div className="text-sm font-semibold text-foreground font-display mb-3">Monthly Budget</div>
            {CLOUD_ACCOUNTS.filter(a => a.status === "connected").map(a => (
              <div key={a.provider} className="mb-3">
                <div className="flex justify-between text-[11px] mb-1">
                  <span className="flex items-center gap-1">
                    <span className="font-semibold" style={{ color: PROVIDER_COLORS[a.provider] }}>{a.provider.toUpperCase()}</span>
                    <span className="text-faint">{a.name}</span>
                  </span>
                  <span className="text-muted-foreground font-medium">${a.monthlySpend.toLocaleString()} / ${a.monthlyBudget.toLocaleString()}</span>
                </div>
                <ProgressBar value={(a.monthlySpend / a.monthlyBudget) * 100} color={PROVIDER_COLORS[a.provider]} />
              </div>
            ))}
          </Card>

          {/* Queue summary */}
          <Card size="sm" className="p-4">
            <div className="text-sm font-semibold text-foreground font-display mb-2.5">Queue Summary</div>
            <div className="grid grid-cols-2 gap-2">
              {[
                { label: "Running", value: running, color: "#22C55E" },
                { label: "Queued", value: queued, color: "#6B7280" },
                { label: "Tasks", value: workloads.filter(w => w.type === "task").length, color: "#06B6D4" },
                { label: "Serving", value: workloads.filter(w => w.type === "serving").length, color: "#22C55E" },
              ].map(s => (
                <div key={s.label} className="flex justify-between py-1 text-[11px]">
                  <span className="text-faint">{s.label}</span>
                  <span className="font-semibold" style={{ color: s.color }}>{s.value}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
