// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect } from "react";
import { Card } from "@/components/ui/card";
import { StatusDot } from "@/components/ui/status-dot";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useAppStore } from "@/stores/app-store";
import { STATUS_COLORS } from "./compute-data";

function getGpuTotal(node: { total?: Record<string, number> }): number {
  return node.total?.["accelerator_count"] ?? 0;
}

function getGpuFree(node: { free?: Record<string, number> }): number {
  return node.free?.["accelerators_available"] ?? 0;
}

export function ClusterNodesTab() {
  const k8sNodes = useAppStore((s) => s.k8sNodes);
  const fetchK8Nodes = useAppStore((s) => s.fetchK8Nodes);

  useEffect(() => {
    fetchK8Nodes();
  }, [fetchK8Nodes]);

  return (
    <div className="animate-in fade-in duration-200">
      <Card size="sm" className="overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-line">
              {["Node", "Status", "GPU", "GPU Used", "CPU", "Memory"].map(h => (
                <th key={h} className="px-3.5 py-2 text-left text-[10px] font-medium text-faint uppercase tracking-wider font-display">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {k8sNodes.map(n => {
              const gpuTotal = getGpuTotal(n);
              const gpuFree = getGpuFree(n);
              const gpuUsed = gpuTotal - gpuFree;
              const cpuUtil = n.metrics?.cpu_utilization_percent ?? 0;
              const memTotalGb = n.memory_gb ?? 0;
              const memFreeGb = n.metrics?.free_memory_bytes
                ? n.metrics.free_memory_bytes / (1024 ** 3)
                : 0;
              const memUsedGb = memTotalGb - memFreeGb;
              const status = n.is_ready ? "active" : "cordoned";

              return (
                <tr
                  key={n.name}
                  className="border-b border-line last:border-0 hover:bg-muted/50 transition-colors"
                  style={{ opacity: n.is_ready ? 1 : 0.5 }}
                >
                  <td className="px-3.5 py-2.5">
                    <div className="text-sm text-foreground font-medium">{n.name}</div>
                    {n.ip_address && <div className="text-[11px] text-faint">{n.ip_address}</div>}
                  </td>
                  <td className="px-3.5 py-2.5">
                    <span className="flex items-center gap-1 text-[11px]">
                      <StatusDot status={n.is_ready ? "running" : "stopped"} />
                      <span style={{ color: STATUS_COLORS[status] }}>{status}</span>
                    </span>
                  </td>
                  <td className="px-3.5 py-2.5 text-[11px] text-muted-foreground">
                    {gpuTotal > 0 ? `${gpuTotal}\u00d7 ${n.accelerator_type ?? "GPU"}` : "\u2014"}
                  </td>
                  <td className="px-3.5 py-2.5 w-24">
                    {gpuTotal > 0
                      ? <><span className="text-[11px] text-muted-foreground mr-1.5">{gpuUsed}/{gpuTotal}</span><ProgressBar value={(gpuUsed / gpuTotal) * 100} color="var(--primary)" /></>
                      : <span className="text-[11px] text-faint">{"\u2014"}</span>}
                  </td>
                  <td className="px-3.5 py-2.5">
                    <span className="text-[11px] text-muted-foreground mr-1.5">{cpuUtil.toFixed(0)}%</span>
                    <ProgressBar value={cpuUtil} color="var(--primary)" />
                  </td>
                  <td className="px-3.5 py-2.5 text-[11px] text-muted-foreground">
                    {memUsedGb.toFixed(0)}/{memTotalGb.toFixed(0)} Gi
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
