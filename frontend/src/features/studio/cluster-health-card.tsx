// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useEffect } from "react";
import { Card } from "@/components/ui/card";
import { ProgressBar } from "@/components/ui/progress-bar";
import { useAppStore } from "@/stores/app-store";

export function ClusterHealthCard() {
  const k8sNodes = useAppStore((s) => s.k8sNodes);
  const fetchK8Nodes = useAppStore((s) => s.fetchK8Nodes);

  useEffect(() => {
    void fetchK8Nodes();
  }, [fetchK8Nodes]);

  const totalGpu = k8sNodes.reduce((s, n) => s + n.accelerator_count, 0);
  const freeGpu = k8sNodes.reduce((s, n) => s + n.accelerator_available, 0);
  const usedGpu = totalGpu - freeGpu;
  const gpuUtil = totalGpu > 0 ? Math.round((usedGpu / totalGpu) * 100) : 0;

  const cpuValues = k8sNodes
    .map((n) => n.metrics?.cpu_utilization_percent)
    .filter((v): v is number => v != null);
  const avgCpu = cpuValues.length > 0 ? Math.round(cpuValues.reduce((s, v) => s + v, 0) / cpuValues.length) : 0;

  const totalMemGb = k8sNodes.reduce((s, n) => s + (n.memory_gb ?? 0), 0);
  const freeMemGb = k8sNodes.reduce((s, n) => {
    const freeBytes = n.metrics?.free_memory_bytes;
    return s + (freeBytes ? freeBytes / (1024 ** 3) : 0);
  }, 0);
  const usedMemGb = Math.round(totalMemGb - freeMemGb);

  const readyNodes = k8sNodes.filter((n) => n.is_ready).length;
  const totalNodes = k8sNodes.length;

  const metrics = [
    { label: "GPU Utilization", value: `${gpuUtil}%`, max: 100, current: gpuUtil, color: "#F59E0B" },
    { label: "CPU Utilization", value: `${avgCpu}%`, max: 100, current: avgCpu, color: "#3B82F6" },
    { label: "Memory", value: `${usedMemGb} / ${Math.round(totalMemGb)} Gi`, max: totalMemGb || 1, current: usedMemGb, color: "#8B5CF6" },
    { label: "Nodes Ready", value: `${readyNodes} / ${totalNodes}`, max: totalNodes || 1, current: readyNodes, color: "#22C55E" },
  ];

  const gpuNodes = k8sNodes.filter((n) => n.accelerator_count > 0).length;
  const totalCpuCores = k8sNodes.reduce((s, n) => s + (n.cpu_count ?? 0), 0);

  const stats = [
    { label: "GPU Nodes", value: String(gpuNodes) },
    { label: "Total GPUs", value: String(totalGpu) },
    { label: "CPU Cores", value: String(totalCpuCores) },
    { label: "Nodes", value: String(totalNodes) },
  ];

  return (
    <Card>
      <div className="flex items-center gap-2 px-4 py-2.5 border-b border-line">
        <span style={{ color: "#22C55E" }}>{"\u229e"}</span>
        <span className="text-sm font-semibold text-foreground font-display">Cluster Health</span>
      </div>
      <div className="px-4 py-3">
        {metrics.map((r, i) => (
          <div key={r.label} className={i < metrics.length - 1 ? "mb-3.5" : ""}>
            <div className="flex justify-between mb-1.5 font-display">
              <span className="text-subtle">{r.label}</span>
              <span className="text-foreground/80 font-medium">{r.value}</span>
            </div>
            <ProgressBar value={r.current} max={r.max} color={r.color} />
          </div>
        ))}
      </div>
      <div className="px-4 py-2 pb-3 grid grid-cols-2 gap-2">
        {stats.map((s) => (
          <div key={s.label} className="flex justify-between">
            <span className="text-faint font-display">{s.label}</span>
            <span className="text-subtle font-medium">{s.value}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
