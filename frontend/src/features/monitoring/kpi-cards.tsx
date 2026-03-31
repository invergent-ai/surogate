// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Sparkline } from "@/components/ui/sparkline";
import { AGENTS_HEALTH, COMBINED_RPS } from "./monitoring-data";

interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
  color: string;
  spark: number[];
}

function KpiCard({ label, value, unit, color, spark }: KpiCardProps) {
  const allZero = spark.every((v) => v === 0);

  return (
    <div className="flex items-end justify-between rounded-lg border border-border bg-card px-3.5 py-3">
      <div>
        <div className="text-[9px] uppercase tracking-wider text-muted-foreground font-display mb-1">
          {label}
        </div>
        <div className="text-xl font-bold text-foreground tracking-tight">
          {value}
          {unit && <span className="text-[9px] font-normal text-muted-foreground ml-1">{unit}</span>}
        </div>
      </div>
      {!allZero && <Sparkline data={spark} color={color} height={26} width={64} />}
    </div>
  );
}

export function KpiCards() {
  const totalRps = AGENTS_HEALTH.reduce((s, a) => s + a.rps, 0);
  const serving = AGENTS_HEALTH.filter((a) => a.p99 > 0);
  const avgP99 = Math.round(serving.reduce((s, a) => s + a.p99, 0) / serving.length);
  const avgError = (serving.reduce((s, a) => s + a.errorRate, 0) / serving.length).toFixed(1);

  const kpis: KpiCardProps[] = [
    { label: "Total RPS", value: totalRps, unit: "req/s", color: "#14B8A6", spark: COMBINED_RPS.map((d) => d.cx + d.code + d.da + d.ob) },
    { label: "Avg P99 Latency", value: `${avgP99}ms`, color: "#3B82F6", spark: AGENTS_HEALTH[0].sparkLatency },
    { label: "Error Rate", value: `${avgError}%`, color: parseFloat(avgError) > 1 ? "#EF4444" : "#22C55E", spark: AGENTS_HEALTH[0].sparkErrors },
    { label: "Tokens (24h)", value: "14.0M", color: "#F59E0B", spark: [800, 850, 920, 980, 1050, 1100, 1080, 1200, 1180, 1220, 1200, 1250, 1280, 1320, 1300, 1350, 1380, 1400, 1380, 1400] },
    { label: "Active Convos", value: "142", unit: "now", color: "#8B5CF6", spark: [80, 90, 95, 110, 100, 115, 120, 125, 130, 128, 135, 140, 138, 142, 140, 136, 140, 142, 138, 142] },
  ];

  return (
    <div className="grid grid-cols-5 gap-2.5 mb-4">
      {kpis.map((kpi) => (
        <KpiCard key={kpi.label} {...kpi} />
      ))}
    </div>
  );
}
