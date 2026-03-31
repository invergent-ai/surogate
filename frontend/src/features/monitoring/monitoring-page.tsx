// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AGENTS_HEALTH, ALERTS, TIME_RANGES, type TimeRangeId } from "./monitoring-data";
import { KpiCards } from "./kpi-cards";
import { RequestRateChart } from "./request-rate-chart";
import { AgentHealthTable } from "./agent-health-table";
import { LatencyHeatmap } from "./latency-heatmap";
import { ModelServingCard } from "./model-serving-card";
import { ErrorLog } from "./error-log";
import { AlertsPanel } from "./alerts-panel";
import { TokenUsageCard } from "./token-usage-card";
import { LiveFeed } from "./live-feed";
import { EndpointsCard } from "./endpoints-card";

function HeaderStatus({ liveEnabled }: { liveEnabled: boolean }) {
  const healthyCount = AGENTS_HEALTH.filter((a) => a.status === "healthy").length;
  const degradedCount = AGENTS_HEALTH.filter((a) => a.status === "degraded").length;
  const downCount = AGENTS_HEALTH.filter((a) => a.status === "down").length;

  return (
    <span className="text-[10px] text-muted-foreground flex items-center gap-1.5">
      <span className="inline-flex items-center gap-1">
        <span
          className="w-1.25 h-1.25 rounded-full inline-block"
          style={{
            background: liveEnabled ? "#22C55E" : "var(--muted-foreground)",
            animation: liveEnabled ? "pulse 2s ease-in-out infinite" : undefined,
          }}
        />
        <span style={{ color: liveEnabled ? "#22C55E" : undefined }}>LIVE</span>
      </span>
      <span className="text-border">·</span>
      <span>{healthyCount} healthy</span>
      {degradedCount > 0 && (
        <>
          <span className="text-border">·</span>
          <span className="text-amber-500">{degradedCount} degraded</span>
        </>
      )}
      {downCount > 0 && (
        <>
          <span className="text-border">·</span>
          <span className="text-destructive">{downCount} down</span>
        </>
      )}
    </span>
  );
}

export function MonitoringPage() {
  const [timeRange, setTimeRange] = useState<TimeRangeId>("15m");
  const [liveEnabled, setLiveEnabled] = useState(true);
  const activeAlerts = ALERTS.filter((a) => !a.acknowledged).length;

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Monitoring"
        subtitle={<HeaderStatus liveEnabled={liveEnabled} />}
        action={
          <div className="flex items-center gap-2.5">
            {/* time range selector */}
            <div className="flex border border-border rounded-md overflow-hidden">
              {TIME_RANGES.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTimeRange(t.id)}
                  className="px-2.5 py-1.25 border-r border-border last:border-r-0 text-[10px] font-display transition-colors cursor-pointer"
                  style={{
                    background: timeRange === t.id ? "var(--primary-foreground)" : undefined,
                    color: timeRange === t.id ? "var(--primary)" : "var(--muted-foreground)",
                    fontWeight: timeRange === t.id ? 600 : 400,
                  }}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* auto-refresh toggle */}
            <Button
              variant="outline"
              size="xs"
              onClick={() => setLiveEnabled(!liveEnabled)}
              className="text-[10px] gap-1.5"
              style={{
                borderColor: liveEnabled ? "#14B8A633" : undefined,
                color: liveEnabled ? "#14B8A6" : undefined,
              }}
            >
              <span
                className="w-1.25 h-1.25 rounded-full"
                style={{ background: liveEnabled ? "#14B8A6" : "var(--muted-foreground)" }}
              />
              {liveEnabled ? "Auto-refresh" : "Paused"}
            </Button>

            {/* active alerts indicator */}
            {activeAlerts > 0 && (
              <Badge variant="danger" className="text-[10px] px-2.5 py-1 font-semibold flex items-center gap-1.5">
                <span className="animate-pulse">&#x25CF;</span> {activeAlerts} active alert{activeAlerts > 1 ? "s" : ""}
              </Badge>
            )}
          </div>
        }
      />

      <div className="flex-1 overflow-y-auto px-7 pt-4 pb-10">
        <KpiCards />

        <div className="grid grid-cols-[1fr_340px] gap-4">
          {/* left column */}
          <div className="flex flex-col gap-4">
            <RequestRateChart timeRange={timeRange} />
            <AgentHealthTable />
            <div className="grid grid-cols-2 gap-4">
              <LatencyHeatmap />
              <ModelServingCard />
            </div>
            <ErrorLog />
          </div>

          {/* right column */}
          <div className="flex flex-col gap-4">
            <AlertsPanel />
            <TokenUsageCard />
            <LiveFeed />
            <EndpointsCard />
          </div>
        </div>
      </div>
    </div>
  );
}
