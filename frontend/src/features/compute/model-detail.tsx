// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useAppStore, type WorkloadMetrics } from "@/stores/app-store";
import { getModelLogs, getModelEvents } from "@/api/models";
import type { ModelEvent } from "@/api/models";
import { STATUS_COLORS } from "./compute-data";
import { X, Terminal, Clock, User, Folder, Cpu, MapPin, ChevronRight, Globe, Layers, Gauge, Server, CalendarClock, Activity } from "lucide-react";
import { statusForDot, InfoRow, EXTENDED_WORKLOAD_COLORS } from "./detail-shared";
import type { ExtendedWorkload } from "./detail-shared";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { AreaChart, Area, XAxis, YAxis } from "recharts";

export function ModelDetail({ item, onClose }: { item: ExtendedWorkload; onClose: () => void }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [events, setEvents] = useState<ModelEvent[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [rightTab, setRightTab] = useState<"logs" | "events" | "metrics">("logs");
  const stopModel = useAppStore((s) => s.stopModel);
  const startModel = useAppStore((s) => s.startModel);
  const restartModel = useAppStore((s) => s.restartModel);
  const dot = statusForDot(item.status);
  const isActive = dot === "running" || dot === "deploying";
  const model = item._model;

  useEffect(() => {
    let cancelled = false;
    const fetchLogs = async () => {
      setLogsLoading(true);
      try {
        const res = await getModelLogs(item.id, { tail: 200 });
        if (!cancelled) setLogs(res.lines);
      } catch {
        if (!cancelled) setLogs(["(could not fetch logs)"]);
      } finally {
        if (!cancelled) setLogsLoading(false);
      }
    };
    void fetchLogs();
    const interval = isActive ? setInterval(() => void fetchLogs(), 3000) : undefined;
    return () => { cancelled = true; clearInterval(interval); };
  }, [item.id, isActive]);

  useEffect(() => {
    let cancelled = false;
    const fetchEvents = async () => {
      setEventsLoading(true);
      try {
        const res = await getModelEvents(item.id, { limit: 200 });
        if (!cancelled) setEvents(res.events);
      } catch {
        if (!cancelled) setEvents([]);
      } finally {
        if (!cancelled) setEventsLoading(false);
      }
    };
    void fetchEvents();
    return () => { cancelled = true; };
  }, [item.id]);

  // ── Live metrics from store ──
  const liveMetrics = useAppStore((s) => s.workloadMetrics[item.id]);
  const snapshots = useAppStore((s) => s.workloadMetricsHistory[item.id]);
  const metricsHistory = useMemo(
    () => (snapshots ?? []).map((s) => buildMetricsPoint(s.time, s.metrics)),
    [snapshots],
  );

  const color = EXTENDED_WORKLOAD_COLORS[item.type] ?? "#6B7280";

  return (
    <td colSpan={9} className="p-0">
      <div className="bg-muted/30 border-t border-line animate-in fade-in duration-150">
        {/* Top bar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-line">
          <div className="flex items-center gap-2.5">
            <StatusDot status={statusForDot(item.status)} />
            <span className="text-sm font-bold text-foreground font-display">{item.name}</span>
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-medium uppercase border"
              style={{ background: color + "12", color, borderColor: color + "20" }}
            >
              {item.type}
            </span>
            <span className="text-[11px] font-medium" style={{ color: STATUS_COLORS[item.status] ?? "var(--muted-foreground)" }}>
              {item.status}
            </span>
            {item.method !== "\u2014" && <span className="text-[11px] text-faint">{"\u00b7"} {item.method}</span>}
          </div>
          <div className="flex items-center gap-2">
            {model && (dot === "running" || dot === "deploying") && (
              <Button
                variant="outline"
                size="xs"
                className="text-destructive border-destructive/30 hover:bg-destructive/10"
                onClick={(e) => { e.stopPropagation(); void stopModel(model.id); }}
              >
                Stop
              </Button>
            )}
            {model && dot === "stopped" && (
              <Button
                variant="outline"
                size="xs"
                onClick={(e) => { e.stopPropagation(); void startModel(model.id); }}
              >
                Start
              </Button>
            )}
            {model && dot === "error" && (
              <Button
                variant="outline"
                size="xs"
                onClick={(e) => { e.stopPropagation(); void restartModel(model.id); }}
              >
                Restart
              </Button>
            )}
            <Button variant="ghost" size="icon-xs" onClick={(e) => { e.stopPropagation(); onClose(); }}>
              <X size={14} />
            </Button>
          </div>
        </div>

        <div className="flex">
          {/* Left: info */}
          <div className="w-80 shrink-0 px-4 py-3 border-r border-line">
            <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
              <InfoRow icon={Folder} label="Project" value={item.project} />
              <InfoRow icon={User} label="Deployed by" value={item.requestedBy} />
              <InfoRow icon={MapPin} label="Location" value={item.location === "local" ? "Local Cluster" : `${item.location.charAt(0).toUpperCase() + item.location.slice(1)} Cloud`} />
              <InfoRow icon={Cpu} label="GPU" value={item.gpu} />
              {item.node !== "\u2014" && <InfoRow icon={ChevronRight} label="Node" value={item.node} />}
              {item.startedAt && <InfoRow icon={Clock} label="Started" value={item.startedAt} />}
              {model && <InfoRow icon={Server} label="Engine" value={model.engine} />}
              {model && <InfoRow icon={Layers} label="Replicas" value={`${model.replicas.current}/${model.replicas.desired}`} />}
              {model?.endpoint && model.endpoint !== "\u2014" && <InfoRow icon={Globe} label="Endpoint" value={model.endpoint} />}
              {model?.uptime && model.uptime !== "\u2014" && <InfoRow icon={Clock} label="Uptime" value={model.uptime} />}
              {model && model.tps > 0 && <InfoRow icon={Gauge} label="TPS" value={String(model.tps)} />}
            </div>
          </div>

          {/* Right: logs / events */}
          <div className="flex-1 min-w-0 flex flex-col">
            <div className="flex items-center gap-3 px-3 py-1.5 border-b border-line shrink-0">
              <button
                type="button"
                onClick={() => setRightTab("logs")}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: rightTab === "logs" ? 1 : 0.45 }}
              >
                <Terminal size={11} className="text-faint" />
                <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Logs</span>
                {isActive && rightTab === "logs" && <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />}
              </button>
              <button
                type="button"
                onClick={() => setRightTab("metrics")}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: rightTab === "metrics" ? 1 : 0.45 }}
              >
                <Activity size={11} className="text-faint" />
                <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Metrics</span>
                {liveMetrics && <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />}
              </button>
              <button
                type="button"
                onClick={() => setRightTab("events")}
                className="flex items-center gap-1.5 cursor-pointer"
                style={{ opacity: rightTab === "events" ? 1 : 0.45 }}
              >
                <CalendarClock size={11} className="text-faint" />
                <span className="text-[10px] font-semibold text-muted-foreground font-display uppercase tracking-wider">Events</span>
                {events.length > 0 && (
                  <span className="text-[9px] px-1 rounded bg-muted text-faint">{events.length}</span>
                )}
              </button>
            </div>

            {rightTab === "logs" ? (
              <ScrollArea className="h-70 bg-background">
                <div className="p-3">
                  {logsLoading && logs.length === 0 ? (
                    <div className="text-[11px] text-faint animate-pulse">Loading logs...</div>
                  ) : logs.length === 0 ? (
                    <div className="text-[11px] text-faint">No log output yet</div>
                  ) : (
                    <pre className="text-[11px] leading-[1.6] text-muted-foreground font-mono whitespace-pre-wrap break-all">
                      {logs.map((line, i) => (
                        <div
                          key={i}
                          className={
                            line.startsWith("ERROR:") ? "text-destructive font-semibold" :
                            line.startsWith("Warning:") ? "text-[#F59E0B]" :
                            ""
                          }
                        >
                          {line}
                        </div>
                      ))}
                    </pre>
                  )}
                </div>
              </ScrollArea>
            ) : rightTab === "metrics" ? (
              <ScrollArea className="h-70 bg-background">
                <MetricsCharts history={metricsHistory} latest={liveMetrics} />
              </ScrollArea>
            ) : (
              <ScrollArea className="h-70 bg-background">
                <div className="p-3">
                  {eventsLoading && events.length === 0 ? (
                    <div className="text-[11px] text-faint animate-pulse">Loading events...</div>
                  ) : events.length === 0 ? (
                    <div className="text-[11px] text-faint">No events recorded</div>
                  ) : (
                    <div className="space-y-1.5">
                      {events.map((ev, i) => (
                        <div key={i} className="flex items-start gap-2 text-[11px]">
                          <span className="shrink-0 text-faint tabular-nums">
                            {new Date(ev.time).toLocaleString(undefined, {
                              month: "short", day: "numeric",
                              hour: "2-digit", minute: "2-digit", second: "2-digit",
                            })}
                          </span>
                          <span
                            className="shrink-0 text-[9px] px-1 py-0.5 rounded font-medium uppercase border"
                            style={{
                              color: ev.type === "job" ? "#3B82F6" : ev.type === "run" ? "#8B5CF6" : "var(--faint)",
                              background: ev.type === "job" ? "#3B82F610" : ev.type === "run" ? "#8B5CF610" : "var(--muted)",
                              borderColor: ev.type === "job" ? "#3B82F620" : ev.type === "run" ? "#8B5CF620" : "var(--border)",
                            }}
                          >
                            {ev.type}
                          </span>
                          <span className="text-muted-foreground">{ev.text}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </ScrollArea>
            )}
          </div>
        </div>
      </div>
    </td>
  );
}


// ── Metrics helpers ─────────────────────────────────────────────

interface MetricsPoint {
  time: number;
  timeLabel: string;
  cpuPercent: number;
  memUsedGiB: number;
  memTotalGiB: number;
  gpus: { util: number; memUsedGiB: number; memTotalGiB: number }[];
}

function buildMetricsPoint(ts: number, m: WorkloadMetrics): MetricsPoint {
  const gpuCount = m.gpus_detected_num ?? 0;
  const gpuMemTotal = (m.gpu_memory_total_bytes ?? 0) / (1024 ** 3);
  const gpus: MetricsPoint["gpus"] = [];
  for (let i = 0; i < gpuCount; i++) {
    gpus.push({
      util: (m[`gpu_util_percent_gpu${i}`] as number) ?? 0,
      memUsedGiB: ((m[`gpu_memory_usage_bytes_gpu${i}`] as number) ?? 0) / (1024 ** 3),
      memTotalGiB: gpuMemTotal,
    });
  }
  return {
    time: ts,
    timeLabel: new Date(ts).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    cpuPercent: m.cpu_usage_percent ?? 0,
    memUsedGiB: (m.memory_usage_bytes ?? 0) / (1024 ** 3),
    memTotalGiB: (m.memory_total_bytes ?? 0) / (1024 ** 3),
    gpus,
  };
}

const GPU_COLORS = ["#22C55E", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"];

function MetricsCharts({ history, latest }: { history: MetricsPoint[]; latest?: WorkloadMetrics }) {
  if (!latest) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="text-[11px] text-faint">No metrics available — metrics appear once the workload is running</div>
      </div>
    );
  }

  if (history.length < 2) {
    return (
      <div className="flex items-center justify-center h-full p-6">
        <div className="text-[11px] text-faint animate-pulse">Collecting metrics...</div>
      </div>
    );
  }

  const gpuCount = history[0].gpus.length;
  const hasGpu = gpuCount > 0;

  // Build recharts data
  const chartData = history.map((p) => {
    const row: Record<string, number | string> = {
      timeLabel: p.timeLabel,
      cpu: p.cpuPercent,
      mem: p.memUsedGiB,
    };
    for (let i = 0; i < gpuCount; i++) {
      row[`gpuUtil${i}`] = p.gpus[i].util;
      row[`gpuMem${i}`] = p.gpus[i].memUsedGiB;
    }
    return row;
  });

  // Chart configs
  const gpuUtilConfig: ChartConfig = {};
  const gpuMemConfig: ChartConfig = {};
  for (let i = 0; i < gpuCount; i++) {
    gpuUtilConfig[`gpuUtil${i}`] = { label: `GPU ${i}`, color: GPU_COLORS[i % GPU_COLORS.length] };
    gpuMemConfig[`gpuMem${i}`] = { label: `GPU ${i}`, color: GPU_COLORS[i % GPU_COLORS.length] };
  }

  const cpuConfig: ChartConfig = { cpu: { label: "CPU", color: "#3B82F6" } };
  const memConfig: ChartConfig = { mem: { label: "Memory", color: "#8B5CF6" } };

  const lastPoint = history[history.length - 1];
  const memTotalGiB = lastPoint.memTotalGiB;
  const gpuMemTotalGiB = hasGpu ? lastPoint.gpus[0].memTotalGiB : 0;

  return (
    <div className="p-3 grid grid-cols-1 gap-3">
      {hasGpu && (
        <>
          <MiniChart
            title="GPU Utilization"
            unit="%"
            data={chartData}
            config={gpuUtilConfig}
            dataKeys={Array.from({ length: gpuCount }, (_, i) => `gpuUtil${i}`)}
            domainMax={100}
            formatValue={(v) => `${Math.round(v as number)}%`}
          />
          <MiniChart
            title="GPU Memory"
            unit="GiB"
            data={chartData}
            config={gpuMemConfig}
            dataKeys={Array.from({ length: gpuCount }, (_, i) => `gpuMem${i}`)}
            domainMax={gpuMemTotalGiB > 0 ? gpuMemTotalGiB : undefined}
            formatValue={(v) => `${(v as number).toFixed(1)} GiB`}
          />
        </>
      )}
      <MiniChart
        title="CPU Usage"
        unit="%"
        data={chartData}
        config={cpuConfig}
        dataKeys={["cpu"]}
        domainMax={100}
        formatValue={(v) => `${Math.round(v as number)}%`}
      />
      <MiniChart
        title="System Memory"
        unit="GiB"
        data={chartData}
        config={memConfig}
        dataKeys={["mem"]}
        domainMax={memTotalGiB > 0 ? memTotalGiB : undefined}
        formatValue={(v) => `${(v as number).toFixed(1)} GiB`}
      />
    </div>
  );
}

function MiniChart({
  title, unit, data, config, dataKeys, domainMax, formatValue,
}: {
  title: string;
  unit: string;
  data: Record<string, number | string>[];
  config: ChartConfig;
  dataKeys: string[];
  domainMax?: number;
  formatValue: (v: number | string) => string;
}) {
  const lastRow = data[data.length - 1];
  const currentValues = dataKeys.map((k) => lastRow[k] as number);
  const currentLabel = currentValues.map(formatValue).join(" / ");

  return (
    <div className="rounded-lg border border-line bg-muted/20 p-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">{title}</span>
        <span className="text-[10px] font-mono text-foreground tabular-nums">{currentLabel}</span>
      </div>
      <ChartContainer config={config} className="h-20 w-full aspect-auto" initialDimension={{ width: 300, height: 80 }}>
        <AreaChart data={data} margin={{ top: 2, right: 2, bottom: 0, left: 0 }}>
          <XAxis dataKey="timeLabel" hide />
          <YAxis
            domain={[0, domainMax ?? "auto"]}
            hide
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                hideLabel={false}
                labelKey="timeLabel"
                formatter={(value, name) => (
                  <span className="font-mono tabular-nums">{formatValue(value as number)} <span className="text-faint">{unit}</span></span>
                )}
              />
            }
          />
          {dataKeys.map((key) => (
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stroke={`var(--color-${key})`}
              fill={`var(--color-${key})`}
              fillOpacity={0.1}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ChartContainer>
    </div>
  );
}
