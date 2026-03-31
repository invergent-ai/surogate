import { Sparkline } from "@/components/ui/sparkline";
import { cn } from "@/utils/cn";
import { ProgressBar } from "@/components/ui/progress-bar";

import { SKILL_TYPE_STYLES } from "./agents-data";
import type { Agent } from "./agents-data";

export function OverviewTab({ agent }: { agent: Agent }) {
  const cpuPct = parseInt(agent.cpu) || 0;
  const memPct =
    agent.mem !== "0Gi"
      ? (parseFloat(agent.mem) / parseFloat(agent.memLimit)) * 100
      : 0;

  return (
    <div className="animate-in fade-in duration-150">
      {/* Key metrics */}
      <div className="grid grid-cols-4 gap-2.5 mb-5">
        {[
          {
            label: "Requests/sec",
            value: agent.rps,
            sub: "current",
            spark: agent.metricsHistory.rps,
            color: "#22C55E",
          },
          {
            label: "P99 Latency",
            value: agent.p99,
            sub: `p50: ${agent.p50} \u00B7 p95: ${agent.p95}`,
            spark: agent.metricsHistory.latency,
            color: "#3B82F6",
          },
          {
            label: "Error Rate",
            value: agent.errorRate,
            sub: "last 24h",
            spark: agent.metricsHistory.errors,
            color: "#EF4444",
          },
          {
            label: "Tokens (24h)",
            value: agent.tokensIn24h,
            sub: `out: ${agent.tokensOut24h}`,
            spark: agent.metricsHistory.tokens,
            color: "#F59E0B",
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
                {m.value}
              </div>
              <div className="text-[9px] text-muted-foreground/40 mt-0.5">
                {m.sub}
              </div>
            </div>
            <Sparkline data={m.spark} color={m.color} height={28} width={70} />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-[1fr_280px] gap-4">
        {/* Left column */}
        <div className="flex flex-col gap-4">
          {/* Deployment info */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Deployment
            </div>
            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  label: "Replicas",
                  value: `${agent.replicas.current} / ${agent.replicas.desired}`,
                },
                { label: "CPU", value: agent.cpu },
                {
                  label: "Memory",
                  value: `${agent.mem} / ${agent.memLimit}`,
                },
                { label: "Namespace", value: agent.namespace },
                { label: "Model", value: agent.model },
                { label: "Base", value: agent.modelBase },
                {
                  label: "Image",
                  value: agent.image.split("/").pop() ?? agent.image,
                },
                {
                  label: "Endpoint",
                  value:
                    agent.endpoint !== "\u2014"
                      ? agent.endpoint.replace("https://", "")
                      : "\u2014",
                },
                { label: "Created", value: agent.createdAt },
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

          {/* Conversation stats */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3">
              Conversations (24h)
            </div>
            <div className="grid grid-cols-4 gap-3">
              {[
                {
                  label: "Total",
                  value: agent.conversations24h,
                  color: "text-foreground",
                },
                {
                  label: "Avg Turns",
                  value: agent.avgTurns,
                  color: "text-blue-500",
                },
                {
                  label: "Satisfaction",
                  value: agent.satisfaction,
                  color: "text-green-500",
                },
                {
                  label: "Tokens In",
                  value: agent.tokensIn24h,
                  color: "text-amber-500",
                },
              ].map((s) => (
                <div key={s.label} className="text-center">
                  <div
                    className={cn(
                      "text-xl font-bold tracking-tight",
                      s.color,
                    )}
                  >
                    {s.value}
                  </div>
                  <div className="text-[9px] text-muted-foreground/40 mt-0.5 font-display">
                    {s.label}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          {/* Resources */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-3.5">
              Resources
            </div>
            <div className="space-y-2.5">
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-muted-foreground">CPU</span>
                  <span className="text-foreground/70 font-medium">
                    {agent.cpu}
                  </span>
                </div>
                <ProgressBar value={cpuPct} max={100} color="#3B82F6" />
              </div>
              <div>
                <div className="flex justify-between text-[10px] mb-1">
                  <span className="text-muted-foreground">
                    Memory &mdash; {agent.mem}
                  </span>
                  <span className="text-foreground/70 font-medium">
                    {Math.round(memPct)}%
                  </span>
                </div>
                <ProgressBar value={memPct} max={100} color="#8B5CF6" />
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-border">
              <div className="grid grid-cols-2 gap-1.5">
                {[
                  { label: "CPU Req", value: agent.resources.cpuReq },
                  { label: "CPU Lim", value: agent.resources.cpuLim },
                  { label: "Mem Req", value: agent.resources.memReq },
                  { label: "Mem Lim", value: agent.resources.memLim },
                ].map((r) => (
                  <div
                    key={r.label}
                    className="flex justify-between text-[9px]"
                  >
                    <span className="text-muted-foreground/40">{r.label}</span>
                    <code className="text-muted-foreground">{r.value}</code>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Connected summary */}
          <section className="bg-muted/40 border border-border rounded-lg p-4">
            <div className="text-xs font-semibold text-foreground font-display mb-2.5">
              Connected
            </div>
            <div className="text-[10px] text-muted-foreground mb-1.5">
              {agent.skills.length} skills &middot; {agent.mcpServers.length}{" "}
              MCP
            </div>
            <div className="flex flex-wrap gap-1">
              {agent.skills.slice(0, 4).map((s) => {
                const style = SKILL_TYPE_STYLES[s.type];
                return (
                  <span
                    key={s.name}
                    className={cn(
                      "text-[9px] px-1.5 py-0.5 rounded",
                      style?.bg,
                      style?.fg,
                    )}
                  >
                    {s.name}
                  </span>
                );
              })}
              {agent.skills.length > 4 && (
                <span className="text-[9px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                  +{agent.skills.length - 4}
                </span>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}