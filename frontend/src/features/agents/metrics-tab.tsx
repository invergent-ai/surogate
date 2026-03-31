import { Sparkline } from "@/components/ui/sparkline";
import type { Agent } from "./agents-data";

export function MetricsTab({ agent }: { agent: Agent }) {
  const charts = [
    {
      label: "Requests per Second",
      data: agent.metricsHistory.rps,
      color: "#22C55E",
      unit: "rps",
    },
    {
      label: "P99 Latency",
      data: agent.metricsHistory.latency,
      color: "#3B82F6",
      unit: "ms",
    },
    {
      label: "Error Rate",
      data: agent.metricsHistory.errors,
      color: "#EF4444",
      unit: "%",
    },
    {
      label: "Token Throughput (K)",
      data: agent.metricsHistory.tokens,
      color: "#F59E0B",
      unit: "K tok",
    },
  ];

  return (
    <div className="animate-in fade-in duration-150 grid grid-cols-2 gap-4">
      {charts.map((chart) => (
        <section
          key={chart.label}
          className="bg-muted/40 border border-border rounded-lg overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <span className="text-xs font-semibold text-foreground font-display">
              {chart.label}
            </span>
            <span className="text-[11px] font-semibold" style={{ color: chart.color }}>
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
  );
}