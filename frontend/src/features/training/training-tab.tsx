// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { ChartCard } from "./chart-card";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function TrainingTab({ run }: { run: any }) {
  const isRL = run.method === "GRPO" || run.method === "PPO";

  const kpis = [
    { label: "Loss", value: run.bestLoss !== null ? run.bestLoss.toFixed(3) : "\u2014", color: "#F59E0B" },
    { label: "Epoch", value: `${run.epochs.current}/${run.epochs.total}`, color: "#3B82F6" },
    { label: "Steps", value: run.steps.current.toLocaleString(), color: "#8B5CF6" },
    { label: "LR", value: run.lr.toExponential(0), color: "#22C55E" },
    { label: "GPU", value: run.gpuUtil > 0 ? `${run.gpuUtil}%` : "\u2014", color: "#06B6D4" },
    ...(isRL ? [
      { label: "Reward", value: run.rewardCurve?.length > 0 ? run.rewardCurve.at(-1).toFixed(3) : "\u2014", color: "#22C55E" },
      { label: "Episodes", value: run.episodes ? run.episodes.completed.toLocaleString() : "\u2014", color: "#3B82F6" },
    ] : []),
  ];

  return (
    <div className="animate-in fade-in duration-200">
      {/* KPI strip */}
      <div className={`grid gap-2.5 mb-5 ${isRL ? "grid-cols-7" : "grid-cols-5"}`}>
        {kpis.map((m, i) => (
          <div key={i} className="bg-card rounded-lg ring-1 ring-foreground/10 px-3 py-2.5">
            <div className="text-[8px] text-muted-foreground/60 uppercase tracking-wide mb-0.5 font-display">
              {m.label}
            </div>
            <div className={`text-lg font-bold tracking-tight ${m.value === "\u2014" ? "text-muted-foreground/30" : "text-foreground"}`}>
              {m.value}
            </div>
          </div>
        ))}
      </div>

      {/* SFT/DPO charts: loss + gradient norm */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <ChartCard
          title={isRL ? "Policy Loss" : "Training Loss"}
          value={run.bestLoss !== null ? run.bestLoss.toFixed(3) : "\u2014"}
          valueColor="#F59E0B"
          datasets={[{ data: isRL ? (run.policyLossCurve || run.lossCurve) : run.lossCurve, color: "#F59E0B" }]}
          xLabel={`Step ${run.steps.current}`}
        />
        <ChartCard
          title="Gradient Norm"
          value={run.gradNormCurve.length > 0 ? run.gradNormCurve.at(-1).toFixed(2) : "\u2014"}
          valueColor="#8B5CF6"
          datasets={[{ data: run.gradNormCurve, color: "#8B5CF6" }]}
        />
      </div>

      {/* RL-specific charts: reward + KL divergence */}
      {isRL && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <ChartCard
            title="Mean Reward"
            value={run.rewardCurve?.at(-1)?.toFixed(3) || "\u2014"}
            valueColor="#22C55E"
            datasets={[{ data: run.rewardCurve || [], color: "#22C55E" }]}
          />
          <ChartCard
            title="KL Divergence"
            value={run.klDivCurve?.at(-1)?.toFixed(4) || "\u2014"}
            valueColor="#EF4444"
            datasets={[{ data: run.klDivCurve || [], color: "#EF4444" }]}
          />
        </div>
      )}

      {/* Learning rate */}
      <ChartCard
        title={`Learning Rate (${run.scheduler})`}
        value={run.lr.toExponential(1)}
        valueColor="#22C55E"
        datasets={[{ data: run.lrCurve, color: "#22C55E" }]}
        h={60}
      />

      {/* Eval results */}
      {Object.keys(run.evalResults).length > 0 && (
        <section className="bg-card rounded-lg ring-1 ring-foreground/10 p-4 mt-4">
          <div className="text-xs font-semibold text-foreground font-display mb-3">
            Evaluation Results
          </div>
          <div className={`grid gap-3 grid-cols-${Math.min(Object.keys(run.evalResults).length, 4)}`}>
            {Object.entries(run.evalResults).map(([k, v]) => (
              <div key={k} className="text-center">
                <div className="text-xl font-bold text-foreground">{v as number}</div>
                <div className="text-[9px] text-muted-foreground/60 mt-0.5 font-display uppercase">{k}</div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
