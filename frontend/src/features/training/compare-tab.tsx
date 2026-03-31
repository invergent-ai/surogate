// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { ChartSVG } from "./chart-card";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function CompareTab({ compareRuns }: { compareRuns: any[] }) {
  if (compareRuns.length === 0) {
    return (
      <div className="py-10 text-center animate-in fade-in duration-200">
        <div className="text-muted-foreground/30 text-xs mb-2">
          Select runs to compare using the checkboxes in the left panel
        </div>
        <div className="text-[10px] text-muted-foreground/50">
          You can compare up to 3 runs simultaneously
        </div>
      </div>
    );
  }

  // Collect all eval keys from compared runs
  const evalKeys = Object.keys(
    compareRuns.reduce((acc, r) => ({ ...acc, ...r.evalResults }), {}),
  );

  const rows: { label: string; get: (r: any) => string | number; highlight?: boolean }[] = [
    { label: "Method", get: r => r.method },
    { label: "Status", get: r => r.status },
    { label: "Dataset", get: r => `${r.dataset} (${r.datasetSamples})` },
    { label: "Learning Rate", get: r => r.lr },
    { label: "Batch Size", get: r => r.batchSize },
    { label: "Epochs", get: r => `${r.epochs.current}/${r.epochs.total}` },
    { label: "Best Loss", get: r => r.bestLoss !== null ? r.bestLoss.toFixed(3) : "\u2014", highlight: true },
    { label: "GPU", get: r => r.gpu },
    { label: "Duration", get: r => r.duration || "in progress" },
    ...evalKeys.map(k => ({
      label: `Eval: ${k}`,
      get: (r: any) => r.evalResults[k] ?? "\u2014",
      highlight: true,
    })),
  ];

  return (
    <div className="animate-in fade-in duration-200">
      {/* Overlaid loss curves */}
      <section className="bg-card rounded-lg ring-1 ring-foreground/10 overflow-hidden mb-4">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Loss Comparison
          </span>
          <div className="flex gap-3">
            {compareRuns.map(r => (
              <span key={r.id} className="flex items-center gap-1 text-[9px]">
                <span className="w-2.5 h-0.5 rounded-full" style={{ background: r.color }} />
                <span className="text-muted-foreground/60">{r.name.substring(0, 18)}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="px-4 py-4">
          <ChartSVG
            datasets={compareRuns.map(r => ({ data: r.lossCurve, color: r.color }))}
            h={140}
          />
        </div>
      </section>

      {/* Comparison table */}
      <section className="bg-card rounded-lg ring-1 ring-foreground/10 overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <span className="text-[13px] font-semibold text-foreground font-display">
            Run Comparison
          </span>
        </div>
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-2 text-left text-[9px] font-medium text-muted-foreground/40 uppercase font-display">
                Metric
              </th>
              {compareRuns.map(r => (
                <th
                  key={r.id}
                  className="px-4 py-2 text-right text-[9px] font-medium font-display"
                  style={{ color: r.color }}
                >
                  {r.name.substring(0, 18)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const values = compareRuns.map(r => row.get(r));
              const numVals = values.map(v => parseFloat(String(v))).filter(v => !isNaN(v));
              const bestVal = row.highlight && numVals.length > 1
                ? (row.label.includes("Loss") ? Math.min(...numVals) : Math.max(...numVals))
                : null;

              return (
                <tr key={i} className="border-b border-border/30">
                  <td className="px-4 py-2 text-[11px] text-muted-foreground">{row.label}</td>
                  {compareRuns.map(r => {
                    const v = row.get(r);
                    const isBest = bestVal !== null && parseFloat(String(v)) === bestVal;
                    return (
                      <td
                        key={r.id}
                        className={`px-4 py-2 text-xs text-right ${isBest ? "text-green-500 font-bold" : "text-foreground/70"}`}
                      >
                        {String(v)} {isBest && <span className="text-[9px]">{"\u2713"}</span>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}
