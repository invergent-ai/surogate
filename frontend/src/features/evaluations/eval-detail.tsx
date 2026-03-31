// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Button } from "@/components/ui/button";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { ResultsTab } from "./results-tab";
import { ComparisonTab } from "./comparison-tab";
import { SamplesTab } from "./samples-tab";
import { DetailsTab } from "./details-tab";
import type { EvalRun } from "./evaluations-data";

function toStatusDot(status: EvalRun["status"]) {
  switch (status) {
    case "completed":
      return "completed" as const;
    case "running":
      return "running" as const;
    case "failed":
      return "error" as const;
    default:
      return "stopped" as const;
  }
}

export function EvalDetail({ run }: { run: EvalRun }) {
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <StatusDot status={toStatusDot(run.status)} />
              <h2 className="text-[15px] font-bold text-foreground font-display tracking-tight">
                {run.name}
              </h2>
            </div>
            <div className="flex gap-3 text-[10px] text-muted-foreground/40 flex-wrap">
              <span>
                model:{" "}
                <span className="text-muted-foreground">
                  {run.modelLabel}
                </span>
              </span>
              {run.compareLabel && (
                <span>
                  vs:{" "}
                  <span className="text-muted-foreground">
                    {run.compareLabel}
                  </span>
                </span>
              )}
              <span>
                by:{" "}
                <span className="text-muted-foreground">{run.runner}</span>
              </span>
              <span>
                compute:{" "}
                <span className="text-muted-foreground">
                  {run.compute === "aws" ? "\u2601 AWS" : "\u229E Local"}{" "}
                  &middot; {run.gpu}
                </span>
              </span>
              {run.duration && (
                <span>
                  duration:{" "}
                  <span className="text-muted-foreground">
                    {run.duration}
                  </span>
                </span>
              )}
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {run.status === "running" && (
              <Button variant="destructive" size="xs">
                Stop
              </Button>
            )}
            <Button variant="outline" size="xs">
              Re-run
            </Button>
            <Button variant="outline" size="xs">
              Export
            </Button>
          </div>
        </div>
        {run.status === "running" && run.progress != null && (
          <div className="mb-2">
            <ProgressBar value={run.progress} color="#8B5CF6" animated />
            <div className="text-[9px] text-violet-500 mt-1">
              {run.progress}% &mdash; running {run.currentBenchmark}
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <Tabs
        defaultValue="results"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
            <TabsTrigger value="samples">Samples</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <TabsContent value="results" className="mt-0">
            <ResultsTab run={run} />
          </TabsContent>
          <TabsContent value="comparison" className="mt-0">
            <ComparisonTab run={run} />
          </TabsContent>
          <TabsContent value="samples" className="mt-0">
            <SamplesTab run={run} />
          </TabsContent>
          <TabsContent value="details" className="mt-0">
            <DetailsTab run={run} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────

export function EvalEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x25C8;</div>
        <div className="font-display text-sm">
          Select an evaluation to view results
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Evaluations run benchmarks and custom evals against your models to
          track quality.
        </div>
      </div>
    </div>
  );
}
