// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ProgressBar } from "@/components/ui/progress-bar";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { cn } from "@/utils/cn";
import {
  BENCHMARKS,
  EVAL_RUNS,
  CATEGORY_COLORS,
} from "./evaluations-data";
import { EvalDetail, EvalEmptyState } from "./eval-detail";
import { BenchmarkSuite } from "./benchmark-suite";
import { Leaderboard } from "./leaderboard";
import type { EvalRun, BenchmarkCategory } from "./evaluations-data";

// ── Run list item ─────────────────────────────────────────────

function RunListItem({
  run,
  selected,
  onSelect,
}: {
  run: EvalRun;
  selected: boolean;
  onSelect: () => void;
}) {
  const statusDot =
    run.status === "completed"
      ? ("completed" as const)
      : run.status === "running"
        ? ("running" as const)
        : run.status === "failed"
          ? ("error" as const)
          : ("stopped" as const);

  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-full text-left px-3.5 py-3 border-l-2 border-b border-b-border/50 transition-colors cursor-pointer",
        selected
          ? "bg-muted/60 border-l-violet-500"
          : "border-l-transparent hover:bg-muted/30",
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <StatusDot status={statusDot} />
        <span className="text-xs font-semibold text-foreground font-display flex-1 truncate">
          {run.name}
        </span>
        <span className="text-[9px] text-muted-foreground/40 shrink-0">
          {run.startedAt}
        </span>
      </div>
      <div className="flex items-center gap-1.5 mb-1 pl-3.5">
        <span className="text-[10px] text-muted-foreground">
          {run.modelLabel}
        </span>
        {run.compareLabel && (
          <>
            <span className="text-[9px] text-muted-foreground/30">vs</span>
            <span className="text-[10px] text-muted-foreground">
              {run.compareLabel}
            </span>
          </>
        )}
      </div>
      {/* Progress for running */}
      {run.status === "running" && run.progress != null && (
        <div className="pl-3.5 mb-1">
          <ProgressBar value={run.progress} color="#8B5CF6" animated />
          <span className="text-[9px] text-violet-500 mt-0.5 block">
            {run.progress}% &mdash; running {run.currentBenchmark}
          </span>
        </div>
      )}
      {/* Benchmark chips */}
      <div className="flex flex-wrap gap-1 pl-3.5">
        {run.benchmarks.slice(0, 5).map((bid) => {
          const b = BENCHMARKS.find((bk) => bk.id === bid);
          const cc =
            CATEGORY_COLORS[b?.category as BenchmarkCategory] ||
            CATEGORY_COLORS.custom;
          const score = run.scores[bid];
          return (
            <span
              key={bid}
              className="text-[8px] px-1.5 py-px rounded font-medium inline-flex items-center gap-1"
              style={{ background: cc.bg, color: cc.fg }}
            >
              {b?.name || bid}
              {score && (
                <span className="text-foreground font-semibold">
                  {score.value}
                </span>
              )}
            </span>
          );
        })}
        {run.benchmarks.length > 5 && (
          <span className="text-[8px] text-muted-foreground/30">
            +{run.benchmarks.length - 5}
          </span>
        )}
      </div>
    </button>
  );
}

// ── New Evaluation dialog ─────────────────────────────────────

function NewEvalDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>New Evaluation</DialogTitle>
          <DialogDescription>
            Configure and run a benchmark evaluation
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3 mb-4">
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Model
            </div>
            <Select defaultValue="llama-3.1-8b-cx">
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="llama-3.1-8b-cx">
                  llama-3.1-8b-cx (v4)
                </SelectItem>
                <SelectItem value="deepseek-r1-code">
                  deepseek-r1-code
                </SelectItem>
                <SelectItem value="qwen-2.5-72b">qwen-2.5-72b</SelectItem>
                <SelectItem value="guard-3b">guard-3b</SelectItem>
                <SelectItem value="mistral-7b-exp">mistral-7b-exp</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Compare Against (optional)
            </div>
            <Select>
              <SelectTrigger size="sm">
                <SelectValue placeholder={"\u2014 none \u2014"} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="llama-3.1-8b-cx-v3">
                  llama-3.1-8b-cx (v3)
                </SelectItem>
                <SelectItem value="mistral-7b-exp">mistral-7b-exp</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="mb-4">
          <div className="text-[9px] text-muted-foreground/50 mb-1.5 font-display uppercase tracking-wide">
            Benchmarks
          </div>
          <div className="flex flex-wrap gap-1.5">
            {BENCHMARKS.map((b) => {
              const cc = CATEGORY_COLORS[b.category];
              return (
                <label
                  key={b.id}
                  className="text-[10px] px-2.5 py-1.5 rounded-[5px] cursor-pointer bg-muted/60 border border-border text-foreground/70 flex items-center gap-1.5 font-display"
                >
                  <Checkbox
                    defaultChecked={["gsm8k", "mmlu", "hellaswag"].includes(
                      b.id,
                    )}
                  />
                  <span className="text-[10px]" style={{ color: cc.fg }}>
                    {b.icon}
                  </span>
                  {b.name}
                </label>
              );
            })}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              Compute
            </div>
            <Select defaultValue="local">
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">Local Cluster</SelectItem>
                <SelectItem value="aws">AWS (SkyPilot)</SelectItem>
                <SelectItem value="gcp">GCP (SkyPilot)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <div className="text-[9px] text-muted-foreground/50 mb-1 font-display uppercase tracking-wide">
              GPU
            </div>
            <Select defaultValue="1xa100">
              <SelectTrigger size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1xa100">
                  1&times; A100 80GB
                </SelectItem>
                <SelectItem value="2xa100">
                  2&times; A100 80GB
                </SelectItem>
                <SelectItem value="4xa100">
                  4&times; A100 80GB
                </SelectItem>
                <SelectItem value="4xh100">
                  4&times; H100 80GB
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={() => onOpenChange(false)}>Run Evaluation</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ── Main page ─────────────────────────────────────────────────

export function EvaluationsPage() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(
    "eval-0018",
  );
  const [showNewEvalModal, setShowNewEvalModal] = useState(false);

  const run = selectedRunId
    ? (EVAL_RUNS.find((r) => r.id === selectedRunId) ?? null)
    : null;

  const runningCount = EVAL_RUNS.filter(
    (r) => r.status === "running",
  ).length;

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Evaluations"
        subtitle={
          <>
            {EVAL_RUNS.length} runs &middot; {runningCount} running &middot;{" "}
            {BENCHMARKS.length} benchmarks
          </>
        }
        action={
          <Button size="sm" onClick={() => setShowNewEvalModal(true)}>
            &#x25C8; New Evaluation
          </Button>
        }
      />

      <Tabs
        defaultValue="runs"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-7 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="runs">Eval Runs</TabsTrigger>
            <TabsTrigger value="benchmarks">Benchmark Suite</TabsTrigger>
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent
          value="runs"
          className="flex-1 flex overflow-hidden mt-0"
        >
          {/* Run list (left) */}
          <div className="w-100 min-w-100 border-r border-border flex flex-col">
            <div className="flex-1 overflow-y-auto">
              {EVAL_RUNS.map((r) => (
                <RunListItem
                  key={r.id}
                  run={r}
                  selected={selectedRunId === r.id}
                  onSelect={() => setSelectedRunId(r.id)}
                />
              ))}
            </div>
          </div>

          {/* Detail (right) */}
          {run ? <EvalDetail run={run} /> : <EvalEmptyState />}
        </TabsContent>

        <TabsContent
          value="benchmarks"
          className="flex-1 flex overflow-hidden mt-0"
        >
          <BenchmarkSuite />
        </TabsContent>

        <TabsContent
          value="leaderboard"
          className="flex-1 flex overflow-hidden mt-0"
        >
          <Leaderboard />
        </TabsContent>
      </Tabs>

      <NewEvalDialog
        open={showNewEvalModal}
        onOpenChange={setShowNewEvalModal}
      />
    </div>
  );
}
