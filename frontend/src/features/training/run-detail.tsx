// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { StatusDot } from "@/components/ui/status-dot";
import { Button } from "@/components/ui/button";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { AGENT_FLOWS, EVALUATORS } from "./training-data";
import { METHOD_COLORS, toStatus } from "./experiment-sidebar";
import { TrainingTab } from "./training-tab";
import { ConfigTab } from "./config-tab";
import { CheckpointsTab } from "./checkpoints-tab";
import { CompareTab } from "./compare-tab";

interface RunDetailProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  run: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  compareRuns: any[];
  isComparing: boolean;
  detailTab: string;
  onTabChange: (tab: string) => void;
}

export function RunDetail({ run, compareRuns, isComparing, detailTab, onTabChange }: RunDetailProps) {
  const mc = METHOD_COLORS[run.method] || METHOD_COLORS.SFT;
  const runFlow = run.agentFlowId ? AGENT_FLOWS.find(f => f.id === run.agentFlowId) : null;
  const runEvaluator = run.evaluatorId ? EVALUATORS.find(e => e.id === run.evaluatorId) : null;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Detail header */}
      <div className="px-6 py-3.5 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <div className="w-1 h-5 rounded-sm" style={{ background: run.color }} />
              <h2 className="text-[15px] font-bold text-foreground font-display">{run.name}</h2>
              <span className={cn("text-[8px] px-1.5 py-0.5 rounded font-semibold border", mc.bg, mc.fg, mc.border)}>
                {run.method}
              </span>
              <StatusDot status={toStatus(run.status)} />
              <span className={cn(
                "text-[10px] font-medium",
                run.status === "running" ? "text-green-500" :
                run.status === "completed" ? "text-blue-500" :
                "text-muted-foreground",
              )}>
                {run.status}
              </span>
            </div>
            <div className="flex gap-3 text-[10px] text-muted-foreground/40 flex-wrap">
              <span>base: <span className="text-muted-foreground/60">{run.baseModel.split("/").pop()}</span></span>
              <span>dataset: <span className="text-muted-foreground/60">{run.dataset} ({run.datasetSamples})</span></span>
              {runFlow && (
                <span>flow: <span className="text-blue-500">{runFlow.icon} {runFlow.name}</span></span>
              )}
              {runEvaluator && (
                <span>eval: <span className="text-green-500">{runEvaluator.icon} {runEvaluator.name}</span></span>
              )}
              <span>compute: <span className="text-muted-foreground/60">
                {run.compute === "aws" ? "\u2601 AWS" : "\u229E Local"} &middot; {run.gpu}
              </span></span>
              {run.duration && (
                <span>duration: <span className="text-muted-foreground/60">{run.duration}</span></span>
              )}
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {run.status === "running" && (
              <Button variant="destructive" size="xs">Stop</Button>
            )}
            {run.status === "completed" && !run.hubRef && (
              <Button variant="outline" size="xs" className="text-green-500 border-green-500/20 hover:bg-green-500/10">
                Publish to Hub
              </Button>
            )}
            {run.status === "completed" && (
              <Button variant="outline" size="xs">Evaluate</Button>
            )}
            <Button variant="outline" size="xs">Clone</Button>
          </div>
        </div>

        {/* Progress bar for running */}
        {run.status === "running" && (
          <div className="mb-2">
            <div className="flex justify-between text-[9px] text-muted-foreground/60 mb-0.5">
              <span>Epoch {run.epochs.current}/{run.epochs.total} &middot; Step {run.steps.current.toLocaleString()}/{run.steps.total.toLocaleString()}</span>
              <span>{run.progress}%</span>
            </div>
            <ProgressBar value={run.progress} color={run.color} />
          </div>
        )}

        {/* Tabs */}
        <Tabs value={detailTab} onValueChange={onTabChange}>
          <TabsList variant="line" className="-mb-px">
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="config">Configuration</TabsTrigger>
            <TabsTrigger value="checkpoints">Checkpoints</TabsTrigger>
            <TabsTrigger value="compare" className="flex items-center gap-1">
              Compare
              {isComparing && <span className="w-1.5 h-1.5 rounded-full bg-primary" />}
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {detailTab === "training" && <TrainingTab run={run} />}
        {detailTab === "config" && <ConfigTab run={run} />}
        {detailTab === "checkpoints" && <CheckpointsTab run={run} />}
        {detailTab === "compare" && <CompareTab compareRuns={compareRuns} />}
      </div>
    </div>
  );
}

export function RunEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/30">
      <div className="text-center">
        <div className="text-3xl mb-3">&#x25EC;</div>
        <div className="text-sm font-display">Select a run to view details</div>
      </div>
    </div>
  );
}
