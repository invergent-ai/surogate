// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { cn } from "@/utils/cn";
import { SFT_EXPERIMENTS, RL_EXPERIMENTS, AGENT_FLOWS, EVALUATORS } from "./training-data";
import { ExperimentSidebar, type RlSelection } from "./experiment-sidebar";
import { RunDetail, RunEmptyState } from "./run-detail";
import { RlItemDetail } from "./rl-item-detail";
import { NewRunDialog } from "./new-run-dialog";
import { NewExperimentDialog } from "./new-experiment-dialog";

const TOP_TABS = [
  { id: "sft" as const, label: "SFT" },
  { id: "rl" as const, label: "RL" },
];

export function TrainingPage() {
  const [topTab, setTopTab] = useState<"sft" | "rl">("sft");
  const [selectedExpId, setSelectedExpId] = useState<string | null>("exp-001");
  const [selectedRunId, setSelectedRunId] = useState<string | null>("ft-0042");
  const [detailTab, setDetailTab] = useState("training");
  const [compareRunIds, setCompareRunIds] = useState<string[]>([]);
  const [selectedRlItem, setSelectedRlItem] = useState<RlSelection | null>(null);
  const [showNewRunModal, setShowNewRunModal] = useState(false);
  const [showNewExpModal, setShowNewExpModal] = useState(false);

  const experiments = topTab === "sft" ? SFT_EXPERIMENTS : RL_EXPERIMENTS;
  const exp = experiments.find(e => e.id === selectedExpId);
  const run = exp?.runs.find(r => r.id === selectedRunId) ?? null;
  const allRuns = experiments.flatMap(e => e.runs);
  const isComparing = compareRunIds.length > 0;

  const selectedFlow = selectedRlItem?.type === "flow" ? AGENT_FLOWS.find(f => f.id === selectedRlItem.id) : null;
  const selectedEval = selectedRlItem?.type === "evaluator" ? EVALUATORS.find(e => e.id === selectedRlItem.id) : null;

  const compareRuns = compareRunIds
    .map(id => allRuns.find(r => r.id === id))
    .filter(Boolean);

  const toggleCompare = (runId: string) => {
    setCompareRunIds(prev =>
      prev.includes(runId)
        ? prev.filter(id => id !== runId)
        : [...prev, runId].slice(-3),
    );
  };

  const switchTopTab = (tab: "sft" | "rl") => {
    setTopTab(tab);
    const exps = tab === "sft" ? SFT_EXPERIMENTS : RL_EXPERIMENTS;
    setSelectedExpId(exps[0]?.id ?? null);
    setSelectedRunId(exps[0]?.runs[0]?.id ?? null);
    setCompareRunIds([]);
    setDetailTab("training");
    setSelectedRlItem(null);
  };

  const runningCount = allRuns.filter(r => r.status === "running").length;

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <PageHeader
        title="Training"
        subtitle={
          <>
            {experiments.length} experiments &middot; {allRuns.length} runs &middot; {runningCount} training now
          </>
        }
        action={
          <div className="flex items-center gap-2">
            {/* SFT / RL tabs */}
            <div className="flex gap-0.5 mr-2">
              {TOP_TABS.map(t => (
                <button
                  key={t.id}
                  onClick={() => switchTopTab(t.id)}
                  className={cn(
                    "px-3.5 py-1 rounded-md border text-[11px] font-display transition-colors cursor-pointer",
                    topTab === t.id
                      ? "border-primary/20 bg-primary/[0.07] text-primary font-semibold"
                      : "border-transparent text-muted-foreground hover:bg-muted/50",
                  )}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <Button variant="outline" size="sm" onClick={() => setShowNewExpModal(true)}>
              + Experiment
            </Button>
            <Button size="sm" onClick={() => setShowNewRunModal(true)}>
              &#x25EC; New Run
            </Button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar */}
        <ExperimentSidebar
          experiments={experiments}
          selectedExpId={selectedExpId}
          selectedRunId={selectedRunId}
          compareRunIds={compareRunIds}
          topTab={topTab}
          selectedRlItem={selectedRlItem}
          onSelectExp={(id) => {
            setSelectedExpId(id);
          }}
          onSelectRun={(id) => {
            setSelectedRunId(id);
            setDetailTab("training");
            setSelectedRlItem(null);
          }}
          onToggleCompare={toggleCompare}
          onClearCompare={() => setCompareRunIds([])}
          onViewComparison={() => setDetailTab("compare")}
          onSelectRlItem={(item) => {
            setSelectedRlItem(item);
            if (item) setSelectedRunId(null);
          }}
        />

        {/* Right detail pane */}
        {topTab === "rl" && (selectedFlow || selectedEval) && !run ? (
          <RlItemDetail
            item={selectedFlow || selectedEval}
            isFlow={!!selectedFlow}
            onClose={() => setSelectedRlItem(null)}
          />
        ) : run ? (
          <RunDetail
            run={run}
            compareRuns={compareRuns}
            isComparing={isComparing}
            detailTab={detailTab}
            onTabChange={setDetailTab}
          />
        ) : (
          <RunEmptyState />
        )}
      </div>

      {/* Modals */}
      <NewRunDialog open={showNewRunModal} onOpenChange={setShowNewRunModal} />
      <NewExperimentDialog open={showNewExpModal} onOpenChange={setShowNewExpModal} />
    </div>
  );
}
