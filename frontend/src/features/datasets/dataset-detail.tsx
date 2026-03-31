// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { FORMAT_STYLES, SOURCE_LABELS, toStatus } from "./datasets-data";
import type { Dataset } from "./datasets-data";
import { OverviewTab } from "./overview-tab";
import { SamplesTab } from "./samples-tab";
import { PipelineTab } from "./pipeline-tab";
import { VersionsTab } from "./versions-tab";

export function DatasetDetail({ dataset }: { dataset: Dataset }) {
  const fs = FORMAT_STYLES[dataset.format] ?? FORMAT_STYLES.SFT;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-start gap-3.5">
            <div
              className="w-11 h-11 rounded-[10px] shrink-0 flex items-center justify-center text-xl border"
              style={{
                backgroundColor: `${dataset.color}12`,
                borderColor: `${dataset.color}30`,
                color: dataset.color,
              }}
            >
              &#x25A4;
            </div>
            <div>
              <div className="flex items-center gap-2 mb-0.5">
                <h2 className="text-base font-bold text-foreground font-display tracking-tight">
                  {dataset.displayName}
                </h2>
                <span className={`text-[8px] font-semibold px-1.5 py-px rounded border ${fs.bg} ${fs.fg} ${fs.border}`}>
                  {dataset.format}
                </span>
                <span className="flex items-center gap-1 text-[10px]">
                  <StatusDot status={toStatus(dataset.status)} />
                  <span className={dataset.status === "error" ? "text-destructive font-medium" : dataset.status === "building" ? "text-amber-500 font-medium" : "text-green-500 font-medium"}>
                    {dataset.status}
                  </span>
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground max-w-[540px] leading-snug">
                {dataset.description}
              </p>
              <div className="flex gap-3 mt-1.5 text-[10px] text-muted-foreground/40 flex-wrap">
                <span>source: <span className="text-muted-foreground">{SOURCE_LABELS[dataset.source]}</span></span>
                <span>{dataset.samples.toLocaleString()} samples</span>
                <span>{dataset.tokens} tokens</span>
                <span>{dataset.size}</span>
                <span>by: <span className="text-muted-foreground">{dataset.createdBy}</span></span>
                {dataset.hubRef && (
                  <span>hub: <span className="text-green-500">{dataset.hubRef}</span></span>
                )}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {!dataset.published && (
              <Button variant="outline" size="xs" className="text-green-500 border-green-500/30 hover:bg-green-500/10">
                Publish to Hub
              </Button>
            )}
            <Button variant="outline" size="xs">Download</Button>
            <Button variant="outline" size="icon-xs">&hellip;</Button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1 mb-2">
          {dataset.tags.map((t) => (
            <span key={t} className="text-[8px] px-1.5 py-px rounded bg-muted text-muted-foreground">
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="flex-1 flex flex-col overflow-hidden">
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="samples">Samples</TabsTrigger>
            <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
          </TabsList>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          <TabsContent value="overview" className="mt-0">
            <OverviewTab dataset={dataset} />
          </TabsContent>
          <TabsContent value="samples" className="mt-0">
            <SamplesTab dataset={dataset} />
          </TabsContent>
          <TabsContent value="pipeline" className="mt-0">
            <PipelineTab dataset={dataset} />
          </TabsContent>
          <TabsContent value="versions" className="mt-0">
            <VersionsTab dataset={dataset} />
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}

export function DatasetEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x25A4;</div>
        <div className="font-display text-sm">Select a dataset to view details</div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Browse datasets, preview samples, inspect pipelines, and track versions.
        </div>
      </div>
    </div>
  );
}
