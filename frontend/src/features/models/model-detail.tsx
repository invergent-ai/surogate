// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { StatusDot } from "@/components/ui/status-dot";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/utils/cn";
import { toStatus, TYPE_STYLES } from "./models-data";
import { OverviewTab } from "./overview-tab";
import { PerformanceTab } from "./performance-tab";
import { ConfigTab } from "./config-tab";
import { FinetunesTab } from "./finetunes-tab";
import { useAppStore } from "@/stores/app-store";
import type { Model } from "./models-data";
import { isProxyModel } from "@/utils/model";

export function ModelDetail({ model }: { model: Model }) {
  const navigate = useNavigate();
  const startModel = useAppStore((s) => s.startModel);
  const stopModel = useAppStore((s) => s.stopModel);
  const restartModel = useAppStore((s) => s.restartModel);
  const deleteModel = useAppStore((s) => s.deleteModel);
  const pending = useAppStore((s) => s.modelPending[model.id] ?? null);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const isProxy = isProxyModel(model);

  const configComplete = isProxy || (
    model.engine !== "\u2014" &&
    (model.engine === "llamacpp" || model.infra === "k8s" || (model.gpu.type !== "\u2014" && model.gpu.count > 0)));

  const handleStart = () => {
    if (confirm(`Deploy ${model.displayName}?`)) {
      startModel(model.id);
    }
  };

  const handleStop = () => {
    if (confirm("Stop serving this model?")) {
      stopModel(model.id);
    }
  };

  const handleRestart = () => {
    if (confirm("Restart this model?")) {
      restartModel(model.id);
    }
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-start gap-3.5">
            <div
              className="w-11 h-11 rounded-[10px] shrink-0 flex items-center justify-center text-xl border"
              style={{
                backgroundColor: toStatus(model.status) === "serving" ? "#3B82F618" : undefined,
                borderColor: toStatus(model.status) === "serving" ? "#3B82F630" : undefined,
                color: toStatus(model.status) === "serving" ? "#3B82F6" : undefined,
              }}
            >
              &#x25C7;
            </div>
            <div>
              <div className="flex items-center gap-2 mb-0.5">
                <h2 className="text-base font-bold text-foreground font-display tracking-tight">
                  {model.displayName}
                </h2>
                {!isProxy && TYPE_STYLES[model.type] && (
                  <span
                    className={cn(
                      "text-sm px-1.5 py-0.5 rounded font-semibold uppercase tracking-wide",
                      TYPE_STYLES[model.type].bg,
                      TYPE_STYLES[model.type].fg,
                    )}
                  >
                    {model.type}
                  </span>
                )}
                {isProxy ? (
                  <span className="flex items-center gap-1 text-sm">
                    <StatusDot status="serving" />
                    <span className="font-medium text-green-500">proxy</span>
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-sm">
                    <StatusDot status={toStatus(model.status)} />
                    <span
                      className={cn(
                        "font-medium",
                        toStatus(model.status) === "error"
                          ? "text-destructive"
                          : toStatus(model.status) === "serving"
                            ? "text-green-500"
                            : "text-muted-foreground",
                      )}
                    >
                      {model.status}
                    </span>
                  </span>
                )}
              </div>
              <p className="text-sm text-muted-foreground max-w-[560px] leading-snug">
                {model.description}
              </p>
              <div className="flex gap-3 mt-1.5 text-xs text-muted-foreground/40 flex-wrap">
                {!isProxy && model.base !== "\u2014" && (
                  <span>
                    base:{" "}
                    <span className="text-muted-foreground">{model.base}</span>
                  </span>
                )}
                {!isProxy && model.family && model.family !== "\u2014" && (
                  <span>
                    family:{" "}
                    <span className="text-muted-foreground">{model.family}</span>
                  </span>
                )}
                {!isProxy && model.paramCount && model.paramCount !== "\u2014" && (
                  <span>
                    params:{" "}
                    <span className="text-muted-foreground">
                      {model.paramCount}
                    </span>
                  </span>
                )}
                {!isProxy && model.quantization && model.quantization !== "\u2014" && (
                  <span>
                    quant:{" "}
                    <span className="text-muted-foreground">
                      {model.quantization}
                    </span>
                  </span>
                )}
                {model.contextWindow > 0 && (
                  <span>
                    ctx:{" "}
                    <span className="text-muted-foreground">
                      {model.contextWindow.toLocaleString()}
                    </span>
                  </span>
                )}
                {model.engine !== "\u2014" && (
                  <span>
                    engine:{" "}
                    <span className="text-muted">
                      {model.engine}
                    </span>
                  </span>
                )}
                {model.endpoint && model.endpoint !== "\u2014" && (
                  <span>
                    endpoint:{" "}
                    <span className="text-blue-500">{model.endpoint}</span>
                  </span>
                )}
                {!isProxy && model.hubRef && (
                  <span>
                    hub:{" "}
                    <span className="text-blue-500">{model.hubRef}</span>
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex gap-1.5 shrink-0">
            {!isProxy && toStatus(model.status) !== "stopped" && toStatus(model.status) !== "error" && (
              <Button variant="outline" size="sm" onClick={handleStop} disabled={!!pending}>
                {pending ? "Stopping\u2026" : "Stop"}
              </Button>
            )}
            {!isProxy && toStatus(model.status) === "stopped" && (
              <Button
                size="sm"
                onClick={handleStart}
                disabled={!!pending || !configComplete}
                title={!configComplete ? "Configure engine and GPU in the Config tab first" : undefined}
              >
                {pending ? "Starting\u2026" : "Start"}
              </Button>
            )}
            {!isProxy && toStatus(model.status) === "error" && (
              <Button variant="destructive" size="sm" onClick={handleRestart} disabled={!!pending}>
                {pending ? "Restarting\u2026" : "Restart"}
              </Button>
            )}
            {toStatus(model.status) === "serving" && (
              <Button variant="outline" size="sm">
                &#x25B7; Playground
              </Button>
            )}
            <Button variant="destructive" size="sm" onClick={() => setDeleteOpen(true)}>
              Delete
            </Button>
            {!isProxy && (
              <Button variant="outline" size="sm" onClick={() => navigate({ to: "/studio/compute/workload-queue", search: { filter: "serving", id: model.id } })}>View job</Button>
            )}
          </div>
        </div>
      {/* Unconfigured alert */}
        {!isProxy && !configComplete && toStatus(model.status) === "stopped" && (
          <div className="px-6 py-3 shrink-0">
            <Alert variant={'destructive'}>
              <AlertDescription className="text-xs">
                This model needs a compute target, GPU, and engine before it can be
                started. Open the <strong>Config</strong> tab to set them up.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </div>

      {/* Tabs */}
      <Tabs
        defaultValue="overview"
        className="flex-1 flex flex-col overflow-hidden"
      >
        <div className="px-6 border-b border-border shrink-0">
          <TabsList variant="line">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
            {!isProxy && <TabsTrigger value="finetunes">Fine-tunes</TabsTrigger>}
          </TabsList>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          <TabsContent value="overview" className="mt-0">
            <OverviewTab model={model} />
          </TabsContent>
          <TabsContent value="performance" className="mt-0">
            <PerformanceTab model={model} />
          </TabsContent>
          <TabsContent value="config" className="mt-0">
            <ConfigTab model={model} />
          </TabsContent>
          {!isProxy && (
            <TabsContent value="finetunes" className="mt-0">
              <FinetunesTab model={model} />
            </TabsContent>
          )}
        </div>
      </Tabs>

      <ConfirmDialog
        open={deleteOpen}
        title="Delete model"
        description={<>This will permanently delete <strong>{model.displayName}</strong> and terminate any running service.</>}
        confirmLabel="Delete"
        variant="destructive"
        onConfirm={async () => {
          await deleteModel(model.id);
          setDeleteOpen(false);
        }}
        onCancel={() => setDeleteOpen(false)}
      />
    </div>
  );
}

// ── Empty state ────────────────────────────────────────────────

export function ModelEmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center text-muted-foreground/40">
      <div className="text-center">
        <div className="text-3xl mb-2">&#x25C7;</div>
        <div className="font-display text-sm">
          Select a model to view details
        </div>
        <div className="text-[10px] mt-1 max-w-[300px] leading-relaxed text-muted-foreground/30">
          Models are LLM endpoints serving inference to your agents and
          applications.
        </div>
      </div>
    </div>
  );
}
