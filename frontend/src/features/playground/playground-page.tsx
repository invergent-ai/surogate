// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { Thread } from "@/components/assistant-ui/thread";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useAppStore } from "@/stores/app-store";
import { cn } from "@/utils/cn";
import {
  ColumnsIcon,
  DownloadIcon,
  LayoutListIcon,
  Trash2Icon,
} from "lucide-react";
import { useSearch } from "@tanstack/react-router";
import {
  type ReactElement,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ContextUsageBar } from "./components/context-usage-bar";
import { ModelInfoBar } from "./model-info-bar";
import { ParametersPanel } from "./parameters-panel";
import { DEFAULT_COLOR, type PlaygroundParams } from "./playground-data";
import { PlaygroundRuntimeProvider } from "./runtime-provider";
import { SessionsPanel } from "./sessions-panel";
import {
  CompareHandlesProvider,
  type CompareHandle,
  RegisterCompareHandle,
  SharedComposer,
} from "./shared-composer";
import { usePlaygroundStore } from "./stores/playground-store";
import { SystemPromptSection } from "./system-prompt-section";
import type { PlaygroundView } from "./types";

// ── Single Content ───────────────────────────────────────────

const SingleContent = memo(function SingleContent({
  threadId,
  newThreadNonce,
}: {
  threadId?: string;
  newThreadNonce?: string;
}): ReactElement {
  return (
    <PlaygroundRuntimeProvider
      initialThreadId={threadId}
      newThreadNonce={newThreadNonce}
    >
      <div className="flex h-full flex-col overflow-hidden">
        <Thread />
      </div>
    </PlaygroundRuntimeProvider>
  );
});

// ── Compare Content ──────────────────────────────────────────

const CompareContent = memo(function CompareContent({
  pairId,
  servingModels,
  selectedModelId,
  compareModelId,
  onCompareModelChange,
}: {
  pairId: string;
  servingModels: { id: string; displayName: string; projectColor: string }[];
  selectedModelId: string | null;
  compareModelId: string | null;
  onCompareModelChange: (modelId: string) => void;
}): ReactElement {
  const handlesRef = useRef<Record<string, CompareHandle>>({});

  const leftModel = servingModels.find((m) => m.id === selectedModelId);
  const leftColor = leftModel?.projectColor || DEFAULT_COLOR;

  const rightCandidates = servingModels.filter((m) => m.id !== selectedModelId);

  return (
    <CompareHandlesProvider handlesRef={handlesRef}>
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="grid min-h-0 flex-1 grid-cols-2">
          <div className="flex min-h-0 flex-col">
            <div className="px-3 py-1.5">
              <span
                className="text-[10px] font-semibold uppercase tracking-wider"
                style={{ color: leftColor }}
              >
                {leftModel?.displayName ?? "Model 1"}
              </span>
            </div>
            <div className="min-h-0 flex-1">
              <PlaygroundRuntimeProvider pairId={pairId} modelId={selectedModelId ?? undefined}>
                <RegisterCompareHandle name="left" />
                <Thread hideComposer hideWelcome />
              </PlaygroundRuntimeProvider>
            </div>
          </div>
          <div className="flex min-h-0 flex-col border-l border-border">
            <div className="flex items-center justify-end gap-1.5 px-3 py-1.5">
              {rightCandidates.map((m) => {
                const color = m.projectColor || DEFAULT_COLOR;
                const isActive = m.id === compareModelId;
                return (
                  <button
                    key={m.id}
                    type="button"
                    className={cn(
                      "rounded-md border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider transition-all",
                      isActive
                        ? "font-bold"
                        : "border-transparent text-muted-foreground hover:text-foreground",
                    )}
                    style={
                      isActive
                        ? { borderColor: `${color}44`, backgroundColor: `${color}12`, color }
                        : undefined
                    }
                    onClick={() => {
                      if (!isActive) onCompareModelChange(m.id);
                    }}
                  >
                    {m.displayName}
                  </button>
                );
              })}
            </div>
            <div className="min-h-0 flex-1">
              <PlaygroundRuntimeProvider pairId={pairId} modelId={compareModelId ?? undefined}>
                <RegisterCompareHandle name="right" />
                <Thread hideComposer hideWelcome />
              </PlaygroundRuntimeProvider>
            </div>
          </div>
        </div>
        <div className="mx-auto w-full max-w-4xl px-4 py-4">
          <SharedComposer handlesRef={handlesRef} />
        </div>
      </div>
    </CompareHandlesProvider>
  );
});

// ── Main Page ────────────────────────────────────────────────

export function PlaygroundPage() {
  const allModels = useAppStore((s) => s.models);
  const fetchModels = useAppStore((s) => s.fetchModels);

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  const servingModels = useMemo(
    () => allModels.filter((m) => m.status === "running"),
    [allModels],
  );

  // Apply model ID from URL search param (e.g. navigating from Models page)
  const { modelId: urlModelId } = useSearch({ strict: false }) as { modelId?: string };
  useEffect(() => {
    if (urlModelId && servingModels.some((m) => m.id === urlModelId)) {
      usePlaygroundStore.getState().setSelectedModelId(urlModelId);
    }
  }, [urlModelId, servingModels]);

  // Playground store
  const selectedModelId = usePlaygroundStore((s) => s.selectedModelId);
  const compareModelId = usePlaygroundStore((s) => s.compareModelId);
  const showSessions = usePlaygroundStore((s) => s.showSessions);
  const systemPrompt = usePlaygroundStore((s) => s.systemPrompt);
  const activePreset = usePlaygroundStore((s) => s.activePreset);
  const params = usePlaygroundStore((s) => s.params);
  const contextUsage = usePlaygroundStore((s) => s.contextUsage);
  const setSelectedModelId = usePlaygroundStore((s) => s.setSelectedModelId);
  const setCompareModelId = usePlaygroundStore((s) => s.setCompareModelId);
  const toggleSessions = usePlaygroundStore((s) => s.toggleSessions);
  const setSystemPrompt = usePlaygroundStore((s) => s.setSystemPrompt);
  const setParams = usePlaygroundStore((s) => s.setParams);
  const applyPreset = usePlaygroundStore((s) => s.applyPreset);

  const [view, setView] = useState<PlaygroundView>({
    mode: "single",
    newThreadNonce: crypto.randomUUID(),
  });

  // Auto-select first serving model
  const effectiveModelId =
    selectedModelId && servingModels.some((m) => m.id === selectedModelId)
      ? selectedModelId
      : (servingModels[0]?.id ?? null);

  // Sync auto-selection back to store
  useEffect(() => {
    if (effectiveModelId && effectiveModelId !== selectedModelId) {
      setSelectedModelId(effectiveModelId);
    }
  }, [effectiveModelId, selectedModelId, setSelectedModelId]);

  const model = servingModels.find((m) => m.id === effectiveModelId);

  const handleParamChange = useCallback(
    (key: keyof PlaygroundParams, value: number) => {
      setParams({ ...usePlaygroundStore.getState().params, [key]: value });
    },
    [setParams],
  );

  const handleSystemPromptChange = useCallback(
    (value: string) => {
      setSystemPrompt(value);
    },
    [setSystemPrompt],
  );

  const toggleCompare = useCallback(() => {
    if (compareModelId) {
      setCompareModelId(null);
      setView({ mode: "single", newThreadNonce: crypto.randomUUID() });
    } else {
      const other = servingModels.find((m) => m.id !== effectiveModelId);
      if (other) {
        setCompareModelId(other.id);
        setView({ mode: "compare", pairId: crypto.randomUUID() });
      }
    }
  }, [compareModelId, servingModels, effectiveModelId, setCompareModelId]);

  const handleCompareModelChange = useCallback((modelId: string) => {
    setCompareModelId(modelId);
    setView({ mode: "compare", pairId: crypto.randomUUID() });
  }, [setCompareModelId]);

  const handleNewThread = useCallback(() => {
    usePlaygroundStore.getState().setActiveThreadId(null);
    setView({ mode: "single", newThreadNonce: crypto.randomUUID() });
    setCompareModelId(null);
  }, [setCompareModelId]);

  const handleNewCompare = useCallback(() => {
    const other = servingModels.find((m) => m.id !== effectiveModelId);
    if (other) {
      setCompareModelId(other.id);
    }
    setView({ mode: "compare", pairId: crypto.randomUUID() });
    usePlaygroundStore.getState().setContextUsage(null);
  }, [servingModels, effectiveModelId, setCompareModelId]);

  const handleThreadSelect = useCallback(
    (nextView: PlaygroundView) => {
      if (nextView.mode === "single") {
        setCompareModelId(null);
        // Ensure we always have a threadId or nonce so the runtime creates a thread
        if (!nextView.threadId && !nextView.newThreadNonce) {
          setView({ mode: "single", newThreadNonce: crypto.randomUUID() });
          return;
        }
      }
      setView(nextView);
    },
    [setCompareModelId],
  );

  // No models running
  if (!model) {
    return (
      <div className="flex flex-1 flex-col overflow-hidden bg-background">
        <PageHeader
          title="Playground"
          subtitle="Interactive testing environment for agents and models."
        />
        <div className="flex flex-1 items-center justify-center">
          <div className="text-center text-muted-foreground">
            <p className="text-sm font-medium">No models running</p>
            <p className="mt-1 text-xs">
              Start a model from the Models page to begin.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-background">
      <PageHeader
        title="Playground"
        action={
          <div className="flex items-center gap-1.5">
            {/* Model selector */}
            {servingModels.map((m) => {
              const color = m.projectColor || DEFAULT_COLOR;
              return (
                <button
                  key={m.id}
                  type="button"
                  className={cn(
                    "flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-display text-[10px] font-medium transition-all",
                    effectiveModelId === m.id
                      ? "font-semibold"
                      : "border-transparent text-muted-foreground hover:text-foreground",
                  )}
                  style={
                    effectiveModelId === m.id
                      ? {
                          borderColor: `${color}44`,
                          backgroundColor: `${color}12`,
                          color,
                        }
                      : undefined
                  }
                  onClick={() => setSelectedModelId(m.id)}
                >
                  <span
                    className="inline-block size-1.25 rounded-full"
                    style={{
                      backgroundColor: color,
                      boxShadow:
                        effectiveModelId === m.id
                          ? `0 0 6px ${color}`
                          : "none",
                    }}
                  />
                  {m.displayName}
                </button>
              );
            })}

            <Separator orientation="vertical" className="mx-1 h-5" />

            {/* Context usage */}
            {contextUsage && model.contextWindow > 0 && (
              <ContextUsageBar
                used={contextUsage.totalTokens}
                total={model.contextWindow}
                cached={contextUsage.cachedTokens}
                promptTokens={contextUsage.promptTokens}
                completionTokens={contextUsage.completionTokens}
              />
            )}

            <Separator orientation="vertical" className="mx-1 h-5" />

            {/* Action buttons */}
            <Button
              variant={showSessions ? "secondary" : "outline"}
              size="xs"
              onClick={toggleSessions}
            >
              <LayoutListIcon className="size-3" data-icon="inline-start" />
              Sessions
            </Button>
            <Button
              variant={view.mode === "compare" ? "secondary" : "outline"}
              size="xs"
              onClick={toggleCompare}
              disabled={servingModels.length < 2}
            >
              <ColumnsIcon className="size-3" data-icon="inline-start" />
              Compare
            </Button>
            <Button variant="outline" size="xs" onClick={handleNewThread}>
              <Trash2Icon className="size-3" data-icon="inline-start" />
              Clear
            </Button>
            <Button
              variant="outline"
              size="xs"
              className="border-amber-500/30 bg-amber-500/10 text-amber-500 hover:bg-amber-500/20"
            >
              <DownloadIcon className="size-3" data-icon="inline-start" />
              Export
            </Button>
          </div>
        }
      />

      <div className="flex flex-1 overflow-hidden">
        {showSessions && (
          <SessionsPanel
            view={view}
            selectedModelId={effectiveModelId}
            onSelect={handleThreadSelect}
            onNewSession={handleNewThread}
            onNewCompare={handleNewCompare}
          />
        )}

        {/* Main content area */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {view.mode === "single" && (
            <>
              <ModelInfoBar
                model={model}
                messageCount={0}
                totalTokens={contextUsage?.totalTokens ?? 0}
                avgLatency={0}
                isStreaming={
                  Object.keys(
                    usePlaygroundStore.getState().runningByThreadId,
                  ).length > 0
                }
              />
              <SystemPromptSection
                value={systemPrompt}
                onChange={handleSystemPromptChange}
                activePreset={activePreset}
                onApplyPreset={applyPreset}
              />
              <SingleContent
                key={view.threadId ?? view.newThreadNonce ?? "new"}
                threadId={view.threadId}
                newThreadNonce={view.newThreadNonce}
              />
            </>
          )}

          {view.mode === "compare" && (
            <CompareContent
              key={view.pairId}
              pairId={view.pairId}
              servingModels={servingModels}
              selectedModelId={effectiveModelId}
              compareModelId={compareModelId}
              onCompareModelChange={handleCompareModelChange}
            />
          )}
        </div>

        <ParametersPanel
          model={model}
          params={params}
          onParamChange={handleParamChange}
          activePreset={activePreset}
          onApplyPreset={applyPreset}
          stats={{
            messageCount: 0,
            totalTokens: contextUsage?.totalTokens ?? 0,
            avgLatency: 0,
            userTurns: 0,
          }}
        />
      </div>
    </div>
  );
}
