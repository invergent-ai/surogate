// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: AGPL-3.0-only
//
import { AGENT_FLOWS, EVALUATORS } from "./training-data";

function ConfigSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="bg-card rounded-lg ring-1 ring-foreground/10 overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <span className="text-[13px] font-semibold text-foreground font-display">{title}</span>
      </div>
      {children}
    </section>
  );
}

function ConfigRow({ label, value, labelColor = "text-blue-500" }: { label: string; value: string; labelColor?: string }) {
  return (
    <div className="px-4 py-[7px] border-b border-border/30 flex gap-3 text-[11px]">
      <span className={`${labelColor} min-w-40`}>{label}</span>
      <span className="text-muted-foreground/30">=</span>
      <span className="text-foreground/70">{value}</span>
    </div>
  );
}

function InfoRow({ label, value, valueColor }: { label: string; value: React.ReactNode; valueColor?: string }) {
  return (
    <div className="flex justify-between py-0.5 text-[10px]">
      <span className="text-muted-foreground/40">{label}</span>
      <span className={valueColor || "text-foreground/70"}>{value}</span>
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function ConfigTab({ run }: { run: any }) {
  const isRL = run.method === "GRPO" || run.method === "PPO";
  const runFlow = run.agentFlowId ? AGENT_FLOWS.find(f => f.id === run.agentFlowId) : null;
  const runEvaluator = run.evaluatorId ? EVALUATORS.find(e => e.id === run.evaluatorId) : null;

  const trainingParams = [
    { k: "method", v: run.method },
    { k: "base_model", v: run.baseModel },
    { k: "dataset", v: `${run.dataset} (${run.datasetSamples})` },
    { k: "learning_rate", v: String(run.lr) },
    { k: "batch_size", v: String(run.batchSize) },
    { k: "grad_accum", v: String(run.gradAccum) },
    { k: "epochs", v: String(run.epochs.total) },
    { k: "warmup_steps", v: String(run.warmupSteps) },
    { k: "weight_decay", v: String(run.weightDecay) },
    { k: "scheduler", v: run.scheduler },
    { k: "optimizer", v: run.optimizer },
  ];

  return (
    <div className="animate-in fade-in duration-200">
      <div className="grid grid-cols-2 gap-4">
        {/* Training parameters */}
        <ConfigSection title="Training Parameters">
          <div className="font-mono">
            {trainingParams.map(({ k, v }) => (
              <ConfigRow key={k} label={k} value={v} />
            ))}
          </div>
        </ConfigSection>

        <div className="flex flex-col gap-4">
          {/* LoRA / Full config */}
          <ConfigSection title={run.lora ? "LoRA Configuration" : "Full Fine-tune"}>
            {run.lora ? (
              <div className="font-mono">
                {Object.entries(run.lora).map(([k, v]) => (
                  <ConfigRow key={k} label={k} value={String(v)} labelColor="text-violet-500" />
                ))}
              </div>
            ) : (
              <div className="px-4 py-3 text-[11px] text-muted-foreground/60">
                Full parameter fine-tuning (no LoRA)
              </div>
            )}
          </ConfigSection>

          {/* DPO-specific config */}
          {run.method === "DPO" && run.beta !== undefined && (
            <ConfigSection title="DPO Parameters">
              <div className="font-mono">
                <ConfigRow label="beta" value={String(run.beta)} labelColor="text-primary" />
                <ConfigRow label="label_smoothing" value={String(run.labelSmoothing)} labelColor="text-primary" />
                <ConfigRow label="ref_model" value={run.refModel} labelColor="text-primary" />
              </div>
            </ConfigSection>
          )}

          {/* RL: Algorithm */}
          {isRL && run.algorithm && (
            <ConfigSection title="Algorithm">
              <div className="font-mono">
                {Object.entries(run.algorithm).map(([k, v]) => (
                  <ConfigRow key={k} label={k} value={String(v)} labelColor="text-primary" />
                ))}
              </div>
            </ConfigSection>
          )}

          {/* RL: Workflow */}
          {isRL && run.workflow && (
            <ConfigSection title="Workflow">
              <div className="font-mono">
                {Object.entries(run.workflow).map(([k, v]) => (
                  <ConfigRow key={k} label={k} value={String(v)} labelColor="text-cyan-500" />
                ))}
              </div>
            </ConfigSection>
          )}

          {/* RL: Pipeline */}
          {isRL && (runFlow || runEvaluator) && (
            <section className="bg-card rounded-lg ring-1 ring-foreground/10 p-3.5">
              <div className="text-xs font-semibold text-foreground font-display mb-2.5">
                RLLM Pipeline
              </div>
              {runFlow && (
                <InfoRow
                  label="AgentFlow"
                  value={<>{runFlow.icon} {runFlow.name} <span className="text-muted-foreground/40">({runFlow.type})</span></>}
                  valueColor="text-blue-500"
                />
              )}
              {runEvaluator && (
                <InfoRow
                  label="Evaluator"
                  value={<>{runEvaluator.icon} {runEvaluator.name} <span className="text-muted-foreground/40">({runEvaluator.rewardType})</span></>}
                  valueColor="text-green-500"
                />
              )}
              {run.episodes && (
                <>
                  <InfoRow label="Episodes" value={`${run.episodes.completed.toLocaleString()} / ${run.episodes.total.toLocaleString()}`} />
                  <InfoRow label="Avg Steps/Episode" value={run.avgStepsPerEpisode} />
                </>
              )}
            </section>
          )}

          {/* Compute */}
          <section className="bg-card rounded-lg ring-1 ring-foreground/10 p-3.5">
            <div className="text-xs font-semibold text-foreground font-display mb-2.5">
              Compute
            </div>
            <InfoRow label="Provider" value={run.compute === "aws" ? "AWS (SkyPilot)" : "Local Cluster"} />
            <InfoRow label="GPU" value={run.gpu} />
            <InfoRow label="GPU Util" value={run.gpuUtil > 0 ? `${run.gpuUtil}%` : "\u2014"} />
            <InfoRow label="Started" value={run.startedAt || "Queued"} />
          </section>
        </div>
      </div>
    </div>
  );
}
