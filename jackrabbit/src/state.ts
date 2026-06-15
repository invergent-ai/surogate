// In-memory view model for the dashboard. Pure data: ingest typed records,
// maintain bounded history, derive ETA / trend. No I/O, no rendering.

import type { ConfigRecord, GpuRecord, Record_, StepRecord } from "./records.ts";

const MAX_HISTORY = 5000;
const MAX_STEP_TS = 200;

export class WatchState {
  // header
  model: string | null = null;
  recipe: string | null = null;
  maxSteps: number | null = null;
  lora: boolean | null = null;
  configFields: Record<string, unknown> = {};
  // latest scalars
  step = 0;
  epoch: number | null = null;
  latestTrainLoss: number | null = null;
  latestEvalLoss: number | null = null;
  lr: number | null = null;
  gradNorm: number | null = null;
  tokensPerSecond: number | null = null;
  phase: string | null = null;
  // history
  lossHistory: number[] = [];
  lossSteps: number[] = [];
  evalHistory: Array<[number, number]> = [];
  recentSteps: StepRecord[] = [];
  // gpus (latest per id)
  private gpus = new Map<number, GpuRecord>();
  // timing
  firstTs: number | null = null;
  lastTs: number | null = null;
  private stepTs: Array<[number, number]> = []; // (step, ts)

  get hasGpus(): boolean {
    return this.gpus.size > 0;
  }

  gpusSorted(): GpuRecord[] {
    return [...this.gpus.keys()].sort((a, b) => a - b).map((k) => this.gpus.get(k)!);
  }

  ingest(records: Record_[]): void {
    for (const r of records) {
      if (r.kind === "config") this.ingestConfig(r);
      else if (r.kind === "step") this.ingestStep(r);
      else this.gpus.set(r.gpuId, r);
    }
  }

  private ingestConfig(r: ConfigRecord): void {
    if (r.recipe !== null) this.recipe = r.recipe;
    if (r.model !== null) this.model = r.model;
    Object.assign(this.configFields, r.fields);
    const ms = r.fields["max_steps"];
    // SFTConfig defaults max_steps to -1 when unset; only a positive value is a real total.
    if (typeof ms === "number" && Number.isInteger(ms) && ms > 0) this.maxSteps = ms;
    if ("lora" in r.fields) this.lora = Boolean(r.fields["lora"]);
  }

  private ingestStep(r: StepRecord): void {
    this.step = r.step;
    if (r.epoch !== null) this.epoch = r.epoch;
    if (r.lr !== null) this.lr = r.lr;
    if (r.gradNorm !== null) this.gradNorm = r.gradNorm;
    if (r.tokensPerSecond !== null) this.tokensPerSecond = r.tokensPerSecond;
    if (r.phase) this.phase = r.phase;
    if (r.trainLoss !== null) {
      this.latestTrainLoss = r.trainLoss;
      this.lossHistory.push(r.trainLoss);
      this.lossSteps.push(r.step);
      if (this.lossHistory.length > MAX_HISTORY) {
        this.lossHistory.shift();
        this.lossSteps.shift();
      }
    }
    if (r.evalLoss !== null) {
      this.latestEvalLoss = r.evalLoss;
      this.evalHistory.push([r.step, r.evalLoss]);
      if (this.evalHistory.length > MAX_HISTORY) this.evalHistory.shift();
    }
    if (this.firstTs === null) this.firstTs = r.ts;
    this.lastTs = r.ts;
    this.stepTs.push([r.step, r.ts]);
    if (this.stepTs.length > MAX_STEP_TS) this.stepTs.shift();
    this.recentSteps.push(r);
    if (this.recentSteps.length > 50) this.recentSteps.shift();
  }

  private secondsPerStep(): number | null {
    if (this.stepTs.length < 2) return null;
    const deltas: number[] = [];
    for (let i = 1; i < this.stepTs.length; i++) {
      const [s0, t0] = this.stepTs[i - 1]!;
      const [s1, t1] = this.stepTs[i]!;
      const ds = s1 - s0;
      if (ds > 0 && t1 >= t0) deltas.push((t1 - t0) / ds);
    }
    if (deltas.length === 0) return null;
    deltas.sort((a, b) => a - b);
    return deltas[Math.floor(deltas.length / 2)]!;
  }

  etaSeconds(): number | null {
    if (!this.maxSteps) return null;
    const sps = this.secondsPerStep();
    if (sps === null) return null;
    return Math.max(0, this.maxSteps - this.step) * sps;
  }

  elapsedSeconds(): number | null {
    if (this.firstTs === null || this.lastTs === null) return null;
    return this.lastTs - this.firstTs;
  }

  lossTrend(window = 10): number {
    const v = this.lossHistory.slice(-window);
    if (v.length < 2) return 0;
    return v[v.length - 1]! - v[0]!;
  }
}
