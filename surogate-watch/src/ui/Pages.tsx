import React from "react";
import { Box, Text } from "ink";
import type { WatchState } from "../state.ts";
import { fmtCount, fmtFloat } from "../format.ts";
import { C, memColor, meterParts, tempColor } from "./theme.ts";
import { Panel } from "./Panel.tsx";
import { Chart } from "./Monitor.tsx";
import type { NavItem } from "./Sidebar.tsx";

function Headline({ s }: { s: WatchState }) {
  const trend = s.lossTrend();
  const arrow = trend < 0 ? "▼" : trend > 0 ? "▲" : "•";
  const arrowColor = trend < 0 ? C.green : trend > 0 ? C.red : C.muted;
  return (
    <Box>
      <Text color={C.muted}>loss </Text>
      <Text bold color={C.train}>
        {fmtFloat(s.latestTrainLoss)}
      </Text>
      <Text color={arrowColor}> {arrow}</Text>
      {s.latestEvalLoss !== null && (
        <Text>
          <Text color={C.muted}>{"   eval "}</Text>
          <Text bold color={C.eval}>
            {fmtFloat(s.latestEvalLoss)}
          </Text>
        </Text>
      )}
      <Text color={C.muted}>{"   lr "}</Text>
      <Text color={C.text}>{fmtFloat(s.lr)}</Text>
      <Text color={C.muted}>{"   |∇| "}</Text>
      <Text color={C.text}>{fmtFloat(s.gradNorm)}</Text>
      <Text color={C.muted}>{"   "}</Text>
      <Text color={C.text}>{fmtCount(s.tokensPerSecond)} tok/s</Text>
    </Box>
  );
}

function GpuRows({ s, big }: { s: WatchState; big: boolean }) {
  if (!s.hasGpus) return <Text color={C.muted}>no GPU telemetry yet</Text>;
  const w = big ? 16 : 10;
  return (
    <Box flexDirection="column">
      {s.gpusSorted().map((g) => {
        const name = (g.name ?? `gpu${g.gpuId}`).slice(0, 12).padEnd(12);
        const temp = g.temp === null ? "  —" : `${Math.round(g.temp)}°`.padStart(3);
        const pwr = g.power === null ? "   —" : `${Math.round(g.power)}W`.padStart(5);
        const sm = meterParts(g.smUtil === null ? null : g.smUtil / 100, w);
        const mem = meterParts(g.memUtil, w);
        return (
          <Box key={g.gpuId}>
            <Text color={C.dim}>{String(g.gpuId).padEnd(2)} </Text>
            <Text color={C.text}>{name} </Text>
            <Text color={tempColor(g.temp)}>{temp}</Text>
            <Text color={C.text}>{"  " + pwr + "   sm "}</Text>
            <Text color={C.accent}>{"█".repeat(sm.filled)}</Text>
            <Text color={C.dim}>{"█".repeat(sm.track)}</Text>
            <Text color={C.text}> {sm.pct}</Text>
            <Text color={C.text}>{"   mem "}</Text>
            <Text color={g.memUtil === null ? C.muted : memColor(g.memUtil)}>{"█".repeat(mem.filled)}</Text>
            <Text color={C.dim}>{"█".repeat(mem.track)}</Text>
            <Text color={C.text}> {mem.pct}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

function MonitorPage({ s, chartImage, chartHeight }: { s: WatchState; chartImage: string; chartHeight: number }) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title="training loss" flexGrow={1} minHeight={chartHeight + 3}>
        <Chart image={chartImage} />
        <Box marginTop={1}>
          <Headline s={s} />
        </Box>
      </Panel>
      <Panel title="devices">
        <Box marginTop={1}>
          <GpuRows s={s} big={false} />
        </Box>
      </Panel>
    </Box>
  );
}

function GpusPage({ s }: { s: WatchState }) {
  return (
    <Panel title="devices" flexGrow={1}>
      <Box marginTop={1}>
        <GpuRows s={s} big />
      </Box>
    </Panel>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <Box>
      <Text color={C.muted}>{label.padEnd(22)}</Text>
      <Text color={C.text}>{value}</Text>
    </Box>
  );
}

function DataPage({ s }: { s: WatchState }) {
  const f = s.configFields;
  const get = (k: string) => (f[k] === undefined || f[k] === null ? "—" : String(f[k]));
  return (
    <Panel title="run config" flexGrow={1}>
      <Box flexDirection="column" marginTop={1}>
        <Fact label="model" value={s.model ?? get("model")} />
        <Fact label="recipe" value={s.recipe ?? get("recipe")} />
        <Fact label="batch / device" value={get("per_device_train_batch_size")} />
        <Fact label="grad accumulation" value={get("gradient_accumulation_steps")} />
        <Fact label="sequence length" value={get("sequence_len")} />
        <Fact label="max steps" value={get("max_steps")} />
        <Fact label="eval steps" value={get("eval_steps")} />
        <Fact label="learning rate" value={get("learning_rate")} />
        <Fact label="lora" value={get("lora")} />
        <Fact label="lora rank" value={get("lora_rank")} />
        <Fact label="optimizer" value={get("optimizer")} />
        <Fact label="output dir" value={get("output_dir")} />
      </Box>
    </Panel>
  );
}

function LogsPage({ s }: { s: WatchState }) {
  const recent = s.recentSteps.slice(-16);
  return (
    <Panel title="log" flexGrow={1}>
      <Box flexDirection="column" marginTop={1}>
        {recent.length === 0 ? (
          <Text color={C.muted}>waiting for metrics…</Text>
        ) : (
          recent.map((r, i) => (
            <Text key={i} color={C.text}>
              step {String(r.step).padStart(4)}
              {r.trainLoss !== null ? `  loss ${fmtFloat(r.trainLoss)}` : ""}
              {r.evalLoss !== null ? `  eval ${fmtFloat(r.evalLoss)}` : ""}
              {r.lr !== null ? `  lr ${fmtFloat(r.lr)}` : ""}
              {r.tokensPerSecond !== null ? `  ${fmtCount(r.tokensPerSecond)} tok/s` : ""}
              {r.phase ? `  ${r.phase}` : ""}
            </Text>
          ))
        )}
      </Box>
    </Panel>
  );
}

export function Page({
  nav,
  s,
  chartImage,
  chartHeight,
}: {
  nav: NavItem;
  s: WatchState;
  chartImage: string;
  chartHeight: number;
}) {
  switch (nav) {
    case "GPUs":
      return <GpusPage s={s} />;
    case "Data":
      return <DataPage s={s} />;
    case "Logs":
      return <LogsPage s={s} />;
    default:
      return <MonitorPage s={s} chartImage={chartImage} chartHeight={chartHeight} />;
  }
}
