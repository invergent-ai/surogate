import React from "react";
import { Box, Text } from "ink";
import type { WatchState } from "../state.ts";
import { fmtCount, fmtDuration, fmtEta, fmtFloat } from "../format.ts";
import { C, memColor, meterParts, tempColor } from "./theme.ts";
import { Panel } from "./Panel.tsx";

export function Chart({ image }: { image: string }) {
  if (!image) return <Text color={C.muted}>waiting for loss data…</Text>;
  return <Text>{image}</Text>;
}

export function Runline({ s }: { s: WatchState }) {
  const total = s.maxSteps && s.maxSteps > 0 ? ` / ${s.maxSteps}` : "";
  return (
    <Text>
      <Text color={C.muted}>step </Text>
      <Text bold color={C.text}>
        {s.step}
        {total}
      </Text>
      {s.epoch !== null && (
        <Text>
          <Text color={C.muted}>{"    epoch "}</Text>
          <Text color={C.text}>{s.epoch.toFixed(2)}</Text>
        </Text>
      )}
      {s.lora && <Text color={C.accent}>{"    LoRA"}</Text>}
      <Text color={C.muted}>{"    ETA "}</Text>
      <Text bold color={C.text}>
        {fmtEta(s.etaSeconds())}
      </Text>
      <Text color={C.muted}>{"    elapsed "}</Text>
      <Text color={C.text}>{fmtDuration(s.elapsedSeconds())}</Text>
    </Text>
  );
}

function VRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Box justifyContent="space-between">
      <Text color={C.muted}>{label}</Text>
      <Text>{children}</Text>
    </Box>
  );
}

export function Vitals({ s }: { s: WatchState }) {
  const trend = s.lossTrend();
  const arrow = trend < 0 ? "▼" : trend > 0 ? "▲" : "•";
  const arrowColor = trend < 0 ? C.green : trend > 0 ? C.red : C.muted;
  return (
    <Box flexDirection="column" marginTop={1}>
      <VRow label="train loss">
        <Text bold color={C.train}>
          {fmtFloat(s.latestTrainLoss)}
        </Text>
        <Text color={arrowColor}> {arrow}</Text>
      </VRow>
      {s.latestEvalLoss !== null && (
        <VRow label="eval loss">
          <Text bold color={C.eval}>
            {fmtFloat(s.latestEvalLoss)}
          </Text>
        </VRow>
      )}
      <VRow label="learning rate">
        <Text color={C.text}>{fmtFloat(s.lr)}</Text>
      </VRow>
      <VRow label="grad norm">
        <Text color={C.text}>{fmtFloat(s.gradNorm)}</Text>
      </VRow>
      <VRow label="throughput">
        <Text color={C.text}>{fmtCount(s.tokensPerSecond)} tok/s</Text>
      </VRow>
      {s.phase && (
        <VRow label="phase">
          <Text color={C.warm}>{s.phase}</Text>
        </VRow>
      )}
    </Box>
  );
}

function Meter({ frac, color }: { frac: number | null; color: string }) {
  const { filled, track, pct } = meterParts(frac);
  return (
    <Text>
      <Text color={color}>{"█".repeat(filled)}</Text>
      <Text color={C.dim}>{"█".repeat(track)}</Text>
      <Text color={C.text}> {pct}</Text>
    </Text>
  );
}

export function Gpus({ s }: { s: WatchState }) {
  if (!s.hasGpus) {
    return (
      <Box marginTop={1}>
        <Text color={C.muted}>no GPU telemetry yet</Text>
      </Box>
    );
  }
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color={C.muted}>{"#  device      temp   power   sm util          memory"}</Text>
      </Box>
      {s.gpusSorted().map((g) => {
        const name = (g.name ?? `gpu${g.gpuId}`).slice(0, 10).padEnd(10);
        const temp = g.temp === null ? "  —" : `${Math.round(g.temp)}°`.padStart(3);
        const pwr = g.power === null ? "   —" : `${Math.round(g.power)}W`.padStart(5);
        return (
          <Box key={g.gpuId}>
            <Text color={C.dim}>{String(g.gpuId).padEnd(3)}</Text>
            <Text color={C.text}>{name} </Text>
            <Text color={tempColor(g.temp)}>{temp}</Text>
            <Text color={C.text}>{"  " + pwr + "  "}</Text>
            <Meter frac={g.smUtil === null ? null : g.smUtil / 100} color={C.accent} />
            <Text> </Text>
            <Meter frac={g.memUtil} color={g.memUtil === null ? C.muted : memColor(g.memUtil)} />
          </Box>
        );
      })}
    </Box>
  );
}

export function LogTail({ s }: { s: WatchState }) {
  const recent = s.recentSteps.slice(-3);
  if (recent.length === 0) return <Text color={C.muted}>waiting for metrics…</Text>;
  return (
    <Box flexDirection="column">
      {recent.map((r, i) => (
        <Text key={i} color={C.text}>
          step {r.step}
          {r.trainLoss !== null ? `  loss ${fmtFloat(r.trainLoss)}` : ""}
          {r.evalLoss !== null ? `  eval ${fmtFloat(r.evalLoss)}` : ""}
          {r.lr !== null ? `  lr ${fmtFloat(r.lr)}` : ""}
          {r.tokensPerSecond !== null ? `  ${fmtCount(r.tokensPerSecond)} tok/s` : ""}
        </Text>
      ))}
    </Box>
  );
}

export function Monitor({ s, chartImage, chartHeight }: { s: WatchState; chartImage: string; chartHeight: number }) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title="run">
        <Runline s={s} />
      </Panel>
      <Panel title="loss" flexGrow={1} minHeight={chartHeight + 3}>
        <Chart image={chartImage} />
      </Panel>
      <Box>
        <Panel title="vitals" width={38}>
          <Vitals s={s} />
        </Panel>
        <Box flexGrow={1}>
          <Panel title="devices" flexGrow={1}>
            <Gpus s={s} />
          </Panel>
        </Box>
      </Box>
      <Panel title="log">
        <LogTail s={s} />
      </Panel>
    </Box>
  );
}
