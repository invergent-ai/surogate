import React from "react";
import { Box, Text } from "ink";
import type { WatchState } from "../state.ts";
import { computeInsights, type Insight, type InsightColor } from "../insights.ts";
import { C } from "./theme.ts";

const COLOR: Record<InsightColor, string> = {
  green: C.green,
  warm: C.warm,
  red: C.red,
  accent: C.accent,
  eval: C.eval,
  muted: C.muted,
};

function Row({ ins }: { ins: Insight }) {
  return (
    <Text color={COLOR[ins.color]}>
      {ins.icon} <Text color={ins.color === "muted" ? C.muted : C.text}>{ins.text}</Text>
    </Text>
  );
}

function Section({ title, items }: { title: string; items: Insight[] }) {
  if (items.length === 0) return null;
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text color={C.dim}>{title}</Text>
      {items.map((ins, i) => (
        <Row key={i} ins={ins} />
      ))}
    </Box>
  );
}

export function InsightsRail({ s, width }: { s: WatchState; width: number }) {
  const g = computeInsights(s, Date.now());
  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="single"
      borderColor={C.border}
      borderTop={false}
      borderBottom={false}
      borderRight={false}
      paddingX={1}
    >
      <Text color={C.accent} bold>
        INSIGHTS
      </Text>
      <Box marginTop={1} flexDirection="column">
        <Section title="HEALTH" items={g.health} />
        <Section title="DEVICES" items={g.alerts} />
        <Section title="TIPS" items={g.tips} />
        <Section title="RUN" items={g.facts} />
      </Box>
    </Box>
  );
}
