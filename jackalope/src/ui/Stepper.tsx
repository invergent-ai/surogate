import React from "react";
import { Box, Text } from "ink";
import { C } from "./theme.ts";

// A horizontal wizard progress rail (PostHog-style): done steps get a ✓, the
// current step a filled number + bold label, upcoming steps a dim number, joined
// by thin connectors.
const CIRCLED = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"];
const glyph = (i: number) => CIRCLED[i] ?? `(${i + 1})`;

export function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (
    <Box>
      {steps.map((label, i) => {
        const done = i < current;
        const cur = i === current;
        const color = done ? C.green : cur ? C.accent : C.dim;
        return (
          <Text key={i}>
            {i > 0 ? <Text color={C.dim}>{" ── "}</Text> : null}
            <Text color={color} bold={cur}>
              {done ? "✓" : glyph(i)} {label}
            </Text>
          </Text>
        );
      })}
    </Box>
  );
}
