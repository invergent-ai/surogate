import React from "react";
import { Box, Text } from "ink";
import type { WatchState } from "../state.ts";
import { C } from "./theme.ts";

export const NAV = ["Monitor", "GPUs", "Data", "Launch", "Logs"] as const;
export type NavItem = (typeof NAV)[number];

export function Sidebar({ active, s, width }: { active: NavItem; s: WatchState; width: number }) {
  return (
    <Box flexDirection="column" width={width} borderStyle="single" borderColor={C.border} borderTop={false} borderBottom={false} borderLeft={false} paddingX={1}>
      <Text color={C.dim}>NAV</Text>
      {NAV.map((item) => {
        const on = item === active;
        return (
          <Text key={item} color={on ? C.accent : C.text} bold={on}>
            {on ? "▸ " : "  "}
            {item}
          </Text>
        );
      })}
      <Box flexDirection="column" marginTop={1}>
        <Text color={C.dim}>RUN</Text>
        <Text color={C.muted}>{s.recipe ?? "—"}</Text>
        {s.lora && <Text color={C.accent}>LoRA</Text>}
        {s.maxSteps && s.maxSteps > 0 ? (
          <Text color={C.muted}>
            {s.step}/{s.maxSteps}
          </Text>
        ) : (
          <Text color={C.muted}>step {s.step}</Text>
        )}
      </Box>
    </Box>
  );
}
