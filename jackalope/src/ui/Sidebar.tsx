import React from "react";
import { Box, Text } from "ink";
import type { WatchState } from "../state.ts";
import { C } from "./theme.ts";

// The nav is grouped into titled sections: a run you watch (Monitor), the pieces
// you assemble to start one (SET UP), and the artifacts a run leaves behind
// (HISTORY). NAV stays a flat ordered list (index = number shortcut) derived from
// the sections, so all the index-based logic elsewhere is unaffected.
const SECTIONS: { title?: string; items: readonly string[] }[] = [
  { items: ["Monitor"] },
  { title: "SET UP", items: ["GPUs", "Models", "Datasets", "Launch"] },
  { title: "HISTORY", items: ["Runs", "Logs", "Files"] },
];

export const NAV = SECTIONS.flatMap((s) => s.items) as readonly NavItem[] as readonly [NavItem, ...NavItem[]];
export type NavItem = "Monitor" | "GPUs" | "Runs" | "Models" | "Datasets" | "Launch" | "Logs" | "Files";

export function Sidebar({
  active,
  s,
  width,
  focusedNav = true,
}: {
  active: NavItem;
  s: WatchState;
  width: number;
  focusedNav?: boolean;
}) {
  return (
    <Box
      flexDirection="column"
      width={width}
      borderStyle="single"
      borderColor={C.border}
      borderTop={false}
      borderBottom={false}
      borderLeft={false}
      paddingX={1}
    >
      {SECTIONS.map((section, si) => (
        <Box key={si} flexDirection="column" marginTop={si > 0 ? 1 : 0}>
          {section.title && <Text color={C.dim}>{section.title}</Text>}
          {section.items.map((item) => {
            const on = item === active;
            const n = NAV.indexOf(item as NavItem) + 1; // global number shortcut
            return (
              <Text key={item} color={on ? C.accent : C.muted} bold={on}>
                {on ? (focusedNav ? "▸ " : "· ") : "  "}
                <Text color={on ? C.gold : C.dim}>{n}</Text> {item}
              </Text>
            );
          })}
        </Box>
      ))}
      <Box flexDirection="column" marginTop={1}>
        <Text color={C.dim}>RUN</Text>
        <Text color={C.text}>{s.recipe ?? "—"}</Text>
        {s.lora && <Text color={C.accent}>LoRA</Text>}
        <Text color={C.muted}>{s.maxSteps && s.maxSteps > 0 ? `step ${s.step}/${s.maxSteps}` : `step ${s.step}`}</Text>
      </Box>
    </Box>
  );
}
