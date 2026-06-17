import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { C } from "./theme.ts";

export interface GpuOption {
  id: number;
  label: string;
}

/**
 * A plain checkbox list for picking GPUs. The @inkjs/ui MultiSelect needed
 * `space` to tick a row and people kept pressing `⏎` (which submitted nothing),
 * so this widget is deliberately forgiving: `⏎` with nothing ticked just uses
 * the highlighted row. Checkboxes are drawn explicitly ([✓]/[ ]) so the current
 * selection is never ambiguous.
 */
export function GpuSelect({
  options,
  active,
  onSubmit,
  initialPicked,
}: {
  options: GpuOption[];
  active: boolean;
  onSubmit: (ids: number[]) => void;
  initialPicked?: number[];
}) {
  const [cursor, setCursor] = useState(0);
  // Pre-tick whatever was earmarked in the GPUs tab (filtered to GPUs present here).
  const [picked, setPicked] = useState<Set<number>>(
    () => new Set((initialPicked ?? []).filter((id) => options.some((o) => o.id === id))),
  );

  useInput(
    (input, key) => {
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      else if (key.downArrow) setCursor((c) => Math.min(options.length - 1, c + 1));
      else if (input === " ") {
        const id = options[cursor]?.id;
        if (id !== undefined)
          setPicked((s) => {
            const n = new Set(s);
            n.has(id) ? n.delete(id) : n.add(id);
            return n;
          });
      } else if (key.return) {
        // forgiving: nothing ticked → use the highlighted row
        const ids = picked.size ? [...picked] : options[cursor] ? [options[cursor]!.id] : [];
        if (ids.length) onSubmit(ids.sort((a, b) => a - b));
      }
    },
    { isActive: active },
  );

  return (
    <Box flexDirection="column">
      {options.map((o, i) => {
        const on = i === cursor;
        const checked = picked.has(o.id);
        return (
          <Text key={o.id} color={on ? C.accent : C.text} bold={on}>
            {on ? "❯ " : "  "}
            <Text color={checked ? C.green : C.dim}>{checked ? "[✓]" : "[ ]"}</Text> {o.label}
          </Text>
        );
      })}
    </Box>
  );
}
