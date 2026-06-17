import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { C } from "./theme.ts";

// Pick an existing training YAML: type a custom path (row 0, inline-editable) or
// ⏎ a discovered example. The chosen file is run as-is with a monitoring overlay.

const WINDOW = 12;

export function YamlPicker({
  examples,
  active,
  onPick,
}: {
  examples: string[];
  active: boolean;
  onPick: (path: string) => void;
}) {
  const [cursor, setCursor] = useState(0);
  const [editing, setEditing] = useState(false);
  const [buf, setBuf] = useState("");
  const last = examples.length; // cursor 0 = custom path, 1..N = examples

  useInput(
    (input, key) => {
      if (editing) {
        if (key.return) {
          if (buf.trim()) onPick(buf.trim());
          setEditing(false);
        } else if (key.escape) setEditing(false);
        else if (key.backspace || key.delete) setBuf((b) => b.slice(0, -1));
        else if (input && !key.ctrl && !key.meta && input >= " ") setBuf((b) => b + input);
        return;
      }
      if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
      else if (key.downArrow) setCursor((c) => Math.min(last, c + 1));
      else if (key.return) {
        if (cursor === 0) {
          setBuf("");
          setEditing(true);
        } else onPick(examples[cursor - 1]!);
      }
    },
    { isActive: active },
  );

  // window over [custom, ...examples]
  const start = Math.max(0, Math.min(cursor - Math.floor(WINDOW / 2), Math.max(0, last + 1 - WINDOW)));
  const idxs: number[] = [];
  for (let i = start; i < Math.min(last + 1, start + WINDOW); i++) idxs.push(i);

  return (
    <Box flexDirection="column">
      {start > 0 && <Text color={C.dim}>  ↑ more</Text>}
      {idxs.map((i) => {
        const on = i === cursor;
        if (i === 0) {
          return (
            <Box key="custom">
              <Text color={on ? C.accent : C.muted} bold={on}>
                {on ? "▸ " : "  "}✎ custom path{"  "}
              </Text>
              {editing && on ? (
                <Text color={C.gold}>
                  {buf}
                  <Text color={C.gold}>█</Text>
                </Text>
              ) : (
                <Text color={C.dim}>{on ? "⏎ to type a .yaml path" : "type a path"}</Text>
              )}
            </Box>
          );
        }
        const p = examples[i - 1]!;
        return (
          <Text key={p} color={on ? C.accent : C.text} bold={on} wrap="truncate-start">
            {on ? "▸ " : "  "}
            {p}
          </Text>
        );
      })}
      {start + WINDOW < last + 1 && <Text color={C.dim}>  ↓ more</Text>}
    </Box>
  );
}
