import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { C } from "./theme.ts";

// A reusable, grouped, scrollable editor over a flat values map. Navigate ↑↓; ⏎
// edits a text/number field inline, space toggles a switch, ←→ cycles an enum.
// Used for both the SFT and GRPO parameter forms (each passes its own schema).

export type FieldKind = "text" | "num" | "bool" | "enum";
export type Values = Record<string, string | boolean>;

export interface FieldDef {
  group: string;
  key: string;
  label: string;
  kind: FieldKind;
  options?: readonly string[];
  show?: (v: Values) => boolean;
  // The "why" shown under the focused row: per-option one-liner for enums,
  // or a single line for other kinds.
  desc?: Record<string, string>;
  help?: string;
}

const VISIBLE_ROWS = 16;

export function FieldEditor<T extends object>({
  schema,
  values,
  setValues,
  active,
  onDone,
  onReset,
  doneLabel = "review & launch",
}: {
  schema: FieldDef[];
  values: T;
  setValues: (v: T) => void;
  active: boolean;
  onDone: () => void;
  onReset?: () => void;
  doneLabel?: string;
}) {
  // T is a concrete struct (LaunchFields/GrpoFields) with no index signature, so
  // we keep one local Record view for the schema-driven dynamic key access.
  const vals = values as Values;
  const visible = schema.filter((f) => !f.show || f.show(vals));
  const launchRow = visible.length;

  const [cursorRaw, setCursor] = useState(0);
  const [editing, setEditing] = useState(false);
  const [buf, setBuf] = useState("");
  const cursor = Math.min(cursorRaw, launchRow);

  const set = (key: string, value: string | boolean) => setValues({ ...values, [key]: value } as T);
  const cycle = (f: FieldDef, dir: 1 | -1) => {
    const opts = f.options!;
    const i = opts.indexOf(String(vals[f.key]));
    // unknown value (not in options) → land on the first option either direction
    const next = i < 0 ? 0 : (i + dir + opts.length) % opts.length;
    set(f.key, opts[next]!);
  };

  useInput(
    (input, key) => {
      if (editing) {
        if (key.return) {
          const f = visible[cursor]!;
          const v = buf.trim();
          // keep the previous value rather than writing an empty number into the
          // YAML (a bare `max_steps:` parses as null and crashes the run)
          if (!(f.kind === "num" && v === "")) set(f.key, v);
          setEditing(false);
        } else if (key.escape) {
          setEditing(false);
        } else if (key.backspace || key.delete) {
          setBuf((b) => b.slice(0, -1));
        } else if (input && !key.ctrl && !key.meta && input >= " ") {
          setBuf((b) => b + input);
        }
        return;
      }
      if (input === "g") return onDone(); // launch now from any field (no scrolling)
      if (input === "r" && onReset) return onReset(); // reset all fields to defaults
      if (key.upArrow) return setCursor(Math.max(0, cursor - 1));
      if (key.downArrow) return setCursor(Math.min(launchRow, cursor + 1));
      if (cursor === launchRow) {
        if (key.return) onDone();
        return;
      }
      const f = visible[cursor]!;
      if (f.kind === "bool") {
        if (input === " " || key.return) set(f.key, !vals[f.key]);
      } else if (f.kind === "enum") {
        if (key.leftArrow) cycle(f, -1);
        else if (key.rightArrow || input === " " || key.return) cycle(f, 1);
      } else if (key.return) {
        setBuf(String(vals[f.key]));
        setEditing(true);
      }
    },
    { isActive: active },
  );

  type Row = { t: "header"; label: string } | { t: "field"; f: FieldDef; i: number } | { t: "done" };
  const rows: Row[] = [];
  let lastGroup = "";
  visible.forEach((f, i) => {
    if (f.group !== lastGroup) {
      rows.push({ t: "header", label: f.group });
      lastGroup = f.group;
    }
    rows.push({ t: "field", f, i });
  });
  rows.push({ t: "done" });
  const cursorDisp = rows.findIndex((r) => (r.t === "field" && r.i === cursor) || (r.t === "done" && cursor === launchRow));
  let start = Math.max(0, cursorDisp - Math.floor(VISIBLE_ROWS / 2));
  start = Math.min(start, Math.max(0, rows.length - VISIBLE_ROWS));
  const windowed = rows.slice(start, start + VISIBLE_ROWS);

  const valueText = (f: FieldDef) => {
    if (f.kind === "bool") return vals[f.key] ? <Text color={C.green}>on</Text> : <Text color={C.dim}>off</Text>;
    return <Text color={C.text}>{String(vals[f.key]) || "—"}</Text>;
  };
  const hint = (f: FieldDef) => (f.kind === "bool" ? "space" : f.kind === "enum" ? "←→" : "⏎ edit");
  // The "why" line shown under the focused field (enum: per-value; else help).
  const descFor = (f: FieldDef): string | undefined =>
    f.kind === "enum" ? f.desc?.[String(vals[f.key])] : f.help;

  return (
    <Box flexDirection="column">
      <Text>
        <Text color={C.gold} bold>
          g
        </Text>
        <Text color={C.dim}> {doneLabel.toLowerCase()} now (uses these settings) · ↑↓ fields · ⏎ edit{onReset ? " · r reset" : ""}</Text>
      </Text>
      {start > 0 && <Text color={C.dim}>  ↑ more</Text>}
      {windowed.map((r, idx) => {
        if (r.t === "header") return <Text key={`h${idx}`} color={C.dim}>{`  ${r.label.toUpperCase()}`}</Text>;
        if (r.t === "done") {
          const on = cursor === launchRow;
          return (
            <Text key="done" color={on ? C.gold : C.muted} bold={on}>
              {on ? "▸ " : "  "}▶ {doneLabel}
            </Text>
          );
        }
        const on = r.i === cursor;
        const desc = on && !editing ? descFor(r.f) : undefined;
        return (
          <Box key={r.f.key} flexDirection="column">
            <Box>
              <Text color={on ? C.accent : C.muted} bold={on}>
                {on ? "▸ " : "  "}
                {r.f.label.padEnd(20)}
              </Text>
              {editing && on ? (
                <Text color={C.gold}>
                  {buf}
                  <Text color={C.gold}>█</Text>
                </Text>
              ) : (
                <>
                  {valueText(r.f)}
                  {on && <Text color={C.dim}>{"   "}{hint(r.f)}</Text>}
                </>
              )}
            </Box>
            {desc && <Text color={C.dim}>{"     ↳ "}{desc}</Text>}
          </Box>
        );
      })}
      {start + VISIBLE_ROWS < rows.length && <Text color={C.dim}>  ↓ more</Text>}
      {onReset && <Text color={C.dim}>  r reset to defaults</Text>}
    </Box>
  );
}
