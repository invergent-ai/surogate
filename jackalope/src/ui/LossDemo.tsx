import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { C, getTheme } from "./theme.ts";
import { goldAt } from "./brand.ts";

// A self-contained, looping "what a training run looks like" animation for the
// welcome screen: a synthetic loss curve descends and draws in left→right, the
// leading edge pulses, and the loss readout ticks down. No real data — purely
// decorative, so it conveys "this watches training" at a glance.

const W = 34; // columns
const H = 7; // rows tall
const PARTIAL = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
const HI = 1.95; // loss at step 0
const LO = 0.38; // loss at the end
const STEP_MS = 130;

// Deterministic curve: exponential decay with a little wobble (seeded by index,
// so it's stable frame-to-frame — only the reveal cursor moves).
const SERIES = Array.from({ length: W }, (_, i) => {
  const t = i / (W - 1);
  const decay = Math.exp(-i / 9);
  const wobble = 0.05 * Math.sin(i * 1.7) * (1 - t);
  return Math.max(0, Math.min(1, decay + wobble));
});

export function LossDemo() {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    // draw once, then hold the finished chart — the carousel remounts this scene
    // each cycle, so it replays cleanly without restarting mid-view.
    const id = setInterval(() => setFrame((f) => (f >= W - 1 ? f : f + 1)), STEP_MS);
    return () => clearInterval(id);
  }, []);

  const cursor = Math.min(frame, W - 1);
  const loss = LO + (HI - LO) * SERIES[cursor]!;
  const leadColor = getTheme() === "light" ? "#000000" : "#ffffff"; // pulse, readable per theme

  // Precompute each column's bar height (in rows) once per frame.
  const heights = SERIES.map((v) => v * (H - 0.001));

  const rows: React.ReactNode[] = [];
  for (let r = 0; r < H; r++) {
    const cells: React.ReactNode[] = [];
    const fromBottom = H - r; // row r holds heights in (fromBottom-1, fromBottom]
    for (let c = 0; c < W; c++) {
      if (c > cursor) {
        cells.push(
          <Text key={c} color={C.dim}>
            {r === H - 1 ? "·" : " "}
          </Text>,
        );
        continue;
      }
      const h = heights[c]!;
      let ch = " ";
      if (h >= fromBottom) ch = "█";
      else if (h > fromBottom - 1) ch = PARTIAL[Math.max(0, Math.min(7, Math.floor((h - (fromBottom - 1)) * 8)))]!;
      const lead = c === cursor;
      cells.push(
        <Text key={c} color={lead ? leadColor : goldAt(r, H)} bold={lead}>
          {ch}
        </Text>,
      );
    }
    rows.push(<Text key={r}>{cells}</Text>);
  }

  return (
    <Box flexDirection="column">
      {rows}
      <Box marginTop={1}>
        <Text color={C.muted}>loss </Text>
        <Text color={C.train} bold>
          {loss.toFixed(2)}
        </Text>
        <Text color={C.green}> ▼</Text>
        <Text color={C.dim}>{`   step ${cursor + 1}/${W}`}</Text>
      </Box>
    </Box>
  );
}
