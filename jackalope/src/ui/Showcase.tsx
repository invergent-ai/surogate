import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { C } from "./theme.ts";
import { goldAt } from "./brand.ts";
import { LossDemo } from "./LossDemo.tsx";
import { FAMILIES } from "../supported.ts";
import { GPU_GENS } from "../gpu-support.ts";

// A rotating, self-animating "what training looks like" panel — modeled on the
// way PostHog's wizard cycles showcase content on its side panel. Each scene
// animates on its own; the carousel advances every few seconds with a title and
// progress dots. All data is synthetic/illustrative — pure decoration.

function useFrame(ms: number): number {
  const [f, setF] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setF((x) => x + 1), ms);
    return () => clearInterval(id);
  }, [ms]);
  return f;
}

const RAMP = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

// Scene: GPU utilization — four bars pulsing high, as during a real run.
function GpuScene() {
  const f = useFrame(160);
  const W = 16;
  return (
    <Box flexDirection="column">
      {[0, 1, 2, 3].map((g) => {
        const util = Math.max(0, Math.min(100, 86 + 12 * Math.sin(f * 0.28 + g * 1.25)));
        const filled = Math.round((util / 100) * W);
        return (
          <Text key={g}>
            <Text color={C.dim}>gpu{g} </Text>
            <Text color={C.green}>{"█".repeat(filled)}</Text>
            <Text color={C.dim}>{"░".repeat(W - filled)}</Text>
            <Text color={C.text}>{` ${Math.round(util)}%`}</Text>
          </Text>
        );
      })}
      <Box marginTop={1}>
        <Text color={C.muted}>4× GPU · sharded with ZeRO-1</Text>
      </Box>
    </Box>
  );
}

// Scene: throughput — a scrolling sparkline + a tok/s readout that ticks.
function ThroughputScene() {
  const f = useFrame(140);
  const N = 30;
  const spark = Array.from({ length: N }, (_, i) => {
    const x = f - (N - 1 - i);
    return Math.max(0.05, Math.min(1, 0.58 + 0.16 * Math.sin(x * 0.24) + 0.08 * Math.sin(x * 1.3)));
  });
  const cur = Math.round(11200 + 2800 * spark[N - 1]!);
  return (
    <Box flexDirection="column">
      <Text>
        {spark.map((v, i) => (
          <Text key={i} color={goldAt(Math.floor((1 - v) * (H_ROWS - 1)), H_ROWS)}>
            {RAMP[Math.min(7, Math.floor(v * 8))]}
          </Text>
        ))}
      </Text>
      <Box marginTop={1}>
        <Text color={C.muted}>throughput </Text>
        <Text color={C.train} bold>
          {cur.toLocaleString()}
        </Text>
        <Text color={C.muted}> tok/s</Text>
      </Box>
      <Text color={C.dim}>sample packing · fused cross-entropy</Text>
    </Box>
  );
}
const H_ROWS = 8;

// Scene: precision speedup — bars grow in, comparing recipe throughput.
function SpeedupScene() {
  const f = useFrame(120);
  const W = 16;
  const grow = Math.min(1, (f % 46) / 28); // fill 0→1, hold, then loop
  const rows: { name: string; x: number; c: string }[] = [
    { name: "bf16", x: 1.0, c: C.accent },
    { name: "fp8-hybrid", x: 1.9, c: C.green },
    { name: "NVFP4 4-bit", x: 2.6, c: C.eval },
  ];
  const maxX = 2.6;
  return (
    <Box flexDirection="column">
      {rows.map((r) => {
        const filled = Math.round((r.x / maxX) * grow * W);
        return (
          <Text key={r.name}>
            <Text color={C.text}>{r.name.padEnd(12)}</Text>
            <Text color={r.c}>{"█".repeat(filled)}</Text>
            <Text color={C.dim}>{"░".repeat(W - filled)}</Text>
            <Text color={C.muted}>{` ${(r.x * grow).toFixed(1)}×`}</Text>
          </Text>
        );
      })}
      <Box marginTop={1}>
        <Text color={C.dim}>relative throughput · illustrative</Text>
      </Box>
    </Box>
  );
}

// Scene: the supported model families, scrolling with a moving highlight, with
// MoE/vision badges. Shows breadth + how to request a missing one.
function ModelsScene() {
  const f = useFrame(750);
  const N = FAMILIES.length;
  const win = Math.min(6, N);
  const start = f % N;
  const rows = Array.from({ length: win }, (_, i) => FAMILIES[(start + i) % N]!);
  return (
    <Box flexDirection="column">
      {rows.map((fam, i) => {
        const on = i === 0;
        return (
          <Text key={fam.type} color={on ? C.accent : C.text} bold={on}>
            {on ? "▸ " : "  "}
            {fam.label.padEnd(16)}
            <Text color={fam.moe ? C.warm : C.dim}>{fam.moe ? "MoE " : "    "}</Text>
            <Text color={fam.vision ? C.eval : C.dim}>{fam.vision ? "vision" : ""}</Text>
          </Text>
        );
      })}
      <Box marginTop={1}>
        <Text color={C.muted}>{`${N} architectures · missing one? `}</Text>
        <Text color={C.eval}>request it in Models</Text>
      </Box>
    </Box>
  );
}

// Scene: which GPU generations surogate supports + the recipes they unlock.
function SupportedGpusScene() {
  const recipeTag = (recipes: string[]) =>
    recipes.includes("nvfp4") ? "bf16 · fp8 · NVFP4" : recipes.includes("fp8-hybrid") ? "bf16 · fp8-hybrid" : "bf16 · QLoRA";
  return (
    <Box flexDirection="column">
      {GPU_GENS.map((g) => {
        const top = g.recipes.includes("nvfp4");
        return (
          <Text key={g.arch}>
            <Text color={top ? C.accent : C.text} bold={top}>
              {g.arch.padEnd(15)}
            </Text>
            <Text color={top ? C.gold : C.muted}>{recipeTag(g.recipes)}</Text>
          </Text>
        );
      })}
      <Box marginTop={1}>
        <Text color={C.muted}>precision gated by GPU arch · </Text>
        <Text color={C.eval}>auto-detected</Text>
      </Box>
    </Box>
  );
}

const SCENES: { title: string; el: React.ReactNode }[] = [
  { title: "TRAINING LOSS", el: <LossDemo /> },
  { title: "SUPPORTED MODELS", el: <ModelsScene /> },
  { title: "SUPPORTED GPUS", el: <SupportedGpusScene /> },
  { title: "GPU UTILIZATION", el: <GpuScene /> },
  { title: "THROUGHPUT", el: <ThroughputScene /> },
  { title: "PRECISION SPEEDUP", el: <SpeedupScene /> },
];

export function Showcase() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((x) => (x + 1) % SCENES.length), 6500);
    return () => clearInterval(id);
  }, []);
  const scene = SCENES[i]!;
  return (
    <Box flexDirection="column">
      <Text color={C.accent} bold>
        {scene.title}
      </Text>
      <Box marginTop={1} minHeight={9}>
        {scene.el}
      </Box>
      <Box>
        {SCENES.map((_, k) => (
          <Text key={k} color={k === i ? C.accent : C.dim}>
            {k === i ? "●" : "○"}{" "}
          </Text>
        ))}
      </Box>
    </Box>
  );
}
