import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import type { Feed } from "../feed.ts";
import { WatchState } from "../state.ts";
import { ChartRenderer } from "../render.ts";
import { C } from "./theme.ts";
import { NAV, Sidebar, type NavItem } from "./Sidebar.tsx";
import { InsightsRail } from "./InsightsRail.tsx";
import { Page } from "./Pages.tsx";
import { Launch } from "./Launch.tsx";

const SIDEBAR_W = 14;
const RAIL_W = 26;

export interface AppProps {
  feed: Feed;
  feedPath: string;
  surogateBin: string;
  repoRoot: string;
}

export function App({ feed, feedPath, surogateBin, repoRoot }: AppProps) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const stateRef = useRef(new WatchState());
  const chartRef = useRef(new ChartRenderer());
  const [, force] = useState(0);
  const [navIdx, setNavIdx] = useState(0);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const cols = stdout?.columns ?? 120;
  const rows = stdout?.rows ?? 40;
  const mainW = Math.max(30, cols - SIDEBAR_W - RAIL_W - 2);
  const chartCols = Math.max(20, mainW - 4);
  const chartHeight = Math.max(6, rows - 18);
  const nav: NavItem = NAV[navIdx]!;

  useEffect(() => {
    feed.start((records) => {
      if (!pausedRef.current) {
        stateRef.current.ingest(records);
        force((n) => n + 1);
      }
    });
    return () => void feed.stop();
  }, [feed]);

  useEffect(() => {
    const t = setInterval(async () => {
      await chartRef.current.maybeRender(stateRef.current, chartCols, chartHeight, Date.now());
      force((n) => n + 1);
    }, 500);
    return () => clearInterval(t);
  }, [chartCols, chartHeight]);

  const onLaunch = nav === "Launch";
  useInput((input, key) => {
    if (input === "q") exit();
    else if (input === "p") setPaused((p) => !p);
    else if (!onLaunch && (key.upArrow || input === "k")) setNavIdx((i) => (i - 1 + NAV.length) % NAV.length);
    else if (!onLaunch && (key.downArrow || input === "j")) setNavIdx((i) => (i + 1) % NAV.length);
    else if (input >= "1" && input <= String(NAV.length)) setNavIdx(Number(input) - 1);
  });

  const s = stateRef.current;
  const model =
    s.model ??
    (typeof s.configFields["output_dir"] === "string" ? (s.configFields["output_dir"] as string).split("/").pop()! : "run");
  const recipe = s.recipe ?? "?";

  return (
    <Box flexDirection="column" width={cols} height={rows}>
      {/* top bar */}
      <Box justifyContent="space-between" paddingX={1}>
        <Text>
          <Text bold color={C.accent}>
            ◆ surogate
          </Text>
          <Text color={C.muted}> · </Text>
          <Text color={C.text}>
            {model} · {recipe}
          </Text>
          {s.lora && <Text color={C.accent}> · LoRA</Text>}
        </Text>
        <Text>
          {paused ? <Text color={C.warm}>‖ paused</Text> : <Text color={C.green}>● live</Text>}
          <Text color={C.dim}>{"  "}{feedPath}</Text>
        </Text>
      </Box>

      {/* middle: sidebar · page · rail */}
      <Box flexGrow={1}>
        <Sidebar active={nav} s={s} width={SIDEBAR_W} />
        <Box flexGrow={1} flexDirection="column" paddingX={1}>
          {nav === "Launch" ? (
            <Launch feedPath={feedPath} surogateBin={surogateBin} repoRoot={repoRoot} onLaunched={() => setNavIdx(0)} />
          ) : (
            <Page nav={nav} s={s} chartImage={chartRef.current.current()} chartHeight={chartHeight} />
          )}
        </Box>
        <InsightsRail s={s} width={RAIL_W} />
      </Box>

      {/* bottom bar */}
      <Box justifyContent="space-between" paddingX={1}>
        <Text color={C.muted}>
          <Hint k="q" label="quit" />
          <Hint k="↑↓" label="nav" />
          <Hint k="enter" label="select" />
          <Hint k="p" label="pause" />
        </Text>
        <Text color={C.dim}>
          {s.hasGpus ? `${s.gpusSorted().length} GPU` : "no GPU"} · {nav.toLowerCase()}
        </Text>
      </Box>
    </Box>
  );
}

function Hint({ k, label }: { k: string; label: string }) {
  return (
    <Text>
      <Text color={C.accent} bold>
        {k}
      </Text>
      <Text color={C.muted}> {label}{"   "}</Text>
    </Text>
  );
}
