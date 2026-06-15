import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import { Feed } from "../feed.ts";
import { WatchState } from "../state.ts";
import { ChartRenderer } from "../render.ts";
import { listRuns, type RunInfo } from "../runs.ts";
import { C } from "./theme.ts";
import { NAV, Sidebar, type NavItem } from "./Sidebar.tsx";
import { InsightsRail } from "./InsightsRail.tsx";
import { Page } from "./Pages.tsx";
import { Launch } from "./Launch.tsx";
import { StartScreen } from "./StartScreen.tsx";

const SIDEBAR_W = 14;
const RAIL_W = 26;

export interface AppProps {
  initialFeedPath: string;
  fromStart: boolean;
  surogateBin: string;
  repoRoot: string;
  version: string;
}

interface FeedDesc {
  path: string;
  fromStart: boolean;
}

export function App({ initialFeedPath, fromStart, surogateBin, repoRoot, version }: AppProps) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const stateRef = useRef(new WatchState());
  const chartRef = useRef(new ChartRenderer());
  const feedRef = useRef<Feed | null>(null);
  const [, force] = useState(0);
  const rerender = () => force((n) => n + 1);

  const [started, setStarted] = useState(false);
  const [feedDesc, setFeedDesc] = useState<FeedDesc>({ path: initialFeedPath, fromStart });
  const [navIdx, setNavIdx] = useState(0);
  const [focus, setFocus] = useState<"nav" | "content">("nav");
  const [runSel, setRunSel] = useState(0);
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const cols = stdout?.columns ?? 120;
  const rows = stdout?.rows ?? 40;
  const mainW = Math.max(30, cols - SIDEBAR_W - RAIL_W - 2);
  const chartCols = Math.max(20, mainW - 4);
  const chartHeight = Math.max(6, rows - 18);
  const nav: NavItem = NAV[navIdx]!;

  // (re)create the feed whenever the watched feed path changes — resets state so
  // runs never bleed into each other.
  useEffect(() => {
    let cancelled = false;
    void feedRef.current?.stop();
    stateRef.current = new WatchState();
    chartRef.current = new ChartRenderer();
    const feed = new Feed(feedDesc.path, feedDesc.fromStart);
    feedRef.current = feed;
    feed.start((records) => {
      if (!cancelled && !pausedRef.current) {
        stateRef.current.ingest(records);
        rerender();
      }
    });
    rerender();
    return () => {
      cancelled = true;
      void feed.stop();
    };
  }, [feedDesc]);

  useEffect(() => {
    const t = setInterval(async () => {
      if (started) await chartRef.current.maybeRender(stateRef.current, chartCols, chartHeight, Date.now());
      rerender();
    }, 500);
    return () => clearInterval(t);
  }, [chartCols, chartHeight, started]);

  const switchFeed = (p: string) => {
    setFeedDesc({ path: p, fromStart: true });
    setFocus("nav");
    setNavIdx(0);
  };

  const runs: RunInfo[] = nav === "Runs" ? listRuns([initialFeedPath, feedDesc.path], Date.now()) : [];

  useInput((input, key) => {
    if (input === "q") return exit();

    if (!started) {
      if (key.return || input === " ") setStarted(true);
      return;
    }

    if (input === "p") {
      setPaused((x) => !x);
      return;
    }

    if (focus === "content") {
      if (key.escape || key.leftArrow) {
        setFocus("nav");
        return;
      }
      if (nav === "Runs") {
        if (key.upArrow || input === "k") setRunSel((i) => Math.max(0, i - 1));
        else if (key.downArrow || input === "j") setRunSel((i) => Math.min(Math.max(0, runs.length - 1), i + 1));
        else if (key.return && runs[runSel]) switchFeed(runs[runSel]!.path);
      }
      return; // Launch widgets handle their own keys
    }

    // focus === "nav"
    if (key.upArrow || input === "k") setNavIdx((i) => (i - 1 + NAV.length) % NAV.length);
    else if (key.downArrow || input === "j") setNavIdx((i) => (i + 1) % NAV.length);
    else if (input >= "1" && input <= String(NAV.length)) setNavIdx(Number(input) - 1);
    else if ((key.return || key.rightArrow) && (nav === "Launch" || nav === "Runs")) setFocus("content");
  });

  const s = stateRef.current;

  if (!started) {
    return (
      <Box width={cols} height={rows}>
        <StartScreen feedPath={feedDesc.path} version={version} />
      </Box>
    );
  }

  const model =
    s.model ??
    (typeof s.configFields["output_dir"] === "string" ? (s.configFields["output_dir"] as string).split("/").pop()! : "run");
  const recipe = s.recipe ?? "?";
  const launchActive = nav === "Launch" && focus === "content";
  const runsActive = nav === "Runs" && focus === "content";

  return (
    <Box flexDirection="column" width={cols} height={rows}>
      <Box justifyContent="space-between" paddingX={1}>
        <Text>
          <Text bold color={C.accent}>
            surogate
          </Text>
          <Text color={C.muted}> · </Text>
          <Text color={C.text}>
            {model} · {recipe}
          </Text>
          {s.lora && <Text color={C.accent}> · LoRA</Text>}
        </Text>
        <Text>
          {paused ? <Text color={C.warm}>‖ paused</Text> : <Text color={C.green}>● live</Text>}
          <Text color={C.dim}>{"  "}{feedDesc.path}</Text>
        </Text>
      </Box>

      <Box flexGrow={1}>
        <Sidebar active={nav} s={s} width={SIDEBAR_W} focusedNav={focus === "nav"} />
        <Box flexGrow={1} flexDirection="column" paddingX={1}>
          {nav === "Launch" ? (
            <Launch
              feedPath={feedDesc.path}
              surogateBin={surogateBin}
              repoRoot={repoRoot}
              active={launchActive}
              onLaunched={(metricsPath) => switchFeed(metricsPath)}
            />
          ) : (
            <Page
              nav={nav}
              s={s}
              chartImage={chartRef.current.current()}
              chartHeight={chartHeight}
              runs={runs}
              runSel={runSel}
              runsActive={runsActive}
              currentFeed={feedDesc.path}
            />
          )}
        </Box>
        <InsightsRail s={s} width={RAIL_W} />
      </Box>

      <Box justifyContent="space-between" paddingX={1}>
        <Text color={C.muted}>
          {launchActive || runsActive ? (
            <>
              <Hint k="esc" label="back" />
              <Hint k="↑↓/⏎" label={runsActive ? "watch run" : "form"} />
            </>
          ) : (
            <>
              <Hint k="q" label="quit" />
              <Hint k="↑↓" label="nav" />
              <Hint k="⏎" label={nav === "Launch" ? "configure" : nav === "Runs" ? "pick" : "select"} />
              <Hint k="p" label="pause" />
            </>
          )}
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
