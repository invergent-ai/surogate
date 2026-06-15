import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import { Feed } from "../feed.ts";
import { WatchState } from "../state.ts";
import { ChartRenderer, type CompareRun } from "../render.ts";
import { listRuns, type RunInfo } from "../runs.ts";
import { AlertEngine, desktopNotify } from "../alerts.ts";
import { pauseRun, resumeRun, runControllable, stopRun } from "../controls.ts";
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
  const alertRef = useRef(new AlertEngine());
  const compareRef = useRef<CompareRun | null>(null);
  const suspendedRef = useRef(new Set<string>());
  const bannerTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [, force] = useState(0);
  const rerender = () => force((n) => n + 1);

  const [started, setStarted] = useState(false);
  const [feedDesc, setFeedDesc] = useState<FeedDesc>({ path: initialFeedPath, fromStart });
  const [navIdx, setNavIdx] = useState(0);
  const [focus, setFocus] = useState<"nav" | "content">("nav");
  const [runSel, setRunSel] = useState(0);
  const [paused, setPaused] = useState(false);
  const [banner, setBannerState] = useState<{ text: string; color: string } | null>(null);
  const [compareFeed, setCompareFeed] = useState<string | null>(null);
  const [stopArmed, setStopArmed] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const flashBanner = (text: string, color: string, ms = 8000) => {
    setBannerState({ text, color });
    if (bannerTimer.current) clearTimeout(bannerTimer.current);
    bannerTimer.current = setTimeout(() => setBannerState(null), ms);
  };

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

  // load the compare run's loss curve (snapshot) when pinned
  useEffect(() => {
    if (!compareFeed) {
      compareRef.current = null;
      return;
    }
    let cancelled = false;
    const label = compareFeed.split("/").pop()!.replace(/\.jsonl$/, "");
    void new Feed(compareFeed, true).snapshot().then((records) => {
      if (cancelled) return;
      const cs = new WatchState();
      cs.ingest(records);
      compareRef.current = { label, steps: cs.lossSteps, train: cs.lossHistory };
      rerender();
    });
    return () => {
      cancelled = true;
    };
  }, [compareFeed]);

  useEffect(() => {
    const t = setInterval(async () => {
      if (!started) return rerender();
      await chartRef.current.maybeRender(stateRef.current, chartCols, chartHeight, Date.now(), compareRef.current ?? undefined);
      // alerts (bell + banner + best-effort desktop notification)
      const a = alertRef.current.check(stateRef.current, feedDesc.path, Date.now());
      if (a) {
        process.stdout.write("\x07");
        flashBanner(`⚑ ${a.text}`, a.color === "red" ? C.red : a.color === "green" ? C.green : C.warm);
        desktopNotify("jackrabbit", a.text);
      }
      rerender();
    }, 500);
    return () => clearInterval(t);
  }, [chartCols, chartHeight, started, feedDesc.path]);

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

    if (stopArmed && input !== "x") setStopArmed(false);

    if (input === "p") {
      setPaused((x) => !x);
      return;
    }

    // pause / resume the current launched run (SIGSTOP / SIGCONT)
    if (input === "z" && !(nav === "Launch" && focus === "content")) {
      if (!runControllable(feedDesc.path)) {
        flashBanner("no controllable run on this feed", C.muted, 3000);
        return;
      }
      const set = suspendedRef.current;
      if (set.has(feedDesc.path)) {
        if (resumeRun(feedDesc.path)) set.delete(feedDesc.path);
        flashBanner("▶ resumed", C.green, 3000);
      } else {
        if (pauseRun(feedDesc.path)) set.add(feedDesc.path);
        flashBanner("⏸ run paused (SIGSTOP) — z to resume", C.warm, 5000);
      }
      return;
    }

    // stop the current launched run (two-press confirm) — except inside the Launch form
    if (input === "x" && !(nav === "Launch" && focus === "content")) {
      if (!runControllable(feedDesc.path)) {
        flashBanner("no controllable run on this feed", C.muted, 3000);
        return;
      }
      if (!stopArmed) {
        setStopArmed(true);
        flashBanner("press x again to STOP this run", C.warm, 4000);
      } else {
        const ok = stopRun(feedDesc.path);
        setStopArmed(false);
        flashBanner(ok ? "⏹ stopping run…" : "stop failed", ok ? C.warm : C.red, 5000);
      }
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
        else if (input === "c" && runs[runSel]) {
          const p = runs[runSel]!.path;
          const clearing = compareFeed === p;
          setCompareFeed(clearing ? null : p);
          flashBanner(clearing ? "compare cleared" : `comparing vs ${runs[runSel]!.name}`, C.eval, 4000);
        }
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
          {suspendedRef.current.has(feedDesc.path) && <Text color={C.warm}>⏸ suspended  </Text>}
          {compareFeed && <Text color={C.eval}>◆ compare  </Text>}
          {paused ? <Text color={C.warm}>‖ paused</Text> : <Text color={C.green}>● live</Text>}
          <Text color={C.dim}>{"  "}{feedDesc.path}</Text>
        </Text>
      </Box>

      {banner && (
        <Box paddingX={1}>
          <Text color={banner.color} bold>
            {banner.text}
          </Text>
        </Box>
      )}

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
          {launchActive ? (
            <>
              <Hint k="esc" label="back" />
              <Hint k="↑↓/⏎" label="form" />
            </>
          ) : runsActive ? (
            <>
              <Hint k="esc" label="back" />
              <Hint k="⏎" label="watch" />
              <Hint k="c" label="compare" />
            </>
          ) : (
            <>
              <Hint k="q" label="quit" />
              <Hint k="↑↓" label="nav" />
              <Hint k="⏎" label={nav === "Launch" ? "configure" : "select"} />
              <Hint k="p" label="pause view" />
              {runControllable(feedDesc.path) && (
                <>
                  <Hint k="z" label={suspendedRef.current.has(feedDesc.path) ? "resume" : "suspend"} />
                  <Hint k="x" label="stop" />
                </>
              )}
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
