import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import type { Feed } from "../feed.ts";
import { WatchState } from "../state.ts";
import { ChartRenderer } from "../render.ts";
import { C } from "./theme.ts";
import { Monitor } from "./Monitor.tsx";
import { Launch } from "./Launch.tsx";

type Tab = "monitor" | "launch";

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
  const [tab, setTab] = useState<Tab>("monitor");
  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(paused);
  pausedRef.current = paused;

  const cols = stdout?.columns ?? 120;
  const rows = stdout?.rows ?? 40;
  const chartCols = Math.max(20, cols - 6);
  const chartHeight = Math.max(6, rows - 26);

  // feed -> state
  useEffect(() => {
    feed.start((records) => {
      if (!pausedRef.current) {
        stateRef.current.ingest(records);
        force((n) => n + 1);
      }
    });
    return () => void feed.stop();
  }, [feed]);

  // chart + clock refresh
  useEffect(() => {
    const t = setInterval(async () => {
      const changed = await chartRef.current.maybeRender(stateRef.current, chartCols, chartHeight, Date.now());
      force((n) => n + 1);
      void changed;
    }, 500);
    return () => clearInterval(t);
  }, [chartCols, chartHeight]);

  useInput((input, key) => {
    if (input === "q" || key.escape) exit();
    else if (input === "m") setTab("monitor");
    else if (input === "l") setTab("launch");
    else if (input === "p") setPaused((p) => !p);
  });

  const s = stateRef.current;
  const model = s.model ?? (typeof s.configFields["output_dir"] === "string"
    ? (s.configFields["output_dir"] as string).split("/").pop()!
    : "run");
  const recipe = s.recipe ?? "?";

  return (
    <Box flexDirection="column" width={cols} height={rows}>
      {/* header */}
      <Box justifyContent="space-between" paddingX={1}>
        <Text>
          <Text bold color={C.accent}>
            ◆ surogate
          </Text>
          <Text color={C.muted}> watch — </Text>
          <Text color={C.text}>
            {model} · {recipe}
          </Text>
          {paused && <Text color={C.warm}> (paused)</Text>}
        </Text>
        <Text color={C.muted}>{feedPath}</Text>
      </Box>
      {/* tab bar */}
      <Box paddingX={1}>
        <Tab label="Monitor" active={tab === "monitor"} />
        <Text> </Text>
        <Tab label="Launch" active={tab === "launch"} />
      </Box>
      {/* body */}
      {tab === "monitor" ? (
        <Monitor s={s} chartImage={chartRef.current.current()} chartHeight={chartHeight} />
      ) : (
        <Launch feedPath={feedPath} surogateBin={surogateBin} repoRoot={repoRoot} onLaunched={() => setTab("monitor")} />
      )}
      {/* footer */}
      <Box paddingX={1}>
        <Text color={C.muted}>
          <Text color={C.accent} bold>
            q
          </Text>{" "}
          quit{"   "}
          <Text color={C.accent} bold>
            m
          </Text>{" "}
          monitor{"   "}
          <Text color={C.accent} bold>
            l
          </Text>{" "}
          launch{"   "}
          <Text color={C.accent} bold>
            p
          </Text>{" "}
          pause
        </Text>
      </Box>
    </Box>
  );
}

function Tab({ label, active }: { label: string; active: boolean }) {
  return active ? (
    <Text backgroundColor={C.accent} color="#0e1018" bold>
      {` ${label} `}
    </Text>
  ) : (
    <Text color={C.muted}>{` ${label} `}</Text>
  );
}
