import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useApp, useInput, useStdout } from "ink";
import { Feed, type FeedStatus } from "../feed.ts";
import { WatchState } from "../state.ts";
import { ChartRenderer, type CompareRun } from "../render.ts";
import { exportRunSummary, listRuns, type RunInfo, type RunMeta, readRunMeta } from "../runs.ts";
import { relaunchFromCheckpoint } from "../launch.ts";
import { parseSshTarget, stopRemoteSession } from "../ssh.ts";
import { dstackStreamAlive, stopDstackRun } from "../dstack.ts";
import { modalStreamAlive, stopModalRun } from "../modal.ts";
import { fetchArtifacts, isFetchable } from "../artifacts.ts";
import { type Alert, AlertEngine, desktopNotify } from "../alerts.ts";
import { pauseRun, resumeRun, runControllable, stopRun } from "../controls.ts";
import { SUROGATE_REPO, copyToClipboard, modelRequestUrl, openUrl } from "../links.ts";
import { C, applyTheme, getTheme, saveThemePref } from "./theme.ts";
import { NAV, Sidebar, type NavItem } from "./Sidebar.tsx";
import { Page } from "./Pages.tsx";
import { Launch } from "./Launch.tsx";
import { Browse } from "./Browse.tsx";
import type { Trainability } from "../supported.ts";
import { StartScreen } from "./StartScreen.tsx";
import { Onboarding } from "./Onboarding.tsx";
import { HelpOverlay } from "./HelpOverlay.tsx";
import { type Compute, loadOnboarding, saveOnboarding } from "../setup.ts";

const SIDEBAR_W = 16;

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
  // Watches a just-launched local run: if it dies producing no metrics, we surface
  // a clear "failed to start" instead of an empty dashboard.
  const launchWatchRef = useRef<{ path: string; at: number } | null>(null);
  const bannerTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [, force] = useState(0);
  const rerender = () => force((n) => n + 1);

  const [started, setStarted] = useState(false);
  // First-run setup wizard: shown before the dashboard until onboarding completes.
  const [needsSetup, setNeedsSetup] = useState(() => !loadOnboarding()?.completed);
  const [setupOpen, setSetupOpen] = useState(false);
  const [setupCompute, setSetupCompute] = useState<{ compute: Compute; sshHost?: string } | null>(null);
  const [feedDesc, setFeedDesc] = useState<FeedDesc>({ path: initialFeedPath, fromStart });
  const [navIdx, setNavIdx] = useState(0);
  const [focus, setFocus] = useState<"nav" | "content">("nav");
  const [runSel, setRunSel] = useState(0);
  const [filesSel, setFilesSel] = useState(0);
  const [paused, setPaused] = useState(false);
  const [banner, setBannerState] = useState<{ text: string; color: string } | null>(null);
  const [compareFeed, setCompareFeed] = useState<string | null>(null);
  const [stopArmed, setStopArmed] = useState(false);
  const [quitArmed, setQuitArmed] = useState(false);
  const [, setThemeName] = useState(getTheme()); // re-render trigger on theme swap
  // Cached so the footer doesn't readFileSync(.pid)+(/proc/pid/cmdline) every frame.
  const [controllable, setControllable] = useState(false);
  const [remoteRun, setRemoteRun] = useState<RunMeta["remote"]>(undefined); // SSH run on this feed
  const [feedState, setFeedState] = useState<FeedStatus>("waiting");
  // Live chart dimensions, read by the render interval without being a dep (so a
  // terminal resize doesn't tear down and recreate the interval).
  const dimsRef = useRef({ chartCols: 80, chartHeight: 20 });
  const [picked, setPicked] = useState<{ model?: { id: string; t: Trainability }; dataset?: string }>({});
  // Search box for the Models/Datasets tabs — reset on tab switch so a query
  // typed in Models doesn't bleed into Datasets (and vice-versa).
  const [browseQuery, setBrowseQuery] = useState("");
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  const [smooth, setSmooth] = useState(false);
  const smoothRef = useRef(smooth);
  smoothRef.current = smooth;
  const [showHelp, setShowHelp] = useState(false);
  const [events, setEvents] = useState<Alert[]>([]); // alert history (newest last)

  const flashBanner = (text: string, color: string, ms = 8000) => {
    setBannerState({ text, color });
    if (bannerTimer.current) clearTimeout(bannerTimer.current);
    bannerTimer.current = setTimeout(() => setBannerState(null), ms);
  };

  const cols = stdout?.columns ?? 120;
  const rows = stdout?.rows ?? 40;
  // the chart fills the middle column (between the nav sidebar and the stats rail)
  const mainW = Math.max(30, cols - SIDEBAR_W - 2);
  const chartCols = Math.max(20, mainW - 4);
  const chartHeight = Math.max(6, rows - 16);
  dimsRef.current = { chartCols, chartHeight };
  const nav: NavItem = NAV[navIdx]!;

  // Switching tabs starts the search box fresh — a query typed in Models must
  // not carry over to Datasets (the two share one box). The onboarding compute
  // hand-off is consumed on the first Launch open; clear it when leaving so a
  // later Launch visit shows the normal compute picker (Launch remounts per tab).
  useEffect(() => {
    setBrowseQuery("");
    if (nav !== "Launch") setSetupCompute(null);
  }, [nav]);

  // (re)create the feed whenever the watched feed path changes — resets state so
  // runs never bleed into each other.
  useEffect(() => {
    let cancelled = false;
    void feedRef.current?.stop();
    stateRef.current = new WatchState();
    chartRef.current = new ChartRenderer();
    const feed = new Feed(feedDesc.path, feedDesc.fromStart);
    feedRef.current = feed;
    setControllable(runControllable(feedDesc.path));
    setRemoteRun(readRunMeta(feedDesc.path)?.remote);
    setFeedState("waiting");
    feed.start(
      (records) => {
        // always ingest (so 'pause view' never loses data); only the redraw is paused
        if (cancelled) return;
        stateRef.current.ingest(records);
        if (!pausedRef.current) rerender();
      },
      (st) => {
        if (!cancelled) setFeedState(st);
      },
    );
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
      const { chartCols: cw, chartHeight: ch } = dimsRef.current;
      await chartRef.current.maybeRender(stateRef.current, cw, ch, Date.now(), compareRef.current ?? undefined, smoothRef.current);
      // alerts (bell + banner + best-effort desktop notification)
      const a = alertRef.current.check(stateRef.current, feedDesc.path, Date.now());
      if (a) {
        process.stdout.write("\x07");
        flashBanner(`⚑ ${a.text}`, a.color === "red" ? C.red : a.color === "green" ? C.green : C.warm);
        desktopNotify("jackalope", a.text);
        setEvents((e) => [...e.slice(-19), a]); // keep the last 20 events
      }
      // controllability can change (a run finishes/dies); the remote meta is
      // immutable after launch, so it's read once on feed-switch (not per tick).
      const ctrl = runControllable(feedDesc.path);
      setControllable(ctrl);

      // Launch watchdog: a just-launched run that produced NO metrics and whose
      // worker is already gone failed to start — say so loudly instead of showing
      // an empty dashboard. The "worker" is the local process (local) or the
      // streaming child: `dstack apply` / the Modal driver (cloud). While the cloud
      // is still provisioning the stream stays alive, so this never false-positives.
      const w = launchWatchRef.current;
      if (w && w.path === feedDesc.path && Date.now() - w.at > 4000) {
        const st = stateRef.current;
        const produced = st.lossHistory.length > 0 || st.step > 0 || st.hasGpus;
        const workerAlive = remoteRun
          ? remoteRun.kind === "dstack"
            ? dstackStreamAlive(remoteRun.session)
            : remoteRun.kind === "modal"
              ? modalStreamAlive(remoteRun.session)
              : true // ssh = fire-and-forget remote tmux; can't cheaply detect
          : ctrl;
        if (produced) {
          launchWatchRef.current = null; // alive and emitting — all good
        } else if (!workerAlive) {
          launchWatchRef.current = null;
          const where = remoteRun ? "the cloud/remote run exited without producing metrics" : "run exited immediately without producing metrics";
          flashBanner(`⚠ ${where} — it failed to start. Open Logs (tab) to see why.`, C.red, 16000);
        }
      }
      rerender();
    }, 500);
    return () => clearInterval(t);
  }, [started, feedDesc.path, remoteRun]);

  const switchFeed = (p: string) => {
    setFeedDesc({ path: p, fromStart: true });
    setFocus("nav");
    setNavIdx(0);
  };

  // Refresh the run list on a slow interval (and on page entry / feed switch)
  // rather than doing synchronous fs.readdir + statSync on every 500ms frame.
  const [runs, setRuns] = useState<RunInfo[]>([]);
  useEffect(() => {
    if (nav !== "Runs" && nav !== "Files") return; // both tabs render from `runs`
    const refresh = () => setRuns(listRuns([initialFeedPath, feedDesc.path], Date.now()));
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, [nav, initialFeedPath, feedDesc.path]);

  useInput((input, key) => {
    // The setup wizard owns all keys while it's open.
    if (setupOpen) return;
    if (!started) {
      // First run (or surogate never set up) leads into setup; otherwise ⏎ opens
      // the dashboard and 's' re-opens setup.
      if (key.return || input === " ") {
        if (needsSetup) setSetupOpen(true);
        else setStarted(true);
      } else if (input === "s") setSetupOpen(true);
      else if (input === "d") setStarted(true); // skip setup → straight to dashboard
      else if (input === "q") exit();
      else if (input === "t") {
        const next = getTheme() === "dark" ? "light" : "dark";
        applyTheme(next);
        saveThemePref(next);
        setThemeName(next);
      } else if (input === "g") openUrl(SUROGATE_REPO);
      return;
    }

    // Help overlay: open with ?, any key closes it.
    if (showHelp) {
      setShowHelp(false);
      return;
    }

    // Launch form: App returns Esc → nav; the form widgets handle the rest.
    if (focus === "content" && nav === "Launch") {
      if (key.escape) setFocus("nav");
      return;
    }
    // Browse and the GPUs picker own all their keys, including back-navigation
    // (← / esc). App stays out so typing/toggling can't quit/pause the run.
    if (focus === "content" && (nav === "Models" || nav === "Datasets" || nav === "GPUs")) {
      return;
    }

    // Models/Datasets are search tabs: typing a printable key from the nav focus
    // (before pressing ⏎) focuses the search and captures the character, so 'q'
    // types "q…" instead of quitting the app.
    if (focus === "nav" && (nav === "Models" || nav === "Datasets") && input && !key.ctrl && !key.meta && input >= " ") {
      setFocus("content");
      setBrowseQuery((q) => q + input);
      return;
    }

    // Quit is deliberate: a single 'q' arms a confirm, a second 'q' exits, any
    // other key cancels. (Ctrl+C still hard-exits instantly.)
    if (quitArmed && input !== "q") setQuitArmed(false);
    if (input === "q") {
      if (quitArmed) return exit();
      setQuitArmed(true); // a persistent confirm bar renders while armed
      return;
    }

    if (stopArmed && input !== "x") setStopArmed(false);

    if (input === "p") {
      setPaused((x) => !x);
      return;
    }

    // toggle dark/light theme (mutates the shared palette in place + re-renders)
    if (input === "t") {
      const next = getTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      saveThemePref(next);
      chartRef.current.invalidate(); // recolor the loss chart right away
      setThemeName(next);
      flashBanner(`theme: ${next}`, C.accent, 2500);
      return;
    }

    // open the surogate repo (★ a star helps the project)
    if (input === "g") {
      openUrl(SUROGATE_REPO);
      copyToClipboard(SUROGATE_REPO);
      flashBanner("✓ link copied to clipboard · opening surogate on GitHub — ★ a star helps!", C.green, 6000);
      return;
    }

    // smooth the loss curve (EMA) — easier to read trends on noisy runs
    if (input === "s") {
      chartRef.current.invalidate();
      setSmooth((x) => !x);
      return;
    }

    if (input === "?") {
      setShowHelp(true);
      return;
    }

    // Run-control keys are skipped while a list/form (Launch, Runs) has focus, so
    // they can't act on the watched run when the user means the highlighted item.
    const listFocused = focus === "content" && (nav === "Launch" || nav === "Runs");

    // pause / resume the current launched run (SIGSTOP / SIGCONT)
    if (input === "z" && !listFocused) {
      if (!controllable) {
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

    // stop the current launched run (two-press confirm) — local PID or remote tmux
    if (input === "x" && !listFocused) {
      if (!controllable && !remoteRun) {
        flashBanner("no controllable run on this feed", C.muted, 3000);
        return;
      }
      if (!stopArmed) {
        setStopArmed(true);
        flashBanner("press x again to STOP this run", C.warm, 4000);
      } else {
        const ok = remoteRun
          ? remoteRun.kind === "dstack"
            ? stopDstackRun(remoteRun.session)
            : remoteRun.kind === "modal"
              ? stopModalRun(feedDesc.path)
              : stopRemoteSession(parseSshTarget(remoteRun.host), remoteRun.session)
          : stopRun(feedDesc.path);
        setStopArmed(false);
        flashBanner(ok ? "⏹ stopping run…" : "stop failed", ok ? C.warm : C.red, 5000);
      }
      return;
    }

    if (focus === "content") {
      // ← / esc returns to the nav menu (Runs/Files use j/k for the list, so the
      // left arrow is free to mean "back").
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
        } else if (input === "e" && runs[runSel]) {
          try {
            const out = exportRunSummary(runs[runSel]!);
            flashBanner(`✓ exported run summary → ${out}`, C.green, 8000);
          } catch (err) {
            flashBanner(`✗ export failed — ${(err as Error).message}`, C.red, 5000);
          }
        } else if (input === "y" && runs[runSel]) {
          copyToClipboard(runs[runSel]!.path);
          flashBanner(`✓ copied run path — ${runs[runSel]!.path}`, C.eval, 6000);
        } else if (input === "f" && runs[runSel]) {
          // fetch a remote/cloud run's artifacts (model, checkpoints, logs) locally
          const run = runs[runSel]!;
          if (!isFetchable(run.path)) {
            flashBanner("⚠ this run's outputs aren't downloadable (dstack) — push to the Hub instead", C.warm, 7000);
          } else {
            flashBanner(`⬇ fetching ${run.name} artifacts…`, C.eval, 60000);
            void fetchArtifacts(run.path)
              .then((res) => {
                if (res.ok) flashBanner(`✓ artifacts → ${res.dest}`, C.green, 10000);
                else flashBanner(`✗ fetch failed — ${res.reason}`, C.red, 9000);
              })
              .catch((err) => flashBanner(`✗ fetch failed — ${(err as Error).message}`, C.red, 9000));
          }
        } else if (input === "r" && runs[runSel]) {
          const run = runs[runSel]!;
          const res = relaunchFromCheckpoint(run, surogateBin);
          if (res.ok) {
            switchFeed(res.feed);
            flashBanner(`▶ resuming ${run.name} from ${run.checkpoints[0]!.name} (pid ${res.pid})`, C.green, 8000);
          } else flashBanner(`✗ ${res.reason}`, C.warm, 6000);
        }
      }
      if (nav === "Files") {
        if (key.upArrow || input === "k") setFilesSel((i) => Math.max(0, i - 1));
        else if (key.downArrow || input === "j") setFilesSel((i) => Math.min(Math.max(0, runs.length - 1), i + 1));
        else if (input === "y" && runs[filesSel]) {
          const folder = runs[filesSel]!.path.replace(/metrics\.jsonl$/, "");
          copyToClipboard(folder);
          flashBanner(`✓ copied run folder — ${folder}`, C.eval, 6000);
        }
      }
      return; // Launch widgets handle their own keys
    }

    // focus === "nav" — the left tab menu: ↑↓ move, ⏎/→ enter the tab
    if (key.upArrow || input === "k") setNavIdx((i) => (i - 1 + NAV.length) % NAV.length);
    else if (key.downArrow || input === "j") setNavIdx((i) => (i + 1) % NAV.length);
    else if (input >= "1" && input <= String(NAV.length)) setNavIdx(Number(input) - 1);
    else if (
      (key.return || key.rightArrow) &&
      (nav === "Launch" || nav === "Runs" || nav === "Files" || nav === "Models" || nav === "Datasets" || nav === "GPUs")
    )
      setFocus("content");
  });

  const s = stateRef.current;

  if (setupOpen) {
    return (
      <Box width={cols} height={rows} backgroundColor={C.bg}>
        <Onboarding
          surogateBin={surogateBin}
          repoRoot={repoRoot}
          onExit={() => setSetupOpen(false)}
          onDone={(r) => {
            saveOnboarding({ completed: true, compute: r.compute, sshHost: r.sshHost, surogateOk: true, ts: new Date().toISOString() });
            setNeedsSetup(false);
            setSetupCompute(r);
            setSetupOpen(false);
            setNavIdx(NAV.indexOf("Launch")); // open Launch with the chosen compute
            setFocus("content");
            setStarted(true);
            flashBanner(`✓ setup complete · compute: ${r.compute}${r.sshHost ? ` (${r.sshHost})` : ""}`, C.green, 6000);
          }}
        />
      </Box>
    );
  }

  if (!started) {
    return (
      <Box width={cols} height={rows} backgroundColor={C.bg}>
        <StartScreen feedPath={feedDesc.path} version={version} needsSetup={needsSetup} />
      </Box>
    );
  }

  if (showHelp) {
    return (
      <Box width={cols} height={rows} backgroundColor={C.bg}>
        <HelpOverlay />
      </Box>
    );
  }

  // last non-empty path segment, so a trailing slash (output_dir: ./watch-out/)
  // doesn't yield an empty name
  const outDir = typeof s.configFields["output_dir"] === "string" ? (s.configFields["output_dir"] as string) : "";
  const outSegs = outDir.split("/").filter(Boolean);
  const model = s.model ?? (outSegs.length ? outSegs[outSegs.length - 1]! : "run");
  const recipe = s.recipe ?? "?";
  // short, friendly name of the watched run (the run folder, not the long path)
  const feedParts = feedDesc.path.split("/");
  const runName =
    feedParts[feedParts.length - 1] === "metrics.jsonl" ? feedParts[feedParts.length - 2]! : feedParts[feedParts.length - 1]!;
  const launchActive = nav === "Launch" && focus === "content";
  const runsActive = nav === "Runs" && focus === "content";
  const filesActive = nav === "Files" && focus === "content";
  const gpusActive = nav === "GPUs" && focus === "content";
  const browseActive = (nav === "Models" || nav === "Datasets") && focus === "content";

  // the active tab's content (rendered either with the right rail — Monitor — or
  // centered in a max-width column — everything else)
  const mainContent =
    nav === "Launch" ? (
      <Launch
        feedPath={feedDesc.path}
        surogateBin={surogateBin}
        repoRoot={repoRoot}
        active={launchActive}
        picked={picked}
        onLaunched={(metricsPath) => {
          launchWatchRef.current = { path: metricsPath, at: Date.now() };
          switchFeed(metricsPath);
        }}
        initialCompute={setupCompute?.compute}
        initialSshHost={setupCompute?.sshHost}
      />
    ) : nav === "Models" || nav === "Datasets" ? (
      <Browse
        kind={nav === "Models" ? "models" : "datasets"}
        active={focus === "content"}
        picked={picked}
        query={browseQuery}
        setQuery={setBrowseQuery}
        onExit={() => setFocus("nav")}
        onPickModel={(id, t) => {
          setPicked((p) => ({ ...p, model: { id, t } }));
          flashBanner(`✓ model selected: ${id}${t.supported ? "" : " (unsupported!)"}`, t.supported ? C.green : C.red, 5000);
        }}
        onPickDataset={(id) => {
          setPicked((p) => ({ ...p, dataset: id }));
          flashBanner(`✓ dataset selected: ${id}`, C.eval, 4000);
        }}
        onRequestModel={(id, arch) => {
          const url = modelRequestUrl(id, arch);
          openUrl(url);
          copyToClipboard(url);
          flashBanner(`✓ request link copied to clipboard · opening GitHub for "${id}"`, C.green, 6000);
        }}
      />
    ) : (
      <Page
        nav={nav}
        s={s}
        chartImage={chartRef.current.current()}
        chartHeight={chartHeight}
        feedState={feedState}
        runs={runs}
        runSel={runSel}
        runsActive={runsActive}
        filesSel={filesSel}
        filesActive={filesActive}
        gpusActive={gpusActive}
        onGpusExit={() => setFocus("nav")}
        currentFeed={feedDesc.path}
        compareFeed={compareFeed}
        events={events}
      />
    );
  return (
    <Box flexDirection="column" width={cols} height={rows} backgroundColor={C.bg}>
      <Box justifyContent="space-between" paddingX={1} backgroundColor={C.panel}>
        <Text>
          <Text bold color={C.gold}>
            ◆ jackalope
          </Text>
          <Text color={C.dim}>{"   "}</Text>
          <Text color={C.text}>{model}</Text>
          <Text color={C.dim}> · </Text>
          <Text color={C.muted}>{recipe}</Text>
          {s.lora && <Text color={C.accent}> · LoRA</Text>}
        </Text>
        <Text>
          {remoteRun && <Text color={C.eval}>⬡ remote{"  "}</Text>}
          {suspendedRef.current.has(feedDesc.path) && <Text color={C.warm}>⏸ suspended{"  "}</Text>}
          {compareFeed && <Text color={C.eval}>◆ compare{"  "}</Text>}
          {paused ? <Text color={C.warm}>‖ paused</Text> : <Text color={C.green}>● live</Text>}
          <Text color={C.dim}>{"   "}{runName}</Text>
        </Text>
      </Box>

      {(picked.model || picked.dataset) && (
        <Box paddingX={1}>
          <Text>
            <Text color={C.gold} bold>
              ◆ your run
            </Text>
            <Text color={C.muted}>{"   model "}</Text>
            {picked.model ? (
              <Text color={picked.model.t.supported ? C.green : C.red}>✓ {picked.model.id}</Text>
            ) : (
              <Text color={C.dim}>— none —</Text>
            )}
            <Text color={C.muted}>{"   dataset "}</Text>
            {picked.dataset ? <Text color={C.eval}>✓ {picked.dataset}</Text> : <Text color={C.dim}>— none —</Text>}
            <Text color={C.dim}>{"    · Launch tab to run"}</Text>
          </Text>
        </Box>
      )}

      {quitArmed ? (
        <Box paddingX={1}>
          <Text backgroundColor={C.red} color="#000000" bold>
            {"  ⚠ QUIT? press "}
          </Text>
          <Text backgroundColor={C.red} color="#ffffff" bold>
            q
          </Text>
          <Text backgroundColor={C.red} color="#000000" bold>
            {" again to exit · any other key cancels  "}
          </Text>
        </Box>
      ) : (
        banner && (
          <Box paddingX={1}>
            <Text color={banner.color} bold>
              {banner.text}
            </Text>
          </Box>
        )
      )}

      {/* left: tab menu · right: content (full width) */}
      <Box flexGrow={1}>
        <Sidebar active={nav} s={s} width={SIDEBAR_W} focusedNav={focus === "nav"} />
        <Box flexGrow={1} flexDirection="column" paddingX={1}>
          {mainContent}
        </Box>
      </Box>

      <Box justifyContent="space-between" paddingX={1} backgroundColor={C.panel}>
        <Text color={C.muted}>
          {browseActive ? (
            <>
              <Hint k="type" label="search" />
              <Hint k="↑↓" label="browse" />
              <Hint k="⏎" label="select" />
              <Hint k="←/esc" label="back" />
              <Hint k="^C" label="quit" />
            </>
          ) : launchActive ? (
            <>
              <Hint k="esc" label="back" />
              <Hint k="^C" label="quit" />
              <Hint k="↑↓" label="move" />
              <Hint k="space" label="select GPUs" />
              <Hint k="⏎" label="next" />
            </>
          ) : runsActive ? (
            <>
              <Hint k="esc" label="back" />
              <Hint k="⏎" label="watch" />
              <Hint k="c" label="compare" />
              <Hint k="r" label="resume" />
              <Hint k="e" label="export" />
              <Hint k="y" label="copy" />
            </>
          ) : (
            <>
              <Hint k={quitArmed ? "q" : "q×2"} label={quitArmed ? "confirm quit" : "quit"} />
              <Hint k="↑↓" label="nav" />
              <Hint k="⏎" label={nav === "Launch" ? "configure" : "select"} />
              <Hint k="t" label={`theme·${getTheme()}`} />
              <Hint k="s" label={smooth ? "raw" : "smooth"} />
              <Hint k="g" label="★ github" />
              <Hint k="?" label="keys" />
              <Hint k="p" label="pause view" />
              {controllable && (
                <>
                  <Hint k="z" label={suspendedRef.current.has(feedDesc.path) ? "resume" : "suspend"} />
                  <Hint k="x" label="stop" />
                </>
              )}
              {!controllable && remoteRun && <Hint k="x" label="stop (remote)" />}
            </>
          )}
        </Text>
        <Text color={C.dim}>
          {nav.toLowerCase()}
          {"  ·  "}
          {s.hasGpus ? `${s.gpusSorted().length} GPU` : "no GPU"}
          {s.maxSteps && s.maxSteps > 0 ? `  ·  ${Math.round((s.step / s.maxSteps) * 100)}%` : ""}
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
