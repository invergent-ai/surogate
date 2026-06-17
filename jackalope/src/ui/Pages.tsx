import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { WatchState } from "../state.ts";
import type { FeedStatus } from "../feed.ts";
import { fmtBytes, fmtCount, fmtFloat } from "../format.ts";
import { type RunInfo, type RunStatus, runArtifacts, runFileEntries, tailBytes } from "../runs.ts";
import type { Alert } from "../alerts.ts";
import { C, lossArrow, memColor, meterParts, tempColor } from "./theme.ts";
import { Panel } from "./Panel.tsx";
import { ProviderManager } from "./ProviderManager.tsx";
import { Chart } from "./Monitor.tsx";
import { ShimmerText } from "./Shimmer.tsx";
import { Sparkline } from "./Sparkline.tsx";
import { discoverGpus, loadGpuSelection, saveGpuSelection } from "../launch.ts";
import { loadProviders, type ProviderRecord } from "../providers.ts";
import { gpuSupport } from "../gpu-support.ts";
import { dstackAvailable } from "../dstack.ts";
import type { NavItem } from "./Sidebar.tsx";

function fmtAge(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

const STATUS: Record<RunStatus, { glyph: string; label: string }> = {
  running: { glyph: "●", label: "running" },
  done: { glyph: "✓", label: "finished" },
  stopped: { glyph: "■", label: "stopped" },
  idle: { glyph: "○", label: "idle" },
};
// live theme color (resolved at render, so it recolors on theme toggle)
function statusColor(s: RunStatus): string {
  return s === "running" || s === "done" ? C.green : s === "stopped" ? C.warm : C.dim;
}

/** Just the basename for long local paths/ids, so cards stay narrow. */
function shortName(s: string | undefined): string {
  if (!s) return "—";
  const base = s.split("/").filter(Boolean).pop() ?? s;
  return base.length > 28 ? base.slice(0, 27) + "…" : base;
}

/** A small inline progress bar (filled/track + percent). */
function MiniBar({ frac, width = 16 }: { frac: number; width?: number }) {
  const p = meterParts(frac, width);
  return (
    <Text>
      <Text color={C.accent}>{"█".repeat(p.filled)}</Text>
      <Text color={C.dim}>{"█".repeat(p.track)}</Text>
      <Text color={C.muted}> {p.pct.trim()}</Text>
    </Text>
  );
}

function RunDetail({ r, watching }: { r: RunInfo; watching: boolean }) {
  const st = STATUS[r.status];
  const m = r.meta;
  const max = m?.maxSteps;
  const step = r.lastStep ?? 0;
  const dur = m ? fmtDuration(r.mtimeMs - m.startedAt) : null;
  const cps = r.checkpoints;
  const Row = ({ label, children }: { label: string; children: React.ReactNode }) => (
    <Box>
      <Text color={C.muted}>{label.padEnd(12)}</Text>
      <Text color={C.text}>{children}</Text>
    </Box>
  );
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color={statusColor(r.status)} bold>
          {st.glyph} {st.label}
        </Text>
        {watching && <Text color={C.green}> · ▸ watching</Text>}
        {m && <Text color={C.muted}>{`   ${m.mode.toUpperCase()}`}</Text>}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        {m?.model && <Row label="model">{shortName(m.model)}</Row>}
        {m?.dataset && <Row label="dataset">{shortName(m.dataset)}</Row>}
        {(m?.recipe || m?.gpus) && (
          <Row label="config">
            {m?.recipe ?? "—"}
            {m?.gpus?.length ? ` · gpu ${m.gpus.join(",")}` : ""}
          </Row>
        )}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Box>
          <Text color={C.muted}>{"progress".padEnd(12)}</Text>
          {max ? <MiniBar frac={step / max} /> : <Text color={C.text}>step {step}</Text>}
          <Text color={C.muted}>{max ? `  ${step}/${max}` : ""}</Text>
        </Box>
        <Row label="loss">{r.lastLoss !== null ? fmtFloat(r.lastLoss) : "—"}</Row>
        {dur && (
          <Row label="duration">
            {dur}
            <Text color={C.dim}>{`  · ${fmtAge(r.ageMs)}`}</Text>
          </Row>
        )}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Box>
          <Text color={C.muted}>{"checkpoints".padEnd(12)}</Text>
          <Text color={cps.length ? C.eval : C.dim}>{cps.length ? `${cps.length} saved` : "none yet"}</Text>
        </Box>
        {cps.slice(0, 5).map((cp, i) => (
          <Text key={cp.name} color={i === 0 ? C.text : C.muted}>
            {"     "}
            {i === 0 ? "◆ " : "· "}
            {cp.name}
            {i === 0 ? <Text color={C.dim}> (latest)</Text> : null}
          </Text>
        ))}
        {cps.length > 5 && <Text color={C.dim}>{`     … +${cps.length - 5} more`}</Text>}
        <Text color={C.dim}>{`  ${r.path.replace(/metrics\.jsonl$/, "")}`}</Text>
        {cps.length > 0 ? (
          <Text>
            <Text color={C.eval} bold>
              {"  r"}
            </Text>
            <Text color={C.dim}>{` resume from ${cps[0]!.name} · e export · y copy path`}</Text>
          </Text>
        ) : (
          <Text color={C.dim}>  e export · y copy path</Text>
        )}
      </Box>
    </Box>
  );
}

function cmpVals(r: RunInfo) {
  const m = r.meta;
  return {
    status: STATUS[r.status].label,
    model: shortName(m?.model),
    recipe: `${m?.recipe ?? "—"}${m ? " · " + m.mode : ""}`,
    loss: r.lastLoss !== null ? fmtFloat(r.lastLoss) : "—",
    step: m?.maxSteps ? `${r.lastStep ?? 0}/${m.maxSteps}` : `${r.lastStep ?? 0}`,
    duration: m ? fmtDuration(r.mtimeMs - m.startedAt) : "—",
    checkpoints: String(r.checkpoints.length),
  };
}

// A two-column metric comparison of the watched run (A) vs a pinned run (B).
// Lower final loss wins (highlighted green).
function CompareTable({ a, b }: { a: RunInfo; b: RunInfo }) {
  const A = cmpVals(a);
  const B = cmpVals(b);
  const COLW = 22;
  const cell = (s: string) => (s.length > COLW - 1 ? s.slice(0, COLW - 1) : s).padEnd(COLW);
  const lossCmp = a.lastLoss !== null && b.lastLoss !== null ? Math.sign(a.lastLoss - b.lastLoss) : 0;
  const Row = ({ label, av, bv, ac, bc }: { label: string; av: string; bv: string; ac?: string; bc?: string }) => (
    <Box>
      <Text color={C.muted}>{label.padEnd(13)}</Text>
      <Text color={ac ?? C.text}>{cell(av)}</Text>
      <Text color={bc ?? C.text}>{cell(bv)}</Text>
    </Box>
  );
  return (
    <Box flexDirection="column" marginTop={1}>
      <Box>
        <Text color={C.muted}>{"compare".padEnd(13)}</Text>
        <Text color={C.accent} bold>{cell("A · " + a.name)}</Text>
        <Text color={C.eval} bold>{cell("B · " + b.name)}</Text>
      </Box>
      <Row label="status" av={A.status} bv={B.status} />
      <Row label="model" av={A.model} bv={B.model} />
      <Row label="config" av={A.recipe} bv={B.recipe} />
      <Row
        label="final loss"
        av={A.loss + (lossCmp < 0 ? "  ✓ lower" : "")}
        bv={B.loss + (lossCmp > 0 ? "  ✓ lower" : "")}
        ac={lossCmp < 0 ? C.green : undefined}
        bc={lossCmp > 0 ? C.green : undefined}
      />
      <Row label="progress" av={A.step} bv={B.step} />
      <Row label="duration" av={A.duration} bv={B.duration} />
      <Row label="checkpoints" av={A.checkpoints} bv={B.checkpoints} />
      <Box marginTop={1}>
        <Text color={C.dim}>c clears compare · loss curves overlay on the Monitor tab</Text>
      </Box>
    </Box>
  );
}

function RunsPage({
  runs,
  runSel,
  active,
  currentFeed,
  compareFeed,
}: {
  runs: RunInfo[];
  runSel: number;
  active: boolean;
  currentFeed: string;
  compareFeed: string | null;
}) {
  const sel = runs[runSel];
  const compared = compareFeed ? runs.find((r) => r.path === compareFeed) : undefined;
  const showCompare = !!compared && !!sel && compared.path !== sel.path;
  return (
    <Panel title="runs" flexGrow={1}>
      <Text color={C.muted}>
        {active ? "↑↓ · ⏎ watch · c compare · f fetch artifacts · e export · r resume · esc back" : "⏎ to browse runs — each launched run is its own folder"}
      </Text>
      {runs.length === 0 ? (
        <Box marginTop={1}>
          <Text color={C.muted}>no runs yet — launch one from the Launch tab</Text>
        </Box>
      ) : (
        <Box marginTop={1}>
          {/* left: the list */}
          <Box flexDirection="column" width={46} marginRight={2}>
            {runs.map((r, i) => {
              const on = active && i === runSel;
              const cur = r.path === currentFeed;
              const cmp = r.path === compareFeed;
              const st = STATUS[r.status];
              return (
                <Text key={r.path} color={on ? C.accent : C.text} bold={on}>
                  {on ? "▸ " : "  "}
                  <Text color={statusColor(r.status)}>{st.glyph}</Text> {r.name.padEnd(24).slice(0, 24)}
                  <Text color={C.muted}>
                    {" "}
                    {(r.meta?.maxSteps ? `${r.lastStep ?? 0}/${r.meta.maxSteps}` : `${r.lastStep ?? 0}`).padStart(8)}
                    {cur ? <Text color={C.green}> ◀</Text> : cmp ? <Text color={C.eval}> ◆</Text> : ""}
                  </Text>
                </Text>
              );
            })}
          </Box>
          {/* right: compare table (when a run is pinned) or the selected summary */}
          {sel && (
            <Box flexDirection="column" flexGrow={1}>
              {showCompare ? <CompareTable a={sel} b={compared!} /> : <RunDetail r={sel} watching={sel.path === currentFeed} />}
            </Box>
          )}
        </Box>
      )}
    </Panel>
  );
}

function Headline({ s }: { s: WatchState }) {
  const { arrow, color: arrowColor } = lossArrow(s.lossTrend());
  return (
    <Box>
      <Text color={C.muted}>loss </Text>
      <Text bold color={C.train}>
        {fmtFloat(s.latestTrainLoss)}
      </Text>
      <Text color={arrowColor}> {arrow}</Text>
      {s.latestEvalLoss !== null && (
        <Text>
          <Text color={C.muted}>{"   eval "}</Text>
          <Text bold color={C.eval}>
            {fmtFloat(s.latestEvalLoss)}
          </Text>
          {(() => {
            const gap = s.evalGapPct();
            if (gap === null) return null;
            // a large positive gap = eval >> train = overfitting
            const col = gap > 25 ? C.red : gap > 10 ? C.warm : C.green;
            return (
              <Text color={col}>
                {"  "}
                {gap >= 0 ? "▲" : "▼"}
                {Math.abs(gap).toFixed(0)}%
              </Text>
            );
          })()}
        </Text>
      )}
      <Text color={C.muted}>{"   lr "}</Text>
      <Text color={C.text}>{fmtFloat(s.lr)}</Text>
      <Text color={C.muted}>{"   |∇| "}</Text>
      <Text color={C.text}>{fmtFloat(s.gradNorm)}</Text>
      <Text color={C.muted}>{"   "}</Text>
      <Text color={C.text}>{fmtCount(s.tokensPerSecond)} tok/s</Text>
    </Box>
  );
}

function GpuRows({ s, big }: { s: WatchState; big: boolean }) {
  if (!s.hasGpus) return <Text color={C.muted}>no GPU telemetry yet</Text>;
  const w = big ? 16 : 10;
  return (
    <Box flexDirection="column">
      {s.gpusSorted().map((g) => {
        const name = (g.name ?? `gpu${g.gpuId}`).slice(0, 12).padEnd(12);
        const temp = g.temp === null ? "  —" : `${Math.round(g.temp)}°`.padStart(3);
        const pwr = g.power === null ? "   —" : `${Math.round(g.power)}W`.padStart(5);
        const sm = meterParts(g.smUtil === null ? null : g.smUtil / 100, w);
        const mem = meterParts(g.memUtil, w);
        return (
          <Box key={g.gpuId}>
            <Text color={C.dim}>{String(g.gpuId).padEnd(2)} </Text>
            <Text color={C.text}>{name} </Text>
            <Text color={tempColor(g.temp)}>{temp}</Text>
            <Text color={C.text}>{"  " + pwr + "   sm "}</Text>
            <Text color={C.accent}>{"█".repeat(sm.filled)}</Text>
            <Text color={C.dim}>{"█".repeat(sm.track)}</Text>
            <Text color={C.text}> {sm.pct}</Text>
            <Text color={C.text}>{"   mem "}</Text>
            <Text color={g.memUtil === null ? C.muted : memColor(g.memUtil)}>{"█".repeat(mem.filled)}</Text>
            <Text color={C.dim}>{"█".repeat(mem.track)}</Text>
            <Text color={C.text}> {mem.pct}</Text>
          </Box>
        );
      })}
    </Box>
  );
}

function MonitorPage({
  s,
  chartImage,
  chartHeight,
  feedState,
  feedPath,
}: {
  s: WatchState;
  chartImage: string;
  chartHeight: number;
  feedState: FeedStatus;
  feedPath: string;
}) {
  const best = s.bestLoss();
  // No metrics yet → the run is starting up / a provider is provisioning. Show its
  // LIVE log + errors here (where the graph will be), so you see exactly what's
  // happening. Once metrics flow, this flips to the loss chart.
  const starting = !chartImage && feedState !== "unavailable";
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title={starting ? "live · starting up" : "training loss"} flexGrow={1} minHeight={chartHeight + 3}>
        {starting ? <LiveActivity feedPath={feedPath} /> : <Chart image={chartImage} feedState={feedState} feedPath={feedPath} />}
        <Box marginTop={1}>
          <Headline s={s} />
        </Box>
        <Box>
          <Text color={C.muted}>throughput </Text>
          <Sparkline values={s.tpsHistory} width={24} color={C.train} />
          <Text color={C.text}>{`  ${fmtCount(s.tokensPerSecond)} tok/s`}</Text>
          {best && (
            <Text color={C.muted}>
              {"     best "}
              <Text color={C.green}>{fmtFloat(best.loss)}</Text>
              <Text color={C.dim}>{` @${best.step}`}</Text>
            </Text>
          )}
        </Box>
      </Panel>
      <Panel title="devices">
        <Box marginTop={1}>
          <GpuRows s={s} big={false} />
        </Box>
      </Panel>
    </Box>
  );
}

const TARGET_GLYPH: Record<string, string> = { local: "▣", ssh: "⇄", modal: "▲", dstack: "☁" };

function GpusPage({ s, active, onExit }: { s: WatchState; active: boolean; onExit: () => void }) {
  const local = useMemo(() => discoverGpus(), []);
  const sup = local[0]?.sm != null ? gpuSupport(local[0]!.sm) : null;
  // Reloaded when returning from the provider manager, so a just-added provider
  // shows up immediately (was memoized once → stale until you left the tab).
  const [providers, setProviders] = useState<ProviderRecord[]>(() => loadProviders());
  const [dstackOk, setDstackOk] = useState<boolean | null>(null);
  useEffect(() => {
    void dstackAvailable().then(setDstackOk);
  }, []);

  // Browse compute targets (default) → pick Local's GPUs, or manage providers.
  const [view, setView] = useState<"targets" | "localgpus" | "providers">("targets");
  const [tcur, setTcur] = useState(0);

  // Local GPU multi-select — persisted, and pre-fills Launch.
  const [cursor, setCursor] = useState(0);
  const [picked, setPicked] = useState<Set<number>>(
    () => new Set(loadGpuSelection().filter((id) => local.some((g) => g.id === id))),
  );
  const toggle = (ids: Set<number>) => {
    saveGpuSelection([...ids]);
    setPicked(ids);
  };

  // the browsable targets: Local first, then every saved provider
  type Target = { kind: string; label: string; sub: string; p?: ProviderRecord };
  const targets: Target[] = useMemo(
    () => [
      { kind: "local", label: "Local", sub: local.length ? `${local.length} GPU${local.length === 1 ? "" : "s"}${sup ? ` · ${sup.arch}` : ""}` : "no NVIDIA GPUs here" },
      ...providers.map((p) => ({ kind: p.kind, label: p.label, sub: p.host ?? p.backend ?? p.workspace ?? "", p })),
    ],
    [local, sup, providers],
  );

  // targets view: browse + select
  useInput(
    (input, key) => {
      if (key.escape || key.leftArrow) return onExit();
      if (input === "p") return setView("providers");
      if (key.upArrow || input === "k") setTcur((c) => Math.max(0, c - 1));
      else if (key.downArrow || input === "j") setTcur((c) => Math.min(targets.length - 1, c + 1));
      else if (key.return && targets[tcur]?.kind === "local" && local.length) {
        setCursor(0);
        setView("localgpus");
      }
    },
    { isActive: active && view === "targets" },
  );

  // local GPU select view
  useInput(
    (input, key) => {
      if (key.escape || key.leftArrow) return setView("targets");
      if (!local.length) return;
      if (key.upArrow || input === "k") setCursor((c) => Math.max(0, c - 1));
      else if (key.downArrow || input === "j") setCursor((c) => Math.min(local.length - 1, c + 1));
      else if (input === " ") {
        const id = local[cursor]?.id;
        if (id === undefined) return;
        const n = new Set(picked);
        n.has(id) ? n.delete(id) : n.add(id);
        toggle(n);
      } else if (input === "a") {
        toggle(picked.size === local.length ? new Set() : new Set(local.map((g) => g.id)));
      }
    },
    { isActive: active && view === "localgpus" },
  );

  if (view === "providers") {
    return (
      <Box flexDirection="column" flexGrow={1}>
        <Panel title="compute providers" flexGrow={1}>
          <Box marginTop={1}>
            <ProviderManager
              active={active && view === "providers"}
              onExit={() => {
                setProviders(loadProviders()); // pick up any newly added provider
                setTcur(0);
                setView("targets");
              }}
            />
          </Box>
        </Panel>
      </Box>
    );
  }

  if (view === "localgpus") {
    return (
      <Box flexDirection="column" flexGrow={1}>
        <Panel title="Local GPUs — pick which to train on" flexGrow={1}>
          <Box marginTop={1} flexDirection="column">
            {local.map((g, i) => {
              const on = active && i === cursor;
              const checked = picked.has(g.id);
              const gs = g.sm != null ? gpuSupport(g.sm) : null;
              return (
                <Text key={g.id} color={on ? C.accent : C.text} bold={on}>
                  {on ? "❯ " : "  "}
                  <Text color={checked ? C.green : C.dim}>{checked ? "[✓]" : "[ ]"}</Text>
                  <Text color={C.dim}>{` gpu${g.id} `}</Text>
                  <Text color={on ? C.accent : C.text}>{g.name}</Text>
                  <Text color={C.muted}>{`  ${g.memMB != null ? Math.round(g.memMB / 1024) + "GB" : "?"}`}</Text>
                  {gs && <Text color={C.dim}>{`  ${gs.arch}`}</Text>}
                  {g.busy && <Text color={C.warm}>{"  busy"}</Text>}
                </Text>
              );
            })}
            <Box marginTop={1} flexDirection="column">
              <Text color={C.dim}>{`space toggle · a all/none · ${picked.size} selected → Launch · ← back`}</Text>
              {sup && (
                <Text>
                  <Text color={C.accent}>{`${local.length}× ${sup.arch}`}</Text>
                  <Text color={C.dim}> · trains </Text>
                  <Text color={C.gold}>{sup.recipes.join(" · ")}</Text>
                </Text>
              )}
            </Box>
          </Box>
        </Panel>
      </Box>
    );
  }

  // default: browsable compute targets + trends
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title="compute targets">
        <Box marginTop={1} flexDirection="column">
          {targets.map((t, i) => {
            const on = active && i === tcur;
            return (
              <Text key={i} color={on ? C.accent : C.text} bold={on}>
                {on ? "❯ " : "  "}
                <Text color={C.eval}>{TARGET_GLYPH[t.kind] ?? "•"}</Text>
                {` ${t.label}`}
                <Text color={C.muted}>{t.sub ? `  ${t.sub}` : ""}</Text>
                {t.kind === "local" && picked.size > 0 && <Text color={C.green}>{`  · ${picked.size} GPUs picked`}</Text>}
              </Text>
            );
          })}
          <Box marginTop={1}>
            <Text color={C.dim}>
              {active ? "↑↓ choose · ⏎ Local → pick GPUs · p add/manage providers · ← back" : "⏎ to browse compute targets"}
            </Text>
          </Box>
          {dstackOk === false && <Text color={C.dim}>{"cloud (dstack) not installed — add it via p"}</Text>}
          <Text color={C.dim}>{"remote/cloud GPUs are chosen at Launch → Compute"}</Text>
        </Box>
      </Panel>
      <Panel title="trends (recent history)" flexGrow={1}>
        <Box flexDirection="column" marginTop={1}>
          {!s.hasGpus ? (
            <Text color={C.muted}>no GPU telemetry yet</Text>
          ) : (
            s.gpusSorted().map((g) => {
              const h = s.gpuHistory(g.gpuId);
              return (
                <Box key={g.gpuId}>
                  <Text color={C.dim}>{`gpu${g.gpuId} `}</Text>
                  <Text color={C.muted}>sm </Text>
                  <Sparkline values={h?.sm ?? []} width={18} color={C.accent} />
                  <Text color={C.muted}>{"  mem "}</Text>
                  <Sparkline values={(h?.mem ?? []).map((m) => m * 100)} width={18} color={C.eval} />
                  <Text color={C.muted}>{"  °C "}</Text>
                  <Sparkline values={h?.temp ?? []} width={12} color={C.warm} />
                  <Text color={C.muted}>{"  W "}</Text>
                  <Sparkline values={h?.power ?? []} width={12} color={C.green} />
                </Box>
              );
            })
          )}
        </Box>
      </Panel>
    </Box>
  );
}

// Tail the last `n` non-empty lines of the run's surogate stdout/stderr log.
// surogate's stdout carries its own ANSI colors + tqdm carriage-returns; strip
// them so log lines render as plain text (the embedded dark colors otherwise show
// as black lines, esp. on a light theme).
// eslint-disable-next-line no-control-regex
const ANSI_ESC = /\x1b\[[0-9;?]*[a-zA-Z]/g;
// Live run/provider activity, shown in the Monitor while a run is starting up
// (before any metrics): the tail of the run log — provisioning steps, surogate
// startup, and any errors (highlighted) — streamed in real time. Covers every
// backend (local spawn, ssh mirror, dstack apply, modal driver) since they all
// land in the same run log.
function LiveActivity({ feedPath }: { feedPath: string }) {
  const lines = tailLog(feedPath, 16);
  return (
    <Box flexDirection="column">
      <ShimmerText text="waiting for the first metrics — live output below" />
      <Box flexDirection="column" marginTop={1}>
        {lines.length === 0 ? (
          <Text color={C.dim}>provisioning / installing… (the provider's output will stream here)</Text>
        ) : (
          lines.map((l, i) => {
            const err = /\b(error|fail(ed|ure)?|traceback|exception|fatal|not found|denied|refused|no such)\b|✗/i.test(l);
            return (
              <Text key={i} color={err ? C.red : C.muted} wrap="truncate-end">
                {l}
              </Text>
            );
          })
        )}
      </Box>
    </Box>
  );
}

function tailLog(feedPath: string, n = 10): string[] {
  return tailBytes(runArtifacts(feedPath).logPath, 8192)
    .replace(/\r/g, "\n") // tqdm rewrites a line with \r — treat as new lines
    .split("\n")
    .map((l) => l.replace(ANSI_ESC, "").replace(/\s+$/, ""))
    .filter(Boolean)
    .slice(-n);
}

const alertColor = (c: Alert["color"]): string => (c === "red" ? C.red : c === "warm" ? C.warm : C.green);

function LogsPage({ s, events, feedPath }: { s: WatchState; events: Alert[]; feedPath: string }) {
  const recent = s.recentSteps.slice(-12);
  const best = s.bestLoss();
  const logLines = tailLog(feedPath, 8);
  return (
    <Box flexDirection="column" flexGrow={1}>
      {events.length > 0 && (
        <Panel title="events">
          <Box flexDirection="column" marginTop={1}>
            {events.slice(-6).reverse().map((a, i) => (
              <Text key={i} color={alertColor(a.color)}>
                ⚑ <Text color={C.text}>{a.text}</Text>
              </Text>
            ))}
          </Box>
        </Panel>
      )}
      <Panel title="run output (surogate stdout/stderr)">
        <Box flexDirection="column" marginTop={1}>
          {logLines.length === 0 ? (
            <Text color={C.dim}>no log yet — it appears once the run writes output</Text>
          ) : (
            logLines.map((l, i) => (
              <Text key={i} color={C.muted} wrap="truncate-end">
                {l}
              </Text>
            ))
          )}
        </Box>
      </Panel>
      <Panel title="metrics" flexGrow={1}>
        <Box flexDirection="column" marginTop={1}>
          {recent.length === 0 ? (
            <Text color={C.muted}>waiting for metrics…</Text>
          ) : (
            recent.map((r, i) => {
              const isBest = best !== null && r.trainLoss !== null && r.step === best.step;
              return (
                <Text key={i} color={isBest ? C.green : C.text}>
                  {isBest ? "★" : " "} step {String(r.step).padStart(4)}
                  {r.trainLoss !== null ? `  loss ${fmtFloat(r.trainLoss)}` : ""}
                  {r.evalLoss !== null ? `  eval ${fmtFloat(r.evalLoss)}` : ""}
                  {r.lr !== null ? `  lr ${fmtFloat(r.lr)}` : ""}
                  {r.tokensPerSecond !== null ? `  ${fmtCount(r.tokensPerSecond)} tok/s` : ""}
                  {r.phase ? `  ${r.phase}` : ""}
                </Text>
              );
            })
          )}
        </Box>
      </Panel>
    </Box>
  );
}

function FilesPage({
  runs,
  filesSel,
  active,
  currentFeed,
}: {
  runs: RunInfo[];
  filesSel: number;
  active: boolean;
  currentFeed: string;
}) {
  const sel = runs[filesSel];
  // runFileEntries does a recursive checkpoint-dir size walk — memoize so it
  // doesn't re-walk on every render tick/keystroke (only when the run changes).
  const entries = useMemo(() => (sel ? runFileEntries(sel.path) : []), [sel?.path, sel?.checkpoints.length, sel?.sizeBytes]);
  const files = entries.filter((e) => e.kind === "file");
  const cps = entries.filter((e) => e.kind === "checkpoint");
  const dir = sel ? sel.path.replace(/metrics\.jsonl$/, "") : "";
  return (
    <Panel title="files & checkpoints" flexGrow={1}>
      <Text color={C.muted}>
        {active ? "↑↓ choose run · y copy run folder · esc back" : "⏎ to browse every run's files & checkpoints"}
      </Text>
      {runs.length === 0 ? (
        <Box marginTop={1}>
          <Text color={C.muted}>no runs yet — launch one from the Launch tab</Text>
        </Box>
      ) : (
        <Box marginTop={1}>
          <Box flexDirection="column" width={40} marginRight={1}>
            {runs.map((r, i) => {
              const on = active && i === filesSel;
              const cur = r.path === currentFeed;
              const st = STATUS[r.status];
              return (
                <Text key={r.path} color={on ? C.accent : C.text} bold={on}>
                  {on ? "▸ " : "  "}
                  <Text color={statusColor(r.status)}>{st.glyph}</Text> {r.name.padEnd(24).slice(0, 24)}
                  <Text color={C.muted}> {`${r.checkpoints.length}ckpt`}</Text>
                  {cur ? <Text color={C.green}> ◀</Text> : null}
                </Text>
              );
            })}
          </Box>
          {sel && (
            <Box
              flexDirection="column"
              flexGrow={1}
              paddingLeft={1}
              borderStyle="single"
              borderColor={C.border}
              borderTop={false}
              borderBottom={false}
              borderRight={false}
            >
              <Text color={C.accent} bold>
                {sel.name}
              </Text>
              <Text color={C.dim}>{dir}</Text>
              <Box flexDirection="column" marginTop={1}>
                <Text color={C.dim}>{`CHECKPOINTS (${cps.length})`}</Text>
                {cps.length === 0 ? (
                  <Text color={C.dim}>none yet</Text>
                ) : (
                  cps.slice(0, 8).map((cp, i) => (
                    <Text key={cp.name} color={i === 0 ? C.text : C.muted}>
                      {i === 0 ? "◆ " : "· "}
                      {cp.name.padEnd(18)}
                      <Text color={C.eval}>{fmtBytes(cp.bytes)}</Text>
                      {i === 0 ? <Text color={C.dim}> (latest)</Text> : null}
                    </Text>
                  ))
                )}
                {cps.length > 8 && <Text color={C.dim}>{`  … +${cps.length - 8} more`}</Text>}
              </Box>
              <Box flexDirection="column" marginTop={1}>
                <Text color={C.dim}>FILES</Text>
                {files.map((f) => (
                  <Text key={f.name} color={C.muted}>
                    {"· "}
                    <Text color={C.text}>{f.name.padEnd(16)}</Text>
                    <Text color={C.muted}> {fmtBytes(f.bytes)}</Text>
                  </Text>
                ))}
              </Box>
            </Box>
          )}
        </Box>
      )}
    </Panel>
  );
}

export function Page({
  nav,
  s,
  chartImage,
  chartHeight,
  feedState,
  runs,
  runSel,
  runsActive,
  filesSel,
  filesActive,
  gpusActive,
  onGpusExit,
  currentFeed,
  compareFeed,
  events,
}: {
  nav: NavItem;
  s: WatchState;
  chartImage: string;
  chartHeight: number;
  feedState: FeedStatus;
  runs: RunInfo[];
  runSel: number;
  runsActive: boolean;
  filesSel: number;
  filesActive: boolean;
  gpusActive: boolean;
  onGpusExit: () => void;
  currentFeed: string;
  compareFeed: string | null;
  events: Alert[];
}) {
  switch (nav) {
    case "GPUs":
      return <GpusPage s={s} active={gpusActive} onExit={onGpusExit} />;
    case "Runs":
      return <RunsPage runs={runs} runSel={runSel} active={runsActive} currentFeed={currentFeed} compareFeed={compareFeed} />;
    case "Files":
      return <FilesPage runs={runs} filesSel={filesSel} active={filesActive} currentFeed={currentFeed} />;
    case "Logs":
      return <LogsPage s={s} events={events} feedPath={currentFeed} />;
    default:
      return (
        <MonitorPage s={s} chartImage={chartImage} chartHeight={chartHeight} feedState={feedState} feedPath={currentFeed} />
      );
  }
}
