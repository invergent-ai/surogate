import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import {
  datasetDetail,
  type DatasetDetail,
  hasHfToken,
  type HfHit,
  modelDetail,
  type ModelDetail,
  paramsB,
  searchDatasets,
  searchModels,
  validateAndSaveHfToken,
} from "../hf.ts";
import { checkTrainable, type Trainability } from "../supported.ts";
import { localDatasetInfo, localModelInfo, looksLikePath } from "../local.ts";
import { fmtBytes, fmtCount } from "../format.ts";
import { C } from "./theme.ts";
import { Panel } from "./Panel.tsx";
import { Spinner } from "./Spinner.tsx";

// HF card data is user-authored and can carry newlines, carriage returns, or
// stray ANSI/control chars that corrupt the terminal grid — they render as
// "black" cells that cut across the panel. Flatten anything Hub-sourced to a
// single clean line before it touches the layout.
function clean(s: string | undefined | null): string {
  if (!s) return "";
  return s
    .replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "") // ANSI escapes
    // Zero-width + bidirectional-override chars: invisible to the user but they
    // desync the terminal's column count (Ink's width math goes wrong → the row's
    // background doesn't fill → "black" gaps that cut across the panel). RTL/LTR
    // overrides (U+202A–202E, U+2066–2069) are the worst offenders.
    .replace(/[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]/g, "")
    .replace(/[\x00-\x1f\x7f]+/g, " ") // control chars
    .replace(/\s+/g, " ") // collapse whitespace/newlines to single spaces
    .trim();
}

// HF's `gated` field is `false` | "false" | true | "auto" | "manual" — anything
// truthy that isn't the string "false" means access is gated.
function isGated(g: boolean | string | undefined): boolean {
  return !!g && g !== "false";
}

// Fetch the highlighted row's detail object: serve from cache instantly, else
// debounce (so arrowing through a list doesn't spam the Hub) and abort on
// change. Disabled (or no id) clears the pane. Both panes share this; only the
// fetcher differs (models also resolve trainability).
function useDetailFetch<T>(
  enabled: boolean,
  id: string | undefined,
  cache: React.MutableRefObject<Map<string, T>>,
  fetcher: (id: string, signal: AbortSignal) => Promise<T>,
  set: (v: T | null) => void,
): void {
  useEffect(() => {
    if (!enabled || !id) {
      set(null);
      return;
    }
    const cached = cache.current.get(id);
    if (cached) {
      set(cached);
      return;
    }
    const ctrl = new AbortController();
    set(null);
    const timer = setTimeout(async () => {
      try {
        const v = await fetcher(id, ctrl.signal);
        cache.current.set(id, v);
        set(v);
      } catch {
        /* ignore detail errors */
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
    // fetcher/set/cache are stable for our purposes — re-run only on id/enabled.
  }, [enabled, id]);
}

// Models also resolve trainability alongside the raw detail.
async function modelEntry(id: string, signal: AbortSignal): Promise<{ d: ModelDetail; t: Trainability }> {
  const d = await modelDetail(id, signal);
  return { d, t: checkTrainable(d.architectures, d.modelType) };
}

export function Browse({
  kind,
  active,
  picked,
  query,
  setQuery,
  onExit,
  onPickModel,
  onPickDataset,
  onRequestModel,
}: {
  kind: "models" | "datasets";
  active: boolean;
  picked?: { model?: { id: string; t: { supported: boolean } }; dataset?: string };
  query: string;
  setQuery: React.Dispatch<React.SetStateAction<string>>;
  onExit: () => void;
  onPickModel: (id: string, t: Trainability) => void;
  onPickDataset: (id: string) => void;
  onRequestModel: (id: string, arch: string | null) => void;
}) {
  const [hits, setHits] = useState<HfHit[]>([]);
  const [sel, setSel] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errMsg, setErrMsg] = useState("");
  const [detail, setDetail] = useState<{ d: ModelDetail; t: Trainability } | null>(null);
  const detailCache = useRef(new Map<string, { d: ModelDetail; t: Trainability }>());
  const [dsDetail, setDsDetail] = useState<DatasetDetail | null>(null);
  const dsCache = useRef(new Map<string, DatasetDetail>());
  // HF token entry (^T) — authed requests dodge the anonymous rate limit.
  const [tokenMode, setTokenMode] = useState(false);
  const [tokenBuf, setTokenBuf] = useState("");
  const [tokenMsg, setTokenMsg] = useState<string | null>(null);
  const [reloadTick, setReloadTick] = useState(0); // bump to re-run the search (e.g. after saving a token)
  const submitToken = () => {
    setTokenMsg("validating…");
    void validateAndSaveHfToken(tokenBuf).then((r) => {
      if (r.ok) {
        setTokenMode(false);
        setTokenBuf("");
        setTokenMsg(null);
        setReloadTick((n) => n + 1); // refetch now that requests are authed (hf.ts has cached the token)
      } else {
        setTokenMsg(`✗ ${r.reason}`);
      }
    });
  };

  // A path-like query (/…, ~/…, ./…) is resolved on disk instead of querying the
  // Hub, so you can train a local model/dataset by just typing its path. The view
  // switches instantly (isLocal), but the on-disk inspection (statSync + reading
  // config.json + safetensors sizes) is debounced so it doesn't run per-keystroke.
  const isLocal = looksLikePath(query);
  const [debouncedQuery, setDebouncedQuery] = useState(query);
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 200);
    return () => clearTimeout(t);
  }, [query]);
  const localModel = useMemo(
    () => (isLocal && kind === "models" ? localModelInfo(debouncedQuery) : null),
    [isLocal, kind, debouncedQuery],
  );
  const localDataset = useMemo(
    () => (isLocal && kind === "datasets" ? localDatasetInfo(debouncedQuery) : null),
    [isLocal, kind, debouncedQuery],
  );

  // debounced search
  useEffect(() => {
    if (isLocal) {
      setStatus("idle");
      setHits([]);
      return;
    }
    const ctrl = new AbortController();
    const q = query.trim();
    setStatus("loading");
    const timer = setTimeout(async () => {
      try {
        const fn = kind === "models" ? searchModels : searchDatasets;
        const res = await fn(q || "the", { limit: 30, signal: ctrl.signal });
        setHits(res);
        setSel(0);
        setStatus("idle");
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        setErrMsg((e as Error).message);
        setStatus("error");
      }
    }, 300);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [query, kind, isLocal, reloadTick]);

  // Full details for the highlighted row, shown inline in the right pane (no
  // separate "expand" step). Keyed on the selected id so re-ordering the results
  // without changing the selection doesn't refetch.
  const selId = hits[sel]?.id;
  useDetailFetch(!isLocal && kind === "models" && selId !== undefined, selId, detailCache, modelEntry, setDetail);
  useDetailFetch(!isLocal && kind === "datasets" && selId !== undefined, selId, dsCache, datasetDetail, setDsDetail);

  const useCurrent = async () => {
    if (isLocal) {
      if (kind === "models" && localModel?.exists && localModel.isDir) onPickModel(localModel.path, localModel.trainability);
      else if (kind === "datasets" && localDataset?.exists) onPickDataset(localDataset.path);
    } else if (hits[sel]) {
      if (kind === "datasets") return onPickDataset(hits[sel]!.id);
      const id = hits[sel]!.id;
      let t = detail?.t;
      // Per-row fetches are shallow; if support is still unknown, do the deeper
      // (config.json / adapter-base) detection now — once, on the explicit pick.
      if (!t || (!t.supported && !t.known)) {
        try {
          const d = await modelDetail(id, undefined, true);
          t = checkTrainable(d.architectures, d.modelType);
          detailCache.current.set(id, { d, t }); // cache the deep result so re-picking doesn't re-fetch
        } catch {
          /* keep the shallow verdict */
        }
      }
      onPickModel(id, t ?? checkTrainable(undefined, undefined));
    }
  };

  // Request support for the highlighted model (only meaningful when it isn't
  // already trainable). Opens a prefilled enhancement issue on the surogate repo.
  const requestCurrent = () => {
    if (kind !== "models") return;
    const id = hits[sel]?.id;
    if (!id || detail?.t.supported) return;
    onRequestModel(id, detail?.d.architectures?.[0] ?? detail?.d.modelType ?? null);
  };

  useInput(
    (input, key) => {
      // HF token entry sub-mode takes over the input line.
      if (tokenMode) {
        if (key.escape) {
          setTokenMode(false);
          setTokenBuf("");
          setTokenMsg(null);
        } else if (key.return) submitToken();
        else if (key.backspace || key.delete) setTokenBuf((b) => b.slice(0, -1));
        else if (input && !key.ctrl && !key.meta && input >= " ") setTokenBuf((b) => b + input);
        return;
      }
      // ← / esc goes back to the nav menu
      if (key.leftArrow || key.escape) return onExit();
      // ctrl-t opens the HF token prompt (plain t must type into the search)
      if (key.ctrl && (input === "t" || input === "T")) {
        setTokenMsg(null);
        return setTokenMode(true);
      }
      if (key.return) void useCurrent();
      // ctrl-r requests an unsupported model (plain r must type into the search)
      else if (key.ctrl && (input === "r" || input === "R")) requestCurrent();
      else if (key.upArrow) setSel((i) => Math.max(0, i - 1));
      else if (key.downArrow) setSel((i) => Math.min(Math.max(0, hits.length - 1), i + 1));
      else if (key.backspace || key.delete) setQuery((q) => q.slice(0, -1));
      else if (input && !key.ctrl && !key.meta && input >= " ") setQuery((q) => q + input);
    },
    { isActive: active },
  );

  // gated model whose config was withheld → we can't judge support, so say
  // "gated" rather than "not supported".
  const gatedUnknown = !!detail && !detail.d.architectures && !detail.d.modelType && isGated(detail.d.gated);

  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title={kind === "models" ? "search models · huggingface" : "search datasets · huggingface"} flexGrow={1}>
        {tokenMode ? (
          <Box marginTop={1} flexDirection="column">
            <Text>
              <Text color={C.muted}>HF token </Text>
              <Text color={C.gold}>{"•".repeat(Math.min(tokenBuf.length, 24))}</Text>
              <Text color={C.gold}>█</Text>
              <Text color={C.dim}>{"   ⏎ save · esc cancel"}</Text>
            </Text>
            <Text color={C.dim}>paste a token from huggingface.co/settings/tokens (saved to ~/.cache/huggingface/token)</Text>
            {tokenMsg && <Text color={tokenMsg.startsWith("✗") ? C.red : C.dim}>{tokenMsg}</Text>}
          </Box>
        ) : (
          <>
            <Box marginTop={1}>
              <Text color={C.muted}>search </Text>
              <Text color={C.gold}>{query}</Text>
              <Text color={active ? C.gold : C.dim}>{active ? "█" : ""}</Text>
              <Text color={C.dim}>{"   "}</Text>
              {!isLocal && status === "loading" ? (
                <Spinner color={C.dim} />
              ) : (
                <Text color={C.dim}>{isLocal ? "local path" : status === "error" ? `⚠ ${errMsg}` : `${hits.length} results`}</Text>
              )}
            </Box>
            {query === "" && (
              <Text color={C.dim}>tip: type a path (/…, ~/…, ./…) to use a local {kind === "models" ? "model" : "dataset"}</Text>
            )}
            {!hasHfToken() && (
              <Text color={C.warm}>
                ⚠ no HF token — <Text color={C.gold}>^T</Text>
                <Text color={C.dim}> to add one (avoids HuggingFace rate limits)</Text>
              </Text>
            )}
          </>
        )}
        {isLocal ? (
          <Box marginTop={1} flexGrow={1}>
            {kind === "models" && localModel && <LocalModelView m={localModel} />}
            {kind === "datasets" && localDataset && <LocalDatasetView d={localDataset} />}
          </Box>
        ) : (
          <Box marginTop={1} flexGrow={1}>
            {/* left: results list (fixed width keeps the divider plumb) */}
            <Box flexDirection="column" width={46}>
              {hits.slice(0, 14).map((h, i) => {
                const on = i === sel;
                const t = kind === "models" && detailCache.current.get(h.id)?.t;
                const mark = listBadge(t || undefined);
                return (
                  <Text key={h.id} color={on ? C.accent : C.text} bold={on} wrap="truncate">
                    {on ? "▸ " : "  "}
                    {kind === "models" ? <Text color={mark.color}>{mark.glyph} </Text> : null}
                    {kind === "models" ? (
                      h.id.slice(0, 42)
                    ) : (
                      <>
                        {h.id.slice(0, 30).padEnd(30)}
                        <Text color={C.muted}> ↓{fmtCount(h.downloads ?? null)} ♥{fmtCount(h.likes ?? null)}</Text>
                      </>
                    )}
                  </Text>
                );
              })}
              {hits.length === 0 && status !== "loading" && <Text color={C.muted}>no results</Text>}
            </Box>
            {/* right: full details, same box, a divider then the detail "window" */}
            <Box
              flexDirection="column"
              flexGrow={1}
              paddingLeft={2}
              borderStyle="single"
              borderColor={C.border}
              borderTop={false}
              borderBottom={false}
              borderRight={false}
            >
              {kind === "models" ? (
                <ModelPane entry={detail} id={hits[sel]?.id ?? ""} gatedUnknown={gatedUnknown} />
              ) : (
                <DatasetPane d={dsDetail} id={hits[sel]?.id ?? ""} />
              )}
            </Box>
          </Box>
        )}
        {(picked?.model || picked?.dataset) && <RunSoFar picked={picked!} />}
      </Panel>
    </Box>
  );
}

// A clean "what you've picked so far" strip at the bottom of the browser, so the
// run you're assembling is always visible while you search models + datasets.
function RunSoFar({ picked }: { picked: { model?: { id: string; t: { supported: boolean } }; dataset?: string } }) {
  return (
    <Box marginTop={1} flexDirection="column">
      <Text color={C.dim}>YOUR RUN</Text>
      <Text>
        <Text color={C.muted}>model    </Text>
        {picked.model ? (
          <Text color={picked.model.t.supported ? C.green : C.red}>✓ {picked.model.id}</Text>
        ) : (
          <Text color={C.dim}>— pick one (⏎) —</Text>
        )}
      </Text>
      <Text>
        <Text color={C.muted}>dataset  </Text>
        {picked.dataset ? <Text color={C.eval}>✓ {picked.dataset}</Text> : <Text color={C.dim}>— pick one (⏎) —</Text>}
      </Text>
    </Box>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Text color={C.muted}>
      {label.padEnd(13)}
      {children}
    </Text>
  );
}

function gatedText(g: boolean | string | undefined): string {
  return isGated(g) ? "yes — needs HF access" : "no";
}

// Shared trainability badge (gatedUnknown only applies to Hub models whose
// config was withheld; local models can't be gated). `known: false` means we
// couldn't read the config — show a neutral "?", not a red "not supported".
// Compact one-glyph trainability marker for list rows. Quieter than SupportBadge:
// a known-but-unsupported model is a muted dot here, not a loud ✗.
function listBadge(t?: Trainability): { glyph: string; color: string } {
  if (!t) return { glyph: " ", color: C.dim };
  if (t.supported) return { glyph: "✓", color: C.green };
  if (t.known) return { glyph: "·", color: C.dim };
  return { glyph: "?", color: C.warm };
}

function SupportBadge({ supported, known = true, gatedUnknown }: { supported: boolean; known?: boolean; gatedUnknown?: boolean }) {
  if (supported) return <Text color={C.green}>✓ trainable by surogate</Text>;
  if (gatedUnknown) return <Text color={C.warm}>⚷ gated — accept the license on HF to check support</Text>;
  if (!known) return <Text color={C.warm}>? couldn't read config — it may still train (⏎ to use)</Text>;
  return <Text color={C.red}>✗ not supported</Text>;
}

// Stack into fixed rows where EACH row is individually truncate-end. Never use
// wrap="wrap" on Hub-sourced text: Ink's column math can't be trusted on
// arbitrary unicode, and an over-wide line leaves an unpainted ("black") gap. A
// per-row truncate is always bounded, so the panel background always fills.
function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}

function FileList({ files }: { files?: string[] }) {
  if (!files || files.length === 0) return null;
  const shown = files.slice(0, 10).map(clean);
  return (
    <Box marginTop={1} flexDirection="column">
      <Text color={C.gold} bold>
        FILES ({files.length})
      </Text>
      {shown.map((f, i) => (
        <Text key={i} color={C.text} wrap="truncate-end">
          {"  "}
          {f}
        </Text>
      ))}
      {files.length > shown.length && <Text color={C.dim}>{`  +${files.length - shown.length} more`}</Text>}
    </Box>
  );
}

function TagList({ tags }: { tags?: string[] }) {
  if (!tags || tags.length === 0) return null;
  const rows = chunk(tags.slice(0, 12).map(clean), 3); // a few tags per row, stacked
  return (
    <Box marginTop={1} flexDirection="column">
      <Text color={C.muted}>tags</Text>
      {rows.map((row, i) => (
        <Text key={i} color={C.text} wrap="truncate-end">
          {row.join("  ·  ")}
        </Text>
      ))}
    </Box>
  );
}

// Full model details, shown inline in the right pane (no separate window).
function ModelPane({
  entry,
  id,
  gatedUnknown,
}: {
  entry: { d: ModelDetail; t: Trainability } | null;
  id: string;
  gatedUnknown: boolean;
}) {
  if (id === "") return <Text color={C.muted}>↑↓ to inspect a model</Text>;
  if (!entry) {
    return (
      <Box flexDirection="column">
        <Text color={C.text} bold wrap="wrap">
          {id}
        </Text>
        <Box marginTop={1}>
          <Spinner color={C.dim} />
          <Text color={C.muted}> loading details…</Text>
        </Box>
      </Box>
    );
  }
  const pb = paramsB(entry.d);
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text color={C.text} bold wrap="truncate">
        {entry.d.id}
      </Text>
      <Box marginTop={1}>
        <SupportBadge supported={entry.t.supported} known={entry.t.known} gatedUnknown={gatedUnknown} />
      </Box>
      {!gatedUnknown && <Text color={C.muted} wrap="truncate-end">{clean(entry.t.reason)}</Text>}
      <Box marginTop={1} flexDirection="column">
        <Fact label="architecture">
          <Text color={C.text}>{entry.d.architectures?.[0] ?? entry.d.modelType ?? "—"}</Text>
        </Fact>
        <Fact label="model type">
          <Text color={C.text}>{entry.d.modelType ?? "—"}</Text>
        </Fact>
        <Fact label="parameters">
          <Text color={C.text}>{pb !== null ? `~${pb}B` : "—"}</Text>
        </Fact>
        {entry.t.supported && (
          <Fact label="recipes">
            <Text color={C.gold}>{entry.t.recipes.join(" · ")}</Text>
          </Fact>
        )}
        <Fact label="pipeline">
          <Text color={C.text}>{entry.d.pipelineTag ?? "—"}</Text>
        </Fact>
        <Fact label="library">
          <Text color={C.text}>{entry.d.libraryName ?? "—"}</Text>
        </Fact>
        <Fact label="downloads">
          <Text color={C.text}>{fmtCount(entry.d.downloads ?? null)}</Text>
          <Text color={C.muted}>{"     likes "}</Text>
          <Text color={C.text}>{fmtCount(entry.d.likes ?? null)}</Text>
        </Fact>
        <Fact label="gated">
          <Text color={isGated(entry.d.gated) ? C.warm : C.text}>{gatedText(entry.d.gated)}</Text>
        </Fact>
        {entry.d.lastModified && (
          <Fact label="updated">
            <Text color={C.text}>{entry.d.lastModified.slice(0, 10)}</Text>
          </Fact>
        )}
      </Box>
      <TagList tags={entry.d.tags} />
      <FileList files={entry.d.files} />
      {entry.t.supported ? (
        <Box marginTop={1}>
          <Text color={C.gold} bold>
            ⏎ use this model in Launch
          </Text>
        </Box>
      ) : !gatedUnknown && entry.t.known ? (
        <Box marginTop={1} flexDirection="column">
          <Text>
            <Text color={C.eval} bold>
              ctrl-r
            </Text>
            <Text color={C.eval}> — request this model on surogate</Text>
          </Text>
          <Text color={C.dim} wrap="truncate-end">opens a prefilled GitHub issue you can submit in one click</Text>
        </Box>
      ) : null}
    </Box>
  );
}

// Full dataset details, shown inline in the right pane.
function DatasetPane({ d, id }: { d: DatasetDetail | null; id: string }) {
  if (id === "") return <Text color={C.muted}>↑↓ to inspect a dataset</Text>;
  if (!d) {
    return (
      <Box flexDirection="column">
        <Text color={C.text} bold wrap="wrap">
          {id}
        </Text>
        <Box marginTop={1}>
          <Spinner color={C.dim} />
          <Text color={C.muted}> loading details…</Text>
        </Box>
      </Box>
    );
  }
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text color={C.text} bold wrap="truncate">
        {d.id}
      </Text>
      <Box marginTop={1} flexDirection="column">
        {d.description && (
          <Fact label="name">
            <Text color={C.text}>{clean(d.description)}</Text>
          </Fact>
        )}
        <Fact label="downloads">
          <Text color={C.text}>{fmtCount(d.downloads ?? null)}</Text>
          <Text color={C.muted}>{"     likes "}</Text>
          <Text color={C.text}>{fmtCount(d.likes ?? null)}</Text>
        </Fact>
        <Fact label="gated">
          <Text color={isGated(d.gated) ? C.warm : C.text}>{gatedText(d.gated)}</Text>
        </Fact>
        {d.lastModified && (
          <Fact label="updated">
            <Text color={C.text}>{d.lastModified.slice(0, 10)}</Text>
          </Fact>
        )}
      </Box>
      <TagList tags={d.tags} />
      <FileList files={d.files} />
      <Box marginTop={1}>
        <Text color={C.gold} bold>
          ⏎ use this dataset in Launch
        </Text>
      </Box>
    </Box>
  );
}

function LocalModelView({ m }: { m: import("../local.ts").LocalModel }) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text color={C.muted}>local model</Text>
      <Text color={C.text} bold>
        {m.path}
      </Text>
      {!m.exists || m.error ? (
        <Text color={C.red}>✗ {m.error ?? "path not found"}</Text>
      ) : (
        <>
          <SupportBadge supported={m.trainability.supported} />
          <Text color={C.muted}>{clean(m.trainability.reason)}</Text>
          <Box marginTop={1} flexDirection="column">
            <Text color={C.muted}>
              arch <Text color={C.text}>{m.architectures?.[0] ?? m.modelType ?? "—"}</Text>
            </Text>
            {m.paramsB !== null && (
              <Text color={C.muted}>
                params <Text color={C.text}>~{m.paramsB}B</Text>
              </Text>
            )}
          </Box>
          <Box marginTop={1}>
            <Text color={C.gold}>⏎ use this local model in Launch</Text>
          </Box>
        </>
      )}
    </Box>
  );
}

function LocalDatasetView({ d }: { d: import("../local.ts").LocalDataset }) {
  return (
    <Box flexDirection="column" flexGrow={1}>
      <Text color={C.muted}>local dataset</Text>
      <Text color={C.text} bold>
        {d.path}
      </Text>
      {!d.exists ? (
        <Text color={C.red}>✗ {d.error ?? "path not found"}</Text>
      ) : (
        <>
          <Text color={C.muted}>
            {d.kind === "dir" ? "directory · load_from_disk / load_dataset" : `file · ${d.dsType}`} · {fmtBytes(d.sizeBytes)}
          </Text>
          <Box marginTop={1}>
            <Text color={C.gold}>⏎ use this local dataset in Launch</Text>
          </Box>
        </>
      )}
    </Box>
  );
}
