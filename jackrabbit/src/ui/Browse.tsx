import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { type HfHit, modelDetail, type ModelDetail, paramsB, searchDatasets, searchModels } from "../hf.ts";
import { checkTrainable, type Trainability } from "../supported.ts";
import { C } from "./theme.ts";
import { Panel } from "./Panel.tsx";

function fmtN(n?: number): string {
  if (n === undefined) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

export function Browse({
  kind,
  active,
  onPickModel,
  onPickDataset,
}: {
  kind: "models" | "datasets";
  active: boolean;
  onPickModel: (id: string, t: Trainability) => void;
  onPickDataset: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<HfHit[]>([]);
  const [sel, setSel] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [errMsg, setErrMsg] = useState("");
  const [detail, setDetail] = useState<{ d: ModelDetail; t: Trainability } | null>(null);
  const detailCache = useRef(new Map<string, { d: ModelDetail; t: Trainability }>());

  // debounced search
  useEffect(() => {
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
  }, [query, kind]);

  // model detail for the highlighted row
  useEffect(() => {
    if (kind !== "models" || hits.length === 0) {
      setDetail(null);
      return;
    }
    const id = hits[sel]?.id;
    if (!id) return;
    if (detailCache.current.has(id)) {
      setDetail(detailCache.current.get(id)!);
      return;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const d = await modelDetail(id, ctrl.signal);
        const t = checkTrainable(d.architectures, d.modelType);
        const entry = { d, t };
        detailCache.current.set(id, entry);
        setDetail(entry);
      } catch {
        /* ignore detail errors */
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [sel, hits, kind]);

  useInput(
    (input, key) => {
      if (key.upArrow) setSel((i) => Math.max(0, i - 1));
      else if (key.downArrow) setSel((i) => Math.min(Math.max(0, hits.length - 1), i + 1));
      else if (key.return && hits[sel]) {
        if (kind === "models") onPickModel(hits[sel]!.id, detail?.t ?? checkTrainable(undefined, undefined));
        else onPickDataset(hits[sel]!.id);
      } else if (key.backspace || key.delete) setQuery((q) => q.slice(0, -1));
      else if (input && !key.ctrl && !key.meta && input >= " ") setQuery((q) => q + input);
    },
    { isActive: active },
  );

  return (
    <Box flexDirection="column" flexGrow={1}>
      <Panel title={kind === "models" ? "search models · huggingface" : "search datasets · huggingface"} flexGrow={1}>
        <Box marginTop={1}>
          <Text color={C.muted}>search </Text>
          <Text color={C.gold}>{query}</Text>
          <Text color={active ? C.gold : C.dim}>{active ? "█" : ""}</Text>
          <Text color={C.dim}>
            {"   "}
            {status === "loading" ? "…" : status === "error" ? `⚠ ${errMsg}` : `${hits.length} results`}
          </Text>
        </Box>
        <Box marginTop={1} flexGrow={1}>
          {/* results */}
          <Box flexDirection="column" width={kind === "models" ? 44 : undefined} flexGrow={kind === "datasets" ? 1 : undefined}>
            {hits.slice(0, 14).map((h, i) => {
              const on = i === sel;
              const t = kind === "models" && detailCache.current.get(h.id)?.t;
              const badge = t ? (t.supported ? "✓" : "·") : " ";
              return (
                <Text key={h.id} color={on ? C.accent : C.text} bold={on}>
                  {on ? "▸ " : "  "}
                  {kind === "models" ? (
                    <Text color={t ? (t.supported ? C.green : C.dim) : C.dim}>{badge} </Text>
                  ) : null}
                  {h.id.slice(0, kind === "models" ? 30 : 44).padEnd(kind === "models" ? 30 : 44)}
                  <Text color={C.muted}> ↓{fmtN(h.downloads)} ♥{fmtN(h.likes)}</Text>
                </Text>
              );
            })}
            {hits.length === 0 && status !== "loading" && <Text color={C.muted}>no results</Text>}
          </Box>
          {/* model detail */}
          {kind === "models" && (
            <Box flexDirection="column" flexGrow={1} marginLeft={1} paddingLeft={1}>
              {detail ? (
                <>
                  <Text color={C.text} bold>
                    {detail.d.id}
                  </Text>
                  <Text>
                    {detail.t.supported ? (
                      <Text color={C.green}>✓ trainable by surogate</Text>
                    ) : (
                      <Text color={C.red}>✗ not supported</Text>
                    )}
                  </Text>
                  <Text color={C.muted}>{detail.t.reason}</Text>
                  <Box marginTop={1} flexDirection="column">
                    <Text color={C.muted}>
                      arch <Text color={C.text}>{detail.d.architectures?.[0] ?? detail.d.modelType ?? "—"}</Text>
                    </Text>
                    {paramsB(detail.d) !== null && (
                      <Text color={C.muted}>
                        params <Text color={C.text}>~{paramsB(detail.d)}B</Text>
                      </Text>
                    )}
                    {detail.t.supported && (
                      <Text color={C.muted}>
                        recipes <Text color={C.gold}>{detail.t.recipes.join(" · ")}</Text>
                      </Text>
                    )}
                    {detail.d.gated && detail.d.gated !== "false" && <Text color={C.warm}>⚷ gated — needs HF access</Text>}
                  </Box>
                  {detail.t.supported && (
                    <Box marginTop={1}>
                      <Text color={C.gold}>⏎ use this model in Launch</Text>
                    </Box>
                  )}
                </>
              ) : (
                <Text color={C.muted}>↑↓ to inspect a model</Text>
              )}
            </Box>
          )}
        </Box>
      </Panel>
    </Box>
  );
}
