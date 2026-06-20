import React, { useEffect, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { C } from "./theme.ts";
import { Spinner } from "./Spinner.tsx";
import {
  type DstackBackend,
  DSTACK_BACKENDS,
  deleteProvider,
  loadProviders,
  type ProviderRecord,
  providerId,
  saveProvider,
} from "../providers.ts";
import { detectRemoteGpus, parseSshTarget, rememberSshHost } from "../ssh.ts";
import { type BackendField, configureDstackBackend, dstackAvailable, DSTACK_BACKEND_FIELDS } from "../dstack.ts";
import { clientInstallCommand, modalAvailable, modalConnect, runShell } from "../setup.ts";

type FormKind = "ssh" | "modal" | DstackBackend;

// Which client a form needs installed (ssh needs none).
function clientFor(kind: FormKind): "modal" | "dstack" | null {
  if (kind === "ssh") return null;
  if (kind === "modal") return "modal";
  return "dstack"; // any cloud backend is provisioned via dstack
}
type View = { step: "list" } | { step: "kind" } | { step: "cloud" } | { step: "form"; kind: FormKind };

const KIND_GLYPH: Record<string, string> = { ssh: "⇄", modal: "▲", dstack: "☁" };

const SSH_FIELDS: BackendField[] = [{ key: "host", label: "user@host[:port]" }];
const MODAL_FIELDS: BackendField[] = [
  { key: "token_id", label: "token-id" },
  { key: "token_secret", label: "token-secret", secret: true },
];

function formSpec(kind: FormKind): BackendField[] {
  if (kind === "ssh") return SSH_FIELDS;
  if (kind === "modal") return MODAL_FIELDS;
  return DSTACK_BACKEND_FIELDS[kind] ?? [];
}
function formTitle(kind: FormKind): string {
  if (kind === "ssh") return "add SSH server";
  if (kind === "modal") return "connect Modal";
  return `connect ${DSTACK_BACKENDS.find((b) => b.id === kind)?.label ?? kind}`;
}

// Add a compute provider — SSH server, Modal workspace, or a dstack cloud backend.
// Connecting verifies + writes creds to each tool's OWN store (SSH keys,
// ~/.modal.toml, ~/.dstack); jackalope saves only non-secret metadata.
export function ProviderManager({ active, onExit }: { active: boolean; onExit: () => void }) {
  const [providers, setProviders] = useState<ProviderRecord[]>(() => loadProviders());
  const [view, setView] = useState<View>({ step: "list" });
  const [cursor, setCursor] = useState(0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; color: string } | null>(null);

  const [vals, setVals] = useState<Record<string, string>>({});
  const [fcur, setFcur] = useState(0);
  // client (modal/dstack) install state for the active form
  const [clientOk, setClientOk] = useState<boolean | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installLog, setInstallLog] = useState<string[]>([]);
  const flash = (text: string, color: string) => setMsg({ text, color });

  // When a form needing a client opens, check whether that client is installed.
  const formTool = view.step === "form" ? clientFor(view.kind) : null;
  useEffect(() => {
    if (!formTool) {
      setClientOk(null);
      return;
    }
    let cancelled = false;
    setClientOk(null);
    void (formTool === "modal" ? modalAvailable() : dstackAvailable()).then((ok) => {
      if (!cancelled) setClientOk(ok);
    });
    return () => {
      cancelled = true;
    };
  }, [formTool]);

  const installAbort = useRef(false);
  const installClient = (tool: "modal" | "dstack") => {
    installAbort.current = false;
    setInstalling(true);
    setInstallLog([]);
    flash(`installing ${tool}…`, C.muted);
    const h = runShell(clientInstallCommand(tool), (line) => setInstallLog((xs) => [...xs.slice(-7), line]));
    void h.done.then(({ ok }) => {
      if (installAbort.current) return; // user navigated away — don't flash/setState
      setInstalling(false);
      if (ok) {
        setClientOk(true);
        flash(`✓ ${tool} installed — now fill the fields and connect`, C.green);
      } else {
        flash(`✗ ${tool} install failed — see the lines above`, C.red);
      }
    });
  };
  const back = () => {
    if (installing) {
      installAbort.current = true; // stop a mid-install completion from flashing
      setInstalling(false);
    }
    setMsg(null);
    setView({ step: "list" });
    setCursor(0);
  };
  const done = (text: string) => {
    setBusy(false);
    setProviders(loadProviders());
    flash(text, C.green);
    setVals({});
    setFcur(0);
    setView({ step: "list" });
    setCursor(0);
  };

  const submit = (kind: FormKind) => {
    const v = (k: string) => (vals[k] ?? "").trim();
    if (kind === "ssh") {
      if (!v("host")) return;
      setBusy(true);
      flash("connecting…", C.muted);
      detectRemoteGpus(parseSshTarget(v("host")))
        .then((g) => {
          rememberSshHost(v("host"));
          saveProvider({ id: providerId("ssh", v("host")), kind: "ssh", label: v("host"), host: v("host"), gpu: g[0]?.name, addedAt: new Date().toISOString() });
          done(`✓ ${v("host")} — ${g.length} GPU${g.length === 1 ? "" : "s"}`);
        })
        .catch((e: Error) => {
          setBusy(false);
          flash(`✗ ${e.message.split("\n")[0] || "ssh failed"}`, C.red);
        });
      return;
    }
    if (kind === "modal") {
      if (!v("token_id") || !v("token_secret")) return;
      setBusy(true);
      flash("verifying token…", C.muted);
      void modalConnect(v("token_id"), v("token_secret")).then((r) => {
        if (!r.ok) {
          setBusy(false);
          return flash(`✗ ${r.reason}`, C.red);
        }
        saveProvider({ id: providerId("modal", v("token_id")), kind: "modal", label: "Modal", workspace: `${v("token_id").slice(0, 10)}…`, addedAt: new Date().toISOString() });
        done("✓ Modal connected");
      });
      return;
    }
    // dstack cloud backend
    const r = configureDstackBackend(kind, vals);
    if (!r.ok) return flash(`✗ ${r.reason}`, C.red);
    const label = DSTACK_BACKENDS.find((b) => b.id === kind)!.label;
    saveProvider({ id: providerId("dstack", kind), kind: "dstack", label, backend: kind, addedAt: new Date().toISOString() });
    done(`✓ ${label} connected — use it in Launch → Compute`);
  };

  useInput(
    (input, key) => {
      if (busy) return;

      if (view.step === "list") {
        if (key.escape || key.leftArrow) return onExit();
        if (key.upArrow) setCursor((c) => Math.max(0, c - 1));
        else if (key.downArrow) setCursor((c) => Math.min(Math.max(0, providers.length - 1), c + 1));
        else if (input === "a" || input === "n") {
          setMsg(null);
          setCursor(0);
          setView({ step: "kind" });
        } else if (input === "d" && providers[cursor]) {
          const p = providers[cursor]!;
          setProviders(deleteProvider(p.id));
          setCursor(0);
          flash(`removed ${p.label} (creds stay in the tool's own store)`, C.muted);
        }
        return;
      }

      if (key.escape) return back();

      if (view.step === "kind") {
        const kinds: FormKind[] = ["ssh", "modal"];
        const total = kinds.length + 1; // + cloud
        if (key.upArrow) setCursor((c) => (c - 1 + total) % total);
        else if (key.downArrow) setCursor((c) => (c + 1) % total);
        else if (key.return) {
          setVals({});
          setFcur(0);
          if (cursor < kinds.length) setView({ step: "form", kind: kinds[cursor]! });
          else {
            setCursor(0);
            setView({ step: "cloud" });
          }
        }
        return;
      }

      if (view.step === "cloud") {
        if (key.upArrow) setCursor((c) => (c - 1 + DSTACK_BACKENDS.length) % DSTACK_BACKENDS.length);
        else if (key.downArrow) setCursor((c) => (c + 1) % DSTACK_BACKENDS.length);
        else if (key.return) {
          setVals({});
          setFcur(0);
          setView({ step: "form", kind: DSTACK_BACKENDS[cursor]!.id });
        }
        return;
      }

      // view.step === "form"
      if (installing) return; // keys ignored while installing the client
      const tool = clientFor(view.kind);
      // Until a needed client is confirmed present, don't capture field input
      // (typed creds would be hidden when the check resolves "not installed").
      // While missing (clientOk false) only 'i' (install) is live; while checking
      // (null) all keys are ignored.
      if (tool && clientOk !== true) {
        if (clientOk === false && input === "i") installClient(tool);
        return;
      }
      const spec = formSpec(view.kind);
      const fld = spec[fcur]!;
      if (key.tab || key.downArrow) setFcur((c) => (c + 1) % spec.length);
      else if (key.upArrow) setFcur((c) => (c - 1 + spec.length) % spec.length);
      else if (key.return) {
        if (fcur < spec.length - 1) setFcur(fcur + 1);
        else submit(view.kind);
      } else if (key.backspace || key.delete) setVals((m) => ({ ...m, [fld.key]: (m[fld.key] ?? "").slice(0, -1) }));
      else if (input && !key.ctrl && !key.meta && input >= " ") setVals((m) => ({ ...m, [fld.key]: (m[fld.key] ?? "") + input }));
    },
    { isActive: active },
  );

  // ── render ──────────────────────────────────────────────────────────────────
  if (view.step === "list") {
    return (
      <Box flexDirection="column">
        <Text color={C.muted}>connected compute — SSH servers, Modal, and cloud backends</Text>
        <Box flexDirection="column" marginTop={1}>
          {providers.length === 0 ? (
            <Text color={C.dim}>none yet — press a to add an SSH server, Modal, or a cloud account</Text>
          ) : (
            providers.map((p, i) => (
              <Text key={p.id} color={active && i === cursor ? C.accent : C.text} bold={active && i === cursor}>
                {active && i === cursor ? "▸ " : "  "}
                <Text color={C.eval}>{KIND_GLYPH[p.kind] ?? "•"}</Text>
                {` ${p.label}`}
                <Text color={C.muted}>{p.host ? `  ${p.host}` : p.backend ? `  ${p.backend}` : p.workspace ? `  ${p.workspace}` : ""}</Text>
                {p.gpu && <Text color={C.dim}>{`  ${p.gpu}`}</Text>}
              </Text>
            ))
          )}
        </Box>
        <Box marginTop={1}>
          <Text color={C.dim}>{active ? "a add · d remove · ↑↓ · ← back" : "↑↓ to GPUs · p manage providers"}</Text>
        </Box>
        {msg && (
          <Text color={msg.color} wrap="truncate-end">
            {msg.text}
          </Text>
        )}
      </Box>
    );
  }

  if (view.step === "kind") {
    const rows = [
      { name: "SSH server", hint: "a remote GPU box you reach over SSH" },
      { name: "Modal", hint: "serverless GPUs · runs surogate in a sandbox" },
      { name: "Cloud (dstack)", hint: "RunPod / Lambda / Vast / AWS / GCP / Azure …" },
    ];
    return (
      <Box flexDirection="column">
        <Text color={C.muted}>add a provider</Text>
        <Box flexDirection="column" marginTop={1}>
          {rows.map((r, i) => (
            <Box key={r.name}>
              <Text color={i === cursor ? C.accent : C.dim}>{i === cursor ? "▸ " : "  "}</Text>
              <Text color={i === cursor ? C.accent : C.text} bold={i === cursor}>
                {r.name}
              </Text>
              <Text color={C.muted}>{`  — ${r.hint}`}</Text>
            </Box>
          ))}
        </Box>
        <Box marginTop={1}>
          <Text color={C.dim}>↑↓ · ⏎ choose · esc back</Text>
        </Box>
      </Box>
    );
  }

  if (view.step === "cloud") {
    return (
      <Box flexDirection="column">
        <Text color={C.muted}>pick a cloud backend (provisioned via dstack)</Text>
        <Box flexDirection="column" marginTop={1}>
          {DSTACK_BACKENDS.map((b, i) => (
            <Box key={b.id}>
              <Text color={i === cursor ? C.accent : C.dim}>{i === cursor ? "▸ " : "  "}</Text>
              <Text color={i === cursor ? C.accent : C.text} bold={i === cursor}>
                {b.label}
              </Text>
              <Text color={C.muted}>{`  — needs ${b.needs.join(", ")}`}</Text>
            </Box>
          ))}
        </Box>
        <Box marginTop={1}>
          <Text color={C.dim}>↑↓ · ⏎ choose · esc back</Text>
        </Box>
      </Box>
    );
  }

  // view.step === "form"
  const spec = formSpec(view.kind);
  const tool = clientFor(view.kind);
  return (
    <Box flexDirection="column">
      <Text color={C.text} bold>
        {formTitle(view.kind)}
      </Text>
      {tool && clientOk === false ? (
        // the client isn't installed → offer to install it (uv, else pip)
        <Box flexDirection="column" marginTop={1}>
          <Text color={C.warm}>● {tool} client not installed</Text>
          {installing ? (
            <Box flexDirection="column" marginTop={1}>
              <Text>
                <Spinner color={C.accent} />
                <Text color={C.muted}>{` installing ${tool}…`}</Text>
              </Text>
              {installLog.map((l, i) => (
                <Text key={i} color={C.dim} wrap="truncate-end">
                  {l}
                </Text>
              ))}
            </Box>
          ) : (
            <Box marginTop={1}>
              <Text color={C.gold} bold>
                i
              </Text>
              <Text color={C.dim}> install it for me (uv, else pip) · esc back</Text>
            </Box>
          )}
        </Box>
      ) : tool && clientOk === null ? (
        <Box marginTop={1}>
          <Spinner color={C.dim} />
          <Text color={C.muted}>{` checking ${tool}…`}</Text>
        </Box>
      ) : (
        <>
          <Box flexDirection="column" marginTop={1}>
            {spec.map((fld, i) => {
              const raw = vals[fld.key] ?? "";
              const shown = fld.secret ? "•".repeat(raw.length) : raw;
              return (
                <Text key={fld.key}>
                  <Text color={C.dim}>{fld.label.padEnd(22)}</Text>
                  <Text color={i === fcur ? C.gold : C.text}>{shown}</Text>
                  {i === fcur && <Text color={C.gold}>█</Text>}
                  {fld.file && i === fcur && <Text color={C.dim}>{"  (path to file)"}</Text>}
                </Text>
              );
            })}
          </Box>
          <Box marginTop={1}>
            {busy ? (
              <Text>
                <Spinner color={C.accent} />
                <Text color={C.muted}> working…</Text>
              </Text>
            ) : (
              <Text color={C.dim}>{spec.length > 1 ? "tab/↑↓ next field · " : ""}⏎ {fcur < spec.length - 1 ? "next" : "connect"} · esc back</Text>
            )}
          </Box>
        </>
      )}
      {msg && (
        <Text color={msg.color} wrap="truncate-end">
          {msg.text}
        </Text>
      )}
    </Box>
  );
}
