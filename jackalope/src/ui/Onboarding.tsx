import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { discoverGpus, type Gpu } from "../launch.ts";
import { detectRemoteGpus, parseSshTarget, recentSshHosts, rememberSshHost, type SshTarget } from "../ssh.ts";
import { dstackAvailable } from "../dstack.ts";
import { gpuSupport } from "../gpu-support.ts";
import {
  type BinCheck,
  type Compute,
  dstackInstallPlan,
  type InstallPlan,
  localInstallPlans,
  remoteInstallPlan,
  type RunHandle,
  runShell,
  runShellRemote,
  surogateAvailable,
  surogateAvailableRemote,
} from "../setup.ts";
import { copyToClipboard } from "../links.ts";
import { C } from "./theme.ts";
import { Panel } from "./Panel.tsx";
import { Spinner } from "./Spinner.tsx";
import { goldAt } from "./brand.ts";

type Step = "compute" | "sshhost" | "install" | "ready";

// Three-step rail; the sshhost sub-step folds under "Compute" so it stays 1-2-3.
const STEP_NAMES = ["Compute", "Get surogate", "Ready"];
const stepIndex = (s: Step): number => (s === "compute" || s === "sshhost" ? 0 : s === "install" ? 1 : 2);

// ── compute discovery → one-line annotation per option ────────────────────────
function gpuSummary(gpus: Gpu[]): string {
  if (gpus.length === 0) return "no NVIDIA GPUs detected here";
  const byName = new Map<string, number>();
  for (const g of gpus) byName.set(g.name, (byName.get(g.name) ?? 0) + 1);
  const names = [...byName.entries()].map(([n, c]) => `${c}× ${n}`).join(" · ");
  const idle = gpus.filter((g) => !g.busy).length;
  const sup = gpus[0]!.sm !== null ? gpuSupport(gpus[0]!.sm) : null;
  return `${names} · ${idle}/${gpus.length} idle${sup ? ` · ${sup.arch} · ${sup.recipes.join("/")}` : ""}`;
}

// ── streaming installer hook ──────────────────────────────────────────────────
type InstPhase = "idle" | "confirm" | "running" | "ok" | "fail";

function useInstaller(verify: () => Promise<BinCheck>) {
  const [phase, setPhase] = useState<InstPhase>("idle");
  const [lines, setLines] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<BinCheck | null>(null);
  const handleRef = useRef<RunHandle | null>(null);
  const startedAt = useRef(0);
  const lastLine = useRef(0);

  const append = (l: string) => {
    lastLine.current = Date.now();
    setLines((xs) => [...xs.slice(-199), l]);
  };

  // elapsed counter + heartbeat: a long-quiet run should never look frozen.
  useEffect(() => {
    if (phase !== "running") return;
    const t = setInterval(() => {
      const secs = Math.floor((Date.now() - startedAt.current) / 1000);
      setElapsed(secs);
      if (Date.now() - lastLine.current > 20000) append(`… still working (${secs}s)`);
    }, 1000);
    return () => clearInterval(t);
  }, [phase]);

  // kill any in-flight install if the wizard unmounts mid-run
  useEffect(() => () => handleRef.current?.cancel(), []);

  const arm = () => setPhase("confirm");
  const run = (start: (onLine: (l: string) => void) => RunHandle) => {
    setLines([]);
    setResult(null);
    setElapsed(0);
    setPhase("running");
    startedAt.current = Date.now();
    lastLine.current = Date.now();
    handleRef.current = start(append);
    void handleRef.current.done.then(async ({ ok, code }) => {
      append(ok ? `✓ finished (exit ${code})` : `✗ exited with code ${code ?? "—"}`);
      const v = await verify().catch(() => ({ ok: false, version: null }) as BinCheck);
      setResult(v);
      setPhase(ok || v.ok ? "ok" : "fail");
    });
  };
  const cancel = () => {
    handleRef.current?.cancel();
    append("✗ cancelled");
    setPhase("fail");
  };
  const reset = () => setPhase("idle");
  return { phase, lines, elapsed, result, arm, run, cancel, reset };
}

// last N activity lines, tail style with status coloring
function ActivityLog({ lines, elapsed, running }: { lines: string[]; elapsed: number; running: boolean }) {
  const tail = lines.slice(-9);
  return (
    <Box flexDirection="column" marginTop={1} paddingLeft={1} borderStyle="single" borderColor={C.border} borderTop={false} borderBottom={false} borderRight={false}>
      <Text>
        {running ? <Spinner color={C.accent} /> : <Text color={C.dim}>·</Text>}
        <Text color={C.dim}>{running ? ` live · ${elapsed}s` : " live"}</Text>
      </Text>
      {tail.length === 0 ? (
        <Text color={C.dim}>waiting…</Text>
      ) : (
        tail.map((l, i) => {
          const c = /^✗|fail|error/i.test(l) ? C.red : /^✓|done|finished|success/i.test(l) ? C.green : C.muted;
          return (
            <Text key={i} color={c} wrap="truncate-end">
              {l || " "}
            </Text>
          );
        })
      )}
    </Box>
  );
}

export function Onboarding({
  surogateBin,
  repoRoot,
  onDone,
  onExit,
}: {
  surogateBin: string;
  repoRoot: string;
  onDone: (r: { compute: Compute; sshHost?: string }) => void;
  onExit: () => void;
}) {
  const gpus = useMemo(() => discoverGpus(), []);
  const hosts = useMemo(() => recentSshHosts(), []);

  const [step, setStep] = useState<Step>("compute");
  const [compute, setCompute] = useState<Compute>("local");
  const [computeSel, setComputeSel] = useState(0);
  const [dstackOk, setDstackOk] = useState<boolean | null>(null);
  useEffect(() => {
    void dstackAvailable().then(setDstackOk);
  }, []);

  // SSH host entry
  const [sshHost, setSshHost] = useState(hosts[0] ?? "");
  const [sshDetect, setSshDetect] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [sshErr, setSshErr] = useState("");
  const [remoteGpus, setRemoteGpus] = useState<Gpu[]>([]);
  const sshTarget: SshTarget = useMemo(() => parseSshTarget(sshHost), [sshHost]);

  // install detection + plan
  const [check, setCheck] = useState<BinCheck | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [planIdx, setPlanIdx] = useState(0);

  const plans: InstallPlan[] = useMemo(() => {
    if (compute === "ssh") return [remoteInstallPlan()];
    if (compute === "dstack") return [dstackInstallPlan()];
    return localInstallPlans(repoRoot);
  }, [compute, repoRoot]);
  const plan = plans[Math.min(planIdx, plans.length - 1)]!;

  const verify = useMemo(() => {
    if (compute === "ssh") return () => surogateAvailableRemote(sshTarget, surogateBin);
    if (compute === "dstack") return async (): Promise<BinCheck> => ({ ok: await dstackAvailable(), version: null });
    return () => surogateAvailable(surogateBin);
  }, [compute, sshTarget, surogateBin]);

  const inst = useInstaller(verify);

  // when we enter the install step, probe the target once
  useEffect(() => {
    if (step !== "install") return;
    setCheck(null);
    setDetecting(true);
    let cancelled = false;
    void verify().then((c) => {
      if (!cancelled) {
        setCheck(c);
        setDetecting(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [step, verify]);

  const ready = (check?.ok ?? false) || inst.phase === "ok";

  // ── SSH detect ──────────────────────────────────────────────────────────────
  const runSshDetect = () => {
    if (!sshHost.trim()) return;
    setSshDetect("busy");
    setSshErr("");
    detectRemoteGpus(sshTarget)
      .then((g) => {
        setRemoteGpus(g);
        setSshDetect("ok");
        rememberSshHost(sshHost);
        setStep("install");
      })
      .catch((e: Error) => {
        setSshDetect("err");
        setSshErr(e.message.split("\n")[0] || "ssh failed");
      });
  };

  // ── install action ────────────────────────────────────────────────────────────
  const startInstall = () => {
    if (compute === "ssh") inst.run((onLine) => runShellRemote(sshTarget, plan.command, onLine));
    else inst.run((onLine) => runShell(plan.command, onLine, plan.cwd));
  };

  // ── keys ──────────────────────────────────────────────────────────────────────
  useInput((input, key) => {
    if (key.escape) {
      // Don't navigate away mid-install — the run keeps going in the background;
      // 'x' cancels it explicitly.
      if (step === "install" && inst.phase === "running") return;
      if (step === "compute") return onExit();
      if (step === "sshhost") return setStep("compute");
      if (step === "install") return setStep(compute === "ssh" ? "sshhost" : "compute");
      if (step === "ready") return setStep("install");
    }

    if (step === "compute") {
      const opts: Compute[] = ["local", "ssh", "dstack"];
      if (key.upArrow) setComputeSel((i) => (i - 1 + opts.length) % opts.length);
      else if (key.downArrow) setComputeSel((i) => (i + 1) % opts.length);
      else if (key.return) {
        const c = opts[computeSel]!;
        setCompute(c);
        setPlanIdx(0);
        setStep(c === "ssh" ? "sshhost" : "install");
      }
      return;
    }

    if (step === "sshhost") {
      if (sshDetect === "busy") return;
      if (key.return) runSshDetect();
      else if (key.backspace || key.delete) setSshHost((q) => q.slice(0, -1));
      else if (key.tab && hosts.length) setSshHost(hosts[0]!);
      else if (input && !key.ctrl && !key.meta && input >= " ") setSshHost((q) => q + input);
      return;
    }

    if (step === "install") {
      if (inst.phase === "running") {
        if (input === "x") inst.cancel();
        return;
      }
      if (inst.phase === "confirm") {
        if (input === "y" || key.return) startInstall();
        else if (input === "n" || key.escape) inst.reset();
        return;
      }
      if (ready) {
        if (key.return) setStep("ready");
        return;
      }
      // not installed yet
      if (input === "i") inst.arm();
      else if (input === "c") copyToClipboard(plan.command);
      else if (input === "r" && inst.phase === "fail") inst.arm();
      else if (input === "m" && plans.length > 1) setPlanIdx((i) => (i + 1) % plans.length);
      else if (key.return) {
        // re-check (e.g. user installed manually in another shell)
        setDetecting(true);
        void verify().then((c) => {
          setCheck(c);
          setDetecting(false);
        });
      }
      return;
    }

    if (step === "ready") {
      if (key.return) onDone({ compute, sshHost: compute === "ssh" ? sshHost : undefined });
      return;
    }
  });

  const stepRunning = step === "install" && inst.phase === "running";
  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1}>
      <Panel title="set up surogate · jackalope" flexGrow={1}>
        <Box marginTop={1} flexGrow={1}>
          {/* left: a cool animation + the wizard checklist filling in one-by-one */}
          <Box flexDirection="column" width={22} marginRight={2}>
            <SetupAnimation height={7} active={stepRunning} />
            <Box marginTop={1}>
              <StepChecklist current={stepIndex(step)} busy={stepRunning} />
            </Box>
          </Box>

          {/* right: the current setup step, happening live */}
          <Box flexDirection="column" flexGrow={1}>
            {step === "compute" && <ComputeStep gpus={gpus} hosts={hosts} dstackOk={dstackOk} sel={computeSel} />}
            {step === "sshhost" && <SshStep host={sshHost} detect={sshDetect} err={sshErr} hosts={hosts} gpus={remoteGpus} />}
            {step === "install" && (
              <InstallStep compute={compute} host={sshHost} plan={plan} plans={plans} detecting={detecting} check={check} ready={ready} inst={inst} />
            )}
            {step === "ready" && <ReadyStep compute={compute} host={sshHost} check={check} />}
            <Box marginTop={1}>
              <Text color={C.dim}>esc back · ctrl-c quit</Text>
            </Box>
          </Box>
        </Box>
      </Panel>
    </Box>
  );
}

// An animated brand-gradient equalizer ("compute warming up") — purely decorative,
// distinct from the splash showcase. Pulses faster while an install is running.
function SetupAnimation({ height = 7, active = false }: { height?: number; active?: boolean }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((x) => x + 1), active ? 90 : 160);
    return () => clearInterval(id);
  }, [active]);
  const cols = 18;
  const speed = active ? 0.45 : 0.28;
  const bars = Array.from({ length: cols }, (_, i) => Math.max(1, Math.round((0.5 + 0.5 * Math.sin(t * speed + i * 0.55)) * height)));
  const rows: string[] = [];
  for (let r = height; r >= 1; r--) rows.push(bars.map((h) => (h >= r ? "█" : " ")).join(""));
  return (
    <Box flexDirection="column">
      {rows.map((line, i) => (
        <Text key={i} color={goldAt(i, height)}>
          {line}
        </Text>
      ))}
    </Box>
  );
}

// Vertical wizard checklist: done steps ✓, the current one spinning/highlighted,
// upcoming dim — PostHog-wizard cadence.
function StepChecklist({ current, busy }: { current: number; busy: boolean }) {
  return (
    <Box flexDirection="column">
      {STEP_NAMES.map((name, i) => {
        const done = i < current;
        const cur = i === current;
        return (
          <Text key={name}>
            {done ? (
              <Text color={C.green}>✓ </Text>
            ) : cur ? (
              busy ? (
                <Spinner color={C.accent} />
              ) : (
                <Text color={C.accent}>▸ </Text>
              )
            ) : (
              <Text color={C.dim}>○ </Text>
            )}
            <Text color={done ? C.muted : cur ? C.accent : C.dim} bold={cur}>
              {cur && busy ? " " : ""}
              {name}
            </Text>
          </Text>
        );
      })}
    </Box>
  );
}

// ── step 1: compute ────────────────────────────────────────────────────────────
function ComputeStep({ gpus, hosts, dstackOk, sel }: { gpus: Gpu[]; hosts: string[]; dstackOk: boolean | null; sel: number }) {
  const rows = [
    { name: "Local GPUs", glyph: "▣", hint: gpuSummary(gpus) },
    { name: "SSH server", glyph: "⇄", hint: hosts.length ? `recent: ${hosts.slice(0, 2).join(", ")}` : "connect to a remote GPU box over SSH" },
    {
      name: "Cloud (dstack)",
      glyph: "☁",
      hint: dstackOk === null ? "checking dstack…" : dstackOk ? "dstack ready · rent a GPU (RunPod/Lambda/AWS…)" : "rent a GPU — dstack CLI not installed yet",
    },
  ];
  return (
    <Box flexDirection="column">
      <Text color={C.muted}>where should training run?</Text>
      <Box flexDirection="column" marginTop={1}>
        {rows.map((r, i) => {
          const on = i === sel;
          return (
            <Box key={r.name} marginBottom={i < rows.length - 1 ? 1 : 0}>
              <Text color={on ? C.accent : C.dim}>{on ? "▸ " : "  "}</Text>
              <Box flexDirection="column">
                <Text color={on ? C.accent : C.text} bold={on}>
                  {r.glyph} {r.name}
                </Text>
                <Text color={C.muted}>{r.hint}</Text>
              </Box>
            </Box>
          );
        })}
      </Box>
      <Box marginTop={1}>
        <Text color={C.dim}>↑↓ choose · </Text>
        <Text color={C.gold} bold>
          ⏎
        </Text>
        <Text color={C.dim}> continue</Text>
      </Box>
    </Box>
  );
}

// ── step 1b: ssh host ──────────────────────────────────────────────────────────
function SshStep({
  host,
  detect,
  err,
  hosts,
  gpus,
}: {
  host: string;
  detect: "idle" | "busy" | "ok" | "err";
  err: string;
  hosts: string[];
  gpus: Gpu[];
}) {
  return (
    <Box flexDirection="column">
      <Text color={C.muted}>connect to a GPU server — uses your ~/.ssh/config (aliases, keys, ports)</Text>
      <Box marginTop={1}>
        <Text color={C.dim}>host </Text>
        <Text color={C.gold}>{host}</Text>
        <Text color={C.gold}>█</Text>
      </Box>
      {hosts.length > 0 && (
        <Text color={C.dim}>
          recent: <Text color={C.muted}>{hosts.slice(0, 4).join("  ")}</Text> · tab fills the latest
        </Text>
      )}
      <Box marginTop={1}>
        {detect === "busy" ? (
          <Text>
            <Spinner color={C.accent} />
            <Text color={C.muted}> connecting & probing GPUs…</Text>
          </Text>
        ) : detect === "ok" ? (
          <Text color={C.green}>✓ {gpus.length} GPU{gpus.length === 1 ? "" : "s"}: {gpuSummary(gpus)}</Text>
        ) : detect === "err" ? (
          <Text color={C.red}>✗ {err}</Text>
        ) : (
          <Text color={C.dim}>type user@host (or an alias), then ⏎ to connect</Text>
        )}
      </Box>
      <Box marginTop={1}>
        <Text color={C.gold} bold>
          ⏎
        </Text>
        <Text color={C.dim}> connect · esc back</Text>
      </Box>
    </Box>
  );
}

// ── step 2: get surogate ready ──────────────────────────────────────────────────
function InstallStep({
  compute,
  host,
  plan,
  plans,
  detecting,
  check,
  ready,
  inst,
}: {
  compute: Compute;
  host: string;
  plan: InstallPlan;
  plans: InstallPlan[];
  detecting: boolean;
  check: BinCheck | null;
  ready: boolean;
  inst: ReturnType<typeof useInstaller>;
}) {
  const where = compute === "ssh" ? host || "remote" : compute === "dstack" ? "cloud" : "local";
  const thing = compute === "dstack" ? "dstack" : "surogate";
  return (
    <Box flexDirection="column">
      <Text>
        <Text color={C.dim}>target </Text>
        <Text color={C.accent}>{where}</Text>
      </Text>

      {detecting ? (
        <Box marginTop={1}>
          <Spinner color={C.accent} />
          <Text color={C.muted}> checking for {thing}…</Text>
        </Box>
      ) : ready ? (
        <>
          <Box marginTop={1}>
            <Text color={C.green}>✓ {thing} is ready{(inst.result?.version ?? check?.version) ? ` · ${inst.result?.version ?? check?.version}` : ""}</Text>
          </Box>
          <Box marginTop={1}>
            <Text color={C.gold} bold>
              ⏎
            </Text>
            <Text color={C.dim}> continue</Text>
          </Box>
        </>
      ) : (
        <>
          <Box marginTop={1}>
            <Text color={C.warm}>● {thing} not found{check?.reason ? ` (${check.reason})` : ""} — let's install it</Text>
          </Box>
          <Box marginTop={1} flexDirection="column">
            <Text>
              <Text color={C.dim}>recommended </Text>
              <Text color={C.text} bold>
                {plan.title}
              </Text>
              {plans.length > 1 && <Text color={C.dim}>{"   "}m other mode</Text>}
            </Text>
            <Text color={C.muted}>{plan.note}</Text>
            <Box marginTop={1}>
              <Text color={C.dim}>$ </Text>
              <Text color={C.gold} wrap="truncate-end">
                {plan.command}
              </Text>
            </Box>
            {plan.prereqs.length > 0 && (
              <Text color={C.dim} wrap="truncate-end">
                you need: {plan.prereqs.join(" · ")}
              </Text>
            )}
            {plan.postHint && <Text color={C.dim}>after: {plan.postHint}</Text>}
          </Box>

          {inst.phase === "confirm" ? (
            <Box marginTop={1}>
              <Text color={C.warm}>run this now? </Text>
              <Text color={C.green} bold>
                y
              </Text>
              <Text color={C.dim}> yes · </Text>
              <Text color={C.red} bold>
                n
              </Text>
              <Text color={C.dim}> no</Text>
            </Box>
          ) : inst.phase === "idle" ? (
            <Box marginTop={1}>
              <Text color={C.gold} bold>
                i
              </Text>
              <Text color={C.dim}> install for me · </Text>
              <Text color={C.gold} bold>
                c
              </Text>
              <Text color={C.dim}> copy command · </Text>
              <Text color={C.gold} bold>
                ⏎
              </Text>
              <Text color={C.dim}> re-check</Text>
            </Box>
          ) : null}

          {(inst.phase === "running" || inst.phase === "fail") && (
            <>
              <ActivityLog lines={inst.lines} elapsed={inst.elapsed} running={inst.phase === "running"} />
              <Box marginTop={1}>
                {inst.phase === "running" ? (
                  <Text color={C.dim}>installing… x to cancel</Text>
                ) : (
                  <Text>
                    <Text color={C.red}>✗ install failed · </Text>
                    <Text color={C.gold} bold>
                      r
                    </Text>
                    <Text color={C.dim}> retry · </Text>
                    <Text color={C.gold} bold>
                      c
                    </Text>
                    <Text color={C.dim}> copy to run manually</Text>
                  </Text>
                )}
              </Box>
            </>
          )}
        </>
      )}
    </Box>
  );
}

// ── step 3: ready ─────────────────────────────────────────────────────────────
function ReadyStep({ compute, host, check }: { compute: Compute; host: string; check: BinCheck | null }) {
  const where = compute === "ssh" ? `SSH · ${host}` : compute === "dstack" ? "Cloud · dstack" : "Local GPUs";
  return (
    <Box flexDirection="column">
      <Text color={C.green} bold>
        ✓ you're all set
      </Text>
      <Box flexDirection="column" marginTop={1}>
        <Text>
          <Text color={C.dim}>compute   </Text>
          <Text color={C.text}>{where}</Text>
        </Text>
        <Text>
          <Text color={C.dim}>surogate  </Text>
          <Text color={C.green}>ready{check?.version ? ` · ${check.version}` : ""}</Text>
        </Text>
      </Box>
      <Box marginTop={1}>
        <Text color={C.muted}>jackalope will open Launch with this compute selected — pick a model, dataset & recipe and go.</Text>
      </Box>
      <Box marginTop={1}>
        <Text color={C.gold} bold>
          ⏎
        </Text>
        <Text color={C.dim}> start training</Text>
      </Box>
    </Box>
  );
}
