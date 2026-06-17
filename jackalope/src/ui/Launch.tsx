import React, { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { ConfirmInput, Select } from "@inkjs/ui";
import fs from "node:fs";
import { looksLikePath, resolveLocalPath } from "../local.ts";
import { RUNS_DIR, newRunFeedPath, runArtifacts, writeRunMeta } from "../runs.ts";
import {
  buildCommand,
  buildConfigYaml,
  buildGrpoCommand,
  DEFAULT_FIELDS,
  discoverGpus,
  estimateRunVramGB,
  ensureRlConfigs,
  fitOnGpu,
  gpuFreeGB,
  grpoStackAvailable,
  grpoStackInstallCommand,
  type GrpoOverlay,
  listExampleConfigs,
  loadGpuSelection,
  overlayExistingSft,
  RECIPES,
  type Gpu,
  type LaunchFields,
  type SpawnResult,
  spawnGrpo,
  spawnTraining,
} from "../launch.ts";
import { FieldEditor } from "./FieldEditor.tsx";
import { DSTACK_DEFAULTS, type DstackFields, GRPO_DEFAULTS, type GrpoFields, dstackSchema, grpoSchema, sftSchema } from "./paramSchemas.ts";
import { type DstackConfig, dstackAvailable, launchDstackRun } from "../dstack.ts";
import { gpuSupport } from "../gpu-support.ts";
import { YamlPicker } from "./YamlPicker.tsx";
import { GpuSelect } from "./GpuSelect.tsx";
import { loadProviders } from "../providers.ts";
import { launchModalRun, MODAL_DEFAULT_IMAGE, MODAL_GPUS, modalReady } from "../modal.ts";
import { Stepper } from "./Stepper.tsx";
import { type SshTarget, detectRemoteGpus, launchRemoteRun, parseSshTarget, rememberSshHost } from "../ssh.ts";
import { runShell } from "../setup.ts";
import { StatusVerb } from "./StatusVerb.tsx";
import { Spinner } from "./Spinner.tsx";
import { C } from "./theme.ts";

type Mode = "sft" | "grpo" | "ruler";
type Source = "new" | "yaml";
type Compute = "local" | "ssh" | "dstack" | "modal";
// Cloud backends provision their own GPUs, so they skip the GPU-select step.
const isCloud = (c: Compute) => c === "dstack" || c === "modal";
type Phase =
  | "compute"
  | "sshhost"
  | "dstackcfg"
  | "modalcfg"
  | "mode"
  | "source"
  | "yaml"
  | "gpus"
  | "params"
  | "tgpus"
  | "vgpus"
  | "jgpus"
  | "gparams"
  | "confirm"
  | "started";

const ORDER: Phase[] = ["compute", "sshhost", "dstackcfg", "modalcfg", "mode", "source", "yaml", "gpus", "params", "tgpus", "vgpus", "jgpus", "gparams", "confirm", "started"];

// The user-facing step rail for the active compute/mode/source. Phases map to
// short labels; the current step is derived from the live phase.
function stepRail(compute: Compute, mode: Mode, source: Source): { phase: Phase; label: string }[] {
  const head: { phase: Phase; label: string }[] = [{ phase: "compute", label: "Compute" }];
  if (compute === "ssh") head.push({ phase: "sshhost", label: "SSH host" });
  if (compute === "dstack") head.push({ phase: "dstackcfg", label: "Cloud" });
  if (compute === "modal") head.push({ phase: "modalcfg", label: "Modal" });
  if (mode === "sft") {
    if (isCloud(compute)) {
      // cloud provisions the GPUs → no GPU-select step
      return [
        ...head,
        { phase: "mode", label: "Mode" },
        { phase: "params", label: "Params" },
        { phase: "confirm", label: "Run" },
      ];
    }
    if (compute === "ssh") {
      // remote = new config only (no source/yaml step)
      return [
        ...head,
        { phase: "mode", label: "Mode" },
        { phase: "gpus", label: "GPUs" },
        { phase: "params", label: "Params" },
        { phase: "confirm", label: "Run" },
      ];
    }
    const tail: { phase: Phase; label: string }[] =
      source === "yaml"
        ? [
            { phase: "source", label: "Config" },
            { phase: "yaml", label: "YAML" },
            { phase: "gpus", label: "GPUs" },
            { phase: "confirm", label: "Run" },
          ]
        : [
            { phase: "source", label: "Config" },
            { phase: "gpus", label: "GPUs" },
            { phase: "params", label: "Params" },
            { phase: "confirm", label: "Run" },
          ];
    return [...head, { phase: "mode", label: "Mode" }, ...tail];
  }
  const rail: { phase: Phase; label: string }[] = [
    ...head,
    { phase: "mode", label: "Mode" },
    { phase: "tgpus", label: "Trainer" },
    { phase: "vgpus", label: "vLLM" },
  ];
  if (mode === "ruler") rail.push({ phase: "jgpus", label: "Judge" });
  rail.push({ phase: "gparams", label: "Params" }, { phase: "confirm", label: "Run" });
  return rail;
}

// One-line "what to do now" guidance per phase.
const PHASE_GUIDANCE: Record<Phase, string> = {
  compute: "where should this run? your local GPUs, or a remote GPU server over SSH",
  sshhost: "enter user@host (or a ~/.ssh/config alias) — we'll detect its GPUs",
  dstackcfg: "pick the cloud GPU + provider — dstack provisions it on launch",
  modalcfg: "pick the Modal GPU — a sandbox runs surogate; outputs land on a Volume",
  mode: "choose how to train — SFT, GRPO, or RULER (RL with an LLM judge)",
  source: "build a fresh config, or run one of your existing YAML files",
  yaml: "pick the YAML to run — it's launched as-is, plus live monitoring",
  gpus: "tick the GPUs to train on (⏎ uses the highlighted one if none ticked)",
  params: "tune the run, or just launch with the sensible defaults",
  tgpus: "GPUs for the trainer",
  vgpus: "GPUs for inference (vLLM) — must be disjoint from the trainer",
  jgpus: "GPUs for the LLM judge (vLLM) — disjoint from trainer + inference",
  gparams: "tune the overlay params, or launch with the example defaults",
  confirm: "review the command and start the run",
  started: "launching…",
};

// A minimal inline text input (the @inkjs/ui one was flaky inside this form).
function TextPrompt({
  value,
  onChange,
  onSubmit,
  active,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  active: boolean;
  placeholder: string;
}) {
  useInput(
    (input, key) => {
      if (key.return) onSubmit();
      else if (key.backspace || key.delete) onChange(value.slice(0, -1));
      else if (input && !key.ctrl && !key.meta && input >= " ") onChange(value + input);
    },
    { isActive: active },
  );
  return (
    <Text>
      {value ? <Text color={C.text}>{value}</Text> : <Text color={C.dim}>{placeholder}</Text>}
      <Text color={C.gold}>{active ? "█" : ""}</Text>
    </Text>
  );
}

/** A local path (vs HF id) gets a dim tag so it's clear where it's loaded from. */
function srcTag(s: string): string {
  return looksLikePath(s) ? " (local)" : "";
}

function gpuLabel(g: Gpu, estGB: number | null): string {
  const free = gpuFreeGB(g);
  const freeStr = free === null ? "" : ` · ${free.toFixed(0)}GB free`;
  const verdict = fitOnGpu(estGB, free);
  const tag = g.busy ? " · busy" : "";
  const fit = verdict === "risk" ? " · ⚠ tight" : verdict === "fits" ? " · ✓ fits" : "";
  const sup = gpuSupport(g.sm);
  const supStr = g.sm !== null ? ` · ${sup.arch} (${sup.best})` : "";
  return `${g.name} · id ${g.id}${freeStr}${supStr}${tag}${fit}`;
}

const EDIT_HINT = "— ↑↓ move · ⏎ edit · space toggle · ←→ change";
const GPU_HINT = "— ↑↓ move · space to tick · ⏎ confirm (uses the highlighted one if none ticked)";

export function Launch({
  feedPath,
  surogateBin,
  repoRoot,
  active,
  picked,
  onLaunched,
  initialCompute,
  initialSshHost,
}: {
  feedPath: string;
  surogateBin: string;
  repoRoot: string;
  active: boolean;
  picked?: { model?: { id: string; t: { supported: boolean; recipes: readonly string[] } }; dataset?: string };
  onLaunched: (metricsPath: string) => void;
  initialCompute?: Compute;
  initialSshHost?: string;
}) {
  const gpus = useMemo(() => discoverGpus(), []);
  const examples = useMemo(() => listExampleConfigs(repoRoot), [repoRoot]);
  const [mode, setMode] = useState<Mode>("sft");
  const [source, setSource] = useState<Source>("new");
  const [compute, setCompute] = useState<Compute>(initialCompute ?? "local");
  const [sshHost, setSshHost] = useState(initialSshHost ?? "");
  const [remoteGpus, setRemoteGpus] = useState<Gpu[]>([]);
  const [detect, setDetect] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [detectErr, setDetectErr] = useState("");
  const [dstackFields, setDstackFields] = useState<DstackFields>({ ...DSTACK_DEFAULTS });
  // Saved compute providers (GPUs tab → p) offered as quick-picks in Compute.
  const savedProviders = useMemo(() => loadProviders().filter((p) => p.kind === "ssh" || p.kind === "dstack"), []);
  const [dstackOk, setDstackOk] = useState<boolean | null>(null); // CLI installed?
  const [modalOk, setModalOk] = useState<boolean | null>(null);
  const [modalGpu, setModalGpu] = useState("H100");
  const [modalCount, setModalCount] = useState(1);
  useEffect(() => {
    void dstackAvailable().then(setDstackOk);
    void modalReady().then(setModalOk);
  }, []);
  const sshTarget: SshTarget = useMemo(() => parseSshTarget(sshHost), [sshHost]);
  const activeGpus = compute === "ssh" ? remoteGpus : gpus;

  // probe the remote host's GPUs (non-blocking) when the user submits a host
  const runDetect = () => {
    if (!sshHost.trim()) return;
    setDetect("busy");
    setDetectErr("");
    detectRemoteGpus(sshTarget)
      .then((g) => {
        setRemoteGpus(g);
        setDetect("ok");
        rememberSshHost(sshHost); // surface it on the GPUs tab next time
        if (g.length > 0) setPhase("mode");
        else setDetectErr("connected, but no NVIDIA GPUs found there");
      })
      .catch((e: Error) => {
        setDetect("err");
        setDetectErr(e.message.split("\n")[0] || "ssh failed");
      });
  };
  const [yamlPath, setYamlPath] = useState("");
  // When the setup wizard hands off a compute target, skip the compute picker and
  // jump straight to that backend's next step — except local-with-no-GPUs, which
  // still needs the picker (so its "no local GPUs, pick Remote/Cloud" guard shows).
  const [phase, setPhase] = useState<Phase>(
    initialCompute && !(initialCompute === "local" && gpus.length === 0)
      ? initialCompute === "ssh"
        ? "sshhost"
        : initialCompute === "dstack"
          ? "dstackcfg"
          : "mode"
      : "compute",
  );
  // ← steps back one visible step in the wizard. Skipped in the FieldEditor
  // phases (params/dstackcfg/gparams), where ← cycles a field value instead — use
  // esc to leave those. compute is the first step (esc exits to nav).
  const FIELD_PHASES: Phase[] = ["dstackcfg", "params", "gparams"];
  useInput(
    (input, key) => {
      // 'i' installs the vLLM stack when an RL mode needs it — but not during a
      // field-editing phase, where 'i' is ordinary typed text.
      if (input === "i" && mode !== "sft" && stackOk === false && !installing && !FIELD_PHASES.includes(phase)) {
        installStack();
        return;
      }
      if (!key.leftArrow || FIELD_PHASES.includes(phase) || phase === "started") return;
      const rail = stepRail(compute, mode, source);
      const cur = rail.findIndex((s) => s.phase === phase);
      if (cur > 0) setPhase(rail[cur - 1]!.phase);
    },
    { isActive: active },
  );

  // Seed from the GPUs tab's earmarked selection (local GPUs present here).
  const [selected, setSelected] = useState<number[]>(() => loadGpuSelection().filter((id) => gpus.some((g) => g.id === id)));
  const [trainerGpus, setTrainerGpus] = useState<number[]>([]);
  const [vllmGpus, setVllmGpus] = useState<number[]>([]);
  const [judgeGpus, setJudgeGpus] = useState<number[]>([]);
  // Every SFT knob lives here, editable in Parameters → buildConfigYaml.
  const [fields, setFields] = useState<LaunchFields>(() => ({
    ...DEFAULT_FIELDS,
    model: picked?.model?.id ?? DEFAULT_FIELDS.model,
    datasetPath: picked?.dataset ?? DEFAULT_FIELDS.datasetPath,
  }));
  const [grpoFields, setGrpoFields] = useState<GrpoFields>({ ...GRPO_DEFAULTS });
  const [pid, setPid] = useState<number | null>(null);
  const [launchErr, setLaunchErr] = useState<string | null>(null);
  // GRPO/RULER preflight: is the vLLM stack present in surogate's venv? Probed
  // lazily (only when an RL mode is first chosen) so SFT-only users never pay for
  // it; re-checked after an in-UI install.
  const [stackOk, setStackOk] = useState<boolean | null>(null);
  const [installing, setInstalling] = useState(false);
  const [installLog, setInstallLog] = useState<string[]>([]);
  const mounted = useRef(true);
  const installBusy = useRef(false); // synchronous guard against double-trigger
  useEffect(() => () => void (mounted.current = false), []);
  const checkStack = () => {
    void grpoStackAvailable(surogateBin).then((ok) => mounted.current && setStackOk(ok));
  };
  const installStack = () => {
    if (installBusy.current) return;
    installBusy.current = true;
    setInstalling(true);
    setInstallLog([]);
    runShell(grpoStackInstallCommand(surogateBin), (line) => setInstallLog((l) => [...l.slice(-12), line]), repoRoot)
      .done.then(() => grpoStackAvailable(surogateBin))
      .then((ok) => {
        installBusy.current = false;
        if (!mounted.current) return; // navigated away mid-install — skip stale setState
        setStackOk(ok);
        setInstalling(false);
        setInstallLog((l) => [...l.slice(-12), ok ? "✓ vLLM stack ready" : "✗ install failed — see output above"]);
      });
  };

  // keep model/dataset in sync when one is picked in the Models/Datasets tabs
  const pickedModel = picked?.model?.id;
  const pickedDataset = picked?.dataset;
  useEffect(() => {
    if (pickedModel) setFields((f) => ({ ...f, model: pickedModel }));
  }, [pickedModel]);
  useEffect(() => {
    if (pickedDataset) setFields((f) => ({ ...f, datasetPath: pickedDataset }));
  }, [pickedDataset]);

  const outAbs = resolveLocalPath(fields.outputDir);
  const recipeChoices: readonly string[] = picked?.model?.t.supported ? picked.model.t.recipes : RECIPES;

  const estGB = estimateRunVramGB(fields);
  // Memoized: ensureRlConfigs does several fs writes and the schemas allocate
  // sizeable arrays — neither should run every render (the status verb re-renders
  // the tree frequently while a run is starting). Configs are always (re)generated,
  // so GRPO/RULER are always available locally.
  const { grpo, ruler } = useMemo(
    () => ({ grpo: ensureRlConfigs("grpo", repoRoot), ruler: ensureRlConfigs("ruler", repoRoot) }),
    [repoRoot],
  );
  const rlConfigs = mode === "ruler" ? ruler : grpo;
  const sftDefs = useMemo(() => sftSchema(recipeChoices), [recipeChoices]);
  const grpoDefs = useMemo(() => grpoSchema(recipeChoices), [recipeChoices]);
  const dstackDefs = useMemo(() => dstackSchema(), []);

  const grpoOverlay = (): GrpoOverlay => ({
    train: {
      learning_rate: grpoFields.learningRate,
      max_steps: grpoFields.maxSteps,
      save_steps: grpoFields.saveSteps,
      per_device_train_batch_size: grpoFields.perDeviceBatch,
      sequence_len: grpoFields.sequenceLen,
      recipe: grpoFields.recipe,
    },
    orch: {
      max_steps: grpoFields.maxSteps,
      seq_len: grpoFields.sequenceLen,
      batch_size: grpoFields.batchSize,
      rollouts_per_example: grpoFields.rolloutsPerExample,
    },
  });

  const doLaunchSft = () => {
    if (!fields.model.trim()) {
      setLaunchErr("pick a model first — Models tab, or type a model id / path in Parameters");
      return;
    }
    if (source === "new" && !fields.datasetPath.trim()) {
      setLaunchErr("pick a dataset first — Datasets tab, or type a path in Parameters");
      return;
    }
    const label = source === "yaml" ? "sft-yaml" : `sft-${fields.recipe}`;

    // Remote: build the config WITHOUT a metrics path (so the remote env var
    // SUROGATE_METRICS_PATH wins) and with a relative output_dir (checkpoints
    // land under the remote run folder). The feed is mirrored back over SSH.
    if (compute === "ssh") {
      const yaml = buildConfigYaml({ ...fields, outputDir: "output" }, selected.length);
      setPhase("started");
      void launchRemoteRun(sshTarget, yaml, label, surogateBin).then((res) => {
        if (res.ok) {
          setLaunchErr(null);
          setTimeout(() => onLaunched(res.feed), 1200);
        } else {
          setLaunchErr(res.reason);
          setPhase("confirm");
        }
      });
      return;
    }

    // Cloud: dstack provisions `count` GPUs and streams metrics back over stdout.
    if (compute === "dstack") {
      const count = Math.max(1, Number(dstackFields.count) || 1);
      const yaml = buildConfigYaml({ ...fields, outputDir: "output" }, count);
      const cfg: DstackConfig = {
        gpu: dstackFields.gpu,
        count,
        image: dstackFields.image,
        backend: dstackFields.backend === "cheapest" ? undefined : dstackFields.backend,
        region: dstackFields.region || undefined,
      };
      const res = launchDstackRun(cfg, yaml, label, surogateBin);
      if (res.ok) {
        setLaunchErr(null);
        setPhase("started");
        setTimeout(() => onLaunched(res.feed), 1200);
      } else {
        setLaunchErr(res.reason);
      }
      return;
    }

    // Modal: a serverless GPU sandbox runs surogate; metrics stream over the
    // driver's stdout and artifacts persist on a Modal Volume.
    if (compute === "modal") {
      const count = Math.max(1, modalCount);
      const yaml = buildConfigYaml({ ...fields, outputDir: "output" }, count);
      const res = launchModalRun({ gpu: modalGpu, count, image: MODAL_DEFAULT_IMAGE }, yaml, label);
      if (res.ok) {
        setLaunchErr(null);
        setPhase("started");
        setTimeout(() => onLaunched(res.feed), 1200);
      } else {
        setLaunchErr(res.reason);
      }
      return;
    }

    const metricsPath = newRunFeedPath(label, Date.now());
    const art = runArtifacts(metricsPath);
    // a freshly-built config writes its checkpoints into the run folder; an
    // existing YAML keeps whatever output_dir the user already set.
    const outDir = source === "new" ? art.outputDir : outAbs;
    const yaml =
      source === "yaml"
        ? overlayExistingSft(yamlPath, selected.length, metricsPath)
        : buildConfigYaml({ ...fields, outputDir: outDir }, selected.length, metricsPath);
    fs.writeFileSync(art.configPath, yaml);
    writeRunMeta(metricsPath, {
      mode: "sft",
      model: fields.model,
      dataset: fields.datasetPath,
      recipe: source === "yaml" ? undefined : fields.recipe,
      gpus: selected,
      maxSteps: Number(fields.maxSteps) || undefined,
      startedAt: Date.now(),
      label,
    });
    finishLaunch(spawnTraining(art.configPath, selected, metricsPath, surogateBin), metricsPath);
  };

  const doLaunchGrpo = () => {
    if (stackOk === false) {
      setLaunchErr("the vLLM stack is missing — press i to install it before launching");
      return;
    }
    const metricsPath = newRunFeedPath(mode, Date.now());
    writeRunMeta(metricsPath, {
      mode,
      recipe: grpoFields.recipe,
      gpus: [...trainerGpus, ...vllmGpus, ...judgeGpus],
      maxSteps: Number(grpoFields.maxSteps) || undefined,
      startedAt: Date.now(),
      label: mode,
    });
    finishLaunch(spawnGrpo(rlConfigs, trainerGpus, vllmGpus, metricsPath, surogateBin, judgeGpus, grpoOverlay()), metricsPath);
  };

  const finishLaunch = (r: SpawnResult, metricsPath: string) => {
    if (r.ok) {
      setPid(r.pid);
      setLaunchErr(null);
      setPhase("started");
      setTimeout(() => onLaunched(metricsPath), 1200);
    } else {
      setLaunchErr(r.reason); // stay on confirm to retry
    }
  };

  const command =
    compute === "modal"
      ? `modal sandbox · ${modalGpu}:${modalCount} · surogate ${mode} config.yaml   (outputs → Volume)`
      : compute === "dstack"
      ? `dstack apply -y -f task.dstack.yml   (provisions ${dstackFields.gpu}:${dstackFields.count} on ${dstackFields.backend})`
      : compute === "ssh"
        ? `ssh ${sshHost} -- ${buildCommand(selected, "config.yaml", surogateBin)}   (detached in tmux)`
        : mode === "sft"
          ? buildCommand(selected, "<run-folder>/config.yaml", surogateBin)
          : buildGrpoCommand(trainerGpus, vllmGpus, rlConfigs, surogateBin, judgeGpus);

  // ORDER-driven step state: `pastOrAt` decides if a step block is visible yet,
  // `done` (strictly past) gets a green ✓ wizard-style. Both replace verbose
  // `phase === "x" || ...` chains so adding a phase only touches ORDER.
  const at = (p: Phase) => ORDER.indexOf(p);
  const pastOrAt = (p: Phase) => at(p) <= at(phase);
  const done = (p: Phase) => at(p) < at(phase);
  const mark = (p: Phase) => (done(p) ? <Text color={C.green}>✓ </Text> : null);

  // the horizontal progress rail for the current compute/mode/source
  const rail = stepRail(compute, mode, source);
  const railCur = phase === "started" ? rail.length : Math.max(0, rail.findIndex((s) => s.phase === phase));
  // SFT requires a model + dataset (no preselection — the user must choose)
  const missingModel = mode === "sft" && !fields.model.trim();
  const missingDataset = mode === "sft" && source === "new" && !fields.datasetPath.trim();
  // pre-flight: warn if the chosen dataset is a local path that doesn't exist
  // (memoized so the existsSync doesn't hit disk on every keystroke/render)
  const datasetMissing = useMemo(
    () => looksLikePath(fields.datasetPath) && !fs.existsSync(resolveLocalPath(fields.datasetPath)),
    [fields.datasetPath],
  );
  // pre-flight: warn if the estimate exceeds the free memory of a chosen GPU
  const selectedGpus = gpus.filter((g) => selected.includes(g.id));
  const vramRisk =
    mode === "sft" &&
    estGB !== null &&
    selectedGpus.length > 0 &&
    selectedGpus.every((g) => fitOnGpu(estGB, gpuFreeGB(g)) === "risk");

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingTop={1}>
      <Text bold color={C.accent}>
        Launch a training run
      </Text>
      <Text color={C.muted}>
        <Text color={fields.model ? (picked?.model ? C.green : C.text) : C.warm}>
          {fields.model || "— no model — pick in Models"}
        </Text>
        <Text color={C.dim}>{srcTag(fields.model)}</Text>
        {" · "}
        <Text color={fields.datasetPath ? C.text : C.warm}>{fields.datasetPath || "— no dataset — pick in Datasets"}</Text>
        <Text color={C.dim}>{srcTag(fields.datasetPath)}</Text>
        {estGB !== null && (
          <Text>
            {" · ≈ est "}
            <Text color={C.gold}>{estGB} GB</Text>
          </Text>
        )}
      </Text>
      {(missingModel || missingDataset) && (
        <Text color={C.dim}>
          tip: open <Text color={C.accent}>Models</Text> / <Text color={C.accent}>Datasets</Text> to search HuggingFace,
          or type a path (/…, ~/…) in Parameters
        </Text>
      )}
      {mode === "sft" && datasetMissing && (
        <Text color={C.warm}>⚠ dataset path not found on disk — double-check it before launching</Text>
      )}
      {vramRisk && (
        <Text color={C.warm}>
          ⚠ ≈{estGB} GB estimate may not fit the selected GPU(s) — lower batch/seq, enable recompute, or add a GPU
        </Text>
      )}

      {active && phase !== "started" && (
        <Box marginTop={1} flexDirection="column">
          <Stepper steps={rail.map((s) => s.label)} current={railCur} />
          <Text color={C.dim}>{PHASE_GUIDANCE[phase]}</Text>
        </Box>
      )}

      {launchErr && <Text color={C.red}>✗ failed to start — {launchErr}</Text>}
      {!active && (
        <Box marginTop={1}>
          <Text color={C.gold}>press ⏎ to configure</Text>
          <Text color={C.dim}> · esc to go back to nav</Text>
        </Box>
      )}

      {/* 0 · compute */}
      <Box marginTop={1} flexDirection="column">
        <Text color={phase === "compute" ? C.accent : C.muted} bold>
          {mark("compute")}Compute
        </Text>
        {phase === "compute" ? (
          <Select
            isDisabled={!active}
            options={[
              { label: `Local — ${gpus.length} GPU${gpus.length === 1 ? "" : "s"} detected`, value: "local" },
              { label: "Remote — a GPU server over SSH", value: "ssh" },
              {
                label: dstackOk === false ? "Cloud — dstack (CLI not installed)" : "Cloud — dstack (AWS/Lambda/RunPod/Vast…)",
                value: "dstack",
              },
              {
                label: modalOk === false ? "Modal — serverless GPU (modal not installed)" : "Modal — serverless GPU sandbox",
                value: "modal",
              },
              // saved providers (added in the GPUs tab) — pick one directly
              ...savedProviders.map((p) => ({
                label: `★ ${p.label}${p.host ? ` · ${p.host}` : p.backend ? ` · ${p.backend}` : ""}`,
                value: `saved:${p.id}`,
              })),
            ]}
            onChange={(v) => {
              // a saved provider → preselect its compute + connection, skip re-entry
              if (v.startsWith("saved:")) {
                const p = savedProviders.find((x) => x.id === v.slice(6));
                if (!p) return;
                if (p.kind === "ssh") {
                  setCompute("ssh");
                  setSshHost(p.host ?? "");
                  setPhase("sshhost");
                } else {
                  setCompute("dstack");
                  if (p.backend) setDstackFields((f) => ({ ...f, backend: p.backend! }));
                  setPhase("dstackcfg");
                }
                return;
              }
              const c = v as Compute;
              setCompute(c);
              // can't train locally with no GPUs — stay on this step so the warning
              // shows and the user picks Remote/Cloud (avoids an empty GPU picker).
              if (c === "local" && gpus.length === 0) return;
              setPhase(c === "ssh" ? "sshhost" : c === "dstack" ? "dstackcfg" : c === "modal" ? "modalcfg" : "mode");
            }}
          />
        ) : (
          <Text color={C.text}>
            {compute === "ssh"
              ? `SSH · ${sshHost}`
              : compute === "dstack"
                ? `Cloud · dstack · ${dstackFields.gpu}:${dstackFields.count}`
                : compute === "modal"
                  ? `Modal · ${modalGpu}:${modalCount}`
                  : `Local · ${gpus.length} GPU${gpus.length === 1 ? "" : "s"}`}
          </Text>
        )}
        {phase === "compute" && compute === "local" && gpus.length === 0 && (
          <Text color={C.warm}>no local GPUs here — pick Remote (SSH) or Cloud (dstack)</Text>
        )}
      </Box>

      {/* 0c · dstack cloud config */}
      {compute === "dstack" && pastOrAt("dstackcfg") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "dstackcfg" ? C.accent : C.muted} bold>
            {mark("dstackcfg")}Cloud (dstack)
          </Text>
          {phase === "dstackcfg" ? (
            <>
              {dstackOk === false && (
                <Text color={C.warm}>⚠ dstack CLI not found — install it (pip install dstack) and start `dstack server`</Text>
              )}
              <FieldEditor
                schema={dstackDefs}
                values={dstackFields}
                setValues={setDstackFields}
                active={active}
                onDone={() => {
                  setSource("new");
                  setPhase("mode");
                }}
                doneLabel="continue"
              />
            </>
          ) : (
            <Text color={C.text}>
              {dstackFields.backend} · {dstackFields.gpu}:{dstackFields.count}
              {dstackFields.region ? ` · ${dstackFields.region}` : ""}
            </Text>
          )}
        </Box>
      )}

      {/* 0d · modal GPU config */}
      {compute === "modal" && pastOrAt("modalcfg") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "modalcfg" ? C.accent : C.muted} bold>
            {mark("modalcfg")}Modal GPU
          </Text>
          {phase === "modalcfg" ? (
            <>
              {modalOk === false && (
                <Text color={C.warm}>⚠ modal client not found — install it (uv tool install modal) and connect a token in GPUs → p</Text>
              )}
              <Select
                isDisabled={!active}
                options={MODAL_GPUS.map((g) => ({ label: `${g}${modalCount > 1 ? ` × ${modalCount}` : ""}`, value: g }))}
                onChange={(g) => {
                  setModalGpu(g);
                  setSource("new");
                  setPhase("mode");
                }}
              />
              <Text color={C.dim}>surogate runs inside a Modal sandbox; outputs persist on a Volume (fetch from Files/Runs)</Text>
            </>
          ) : (
            <Text color={C.text}>
              {modalGpu}:{modalCount}
            </Text>
          )}
        </Box>
      )}

      {/* 0b · ssh host */}
      {compute === "ssh" && pastOrAt("sshhost") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "sshhost" ? C.accent : C.muted} bold>
            {mark("sshhost")}SSH host
          </Text>
          {phase === "sshhost" ? (
            <>
              <Box>
                <Text color={C.muted}>host </Text>
                <TextPrompt
                  value={sshHost}
                  onChange={setSshHost}
                  onSubmit={runDetect}
                  active={active}
                  placeholder="user@host  (or a ~/.ssh/config alias) · ⏎ to connect"
                />
              </Box>
              {detect === "busy" && (
                <Box>
                  <Spinner color={C.muted} />
                  <Text color={C.muted}> connecting & detecting GPUs…</Text>
                </Box>
              )}
              {detect === "err" && <Text color={C.red}>✗ {detectErr}</Text>}
              {detect === "ok" && detectErr && <Text color={C.warm}>⚠ {detectErr}</Text>}
            </>
          ) : (
            <Text color={C.text}>
              {sshHost}{" "}
              <Text color={C.dim}>
                · {remoteGpus.length} remote GPU{remoteGpus.length === 1 ? "" : "s"}
                {remoteGpus[0]?.sm != null ? ` · ${gpuSupport(remoteGpus[0]!.sm).arch}` : ""}
              </Text>
            </Text>
          )}
        </Box>
      )}

      {/* 1 · mode */}
      <Box marginTop={1} flexDirection="column">
        <Text color={phase === "mode" ? C.accent : C.muted} bold>
          {mark("mode")}Mode
        </Text>
        {phase === "mode" ? (
          <Select
            isDisabled={!active}
            options={[
              { label: "SFT — supervised fine-tuning", value: "sft" },
              {
                label:
                  compute !== "local"
                    ? "GRPO — local only (remote RL coming soon)"
                    : "GRPO — RL (split GPUs)",
                value: "grpo",
              },
              {
                label:
                  compute !== "local"
                    ? "RULER — local only (remote RL coming soon)"
                    : "RULER — RL + LLM judge (3-way GPU split)",
                value: "ruler",
              },
            ]}
            onChange={(v) => {
              const m = v as Mode;
              if (m !== "sft" && compute !== "local") return; // remote/cloud = SFT only for now
              setMode(m);
              if (m !== "sft" && stackOk === null) checkStack(); // probe the vLLM stack on first RL pick
              if (m !== "sft") setPhase("tgpus");
              else if (isCloud(compute)) {
                setSource("new");
                setPhase("params"); // cloud (dstack/modal) has no source/GPU-select step
              } else if (compute === "ssh") {
                setSource("new"); // remote skips the source step
                setPhase("gpus");
              } else setPhase("source");
            }}
          />
        ) : (
          <Text color={C.text}>{mode.toUpperCase()}</Text>
        )}
      </Box>

      {/* GRPO/RULER preflight: the vLLM stack must be in surogate's venv */}
      {mode !== "sft" && phase !== "mode" && (stackOk === false || installing) && (
        <Box marginTop={1} flexDirection="column">
          {installing ? (
            <>
              <Text color={C.accent}>
                <Spinner /> installing the vLLM stack (vllm · msgspec · uvloop) into surogate's venv…
              </Text>
              {installLog.slice(-6).map((l, i) => (
                <Text key={i} color={C.dim} wrap="truncate-end">
                  {"  "}
                  {l}
                </Text>
              ))}
            </>
          ) : (
            <Text>
              <Text color={C.warm} bold>
                ⚠ vLLM stack missing
              </Text>
              <Text color={C.dim}> — GRPO/RULER need vllm · msgspec · uvloop. Press </Text>
              <Text color={C.gold} bold>
                i
              </Text>
              <Text color={C.dim}> to install into surogate's venv.</Text>
            </Text>
          )}
        </Box>
      )}

      {/* SFT 2 · config source (local only) */}
      {mode === "sft" && compute === "local" && phase !== "mode" && pastOrAt("source") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "source" ? C.accent : C.muted} bold>
            {mark("source")}2 · Config
          </Text>
          {phase === "source" ? (
            <Select
              isDisabled={!active}
              options={[
                { label: "Build a new config (edit parameters)", value: "new" },
                { label: "Use an existing YAML file", value: "yaml" },
              ]}
              onChange={(v) => {
                const s = v as Source;
                setSource(s);
                setPhase(s === "new" ? "gpus" : "yaml");
              }}
            />
          ) : (
            <Text color={C.text}>{source === "yaml" ? "existing YAML" : "build new"}</Text>
          )}
        </Box>
      )}

      {/* SFT existing-YAML picker */}
      {mode === "sft" && source === "yaml" && pastOrAt("yaml") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "yaml" ? C.accent : C.muted} bold>
            {mark("yaml")}2b · YAML file
          </Text>
          {phase === "yaml" ? (
            <YamlPicker
              examples={examples}
              active={active}
              onPick={(p) => {
                setYamlPath(p);
                setPhase("gpus");
              }}
            />
          ) : (
            <Text color={C.text}>{yamlPath}</Text>
          )}
        </Box>
      )}

      {/* SFT 3 · gpus (skipped for cloud — dstack provisions them) */}
      {mode === "sft" && compute !== "dstack" && pastOrAt("gpus") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "gpus" ? C.accent : C.muted} bold>
            {mark("gpus")}3 · GPUs <Text color={C.dim}>{phase === "gpus" ? GPU_HINT : ""}</Text>
          </Text>
          {phase === "gpus" ? (
            <GpuSelect
              active={active}
              options={activeGpus.map((g) => ({ id: g.id, label: gpuLabel(g, estGB) }))}
              initialPicked={compute === "local" ? selected : undefined}
              onSubmit={(ids) => {
                setSelected(ids);
                setPhase(source === "yaml" ? "confirm" : "params");
              }}
            />
          ) : (
            <Text color={C.text}>{selected.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
          )}
        </Box>
      )}

      {/* SFT 4 · parameters (new config only) */}
      {mode === "sft" && source === "new" && pastOrAt("params") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "params" ? C.accent : C.muted} bold>
            {mark("params")}4 · Parameters <Text color={C.dim}>{phase === "params" ? EDIT_HINT : ""}</Text>
          </Text>
          {phase === "params" ? (
            <FieldEditor
              schema={sftDefs}
              values={fields}
              setValues={setFields}
              active={active}
              onDone={() => setPhase("confirm")}
              onReset={() => setFields({ ...DEFAULT_FIELDS, model: fields.model, datasetPath: fields.datasetPath })}
            />
          ) : (
            <Text color={C.text}>
              {fields.recipe} · {fields.maxSteps} steps · lr {fields.learningRate} · bs {fields.perDeviceBatch}×
              {fields.gradAccum} · seq {fields.sequenceLen}
              {fields.lora ? ` · LoRA r${fields.loraRank}` : " · full FT"}
            </Text>
          )}
        </Box>
      )}

      {/* GRPO / RULER: trainer + vllm (+ judge) gpus */}
      {(mode === "grpo" || mode === "ruler") &&
        pastOrAt("tgpus") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "tgpus" ? C.accent : C.muted} bold>
            {mark("tgpus")}2 · Trainer GPUs <Text color={C.dim}>{phase === "tgpus" ? GPU_HINT : ""}</Text>
          </Text>
          {phase === "tgpus" ? (
            <GpuSelect
              active={active}
              options={gpus.map((g) => ({ id: g.id, label: gpuLabel(g, estGB) }))}
              onSubmit={(ids) => {
                setTrainerGpus(ids);
                setPhase("vgpus");
              }}
            />
          ) : (
            <Text color={C.text}>{trainerGpus.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
          )}
          {pastOrAt("vgpus") && (
            <Box flexDirection="column" marginTop={1}>
              <Text color={phase === "vgpus" ? C.accent : C.muted} bold>
                {mark("vgpus")}3 · Inference (vLLM) GPUs — must be disjoint{" "}
                <Text color={C.dim}>{phase === "vgpus" ? GPU_HINT : ""}</Text>
              </Text>
              {phase === "vgpus" ? (
                <GpuSelect
                  active={active}
                  options={gpus
                    .filter((g) => !trainerGpus.includes(g.id))
                    .map((g) => ({ id: g.id, label: gpuLabel(g, null) }))}
                  onSubmit={(ids) => {
                    setVllmGpus(ids);
                    setPhase(mode === "ruler" ? "jgpus" : "gparams");
                  }}
                />
              ) : (
                <Text color={C.text}>{vllmGpus.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
              )}
            </Box>
          )}
          {mode === "ruler" && pastOrAt("jgpus") && (
            <Box flexDirection="column" marginTop={1}>
              <Text color={phase === "jgpus" ? C.accent : C.muted} bold>
                {mark("jgpus")}4 · Judge GPUs — LLM-as-judge vLLM (disjoint){" "}
                <Text color={C.dim}>{phase === "jgpus" ? GPU_HINT : ""}</Text>
              </Text>
              {phase === "jgpus" ? (
                <GpuSelect
                  active={active}
                  options={gpus
                    .filter((g) => !trainerGpus.includes(g.id) && !vllmGpus.includes(g.id))
                    .map((g) => ({ id: g.id, label: gpuLabel(g, null) }))}
                  onSubmit={(ids) => {
                    setJudgeGpus(ids);
                    setPhase("gparams");
                  }}
                />
              ) : (
                <Text color={C.text}>{judgeGpus.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* GRPO / RULER parameters (overlay the example train + orch yaml) */}
      {(mode === "grpo" || mode === "ruler") && pastOrAt("gparams") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "gparams" ? C.accent : C.muted} bold>
            {mark("gparams")}{mode === "ruler" ? "5" : "4"} · Parameters{" "}
            <Text color={C.dim}>{phase === "gparams" ? "— overlays examples/" + mode + "/train+orch.yaml" : ""}</Text>
          </Text>
          {phase === "gparams" ? (
            <FieldEditor
              schema={grpoDefs}
              values={grpoFields}
              setValues={setGrpoFields}
              active={active}
              onDone={() => setPhase("confirm")}
              onReset={() => setGrpoFields({ ...GRPO_DEFAULTS })}
            />
          ) : (
            <Text color={C.text}>
              lr {grpoFields.learningRate} · {grpoFields.maxSteps} steps · group {grpoFields.rolloutsPerExample} ·{" "}
              {grpoFields.recipe}
            </Text>
          )}
        </Box>
      )}

      {/* confirm */}
      {phase === "confirm" && (
        <Box marginTop={1} flexDirection="column">
          <Text color={C.accent} bold>
            Confirm
          </Text>
          <Box flexDirection="column" marginBottom={1}>
            <Text color={C.muted}>
              compute{"  → "}
              {compute === "dstack" ? (
                <Text color={C.eval}>
                  dstack · {dstackFields.gpu}:{dstackFields.count} · {dstackFields.backend}
                  <Text color={C.dim}> (provisioned on launch · metrics streamed back)</Text>
                </Text>
              ) : compute === "ssh" ? (
                <Text color={C.eval}>{sshHost} <Text color={C.dim}>(remote · feed mirrored back over SSH)</Text></Text>
              ) : (
                <Text color={C.text}>local · {selected.map((g) => `gpu${g}`).join(",") || "—"}</Text>
              )}
            </Text>
            <Text color={C.muted}>
              outputs{"   → "}
              {compute === "dstack" ? (
                <Text color={C.text}>on the cloud instance (output_dir) — stream metrics here</Text>
              ) : compute === "ssh" ? (
                <Text color={C.text}>~/.surogate-watch/remote/&lt;run&gt;/output (on the server)</Text>
              ) : mode !== "sft" ? (
                <Text color={C.text}>per examples/{mode}/train.yaml (output_dir)</Text>
              ) : source === "yaml" ? (
                <Text color={C.text}>per {yamlPath} (output_dir)</Text>
              ) : (
                <>
                  <Text color={C.text}>{outAbs}/</Text>
                  <Text color={C.dim}>{"   weights · checkpoints · adapter"}</Text>
                </>
              )}
            </Text>
            <Text color={C.muted}>
              live feed{" → "}
              <Text color={C.text}>{RUNS_DIR}/</Text>
              <Text color={C.dim}>{"   one .jsonl + .log per run"}</Text>
            </Text>
          </Box>
          <Text color={C.green}>$ {command}</Text>
          <Box>
            <Text color={C.text}>Start this {mode.toUpperCase()} run? </Text>
            <ConfirmInput
              isDisabled={!active}
              onConfirm={mode === "sft" ? doLaunchSft : doLaunchGrpo}
              onCancel={() => setPhase("mode")}
            />
          </Box>
        </Box>
      )}

      {phase === "started" && (
        <Box marginTop={1} flexDirection="column">
          <Text color={C.green}>
            ✓ {mode.toUpperCase()} started{" "}
            <Text color={C.dim}>
              {compute === "dstack"
                ? `on dstack (${dstackFields.gpu}:${dstackFields.count} · provisioning…)`
                : compute === "ssh"
                  ? `on ${sshHost} (tmux)`
                  : `(pid ${pid})`}
            </Text>
          </Text>
          <StatusVerb />
        </Box>
      )}
    </Box>
  );
}
