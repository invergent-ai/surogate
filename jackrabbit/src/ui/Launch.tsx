import React, { useMemo, useState } from "react";
import { Box, Text } from "ink";
import { ConfirmInput, MultiSelect, Select } from "@inkjs/ui";
import fs from "node:fs";
import path from "node:path";
import {
  buildCommand,
  buildConfigYaml,
  buildGrpoCommand,
  DEFAULT_FIELDS,
  discoverGpus,
  estimateRunVramGB,
  exampleGrpoConfigs,
  fitOnGpu,
  gpuFreeGB,
  grpoConfigsExist,
  RECIPES,
  type Gpu,
  type Recipe,
  spawnGrpo,
  spawnTraining,
} from "../launch.ts";
import { newRunFeedPath } from "../runs.ts";
import { C } from "./theme.ts";

type Mode = "sft" | "grpo";
type Phase = "mode" | "gpus" | "tgpus" | "vgpus" | "recipe" | "confirm" | "started";

function gpuLabel(g: Gpu, estGB: number | null): string {
  const free = gpuFreeGB(g);
  const freeStr = free === null ? "" : ` · ${free.toFixed(0)}GB free`;
  const verdict = fitOnGpu(estGB, free);
  const tag = g.busy ? " · busy" : "";
  const fit = verdict === "risk" ? " · ⚠ tight" : verdict === "fits" ? " · ✓ fits" : "";
  return `${g.name} · id ${g.id}${freeStr}${tag}${fit}`;
}

export function Launch({
  feedPath,
  surogateBin,
  repoRoot,
  active,
  picked,
  onLaunched,
}: {
  feedPath: string;
  surogateBin: string;
  repoRoot: string;
  active: boolean;
  picked?: { model?: { id: string; t: { supported: boolean; recipes: readonly string[] } }; dataset?: string };
  onLaunched: (metricsPath: string) => void;
}) {
  const gpus = useMemo(() => discoverGpus(), []);
  const [mode, setMode] = useState<Mode>("sft");
  const [phase, setPhase] = useState<Phase>("mode");
  const [selected, setSelected] = useState<number[]>([]);
  const [trainerGpus, setTrainerGpus] = useState<number[]>([]);
  const [vllmGpus, setVllmGpus] = useState<number[]>([]);
  const [recipe, setRecipe] = useState<Recipe>(DEFAULT_FIELDS.recipe);
  const [pid, setPid] = useState<number | null>(null);

  // models/datasets picked from the HF browser override the defaults
  const recipeChoices = (picked?.model?.t.supported ? picked.model.t.recipes : RECIPES).filter((r): r is Recipe =>
    (RECIPES as readonly string[]).includes(r),
  );
  const fields = {
    ...DEFAULT_FIELDS,
    recipe,
    model: picked?.model?.id ?? DEFAULT_FIELDS.model,
    datasetPath: picked?.dataset ?? DEFAULT_FIELDS.datasetPath,
  };
  const estGB = estimateRunVramGB(fields);
  const grpo = exampleGrpoConfigs(repoRoot);
  const grpoOk = grpoConfigsExist(grpo);

  const doLaunchSft = () => {
    const metricsPath = newRunFeedPath(`sft-${recipe}`, Date.now());
    const configPath = path.join(repoRoot, "watch-run.yaml");
    fs.writeFileSync(configPath, buildConfigYaml(fields, selected.length, metricsPath));
    const p = spawnTraining(configPath, selected, metricsPath, surogateBin);
    setPid(p);
    setPhase("started");
    setTimeout(() => onLaunched(metricsPath), 1200);
  };

  const doLaunchGrpo = () => {
    const metricsPath = newRunFeedPath("grpo", Date.now());
    const p = spawnGrpo(grpo, trainerGpus, vllmGpus, metricsPath, surogateBin);
    setPid(p);
    setPhase("started");
    setTimeout(() => onLaunched(metricsPath), 1200);
  };

  const command =
    mode === "sft"
      ? buildCommand(selected, path.join(repoRoot, "watch-run.yaml"), surogateBin)
      : buildGrpoCommand(trainerGpus, vllmGpus, grpo, surogateBin);

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingTop={1}>
      <Text bold color={C.accent}>
        Launch a training run
      </Text>
      <Text color={C.muted}>
        <Text color={picked?.model ? C.green : C.dim}>{fields.model}</Text>
        {" · "}
        {fields.datasetPath}
        {estGB !== null && (
          <Text>
            {" · ≈ est "}
            <Text color={C.gold}>{estGB} GB</Text>
          </Text>
        )}
      </Text>
      {!picked?.model && <Text color={C.dim}>tip: open Models to search HF + check surogate support</Text>}
      {!active && (
        <Box marginTop={1}>
          <Text color={C.gold}>press ⏎ to configure</Text>
          <Text color={C.dim}> · esc to go back to nav</Text>
        </Box>
      )}

      {/* mode */}
      <Box marginTop={1} flexDirection="column">
        <Text color={phase === "mode" ? C.accent : C.muted} bold>
          1 · Mode
        </Text>
        {phase === "mode" ? (
          <Select
            isDisabled={!active}
            options={[
              { label: "SFT — supervised fine-tuning", value: "sft" },
              { label: grpoOk ? "GRPO — RL (split GPUs)" : "GRPO — needs examples/grpo/*.yaml", value: "grpo" },
            ]}
            onChange={(v) => {
              const m = v as Mode;
              if (m === "grpo" && !grpoOk) return;
              setMode(m);
              setPhase(m === "sft" ? "gpus" : "tgpus");
            }}
          />
        ) : (
          <Text color={C.text}>{mode.toUpperCase()}</Text>
        )}
      </Box>

      {/* SFT: gpus */}
      {mode === "sft" && phase !== "mode" && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "gpus" ? C.accent : C.muted} bold>
            2 · GPUs
          </Text>
          {phase === "gpus" ? (
            <MultiSelect
              isDisabled={!active}
              options={gpus.map((g) => ({ label: gpuLabel(g, estGB), value: String(g.id) }))}
              onSubmit={(values) => {
                setSelected(values.map(Number));
                setPhase("recipe");
              }}
            />
          ) : (
            <Text color={C.text}>{selected.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
          )}
        </Box>
      )}

      {/* GRPO: trainer + vllm gpus */}
      {mode === "grpo" && (phase === "tgpus" || phase === "vgpus" || phase === "confirm" || phase === "started") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "tgpus" ? C.accent : C.muted} bold>
            2 · Trainer GPUs
          </Text>
          {phase === "tgpus" ? (
            <MultiSelect
              isDisabled={!active}
              options={gpus.map((g) => ({ label: gpuLabel(g, estGB), value: String(g.id) }))}
              onSubmit={(values) => {
                setTrainerGpus(values.map(Number));
                setPhase("vgpus");
              }}
            />
          ) : (
            <Text color={C.text}>{trainerGpus.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
          )}
          {(phase === "vgpus" || phase === "confirm" || phase === "started") && (
            <Box flexDirection="column" marginTop={1}>
              <Text color={phase === "vgpus" ? C.accent : C.muted} bold>
                3 · Inference (vLLM) GPUs — must be disjoint
              </Text>
              {phase === "vgpus" ? (
                <MultiSelect
                  isDisabled={!active}
                  options={gpus
                    .filter((g) => !trainerGpus.includes(g.id))
                    .map((g) => ({ label: gpuLabel(g, null), value: String(g.id) }))}
                  onSubmit={(values) => {
                    setVllmGpus(values.map(Number));
                    setPhase("confirm");
                  }}
                />
              ) : (
                <Text color={C.text}>{vllmGpus.map((g) => `gpu${g}`).join(", ") || "—"}</Text>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* SFT: recipe */}
      {mode === "sft" && (phase === "recipe" || phase === "confirm" || phase === "started") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "recipe" ? C.accent : C.muted} bold>
            3 · Precision recipe
          </Text>
          {phase === "recipe" ? (
            <Select
              isDisabled={!active}
              options={recipeChoices.map((r) => ({ label: r, value: r }))}
              onChange={(v) => {
                setRecipe(v as Recipe);
                setPhase("confirm");
              }}
            />
          ) : (
            <Text color={C.text}>{recipe}</Text>
          )}
        </Box>
      )}

      {/* confirm */}
      {phase === "confirm" && (
        <Box marginTop={1} flexDirection="column">
          <Text color={C.accent} bold>
            Confirm
          </Text>
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
        <Box marginTop={1}>
          <Text color={C.green}>✓ {mode.toUpperCase()} started (pid {pid}) — following its feed…</Text>
        </Box>
      )}
    </Box>
  );
}
