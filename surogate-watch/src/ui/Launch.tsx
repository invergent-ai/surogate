import React, { useMemo, useState } from "react";
import { Box, Text } from "ink";
import { ConfirmInput, MultiSelect, Select } from "@inkjs/ui";
import fs from "node:fs";
import path from "node:path";
import {
  buildCommand,
  buildConfigYaml,
  DEFAULT_FIELDS,
  discoverGpus,
  RECIPES,
  type Recipe,
  spawnTraining,
} from "../launch.ts";
import { C } from "./theme.ts";

type Phase = "gpus" | "recipe" | "confirm" | "started";

export function Launch({
  feedPath,
  surogateBin,
  repoRoot,
  onLaunched,
}: {
  feedPath: string;
  surogateBin: string;
  repoRoot: string;
  onLaunched: () => void;
}) {
  const gpus = useMemo(() => discoverGpus(), []);
  const [phase, setPhase] = useState<Phase>("gpus");
  const [selected, setSelected] = useState<number[]>([]);
  const [recipe, setRecipe] = useState<Recipe>(DEFAULT_FIELDS.recipe);
  const [pid, setPid] = useState<number | null>(null);

  const fields = { ...DEFAULT_FIELDS, recipe };
  const configPath = path.join(repoRoot, "watch-run.yaml");
  const command = buildCommand(selected, configPath, surogateBin);

  const doLaunch = () => {
    fs.writeFileSync(configPath, buildConfigYaml(fields, selected.length));
    const p = spawnTraining(configPath, selected, feedPath, surogateBin);
    setPid(p);
    setPhase("started");
    setTimeout(onLaunched, 1500);
  };

  return (
    <Box flexDirection="column" flexGrow={1} paddingX={1} paddingTop={1}>
      <Text bold color={C.accent}>
        Launch a training run
      </Text>
      <Text color={C.muted}>model {fields.model} · {fields.maxSteps} steps · bsz {fields.perDeviceBatch} · seq {fields.sequenceLen}</Text>

      <Box marginTop={1} flexDirection="column">
        <Text color={phase === "gpus" ? C.accent : C.muted} bold>
          1 · Select GPUs
        </Text>
        {phase === "gpus" ? (
          <MultiSelect
            options={gpus.map((g) => ({
              label: `${g.name}  ·  id ${g.id}${g.memMB ? `  ·  ${(g.memMB / 1024).toFixed(0)} GB` : ""}`,
              value: String(g.id),
            }))}
            onSubmit={(values) => {
              setSelected(values.map(Number));
              setPhase("recipe");
            }}
          />
        ) : (
          <Text color={C.text}>{selected.length ? selected.map((g) => `gpu${g}`).join(", ") : "—"}</Text>
        )}
      </Box>

      {(phase === "recipe" || phase === "confirm" || phase === "started") && (
        <Box marginTop={1} flexDirection="column">
          <Text color={phase === "recipe" ? C.accent : C.muted} bold>
            2 · Precision recipe
          </Text>
          {phase === "recipe" ? (
            <Select
              options={RECIPES.map((r) => ({ label: r, value: r }))}
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

      {phase === "confirm" && (
        <Box marginTop={1} flexDirection="column">
          <Text color={C.accent} bold>
            3 · Confirm
          </Text>
          <Text color={C.green}>$ {command}</Text>
          <Box>
            <Text color={C.text}>Start this training run? </Text>
            <ConfirmInput onConfirm={doLaunch} onCancel={() => setPhase("gpus")} />
          </Box>
        </Box>
      )}

      {phase === "started" && (
        <Box marginTop={1}>
          <Text color={C.green}>✓ training started (pid {pid}) — switching to Monitor…</Text>
        </Box>
      )}
    </Box>
  );
}
