import React, { useMemo } from "react";
import { Box, Text } from "ink";
import fs from "node:fs";
import { discoverGpus } from "../launch.ts";
import { C } from "./theme.ts";
import { GOLD, RABBIT, TAGLINE, WORDMARK, goldAt } from "./brand.ts";

const TIPS = [
  "↑↓ navigate pages · l launch a run · p pause · q quit",
  "report_to: [surogate] in your training config feeds this UI",
  "fp8-hybrid ≈ 1.8× faster than bf16 on Blackwell",
  "real inline graphs on Kitty · Ghostty · iTerm2 · WezTerm",
];

export function StartScreen({ feedPath, version }: { feedPath: string; version: string }) {
  const gpus = useMemo(() => discoverGpus(), []);
  const live = useMemo(() => {
    try {
      return fs.statSync(feedPath).size > 0;
    } catch {
      return false;
    }
  }, [feedPath]);

  // group GPUs by name for a "4× RTX 5090" summary
  const byName = new Map<string, number>();
  for (const g of gpus) byName.set(g.name, (byName.get(g.name) ?? 0) + 1);

  return (
    <Box flexDirection="column" paddingX={3} paddingY={1}>
      {/* mascot + wordmark */}
      <Box>
        <Box flexDirection="column">
          {RABBIT.map((line, i) => (
            <Text key={i} color={goldAt(i, RABBIT.length)}>
              {line}
            </Text>
          ))}
        </Box>
        <Box flexDirection="column" marginLeft={4} marginTop={3}>
          {WORDMARK.map((line, i) => (
            <Text key={i} color={goldAt(i, WORDMARK.length)} bold>
              {line}
            </Text>
          ))}
          <Box marginTop={1}>
            <Text color={C.text}>{TAGLINE}</Text>
          </Box>
          <Text color={C.dim}>live training dashboard {version ? `· v${version}` : ""}</Text>
        </Box>
      </Box>

      {/* devices + tips */}
      <Box marginTop={1}>
        <Box flexDirection="column" width={36}>
          <Text color={GOLD} bold>
            DEVICES
          </Text>
          {gpus.length ? (
            <>
              {[...byName.entries()].map(([name, n]) => (
                <Text key={name} color={C.text}>
                  <Text color={C.green}>●</Text> {n}× {name}
                </Text>
              ))}
              {gpus.slice(0, 8).map((g) => {
                const free = g.memMB === null ? null : (g.memMB - (g.memUsedMB ?? 0)) / 1024;
                return (
                  <Text key={g.id} color={C.muted}>
                    {"  "}gpu{g.id}
                    {free !== null ? ` · ${free.toFixed(0)}/${((g.memMB ?? 0) / 1024).toFixed(0)} GB free` : ""}
                    {g.busy ? <Text color={C.warm}> · busy</Text> : <Text color={C.green}> · idle</Text>}
                  </Text>
                );
              })}
            </>
          ) : (
            <Text color={C.muted}>no GPUs detected</Text>
          )}
        </Box>
        <Box flexDirection="column">
          <Text color={GOLD} bold>
            TIPS
          </Text>
          {TIPS.map((t, i) => (
            <Text key={i} color={C.text}>
              <Text color={GOLD}>💡</Text> {t}
            </Text>
          ))}
        </Box>
      </Box>

      {/* feed + prompt */}
      <Box marginTop={1} flexDirection="column">
        <Text>
          <Text color={C.dim}>feed </Text>
          <Text color={C.muted}>{feedPath}</Text>
          {"  "}
          {live ? <Text color={C.green}>● live</Text> : <Text color={C.warm}>○ waiting</Text>}
        </Text>
        <Box marginTop={1}>
          <Text color={C.dim}>press </Text>
          <Text color={GOLD} bold>
            ⏎ enter
          </Text>
          <Text color={C.dim}> to open the dashboard · </Text>
          <Text color={GOLD} bold>
            q
          </Text>
          <Text color={C.dim}> quit</Text>
        </Box>
      </Box>
    </Box>
  );
}
