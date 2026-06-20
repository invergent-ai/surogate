import React, { useEffect, useMemo, useState } from "react";
import { Box, Text } from "ink";
import fs from "node:fs";
import { discoverGpus } from "../launch.ts";
import { TIPS } from "../tips.ts";
import { C, getTheme } from "./theme.ts";
import { RABBIT, TAGLINE, WORDMARK, goldAt } from "./brand.ts";
import { Showcase } from "./Showcase.tsx";
import { gpuSupport } from "../gpu-support.ts";

// What surogate actually does — a compact, real capability showcase (PostHog's
// "colored dot + name + one-line problem" pattern). Grounded in surogate's docs.
type DotColor = "accent" | "green" | "eval" | "warm" | "gold";
const CAPABILITIES: { name: string; desc: string; color: DotColor }[] = [
  { name: "Precision recipes", desc: "bf16 · fp8-hybrid · NVFP4 4-bit", color: "accent" },
  { name: "LoRA / QLoRA", desc: "adapters on tiny VRAM (NF4/fp8/fp4)", color: "green" },
  { name: "RL fine-tuning", desc: "GRPO · RULER with an LLM judge", color: "eval" },
  { name: "Mixture-of-Experts", desc: "expert-parallel · router LoRA", color: "warm" },
  { name: "Scale & offload", desc: "multi-GPU ZeRO 1/2/3 · CPU offload", color: "gold" },
];

const GENERAL_TIPS = TIPS.filter((t) => !t.tags || t.tags.length === 0).map((t) => t.t);

function RotatingTip() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setI((x) => x + 1), 7000);
    return () => clearInterval(id);
  }, []);
  if (GENERAL_TIPS.length === 0) return null;
  return (
    <Box>
      <Text>💡 </Text>
      <Box flexShrink={1}>
        <Text color={C.text}>{GENERAL_TIPS[i % GENERAL_TIPS.length]}</Text>
      </Box>
    </Box>
  );
}

export function StartScreen({ feedPath, version, needsSetup }: { feedPath: string; version: string; needsSetup?: boolean }) {
  const gpus = useMemo(() => discoverGpus(), []);
  // "live" = written to recently; a finished run's file is non-empty but stale
  // (that's the "last run, idle" case), so size alone would mislabel it as live.
  const feedStatus = useMemo<"live" | "idle" | "waiting">(() => {
    try {
      const st = fs.statSync(feedPath);
      if (st.size === 0) return "waiting";
      return Date.now() - st.mtimeMs < 15000 ? "live" : "idle";
    } catch {
      return "waiting";
    }
  }, [feedPath]);

  const byName = new Map<string, number>();
  for (const g of gpus) byName.set(g.name, (byName.get(g.name) ?? 0) + 1);
  const idle = gpus.filter((g) => !g.busy).length;

  return (
    <Box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} paddingY={1}>
      {/* hero — centered */}
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
          <Text>
            <Text color={C.gold} bold>
              jackalope
            </Text>
            <Text color={C.dim}> · live training dashboard {version ? `· v${version}` : ""}</Text>
          </Text>
        </Box>
      </Box>

      {/* two columns, centered (animation + tip · what it trains + devices) */}
      <Box marginTop={1}>
        <Box flexDirection="column" width={40}>
          <Showcase />
          <Box marginTop={1}>
            <RotatingTip />
          </Box>
        </Box>

        <Box flexDirection="column" marginLeft={4} width={56}>
          <Text color={C.dim}>WHAT SUROGATE TRAINS</Text>
          <Box flexDirection="column" marginTop={1}>
            {CAPABILITIES.map((cap) => (
              <Text key={cap.name}>
                <Text color={C[cap.color]}>● </Text>
                <Text color={C.text} bold>
                  {cap.name}
                </Text>
                <Text color={C.dim}> — </Text>
                <Text color={C.muted}>{cap.desc}</Text>
              </Text>
            ))}
          </Box>
          <Box flexDirection="column" marginTop={1}>
            <Text color={C.dim}>DEVICES</Text>
            {gpus.length ? (
              <>
                <Text>
                  {[...byName.entries()].map(([name, n], i) => (
                    <Text key={name}>
                      {i > 0 ? <Text color={C.dim}> · </Text> : null}
                      <Text color={C.green}>●</Text>
                      <Text color={C.text}>
                        {" "}
                        {n}× {name}
                      </Text>
                    </Text>
                  ))}
                  <Text color={C.dim}>{`   ${idle}/${gpus.length} idle`}</Text>
                </Text>
                {gpus[0]!.sm !== null && (
                  <Text>
                    <Text color={C.accent}>{gpuSupport(gpus[0]!.sm).arch}</Text>
                    <Text color={C.dim}> · trains </Text>
                    <Text color={C.gold}>{gpuSupport(gpus[0]!.sm).recipes.join(" · ")}</Text>
                  </Text>
                )}
              </>
            ) : (
              <Text color={C.muted}>no NVIDIA GPUs here — use Remote/Cloud in Launch</Text>
            )}
          </Box>
        </Box>
      </Box>

      {/* centered footer: feed status + star + prompt */}
      <Box marginTop={1} flexDirection="column" alignItems="center">
        <Text>
          <Text color={C.dim}>feed </Text>
          <Text color={C.muted}>{feedPath}</Text>
          {"  "}
          {feedStatus === "live" ? (
            <Text color={C.green}>● live</Text>
          ) : feedStatus === "idle" ? (
            <Text color={C.muted}>○ idle · last run</Text>
          ) : (
            <Text color={C.warm}>○ waiting</Text>
          )}
        </Text>
        <Box marginTop={1}>
          <Text color={C.gold}>★ </Text>
          <Text color={C.dim}>star </Text>
          <Text color={C.eval}>github.com/invergent-ai/surogate</Text>
        </Box>
        {needsSetup ? (
          <Box marginTop={1} flexDirection="column" alignItems="center">
            <Text>
              <Text color={C.accent}>▸ </Text>
              <Text color={C.gold} bold>
                ⏎ set up surogate
              </Text>
              <Text color={C.dim}> — pick compute & install in one flow</Text>
            </Text>
            <Box marginTop={1}>
              <Text color={C.gold} bold>
                d
              </Text>
              <Text color={C.dim}> skip to dashboard · </Text>
              <Text color={C.gold} bold>
                t
              </Text>
              <Text color={C.dim}>{` theme (${getTheme()}) · `}</Text>
              <Text color={C.gold} bold>
                q
              </Text>
              <Text color={C.dim}> quit</Text>
            </Box>
          </Box>
        ) : (
          <Box marginTop={1}>
            <Text color={C.gold} bold>
              ⏎ enter
            </Text>
            <Text color={C.dim}> open dashboard · </Text>
            <Text color={C.gold} bold>
              s
            </Text>
            <Text color={C.dim}> set up · </Text>
            <Text color={C.gold} bold>
              t
            </Text>
            <Text color={C.dim}>{` theme (${getTheme()}) · `}</Text>
            <Text color={C.gold} bold>
              g
            </Text>
            <Text color={C.dim}> github · </Text>
            <Text color={C.gold} bold>
              q
            </Text>
            <Text color={C.dim}> quit</Text>
          </Box>
        )}
      </Box>
    </Box>
  );
}
