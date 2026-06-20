import React from "react";
import { Box, Text } from "ink";
import { C } from "./theme.ts";
import { RABBIT, goldAt } from "./brand.ts";

// A full-screen keybinding reference (toggled with ?). Any key dismisses it.
const SECTIONS: { title: string; keys: [string, string][] }[] = [
  {
    title: "NAVIGATE",
    keys: [
      ["↑ ↓ / j k", "move through the tab menu"],
      ["1 – 8", "jump straight to a tab"],
      ["⏎ / →", "enter the tab / select"],
      ["esc / ←", "back to the tab menu"],
    ],
  },
  {
    title: "VIEW",
    keys: [
      ["s", "smooth the loss curve (EMA)"],
      ["t", "toggle dark / light theme"],
      ["p", "pause the live view"],
      ["c", "compare a run (in Runs)"],
      ["?", "this help"],
    ],
  },
  {
    title: "RUN CONTROL",
    keys: [
      ["z", "suspend / resume (SIGSTOP)"],
      ["x", "stop the run (press twice)"],
      ["r", "resume ckpt · request model"],
      ["e", "export run metrics (Runs)"],
      ["y", "copy run path (Runs)"],
    ],
  },
  {
    title: "GENERAL",
    keys: [
      ["g", "★ open surogate on GitHub"],
      ["q q", "quit (press twice) · ^C force"],
    ],
  },
];

function Section({ title, keys }: { title: string; keys: [string, string][] }) {
  return (
    <Box flexDirection="column" marginBottom={1} width={42}>
      <Text color={C.dim}>{title}</Text>
      {keys.map(([k, desc]) => (
        <Text key={k}>
          <Text color={C.gold} bold>
            {k.padEnd(13)}
          </Text>
          <Text color={C.text}>{desc}</Text>
        </Text>
      ))}
    </Box>
  );
}

export function HelpOverlay() {
  const left = SECTIONS.slice(0, 2);
  const right = SECTIONS.slice(2);
  return (
    <Box flexDirection="column" alignItems="center" justifyContent="center" flexGrow={1} paddingY={1}>
      {/* full mascot + title, centered */}
      <Box>
        <Box flexDirection="column">
          {RABBIT.map((line, i) => (
            <Text key={i} color={goldAt(i, RABBIT.length)}>
              {line}
            </Text>
          ))}
        </Box>
        <Box flexDirection="column" marginLeft={4} marginTop={4}>
          <Text color={C.gold} bold>
            ◆ jackalope
          </Text>
          <Text color={C.text}>keyboard reference</Text>
          <Text color={C.dim}>press any key to close</Text>
        </Box>
      </Box>

      {/* two columns of shortcut groups, centered */}
      <Box marginTop={1}>
        <Box flexDirection="column" marginRight={2}>
          {left.map((sec) => (
            <Section key={sec.title} title={sec.title} keys={sec.keys} />
          ))}
        </Box>
        <Box flexDirection="column">
          {right.map((sec) => (
            <Section key={sec.title} title={sec.title} keys={sec.keys} />
          ))}
        </Box>
      </Box>
    </Box>
  );
}
