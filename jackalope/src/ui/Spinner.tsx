import React, { useEffect, useState } from "react";
import { Text } from "ink";
import { C } from "./theme.ts";

// A tiny animated braille spinner (like the PostHog wizard / ora) for async
// states: launching a run, waiting for the feed, searching the Hub.
const FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function Spinner({ color = C.gold }: { color?: string }) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % FRAMES.length), 90);
    return () => clearInterval(t);
  }, []);
  return <Text color={color}>{FRAMES[i]}</Text>;
}
