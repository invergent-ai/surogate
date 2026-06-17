import React, { useEffect, useState } from "react";
import { Text } from "ink";
import { C } from "./theme.ts";

// A terminal take on PostHog's signature "shimmer": a soft highlight band sweeps
// left→right across the text, ~3s per pass, to signal active-but-waiting work
// without the frantic feel of a fast spinner. Used for loading/connecting states.
const STEP_MS = 110;

export function ShimmerText({ text, dim = C.dim }: { text: string; dim?: string }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setT((x) => x + 1), STEP_MS);
    return () => clearInterval(id);
  }, []);

  // read the live theme each render so the band recolors on a theme toggle
  const BAND = [C.gold, "#ffe39a", "#fff4d6", "#ffe39a", C.gold];
  const chars = [...text];
  const period = chars.length + 8; // a gap of fully-dim frames between passes
  const crest = t % period;
  return (
    <Text>
      {chars.map((ch, i) => {
        const d = crest - i; // distance from the crest, only lights chars behind it
        const color = d >= 0 && d < BAND.length ? BAND[d]! : dim;
        return (
          <Text key={i} color={color}>
            {ch}
          </Text>
        );
      })}
    </Text>
  );
}
