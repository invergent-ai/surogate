// Shared color palette — Surogate brand gold.
export const C = {
  accent: "#ffd15c", // brand gold
  gold: "#ffd15c",
  goldDeep: "#e4961c",
  eval: "#9fd0ff", // cool blue, contrasts with gold
  train: "#ffd15c",
  muted: "#8a8f9a",
  dim: "#54585f",
  text: "#d7d3c8",
  warm: "#ffb454",
  red: "#ff6b5c",
  green: "#8fd17a",
  border: "#4a4434",
} as const;

export function tempColor(t: number | null): string {
  if (t === null) return C.muted;
  if (t >= 80) return C.red;
  if (t >= 70) return C.warm;
  return C.green;
}

export function memColor(frac: number): string {
  if (frac >= 0.9) return C.red;
  if (frac >= 0.75) return C.warm;
  return C.green;
}

/** A solid block meter: filled portion + dim track + percent. */
export function meterParts(frac: number | null, width = 10): { filled: number; track: number; pct: string } {
  if (frac === null) return { filled: 0, track: width, pct: "  —" };
  const f = Math.max(0, Math.min(1, frac));
  const filled = Math.round(f * width);
  return { filled, track: width - filled, pct: `${Math.round(f * 100)}`.padStart(3) + "%" };
}
