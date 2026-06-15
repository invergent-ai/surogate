// Shared color palette (matches the chart).
export const C = {
  accent: "#5ad1c4",
  eval: "#d18fff",
  train: "#5ad1c4",
  muted: "#6b7280",
  dim: "#4b5163",
  text: "#c7ccd6",
  warm: "#ffb454",
  red: "#ff5c5c",
  green: "#5ad17a",
  border: "#3a3f4b",
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
