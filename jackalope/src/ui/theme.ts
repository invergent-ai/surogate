// Shared color palette — Surogate brand gold, with dark/light variants.
//
// `C` is a single mutable object that every component reads (`C.accent`, …).
// Switching theme mutates `C` in place (Object.assign) and the app re-renders,
// so no component needs a context or prop threading. Values are foreground-only
// (terminals own their background); the light palette uses darker inks so it
// stays legible on a white terminal. Contrast targets are adapted from PostHog's
// light/dark token families.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export interface Palette {
  accent: string;
  gold: string;
  goldDeep: string;
  eval: string;
  train: string;
  muted: string;
  dim: string;
  text: string;
  warm: string;
  red: string;
  green: string;
  border: string;
  bg: string; // full-screen canvas background (painted so light mode is real)
  panel: string; // subtle panel background, a hair off the canvas
  brandTop: string; // mascot/wordmark gradient stops
  brandBot: string;
}

const DARK: Palette = {
  accent: "#ffd15c",
  gold: "#ffd15c",
  goldDeep: "#e4961c",
  eval: "#9fd0ff",
  train: "#ffd15c",
  muted: "#8a8f9a",
  dim: "#54585f",
  text: "#d7d3c8",
  warm: "#ffb454",
  red: "#ff6b5c",
  green: "#8fd17a",
  border: "#262626", // neutral dark gray (no warm tint)
  bg: "#000000", // pure black — blends with the chart + logs
  panel: "#0b0b0b", // a hair off black for subtle bars/cards
  brandTop: "#ffd15c",
  brandBot: "#e4961c",
};

const LIGHT: Palette = {
  accent: "#b26a00", // darker amber: readable on white
  gold: "#a86d12",
  goldDeep: "#7a4e0c",
  eval: "#1d4aff", // PostHog data-color-1 blue
  train: "#b26a00",
  muted: "#5f5f5f",
  dim: "#a0a0a0",
  text: "#1a1a1a",
  warm: "#874d00",
  red: "#a01f0f",
  green: "#3a7a00",
  border: "#e4e4e7", // neutral light gray
  bg: "#ffffff", // pure white canvas
  panel: "#f4f4f5", // very light gray surface
  brandTop: "#c8881c",
  brandBot: "#8a5a10",
};

export type ThemeName = "dark" | "light";

// The live palette. Starts as dark; `applyTheme` swaps the contents in place.
export const C: Palette = { ...DARK };

// Compare-series color (theme-independent). Lives here, not in chart.ts, so the
// braille renderer can use it without statically importing chart.ts (which would
// drag echarts + the resvg native addon into startup / the standalone binary).
export const COMPARE = "#7b93c4"; // muted blue, distinct from eval

let current: ThemeName = "dark";
export function getTheme(): ThemeName {
  return current;
}

const PREF_FILE = path.join(os.homedir(), ".surogate-watch", "theme");

export function applyTheme(name: ThemeName): void {
  current = name;
  Object.assign(C, name === "light" ? LIGHT : DARK);
}

export function saveThemePref(name: ThemeName): void {
  try {
    fs.mkdirSync(path.dirname(PREF_FILE), { recursive: true });
    fs.writeFileSync(PREF_FILE, name);
  } catch {
    /* best effort */
  }
}

/** Preferred theme: saved choice → terminal background ($COLORFGBG) → dark. */
export function detectDefaultTheme(): ThemeName {
  try {
    const saved = fs.readFileSync(PREF_FILE, "utf8").trim();
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* no saved pref */
  }
  // COLORFGBG="fg;bg"; bg 7 or 15 = a light terminal (8 is "default", treat as dark)
  const v = process.env["COLORFGBG"];
  if (v) {
    const parts = v.split(";");
    const bg = Number(parts[parts.length - 1]);
    if (!Number.isNaN(bg) && bg >= 7 && bg !== 8) return "light";
  }
  return "dark";
}

// Apply the preferred theme at import time so the first paint is already correct.
applyTheme(detectDefaultTheme());

export function hexToRgb(h: string): [number, number, number] {
  const n = h.replace("#", "");
  return [parseInt(n.slice(0, 2), 16), parseInt(n.slice(2, 4), 16), parseInt(n.slice(4, 6), 16)];
}

/** Loss-trend → arrow glyph + color (▼ down = good/green, ▲ up = bad/red). */
export function lossArrow(trend: number): { arrow: string; color: string } {
  if (trend < 0) return { arrow: "▼", color: C.green };
  if (trend > 0) return { arrow: "▲", color: C.red };
  return { arrow: "•", color: C.muted };
}

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
