// Surogate brand: gold gradient + leaping-rabbit mascot + "SUROGATE" wordmark,
// rendered as clean half-block ASCII art (no raster image — crisp at any size).
import { C, hexToRgb } from "./theme.ts";

// Leaping rabbit mascot (half-block art, 28 wide).
export const RABBIT: string[] = [
  "          ██▄",
  "         ████▄",
  "         █████▄  ▄▄▄",
  "        ███████▀▀▀▀▀▀▀█▄",
  "▄▄▄▄████▀▀▀           █▀",
  "▀█████▀           ▄▄█▀",
  "  ▀████▄        ▀███",
  "     ▀███▄         ▀█▄",
  "      █▀             ▀█▄",
  "    ▄█▄▄▄▄▄▄▄▄▄▄      ▀█▄",
  "    ▀▀████▀▀  ▀▀▀█▄▄    ██",
  "      ▀██▀         ▀█▄   █▄",
  "                     ▀██  █",
  "                       ▀█▄██",
  "                         ▀██",
  "                          ▀█",
];

// Bold block "SUROGATE" wordmark (5 rows).
export const WORDMARK: string[] = [
  "████  ██ ██ ████   ███   ███   ███  █████ ████ ",
  "██    ██ ██ ██ ██ ██ ██ ██    ██ ██   ██  ██   ",
  " ███  ██ ██ ████  ██ ██ ██ ██ █████   ██  ███  ",
  "   ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██   ██  ██   ",
  "████   ███  ██ ██  ███   ███  ██ ██   ██  ████ ",
];

export const TAGLINE = "FP8 / FP4 · Training · Fine-tuning · RL";

/** Linear brand gradient color for row `i` of `n` (top stop -> bottom stop).
 *  Reads the live theme's brand stops so the wordmark/mascot recolor with the
 *  dark/light theme. */
export function goldAt(i: number, n: number): string {
  const top = hexToRgb(C.brandTop);
  const bot = hexToRgb(C.brandBot);
  const t = n <= 1 ? 0 : i / (n - 1);
  const c = top.map((v, k) => Math.round(v + (bot[k]! - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
