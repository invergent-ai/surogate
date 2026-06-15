// Surogate brand: gold gradient + leaping-rabbit mascot + wordmark.
// Ported from surogate/utils/banner.py so the dashboard matches the CLI brand.

export const GOLD = "#FFD15C"; // bright gold (top)
export const GOLD_DEEP = "#E4961C"; // deep amber (bottom)
export const GOLD_MID = "#F2B441";

// Leaping rabbit mascot (half-block render of the brand favicon, 28 wide).
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

/** Linear gold gradient color for row `i` of `n` (top bright -> bottom amber). */
export function goldAt(i: number, n: number): string {
  const top = [255, 209, 92];
  const bot = [228, 150, 28];
  const t = n <= 1 ? 0 : i / (n - 1);
  const c = top.map((v, k) => Math.round(v + (bot[k]! - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
