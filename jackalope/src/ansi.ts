// Shared truecolor ANSI escape helpers used by the chart + logo renderers.

export const RESET = "\x1b[0m";
export const DEFAULT_FG = "\x1b[39m";
export const DEFAULT_BG = "\x1b[49m";

export function fgRgb(r: number, g: number, b: number): string {
  return `\x1b[38;2;${r};${g};${b}m`;
}
export function bgRgb(r: number, g: number, b: number): string {
  return `\x1b[48;2;${r};${g};${b}m`;
}
/** Foreground escape from a #rrggbb hex string. */
export function fgHex(hex: string): string {
  return fgRgb(parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16));
}

/** Strip ANSI escape sequences and non-printable control chars (keeping tab) so
 *  untrusted/streamed output can't corrupt the terminal grid when rendered. */
export function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/[\x00-\x08\x0b-\x1f\x7f]/g, "");
}
