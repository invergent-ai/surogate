// Terminal image rendering. terminal-image handles Kitty + iTerm2 (plus a
// colored half-block fallback that works on any truecolor terminal). When the
// terminal speaks Sixel but not Kitty/iTerm2 (xterm, foot, Konsole, Windows
// Terminal…), we encode RGBA pixels to Sixel ourselves for real raster output.

import terminalImage from "terminal-image";
import { image2sixel } from "sixel";

let _sixelPreferred = false;

/** Set by startup detection (DA1 query): terminal supports Sixel, not Kitty/iTerm2. */
export function setSixelPreferred(on: boolean): void {
  _sixelPreferred = on;
}
export function sixelPreferred(): boolean {
  return _sixelPreferred;
}

/** Kitty / iTerm2 / half-block via terminal-image. */
export function viaTerminalImage(png: Buffer, cols: number, rows: number): Promise<string> {
  return terminalImage.buffer(png, { width: cols, height: rows, preserveAspectRatio: true });
}

/** Encode RGBA pixels to a Sixel escape string (256-color). */
export function viaSixel(rgba: Uint8Array, width: number, height: number): string {
  return image2sixel(Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width, height, 256);
}
