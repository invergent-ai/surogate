// Raster terminal output for the image/sixel graphics modes. terminal-image
// drives the Kitty + iTerm2 protocols; for terminals that speak Sixel (xterm,
// foot, Konsole, Windows Terminal…) we encode RGBA pixels to Sixel ourselves.
// (The default renderer is the native braille line chart — see linechart.ts.)

import terminalImage from "terminal-image";
import { image2sixel } from "sixel";

/** Kitty / iTerm2 via terminal-image. */
export function viaTerminalImage(png: Buffer, cols: number, rows: number): Promise<string> {
  return terminalImage.buffer(png, { width: cols, height: rows, preserveAspectRatio: true });
}

/** Encode RGBA pixels to a Sixel escape string (256-color). */
export function viaSixel(rgba: Uint8Array, width: number, height: number): string {
  return image2sixel(Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength), width, height, 256);
}
