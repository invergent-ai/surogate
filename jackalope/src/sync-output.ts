// Flicker-safe rendering via DEC Synchronized Output (mode ?2026).
//
// Inline images (Kitty/Sixel) re-drawn every frame can tear/flicker. Wrapping
// each terminal frame in BSU/ESU tells the terminal to buffer the update and
// present it atomically. Terminals that don't support ?2026 ignore the private
// mode sequences, so this is safe to emit unconditionally.

const BSU = "\x1b[?2026h"; // Begin Synchronized Update
const ESU = "\x1b[?2026l"; // End Synchronized Update

/** Monkeypatch stdout so each escape-bearing write (an Ink frame) is presented
 *  atomically. Returns a restore function. No-op on non-TTY. */
export function enableSynchronizedOutput(): () => void {
  const stream = process.stdout;
  if (!stream.isTTY) return () => {};
  const orig = stream.write.bind(stream) as typeof stream.write;

  const patched = ((chunk: unknown, ...rest: unknown[]) => {
    if (typeof chunk === "string" && chunk.includes("\x1b") && !chunk.startsWith(BSU)) {
      return (orig as (c: unknown, ...r: unknown[]) => boolean)(BSU + chunk + ESU, ...rest);
    }
    return (orig as (c: unknown, ...r: unknown[]) => boolean)(chunk, ...rest);
  }) as typeof stream.write;

  stream.write = patched;
  return () => {
    stream.write = orig;
  };
}
