// External links + a best-effort URL opener. Opening a browser from a TUI isn't
// guaranteed (ssh, headless), so callers should also show the URL so it stays
// accessible/copyable.

import { spawn } from "node:child_process";

export const SUROGATE_REPO = "https://github.com/invergent-ai/surogate";

/** A prefilled "add this model" enhancement issue on the surogate repo. */
export function modelRequestUrl(modelId: string, arch?: string | null): string {
  const title = `Add model support: ${modelId}`;
  const body = [
    "### Model support request",
    "",
    `Please add training support for **${modelId}** to surogate.`,
    "",
    `- Model: https://huggingface.co/${modelId}`,
    `- Architecture: ${arch || "(unknown)"}`,
    "",
    "_Requested via the jackalope dashboard._",
  ].join("\n");
  const q = new URLSearchParams({ labels: "enhancement", title, body });
  return `${SUROGATE_REPO}/issues/new?${q.toString()}`;
}

/** Copy text to the terminal clipboard via OSC 52 (works in most modern
 *  terminals; tmux needs `set -g set-clipboard on`). Best-effort. */
export function copyToClipboard(text: string): void {
  try {
    const b64 = Buffer.from(text, "utf8").toString("base64");
    process.stdout.write(`\x1b]52;c;${b64}\x07`);
  } catch {
    /* ignore */
  }
}

/** Open a URL in the user's default browser (detached, non-blocking). Returns
 *  false only if spawn throws synchronously; async failures are swallowed (the
 *  caller shows the URL regardless). */
export function openUrl(url: string): boolean {
  const platform = process.platform;
  const cmd = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
  const args = platform === "win32" ? ["/c", "start", "", url] : [url];
  try {
    const child = spawn(cmd, args, { stdio: "ignore", detached: true });
    child.on("error", () => {});
    child.unref();
    return true;
  } catch {
    return false;
  }
}
