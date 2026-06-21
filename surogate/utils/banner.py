"""Startup banner for the Surogate CLI.

Renders a compact side-by-side banner: the bold ``SUROGATE`` wordmark (plus a
tagline and version) on the left, and the Surogate leaping-rabbit mascot on the
right, in brand gold. The rabbit is drawn with half-block glyphs (``▀▄█``) for
smooth, high-resolution edges. The wordmark uses a solid block alphabet. All art
is baked in as plain text, so there is no Pillow or asset dependency at runtime.

Color is truecolor gold by default and degrades gracefully:
  * ``NO_COLOR`` set, ``TERM=dumb``, or a non-tty stdout -> no ANSI codes
  * terminal narrower than the art       -> a compact one-liner
Only the local-master rank prints (respects ``LOCAL_RANK``).
"""

import os
import shutil
import sys

# Leaping rabbit mascot, half-block render of the brand favicon (28 wide).
_RABBIT = [
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
]

# Bold block "SUROGATE" wordmark (5 rows).
_WORDMARK = [
    "████  ██ ██ ████   ███   ███   ███  █████ ████ ",
    "██    ██ ██ ██ ██ ██ ██ ██    ██ ██   ██  ██   ",
    " ███  ██ ██ ████  ██ ██ ██ ██ █████   ██  ███  ",
    "   ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██   ██  ██   ",
    "████   ███  ██ ██  ███   ███  ██ ██   ██  ████ ",
]

_TAGLINE = "BF16/FP8/FP4 · Training · Fine-tuning · RL"
_GAP = "   "  # space between wordmark column and rabbit column

# Brand gold, top->bottom gradient (bright gold -> deep amber).
_GOLD_TOP = (255, 209, 92)
_GOLD_BOT = (228, 150, 28)

_RESET = "\033[0m"
_BOLD = "\033[1m"

_WORD_W = max(len(l) for l in _WORDMARK)
_RABBIT_W = max(len(l) for l in _RABBIT)
_LEFT_W = max(_WORD_W, len(_TAGLINE))
_ART_WIDTH = _LEFT_W + len(_GAP) + _RABBIT_W
_FALLBACK = "(\\(\\  SUROGATE"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:
        return default


def _gold(t: float) -> str:
    r = round(_GOLD_TOP[0] + (_GOLD_BOT[0] - _GOLD_TOP[0]) * t)
    g = round(_GOLD_TOP[1] + (_GOLD_BOT[1] - _GOLD_TOP[1]) * t)
    b = round(_GOLD_TOP[2] + (_GOLD_BOT[2] - _GOLD_TOP[2]) * t)
    return f"\033[38;2;{r};{g};{b}m"


def render(version: str = "") -> str:
    """Return the banner as a single multi-line string."""
    width = _term_width()
    color = _supports_color()
    ver = f"v{version}" if version else ""

    if width < _ART_WIDTH + 2:
        head = _FALLBACK + (f"  {ver}" if ver else "")
        pad = " " * max((width - len(head)) // 2, 0)
        if color:
            return f"\n{pad}{_BOLD}{_gold(0.0)}{head}{_RESET}\n"
        return f"\n{pad}{head}\n"

    # Left column: wordmark, blank, tagline, version.
    left = list(_WORDMARK) + ["", _TAGLINE]
    if ver:
        left.append(ver)
    left = [l.ljust(_LEFT_W) for l in left]
    # The rabbit is the last column on each line, so trailing-space padding here
    # would only be stripped again below — keep the raw art.
    rabbit = list(_RABBIT)

    rows = max(len(left), len(rabbit))

    def vpad(block, w):
        top = (rows - len(block)) // 2
        return [" " * w] * top + block + [" " * w] * (rows - len(block) - top)

    left = vpad(left, _LEFT_W)
    rabbit = vpad(rabbit, _RABBIT_W)

    margin = " " * ((width - _ART_WIDTH) // 2)
    out = [""]
    for i in range(rows):
        line = margin + left[i] + _GAP + rabbit[i]
        if color:
            line = f"{_BOLD}{_gold(i / max(rows - 1, 1))}{line.rstrip()}{_RESET}"
        else:
            line = line.rstrip()
        out.append(line)
    out.append("")
    return "\n".join(out) + "\n"


def print_banner(version: str = "") -> None:
    """Print the startup banner to stdout (local master rank only).

    Best-effort and decorative: any failure (e.g. a malformed ``LOCAL_RANK`` or
    a missing ``sys.stdout``) is swallowed so it can never break CLI startup.
    """
    try:
        if int(os.environ.get("LOCAL_RANK", -1) or -1) not in (-1, 0):
            return
        sys.stdout.write(render(version))
        sys.stdout.flush()
    except Exception:
        pass


if __name__ == "__main__":
    print_banner("0.4.2")
