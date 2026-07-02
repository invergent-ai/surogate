"""Graders: map a worker's raw text output + ground truth to a reward in [0, 1].

Each grader has signature ``(output: str, solution: Any) -> float``. The registry
lets tasks reference a grader by name (and keeps datasets decoupled from grading).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

Grader = Callable[[str, Any], float]


# ---------------------------------------------------------------------------
# extraction helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_SIMPLE_FRAC_RE = re.compile(r"-?\d+\s*/\s*-?\d+")
_LATEX_FRAC_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")


def _last_number(text: str) -> str | None:
    # Prefer a GSM8K-style "#### <answer>" marker if present.
    if "####" in text:
        tail = text.rsplit("####", 1)[1]
        m = _NUM_RE.findall(tail)
        if m:
            return m[0].replace(",", "")
    m = _NUM_RE.findall(text)
    return m[-1].replace(",", "") if m else None


def _last_fraction(text: str) -> str | None:
    latex = _LATEX_FRAC_RE.findall(text)
    if latex:
        num, den = latex[-1]
        return f"\\frac{{{num}}}{{{den}}}"
    simple = _SIMPLE_FRAC_RE.findall(text)
    return simple[-1].replace(" ", "") if simple else None


def extract_boxed(text: str) -> str | None:
    """Extract the contents of the last ``\\boxed{...}`` (brace-balanced)."""
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    i = idx + len(r"\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_code(text: str) -> str:
    """Pull a fenced code block if present, else return the text as-is."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def extract_choice(text: str) -> str | None:
    """Extract a multiple-choice letter A-J (prefers 'answer is X' phrasing)."""
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-J])\b", text)
    return m[-1].upper() if m else None


# ---------------------------------------------------------------------------
# graders
# ---------------------------------------------------------------------------


def gsm8k_exact(output: str, solution: Any) -> float:
    pred = _last_number(output)
    gold = _last_number(str(solution))
    if pred is None or gold is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gold)) < 1e-6 else 0.0
    except ValueError:
        return 1.0 if pred == gold else 0.0


def _math_verify_equal(output: str, gold: str, timeout: float = 5.0) -> bool:
    """OFFICIAL grader: HuggingFace math_verify (the Qwen2.5-Math/SymPy path used by MATH-500,
    NuminaMath, and Omni-MATH rule-based eval). The gold is wrapped in ``$...$`` so math_verify's
    LaTeX extractor parses it (bare strings extract empty); ``verify(gold, pred)`` is
    order-sensitive (gold first). Runs in a daemon thread with a hard timeout so a pathological
    sympy parse can never freeze the grader -- using math_verify's OWN ``timeout_seconds`` (a
    nested daemon thread breaks math_verify, so we rely on its built-in timeout)."""
    try:
        from math_verify import parse as mv_parse, verify as mv_verify
        return bool(mv_verify(mv_parse(f"${gold}$"), mv_parse(output)))
    except Exception:
        return False


def math_equal(output: str, solution: Any) -> float:
    gold = str(solution)
    # Primary: official math_verify (handles tuples/intervals/sets/matrices/sci-notation/percent/...).
    if _math_verify_equal(output, gold):
        return 1.0
    # Fallback: hand-rolled normalizer -- covers plain text answers (e.g. "Evelyn") that
    # math_verify (a math parser) does not, plus a sympy equality check.
    pred = extract_boxed(output) or _last_fraction(output) or (output.strip().splitlines()[-1] if output.strip() else "")
    gold_x = extract_boxed(gold) or gold
    pred_n, gold_n = _normalize_math(pred), _normalize_math(gold_x)
    if pred_n == gold_n:
        return 1.0
    return 1.0 if _sympy_equal_timed(pred_n, gold_n, timeout=5.0) else 0.0


def _sympy_equal_timed(pred_n: str, gold_n: str, timeout: float = 5.0) -> bool:
    """sympy.simplify can hang indefinitely on pathological inputs (and a long max-reasoning
    answer is a prime trigger). Run it in a daemon thread with a hard timeout so it can NEVER
    block the caller — if it doesn't finish in time, treat the answer as not-verified (False).
    Without this, a synchronous hang inside an async grader freezes the whole event loop."""
    result = [False]

    def work():
        try:
            import sympy
            result[0] = bool(sympy.simplify(sympy.sympify(pred_n) - sympy.sympify(gold_n)) == 0)
        except Exception:
            result[0] = False

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    return result[0]


def _normalize_math(s: str) -> str:
    s = s.strip().strip("`").strip("$").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)  # \text{Evelyn} -> Evelyn
    s = re.sub(r"\^\{?\\circ\}?", "", s)          # 90^\circ / 90^{\circ} -> 90 (strip degree unit)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = s.rstrip(".")
    return s


def mc_letter(output: str, solution: Any) -> float:
    boxed = extract_boxed(output)
    pred = (boxed.strip()[:1].upper() if boxed else None) or extract_choice(output)
    gold = (extract_boxed(str(solution)) or str(solution)).strip().upper()[:1]
    return 1.0 if pred == gold else 0.0


def _run_capped(cmd, *, stdin_data=None, timeout=10):
    """Run ``cmd`` under a hard wall-clock cap. The child gets its own session, so on
    timeout we SIGKILL the whole process group — otherwise grandchildren spawned by
    untrusted model code keep the stdout pipe open and ``communicate`` hangs forever
    despite the timeout. stdin is closed (DEVNULL) unless input is given, so ``input()``
    gets EOF instead of blocking. Returns ``(returncode|None, stdout)``; None == failed/timed out."""
    import os
    import signal
    import subprocess

    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except Exception:
        return None, ""
    try:
        out, _ = p.communicate(input=stdin_data, timeout=timeout)
        return p.returncode, out
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return None, ""


def code_exec(output: str, solution: Any) -> float:
    """Run candidate code + a test snippet in a subprocess; reward 1.0 on exit 0.

    ``solution`` is a dict ``{"test": <python source>, "entry_point": <name>}`` in the
    HumanEval style (the test calls ``check(<entry_point>)``).
    """
    import sys
    import tempfile
    import textwrap

    if not isinstance(solution, dict):
        return 0.0
    code = extract_code(output)
    test = solution.get("test", "")
    entry = solution.get("entry_point", "")
    program = f"{code}\n\n{test}\n\ncheck({entry})\n" if entry else f"{code}\n\n{test}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(textwrap.dedent(program))
        f.flush()
        rc, _ = _run_capped([sys.executable, f.name], timeout=solution.get("timeout", 10))
        return 1.0 if rc == 0 else 0.0


def code_exec_stdio(output: str, solution: Any) -> float:
    """Run candidate as a stdin->stdout program; reward 1.0 iff all tests match.

    ``solution`` = ``{"tests": [{"input": str, "output": str}, ...], "timeout": int}``
    (competitive-programming style, e.g. code_contests).
    """
    import sys
    import tempfile

    if not isinstance(solution, dict):
        return 0.0
    tests = solution.get("tests", [])
    if not tests:
        return 0.0
    code = extract_code(output)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()
        for t in tests:
            rc, out = _run_capped(
                [sys.executable, f.name],
                stdin_data=t.get("input", ""),
                timeout=solution.get("timeout", 10),
            )
            if rc != 0:
                return 0.0
            if out.strip() != str(t.get("output", "")).strip():
                return 0.0
    return 1.0


def parse_grid(text: str) -> list[list[int]] | None:
    """Extract a 2-D integer grid from model output.

    Accepts a JSON-style ``[[...],[...]]`` literal (anywhere in the text, last match
    wins) or, failing that, the trailing block of whitespace-separated digit rows.
    """
    import ast

    matches = re.findall(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    for m in reversed(matches):
        try:
            g = ast.literal_eval(m)
            if isinstance(g, list) and g and all(isinstance(r, list) for r in g):
                return [[int(x) for x in r] for r in g]
        except (ValueError, SyntaxError):
            continue
    # fallback: trailing run of digit-only rows
    rows: list[list[int]] = []
    for line in reversed(text.strip().splitlines()):
        toks = line.replace(",", " ").split()
        if toks and all(t.lstrip("-").isdigit() for t in toks):
            rows.append([int(t) for t in toks])
        elif rows:
            break
    return list(reversed(rows)) if rows else None


def grid_exact(output: str, solution: Any) -> float:
    """Exact-match a predicted grid against the gold grid (ARC-AGI)."""
    gold = solution if isinstance(solution, list) else None
    if gold is None:
        return 0.0
    pred = parse_grid(output)
    return 1.0 if pred == [[int(x) for x in r] for r in gold] else 0.0


def contains(output: str, solution: Any) -> float:
    """Reward when the expected string appears in the worker output."""
    gold = str(solution).strip().lower()
    if not gold:
        return 0.0
    return 1.0 if gold in output.lower() else 0.0


def contains_all_absent(output: str, solution: Any) -> float:
    """Reward when all required substrings appear and forbidden stale values do not.

    ``solution`` = ``{"must_contain": [...], "must_not_contain": [...]}``.
    This is useful for long-context revision tasks where verbose answers that include
    obsolete values in the final answer should fail instead of receiving partial
    credit. If a single output line contains all required fields, forbidden checks
    are applied to that line so reviewer explanations do not create false negatives.
    """
    if not isinstance(solution, dict):
        return 0.0
    out = output.lower()
    required = [str(x).strip().lower() for x in solution.get("must_contain", []) if str(x).strip()]
    forbidden = [str(x).strip().lower() for x in solution.get("must_not_contain", []) if str(x).strip()]
    if not required:
        return 0.0
    lines = [line.strip().lower() for line in output.splitlines() if line.strip()]
    matching_lines = [line for line in lines if all(item in line for item in required)]
    judged = matching_lines[-1] if matching_lines else out
    if any(item not in judged for item in required):
        return 0.0
    if any(item in judged for item in forbidden):
        return 0.0
    return 1.0


REGISTRY: dict[str, Grader] = {
    "gsm8k_exact": gsm8k_exact,
    "math_equal": math_equal,
    "mc_letter": mc_letter,
    "code_exec": code_exec,
    "code_exec_stdio": code_exec_stdio,
    "grid_exact": grid_exact,
    "contains": contains,
    "contains_all_absent": contains_all_absent,
}


def get_grader(name: str) -> Grader:
    if name not in REGISTRY:
        raise KeyError(f"unknown grader {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]
