"""Per-step cost-ESCALATION LADDER probe on CODING (the product domain). The product is a 3-rung
cost ladder GLM($4) -> Gemini($12) -> Opus($25): cheap model drives, escalate only on failure.

Per SWE-smith task we measure, in containers where /testbed is the shared, mutable repo state:
  Ladder container (one shared state, escalate in place on tests-red):
    1. GLM attempts the fix (OpenCode). grade -> r_glm  = always-GLM (rung 1).
    2. if GLM fails: Gemini CONTINUES in the SAME container (inherits GLM's edits) -> r_gem
       = esc-GLM->Gemini.
    3. if Gemini also fails: Opus CONTINUES in the SAME container -> r_opus_ladder
       = esc-GLM->Gemini->Opus (full ladder).
  Solo containers (constants for comparison):
    always-Gemini (fresh), always-Opus (fresh).

Key product distribution: how often GLM suffices (served at $4), how often Gemini rescues the rest
($12), how rarely Opus is actually needed ($25). Cost not tracked numerically (per request); the
escalation RATE per rung is the cost story. OpenCode drives OpenRouter slugs directly.

HARNESS (see memory swesmith-harness-gotchas): MUST `git checkout <instance_id>` (bug is per-branch;
base is clean) and run oc inside the `testbed` conda env (login shell), else all-zero.
"""
from __future__ import annotations

import asyncio
import os
import subprocess

OC = os.path.expanduser("~/.opencode/bin/opencode")
KEY = os.environ["OPENROUTER_API_KEY"]
N = int(os.getenv("NTASKS", "20"))
TIMEOUT = int(os.getenv("OC_TIMEOUT", "600"))
CONC = int(os.getenv("CONC", "3"))
GLM, GEMINI, OPUS = "z-ai/glm-5.2", "google/gemini-3.1-pro-preview", "anthropic/claude-opus-4.8"
CONDA = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"


def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)


def oc_run(cid, slug, msg):
    """Run OpenCode with a worker at /testbed inside conda `testbed`. Returns 'ok'|'timeout'."""
    inner = f"{CONDA} && exec /usr/local/bin/oc run -m openrouter/{slug} --dangerously-skip-permissions \"$1\""
    try:
        subprocess.run(["docker", "exec", "-e", f"OPENROUTER_API_KEY={KEY}", "-e", "HOME=/root",
                        "-w", "/testbed", cid, "bash", "-lc", inner, "_", msg],
                       capture_output=True, text=True, timeout=TIMEOUT)
        return "ok"
    except subprocess.TimeoutExpired:
        return "timeout"


def diff_of(cid):
    sh("docker", "exec", cid, "bash", "-c", "cd /testbed && git add -A")
    return sh("docker", "exec", cid, "bash", "-c", "cd /testbed && git diff --cached").stdout


def start(img, iid):
    """Start a container and CHECK OUT THE BUGGY BRANCH (instance_id). Returns cid or ''."""
    cid = sh("docker", "run", "-d", "--rm", "-v", f"{OC}:/usr/local/bin/oc:ro", img, "sleep", "9000").stdout.strip()
    if not cid:
        return ""
    co = sh("docker", "exec", cid, "bash", "-c", f"cd /testbed && git checkout {iid} 2>&1")
    blob = (co.stderr + co.stdout).lower()
    if "error" in blob and "set up to track" not in blob:
        sh("docker", "rm", "-f", cid)
        return ""
    return cid


def rm(cid):
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def main():
    from director.agentic.runners import load_swesmith_tasks
    from director.agentic.swebench_mini import grade_swesmith

    tasks = load_swesmith_tasks(N)
    print(f"per-step LADDER probe: {len(tasks)} SWE-smith tasks  GLM->Gemini->Opus\n"
          f"  conc={CONC} timeout={TIMEOUT}s  [checkout instance branch + conda testbed]", flush=True)
    sem = asyncio.Semaphore(CONC)
    res = {k: [] for k in ["always-glm", "always-gemini", "always-opus", "esc-glm->gem", "esc-glm->gem->opus"]}
    # landing: where each task is first solved along the ladder
    land = {"glm": 0, "gemini": 0, "opus": 0, "unsolved": 0}
    lock = asyncio.Lock()

    async def grade(inst, diff):
        if not diff.strip():
            return 0.0
        return float(await asyncio.to_thread(grade_swesmith, inst, diff))

    def solo(img, iid, slug, inst, msg):
        cid = start(img, iid)
        if not cid:
            return 0.0
        try:
            oc_run(cid, slug, msg)
            d = diff_of(cid)
        finally:
            pass
        # grade outside (needs async); return diff via closure not possible -> grade here sync
        r = 0.0
        try:
            r = float(grade_swesmith(inst, d)) if d.strip() else 0.0
        finally:
            rm(cid)
        return r

    async def one(t):
        async with sem:
            inst = t["payload"]; img = inst["image_name"]; iid = inst["instance_id"]
            ps = inst["problem_statement"].strip()
            fix_msg = ps + "\n\nThe repo in /testbed has the bug above. Fix it by editing the SOURCE (do NOT modify tests). Run the project's tests to verify."
            cont_msg = ps + "\n\nThe repo in /testbed contains a previous engineer's partial attempt at this fix, but the failing tests still do not pass. Review the current code, find what's wrong or incomplete, and finish the fix. Do NOT modify tests. Run the tests to verify."

            # --- ladder container: GLM -> Gemini -> Opus in shared state ---
            cid = await asyncio.to_thread(start, img, iid)
            if not cid:
                async with lock:
                    print(f"  {iid[:40]:40} SKIP (container/checkout failed)", flush=True)
                return
            r_glm = r_gem = r_opus = 0.0; where = "unsolved"
            try:
                await asyncio.to_thread(oc_run, cid, GLM, fix_msg)
                r_glm = await grade(inst, await asyncio.to_thread(diff_of, cid))
                if r_glm >= 1.0:
                    r_gem = r_opus = 1.0; where = "glm"
                else:
                    await asyncio.to_thread(oc_run, cid, GEMINI, cont_msg)
                    r_gem = await grade(inst, await asyncio.to_thread(diff_of, cid))
                    if r_gem >= 1.0:
                        r_opus = 1.0; where = "gemini"
                    else:
                        await asyncio.to_thread(oc_run, cid, OPUS, cont_msg)
                        r_opus = await grade(inst, await asyncio.to_thread(diff_of, cid))
                        where = "opus" if r_opus >= 1.0 else "unsolved"
            finally:
                await asyncio.to_thread(rm, cid)

            # --- solo constants (fresh containers) ---
            r_gem_solo = await asyncio.to_thread(solo, img, iid, GEMINI, inst, fix_msg)
            r_opus_solo = await asyncio.to_thread(solo, img, iid, OPUS, inst, fix_msg)

            async with lock:
                res["always-glm"].append(r_glm)
                res["always-gemini"].append(r_gem_solo)
                res["always-opus"].append(r_opus_solo)
                res["esc-glm->gem"].append(r_gem)
                res["esc-glm->gem->opus"].append(r_opus)
                land[where] += 1
                print(f"  {iid[:40]:40} glm={r_glm:.0f} gem_solo={r_gem_solo:.0f} opus_solo={r_opus_solo:.0f} "
                      f"| ladder: g={r_glm:.0f}->gem={r_gem:.0f}->opus={r_opus:.0f}  [land={where}]", flush=True)

    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    loop.run_until_complete(asyncio.gather(*[one(t) for t in tasks]))

    n = len(res["always-glm"])
    print(f"\n=== PER-POLICY ({n} tasks) ===", flush=True)
    for k in ["always-glm", "always-gemini", "always-opus", "esc-glm->gem", "esc-glm->gem->opus"]:
        s = sum(1 for x in res[k] if x >= 1)
        print(f"  {k:20} success {s}/{n} = {s/max(n,1):.2f}", flush=True)
    print(f"\n=== LADDER LANDING (where each task is first solved) ===", flush=True)
    print(f"  GLM   suffices ($4) : {land['glm']}/{n} ({land['glm']/max(n,1):.0%})", flush=True)
    print(f"  Gemini rescues ($12): {land['gemini']}/{n} ({land['gemini']/max(n,1):.0%})", flush=True)
    print(f"  Opus   needed  ($25): {land['opus']}/{n} ({land['opus']/max(n,1):.0%})", flush=True)
    print(f"  unsolved           : {land['unsolved']}/{n} ({land['unsolved']/max(n,1):.0%})", flush=True)
    solved = land['glm'] + land['gemini'] + land['opus']
    print(f"\n  full-ladder solve rate: {solved}/{n} = {solved/max(n,1):.2f}  "
          f"(vs always-opus {sum(1 for x in res['always-opus'] if x>=1)}/{n})", flush=True)
    print("  verdict: most traffic should land at GLM($4); Gemini($12) should rescue most of the rest; "
          "Opus($25) rare. That distribution = the cost win at frontier solve rate.", flush=True)


if __name__ == "__main__":
    main()
