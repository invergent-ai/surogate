"""PROOF: OpenCode and Claude Code each drive a REAL agentic fix on SWE-smith bugs, end to end.

2 contested tasks x 2 harness arms = 4 isolated containers, run concurrently:
  arm A: OpenCode + GLM-5.2 (openrouter)          -- study-proven mechanics, verbatim
  arm B: Claude Code + Opus-4.8 via YUNWU native  -- claude ELF mounted into container,
         ANTHROPIC_BASE_URL=https://yunwu.ai (Anthropic-native /v1/messages, probe-verified)

Each run: start container -> checkout buggy branch -> agent edits source + runs tests inside
conda testbed -> git diff -> grade with SWE-smith's OWN harness (fresh grading container).
Success criterion of the DEMO = the full cycle executes and produces a graded diff;
resolved=1 is a bonus (single attempts are stochastic; both tasks are contested by design).

Meters: agent wall-time, grade wall-time, diff size, claude cost/turns (from --output-format json).
No training-run interaction whatsoever (CPU containers + provider APIs only).
"""
import asyncio, json, os, subprocess, sys, time

REPO = "/home/densemax/work/flavius/surogate"
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "director"))

OC = os.path.expanduser("~/.opencode/bin/opencode")
CLAUDE = "/home/densemax/.local/share/claude/versions/2.1.198"   # resolved self-contained ELF
ORKEY = os.environ["OPENROUTER_API_KEY"]
YWKEY = os.environ["YUNWU_API_KEY"]
GLM = "z-ai/glm-5.2"
OPUS = "claude-opus-4-8"
CONDA = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed"
OC_TIMEOUT = 600
CC_TIMEOUT = 900

def local_image_repos():
    p = subprocess.run(["docker", "images", "--format", "{{.Repository}}"],
                       capture_output=True, text=True)
    return {l.strip() for l in p.stdout.splitlines() if l.strip()}

def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)

def start(img, iid):
    cid = sh("docker", "run", "-d", "--rm",
             "-v", f"{OC}:/usr/local/bin/oc:ro",
             "-v", f"{CLAUDE}:/usr/local/bin/claude:ro",
             img, "sleep", "9000").stdout.strip()
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

def diff_of(cid):
    sh("docker", "exec", cid, "bash", "-c", "cd /testbed && git add -A")
    return sh("docker", "exec", cid, "bash", "-c", "cd /testbed && git diff --cached").stdout

def run_opencode(cid, msg):
    inner = f"{CONDA} && exec /usr/local/bin/oc run -m openrouter/{GLM} --dangerously-skip-permissions \"$1\""
    try:
        subprocess.run(["docker", "exec", "-e", f"OPENROUTER_API_KEY={ORKEY}", "-e", "HOME=/root",
                        "-w", "/testbed", cid, "bash", "-lc", inner, "_", msg],
                       capture_output=True, text=True, timeout=OC_TIMEOUT)
        return {"status": "ok"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}

def run_claude_code(cid, msg):
    # fresh HOME in container: mark onboarding done so headless -p starts clean
    sh("docker", "exec", "-e", "HOME=/root", cid, "bash", "-c",
       "echo '{\"hasCompletedOnboarding\":true}' > /root/.claude.json")
    inner = (f"{CONDA} && exec /usr/local/bin/claude -p \"$1\" --model {OPUS} "
             f"--dangerously-skip-permissions --output-format json --max-turns 30")
    try:
        p = subprocess.run(["docker", "exec",
                            "-e", "ANTHROPIC_BASE_URL=https://yunwu.ai",
                            "-e", f"ANTHROPIC_API_KEY={YWKEY}",
                            "-e", "HOME=/root", "-e", "IS_SANDBOX=1",
                            "-e", "DISABLE_TELEMETRY=1", "-e", "DISABLE_ERROR_REPORTING=1",
                            "-w", "/testbed", cid, "bash", "-lc", inner, "_", msg],
                           capture_output=True, text=True, timeout=CC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    out = {"status": "ok", "rc": p.returncode, "stderr_tail": p.stderr[-300:]}
    try:
        j = json.loads(p.stdout.strip().splitlines()[-1])
        out.update({"turns": j.get("num_turns"), "cost_usd": j.get("total_cost_usd"),
                    "duration_ms": j.get("duration_ms"), "is_error": j.get("is_error"),
                    "result_tail": (j.get("result") or "")[-220:]})
    except Exception:
        out["stdout_tail"] = p.stdout[-300:]
    return out

async def one(task, arm):
    from director.agentic.swebench_mini import grade_swesmith
    inst = task["payload"]; iid = inst["instance_id"]; img = inst["image_name"]
    tag = f"{iid.split('.')[-1]}/{arm}"
    msg = (inst["problem_statement"].strip()
           + "\n\nThe repo in /testbed has the bug above. Fix it by editing the SOURCE "
             "(do NOT modify tests). Run the project's tests to verify.")
    t0 = time.time()
    cid = await asyncio.to_thread(start, img, iid)
    if not cid:
        print(f"[{tag}] container/checkout FAILED", flush=True); return
    t_start = time.time() - t0
    print(f"[{tag}] container up ({t_start:.0f}s), agent running...", flush=True)
    try:
        if arm == "claude_code":
            # preflight: does the mounted ELF execute in this image?
            v = sh("docker", "exec", "-e", "HOME=/root", cid, "bash", "-c", "claude --version 2>&1")
            if v.returncode != 0:
                print(f"[{tag}] claude binary preflight FAILED: {v.stdout[-160:]} {v.stderr[-160:]}", flush=True)
                return
            t1 = time.time()
            meta = await asyncio.to_thread(run_claude_code, cid, msg)
        else:
            t1 = time.time()
            meta = await asyncio.to_thread(run_opencode, cid, msg)
        t_agent = time.time() - t1
        diff = await asyncio.to_thread(diff_of, cid)
    finally:
        await asyncio.to_thread(rm, cid)
    nfiles = diff.count("diff --git")
    plus = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    minus = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    open(f"{SCRATCH}/proof_{tag.replace('/','_')}.diff", "w").write(diff)
    print(f"[{tag}] agent done in {t_agent:.0f}s ({meta}); diff: {nfiles} files +{plus}/-{minus}; grading...", flush=True)
    t2 = time.time()
    resolved = 0.0
    if diff.strip():
        resolved = float(await asyncio.to_thread(grade_swesmith, inst, diff, f"proof_{arm}"))
    t_grade = time.time() - t2
    print(f"[{tag}] RESOLVED={resolved:.0f}  (agent {t_agent:.0f}s, grade {t_grade:.0f}s)", flush=True)
    return {"task": iid, "arm": arm, "resolved": resolved, "agent_s": round(t_agent),
            "grade_s": round(t_grade), "files": nfiles, "plus": plus, "minus": minus,
            "meta": {k: v for k, v in (meta or {}).items() if k in
                     ("status", "turns", "cost_usd", "is_error", "rc")}}

async def main():
    from director.agentic.runners import load_swesmith_tasks
    print("streaming SWE-smith; selecting 2 tasks whose repo image exists locally...", flush=True)
    rows = load_swesmith_tasks(800)
    have = local_image_repos()
    seen_repos, tasks = set(), []
    for t in rows:
        img = t["payload"]["image_name"].split(":")[0].removeprefix("docker.io/")
        repo = t["item_id"].split(".")[0]
        if img in have and repo not in seen_repos and "pandas" not in repo:  # pandas grading is slow
            seen_repos.add(repo); tasks.append(t)
        if len(tasks) == 2:
            break
    print(f"selected: {[t['item_id'] for t in tasks]}", flush=True)
    assert len(tasks) == 2, f"no locally-imaged tasks in first 800 rows; have={len(have)} images"
    combos = [(t, arm) for t in tasks for arm in ("opencode_glm", "claude_code")]
    results = [r for r in await asyncio.gather(*[one(t, a) for t, a in combos]) if r]
    print("\n================ PROOF RESULTS ================", flush=True)
    for r in results:
        print(f"{r['task'][:52]:52} {r['arm']:13} resolved={r['resolved']:.0f} "
              f"agent={r['agent_s']}s grade={r['grade_s']}s diff={r['files']}f +{r['plus']}/-{r['minus']} {r['meta']}", flush=True)
    json.dump(results, open(f"{SCRATCH}/proof_agentic_results.json", "w"), indent=1)
    print("\nPROOF DONE (full cycle = container -> agent edits+tests -> diff -> official grade)", flush=True)

asyncio.run(main())
