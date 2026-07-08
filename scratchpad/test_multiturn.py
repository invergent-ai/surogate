"""Offline unit test of the multi-turn control flow — NO network, NO live-run touch.
Fakes _execute_last_workflow to script turn outcomes and asserts: feedback-vs-terminate,
shared-memory accumulation, terminal reward = last executed grade. Plus a real parse-path
check and a tokenizer seq-len risk-retirement for the 2-turn+code-artifact prompt."""
import asyncio, importlib.util, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")

spec = importlib.util.spec_from_file_location(
    "mt", "/home/densemax/work/flavius/surogate/scratchpad/fugu_ultra_multiturn.py")
mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)

Env = mt.FuguUltraMultiTurnEnv


def new_env():
    e = Env.__new__(Env)      # skip vf.__init__ heavy machinery for a pure logic test
    e.rt = None
    return e


async def run():
    # --- 1. failing turn 0 -> revise instruction + shared memory grows, NO terminate ---
    e = new_env()
    scripted = [{"reward": 0.5, "success": False, "parse_valid": True, "outcome_text": "def f(): return 1  # wrong"}]
    async def fake_exec(state, _s=scripted): return _s[len(state["turn_records"])]
    e._execute_last_workflow = fake_exec
    st = await e.setup_state({"info": {"task_id": "t", "lane": "single_turn"},
                              "trajectory": [{"completion": "workflow_1"}]})
    resp = await e.env_response(None, st)
    assert "final_env_response" not in st, "should NOT terminate on failure"
    assert "INCORRECT" in resp[0]["content"] and "wrong" in resp[0]["content"], "feedback carries outcome"
    assert st["shared_memory"] == ["def f(): return 1  # wrong"], st["shared_memory"]
    assert len(st["turn_records"]) == 1
    print("1. failing turn0 -> revise + shared-memory + no-terminate  OK")

    # --- 2. succeeding turn 0 -> early terminate ---
    e2 = new_env()
    async def fake_ok(state): return {"reward": 1.0, "success": True, "parse_valid": True, "outcome_text": "42"}
    e2._execute_last_workflow = fake_ok
    st2 = await e2.setup_state({"info": {"task_id": "t"}, "trajectory": [{"completion": "wf"}]})
    await e2.env_response(None, st2)
    assert st2.get("final_env_response"), "should terminate early on success"
    print("2. succeeding turn0 -> early terminate  OK")

    # --- 3. reward = last executed grade (turn 1 repair succeeds after turn 0 failed) ---
    e3 = new_env()
    seq = [{"reward": 0.5, "success": False, "parse_valid": True, "outcome_text": "bug"},
           {"reward": 1.0, "success": True, "parse_valid": True, "outcome_text": "fixed"}]
    async def fake_seq(state, _s=seq): return _s[len(state["turn_records"])]
    e3._execute_last_workflow = fake_seq
    st3 = await e3.setup_state({"info": {"task_id": "t"},
                                "trajectory": [{"completion": "wf1"}]})
    await e3.env_response(None, st3)            # executes turn0 (fail) -> records=1
    st3["trajectory"].append({"completion": "wf2"})   # turn1 generated (repair)
    r = await e3.reward(st3)                    # executes turn1 -> reward=1.0
    assert r == 1.0 and len(st3["turn_records"]) == 2, (r, st3["turn_records"])
    print("3. reward = last-executed grade (0.5 fail -> 1.0 repair)  OK")

    # --- 4. real parse path (no network): a genuine conductor completion parses ---
    raw = ('model_id = [2, 0]\n'
           'subtasks = ["Find the bug in the attempt and fix it.", "Verify and format the final answer."]\n'
           'access_list = [[], ["all"]]')
    wf = mt.parse_workflow(mt._extract_workflow_payload(raw))
    assert len(wf.steps) == 2 and wf.steps[0].worker_id == 2, wf
    print("4. real parse path OK (2-step repair workflow)")

    # --- 5. seq-len risk: 2-turn prompt with a code artifact must fit 8192 ---
    from transformers import AutoTokenizer
    RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    tok = AutoTokenizer.from_pretrained(RAW)
    fat_artifact = "def solve():\n" + "    x = compute_something(y)  # plausible line\n" * 120  # ~large wrong solution
    two_turn = ("SYSTEM+FEWSHOTS " * 400 + "\nUSER QUESTION: " + "problem statement " * 200
                + "\nworkflow_1 " * 60 + mt.REVISE_INSTRUCTION.format(outcome=fat_artifact[:1500])
                + "\nworkflow_2 " * 60)
    n = len(tok.encode(two_turn))
    print(f"5. worst-case 2-turn prompt tokens = {n}  ({'FITS 8192' if n < 8192 else 'RISK: exceeds 8192 — cap artifact tighter'})")

    print("\nALL MULTI-TURN LOGIC TESTS PASSED")

asyncio.run(run())
