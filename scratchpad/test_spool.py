"""Unit tests for the in-flight rollout spool + scheduler restore semantics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/densemax/work/flavius/surogate")

from surogate.grpo.orchestrator.spool import InflightSpool

TMP = Path("/tmp/claude-1000/-home-densemax-work-flavius-surogate/1636be7a-c882-47c0-8ed5-6ece7392008f/scratchpad/spool_test")
TMP.mkdir(parents=True, exist_ok=True)

def rollout(i, reward=0.5):
    return {"example_id": 7, "task": "fugu_ultra_paper", "reward": reward,
            "trajectory": [{"tokens": [1, 2, 3 + i]}], "error": None}

def fresh(name):
    p = TMP / name
    if p.exists():
        p.unlink()
    return InflightSpool(p)

# 1. round-trip: live partial group survives; done and dropped groups excluded
sp = fresh("rt.jsonl")
sp.group_start(0, 64, {"example_id": 7, "task": "fugu_ultra_paper", "prompt": "q"})
for i in range(3):
    sp.rollout(0, rollout(i))
sp.group_start(1, 64, {"example_id": 8, "task": "fugu_ultra_paper", "prompt": "q2"})
sp.rollout(1, rollout(9))
sp.group_done(1)
sp.group_start(2, 64, {"example_id": 9, "task": "fugu_ultra_paper", "prompt": "q3"})
sp.group_dropped(2)
sp.close()
live = InflightSpool(sp.path).load(current_step=65, max_off_policy_steps=8)
assert len(live) == 1 and len(live[0].completed_rollouts) == 3, live
assert live[0].example["example_id"] == 7
assert live[0].completed_rollouts[2]["trajectory"][0]["tokens"] == [1, 2, 5]
print("1. round-trip + done/dropped exclusion OK")

# 2. torn trailing line (crash mid-append) tolerated
with open(sp.path, "a") as f:
    f.write('{"event": "rollout", "gid": 0, "rollout": {"trunc')
live = InflightSpool(sp.path).load(65, 8)
assert len(live) == 1 and len(live[0].completed_rollouts) == 3
print("2. torn tail OK")

# 3. staleness: group beyond max_off_policy_steps discarded
sp = fresh("stale.jsonl")
sp.group_start(0, 50, {"example_id": 1, "task": "t", "prompt": "q"})
sp.rollout(0, rollout(0))
sp.group_start(1, 60, {"example_id": 2, "task": "t", "prompt": "q"})
sp.rollout(1, rollout(1))
sp.close()
live = InflightSpool(sp.path).load(current_step=64, max_off_policy_steps=8)  # 64-50=14 > 8 stale; 64-60=4 ok
assert len(live) == 1 and live[0].example["example_id"] == 2, live
# future guard: a group from step 100 must not survive a resume at 64 (stale spool of a later run)
sp2 = fresh("future.jsonl")
sp2.group_start(0, 100, {"example_id": 3, "task": "t", "prompt": "q"})
sp2.rollout(0, rollout(0)); sp2.close()
assert InflightSpool(sp2.path).load(current_step=64, max_off_policy_steps=8) == []
print("3. staleness + future-step discard OK")

# 4. compaction: regenerates file from live GroupStates only, atomically
from surogate.grpo.orchestrator.scheduler import GroupState
sp = fresh("compact.jsonl")
sp.group_start(0, 64, {"example_id": 1, "task": "t", "prompt": "q"})
sp.group_done(0)
sp.group_start(1, 64, {"example_id": 2, "task": "t", "prompt": "q"})
sp.rollout(1, rollout(0))
groups = {1: GroupState(example={"example_id": 2, "task": "t", "prompt": "q"},
                        rollouts_to_schedule=63, completed_rollouts=[rollout(0)])}
sp.compact(groups, {1: 64})
lines = [json.loads(l) for l in open(sp.path)]
assert len(lines) == 2 and lines[0]["event"] == "group_start" and lines[1]["event"] == "rollout"
live = InflightSpool(sp.path).load(65, 8)
assert len(live) == 1 and len(live[0].completed_rollouts) == 1
print("4. compaction OK")

# 5. fail-open: unwritable path disables spool without raising
bad = InflightSpool(Path("/proc/nonexistent/spool.jsonl"))
bad.group_start(0, 1, {"a": 1})
bad.rollout(0, rollout(0))
assert bad._disabled
print("5. fail-open OK")

# 6. missing file → clean empty load
assert InflightSpool(TMP / "never_written.jsonl").load(10, 8) == []
print("6. missing-file load OK")

# 7. scheduler restore: rebuilds groups with fresh ids, top-up counts, complete groups intact
class _SchedStub:
    # borrow the real method; provide only what it touches
    restore_from_spool = __import__("surogate.grpo.orchestrator.scheduler", fromlist=["Scheduler"]).Scheduler.restore_from_spool
    def __init__(self, spool, rpe):
        import logging, types
        self.spool, self.rollouts_per_example, self.max_off_policy_steps = spool, rpe, 8
        self.next_group_id, self.groups, self._group_ckpt_step = 5, {}, {}
        self.logger = logging.getLogger("stub")
        self.buffer = types.SimpleNamespace(env_names=["t"])

sp = fresh("restore.jsonl")
sp.group_start(0, 64, {"example_id": 1, "task": "t", "prompt": "q"})
for i in range(3):
    sp.rollout(0, rollout(i))
sp.group_start(1, 64, {"example_id": 2, "task": "t", "prompt": "q2"})
for i in range(4):
    sp.rollout(1, rollout(i, reward=1.0))
sp.group_start(2, 64, {"example_id": 9, "task": "REMOVED_ENV", "prompt": "q3"})
sp.rollout(2, rollout(0))
sp.close()
stub = _SchedStub(InflightSpool(sp.path), rpe=4)
stub.restore_from_spool(current_step=65)
assert set(stub.groups) == {5, 6}, stub.groups
by_missing = sorted((g.rollouts_to_schedule, len(g.completed_rollouts)) for g in stub.groups.values())
assert by_missing == [(0, 4), (1, 3)], by_missing  # complete group drains at next generate_batch; partial tops up 1
assert stub._group_ckpt_step == {5: 64, 6: 64}
print("7. scheduler restore OK")

print("\nALL SPOOL TESTS PASSED")

# 8. keep_last/keep_interval cleanup
import shutil
from surogate.grpo.orchestrator.ckpt import CheckpointManager
from surogate.core.config.grpo_orch_config import GRPOCheckpointConfig
from surogate.utils.dict import DictDefault
root = TMP / "ckpt_root"
shutil.rmtree(root, ignore_errors=True)
ck = root / "checkpoints"
for s in range(1, 13):
    (ck / f"step_{s}" / "orchestrator").mkdir(parents=True)
(ck / "inflight_spool.jsonl").write_text('{"event":"group_done","gid":0}\n')
mgr = CheckpointManager(root, GRPOCheckpointConfig(DictDefault({"keep_last": 3, "keep_interval": 5})))
mgr._cleanup_old_ckpts()
left = sorted(int(p.name.split("_")[1]) for p in ck.glob("step_*"))
assert left == [5, 10, 11, 12], left  # newest 3 + multiples of 5
assert (ck / "inflight_spool.jsonl").exists()
print("8. keep_last/keep_interval cleanup OK (spool file untouched)")
print("\nALL EXTENDED TESTS PASSED")
