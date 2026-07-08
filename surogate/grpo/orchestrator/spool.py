import json
from dataclasses import dataclass, field
from pathlib import Path

import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class RestoredGroup:
    """An in-flight group reconstructed from the spool after a restart."""

    example: dict
    ckpt_step: int
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    failed_rollouts: int = 0


class InflightSpool:
    """Append-only event log of in-flight rollout groups, so a restart can resume
    partially generated groups instead of regenerating them from scratch.

    Events (one JSON object per line):
      {"event": "group_start", "gid": int, "ckpt_step": int, "example": {...}}
      {"event": "rollout", "gid": int, "rollout": {...}}
      {"event": "group_done", "gid": int}
      {"event": "group_dropped", "gid": int}

    A group is live iff started and neither done nor dropped. The scheduler no longer
    writes `group_done` at buffer-accept (2026-07-05: accepted rollouts live only in the
    local batch list until the bin is written — a mid-step kill must resurrect them;
    re-accepting is idempotent). Banked groups are erased by the step-boundary compaction
    instead, once the batch is safely on disk. The event stays honored by the loader.

    Durability: lines are flushed (not fsynced) — a process kill loses nothing
    (page cache survives), a power loss can lose the last-seconds tail. All write
    errors fail open: the spool disables itself and generation continues.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file = None
        self._disabled = False

    def _write(self, event: dict) -> None:
        if self._disabled:
            return
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "a")
            self._file.write(json.dumps(event, default=make_serializable) + "\n")
            self._file.flush()
        except Exception as e:
            self._disabled = True
            logger.warning(f"In-flight spool disabled after write error: {e}")

    def group_start(self, gid: int, ckpt_step: int, example: dict) -> None:
        self._write({"event": "group_start", "gid": gid, "ckpt_step": ckpt_step, "example": example})

    def rollout(self, gid: int, rollout: vf.RolloutOutput) -> None:
        self._write({"event": "rollout", "gid": gid, "rollout": rollout})

    def group_done(self, gid: int) -> None:
        self._write({"event": "group_done", "gid": gid})

    def group_dropped(self, gid: int) -> None:
        self._write({"event": "group_dropped", "gid": gid})

    def load(self, current_step: int, max_off_policy_steps: int) -> list[RestoredGroup]:
        """Replays the event log and returns live groups within the off-policy window.

        Tolerates a torn trailing line (crash mid-append). Groups whose generation
        weights are more than `max_off_policy_steps` behind `current_step` are
        discarded — the same staleness rule the scheduler enforces in memory.
        """
        if not self.path.exists():
            return []
        groups: dict[int, RestoredGroup] = {}
        terminal: set[int] = set()
        malformed = 0
        try:
            with open(self.path) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    gid = event.get("gid")
                    kind = event.get("event")
                    if kind == "group_start":
                        groups[gid] = RestoredGroup(example=event["example"], ckpt_step=event.get("ckpt_step", 0))
                    elif kind == "rollout" and gid in groups:
                        groups[gid].completed_rollouts.append(event["rollout"])
                    elif kind in ("group_done", "group_dropped"):
                        terminal.add(gid)
        except Exception as e:
            logger.warning(f"In-flight spool unreadable, starting clean: {e}")
            return []
        if malformed > 1:
            # one torn trailing line is expected after a crash; more means corruption
            logger.warning(f"In-flight spool had {malformed} malformed line(s)")

        live, stale = [], 0
        for gid, group in groups.items():
            if gid in terminal:
                continue
            # too old = beyond the off-policy window; from the future = a stale spool left
            # by a later run polluting a fresh/rewound start. Both are off-policy garbage.
            if current_step - group.ckpt_step > max_off_policy_steps or group.ckpt_step > current_step:
                stale += 1
                continue
            live.append(group)
        if stale:
            logger.info(f"Discarded {stale} spooled group(s) outside the off-policy window")
        return live

    def compact(self, live_groups: dict[int, "object"], ckpt_step_by_gid: dict[int, int]) -> None:
        """Rewrites the spool to contain only the given live groups (atomic tmp+rename).

        Called at step boundaries; the current GroupStates are the source of truth,
        so the file is regenerated from them rather than filtered.
        """
        if self._disabled:
            return
        try:
            if self._file is not None:
                self._file.close()
                self._file = None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for gid, group in live_groups.items():
                    ckpt_step = ckpt_step_by_gid.get(gid, 0)
                    f.write(
                        json.dumps(
                            {"event": "group_start", "gid": gid, "ckpt_step": ckpt_step, "example": group.example},
                            default=make_serializable,
                        )
                        + "\n"
                    )
                    for rollout in group.completed_rollouts:
                        f.write(
                            json.dumps({"event": "rollout", "gid": gid, "rollout": rollout}, default=make_serializable)
                            + "\n"
                        )
            tmp.rename(self.path)
        except Exception as e:
            self._disabled = True
            logger.warning(f"In-flight spool disabled after compaction error: {e}")

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
