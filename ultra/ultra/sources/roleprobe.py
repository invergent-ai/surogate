"""Derived role-probe datasets (ultra-data2 §7).

Not a dataset loader and NOT GRPO training data — probes are generated from a base task plus
worker outputs to measure which workers are useful as planners, critics, verifiers, and
synthesizers (repair_rate, damage_rate, selection accuracy). policy = diagnostic_only.

Each probe reuses the base task's grader and harness, so the repaired / selected / synthesized
answer is scored exactly as the original task would be.
"""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import SourceManifest, TaskSpec
from .hf import make_taskspec


def _user(task: TaskSpec) -> str:
    for m in reversed(task.input.messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


class RoleProbeAdapter:
    source_name = "role_probe"
    capability = "role_probe"
    policy = SOURCE_POLICY["role_probe"]
    version = "v1"

    def critic_probe(self, base: TaskSpec, draft_text: str, draft_correct: bool) -> TaskSpec:
        prompt = (
            "Review the candidate solution to the task below. Identify any concrete errors, "
            "explain them, and return the smallest corrected solution.\n\n"
            f"TASK:\n{_user(base)}\n\nCANDIDATE SOLUTION:\n{draft_text}"
        )
        return self._probe("critic", base, prompt, ["correct" if draft_correct else "incorrect"])

    def verifier_probe(self, base: TaskSpec, candidates: list[str]) -> TaskSpec:
        listing = "\n\n".join(f"[{j}] {c}" for j, c in enumerate(candidates))
        prompt = (
            "One of the candidate answers below is correct. Choose the correct one and explain "
            f"why.\n\nTASK:\n{_user(base)}\n\nCANDIDATES:\n{listing}"
        )
        return self._probe("verifier", base, prompt, [f"n{len(candidates)}"])

    def synthesizer_probe(self, base: TaskSpec, attempts: list[str]) -> TaskSpec:
        listing = "\n\n".join(f"[ATTEMPT {j}]\n{a}" for j, a in enumerate(attempts))
        prompt = (
            "Combine the partial attempts below into one correct final solution. Preserve correct "
            f"parts and resolve disagreements.\n\nTASK:\n{_user(base)}\n\nATTEMPTS:\n{listing}"
        )
        return self._probe("synthesizer", base, prompt, [f"n{len(attempts)}"])

    def _probe(self, role: str, base: TaskSpec, prompt: str, extra_tags: list[str]) -> TaskSpec:
        return make_taskspec(
            task_id=f"role_probe__{role}__{base.task_id}",
            capability="role_probe",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness=base.environment.harness,
            grader_type=base.grader.type,
            expected_answer=base.grader.expected_answer,
            messages=[{"role": "user", "content": prompt}],
            split="diagnostic",
            group_id=f"role_probe_{role}",
            contamination_group=f"roleprobe::{base.task_id}",
            domain="role_probe",
            tags=["probe", role, *extra_tags],
        )

    def manifest(self) -> SourceManifest:
        return SourceManifest(
            source_name=self.source_name,
            source_type="derived",
            version=self.version,
            allowed_uses=["diagnostic"],
            forbidden_uses=["grpo_train"],
            known_issues=["derived from worker outputs; diagnostic only (pool selection)"],
        )
