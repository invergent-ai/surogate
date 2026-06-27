"""Direct-QA source adapters: math + multiple-choice science/knowledge.

Ported from the router's verified HF loaders (same dataset IDs, fields, graders, and
system prompts). All route to the ``direct_qa`` harness; math uses ``math_equal``,
multiple-choice uses ``mc_letter``.
"""

from __future__ import annotations

import string

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import HFTaskAdapter, make_taskspec

_LETTERS = string.ascii_uppercase
_MATH_SYS = "Solve the problem. Put your final answer in \\boxed{}."
_MC_SYS = "Answer the question. Put the final answer letter in \\boxed{}."

# Bank/MC domain -> Ultra capability.
_DOMAIN_CAP = {
    "math": "math",
    "science": "science_knowledge",
    "general": "factual_qa",
    "code": "unit_code",
    "reasoning": "planning",
}


# --------------------------------------------------------------------------- math
class NuminaMathAdapter(HFTaskAdapter):
    """NuminaMath-1.5 — the large math TRAIN source (final-answer rows only)."""

    source_name = "numina_math"
    capability = "math"
    dataset_id = "AI-MO/NuminaMath-1.5"
    hf_split = "train"
    streaming = True
    policy = SOURCE_POLICY["numina_math"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        ans = (r.get("answer") or "").strip()
        if not ans or ans.lower() in {"proof", "notfound", "none"}:
            return None
        if str(r.get("problem_is_valid", "Yes")).lower() in {"no", "false"}:
            return None
        return make_taskspec(
            task_id=f"numina_math__{i}",
            capability="math",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="math_equal",
            expected_answer=ans,
            prompt=r["problem"],
            system=_MATH_SYS,
            group_id="numina_math",
            contamination_group=f"numina::{i}",
            domain="math",
            subdomain=r.get("problem_type"),
            tags=["math"],
            url_or_ref=self.dataset_id,
        )


# "MathAdapter" in ultra-data2 §5 is the NuminaMath train source.
MathAdapter = NuminaMathAdapter


class Math500Adapter(HFTaskAdapter):
    """MATH-500 — held out for final evaluation."""

    source_name = "math500"
    capability = "math"
    dataset_id = "HuggingFaceH4/MATH-500"
    hf_split = "test"
    policy = SOURCE_POLICY["math500"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        return make_taskspec(
            task_id=f"math500__{r.get('unique_id', i)}",
            capability="math",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="math_equal",
            expected_answer=r["answer"],
            prompt=r["problem"],
            system=_MATH_SYS,
            group_id="math500",
            domain="math",
            tags=["math", "eval"],
            url_or_ref=self.dataset_id,
        )


class AIMEAdapter(HFTaskAdapter):
    """AIME (older years) — train_allowed per ultra-data2 §3 (2025+ would be final_eval)."""

    source_name = "aime_old"
    capability = "math"
    dataset_id = "Maxwell-Jia/AIME_2024"
    hf_split = "train"
    policy = SOURCE_POLICY["aime_old"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        return make_taskspec(
            task_id=f"aime_old__{r['ID']}",
            capability="math",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="math_equal",
            expected_answer=str(r["Answer"]),
            prompt=r["Problem"],
            system="Solve the problem. Put the final integer answer in \\boxed{}.",
            group_id="aime",
            domain="math",
            tags=["math", "competition"],
            url_or_ref=self.dataset_id,
        )


class OmniMathAdapter(HFTaskAdapter):
    """Omni-MATH — olympiad problems with verifiable final answers."""

    source_name = "omni_math"
    capability = "math"
    dataset_id = "KbsdJames/Omni-MATH"
    hf_split = "test"
    policy = SOURCE_POLICY["omni_math"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        ans = (r.get("answer") or "").strip()
        if not ans:
            return None
        return make_taskspec(
            task_id=f"omni_math__{i}",
            capability="math",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="math_equal",
            expected_answer=ans,
            prompt=r["problem"],
            system=_MATH_SYS,
            group_id="omni_math",
            domain="math",
            subdomain=str(r.get("difficulty")),
            tags=["math", "olympiad"],
            url_or_ref=self.dataset_id,
        )


# --------------------------------------------------------- multiple-choice science/general
_MMLU_DOMAIN = {
    "physics": "science",
    "chemistry": "science",
    "biology": "science",
    "health": "science",
    "math": "math",
}


class MMLUProAdapter(HFTaskAdapter):
    """MMLU-Pro — multiple-choice, each row routed to a domain by its category."""

    source_name = "mmlu_pro"
    capability = "factual_qa"
    dataset_id = "TIGER-Lab/MMLU-Pro"
    hf_split = "test"
    policy = SOURCE_POLICY["mmlu_pro"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        domain = _MMLU_DOMAIN.get(r["category"], "general")
        lines = [f"{_LETTERS[j]}) {opt}" for j, opt in enumerate(r["options"])]
        prompt = r["question"] + "\n\nOptions:\n" + "\n".join(lines)
        gold = str(r["answer"]).strip()[:1].upper()
        return make_taskspec(
            task_id=f"mmlu_pro__{r['question_id']}",
            capability=_DOMAIN_CAP.get(domain, "factual_qa"),
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="mc_letter",
            expected_answer=gold,
            prompt=prompt,
            system=_MC_SYS,
            group_id="mmlu_pro",
            domain=domain,
            subdomain=r.get("category"),
            tags=["mc", "knowledge"],
            url_or_ref=self.dataset_id,
        )


_SUPERGPQA_SCI = {"Science", "Medicine", "Engineering", "Agronomy"}


class SuperGPQAAdapter(HFTaskAdapter):
    """SuperGPQA — graduate multiple-choice across 285 disciplines."""

    source_name = "supergpqa"
    capability = "science_knowledge"
    dataset_id = "m-a-p/SuperGPQA"
    hf_split = "train"
    policy = SOURCE_POLICY["supergpqa"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        domain = "science" if r["discipline"] in _SUPERGPQA_SCI else "general"
        lines = [f"{_LETTERS[j]}) {opt}" for j, opt in enumerate(r["options"])]
        prompt = r["question"] + "\n\nOptions:\n" + "\n".join(lines)
        return make_taskspec(
            task_id=f"supergpqa__{r['uuid']}",
            capability=_DOMAIN_CAP.get(domain, "factual_qa"),
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="mc_letter",
            expected_answer=str(r["answer_letter"]).strip()[:1].upper(),
            prompt=prompt,
            system=_MC_SYS,
            group_id="supergpqa",
            domain=domain,
            subdomain=r.get("field"),
            tags=["mc", "graduate"],
            url_or_ref=self.dataset_id,
        )


class GPQAStyleAdapter(HFTaskAdapter):
    """GPQA-Diamond — held out for final evaluation (expert science MC)."""

    source_name = "gpqa_diamond"
    capability = "science_knowledge"
    dataset_id = "hendrydong/gpqa_diamond_mc"
    hf_split = "test"
    policy = SOURCE_POLICY["gpqa_diamond"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        from ..grading.verifiers import extract_boxed

        letter = (extract_boxed(r["solution"]) or str(r["solution"])).strip()[:1].upper()
        return make_taskspec(
            task_id=f"gpqa_diamond__{i}",
            capability="science_knowledge",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="mc_letter",
            expected_answer=letter,
            prompt=r["problem"],
            system=_MC_SYS,
            group_id="gpqa_diamond",
            domain="science",
            subdomain=r.get("domain"),
            tags=["mc", "expert", "eval"],
            url_or_ref=self.dataset_id,
        )


class HLEStyleAdapter(HFTaskAdapter):
    """Humanity's Last Exam — expert QA, held out for final evaluation.

    SCHEMA UNVERIFIED: confirm field names (``question``/``answer``) against the live
    dataset before enabling materialize_all; ``_row_to_spec`` mapping is best-effort.
    """

    source_name = "hle"
    capability = "science_knowledge"
    dataset_id = "cais/hle"
    hf_split = "test"
    policy = SOURCE_POLICY["hle"]

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        q = r.get("question")
        a = r.get("answer")
        if not q or a is None:
            return None
        return make_taskspec(
            task_id=f"hle__{r.get('id', i)}",
            capability="science_knowledge",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="direct_qa",
            grader_type="math_equal",  # short-answer; swap to a dedicated HLE grader when added
            expected_answer=str(a),
            prompt=str(q),
            system="Answer the question. Put your final answer in \\boxed{}.",
            group_id="hle",
            domain="expert",
            tags=["expert", "eval", "schema-unverified"],
            url_or_ref=self.dataset_id,
        )
