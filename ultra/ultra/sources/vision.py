"""Multimodal/chart source adapter (EXECUTION-PENDING: vision_qa harness).

Only usable when the worker pool includes vision-capable models (ultra-data2 §2).
"""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import HFTaskAdapter, make_taskspec


class CharXivAdapter(HFTaskAdapter):
    """CharXiv chart reasoning — held out for final evaluation.

    SCHEMA UNVERIFIED: confirm field names (image / question / answer) against the live
    dataset before materialize_all.
    """

    source_name = "charxiv"
    capability = "multimodal"
    dataset_id = "princeton-nlp/CharXiv"
    hf_split = "validation"
    policy = SOURCE_POLICY["charxiv"]
    harness = "vision_qa"

    def _row_to_spec(self, r: dict, i: int) -> TaskSpec | None:
        q = r.get("reasoning_q") or r.get("question")
        a = r.get("reasoning_a") or r.get("answer")
        if not q or a is None:
            return None
        return make_taskspec(
            task_id=f"charxiv__{r.get('figure_id', i)}",
            capability="multimodal",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="vision_qa",
            grader_type="vision_match",
            expected_answer=str(a),
            messages=[{"role": "user", "content": str(q)}],
            multimodal_assets=[{"type": "image", "ref": r.get("image_path") or r.get("image")}],
            group_id="charxiv",
            domain="chart",
            requires_vision=True,
            tags=["vision", "chart", "eval", "schema-unverified"],
            url_or_ref=self.dataset_id,
        )
