"""Long-context source adapters (EXECUTION-PENDING: long_context harness).

The answer must require the provided documents (not solvable from metadata); context source
documents are held out by split (ultra-data2 §10).
"""

from __future__ import annotations

from ..policy import SOURCE_POLICY
from ..schemas import TaskSpec
from .hf import make_taskspec
from .raw import RawRecordAdapter


class LongContextDocPackAdapter(RawRecordAdapter):
    """Generated long-context document packs: retrieval + synthesis across long inputs."""

    source_name = "longctx_generated"
    capability = "long_context"
    policy = SOURCE_POLICY["longctx_generated"]
    harness = "long_context"

    def _to_spec(self, raw: dict, i: int) -> TaskSpec | None:
        q = raw.get("question")
        a = raw.get("answer")
        if not q or a is None or not raw.get("documents"):
            return None
        return make_taskspec(
            task_id=f"{self.source_name}__{raw.get('id', i)}",
            capability="long_context",
            source_name=self.source_name,
            source_version=self.version,
            policy=self.policy,
            harness="long_context",
            grader_type="contains",
            expected_answer=str(a),
            messages=[{"role": "user", "content": str(q)}],
            context_documents=raw["documents"],
            group_id=raw.get("corpus", "longctx"),
            contamination_group=raw.get("corpus"),
            domain="long_context",
            requires_long_context=True,
            tags=["long-context"],
        )


class MRCRStyleAdapter(LongContextDocPackAdapter):
    """MRCR / Michelangelo-style multi-needle retrieval — held out for final evaluation."""

    source_name = "mrcr"
    policy = SOURCE_POLICY["mrcr"]
