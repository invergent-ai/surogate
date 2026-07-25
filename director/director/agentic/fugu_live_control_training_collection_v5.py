"""Prepared-index-safe live-control trajectory collection."""

from __future__ import annotations

from typing import override

from harbor.models.agent.context import AgentContext

from director.agentic.fugu_live_control_training_collection_v4 import (
    FuguLiveControlTrainingCollectionAgentV4,
)


COLLECTION_REVISION = "20260717-live-control-current-pool-v5-prepared-index-tests"
PROTECTION_POLICY = "prepared_index_test_blobs_restored_after_each_batch"


class FuguLiveControlTrainingCollectionAgentV5(
    FuguLiveControlTrainingCollectionAgentV4
):
    """Collect recovery traces while preserving the prepared verifier baseline."""

    @staticmethod
    def name() -> str:
        return "fugu-live-control-training-collection-v5"

    def version(self) -> str | None:
        return COLLECTION_REVISION

    @override
    def _record_fugu_metadata(self, context: AgentContext) -> None:
        super()._record_fugu_metadata(context)
        metadata = dict(context.metadata or {})
        metadata.update(
            {
                "collection_revision": COLLECTION_REVISION,
                "protected_test_restore_policy": PROTECTION_POLICY,
            }
        )
        context.metadata = metadata
