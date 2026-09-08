"""Draft shortlist resolved against this checkpoint's tokenizer and output head."""

from surogate.serve.convert.qwen3_5.draft_head import (
    DEFAULT_RANKING, DRAFT_HEAD_OBJECT, DRAFT_HEAD_TOKEN_IDS_OBJECT,
    DraftHeadContext, compute_shortlist, materialize_draft_head,
    materialize_draft_head_token_ids,
)
