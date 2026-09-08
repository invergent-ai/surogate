"""Frequency shortlist bounded by the checkpoint and its tokenizer."""

from pathlib import Path

from surogate.serve.convert.common.checkpoint import tokenizer_domain
from surogate.serve.convert.common.draft_head import (
    DraftHeadContext,
    compute_shortlist as _compute_shortlist,
    load_total_counts,
    materialize_draft_head,
    materialize_draft_head_token_ids,
    read_special_ids,
    select_shortlist,
)

DRAFT_HEAD_OBJECT = "text/draft_head"
DRAFT_HEAD_TOKEN_IDS_OBJECT = "text/draft_head_token_ids"
DEFAULT_RANKING = Path("freq_corpus/fixtures/ranking/ranking.train.counts.i64")


def compute_shortlist(ranking_path, tokenizer_dir, *, geometry) -> DraftHeadContext:
    domain = tokenizer_domain(tokenizer_dir)
    if domain != geometry.token_domain:
        raise ValueError("resolved token domain disagrees with the checkpoint tokenizer")
    return _compute_shortlist(ranking_path, tokenizer_dir, n=geometry.draft_vocab,
                              vocab=geometry.vocab, tokenizer_vocab_size=domain,
                              require_tokenizer_match=(Path(ranking_path) == DEFAULT_RANKING or
                                  Path(ranking_path).resolve() ==
                                  (Path(__file__).resolve().parents[2] / "tools" / DEFAULT_RANKING).resolve()))
