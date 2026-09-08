import json
from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for
from surogate.serve.convert.qwen3_5_moe import draft_head, inventory


def test_missing_bundled_ranking_uses_this_tokenizers_ids(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {str(i): i for i in range(50)}}}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    g = inventory.geometry_from_config(config_for(), token_domain=50)
    context = draft_head.compute_shortlist(draft_head.DEFAULT_RANKING, tmp_path, geometry=g)
    assert context.selected.tolist() == list(range(50))
    assert context.ranking is None
