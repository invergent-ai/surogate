from pathlib import Path

import torch

from surogate.serve.convert.qwen3_5_moe import (
    convert,
    draft_head,
    inventory,
    recipe,
)


def test_report_retains_target_specific_provenance_and_component_bytes(
    tmp_path: Path,
) -> None:
    from tests.serve.test_qwen3_5_moe_checkpoint_config import config_for
    g = inventory.geometry_from_config(config_for(), token_domain=500)
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources, geometry=g)
    base_source = recipe.SourcePreflight(883, 1045, 26, {"BF16": 1045})
    dflash_source = recipe.SourcePreflight(51, 69, 1, {"BF16": 69})
    report = convert.build_conversion_report(
        model_dir=tmp_path / "model",
        dflash_model_dir=tmp_path / "dflash",
        out_path=tmp_path / "model.sinfer",
        arguments={},
        base_config_summary={"hidden": 128, "draft_vocab": 500, "token_domain": 500},
        dflash_config_summary={"hidden_size": 2048},
        base_source_preflight=base_source,
        dflash_source_preflight=dflash_source,
        objects=plan.objects,
        elapsed_seconds=1.0,
        final_bytes=123,
        device=torch.device("cpu"),
        ranking_path=draft_head.DEFAULT_RANKING,
        revision="test-revision",
        environment={"python": "test"},
    )

    assert report["identity"] == {
        "model_id": inventory.MODEL_ID,
        "weights_id": inventory.WEIGHTS_ID,
    }
    assert report["target_key"] == inventory.TARGET_KEY
    assert report["recipe_id"] == convert.RECIPE_ID
    assert report["source"]["base_model_path"] == str(
        (tmp_path / "model").resolve()
    )
    assert report["source"]["dflash_model_path"] == str(
        (tmp_path / "dflash").resolve()
    )
    assert report["source_preflight"]["base"]["tensors"] == 1045
    assert report["source_preflight"]["dflash"]["tensors"] == 69
    assert report["source_preflight"]["combined"]["tensors"] == 1114
    assert report["draft_head"] == {"rows": 500, "tokenizer_vocab_size": 500}
    assert report["artifact"]["bytes"] == 123
    assert "gguf_evidence_path" not in report["source"]
