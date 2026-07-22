"""`type: preference` datasets load through the shared dataset framework."""

import json

import pytest

from surogate.core.config.dataset_config import PreferenceDatasetConfig, create_dataset_config
from surogate.dpo.trainer import _load_pref_datasets
from surogate.utils.dict import DictDefault


def _pref_cfg(path, **overrides) -> PreferenceDatasetConfig:
    return create_dataset_config(DictDefault({"path": str(path), "type": "preference", **overrides}))


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_create_dataset_config_returns_preference_config():
    cfg = _pref_cfg("x.jsonl")
    assert isinstance(cfg, PreferenceDatasetConfig)
    assert cfg.prompt_field == "prompt"
    assert cfg.chosen_field == "chosen"
    assert cfg.rejected_field == "rejected"


def test_load_preserves_optional_thinking_mode_and_drops_metadata(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "prompt": [{"role": "user", "content": "Răspunde direct."}],
                "chosen": "Da",
                "rejected": "Nu",
                "enable_thinking": False,
                "ignored_metadata": "x",
            },
            {
                "prompt": [{"role": "user", "content": "Încă una."}],
                "chosen": "Merge",
                "rejected": "Nu merge",
            },
        ],
    )

    rows = _load_pref_datasets([_pref_cfg(path)])
    assert rows == [
        {
            "prompt": [{"role": "user", "content": "Răspunde direct."}],
            "chosen": "Da",
            "rejected": "Nu",
            "enable_thinking": False,
        },
        {
            "prompt": [{"role": "user", "content": "Încă una."}],
            "chosen": "Merge",
            "rejected": "Nu merge",
        },
    ]


def test_load_concatenates_shards_in_declaration_order(tmp_path):
    cfgs = []
    for index in range(2):
        path = tmp_path / f"pairs-{index}.jsonl"
        _write_jsonl(path, [{"prompt": f"p{index}", "chosen": f"c{index}", "rejected": f"r{index}"}])
        cfgs.append(_pref_cfg(path))

    assert _load_pref_datasets(cfgs) == [
        {"prompt": "p0", "chosen": "c0", "rejected": "r0"},
        {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
    ]


def test_load_requires_at_least_one_dataset():
    with pytest.raises(ValueError, match="at least one preference dataset"):
        _load_pref_datasets([])


def test_load_honors_field_mappings(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [{"question": "p", "better": "c", "worse": "r"}])

    rows = _load_pref_datasets(
        [_pref_cfg(path, prompt_field="question", chosen_field="better", rejected_field="worse")]
    )
    assert rows == [{"prompt": "p", "chosen": "c", "rejected": "r"}]


def test_load_rejects_missing_required_column(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [{"prompt": "p", "chosen": "c"}])

    with pytest.raises(ValueError, match="Rejected field 'rejected' is missing"):
        _load_pref_datasets([_pref_cfg(path)])


def test_load_drops_invalid_rows(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {"prompt": "p0", "chosen": "c0", "rejected": "r0"},
            {"prompt": "p1", "chosen": None, "rejected": "r1"},
        ],
    )

    assert _load_pref_datasets([_pref_cfg(path)]) == [{"prompt": "p0", "chosen": "c0", "rejected": "r0"}]


def test_load_supports_parquet(tmp_path):
    from datasets import Dataset

    path = tmp_path / "pairs.parquet"
    Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}]).to_parquet(str(path))

    assert _load_pref_datasets([_pref_cfg(path)]) == [{"prompt": "p", "chosen": "c", "rejected": "r"}]


def test_load_honors_samples_limit(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(path, [{"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(4)])

    rows = _load_pref_datasets([_pref_cfg(path, samples=2)])
    assert len(rows) == 2
    assert all(row["prompt"].startswith("p") for row in rows)
