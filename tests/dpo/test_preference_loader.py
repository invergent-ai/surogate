import json

from surogate.dpo.trainer import _load_pref_datasets, _load_pref_rows


def test_load_pref_rows_preserves_optional_thinking_mode(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = {
        "prompt": [{"role": "user", "content": "Răspunde direct."}],
        "chosen": "Da",
        "rejected": "Nu",
        "enable_thinking": False,
        "ignored_metadata": "x",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    assert _load_pref_rows(str(path)) == [
        {
            "prompt": row["prompt"],
            "chosen": "Da",
            "rejected": "Nu",
            "enable_thinking": False,
        }
    ]


def test_load_pref_datasets_concatenates_shards_in_order(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"pairs-{index}.jsonl"
        path.write_text(
            json.dumps({"prompt": f"p{index}", "chosen": f"c{index}", "rejected": f"r{index}"}) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))

    assert _load_pref_datasets(paths) == [
        {"prompt": "p0", "chosen": "c0", "rejected": "r0"},
        {"prompt": "p1", "chosen": "c1", "rejected": "r1"},
    ]


def test_load_pref_datasets_requires_at_least_one_path():
    import pytest

    with pytest.raises(ValueError, match="at least one preference dataset"):
        _load_pref_datasets([])
