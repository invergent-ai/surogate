import json
import pytest
from surogate.serve.convert.qwen3_5 import convert as dense
from surogate.serve.convert.qwen3_5_moe import convert as moe


@pytest.mark.parametrize("loader", [dense.load_resources, moe.load_resources])
def test_resources_are_read_from_the_selected_checkpoint(tmp_path, loader):
    files = {"tokenizer.json": b'{"model":{"vocab":{"changed":0}}}',
             "tokenizer_config.json": b'{}', "chat_template.jinja": b'{{ messages }}',
             "generation_config.json": b'{"eos_token_id":0}'}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    resources = loader(tmp_path)
    assert {r.name: r.data for r in resources} == {"frontend/" + n: d for n, d in files.items()}
    (tmp_path / "tokenizer.json").write_text('{"model":{"vocab":{"renamed":0}}}')
    changed = {r.name: r.data for r in loader(tmp_path)}
    assert changed["frontend/tokenizer.json"] != files["tokenizer.json"]
    assert json.loads(changed["frontend/generation_config.json"])["eos_token_id"] == 0
