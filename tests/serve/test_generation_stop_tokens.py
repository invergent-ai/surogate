"""Chat EOS must survive synthesized generation config for native serving."""

import json

import pytest

from surogate.serve.convert.common.conversion import _synthesize_generation_config


@pytest.mark.parametrize("decoder", [True, False])
def test_chat_eos_is_added_to_base_model_eos(tmp_path, decoder):
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config=dict(eos_token_id=10))))
    tokenizer = dict(eos_token="<|im_end|>")
    if decoder:
        tokenizer["added_tokens_decoder"] = {"12": dict(content="<|im_end|>")}
    else:
        (tmp_path / "tokenizer.json").write_text(json.dumps(dict(added_tokens=[dict(id=12, content="<|im_end|>")])))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer))
    result = json.loads(_synthesize_generation_config(tmp_path))
    assert result["eos_token_id"] == [10, 12]


def test_existing_stop_list_is_not_duplicated(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(eos_token_id=[10, 12], pad_token_id=0)))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(dict(eos_token=dict(content="end"), added_tokens_decoder={"12": dict(content="end")}))
    )
    assert json.loads(_synthesize_generation_config(tmp_path)) == dict(eos_token_id=[10, 12], pad_token_id=0)
