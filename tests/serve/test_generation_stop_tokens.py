"""Chat EOS must survive synthesized generation config for native serving."""

import json

import pytest

from surogate.serve.convert.common.conversion import _synthesize_generation_config, load_resources
from surogate.serve.convert.common.inventory import ResourceSpec
from surogate.serve.ingest import _gguf_fingerprint, source_fingerprint


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


def test_generation_sampling_settings_survive_conversion_and_invalidate_cache(tmp_path):
    config = dict(eos_token_id=[10, 12], temperature=0.42, top_k=7, top_p=0.91,
                  min_p=0.05, repetition_penalty=1.12)
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps(config))
    fingerprint = source_fingerprint(tmp_path)
    spec = ResourceSpec("frontend/generation_config.json", "raw-bytes-v1")
    resources = load_resources(tmp_path, [spec])
    assert json.loads(resources[0].data) == config
    config["temperature"] = 0.8
    path.write_text(json.dumps(config))
    assert source_fingerprint(tmp_path) != fingerprint
    assert json.loads(load_resources(tmp_path, [spec])[0].data) == config


def test_gguf_sampling_settings_invalidate_only_affected_caches(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fingerprint fixture")
    previous = _gguf_fingerprint(path)
    assert _gguf_fingerprint(path, generation_defaults={}) == previous
    configured = _gguf_fingerprint(path, generation_defaults={"temperature": 0.0})
    assert configured != previous
    assert _gguf_fingerprint(path, generation_defaults={"temperature": 0.5}) != configured
