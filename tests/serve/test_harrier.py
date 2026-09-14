"""Harrier checkpoint geometry and tokenizer contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from surogate.serve.convert.harrier.convert import frontend, geometry_from_gguf
from surogate.serve.gguf.lean import LeanGguf


@pytest.mark.parametrize("arch,layers,hidden,heads,kv,dim,window,scale", [
    ("gemma3", 18, 640, 4, 1, 256, 0, 1.0),
    ("qwen3", 28, 1024, 16, 8, 128, 0, 1.0),
    ("gemma-embedding", 62, 5376, 32, 16, 128, 1024, 8.0),
])
def test_harrier_family_metadata(arch, layers, hidden, heads, kv, dim, window, scale):
    metadata = {"general.architecture": arch, "tokenizer.ggml.tokens": ["a", "b"],
                **{f"{arch}.{key}": value for key, value in {
                    "pooling_type": 3, "block_count": layers, "embedding_length": hidden,
                    "feed_forward_length": 21504 if layers == 62 else 2048,
                    "attention.head_count": heads, "attention.head_count_kv": kv,
                    "attention.key_length": dim, "attention.value_length": dim,
                    "attention.sliding_window": window, "context_length": 131072 if layers == 62 else 32768,
                    "attention.layer_norm_rms_epsilon": 1e-6, "rope.freq_base": 1e6,
                    "rope.scaling.type": "linear", "rope.scaling.factor": scale,
                }.items()}}
    source = SimpleNamespace(kv=metadata.get, tensor=lambda _: SimpleNamespace(shape=(2, hidden)))
    target, geometry, kinds = geometry_from_gguf(source)
    assert target == ("qwen3_embedding" if arch == "qwen3" else "gemma3_embedding")
    assert geometry["max_context"] == 32768
    assert geometry["hidden"] == hidden and geometry["kv_heads"] == kv
    assert geometry["rope_frequency_scale"] == 1 / scale
    assert geometry["attention_scale"] == pytest.approx((168 if layers == 62 else dim) ** -.5)
    if layers == 62:
        assert kinds[:7] == ["sliding_attention"] * 5 + ["full_attention", "sliding_attention"]
    else:
        assert set(kinds) == {"full_attention"}
    metadata[f"{arch}.pooling_type"] = 1
    with pytest.raises(ValueError, match="last-token pooling"):
        geometry_from_gguf(source)


@pytest.mark.parametrize("name", ["harrier-oss-v1-0.6B-Q8_0.gguf", "harrier-oss-v1-270M-Q8_0.gguf",
                                 "harrier-oss-v1-27B-Q8_0.gguf"])
def test_local_harrier_tokenizer_is_self_contained(name):
    path = Path(__file__).resolve().parents[2] / "models" / name
    if not path.is_file():
        pytest.skip("requires the local Harrier GGUF")
    from sentencepiece import SentencePieceProcessor
    from tokenizers import Tokenizer
    with LeanGguf(path) as reader:
        resources = frontend(reader)
        eos = reader.kv("tokenizer.ggml.eos_token_id")
    if "frontend/tokenizer.model" in resources:
        tokenizer = SentencePieceProcessor(model_proto=resources["frontend/tokenizer.model"])
        assert tokenizer.bos_id() == 2 and tokenizer.eos_id() == eos
        assert tokenizer.encode("hello")
    else:
        tokenizer = Tokenizer.from_str(resources["frontend/tokenizer.json"].decode())
        assert tokenizer.encode("Hello world", add_special_tokens=False).ids == (
            [9707, 1879] if "0.6B" in name else [9259, 1902])
        if "27B" in name:
            assert tokenizer.get_vocab_size() == 262208
            assert tokenizer.encode("a  b", add_special_tokens=False).ids == [236746, 138, 236763]
