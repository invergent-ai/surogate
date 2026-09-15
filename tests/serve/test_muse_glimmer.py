"""Muse-Glimmer text geometry and tokenizer contracts."""
from types import SimpleNamespace

import pytest

from surogate.serve.convert.muse_glimmer.convert import geometry


def source(**overrides):
    values = {
        'general.architecture': 'muse-glimmer',
        'muse-glimmer.block_count': 52, 'muse-glimmer.embedding_length': 6656,
        'muse-glimmer.feed_forward_length': 19968, 'muse-glimmer.context_length': 131072,
        'muse-glimmer.attention.head_count': 32, 'muse-glimmer.attention.head_count_kv': 2,
        'muse-glimmer.attention.key_length': 128, 'muse-glimmer.attention.value_length': 128,
        'muse-glimmer.attention.layer_norm_rms_epsilon': 1e-5,
        'muse-glimmer.attention.sliding_window': 2048,
        'muse-glimmer.attention.sliding_window_pattern': 4,
        'muse-glimmer.rope.freq_base': 500000,
        'muse-glimmer.logit_scale': 0.19611613513818404,
        'muse-glimmer.final_logit_softcapping': 20,
        'tokenizer.ggml.tokens': ['x'] * 202048,
    }
    values.update(overrides)
    return SimpleNamespace(kv=lambda key, default=None: values.get(key, default),
                           tensor=lambda key: SimpleNamespace(shape=(202048, 6656)))


def test_muse_preserves_nope_schedule_and_output_arithmetic():
    g, layers = geometry(source())
    assert layers == ['sliding_attention'] * 3 + ['full_attention'] + (['sliding_attention'] * 3 + ['full_attention']) * 12
    assert g['rotary_dim'] == g['rope_theta'] == 0
    assert g['sliding_rotary_dim'] == 128
    assert g['sliding_rope_theta'] == 500000
    assert g['post_norm_epsilon'] == 1e-8
    assert g['rms_epsilon'] == 1e-5
    assert g['logit_scale'] == 0.19611613513818404
    assert g['logit_softcap'] == 20
    assert g['max_context'] == 131072


def test_muse_explicit_window_pattern_is_not_replaced():
    pattern = [False, True] * 26
    _, layers = geometry(source(**{'muse-glimmer.attention.sliding_window_pattern': pattern}))
    assert layers == ['full_attention', 'sliding_attention'] * 26


@pytest.mark.parametrize('key,value', [
    ('attention.head_count', 32.5), ('attention.head_count_kv', 3),
    ('block_count', 52.5), ('attention.value_length', 64), ('logit_scale', float('nan')),
    ('final_logit_softcapping', 0), ('attention.sliding_window_pattern', [2] * 52),
    ('attention.sliding_window_pattern', [True] * 51), ('attention.sliding_window_pattern', True),
])
def test_muse_rejects_invalid_geometry(key, value):
    with pytest.raises(ValueError):
        geometry(source(**{'muse-glimmer.' + key: value}))


def test_temporal_download_refuses_full_shard_response(tmp_path, monkeypatch):
    import numpy as np
    from surogate.serve.convert.muse_glimmer import temporal
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    class Response:
        status = 200
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size): raise AssertionError('must not read a full model shard')
    monkeypatch.setattr(temporal.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match='requested Muse temporal weight range'):
        temporal.temporal_projection(np.zeros((1536, 588), dtype=np.float32))
    assert not list(tmp_path.rglob('*.bf16'))


def test_temporal_download_checks_content(tmp_path, monkeypatch):
    import numpy as np
    from surogate.serve.convert.muse_glimmer import temporal
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path))
    class Response:
        status = 206
        headers = {'Content-Range': f'bytes {temporal._START}-{temporal._START + temporal._SIZE - 1}/50000000000'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            assert size == temporal._SIZE + 1
            return bytes(temporal._SIZE)
    monkeypatch.setattr(temporal.urllib.request, 'urlopen', lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match='checksum mismatch'):
        temporal.temporal_projection(np.zeros((1536, 588), dtype=np.float32))
    assert not list(tmp_path.rglob('*.bf16'))


def test_text_gguf_conversion_without_optional_drafter(tmp_path):
    import numpy as np
    gguf = pytest.importorskip('gguf')
    from surogate.serve.convert.muse_glimmer.convert import convert
    from surogate.serve.artifact.container import Artifact
    path = tmp_path / 'tiny.gguf'
    writer = gguf.GGUFWriter(str(path), 'muse-glimmer')
    metadata = {'block_count': 1, 'embedding_length': 32, 'feed_forward_length': 64,
        'context_length': 4096, 'attention.head_count': 2, 'attention.head_count_kv': 1,
        'attention.key_length': 16, 'attention.sliding_window': 2048,
        'attention.sliding_window_pattern': 1, 'rope.freq_base': 500000.,
        'attention.layer_norm_rms_epsilon': 1e-5, 'logit_scale': .2,
        'final_logit_softcapping': 20.}
    for key, value in metadata.items():
        (writer.add_float32 if isinstance(value, float) else writer.add_uint32)('muse-glimmer.' + key, value)
    writer.add_token_list([f't{i}' for i in range(32)])
    tensors = {'token_embd.weight': (32, 32), 'output.weight': (32, 32), 'output_norm.weight': (32,)}
    for name in ['attn_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm']:
        tensors['blk.0.' + name + '.weight'] = (32,)
    for name, shape in {'attn_q_norm': (16,), 'attn_k_norm': (16,), 'attn_q': (32,32),
            'attn_k': (16,32), 'attn_v': (16,32), 'attn_gate': (32,32), 'attn_output': (32,32),
            'ffn_gate': (64,32), 'ffn_up': (64,32), 'ffn_down': (32,64)}.items():
        tensors['blk.0.' + name + '.weight'] = shape
    for name, shape in tensors.items():
        writer.add_tensor(name, np.ones(shape, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    front = tmp_path / 'frontend'
    front.mkdir()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        (front / name).write_text('{}')
    (front / 'chat_template.jinja').write_text('{{ messages }}')
    out = convert(path, front, tmp_path / 'tiny.sinfer')
    with Artifact(out) as artifact:
        assert artifact.geometry['hidden'] == 32
        assert not artifact.dflash_geometry and not artifact.dflash_target_layers
        assert not artifact.vision_geometry
        assert len([o for o in artifact.objects if o.name.startswith('text/')]) == len(tensors)
