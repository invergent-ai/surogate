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
