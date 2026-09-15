"""Import Muse-Glimmer's separately published DFlash GGUF assistant."""
from contextlib import closing
from pathlib import Path
import math
import numpy as np
import torch

from surogate.serve.artifact import TensorSpec
from surogate.serve.artifact.layouts import encode_direct
from surogate.serve.artifact.geometry import validate_dflash_geometry
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.convert.common.quantize import quantize_and_encode


def find_drafter(model, explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError(f'DFlash checkpoint does not exist: {path}')
        return path
    candidates = sorted(Path(model).parent.glob('*dflash*.gguf'))
    matches = []
    for path in candidates:
        with closing(GgufSource(path)) as source:
            if source.kv('general.architecture') == 'dflash':
                matches.append(path)
    if len(matches) > 1:
        raise ValueError('multiple DFlash checkpoints found; select one with --dflash-model')
    return matches[0] if matches else None


def import_drafter(path, text, mask_token):
    specs, data = [], {}
    with closing(GgufSource(Path(path))) as source:
        if source.kv('general.architecture') != 'dflash':
            raise ValueError('expected a DFlash GGUF checkpoint')
        def get(key):
            value = source.kv('dflash.' + key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid dflash.{key}')
            return value
        def integer(key):
            value = get(key)
            if value != int(value):
                raise ValueError(f'invalid dflash.{key}')
            return int(value)
        h, n, m = (integer(k) for k in ('embedding_length', 'block_count', 'feed_forward_length'))
        q, kv, dim = (integer(k) for k in ('attention.head_count', 'attention.head_count_kv', 'attention.key_length'))
        pattern = source.kv('dflash.attention.sliding_window_pattern')
        # This export stores HF output indices plus one, including the embedding slot.
        targets = source.kv('dflash.target_layers')
        if not isinstance(targets, list) or any(type(i) is not int for i in targets):
            raise ValueError('DFlash target_layers must contain integer output indices')
        targets = [i - 1 for i in targets]
        if pattern not in ([True] * n, [True] * (n - 1) + [False]):
            raise ValueError('DFlash must use local attention with at most one final full layer')
        # The published Muse assistant omits the mask ID from its GGUF metadata.
        # Resolve it from the target tokenizer rather than assuming another family's ID.
        mask = mask_token
        g = dict(hidden=h, layers=n, local_layers=sum(pattern), intermediate=m,
                 query_heads=q, kv_heads=kv, head_dim=dim,
                 local_capacity=integer('attention.sliding_window'), mask_token=mask,
                 block_size=integer('block_size'), max_context=integer('context_length'),
                 rms_epsilon=get('attention.layer_norm_rms_epsilon'), rope_theta=get('rope.freq_base'),
                 attention_scale=dim ** -.5, feature_layers=len(targets), feature_rows=h * len(targets))
        validate_dflash_geometry(g, targets, text)
        entries = [('feature_projection', ['fc.weight'], (h, g['feature_rows'])),
                   ('context_norm', ['enc.output_norm.weight'], (h,)),
                   ('final_norm', ['output_norm.weight'], (h,))]
        for i in range(n):
            for name, roles, shape in (
                ('input_norm', ['attn_norm'], (h,)),
                ('attention/query_key_value', ['attn_q', 'attn_k', 'attn_v'], ((q+2*kv)*dim,h)),
                ('attention/query_norm', ['attn_q_norm'], (dim,)),
                ('attention/key_norm', ['attn_k_norm'], (dim,)),
                ('attention/output', ['attn_output'], (h,q*dim)),
                ('post_attention_norm', ['ffn_norm'], (h,)),
                ('mlp/gate_up', ['ffn_gate','ffn_up'], (2*m,h)),
                ('mlp/down', ['ffn_down'], (h,m))):
                entries.append((f'layers/{i}/{name}', [f'blk.{i}.{r}.weight' for r in roles], shape))
        consumed = set()
        for name, roles, shape in entries:
            values = [source.float32(r) for r in roles]
            if len(roles) > 1:
                rows = [q*dim, kv*dim, kv*dim] if name.endswith('query_key_value') else [m, m]
                if any(tuple(value.shape) != (rows[i], h) for i, value in enumerate(values)):
                    raise ValueError(f'DFlash {name}: component projection dimensions do not match the checkpoint')
            value = torch.from_numpy(np.concatenate(values, axis=0) if len(values) > 1 else values[0].copy())
            if tuple(value.shape) != shape:
                raise ValueError(f'DFlash {name}: expected {shape}, got {tuple(value.shape)}')
            name = 'dflash/' + name
            norm = len(shape) == 1
            fmt = 'BF16' if norm else 'W8G32_F16S'
            specs.append(TensorSpec(name, shape, fmt, 'contiguous-le-v1' if norm else 'row-split-k128-v1'))
            data[name] = encode_direct(value.to(torch.bfloat16), fmt) if norm else quantize_and_encode(value, fmt, device='cpu')
            consumed.update(roles)
        if source.tensors.keys() - consumed:
            raise ValueError(f'unsupported DFlash tensors: {sorted(source.tensors.keys() - consumed)}')
    return g, targets, specs, data
