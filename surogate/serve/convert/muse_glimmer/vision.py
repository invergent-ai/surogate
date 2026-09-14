"""Muse-Glimmer vision projector metadata and GGUF bindings."""
from pathlib import Path
import math

from surogate.serve.convert.common.gguf_source import GgufSource


def find_projector(model: Path, explicit=None) -> Path:
    candidates = [Path(explicit)] if explicit else sorted(Path(model).parent.glob('*mmproj*.gguf'))
    matches = []
    for candidate in candidates:
        try:
            source = GgufSource(candidate)
            try:
                if source.kv('clip.projector_type') == 'muse-glimmer':
                    matches.append(candidate)
            finally:
                source.close()
        except (ValueError, OSError):
            if explicit:
                raise
    if len(matches) != 1:
        raise ValueError('Muse-Glimmer needs one matching vision projector; select it with --mmproj')
    return matches[0]


def vision_objects(source, text):
    if source.kv('clip.projector_type') != 'muse-glimmer':
        raise ValueError('expected a Muse-Glimmer vision projector')
    def integer(key):
        value = source.kv('clip.vision.' + key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'invalid clip.vision.{key}')
        return value
    h, m, heads = (integer(k) for k in ('embedding_length', 'feed_forward_length', 'attention.head_count'))
    patch, merge = integer('patch_size'), integer('spatial_merge_size')
    pos, ph = source.tensor('v.position_embd.weight').shape, source.tensor('mm.0.weight').shape[0]
    epsilon = source.kv('clip.vision.attention.layer_norm_epsilon')
    if (h % heads or h // heads not in (64, 72, 96) or merge != 2 or pos[1] != h or
            math.isqrt(pos[0]) ** 2 != pos[0] or not isinstance(epsilon, (float, int)) or
            not math.isfinite(epsilon) or epsilon <= 0 or
            integer('projection_dim') != text['hidden']):
        raise ValueError('unsupported Muse-Glimmer vision geometry')
    g = dict(muse_glimmer=1, layers=integer('block_count'), hidden=h, intermediate=m,
             heads=heads, patch_dim=3*patch*patch, merge=merge, position_embeddings=pos[0],
             rotary_dim=h//heads, output_hidden=text['hidden'], rope_theta=10000,
             norm_epsilon=epsilon, projector_hidden=ph, max_image_tokens=4096)
    objects = [('vision/patch_embedding', 'v.patch_embd.weight', (h, 3*patch*patch), 0),
               ('vision/position_embedding', 'v.position_embd.weight', pos, 0)]
    for norm in ('pre', 'post'):
        for part in ('weight', 'bias'):
            objects.append((f'vision/{norm}_norm/{part}', f'v.{norm}_ln.{part}', (h,), 0))
    for i, shape in enumerate(((ph, h*merge*merge), (ph, ph), (text['hidden'], ph))):
        objects.append((f'vision/projector/{i}', f'mm.{i}.weight', shape, 0))
    for i in range(g['layers']):
        for role, source_role, rows, cols in (
                ('attention/query', 'attn_q', h, h), ('attention/key', 'attn_k', h, h),
                ('attention/value', 'attn_v', h, h), ('attention/output', 'attn_out', h, h),
                ('mlp/fc1', 'ffn_up', m, h), ('mlp/fc2', 'ffn_down', h, m)):
            permute = heads if role in ('attention/query', 'attention/key') else 0
            for part, shape in (('weight', (rows, cols)), ('bias', (rows,))):
                name = role if part == 'weight' else role + '_bias'
                objects.append((f'vision/layers/{i}/{name}', f'v.blk.{i}.{source_role}.{part}', shape, permute))
        for norm, source_norm in (('norm1', 'ln1'), ('norm2', 'ln2')):
            for part in ('weight', 'bias'):
                objects.append((f'vision/layers/{i}/{norm}/{part}', f'v.blk.{i}.{source_norm}.{part}', (h,), 0))
    processor = dict(muse_glimmer=True, gemma_version=0, image_token_id=200092,
                     video_token_id=200091, patch_size=patch, spatial_merge_size=merge,
                     position_embeddings=pos[0], max_soft_tokens=4096, resample=1,
                     image_token='<|patch|>', video_token='<|video|>',
                     boi_token='<|image_start|>', eoi_token='<|image_end|>')
    return g, objects, processor
