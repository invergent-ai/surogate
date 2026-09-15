"""Real Muse GGUF image encoder against Transformers, including ragged windows.

Set SUROGATE_MUSE_FIXTURE to a directory containing config.json, muse-vision.sinfer,
and mmproj-Muse-Glimmer-30B-BF16.gguf. Uses the standalone serving tower executable.
"""
import json
from contextlib import closing
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize('height,width,video', [(6, 10, False), (34, 38, False), (6, 10, True), (24, 24, True)])
def test_muse_checkpoint_vision(tmp_path, height, width, video):
    fixture = os.environ.get('SUROGATE_MUSE_FIXTURE')
    if not fixture:
        pytest.skip('set SUROGATE_MUSE_FIXTURE for real Muse-Glimmer vision validation')
    import torch
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerVisionConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerVisionModel, MuseGlimmerVisionRotaryEmbedding
    from surogate.serve.convert.common.gguf_source import GgufSource

    fixture = Path(fixture)
    cfg = MuseGlimmerVisionConfig(**json.loads((fixture / 'config.json').read_text())['vision_config'])
    # The GGUF projector sums temporal slabs for still images. Its 2D patch matrix
    # is equivalent to the original temporal convolution on two identical frames.
    temporal_file = fixture / 'temporal_patch.bin'
    cfg.patch_temporal = 2 if temporal_file.is_file() else 1
    if video and cfg.patch_temporal != 2:
        pytest.skip('supply original temporal_patch.bin to validate distinct video frames')
    cfg._attn_implementation = 'sdpa'
    with torch.device('meta'):
        tower = MuseGlimmerVisionModel(cfg)
    state = {}
    projector = []
    with closing(GgufSource(fixture / 'mmproj-Muse-Glimmer-30B-BF16.gguf')) as source:
        for name in tower.state_dict():
            if name == 'patch_embedder.patch_embedding.weight':
                src = 'v.patch_embd.weight'
            elif name == 'patch_embedder.position_embedding_table.weight':
                src = 'v.position_embd.weight'
            elif name.startswith('ln_'):
                src = 'v.' + name.removeprefix('ln_').replace('pre.', 'pre_ln.').replace('post.', 'post_ln.')
            else:
                _, i, *rest = name.split('.')
                role = '.'.join(rest)
                for hf, gg in [('attn.q_proj', 'attn_q'), ('attn.k_proj', 'attn_k'),
                               ('attn.v_proj', 'attn_v'), ('attn.proj', 'attn_out'),
                               ('mlp.fc1', 'ffn_up'), ('mlp.fc2', 'ffn_down'),
                               ('norm1', 'ln1'), ('norm2', 'ln2')]:
                    role = role.replace(hf, gg)
                src = f'v.blk.{i}.{role}'
            value = (torch.frombuffer(bytearray(temporal_file.read_bytes()), dtype=torch.bfloat16).float()
                     if name == 'patch_embedder.patch_embedding.weight' and cfg.patch_temporal == 2
                     else torch.from_numpy(source.float32(src).copy()))
            if '.attn.q_proj.' in name or '.attn.k_proj.' in name:
                heads = cfg.num_attention_heads
                value = value.reshape(heads, -1, 2, *value.shape[1:]).transpose(1, 2).reshape_as(value)
            state[name] = value.reshape(tower.state_dict()[name].shape).bfloat16().cuda()
        for i in range(3):
            projector.append(torch.from_numpy(source.float32(f'mm.{i}.weight').copy()).bfloat16().cuda())
    tower.load_state_dict(state, assign=True)
    tower.rotary_emb = MuseGlimmerVisionRotaryEmbedding(cfg).cuda()
    tower.eval()
    torch.manual_seed(851)
    patches = (torch.rand(height * width, cfg.patch_temporal, 3 * cfg.patch_size ** 2, device='cuda') * 2 - 1).bfloat16()
    if not video and cfg.patch_temporal == 2:
        patches[:, 1] = patches[:, 0]
    patches = patches.flatten(1)
    with torch.no_grad():
        result = tower(pixel_values=patches, grid_thw=torch.tensor([[1, height, width]], device='cuda'))
        expected = result.last_hidden_state
        for i, weight in enumerate(projector):
            expected = torch.nn.functional.linear(expected, weight)
            if i != 2:
                expected = torch.nn.functional.gelu(expected, approximate='none')
        values = expected.float()
        expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)).bfloat16().float().cpu().reshape(-1)
    grouped = patches.reshape(height // 2, 2, width // 2, 2, -1).permute(0, 2, 1, 3, 4).contiguous()
    input_path, output_path = tmp_path / 'patches.bin', tmp_path / 'output.bin'
    input_path.write_bytes(grouped.bfloat16().cpu().view(torch.uint16).numpy().tobytes())
    del tower, state, projector, grouped, patches, result
    torch.cuda.empty_cache()
    binary = Path('csrc/build-serve/serve_tests/sinfer_gemma_vision_test').resolve()
    subprocess.run([str(binary), str(fixture / 'muse-vision.sinfer'), str(input_path),
                    str(height), str(width), str(output_path), '1'], check=True)
    actual = torch.frombuffer(bytearray(output_path.read_bytes()), dtype=torch.bfloat16).float()
    assert actual.shape == expected.shape
    relative = (actual - expected).norm() / expected.norm()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    print(f'Muse {"video" if video else "image"} {height}x{width}: relative L2={relative:.6f}, cosine={cosine:.7f}')
    assert torch.isfinite(actual).all()
    # Fifty BF16 residual blocks amplify rounding on the deliberately tiny random
    # grid. Layer probes show gradual drift (0.003 L2 at layer 0, 0.089 at 49),
    # rather than a permutation/window mismatch. Require close whole-vector
    # agreement alongside the independent attention, rotation and shuffle tests.
    assert relative < 0.10
    assert cosine > 0.995
