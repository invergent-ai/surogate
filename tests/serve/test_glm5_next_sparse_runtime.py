"""Opt-in native long-context, cache and MTP regression.

Set SUROGATE_GLM_INDEXER_ARTIFACT to a converted GLM checkpoint and expose a GPU.
The fixture keeps the first four real layers and the draft head, referencing the original weights
without copying the expert banks. It tests serving mechanics, not model quality.
"""
import os
from pathlib import Path
import subprocess

import pytest


def make_artifact(path, source_path):
    from surogate.serve.artifact.container import Artifact, ArtifactWriter, ResourceSpec, TensorSpec

    with Artifact(source_path) as source:
        assert source.identity.architecture == 'glm5_next'
        linear = source.layer_types.index('linear_attention')
        attention = source.layer_types.index('full_attention')
        assert linear < source.geometry['leading_dense_layers'] <= attention
        geometry = dict(source.geometry, layers=attention + 1, max_context=4096)
        entries = []
        for obj in source.objects:
            name = obj.name
            if name.startswith('text/layers/'):
                layer = int(name.split('/')[2])
                if layer > attention: continue
            if obj.kind == 'resource': spec = ResourceSpec(name, obj.encoding, obj.bytes)
            else:
                spec = TensorSpec(name, obj.shape, obj.format, obj.layout,
                                  obj.runs, obj.transform, obj.group_map, obj.segments)
            entries.append((spec, obj))
        with ArtifactWriter(path, source.identity, [spec for spec, _ in entries],
                            external=source.external, geometry=geometry,
                            layer_types=source.layer_types[:attention + 1]) as writer:
            for spec, obj in entries:
                if not getattr(spec, 'runs', ()): writer.write(spec.name, bytes(source.payload(obj)))
    return path


@pytest.mark.skipif(not os.environ.get('SUROGATE_GLM_INDEXER_ARTIFACT'), reason='opt-in native GPU regression')
@pytest.mark.parametrize('mtp,kv_dtype', [(False, 'bf16'), (False, 'fp8'), (True, 'bf16'), (True, 'fp8')])
def test_long_context_cache_and_mtp(tmp_path, mtp, kv_dtype):
    root = Path(__file__).resolve().parents[2]
    binary = root / 'csrc/build-serve/serve_tests/sinfer_multi_device_test'
    if not binary.exists(): pytest.skip('build serve-tests first')
    artifact = make_artifact(tmp_path / 'glm.sinfer', os.environ['SUROGATE_GLM_INDEXER_ARTIFACT'])
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               SUROGATE_MULTI_DEVICE_TEST_ARTIFACT=str(artifact),
               SUROGATE_MULTI_DEVICE_TEST_DEVICES='0:0',
               SUROGATE_MULTI_DEVICE_TEST_CONTEXT='4096', SUROGATE_MULTI_DEVICE_TEST_SINGLE_ONLY='1', SUROGATE_MULTI_DEVICE_TEST_PROMPT_TOKENS='2201',
               SUROGATE_MULTI_DEVICE_TEST_CACHE='1', SUROGATE_MULTI_DEVICE_TEST_LOGPROBS='1',
               SUROGATE_MULTI_DEVICE_TEST_KV_DTYPE=kv_dtype)
    if mtp: env['SUROGATE_MULTI_DEVICE_TEST_MTP'] = '1'
    else: env.pop('SUROGATE_MULTI_DEVICE_TEST_MTP', None)
    result = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
