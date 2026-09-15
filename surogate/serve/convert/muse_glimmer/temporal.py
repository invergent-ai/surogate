"""Recover the temporal projection omitted by the released image-only GGUF projector."""
from pathlib import Path
import hashlib
import os
import tempfile
import urllib.request
import numpy as np

# The original projection is only 3.45 MiB in a 50 GB safetensors shard. Pin both
# revision and content, and never accept an HTTP server that ignores Range.
_REVISION = 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68'
_URL = f'https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/{_REVISION}/model-00001-of-00002.safetensors?range=muse-temporal-v1'
_START = 8 + 170160 + 49943184384
_SIZE = 1536 * 1176 * 2
_SHA256 = 'b00e07d9ccf9be48df03586b0db521dbc8e030600474b52963bc783724b4218a'


def temporal_projection(folded):
    if folded.shape != (1536, 588):
        raise ValueError('Muse video requires the matching original temporal patch projection')
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'surogate' / 'serve'
    path = cache / f'muse-temporal-{_SHA256}.bf16'
    data = path.read_bytes() if path.is_file() else b''
    if len(data) != _SIZE or hashlib.sha256(data).hexdigest() != _SHA256:
        request = urllib.request.Request(_URL, headers={'Range': f'bytes={_START}-{_START + _SIZE - 1}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range', '').split('/')[0] != f'bytes {_START}-{_START + _SIZE - 1}':
                raise ValueError('server did not return the requested Muse temporal weight range')
            data = response.read(_SIZE + 1)
        if len(data) != _SIZE or hashlib.sha256(data).hexdigest() != _SHA256:
            raise ValueError('Muse temporal projection checksum mismatch')
        cache.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='muse-temporal-', suffix='.partial', dir=cache)
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(data)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
    values = (np.frombuffer(data, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(1536, 2, 588)
    if not np.array_equal(values.sum(axis=1), folded):
        raise ValueError('GGUF projector does not match the published Muse temporal weights; refusing to mix checkpoints')
    return data
