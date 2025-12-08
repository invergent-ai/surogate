import base64
import math
import os
import re
from io import BytesIO
from typing import Union, TypeVar, List, Callable, Any

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from surogate.utils.env import get_env_args

_T = TypeVar('_T')

def load_image(image: Union[str, bytes, Image.Image]) -> Image.Image:
    image = load_file(image)
    if isinstance(image, BytesIO):
        image = Image.open(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def load_file(path: Union[str, bytes, _T]) -> Union[BytesIO, _T]:
    res = path
    if isinstance(path, str):
        path = path.strip()
        if path.startswith('http'):
            retries = Retry(total=3, backoff_factor=1, allowed_methods=['GET'])
            with requests.Session() as session:
                session.mount('http://', HTTPAdapter(max_retries=retries))
                session.mount('https://', HTTPAdapter(max_retries=retries))

                timeout = float(os.getenv('LOAD_IMAGE_TIMEOUT', '20'))
                request_kwargs = {'timeout': timeout} if timeout > 0 else {}

                response = session.get(path, **request_kwargs)
                response.raise_for_status()
                content = response.content
                res = BytesIO(content)

        elif os.path.exists(path) or (not path.startswith('data:') and len(path) <= 200):
            ROOT_IMAGE_DIR = get_env_args('ROOT_IMAGE_DIR', str, None)
            if ROOT_IMAGE_DIR is not None and not os.path.exists(path):
                path = os.path.join(ROOT_IMAGE_DIR, path)
            path = os.path.abspath(os.path.expanduser(path))
            with open(path, 'rb') as f:
                res = BytesIO(f.read())
        else:  # base64_str
            data = path
            if data.startswith('data:'):
                match_ = re.match(r'data:(.+?);base64,(.+)', data)
                assert match_ is not None
                data = match_.group(2)
            data = base64.b64decode(data)
            res = BytesIO(data)
    elif isinstance(path, bytes):
        res = BytesIO(path)
    return res

def load_batch(path_list: List[Union[str, None, Any, BytesIO]],
               load_func: Callable[[Any], _T] = load_image) -> List[_T]:
    res = []
    assert isinstance(path_list, (list, tuple)), f'path_list: {path_list}'
    for path in path_list:
        if path is None:  # ignore None
            continue
        res.append(load_func(path))
    return res

def load_audio(audio: Union[str, bytes], sampling_rate: int, return_sr: bool = False):
    import librosa
    audio_io = load_file(audio)
    res = librosa.load(audio_io, sr=sampling_rate)
    return res if return_sr else res[0]

def rescale_image(img: Image.Image, max_pixels: int) -> Image.Image:
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return img

    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(img)