import os

from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.chat_templates.vision_utils import load_file
from surogate.core.model.patcher import patch_get_input_embeddings
from surogate.core.model.registry import register_model, ModelTemplate, MLLMModelType
from surogate.core.model.loader import get_model_tokenizer_multimodal
from surogate.utils.env import get_env_args


def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return
    if os.getenv('VIDEO_MAX_PIXELS') and not os.getenv('VIDEO_TOTAL_PIXELS'):
        # https://github.com/QwenLM/Qwen2.5-VL/issues/1120
        os.environ['VIDEO_TOTAL_PIXELS'] = str(int(128000 * 28 * 28 * 0.9))
    res = {}
    for key in [
        'image_factor',  # image_patch_size * SPATIAL_MERGE_SIZE
        'min_pixels',  # IMAGE_MIN_TOKEN_NUM * image_factor ** 2
        'max_pixels',
        'video_min_pixels',
        'video_max_pixels',
        'video_total_pixels',
        #
        'max_ratio',
        'frame_factor',
        'fps',
        'fps_min_frames',
        'fps_max_frames',
        # qwen3_vl
        'image_max_token_num',
        'image_min_token_num',
        'spatial_merge_size',
        'video_max_token_num',
        'video_min_token_num',
    ]:
        type_func = float if key == 'fps' else int
        default_value = getattr(vision_process, key.upper(), None)
        if default_value is None:
            # Skip keys not supported by the specific vision_process implementation
            continue
        val = get_env_args(key, type_func, default_value)
        setattr(vision_process, key.upper(), val)
        res[key] = val
    # Patch decord video reader if available
    _read_video_decord = getattr(vision_process, '_read_video_decord', None)
    if _read_video_decord is not None:

        def _new_read_video_decord(ele: dict):
            ele['video'] = load_file(ele['video'])
            return _read_video_decord(ele)

        backends = getattr(vision_process, 'VIDEO_READER_BACKENDS', None)
        if isinstance(backends, dict):
            backends['decord'] = _new_read_video_decord
        elif backends is None:  # keye_vl
            vision_process._read_video_decord = _new_read_video_decord
    vision_process._patch = True
    return res


def compat_qwen_vl_utils(image_patch_size: int):
    spatial_merge_size = int(os.getenv('SPATIAL_MERGE_SIZE', '2'))
    image_factor = image_patch_size * spatial_merge_size
    env_vars_to_process = {
        'MAX_PIXELS': 'IMAGE_MAX_TOKEN_NUM',
        'MIN_PIXELS': 'IMAGE_MIN_TOKEN_NUM',
        'VIDEO_MAX_PIXELS': 'VIDEO_MAX_TOKEN_NUM',
        'VIDEO_MIN_PIXELS': 'VIDEO_MIN_TOKEN_NUM',
    }
    for source_var, target_var in env_vars_to_process.items():
        value = os.getenv(source_var)
        if value and not os.getenv(target_var):
            os.environ[target_var] = str(int(value) // image_factor ** 2)


def get_model_tokenizer_qwen2_vl(*args, **kwargs):
    from transformers import Qwen2VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None:
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        patch_get_input_embeddings(base_model.visual, 'patch_embed')

    from qwen_vl_utils import vision_process
    check_qwen_vl_utils = kwargs.get('_check_qwen_vl_utils', True)
    if check_qwen_vl_utils:
        compat_qwen_vl_utils(image_patch_size=14)

    global_vars = patch_qwen_vl_utils(vision_process)
    tokenizer.global_vars = global_vars  # In order to have different hashes for the template.
    return model, tokenizer


def get_model_tokenizer_qwen2_5_vl(*args, **kwargs):
    from transformers import Qwen2_5_VLForConditionalGeneration
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen2_5_VLForConditionalGeneration
    return get_model_tokenizer_qwen2_vl(*args, **kwargs)


"""
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
"""
register_model(
    ModelTemplate(
        MLLMModelType.qwen2_5_vl,
        ChatTemplateType.qwen2_5_vl,
        get_model_tokenizer_qwen2_5_vl,
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        tags=['vision', 'video']))
