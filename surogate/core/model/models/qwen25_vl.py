import importlib
import os

from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.chat_templates.vision_utils import load_file
from surogate.core.model.patcher import patch_get_input_embeddings
from surogate.core.model.registry import ModelLoader, register_model, ModelTemplate
from surogate.core.model.utils import Processor
from surogate.utils.env import get_env_args
from transformers import PretrainedConfig, PreTrainedModel
from packaging import version
from transformers.utils.versions import require_version

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


class Qwen2VLLoader(ModelLoader):
    def get_model(self, model_dir: str, config, processor, model_kwargs) -> PreTrainedModel:
        from transformers import Qwen2VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen2VLForConditionalGeneration
        model = super().get_model(model_dir, config, processor, model_kwargs)
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        patch_get_input_embeddings(base_model.visual, 'patch_embed')
        return model
    
    def _check_qwen_vl_utils(self):
        try:
            qwen_vl_utils_version = importlib.metadata.version('qwen_vl_utils')
        except importlib.metadata.PackageNotFoundError:
            raise importlib.metadata.PackageNotFoundError(
                "The 'qwen_vl_utils' distribution was not found and is required by this application.")
        if version.parse(qwen_vl_utils_version) >= version.parse('0.0.14'):
            compat_qwen_vl_utils(image_patch_size=14)
        else:
            require_version('qwen_vl_utils<0.0.12')
            
    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        self._check_qwen_vl_utils()
        from qwen_vl_utils import vision_process
        processor = super().get_processor(model_dir, config)
        global_vars = patch_qwen_vl_utils(vision_process)
        processor.global_vars = global_vars  # In order to have different hashes for the template.
        return processor

class Qwen2_5VLLoader(Qwen2VLLoader):

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.auto_model_cls = self.auto_model_cls or Qwen2_5_VLForConditionalGeneration
        return super().get_model(model_dir, *args, **kwargs)
    
"""
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
"""
register_model(
    ModelTemplate(
        model_type='Qwen2_5_VLForConditionalGeneration',
        chat_templates=[ChatTemplateType.qwen2_5_vl],
        loader=Qwen2_5VLLoader,
        is_multimodal=True,
        tags=['vision', 'video']))
