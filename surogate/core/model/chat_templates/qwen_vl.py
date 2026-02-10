from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, List

from pyparsing import Any
import torch

from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.chat_templates.inputs import StdChatTemplateInputs
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.core.model.chat_templates.qwen import QwenChatTemplate
from surogate.utils.env import get_env_args
from .base import Word, register_chat_template
from .chatml import ChatmlChatTemplate, DEFAULT_SYSTEM
from .utils import findall, get_packed_seq_params

class Qwen2VLTemplate(ChatTemplateProcessor):
    image_token_id = 151655
    video_token_id = 151656
    placeholder_tokens = ['<|image_pad|>', '<|video_pad|>']
    version = 'v2'
    use_model = True
    support_padding_free = True

    def _extend_tokens(
            self,
            input_ids: List[int],
            labels: Optional[List[int]],
            loss_scale: Optional[List[float]],
            idx_list: List[int],
            get_new_tokens,
    ) -> tuple[List[int], Optional[List[int]], Optional[List[float]]]:
        """Replace placeholder tokens at idx_list with expanded multimodal tokens."""
        if not idx_list:
            return input_ids, labels, loss_scale

        input_len = len(input_ids)
        for idx in idx_list:
            if idx < 0 or idx >= input_len:
                raise ValueError("multimodal token index out of range")

        repl_map = {}
        for i, idx in enumerate(idx_list):
            repl = get_new_tokens(i)
            if not isinstance(repl, list):
                repl = [repl]
            repl_map[idx] = repl

        new_input_ids: List[int] = []
        new_labels: Optional[List[int]] = [] if labels is not None else None
        new_loss_scale: Optional[List[float]] = [] if loss_scale is not None else None

        for i, tok in enumerate(input_ids):
            repl = repl_map.get(i)
            if repl is None:
                new_input_ids.append(tok)
                if new_labels is not None:
                    new_labels.append(labels[i])
                if new_loss_scale is not None:
                    new_loss_scale.append(loss_scale[i])
                continue

            new_input_ids.extend(repl)
            if new_labels is not None:
                new_labels.extend([-100] * len(repl))
            if new_loss_scale is not None:
                new_loss_scale.extend([0.0] * len(repl))

        return new_input_ids, new_labels, new_loss_scale

    def init_env_args(self):
        super().init_env_args()
        self.bbox_format = get_env_args('QWENVL_BBOX_FORMAT', str, 'legacy')

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdChatTemplateInputs) -> List[Word]:
        from qwen_vl_utils import fetch_image, fetch_video
        assert media_type in {'image', 'video'}
        kwargs = {'image_patch_size': self.processor.image_processor.patch_size} if self.version == 'v3' else {}
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]}, **kwargs)
            if self.mode == 'lmdeploy':
                return ['<|vision_start|>', [-100], '<|vision_end|>']
            else:
                return ['<|vision_start|><|image_pad|><|vision_end|>']
        else:
            if self.version == 'v3':
                kwargs['return_video_metadata'] = True
            video = inputs.videos[index]
            video_inputs = {'video': video}
            if isinstance(video, list):  # image list
                from qwen_vl_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
            video, video_kwargs = fetch_video(video_inputs, return_video_sample_fps=True, **kwargs)
            tokens = ['<|vision_start|><|video_pad|><|vision_end|>']
            if self.version == 'v2_5':
                inputs.mm_processor_kwargs.setdefault('fps', []).append(video_kwargs)
            elif self.version == 'v3':
                if self.mode != 'vllm':
                    video, video_metadata = video
                    inputs.mm_processor_kwargs.setdefault('video_metadata', []).append(video_metadata)
                    tokens = ['<|video_pad|>']
                inputs.mm_processor_kwargs['do_sample_frames'] = False
            if isinstance(video, torch.Tensor):
                video = video.to(torch.uint8)
            inputs.videos[index] = video
            return tokens

    def replace_ref(self, ref: str, index: int, inputs: StdChatTemplateInputs) -> List[Word]:
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdChatTemplateInputs) -> List[Word]:
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def _encode(self, inputs: StdChatTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    kwargs = {}
                    if hasattr(processor, 'video_processor'):
                        processor_func = processor.video_processor
                    else:
                        processor_func = processor.image_processor
                        kwargs['images'] = None
                    media_inputs = processor_func(videos=mm_data, return_tensors='pt', do_resize=False, **kwargs)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        fps = inputs.mm_processor_kwargs['fps']
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / tmp for tmp in fps
                        ]
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    return [media_token] * token_len

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        base_model = self.get_base_model(model)
        if hasattr(base_model.model, 'embed_tokens'):
            inputs_embeds = base_model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = base_model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.visual, self.processor, model.config)
        return {'inputs_embeds': inputs_embeds}

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)
        second_per_grid_ts = self.gather_list(batch, 'second_per_grid_ts')
        if second_per_grid_ts:
            res['second_per_grid_ts'] = second_per_grid_ts
        return res

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        for r in row:
            r_copy = r.copy()
            r_copy['input_ids'] = torch.tensor(r_copy['input_ids'])[None]
            r['position_ids'] = self._get_position_ids(r_copy)
        packed = super().packing_row(row)
        return packed

    def _get_position_ids(self, inputs: Dict[str, Any]):
        # fix https://github.com/huggingface/transformers/pull/33487
        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': inputs.get('second_per_grid_ts')}
        base_model = self.get_base_model(self._get_model())
        if hasattr(base_model, 'get_rope_index'):
            get_rope_index = base_model.get_rope_index
        elif hasattr(base_model, 'model') and hasattr(base_model.model, 'get_rope_index'):
            get_rope_index = base_model.model.get_rope_index
        else:
            get_rope_index = lambda *args, **kw: self._compute_qwen3_vl_rope_index(base_model, *args, **kw)
        attention_mask = inputs.get('attention_mask_2d')
        if attention_mask is None:
            attention_mask = inputs.get('attention_mask')
        position_ids, _ = get_rope_index(
            inputs['input_ids'],
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask=attention_mask,
            **kwargs)
        return self._concat_text_position_ids(position_ids)

    @staticmethod
    def _compute_qwen3_vl_rope_index(
            base_model,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # Fallback for Qwen3-VL when HF get_rope_index is unavailable.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1

        config = getattr(base_model, 'config', None)
        if config is None and hasattr(base_model, 'model'):
            config = getattr(base_model.model, 'config', None)
        if config is None:
            raise ValueError('Qwen3-VL rope index fallback requires a model config.')
        vision_cfg = getattr(config, 'vision_config', None)
        spatial_merge_size = getattr(vision_cfg, 'spatial_merge_size', None)
        if spatial_merge_size is None:
            spatial_merge_size = getattr(config, 'spatial_merge_size', 1)

        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id

        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids_row in enumerate(total_input_ids):
                input_ids_row = input_ids_row[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids_row == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids_row[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids_row.tolist()
                llm_pos_ids_list: List[torch.Tensor] = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            if text_position_ids.shape[0] == 1:
                # https://github.com/huggingface/transformers/pull/40194
                res.update(get_packed_seq_params(text_position_ids))
        return res
    
class Qwen3VLTemplate(Qwen2VLTemplate):
    version = 'v3'

    def _encode(self, inputs: StdChatTemplateInputs) -> Dict[str, Any]:
        encoded = ChatTemplateProcessor._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        for media_type in ['images', 'videos']:
            mm_data = getattr(inputs, media_type)
            if mm_data:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(images=mm_data, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    split_token = self._tokenize('\n')[0]
                    media_inputs = processor(
                        text=['\n'.join(['<|vision_start|><|video_pad|><|vision_end|>'] * len(mm_data))],
                        videos=mm_data,
                        return_tensors='pt',
                        do_resize=False,
                        **inputs.mm_processor_kwargs)
                    splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_inputs.pop('input_ids', None)
                    media_inputs.pop('attention_mask', None)
                    media_token = self.video_token_id
                idx_list = findall(input_ids, media_token)
                merge_length = processor.image_processor.merge_size**2

                def _get_new_tokens(i):
                    if media_type == 'images':
                        token_len = (media_grid_thw[i].prod() // merge_length)
                        return [media_token] * token_len
                    else:
                        return splited_tokens[i]

                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                    _get_new_tokens)
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


register_chat_template(
    QwenChatTemplate(
        ChatTemplateType.qwen3_vl, template_processor_cls=Qwen3VLTemplate, default_system=None, thinking_prefix='<think>\n'))
