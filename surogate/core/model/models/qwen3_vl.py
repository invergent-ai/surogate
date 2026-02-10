from types import MethodType
from typing import Optional, Union

import torch
from PIL import Image

from surogate.core.config.enums import ChatTemplateType
from surogate.core.model.models.qwen25_vl import compat_qwen_vl_utils, get_model_tokenizer_qwen2_vl
from surogate.core.model.registry import register_model, ModelTemplate, MLLMModelType
from surogate.utils.tensor import to_device

def _forward_qwen3_vl_or_qwen3_omni(
        self,
        processor,
        input_ids,
        inputs_embeds,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
):
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    dtype = self.visual.dtype
    if pixel_values is None and pixel_values_videos is None:  # plain-text
        images = [Image.new('RGB', (32, 32), (0, 0, 0))]
        media_inputs = processor.image_processor(images=images, return_tensors='pt')
        media_inputs = to_device(media_inputs, input_ids.device)
        pixel_values = media_inputs['pixel_values'].type(dtype)
        image_embeds, deepstack_visual_embeds = self.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
        inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
        visual_pos_masks = None
    else:
        if pixel_values is None:
            pixel_values_mixed = pixel_values_videos
            grid_thw = video_grid_thw
        elif pixel_values_videos is None:
            pixel_values_mixed = pixel_values
            grid_thw = image_grid_thw
        else:
            pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
            grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
        pixel_values_mixed = pixel_values_mixed.type(dtype)
        mixed_embeds, deepstack_visual_embeds = self.visual(pixel_values_mixed, grid_thw=grid_thw)
        if pixel_values is None:
            image_embeds = None
            video_embeds = mixed_embeds
        elif pixel_values_videos is None:
            image_embeds = mixed_embeds
            video_embeds = None
        else:
            merge_length = processor.image_processor.merge_size**2
            image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
            image_embeds = mixed_embeds[:image_tokens]
            video_embeds = mixed_embeds[image_tokens:]

        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        if image_embeds is not None:
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None:
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        image_mask, video_mask = image_mask[..., 0], video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        if image_embeds is not None and video_embeds is not None:
            deepstack_image_embeds = [tensor[:image_tokens] for tensor in deepstack_visual_embeds]
            deepstack_video_embeds = [tensor[image_tokens:] for tensor in deepstack_visual_embeds]
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

def _compat_qwen3_vl_mixed_data(model, processor, is_moe: bool = False):
    if hasattr(model, 'origin_forward'):
        return
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLModelOutputWithPast, TransformersKwargs, Unpack,
                                                                check_model_inputs, Cache, is_torchdynamo_compiling)
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeModelOutputWithPast
    output_cls = Qwen3VLMoeModelOutputWithPast if is_moe else Qwen3VLModelOutputWithPast

    check_model_inputs = check_model_inputs()

    @check_model_inputs
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, output_cls]:
        if not self.training:
            return self.origin_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError('You must specify exactly one of input_ids or inputs_embeds')

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = _forward_qwen3_vl_or_qwen3_omni(
            self, processor, input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw,
            video_grid_thw)
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask['full_attention'])
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                    (input_ids is not None and input_ids.shape[1] != 1) or
                    (inputs_embeds is not None and inputs_embeds.shape[1] != 1))
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                    (cache_position is not None and cache_position[0] == 0) or
                    (past_key_values is None or past_key_values.get_seq_length() == 0))
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = ((cache_position[0]
                          + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return output_cls(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    model.origin_forward = model.forward
    model.forward = MethodType(forward, model)

def get_model_tokenizer_qwen3_vl(model_dir, *args, **kwargs):
    from transformers import Qwen3VLForConditionalGeneration
    compat_qwen_vl_utils(image_patch_size=16)
    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3VLForConditionalGeneration
    kwargs['_check_qwen_vl_utils'] = False
    model, processor = get_model_tokenizer_qwen2_vl(model_dir, *args, **kwargs)
    if model is not None:
        _compat_qwen3_vl_mixed_data(model.model, processor)
    return model, processor


"""
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
"""
register_model(
    ModelTemplate(
        MLLMModelType.qwen3_vl,
        ChatTemplateType.qwen3_vl,
        get_model_tokenizer_qwen3_vl,
        architectures=['Qwen3VLForConditionalGeneration'],
        tags=['vision', 'video']))