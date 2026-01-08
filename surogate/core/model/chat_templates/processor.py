import hashlib
import inspect
import os
import re
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Optional, Literal, List, Dict, Any, Union, Tuple

import torch
from PIL import Image
from peft import PeftModel
from torch import nn
from transformers import PreTrainedTokenizerBase

from surogate.core.datasets.preprocessor.row import MaxLengthError, RowPreprocessor
from surogate.core.model.agent_templates import agent_templates
from surogate.core.model.agent_templates.base import BaseAgentTemplate
from surogate.core.model.chat_templates.inputs import InferRequest, ChatTemplateInputs, StdChatTemplateInputs
from surogate.core.model.chat_templates.utils import fetch_one, get_last_user_round, split_str_parts_by
from surogate.core.model.chat_templates.vision_utils import load_image, rescale_image, load_batch, load_audio
from surogate.core.model.loss_scale.loss_scale import LossScale, get_loss_scale
from surogate.core.model.utils import Processor, Context, ContextType
from surogate.utils.env import get_env_args
from surogate.utils.fs import get_cache_dir
from surogate.utils.logger import get_logger
from surogate.utils.tensor import to_device

logger = get_logger()


class ChatTemplateProcessor:
    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>', '<cot-process>', '<start-image>']
    special_keys = ['images', 'videos', 'audios', 'objects']

    image_placeholder = ['<image>']
    video_placeholder = ['<video>']
    audio_placeholder = ['<audio>']
    cot_process_placeholder = ['ки']
    placeholder_tokens = []  # For clearer printing
    load_images = True
    skip_prompt = True
    use_model = False
    norm_bbox = 'norm1000'
    support_padding_free = False  # It only takes effect for multimodal models.

    is_encoder_decoder = False

    agent_template: BaseAgentTemplate = None
    chat_template: 'ChatTemplate' = None

    def __init__(
            self,
            processor: Optional['Processor'],
            chat_template: 'ChatTemplate',
            default_system: Optional[str] = None,
            max_length: Optional[int] = None,
            *,
            truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
            max_pixels: Optional[int] = None,
            agent_template: Optional[str] = None,
            norm_bbox: Literal['norm1000', 'none', None] = None,
            use_chat_template: bool = True,
            # only for train
            padding_free: bool = False,
            padding_side: Literal['left', 'right'] = 'right',
            loss_scale: str = 'default',
            sequence_parallel_size: int = 1,
            # infer/deploy
            response_prefix: Optional[str] = None,
            enable_thinking: Optional[bool] = None,
            add_non_thinking_prefix: bool = True,
    ) -> None:
        self._processor_inited = False
        self.max_length = max_length
        self.model = None
        self.dummy_model = None
        self.template_backend = 'native'

        if not use_chat_template:
            chat_template = chat_template.to_generate_chat_template()
        else:
            chat_template = deepcopy(chat_template)

        chat_template.check_system(default_system)

        if default_system is not None:
            chat_template.default_system = default_system
        if enable_thinking is None:
            enable_thinking = chat_template.is_thinking
        if response_prefix is None:
            if use_chat_template:
                response_prefix = (
                    chat_template.thinking_prefix if enable_thinking else chat_template.non_thinking_prefix)
            else:
                response_prefix = ''

        self.response_prefix = response_prefix
        self.chat_template = chat_template
        self.use_chat_template = use_chat_template
        self.enable_thinking = enable_thinking
        self.add_non_thinking_prefix = add_non_thinking_prefix
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.loss_scale: 'LossScale' = get_loss_scale(loss_scale)
        self.max_pixels = max_pixels
        self.padding_side = padding_side
        self.sequence_parallel_size = sequence_parallel_size
        self.padding_free = padding_free  # padding_free/packing
        self.packing = False
        agent_template = agent_template or chat_template.agent_template
        self._agent_template = agent_template
        self.agent_template = agent_templates[agent_template]()
        self.norm_bbox = norm_bbox or self.norm_bbox
        if self.is_encoder_decoder:
            self.skip_prompt = False
        self.mode: Literal['pt', 'vllm', 'sglang',  'train'] = 'pt'
        self._handles = []

        if processor is not None:
            self.init_processor(processor)

    def init_processor(self, processor: 'Processor') -> None:
        if processor is None or self._processor_inited:
            return
        self._processor_inited = True
        self.processor = processor
        self.model_info = processor.model_info
        self.config = self.model_info.config
        self.model_template = processor.model_template
        if self.max_length is None:
            self.max_length = self.model_info.max_model_len
        tokenizer = self.tokenizer

        for i, token in enumerate(self.placeholder_tokens):
            if isinstance(token, str):
                self.placeholder_tokens[i] = tokenizer.convert_tokens_to_ids(token)
        self.chat_template.init(tokenizer)
        self.init_env_args()

    def init_env_args(self):
        if self.model_template.is_multimodal:
            self.root_image_dir = get_env_args('ROOT_IMAGE_DIR', str, None)
        else:
            self.root_image_dir = None

    @property
    def tokenizer(self):
        tokenizer = self.processor
        if not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self.processor = value

    @torch.inference_mode()
    def encode(self,
               inputs: Union[Dict[str, Any], InferRequest],
               return_template_inputs: bool = False,
               return_length: bool = False) -> Dict[str, Any]:
        """
        Returns:
            return {'input_ids': List[int], 'labels': Optional[List[int]], ...}
        """
        assert self._processor_inited, ('Please initialize the processor before calling the template.encode method: '
                                        'template.init_processor(processor).')
        if isinstance(inputs, dict):
            if not self.is_training:
                InferRequest.remove_response(inputs['messages'])
            inputs = ChatTemplateInputs.from_dict(inputs)
        elif isinstance(inputs, ChatTemplateInputs):
            inputs = deepcopy(inputs)
        assert isinstance(inputs, ChatTemplateInputs)

        chosen = inputs.chosen
        encoded = self._encode_truncated(chosen)

        batched = encoded
        if not isinstance(batched, (list, tuple)):
            batched = [batched]
        for encoded in batched:
            if chosen.channel is not None:
                encoded['channel'] = chosen.channel

            lengths = []
            for key in list(encoded.keys()):
                if encoded[key] is None:
                    encoded.pop(key)
                elif key.endswith('length'):
                    value = encoded[key]
                    if isinstance(value, int):
                        lengths.append(value)
                    elif isinstance(value, (tuple, list)):
                        lengths += value
            if return_length:
                if not lengths:
                    raise ValueError(f'lengths should not be empty. batched: {batched}')
                encoded['length'] = lengths[0] if len(lengths) == 1 else lengths
            else:
                encoded.pop('length', None)

            if return_template_inputs:
                encoded['template_inputs'] = chosen

            encoded['_extra_kwargs'] = chosen.extra_kwargs

        return batched[0] if len(batched) == 1 else batched

    @torch.inference_mode()
    def encode_batch(self,
                     inputs_list: List[Union[Dict[str, Any], InferRequest]],
                     return_template_inputs: bool = False,
                     return_length: bool = False) -> List[Dict[str, Any]]:
        """Encode a batch of inputs using batched tokenization for better performance.

        This method processes multiple inputs simultaneously, which is significantly faster
        than calling encode() repeatedly because:
        1. Tokenizer batch operations are highly optimized (10-50x faster)
        2. Reduces Python interpreter overhead
        3. Better CPU cache utilization

        Args:
            inputs_list: List of input dictionaries or InferRequest objects
            return_template_inputs: Whether to include template_inputs in output
            return_length: Whether to include length in output

        Returns:
            List of encoded dictionaries, one per input
        """
        assert self._processor_inited, ('Please initialize the processor before calling the template.encode_batch method: '
                                        'template.init_processor(processor).')

        # Normalize inputs to ChatTemplateInputs
        normalized_inputs = []
        for inputs in inputs_list:
            if isinstance(inputs, dict):
                if not self.is_training:
                    InferRequest.remove_response(inputs['messages'])
                inputs = ChatTemplateInputs.from_dict(inputs)
            elif isinstance(inputs, ChatTemplateInputs):
                inputs = deepcopy(inputs)
            assert isinstance(inputs, ChatTemplateInputs)
            normalized_inputs.append(inputs)

        # Process all inputs - collect context lists for batched tokenization
        results = []
        context_lists_batch = []
        loss_scale_lists_batch = []
        metadata_batch = []

        for inputs in normalized_inputs:
            chosen = inputs.chosen
            # Prepare inputs (image loading, preprocessing, etc.)
            self._preprocess_inputs(chosen)

            # Get context list and loss scales (string concatenation phase)
            if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
                encoded = ChatTemplateProcessor._encode(self, chosen)
                keys = ['images', 'audios', 'videos']
                if self.mode == 'vllm':
                    keys.append('mm_processor_kwargs')
                for key in keys:
                    value = getattr(chosen, key)
                    if value:
                        encoded[key] = value
                # For these modes, encode returns complete results
                results.append((encoded, chosen))
                context_lists_batch.append(None)
                loss_scale_lists_batch.append(None)
                metadata_batch.append(None)
            else:
                # Get context list (this does string operations but not tokenization yet)
                chosen_copy = deepcopy(chosen)
                chosen_copy.messages = deepcopy(chosen_copy.messages)
                self._native_prepare_inputs(chosen_copy)

                template_backend = self.template_backend
                if (self.chat_template.template_type == 'dummy' and self.use_chat_template and not self.is_training):
                    template_backend = 'jinja'

                res_context_list, loss_scale_list, answer_len = (
                    self._native_encode(chosen_copy) if template_backend == 'native' else self._jinja_encode(chosen_copy))

                # Simplify context lists (merge strings, handle special tokens)
                res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, chosen_copy)

                context_lists_batch.append(res_context_list)
                loss_scale_lists_batch.append(loss_scale_list)
                metadata_batch.append({
                    'chosen': chosen,
                    'answer_len': answer_len,
                    'suffix_tokens': self._encode_context_list(self.chat_template.suffix)[0]
                })
                results.append(None)

        # Batched tokenization: collect all strings that need tokenization
        strings_to_tokenize = []
        string_indices = []  # (result_idx, context_idx)

        for result_idx, context_list in enumerate(context_lists_batch):
            if context_list is None:
                continue
            for context_idx, context in enumerate(context_list):
                if isinstance(context, str):
                    strings_to_tokenize.append(context)
                    string_indices.append((result_idx, context_idx))

        # Perform batched tokenization on all strings at once
        if strings_to_tokenize:
            tokenized_batch = self._tokenize_batch(strings_to_tokenize)

            # Distribute tokenized results back to context lists
            for tokenized_ids, (result_idx, context_idx) in zip(tokenized_batch, string_indices):
                context_lists_batch[result_idx][context_idx] = tokenized_ids

        # Now encode each context list (this is fast now that tokenization is done)
        final_results = []
        for result_idx, (result, context_list, loss_scale_list, metadata) in enumerate(
                zip(results, context_lists_batch, loss_scale_lists_batch, metadata_batch)):

            if result is not None:
                # Already encoded (vllm/lmdeploy/sglang mode)
                encoded, chosen = result
            else:
                # Encode from tokenized context list
                input_ids, labels, loss_scale = self._encode_context_list(context_list, loss_scale_list)
                self._add_dynamic_eos(input_ids, labels, loss_scale, metadata['suffix_tokens'])

                encoded = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'loss_scale': loss_scale
                }
                if encoded.get('labels') is not None:
                    encoded['labels'][0] = -100
                if encoded.get('loss_scale') is not None:
                    encoded['loss_scale'][0] = 0
                if not self.is_training:
                    for k in list(encoded.keys()):
                        if k.endswith('labels') or k.endswith('loss_scale'):
                            encoded[k] = None

                chosen = metadata['chosen']

            # Apply truncation if needed
            input_ids = encoded.get('input_ids')
            labels = encoded.get('labels')
            loss_scale_val = encoded.get('loss_scale')
            length = self._get_length(input_ids, labels)

            if self.max_length is not None and length > self.max_length:
                if self.truncation_strategy in {'right', 'left'}:
                    input_ids, labels, loss_scale_val = self._truncate(
                        input_ids, labels, loss_scale_val, truncation_strategy=self.truncation_strategy)
                    length = self._get_length(input_ids, labels)
                elif self.truncation_strategy == 'raise':
                    from surogate.core.datasets.preprocessor.row import MaxLengthError
                    raise MaxLengthError(f'Current length of row({length}) is larger'
                                         f' than the max_length({self.max_length}).')

            encoded['length'] = length
            encoded['input_ids'] = input_ids
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale_val

            # Post-process like in encode()
            if chosen.channel is not None:
                encoded['channel'] = chosen.channel

            lengths = []
            for key in list(encoded.keys()):
                if encoded[key] is None:
                    encoded.pop(key)
                elif key.endswith('length'):
                    value = encoded[key]
                    if isinstance(value, int):
                        lengths.append(value)
                    elif isinstance(value, (tuple, list)):
                        lengths += value

            if return_length:
                if not lengths:
                    raise ValueError(f'lengths should not be empty. encoded: {encoded}')
                encoded['length'] = lengths[0] if len(lengths) == 1 else lengths
            else:
                encoded.pop('length', None)

            if return_template_inputs:
                encoded['template_inputs'] = chosen

            encoded['_extra_kwargs'] = chosen.extra_kwargs

            final_results.append(encoded)

        return final_results
 

    def _encode_truncated(self, inputs: StdChatTemplateInputs):
        self._preprocess_inputs(inputs)
        if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
            # For multi-modal models, images do not need to be pre processed here
            # vllm/lmdeploy/sglang will handle the logic
            encoded = ChatTemplateProcessor._encode(self, inputs)
            keys = ['images', 'audios', 'videos']
            if self.mode == 'vllm':
                keys.append('mm_processor_kwargs')
            for key in keys:
                value = getattr(inputs, key)
                if value:
                    encoded[key] = value
        else:
            encoded = self._encode(inputs)

        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        loss_scale = encoded.get('loss_scale')
        length = self._get_length(input_ids, labels)
        if self.max_length is not None and length > self.max_length:
            if self.truncation_strategy in {'right', 'left'}:
                input_ids, labels, loss_scale = self._truncate(
                    input_ids, labels, loss_scale, truncation_strategy=self.truncation_strategy)
                length = self._get_length(input_ids, labels)
            elif self.truncation_strategy == 'raise':
                raise MaxLengthError(f'Current length of row({length}) is larger'
                                     f' than the max_length({self.max_length}).')
            elif self.truncation_strategy == 'split':
                i = 0
                batched = []
                while i < length:
                    splited = {}
                    for key in ['input_ids', 'labels', 'loss_scale']:
                        value = encoded.get(key)
                        if value is not None:
                            value = value[i:i + self.max_length]
                            if key == 'labels' and len(value) > 0:
                                value[0] = -100
                            elif key == 'loss_scale' and len(value) > 0:
                                value[0] = 0
                        splited[key] = value
                    splited['length'] = self._get_length(splited.get('input_ids'), splited.get('labels'))
                    batched.append(splited)
                    i += self.max_length
                return batched
            else:
                raise ValueError(f'Invalid truncation_strategy: {self.truncation_strategy}')
            
        encoded['length'] = length
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _preprocess_inputs(self, inputs: StdChatTemplateInputs, ) -> None:
        self._preprocess_function_call(inputs)
        if self.model_template.is_multimodal:
            self._replace_image_tags(inputs)
            self._replace_start_image_tags(inputs)

        images = inputs.images
        load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}
        load_images_origin = load_images
        if self.max_pixels is not None or inputs.objects:
            load_images = True
        if images:
            for i, image in enumerate(images):
                images[i] = self._load_image(images[i], load_images)
        if inputs.objects:
            self._get_height_width(inputs)
        if self.max_pixels is not None:
            # Scale the image proportionally without affecting the scaled objects.
            images = [rescale_image(img, self.max_pixels) for img in images]
        if images and not load_images_origin:  # fix pt & qwen-vl
            for i, image in enumerate(images):
                if isinstance(image, Image.Image):
                    images[i] = self._save_pil_image(image)
        inputs.images = images

        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))
        if inputs.is_multimodal:
            self._add_default_tags(inputs)

    def _preprocess_function_call(self, inputs: StdChatTemplateInputs) -> None:
        agent_template = self.agent_template
        agent_template.chat_template = self.chat_template
        if inputs.tools:
            if isinstance(inputs.tools, str):
                inputs.tools = agent_template._parse_json(inputs.tools)
                if not isinstance(inputs.tools, (list, tuple)):
                    inputs.tools = [inputs.tools]
            elif isinstance(inputs.tools, (list, tuple)):
                inputs.tools = [agent_template._parse_json(tool) for tool in inputs.tools]
            else:
                raise ValueError(f'inputs.tools: {inputs.tools}')
            for i, tool in enumerate(inputs.tools):
                inputs.tools[i] = agent_template.wrap_tool(tool)
        i = 0
        messages = inputs.messages
        while i < len(messages):
            if messages[i]['role'] == 'tool_call':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool_call':
                    i += 1
                tool_content = self.agent_template._format_tool_calls(messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'assistant', 'content': tool_content}]
                i = i_start + 1
            else:
                i += 1

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdChatTemplateInputs) -> Tuple[List[Context], List[float]]:
        """This method happens before tokenization, replace standard tags to the contents or input_ids needed by
        the model.

        Args:
            context_list: The content list
            loss_scale_list: The loss scale list
        Returns:
            The context_list and loss_scale_list after replacement.
        """
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)
        # replace tag/object/box
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        # reset
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['video', 'audio']:
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(
                        getattr(inputs, f'{k}s')):
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    loss_scale = 0.
                    break
            else:
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox):
                    idx = inputs.bbox_idx
                    c_list = self.replace_bbox(bbox[idx], idx, inputs)
                    inputs.bbox_idx += 1
                else:
                    c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    def replace_ref(self, ref: str, index: int, inputs: StdChatTemplateInputs) -> List[Context]:
        """Replace objects referenced by the bbox to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            ref: Description of the bbox
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [ref]

    def replace_cot_process(self, inputs: StdChatTemplateInputs) -> List[Context]:
        """Replace the cot process label for PRM training or inference.
        Override this function to do your own replace operation.

        Args:
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [self.cot_process_placeholder]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdChatTemplateInputs) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            bbox: [x, y] or [x1, y1, x2, y2]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [f'[{self._get_bbox_str(bbox)}]']

    def _pre_tokenize_images(self, context_list: List[Context], loss_scale_list: List[float],
                             inputs: StdChatTemplateInputs) -> Tuple[List[Context], List[float]]:
        # Fix the bounding box position offset issue in the Qwen2.5-VL grounding task.
        res: List[Context] = []
        res_loss_scale: List[float] = []
        inputs.image_idx = 0

        for context, loss_scale in zip(context_list, loss_scale_list):
            if context == '<image>' and inputs.is_multimodal and inputs.image_idx < len(inputs.images):
                c_list = self.replace_tag('image', inputs.image_idx, inputs)
                inputs.image_idx += 1
                loss_scale = 0. if self.template_backend == 'native' else 1.
            else:
                c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdChatTemplateInputs) -> Optional[List[Context]]:
        """Override this function to do your own replace operation.

        This method is used to replace standard tags like `<image>` to some tokens that the model needs.

        Args:
            media_type: The modal.
            index: The index of the medias, for index 0 represents the first elements in `images`
            inputs: The inputs

        Returns:
            The content or input_ids after replacement.
        """
        if media_type == 'image':
            if self.mode == 'lmdeploy':
                return [[-100]]
            return self.image_placeholder
        elif media_type == 'video':
            if self.mode == 'vllm':
                # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
                from vllm.assets.video import video_to_ndarrays, video_get_metadata
                num_frames = get_env_args('vllm_num_frames', int, 16)
                video_data = video_to_ndarrays(inputs.videos[index], num_frames)
                video_metadatas = video_get_metadata(inputs.videos[index], num_frames)
                inputs.videos[index] = [(video_data, video_metadatas)]
                return self.video_placeholder
            else:
                return self.video_placeholder
        elif media_type == 'audio':
            return self.audio_placeholder

    def normalize_bbox(self, inputs: StdChatTemplateInputs) -> None:
        objects = inputs.objects
        bbox_list = objects['bbox']
        width_list = objects['width']
        height_list = objects['height']
        bbox_type = objects.pop('bbox_type', None) or 'real'
        image_id_list = objects.pop('image_id', None) or []
        image_id_list += [0] * (len(bbox_list) - len(image_id_list))
        for bbox, image_id in zip(bbox_list, image_id_list):
            if bbox_type == 'norm1':
                width, height = 1, 1
            else:
                width, height = width_list[image_id], height_list[image_id]
            for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):
                if self.norm_bbox == 'norm1000':
                    norm_width, norm_height = 1000, 1000
                elif self.norm_bbox == 'none':
                    image = inputs.images[image_id]
                    norm_width, norm_height = image.width, image.height
                bbox[2 * i] = int(round(x / width * norm_width))
                bbox[2 * i + 1] = int(round(y / height * norm_height))

    def _encode(self, inputs: StdChatTemplateInputs) -> Dict[str, Any]:
        inputs.messages = deepcopy(inputs.messages)
        template_backend = self.template_backend
        if (self.chat_template.template_type == 'dummy' and self.use_chat_template and not self.is_training):
            template_backend = 'jinja'
            logger.info_once(f'Setting template_backend: {template_backend}')
        self._native_prepare_inputs(inputs)
        res_context_list, loss_scale_list, answer_len = (
            self._native_encode(inputs) if template_backend == 'native' else self._jinja_encode(inputs))
        encoded = {}
        if self.is_encoder_decoder or self.mode == 'gkd':
            total_len = len(res_context_list)
            for key, _slice in zip(['prompt', 'answer'],
                                   [slice(0, total_len - answer_len),
                                    slice(total_len - answer_len, total_len)]):
                context_list, loss_scale = self._simplify_context_list(res_context_list[_slice],
                                                                       loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale = self._encode_context_list(context_list, loss_scale)
                encoded[f'{key}_input_ids'] = input_ids
                encoded[f'{key}_labels'] = labels
                encoded[f'{key}_loss_scale'] = loss_scale
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
            labels = encoded['prompt_labels'] + encoded['answer_labels']
            loss_scale = None
            if isinstance(encoded['prompt_loss_scale'], list):
                loss_scale = encoded['prompt_loss_scale'] + encoded['answer_loss_scale']
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale = self._encode_context_list(res_context_list, loss_scale_list)
        self._add_dynamic_eos(input_ids, labels, loss_scale, self._encode_context_list(self.chat_template.suffix)[0])

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if encoded.get('labels') is not None:
            encoded['labels'][0] = -100
        if encoded.get('loss_scale') is not None:
            encoded['loss_scale'][0] = 0
        if not self.is_training:
            for k in list(encoded.keys()):
                if k.endswith('labels') or k.endswith('loss_scale'):
                    encoded[k] = None
        return encoded

    def register_post_encode_hook(self, models: List[nn.Module]) -> None:
        """This function is important for multi-modal training, as it registers the post_encode method
            as a forward hook, converting input_ids into inputs_embeds.
        """
        if self._handles:
            return

        for model in models:
            # please use torch>=2.0
            handle = model.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            self._handles.append((model, handle))

    def remove_post_encode_hook(self):
        models = []
        for model, handle in self._handles:
            models.append(model)
            handle.remove()
        self._handles = []
        return models

    def _jinja_encode(self, inputs: StdChatTemplateInputs):
        messages = inputs.messages.copy()
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        if messages[-1]['content'] is None:
            messages.pop()
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        kwargs = {}
        if inputs.tools:
            kwargs['tools'] = inputs.tools
        if 'thinking_budget' in inputs.extra_kwargs:
            kwargs['thinking_budget'] = inputs.extra_kwargs.get('thinking_budget', 0)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        answer_len = 1 if self.is_training else 0
        return [text], [1.], answer_len

    def _native_encode(self, inputs: StdChatTemplateInputs):
        chat_template = self.chat_template
        if self.use_chat_template:
            if self.add_non_thinking_prefix:
                self._add_non_thinking_prefix(inputs)
            if chat_template.is_thinking or self.enable_thinking:
                self._remove_history_thinking(inputs)
                
        system = self._get_system(inputs)

        self._get_std_messages(inputs.messages)
        n_round = len(inputs.messages) // 2
        if n_round > 1 and not self.chat_template.support_multi_round:
            logger.warning_once(
                'The template does not support multi-round chat. Only use the last round of the conversation.')
            # TODO: Multimodal models may encounter image mismatch issues.
            inputs.messages = inputs.messages[-2:]

        res_context_list: List[Context] = []
        res_context_types: List[ContextType] = []
        sep_token = None
        if chat_template.auto_add_bos:
            all_tokens = self.tokenizer.encode('a')
            single_token = self.tokenizer.encode('a', add_special_tokens=False)
            assert len(single_token) == 1
            idx = all_tokens.index(single_token[0])
            bos_token = all_tokens[:idx]
            sep_token = all_tokens[idx + 1:]
            if bos_token:
                res_context_list.append(bos_token)
                res_context_types.append(ContextType.OTHER)

        if not system:
            prefix = chat_template.prefix
        else:
            prefix = chat_template.system_prefix

        self._concat_context_list(prefix, res_context_list, res_context_types, system=system)

        assert len(inputs.messages) > 0, f'inputs.messages: {inputs.messages}'
        n_round = len(inputs.messages) // 2
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']
            response_role, response = response_message['role'], response_message['content']
            # TODO: Optimize the Template mechanism.
            assert query_role in {'user', 'tool'}, f'query_role: "{query_role}"'
            assert response_role in {'assistant'}, f'response_role: "{response_role}"'
            if query_role == 'tool':
                prompt = query
                query = ''
            elif chat_template.is_post_system and i == n_round - 1:
                prompt = chat_template.system_prompt
            else:
                prompt = chat_template.prompt

            context_list = prompt.copy()
            extra_context_list = []
            extra_context_type = None
            if i < n_round - 1:
                # Not the last round.
                context_list.append('{{RESPONSE}}')
                if inputs.messages[2 * (i + 1)]['role'] != 'tool':
                    extra_context_list = chat_template.chat_sep
                    extra_context_type = ContextType.OTHER
            elif response is not None:
                # It is the final round, and the response exists (during training).
                context_list.append('{{RESPONSE}}')
                # The GLM-4.5 assistant part (tool call) may end with <|observation|>,
                # and here we avoid adding <|user|>.
                response_content = response
                if not isinstance(response_content, str):
                    if isinstance(response, list):
                        token_ids = response
                    else:
                        token_ids = response['token_ids']
                    response_content = self.tokenizer.decode(token_ids[-20:])
                endswith_stop_words = any(
                    response_content.endswith(stop_word) for stop_word in chat_template.stop_words
                    if isinstance(stop_word, str))
                # self.is_training needed because we may want to continue generation from
                # the current response
                add_eos = inputs.extra_kwargs.get('add_eos')
                if add_eos is None:
                    add_eos = self.is_training and not sep_token and not endswith_stop_words
                if add_eos:
                    extra_context_list = chat_template.suffix
                    extra_context_type = ContextType.SUFFIX
            elif self.response_prefix:
                # final round and during inference.
                context_list.append(chat_template.response_prefix)

            self._concat_context_list(
                context_list,
                res_context_list,
                res_context_types,
                query=query,
                response=response,
                system=system,
                round0=i)
            res_context_list += extra_context_list
            res_context_types += [extra_context_type] * len(extra_context_list)
        if chat_template.auto_add_bos and sep_token:
            res_context_list.append(sep_token)
            res_context_types.append(ContextType.SUFFIX)
        res_context_list, loss_scale_list = self.loss_scale(res_context_list, res_context_types, inputs.messages,
                                                            **inputs.extra_kwargs)
        if self.is_training:
            answer_len = len(extra_context_list) + bool(response is not None)
        else:
            answer_len = 0
        return res_context_list, loss_scale_list, answer_len

    def _add_non_thinking_prefix(self, inputs) -> None:
        messages = inputs.messages
        non_thinking_prefix = self.chat_template.non_thinking_prefix
        if non_thinking_prefix:
            if not self.is_training or self.loss_scale.base_strategy == 'last_round':
                start_idx = get_last_user_round(messages)
            else:
                start_idx = -1
            for i, message in enumerate(messages):
                if i < start_idx:
                    continue
                if message['role'] == 'assistant' and isinstance(message['content'], str):
                    if not message['content'].startswith(('<think>', non_thinking_prefix)):
                        # During multi-turn SFT training/validation:
                        # If the message has no <think> block and does not start with the non_thinking_prefix,
                        # prepend the non_thinking_prefix to the content.
                        message['content'] = non_thinking_prefix + message['content']
                        
    def _remove_history_thinking(self, inputs) -> None:
        if self.is_training and self.loss_scale.base_strategy != 'last_round':
            return
        messages = inputs.messages
        # Only during inference or training, and only if the loss_scale is set to 'last_round',
        # will the previous 'think' entries be deleted.
        last_user_round = get_last_user_round(messages)
        for i, message in enumerate(messages):
            # Delete the content before '</think>' in all assistant turns except the last round.
            if message['role'] == 'assistant' and isinstance(message['content'], str) and i < last_user_round:
                message['content'] = self._remove_thinking_content(message['content'])
                
    
    def _native_prepare_inputs(self, inputs: StdChatTemplateInputs):
        """
        Preprocesses the list of messages in the input by merging and formatting consecutive messages
        according to their roles.

        Specifically, this method:
            - Merges consecutive messages from the same role ('assistant' or 'user') to prevent downstream errors.
            - Detects consecutive tool-related messages following an assistant message, then formats and
            combines them using `agent_template._format_tool_responses` for structured output.
            - Updates the messages list in-place for further processing.

        Args:
            inputs: An StdTemplateInputs object which contains a 'messages' attribute, which is a list of dictionaries.
                    Each message dictionary should have at least the keys 'role' and 'content'.

        Returns:
            None. The input messages list is updated in-place.
        """
        messages = inputs.messages
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                # where tool_content is a List.
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif pre_role == 'assistant' and role == 'assistant' or pre_role == 'user' and role == 'user':
                # Consecutive messages from the assistant/user role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    def _get_system(self, inputs: StdChatTemplateInputs) -> Optional[str]:
        chat_template = self.chat_template
        system = inputs.system
        tools = inputs.tools
        chat_template.check_system(system)
        if system is None:
            system = chat_template.default_system

        if tools is not None:
            system = self.agent_template._format_tools(tools, system, inputs.messages[0])
        return system

    def _encode_context_list(self,
                             context_list: List[Context],
                             loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float]]:
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                token_list = self._tokenize(context)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            if not self.loss_scale.is_loss_scale_binary:
                loss_scale.extend([loss_weight] * len(token_list))
        if self.loss_scale.is_loss_scale_binary:
            loss_scale = None
        return input_ids, labels, loss_scale

    def _tokenize(self, context, **kwargs):
        return self.tokenizer(context, return_attention_mask=False, add_special_tokens=False, **kwargs)['input_ids']

    def _tokenize_batch(self, contexts: List[str], **kwargs) -> List[List[int]]:
        """Tokenize a batch of strings efficiently using the tokenizer's batch processing.

        This is significantly faster than calling _tokenize() in a loop because:
        1. HuggingFace tokenizers use Rust implementations that parallelize across batch
        2. Reduces Python<->Rust FFI overhead
        3. Better memory locality and cache utilization

        Args:
            contexts: List of strings to tokenize
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            List of token ID lists, one per input string
        """
        if not contexts:
            return []

        # Use batch encoding for maximum performance
        batch_result = self.tokenizer(
            contexts,
            return_attention_mask=False,
            add_special_tokens=False,
            padding=False,  # Don't pad - we handle variable lengths
            truncation=False,  # Don't truncate - we handle this separately
            **kwargs
        )

        return batch_result['input_ids']

    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        old_kwargs = to_device(kwargs, model.device)
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        for k, v in old_kwargs.items():
            if k in {
                'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                'max_length_q', 'max_length_k', 'cu_seq_lens_q', 'cu_seq_lens_k'
            } and k not in kwargs:
                kwargs[k] = v
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        base_model = self.get_base_model(model)
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               inputs: StdChatTemplateInputs) -> Tuple[List[Context], List[float]]:
        """Merge anything in the context to simplify the inputs"""
        context_list, loss_scale_list = self._split_special_tokens(context_list, loss_scale_list)
        context_list, loss_scale_list = self._pre_tokenize(context_list, loss_scale_list, inputs)

        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list
        temp: List[str] = []
        temp_loss_scale = 0.
        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str) and (loss_scale == temp_loss_scale):
                temp.append(context)
            else:
                if len(temp) > 0:
                    res.append(''.join(temp))
                    res_loss_scale.append(temp_loss_scale)
                    temp.clear()
                if isinstance(context, str):  # loss_scale diff
                    temp.append(context)
                else:
                    res.append(context)
                    res_loss_scale.append(loss_scale)
                temp_loss_scale = loss_scale
        if len(temp) > 0:
            res.append(''.join(temp))
            res_loss_scale.append(temp_loss_scale)

        return res, res_loss_scale

    def forward_context(self, model, inputs):
        # This function encis only used to handle scenarios where the model needs
        # to be patched during the forward pass.
        return nullcontext()

    @property
    def is_training(self):
        return self.mode not in {'pt', 'vllm', 'sglang'}

    def set_mode(self, mode: Literal['pt', 'vllm', 'sglang', 'train', 'rlhf', 'gkd']) -> None:
        self.mode = mode

    def _truncate(self, input_ids: List[int], labels: Optional[List[int]], loss_mask: Optional[List[float]],
                  truncation_strategy: Literal['left', 'right']):
        placeholder_tokens = torch.tensor(self.placeholder_tokens)
        input_ids_tensor = torch.tensor(input_ids)
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
        n_protected = protected.sum().item()
        if n_protected < self.max_length:
            non_protected = (~protected).nonzero(as_tuple=True)[0]
            if truncation_strategy == 'left':
                idx = non_protected[-(self.max_length - n_protected):]
            else:
                idx = non_protected[:self.max_length - n_protected]
            protected[idx] = True
        input_ids = input_ids_tensor[protected].tolist()
        if labels is not None:
            labels = torch.tensor(labels)[protected].tolist()
        if loss_mask is not None:
            loss_mask = torch.tensor(loss_mask)[protected].tolist()
        return input_ids, labels, loss_mask

    @staticmethod
    def _concat_context_list(
            context_list: List[Context],
            res_context_list: List[Context],  # inplace
            res_context_type: List[ContextType],  # inplace
            system: Optional[str] = None,
            query: Optional[str] = None,
            response: Optional[str] = None,
            round0: Optional[int] = None) -> None:
        """Concat context list and replace placeholder"""
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    res_context_list.append(response)
                    res_context_type.append(ContextType.RESPONSE)
                    continue
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        assert isinstance(new_str, str), f'new_str: {new_str}'
                        context = context.replace(old_str, new_str)
            if len(context) == 0:
                continue
            res_context_list.append(context)
            res_context_type.append(ContextType.OTHER)

    @staticmethod
    def _get_std_messages(messages):
        if messages and messages[0]['role'] == 'assistant':
            messages.insert(0, {'role': 'user', 'content': ''})  # pretrain
        if len(messages) % 2 == 1:
            messages.append({'role': 'assistant', 'content': None})  # inference

    @staticmethod
    def _add_dynamic_eos(input_ids: List[int], labels: List[int], loss_scale: Optional[List[int]],
                         suffix_tokens_id: List[int]) -> None:
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels) + 1):
            if labels[i - 1] >= 0 and i < len(labels) and labels[i] == -100:
                start = i
            elif start > 0 and labels[i - 1] == -100 and (i == len(labels) or labels[i] >= 0):
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len and input_ids[start:start + suffix_len] == suffix_tokens_id:
                    labels[start:start + suffix_len] = suffix_tokens_id
                    if loss_scale and loss_scale[start:start + suffix_len] == [0] * suffix_len:
                        loss_scale[start:start + suffix_len] = [1] * suffix_len

    @staticmethod
    def _split_special_tokens(context_list: List[Context],
                              loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        """Split special tokens, for example `<image>`, `<video>`, this will help the replace_tag operation"""
        res: List[Context] = []
        loss_scale_res: List[float] = []
        for context, loss_scale in zip(context_list, loss_scale_list):
            contexts = []
            if isinstance(fetch_one(context), str):
                for d in split_str_parts_by(context, ChatTemplateProcessor.special_tokens):
                    contexts.extend([d['key'], d['content']])
                contexts = [c for c in contexts if c]
                res.extend(contexts)
                loss_scale_res.extend([loss_scale] * len(contexts))
            else:
                res.append(context)
                loss_scale_res.append(loss_scale)
        return res, loss_scale_res

    @staticmethod
    def _get_length(input_ids, labels):
        # input_ids might be a tensor.
        lengths = [0]
        if input_ids is not None:
            lengths.append(len(input_ids))
        if labels is not None:
            lengths.append(len(labels))
        length = max(lengths)
        return length

    @staticmethod
    def get_base_model(model):
        if isinstance(model, PeftModel):
            return model.model
        else:
            return model

    @staticmethod
    def _add_default_tags(inputs: StdChatTemplateInputs):
        total_content = []
        for message in inputs.messages:
            content = message['content'] or ''
            if not isinstance(content, str):
                if message['role'] == 'user':
                    # Give up adding the default tag
                    return
                elif message['role'] == 'assistant':
                    continue
            total_content.append(content)
        total_content = '\n'.join(total_content)
        if inputs.system:
            total_content = f'{inputs.system}\n{total_content}'
        for media_type in ['image', 'audio', 'video']:
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            medias = getattr(inputs, media_key)
            if not isinstance(medias, list):
                medias = [medias]
            if medias:
                num_media_tags = len(re.findall(media_tag, total_content))
                num_media = len(medias)
                num_new_tags = num_media - num_media_tags
                if num_new_tags > 0:
                    inputs.messages[0]['content'] = media_tag * num_new_tags + inputs.messages[0]['content']
                elif num_new_tags < 0:
                    logger.warning(
                        f'num_media: {num_media}, num_media_tags: {num_media_tags}, total_content: {total_content}. '
                        'We will only replace the frontmost media_tags while keeping the subsequent media_tags.')

    @staticmethod
    def _replace_image_tags(inputs: StdChatTemplateInputs):
        # compat
        if inputs.images:
            return
        images = []
        pattern = r'<img>(.+?)</img>'
        for message in inputs.messages:
            content = message['content']
            if not isinstance(content, str):
                continue
            for image in re.findall(pattern, content):
                # only support local_path
                if os.path.isfile(image):
                    images.append(image)
                else:
                    logger.warning_once(f'Failed to parse image path: `{content}`.', hash_id='<img></img>')
            message['content'] = re.sub(pattern, '<image>', content)
        inputs.images = images

    @staticmethod
    def _replace_start_image_tags(inputs: StdChatTemplateInputs):
        # compat
        generate_mode = False
        message = inputs.messages[-1]
        content = message['content']
        if message['role'] == 'user' and content.endswith('<start-image>'):
            generate_mode = True
            message['content'] = message['content'][:-len('<start-image>')]  # remove the <start-image>
        inputs.generate_mode = generate_mode

    @staticmethod
    def _load_image(image, load_images: bool):
        if load_images:
            if isinstance(image, dict) and 'bytes' in image:
                image = image['bytes'] or image['path']
            image = load_image(image)
        else:
            if isinstance(image, dict):
                path = image['path']
                if path and (path.startswith('http') or os.path.exists(path)):
                    image = path
                else:
                    image = load_image(image['bytes'])
            elif not isinstance(image, str):
                image = load_image(image)
        return image

    @staticmethod
    def _get_height_width(inputs: StdChatTemplateInputs) -> None:
        width = []
        height = []
        for image in inputs.images:
            width.append(image.width)
            height.append(image.height)
        inputs.objects['width'] = width
        inputs.objects['height'] = height

    @staticmethod
    def _save_pil_image(image: Image.Image) -> str:
        img_bytes = image.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        tmp_dir = os.path.join(get_cache_dir(), 'tmp', 'images')
        logger.info_once(f'create tmp_dir: {tmp_dir}')
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, f'{img_hash}.png')
        if not os.path.exists(img_path):
            image.save(img_path)
        return img_path

    @staticmethod
    def _get_bbox_str(bbox: List[int]) -> str:
        point = []
        for x, y in zip(bbox[::2], bbox[1::2]):
            point.append(f'({x},{y})')
        return ','.join(point)

    @staticmethod
    def _fetch_inputs_startswith(batch: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        new_batch = []
        for inputs in batch:
            new_inputs = {}
            for k, v in inputs.items():
                if k.startswith(prefix):
                    new_inputs[k[len(prefix):]] = v
            new_batch.append(new_inputs)
        return new_batch

    @staticmethod
    def fetch_inputs(batch: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, Any]:
        keys = keys or []
        rows = RowPreprocessor.rows_to_batched(batch)
        return {k: rows[k] for k in keys if rows.get(k) is not None}


def get_chat_template_processor(
        template_type: str,
        processor: Optional['Processor'],
        default_system: Optional[str] = None,
        max_length: Optional[int] = None,
        *,
        truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
        max_pixels: Optional[int] = None,  # h * w
        agent_template: Optional[str] = None,
        norm_bbox: Literal['norm1000', 'none', None] = None,
        use_chat_template: bool = True,
        # train
        padding_free: bool = False,
        padding_side: Literal['left', 'right'] = 'right',
        loss_scale: str = 'default',
        sequence_parallel_size: int = 1,
        # infer/deploy
        response_prefix: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        add_non_thinking_prefix: bool = True,
) -> 'ChatTemplateProcessor':
    from .base import CHAT_TEMPLATE_MAPPING

    chat_template = CHAT_TEMPLATE_MAPPING[template_type]
    template_processor_cls = chat_template.template_processor_cls
    return template_processor_cls(
        processor,
        chat_template,
        default_system,
        max_length,
        truncation_strategy=truncation_strategy,
        max_pixels=max_pixels,
        agent_template=agent_template,
        norm_bbox=norm_bbox,
        use_chat_template=use_chat_template,
        # train
        padding_free=padding_free,
        padding_side=padding_side,
        loss_scale=loss_scale,
        sequence_parallel_size=sequence_parallel_size,
        # infer/deploy
        response_prefix=response_prefix,
        enable_thinking=enable_thinking,
        add_non_thinking_prefix=add_non_thinking_prefix,
    )
