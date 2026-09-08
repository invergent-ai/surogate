"""Library-backed hybrid frontend loaded from artifact resources."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

from .multimodal import MultimodalBatch, batch_from_processor_output


def _fetch_videos_opencv(_processor, video_or_videos, sample_indices_fn=None):
    """Use the Transformers sampler with its local OpenCV decoder."""

    from transformers.video_utils import load_video

    if isinstance(video_or_videos, list):
        fetched = [
            _fetch_videos_opencv(_processor, item, sample_indices_fn=sample_indices_fn) for item in video_or_videos
        ]
        return list(zip(*fetched))
    return load_video(
        video_or_videos,
        backend="opencv",
        sample_indices_fn=sample_indices_fn,
    )


class Frontend:
    """Processor, tokenizer, template, and generation defaults for one artifact."""

    def __init__(self, binding: Any):
        try:
            from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
            from transformers.utils import is_torchcodec_available
        except ImportError as exc:
            raise RuntimeError("hybrid reference inference requires Transformers with Qwen3-VL support") from exc

        resources = binding.frontend
        self.vision_config = binding.vision_config
        self.token_domain = binding.config.token_domain
        files: tuple[tuple[str, Any], ...] = (
            ("tokenizer.json", resources.tokenizer_json),
            ("tokenizer_config.json", resources.tokenizer_config_json),
            ("chat_template.jinja", resources.chat_template_jinja),
            ("generation_config.json", resources.generation_config_json),
            ("preprocessor_config.json", resources.preprocessor_config_json),
            (
                "video_preprocessor_config.json",
                resources.video_preprocessor_config_json,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="sinfer-frontend-") as temporary:
            directory = Path(temporary)
            for filename, resource in files:
                if resource is not None:
                    (directory / filename).write_bytes(binding.resource_bytes(resource))
            if self.vision_config is not None:
                self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
                self.processor = self.tokenizer
            self.generation_config = GenerationConfig.from_pretrained(directory, local_files_only=True)

        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is not None and not is_torchcodec_available():
            video_processor.fetch_videos = MethodType(_fetch_videos_opencv, video_processor)
        vocabulary = self.tokenizer.get_vocab()
        domain = max(vocabulary.values(), default=-1) + 1
        if domain != self.token_domain:
            raise ValueError(f"artifact tokenizer has ID domain {domain}; checkpoint declares {self.token_domain}")
        if self.vision_config is not None:
            for name in ("image_processor", "video_processor"):
                processor = getattr(self.processor, name, None)
                if processor is not None and getattr(processor, "merge_size", None) != self.vision_config.spatial_merge:
                    raise ValueError(f"{name} merge_size disagrees with checkpoint vision geometry")

    @property
    def default_stop_token_ids(self) -> set[int]:
        values = self.generation_config.eos_token_id
        if values is None:
            return set()
        if isinstance(values, int):
            return {values}
        return {int(value) for value in values}

    def process(self, messages: list[dict[str, Any]], *, thinking: bool) -> MultimodalBatch:
        output = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        if self.vision_config is None and "mm_token_type_ids" not in output:
            import torch

            output["mm_token_type_ids"] = torch.zeros_like(output["input_ids"])
        return batch_from_processor_output(
            output, spatial_merge=self.vision_config.spatial_merge if self.vision_config else 1
        )

    def process_text(self, text: str, *, thinking: bool) -> MultimodalBatch:
        if not text.strip():
            raise ValueError("prompt text must not be empty")
        return self.process(
            [{"role": "user", "content": text}],
            thinking=thinking,
        )

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


__all__ = ["Frontend"]
