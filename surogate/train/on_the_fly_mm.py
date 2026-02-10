import numpy as np
import torch

from surogate.core.datasets.datasets import disable_datasets_caching
from surogate.core.datasets.loader import load_dataset_with_config, pre_process, post_process, concat_datasets, \
    shuffle_dataset
from surogate.core.model.loader import get_model_info_and_tokenizer
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed


logger = get_logger()


def _labels_to_input_mask(labels: np.ndarray) -> np.ndarray:
    mask = (labels != -100).astype(np.int32, copy=False)
    out = np.zeros_like(mask, dtype=np.int32)
    if out.ndim != 2:
        raise ValueError("labels_to_input_mask expects a 2D labels array")
    if out.shape[1] > 0:
        out[:, :-1] = mask[:, 1:]
    return out


def init_mm_helpers(config):
    _, _, hf_model, processor = get_model_info_and_tokenizer(
        config.model_dir,
        torch_dtype=config.torch_dtype,
        load_model=True,
        max_model_len=config.max_model_len,
        rope_scaling=config.rope_scaling,
        model_type=config.model_type,
    )

    hf_model.eval()
    hf_model.requires_grad_(False)

    vision_device = torch.device("cuda:0") if torch.cuda.is_available() and config.gpus > 0 else torch.device("cpu")
    try:
        hf_model.model.visual.to(vision_device)
    except Exception as exc:
        logger.warning(f"Failed to move visual encoder to {vision_device}: {exc}. Falling back to CPU.")
        vision_device = torch.device("cpu")
        hf_model.model.visual.to(vision_device)

    try:
        hf_model.model.language_model.to("cpu")
        if hasattr(hf_model, "lm_head"):
            hf_model.lm_head.to("cpu")
    except Exception:
        pass

    template_processor = config.get_template_processor(processor)
    template_processor.set_mode('train')
    if template_processor.use_model:
        template_processor.model = hf_model

    rope_fn = hf_model.get_rope_index if hasattr(hf_model, "get_rope_index") else hf_model.model.get_rope_index

    return hf_model, processor, template_processor, vision_device, rope_fn


def load_multimodal_datasets(config, *, node_rank=None, num_nodes=None):
    train_datasets, val_datasets = [], []
    train_seed = np.random.RandomState(config.train_seed)
    eval_seed = np.random.RandomState(config.eval_seed)
    has_validation_datasets = len(config.validation_datasets) > 0

    with disable_datasets_caching():
        for ds_config in config.datasets:
            dataset = load_dataset_with_config(
                ds_config,
                num_workers=config.dataloader_num_workers,
                node_rank=node_rank,
                num_nodes=num_nodes,
            )
            dataset = pre_process(dataset, ds_config, num_proc=config.dataloader_num_workers)
            train_dataset, val_dataset = post_process(
                dataset,
                dataset_sample=ds_config.samples,
                split_dataset_ratio=config.validation_split_ratio if not has_validation_datasets else 0.0,
                random_state=train_seed,
            )
            train_datasets.append(train_dataset)
            if val_dataset is not None:
                val_datasets.append(val_dataset)

        for ds_config in config.validation_datasets:
            dataset = load_dataset_with_config(ds_config, num_workers=config.dataloader_num_workers)
            dataset = pre_process(dataset, ds_config, num_proc=config.dataloader_num_workers)
            _, val_dataset = post_process(
                dataset,
                dataset_sample=ds_config.samples,
                split_dataset_ratio=1.0,
                random_state=eval_seed,
            )
            val_datasets.append(val_dataset)

        train_dataset = concat_datasets(train_datasets)
        train_dataset = shuffle_dataset(
            train_dataset, seed=get_seed(train_seed), buffer_size=1000)

        val_dataset = None
        if len(val_datasets) > 0:
            val_dataset = concat_datasets(val_datasets)
            val_dataset = shuffle_dataset(
                val_dataset, seed=get_seed(eval_seed), buffer_size=1000)

    return train_dataset, val_dataset


class OnTheFlyMultimodalBatcher:
    def __init__(
        self,
        *,
        dataset,
        template_processor,
        hf_model,
        vision_device: torch.device,
        rope_fn,
        batch_size: int,
        seq_len: int,
        pad_token_id: int,
        seed: int,
        shuffle: bool = True,
        repeat: bool = True,
    ) -> None:
        self.dataset = dataset
        self.template = template_processor
        self.hf_model = hf_model
        self.vision_device = vision_device
        self.rope_fn = rope_fn
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.shuffle = shuffle
        self.repeat = repeat

        cfg = hf_model.config
        self.image_token_id = cfg.image_token_id
        self.video_token_id = cfg.video_token_id
        self.hidden_size = cfg.text_config.hidden_size
        self.deepstack_layers = len(getattr(hf_model.visual, "deepstack_visual_indexes", []))
        vision_cfg = getattr(cfg, "vision_config", None)
        merge_size = getattr(getattr(hf_model, "visual", None), "spatial_merge_size", None)
        if merge_size is None and vision_cfg is not None:
            merge_size = getattr(vision_cfg, "spatial_merge_size", None)
        self._merge_length = int(merge_size) ** 2 if merge_size is not None else 1

        self._epoch = 0
        self._samples_seen = 0
        self._encoded_buffer: list[dict] = []
        self._epoch_len = None
        try:
            self._epoch_len = len(dataset)
        except Exception:
            self._epoch_len = None

        self.steps_per_epoch = 0
        if self._epoch_len is not None and self.batch_size > 0:
            self.steps_per_epoch = max(1, self._epoch_len // self.batch_size)

        self._reset_iterator()

    def _reset_iterator(self) -> None:
        ds = self.dataset
        if self.shuffle:
            ds = shuffle_dataset(ds, seed=self.seed + self._epoch, buffer_size=1000)
        self._iter = iter(ds)
        self._samples_seen = 0

    def epoch(self) -> int:
        return self._epoch

    def progress(self) -> float:
        if not self._epoch_len:
            return 0.0
        return 100.0 * (float(self._samples_seen) / float(self._epoch_len))

    def _next_raw_rows(self, n: int) -> list[dict]:
        rows: list[dict] = []
        while len(rows) < n:
            try:
                row = next(self._iter)
            except StopIteration:
                if not self.repeat:
                    break
                self._epoch += 1
                self._reset_iterator()
                continue
            rows.append(row)
            self._samples_seen += 1
        return rows

    def _encode_rows(self, rows: list[dict]) -> list[dict]:
        # Multimodal templates need per-row encode to expand visual placeholders.
        if getattr(self.template, "model_template", None) is not None and self.template.model_template.is_multimodal:
            return self._encode_rows_single(rows)

        try:
            encoded = self.template.encode_batch(rows, return_length=True)
            encoded_list: list[dict] = []
            for item in encoded:
                if item is None:
                    continue
                if isinstance(item, list):
                    encoded_list.extend([x for x in item if x is not None])
                else:
                    encoded_list.append(item)
            return [e for e in encoded_list if self._is_row_usable(e)]
        except Exception:
            return self._encode_rows_single(rows)

    def _encode_rows_single(self, rows: list[dict]) -> list[dict]:
        encoded_list: list[dict] = []
        for row in rows:
            try:
                out = self.template.encode(row, return_length=True)
            except Exception:
                continue
            if out is None:
                continue
            if isinstance(out, list):
                encoded_list.extend([x for x in out if x is not None])
            else:
                encoded_list.append(out)
        return [e for e in encoded_list if self._is_row_usable(e)]

    def _is_row_usable(self, row: dict) -> bool:
        tokens = row.get("input_ids", None)
        if tokens is None:
            return False
        tokens_np = self._to_numpy_int(tokens)
        if tokens_np.size <= self.seq_len:
            return self._visual_tokens_aligned(row, tokens_np)
        tail = tokens_np[self.seq_len:]
        if np.any(tail == self.image_token_id) or np.any(tail == self.video_token_id):
            return False
        return self._visual_tokens_aligned(row, tokens_np)

    def _expected_tokens_from_grid(self, grid_thw) -> int:
        if grid_thw is None:
            return 0
        grid_t = self._to_torch(grid_thw, dtype=torch.long)
        if grid_t.ndim == 1:
            grid_t = grid_t.unsqueeze(0)
        if grid_t.numel() == 0:
            return 0
        return int((grid_t.prod(dim=-1) // self._merge_length).sum().item())

    def _visual_tokens_aligned(self, row: dict, tokens_np: np.ndarray) -> bool:
        image_tokens = int((tokens_np == self.image_token_id).sum())
        video_tokens = int((tokens_np == self.video_token_id).sum())

        image_grid = row.get("image_grid_thw", None)
        video_grid = row.get("video_grid_thw", None)

        expected_image = self._expected_tokens_from_grid(image_grid)
        expected_video = self._expected_tokens_from_grid(video_grid)

        if image_tokens > 0 and image_grid is None:
            logger.warning_once("Dropping row with image tokens but missing image_grid_thw/pixel_values.")
            return False
        if video_tokens > 0 and video_grid is None:
            logger.warning_once("Dropping row with video tokens but missing video_grid_thw/pixel_values.")
            return False
        if expected_image and image_tokens != expected_image:
            logger.warning_once(
                f"Dropping row with mismatched image tokens (tokens={image_tokens}, expected={expected_image})."
            )
            return False
        if expected_video and video_tokens != expected_video:
            logger.warning_once(
                f"Dropping row with mismatched video tokens (tokens={video_tokens}, expected={expected_video})."
            )
            return False
        if image_tokens == 0 and expected_image > 0:
            logger.warning_once("Dropping row with image_grid_thw but no image tokens.")
            return False
        if video_tokens == 0 and expected_video > 0:
            logger.warning_once("Dropping row with video_grid_thw but no video tokens.")
            return False
        return True

    @staticmethod
    def _to_numpy_int(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.int32)

    @staticmethod
    def _to_torch(x, *, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype) if dtype is not None else x
        return torch.as_tensor(x, dtype=dtype)

    def next_batch(self) -> dict:
        attempts = 0
        while len(self._encoded_buffer) < self.batch_size:
            need = self.batch_size - len(self._encoded_buffer)
            rows = self._next_raw_rows(need)
            if not rows:
                if not self.repeat:
                    if not self._encoded_buffer:
                        raise StopIteration
                    self._encoded_buffer = []
                    raise StopIteration
                attempts += 1
                if attempts > 100:
                    raise RuntimeError("Failed to build a batch from the dataset")
                continue
            encoded = self._encode_rows(rows)
            self._encoded_buffer.extend(encoded)
            attempts += 1
            if attempts > 100 and len(self._encoded_buffer) < self.batch_size:
                raise RuntimeError("Failed to build a batch from the dataset after multiple attempts")

        batch_rows = self._encoded_buffer[:self.batch_size]
        self._encoded_buffer = self._encoded_buffer[self.batch_size:]
        return self._build_batch(batch_rows)

    def _build_batch(self, rows: list[dict]) -> dict:
        B = self.batch_size
        T = self.seq_len
        pad_id = self.pad_token_id

        inputs = np.full((B, T), pad_id, dtype=np.int32)
        labels = np.full((B, T), -100, dtype=np.int32)

        image_grid_thw_list: list[torch.Tensor] = []
        image_pixel_values_list: list[torch.Tensor] = []
        video_grid_thw_list: list[torch.Tensor] = []
        video_pixel_values_list: list[torch.Tensor] = []

        for i, row in enumerate(rows):
            if i >= B:
                break
            tokens = self._to_numpy_int(row.get("input_ids", []))
            lbls = row.get("labels", None)
            if lbls is None:
                lbls = tokens
            labels_np = self._to_numpy_int(lbls)

            if tokens.ndim != 1 or labels_np.ndim != 1:
                raise ValueError("input_ids/labels must be 1D arrays")

            length = min(tokens.size, T)
            if length > 0:
                inputs[i, :length] = tokens[:length]
                labels[i, :length] = labels_np[:length]

            image_grid = row.get("image_grid_thw", None)
            if image_grid is not None:
                image_grid_t = self._to_torch(image_grid, dtype=torch.long)
                image_grid_thw_list.append(image_grid_t)
                pixel_values = row.get("pixel_values", None)
                if pixel_values is None:
                    raise ValueError("image_grid_thw present but pixel_values missing")
                image_pixel_values_list.append(self._to_torch(pixel_values))

            video_grid = row.get("video_grid_thw", None)
            if video_grid is not None:
                video_grid_t = self._to_torch(video_grid, dtype=torch.long)
                video_grid_thw_list.append(video_grid_t)
                pixel_values_videos = row.get("pixel_values_videos", None)
                if pixel_values_videos is None and image_grid is None:
                    pixel_values_videos = row.get("pixel_values", None)
                if pixel_values_videos is None:
                    raise ValueError("video_grid_thw present but pixel_values_videos missing")
                video_pixel_values_list.append(self._to_torch(pixel_values_videos))

        attention_mask = (inputs != pad_id).astype(np.int32, copy=False)

        targets = inputs.copy()
        if T > 1:
            targets[:, :-1] = inputs[:, 1:]
        targets[:, -1] = -100

        input_mask = _labels_to_input_mask(labels)
        targets[input_mask == 0] = -100

        image_grid_thw_cpu = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None
        video_grid_thw_cpu = torch.cat(video_grid_thw_list, dim=0) if video_grid_thw_list else None

        input_ids_t = torch.from_numpy(inputs.astype(np.int64, copy=False))
        attn_t = torch.from_numpy(attention_mask.astype(np.int64, copy=False))
        position_ids, _ = self.rope_fn(
            input_ids_t,
            image_grid_thw=image_grid_thw_cpu,
            video_grid_thw=video_grid_thw_cpu,
            attention_mask=attn_t,
        )

        visual_mask = (inputs == self.image_token_id) | (inputs == self.video_token_id)
        num_visual = int(visual_mask.sum())
        visual_pos_masks = visual_mask.astype(np.int32, copy=False)

        visual_embeds = np.zeros((B * T, self.hidden_size), dtype=np.float32)
        deepstack_visual_embeds: list[np.ndarray] = []

        if num_visual > 0:
            with torch.no_grad():
                image_embeds_flat = torch.empty((0, self.hidden_size), device=self.vision_device)
                video_embeds_flat = torch.empty((0, self.hidden_size), device=self.vision_device)
                deepstack_image = []
                deepstack_video = []

                if image_pixel_values_list:
                    if image_grid_thw_cpu is None:
                        raise ValueError("pixel_values present but image_grid_thw missing")
                    pixel_values = torch.cat(image_pixel_values_list, dim=0).to(self.vision_device)
                    image_grid_thw_vis = image_grid_thw_cpu.to(self.vision_device)
                    image_embeds_list, deepstack_image = self.hf_model.get_image_features(
                        pixel_values, image_grid_thw=image_grid_thw_vis
                    )
                    if image_embeds_list:
                        image_embeds_flat = torch.cat(image_embeds_list, dim=0)

                if video_pixel_values_list:
                    if video_grid_thw_cpu is None:
                        raise ValueError("pixel_values_videos present but video_grid_thw missing")
                    pixel_values_v = torch.cat(video_pixel_values_list, dim=0).to(self.vision_device)
                    video_grid_thw_vis = video_grid_thw_cpu.to(self.vision_device)
                    video_embeds_list, deepstack_video = self.hf_model.get_video_features(
                        pixel_values_v, video_grid_thw=video_grid_thw_vis
                    )
                    if video_embeds_list:
                        video_embeds_flat = torch.cat(video_embeds_list, dim=0)

                if self.deepstack_layers:
                    if deepstack_image and len(deepstack_image) != self.deepstack_layers:
                        raise ValueError("deepstack image layers mismatch")
                    if deepstack_video and len(deepstack_video) != self.deepstack_layers:
                        raise ValueError("deepstack video layers mismatch")
                    if not deepstack_image:
                        deepstack_image = [
                            torch.empty((0, self.hidden_size), device=self.vision_device)
                            for _ in range(self.deepstack_layers)
                        ]
                    if not deepstack_video:
                        deepstack_video = [
                            torch.empty((0, self.hidden_size), device=self.vision_device)
                            for _ in range(self.deepstack_layers)
                        ]

                expected_image = int((inputs == self.image_token_id).sum())
                expected_video = int((inputs == self.video_token_id).sum())

                if image_embeds_flat.shape[0] < expected_image:
                    raise ValueError("not enough image embeddings for visual tokens")
                if video_embeds_flat.shape[0] < expected_video:
                    raise ValueError("not enough video embeddings for visual tokens")
                if image_embeds_flat.shape[0] > expected_image:
                    image_embeds_flat = image_embeds_flat[:expected_image]
                    if self.deepstack_layers:
                        deepstack_image = [d[:expected_image] for d in deepstack_image]
                if video_embeds_flat.shape[0] > expected_video:
                    video_embeds_flat = video_embeds_flat[:expected_video]
                    if self.deepstack_layers:
                        deepstack_video = [d[:expected_video] for d in deepstack_video]

                visual_seq = torch.empty((num_visual, self.hidden_size), device=self.vision_device)
                deepstack_seq = [
                    torch.empty_like(visual_seq) for _ in range(self.deepstack_layers)
                ]

                flat_tokens = inputs.reshape(-1)
                image_idx = 0
                video_idx = 0
                out_idx = 0
                for tok in flat_tokens:
                    if tok == self.image_token_id:
                        visual_seq[out_idx] = image_embeds_flat[image_idx]
                        for li in range(self.deepstack_layers):
                            deepstack_seq[li][out_idx] = deepstack_image[li][image_idx]
                        image_idx += 1
                        out_idx += 1
                    elif tok == self.video_token_id:
                        visual_seq[out_idx] = video_embeds_flat[video_idx]
                        for li in range(self.deepstack_layers):
                            deepstack_seq[li][out_idx] = deepstack_video[li][video_idx]
                        video_idx += 1
                        out_idx += 1

                if image_idx != expected_image or video_idx != expected_video or out_idx != num_visual:
                    raise ValueError("visual embedding alignment mismatch")

                visual_seq_cpu = visual_seq.float().cpu().numpy()
                visual_embeds[:num_visual] = visual_seq_cpu

                for li in range(self.deepstack_layers):
                    buf = np.zeros_like(visual_embeds)
                    buf[:num_visual] = deepstack_seq[li].float().cpu().numpy()
                    deepstack_visual_embeds.append(buf)
        else:
            if self.deepstack_layers:
                for _ in range(self.deepstack_layers):
                    deepstack_visual_embeds.append(np.zeros_like(visual_embeds))

        return {
            "inputs": inputs,
            "targets": targets,
            "position_ids": position_ids.cpu().numpy().astype(np.int32),
            "visual_pos_masks": visual_pos_masks,
            "visual_embeds": visual_embeds,
            "deepstack_visual_embeds": deepstack_visual_embeds,
        }
