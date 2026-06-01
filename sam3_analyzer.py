import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Config, Sam3Model, Sam3Processor


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    return [item.strip() for item in value.split("|") if item.strip()]


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_torch_dtype(value: str, device: str) -> torch.dtype | None:
    value = value.lower().strip()
    if value in {"", "auto"}:
        return torch.float16 if device == "cuda" else torch.float32
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"unsupported VIDEO_SAM3_DTYPE: {value}")


@dataclass(frozen=True)
class Sam3Settings:
    model_id: str
    sample_fps: float
    threshold: float
    mask_threshold: float
    min_mask_area_ratio: float
    approve_min_hits: int
    approve_min_unique_seconds: int
    prompts: list[str]
    device: str
    dtype: torch.dtype | None
    image_size: int
    early_approve: bool
    max_sampled_frames: int


def settings_from_env(
    *,
    model_id: str | None = None,
    sample_fps: float | None = None,
    threshold: float | None = None,
    mask_threshold: float | None = None,
    min_mask_area_ratio: float | None = None,
    approve_min_hits: int | None = None,
    approve_min_unique_seconds: int | None = None,
    prompts: list[str] | None = None,
    device: str | None = None,
) -> Sam3Settings:
    resolved_device = (device or os.getenv("VIDEO_SAM3_DEVICE") or default_device()).strip()
    dtype = normalize_torch_dtype(os.getenv("VIDEO_SAM3_DTYPE", "auto"), resolved_device)
    resolved_prompts = prompts or _env_list(
        "VIDEO_SAM3_PROMPTS",
        ["lit energy can", "lit energy drink can", "energy drink can"],
    )
    if not resolved_prompts:
        raise ValueError("at least one SAM3 prompt is required")

    resolved_sample_fps = sample_fps if sample_fps is not None else _env_float("VIDEO_SAM3_SAMPLE_FPS", 2.0)
    if resolved_sample_fps <= 0:
        raise ValueError("VIDEO_SAM3_SAMPLE_FPS must be > 0")

    return Sam3Settings(
        model_id=model_id or os.getenv("VIDEO_SAM3_MODEL_ID", "facebook/sam3"),
        sample_fps=resolved_sample_fps,
        threshold=threshold if threshold is not None else _env_float("VIDEO_SAM3_SCORE_THRESHOLD", 0.5),
        mask_threshold=mask_threshold
        if mask_threshold is not None
        else _env_float("VIDEO_SAM3_MASK_THRESHOLD", 0.5),
        min_mask_area_ratio=min_mask_area_ratio
        if min_mask_area_ratio is not None
        else _env_float("VIDEO_SAM3_MIN_MASK_AREA_RATIO", 0.001),
        approve_min_hits=approve_min_hits
        if approve_min_hits is not None
        else _env_int("VIDEO_SAM3_APPROVE_MIN_HITS", 3),
        approve_min_unique_seconds=approve_min_unique_seconds
        if approve_min_unique_seconds is not None
        else _env_int("VIDEO_SAM3_APPROVE_MIN_UNIQUE_SECONDS", 2),
        prompts=resolved_prompts,
        device=resolved_device,
        dtype=dtype,
        image_size=_env_int("VIDEO_SAM3_IMAGE_SIZE", 0),
        early_approve=_env_bool("VIDEO_SAM3_EARLY_APPROVE", True),
        max_sampled_frames=_env_int("VIDEO_SAM3_MAX_SAMPLED_FRAMES", 0),
    )


class Sam3VideoAnalyzer:
    def __init__(self, settings: Sam3Settings):
        self.settings = settings
        self.model: Sam3Model | None = None
        self.processor: Sam3Processor | None = None
        self.text_inputs_by_prompt: dict[str, Any] = {}

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        if self.settings.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_kwargs: dict[str, Any] = {}
        if self.settings.dtype is not None and self.settings.dtype != torch.float32:
            model_kwargs["torch_dtype"] = self.settings.dtype

        processor_kwargs: dict[str, Any] = {}
        if self.settings.image_size > 0:
            config = Sam3Config.from_pretrained(self.settings.model_id)
            config.image_size = self.settings.image_size
            model_kwargs["config"] = config
            processor_kwargs["size"] = {"height": self.settings.image_size, "width": self.settings.image_size}

        self.model = Sam3Model.from_pretrained(self.settings.model_id, **model_kwargs).to(self.settings.device)
        self.processor = Sam3Processor.from_pretrained(self.settings.model_id, **processor_kwargs)
        self.model.eval()

        self.text_inputs_by_prompt = {
            prompt: self.processor(text=prompt, return_tensors="pt").to(self.settings.device)
            for prompt in self.settings.prompts
        }

    def analyze(self, video_path: str, video_id: str = "") -> dict[str, Any]:
        if not os.path.exists(video_path):
            return {"status": "invalid", "result": {"reason": "video_not_found", "video_path": video_path}}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "invalid", "result": {"reason": "video_open_failed", "video_path": video_path}}

        self.load()
        assert self.model is not None
        assert self.processor is not None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = float(total_frames / fps) if total_frames > 0 and fps > 0 else None
        frame_step = max(1, int(round(fps / self.settings.sample_fps))) if fps > 0 else 1

        frame_idx = 0
        frames_checked = 0
        hits: list[dict[str, Any]] = []
        early_stopped = False

        while True:
            grabbed = cap.grab()
            if not grabbed:
                break

            should_sample = frame_idx % frame_step == 0
            if not should_sample:
                frame_idx += 1
                continue

            ok, bgr = cap.retrieve()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            t_sec = float(frame_idx / fps) if fps > 0 else 0.0
            frames_checked += 1

            frame_hits = self.detect_frame(image)
            if frame_hits:
                best_hit = max(frame_hits, key=lambda item: item["score"])
                best_hit = dict(best_hit)
                best_hit["frame_idx"] = frame_idx
                best_hit["timestamp_sec"] = round(t_sec, 3)
                hits.append(best_hit)

            if self._should_stop_early(hits):
                early_stopped = True
                break
            if self.settings.max_sampled_frames > 0 and frames_checked >= self.settings.max_sampled_frames:
                break

            frame_idx += 1

        cap.release()
        return self._result_payload(
            video_path=video_path,
            video_id=video_id,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            frame_step=frame_step,
            frames_checked=frames_checked,
            hits=hits,
            early_stopped=early_stopped,
        )

    def detect_frame(self, image: Image.Image) -> list[dict[str, Any]]:
        assert self.model is not None
        assert self.processor is not None

        img_inputs = self.processor(images=image, return_tensors="pt").to(self.settings.device)
        target_sizes = _target_sizes(img_inputs, image)
        frame_hits: list[dict[str, Any]] = []

        with torch.inference_mode(), self._autocast_context():
            vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)
            for prompt, text_inputs in self.text_inputs_by_prompt.items():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
                frame_hits.extend(
                    _hits_from_outputs(
                        processor=self.processor,
                        outputs=outputs,
                        prompt=prompt,
                        image=image,
                        target_sizes=target_sizes,
                        threshold=self.settings.threshold,
                        mask_threshold=self.settings.mask_threshold,
                        min_mask_area_ratio=self.settings.min_mask_area_ratio,
                    )
                )

        return frame_hits

    def _autocast_context(self):
        if self.settings.device != "cuda":
            return nullcontext()
        if self.settings.dtype not in {torch.float16, torch.bfloat16}:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.settings.dtype)

    def _should_stop_early(self, hits: list[dict[str, Any]]) -> bool:
        if not self.settings.early_approve:
            return False
        if len(hits) < self.settings.approve_min_hits:
            return False
        unique_seconds = len({int(math.floor(hit["timestamp_sec"])) for hit in hits})
        return unique_seconds >= self.settings.approve_min_unique_seconds

    def _result_payload(
        self,
        *,
        video_path: str,
        video_id: str,
        fps: float,
        total_frames: int,
        duration_sec: float | None,
        frame_step: int,
        frames_checked: int,
        hits: list[dict[str, Any]],
        early_stopped: bool,
    ) -> dict[str, Any]:
        unique_seconds = len({int(math.floor(hit["timestamp_sec"])) for hit in hits})
        top_hits = sorted(hits, key=lambda item: item["score"], reverse=True)[:10]
        max_score = max((hit["score"] for hit in hits), default=0.0)

        if len(hits) >= self.settings.approve_min_hits and unique_seconds >= self.settings.approve_min_unique_seconds:
            status = "approved"
        elif hits:
            status = "manual"
        else:
            status = "rejected"

        return {
            "status": status,
            "result": {
                "model": self.settings.model_id,
                "device": self.settings.device,
                "dtype": None if self.settings.dtype is None else str(self.settings.dtype).replace("torch.", ""),
                "image_size": self.settings.image_size or None,
                "video_id": video_id,
                "video_path": video_path,
                "fps": fps,
                "sample_fps": self.settings.sample_fps,
                "frame_step": frame_step,
                "frames_checked": frames_checked,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "prompts": self.settings.prompts,
                "hits_count": len(hits),
                "unique_seconds_with_hits": unique_seconds,
                "max_score": max_score,
                "top_hits": top_hits,
                "early_stopped": early_stopped,
                "max_sampled_frames": self.settings.max_sampled_frames,
            },
        }


def _target_sizes(inputs: Any, image: Image.Image) -> list[list[int]]:
    original_sizes = inputs.get("original_sizes")
    if original_sizes is None:
        return [[image.height, image.width]]
    return original_sizes.tolist()


def _hits_from_outputs(
    *,
    processor: Sam3Processor,
    outputs: Any,
    prompt: str,
    image: Image.Image,
    target_sizes: list[list[int]],
    threshold: float,
    mask_threshold: float,
    min_mask_area_ratio: float,
) -> list[dict[str, Any]]:
    post = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )
    if not post:
        return []

    result = post[0]
    masks = result.get("segmentation", result.get("masks", []))
    labels = result.get("labels", [])
    scores = result.get("scores", [])

    image_area = float(image.width * image.height)
    hits: list[dict[str, Any]] = []
    for idx, mask in enumerate(masks):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score < threshold:
            continue

        mask_np = _to_numpy_mask(mask)
        mask_area = int(mask_np.sum())
        area_ratio = float(mask_area / image_area) if image_area > 0 else 0.0
        if area_ratio < min_mask_area_ratio:
            continue

        bbox = _mask_bbox(mask_np)
        if bbox is None:
            continue

        label = labels[idx] if idx < len(labels) else None
        hits.append(
            {
                "prompt": prompt,
                "label": None if label is None else str(label),
                "score": score,
                "bbox": bbox,
                "mask_area": mask_area,
                "mask_area_ratio": area_ratio,
            }
        )

    return hits


def _to_numpy_mask(mask: Any) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
