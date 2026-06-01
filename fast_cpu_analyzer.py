import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.utils import logging as transformers_logging


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


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name, "")
    if not value.strip():
        return default
    return [item.strip() for item in value.split("|") if item.strip()]


@dataclass(frozen=True)
class FastCpuSettings:
    model_id: str
    refs_dirs: list[str]
    text_prompts: list[str]
    device: str
    sample_fps: float
    max_sampled_frames: int
    max_width: int
    crop_mode: str
    approve_ref_threshold: float
    approve_text_threshold: float
    manual_ref_threshold: float
    manual_text_threshold: float
    approve_min_hits: int
    max_refs: int


def settings_from_env() -> FastCpuSettings:
    sample_fps = _env_float("VIDEO_FAST_SAMPLE_FPS", 0.25)
    if sample_fps <= 0:
        raise ValueError("VIDEO_FAST_SAMPLE_FPS must be > 0")

    return FastCpuSettings(
        model_id=os.getenv("VIDEO_FAST_MODEL_ID", "openai/clip-vit-base-patch32").strip(),
        refs_dirs=_env_list("VIDEO_FAST_REFS_DIRS", ["refs/can", "refs/chips", "refs/logo"]),
        text_prompts=_env_list(
            "VIDEO_FAST_TEXT_PROMPTS",
            [
                "lit energy drink can",
                "lit energy chips",
                "lit energy logo",
                "energy drink can",
            ],
        ),
        device=os.getenv("VIDEO_FAST_DEVICE", "cpu").strip() or "cpu",
        sample_fps=sample_fps,
        max_sampled_frames=_env_int("VIDEO_FAST_MAX_SAMPLED_FRAMES", 8),
        max_width=_env_int("VIDEO_FAST_MAX_WIDTH", 512),
        crop_mode=os.getenv("VIDEO_FAST_CROP_MODE", "5").strip() or "5",
        approve_ref_threshold=_env_float("VIDEO_FAST_APPROVE_REF_THRESHOLD", 0.30),
        approve_text_threshold=_env_float("VIDEO_FAST_APPROVE_TEXT_THRESHOLD", 0.20),
        manual_ref_threshold=_env_float("VIDEO_FAST_MANUAL_REF_THRESHOLD", 0.25),
        manual_text_threshold=_env_float("VIDEO_FAST_MANUAL_TEXT_THRESHOLD", 0.16),
        approve_min_hits=_env_int("VIDEO_FAST_APPROVE_MIN_HITS", 1),
        max_refs=_env_int("VIDEO_FAST_MAX_REFS", 24),
    )


class FastCpuVideoAnalyzer:
    def __init__(self, settings: FastCpuSettings):
        self.settings = settings
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self.ref_vec: torch.Tensor | None = None
        self.text_vec: torch.Tensor | None = None
        self.ref_paths: list[str] = []

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        transformers_logging.set_verbosity_error()
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("VIDEO_FAST_DEVICE=cuda but CUDA is not available")

        torch_threads = _env_int("VIDEO_FAST_TORCH_THREADS", 0)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self.model = CLIPModel.from_pretrained(self.settings.model_id).to(self.settings.device)
        self.processor = CLIPProcessor.from_pretrained(self.settings.model_id)
        self.model.eval()

        ref_paths, ref_images = _load_reference_images(self.settings.refs_dirs, self.settings.max_refs)
        if not ref_images:
            raise FileNotFoundError(f"no reference images found in {self.settings.refs_dirs}")
        self.ref_paths = ref_paths

        ref_feats = self._encode_images(ref_images)
        self.ref_vec = _normalize(ref_feats.mean(dim=0, keepdim=True))

        text_feats = self._encode_text(self.settings.text_prompts)
        self.text_vec = _normalize(text_feats.mean(dim=0, keepdim=True))

    def analyze(self, video_path: str, video_id: str = "") -> dict[str, Any]:
        if not os.path.exists(video_path):
            return {"status": "invalid", "result": {"reason": "video_not_found", "video_path": video_path}}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "invalid", "result": {"reason": "video_open_failed", "video_path": video_path}}

        self.load()
        assert self.model is not None
        assert self.processor is not None
        assert self.ref_vec is not None
        assert self.text_vec is not None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = float(total_frames / fps) if total_frames > 0 and fps > 0 else None
        frame_indices = _sample_frame_indices(
            fps=fps,
            total_frames=total_frames,
            sample_fps=self.settings.sample_fps,
            max_sampled_frames=self.settings.max_sampled_frames,
        )

        samples: list[dict[str, Any]] = []
        hits: list[dict[str, Any]] = []
        manual_candidates: list[dict[str, Any]] = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = _resize_rgb(rgb, self.settings.max_width)
            crops = _make_crops(rgb, self.settings.crop_mode)
            pil_crops = [Image.fromarray(crop) for crop in crops]

            crop_feats = self._encode_images(pil_crops)
            sims_ref = (crop_feats @ self.ref_vec.T).squeeze(1).detach().cpu().numpy()
            sims_text = (crop_feats @ self.text_vec.T).squeeze(1).detach().cpu().numpy()

            combined = (sims_ref + sims_text) / 2.0
            best_crop = int(np.argmax(combined))
            ref_score = float(sims_ref[best_crop])
            text_score = float(sims_text[best_crop])
            combined_score = float(combined[best_crop])
            timestamp_sec = float(frame_idx / fps) if fps > 0 else 0.0

            sample = {
                "frame_idx": int(frame_idx),
                "timestamp_sec": round(timestamp_sec, 3),
                "crop": best_crop,
                "ref_score": ref_score,
                "text_score": text_score,
                "combined_score": combined_score,
            }
            samples.append(sample)

            if (
                ref_score >= self.settings.approve_ref_threshold
                and text_score >= self.settings.approve_text_threshold
            ):
                hits.append(sample)
                if len(hits) >= self.settings.approve_min_hits:
                    break
                continue

            if (
                ref_score >= self.settings.manual_ref_threshold
                or text_score >= self.settings.manual_text_threshold
            ):
                manual_candidates.append(sample)

        cap.release()
        return self._result_payload(
            video_path=video_path,
            video_id=video_id,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            frames_requested=len(frame_indices),
            frames_checked=len(samples),
            samples=samples,
            hits=hits,
            manual_candidates=manual_candidates,
        )

    @torch.inference_mode()
    def _encode_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        assert self.model is not None
        assert self.processor is not None
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.settings.device)
        feats = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        return _normalize(_feature_tensor(feats))

    @torch.inference_mode()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        assert self.model is not None
        assert self.processor is not None
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.settings.device)
        feats = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )
        return _normalize(_feature_tensor(feats))

    def _result_payload(
        self,
        *,
        video_path: str,
        video_id: str,
        fps: float,
        total_frames: int,
        duration_sec: float | None,
        frames_requested: int,
        frames_checked: int,
        samples: list[dict[str, Any]],
        hits: list[dict[str, Any]],
        manual_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        top_samples = sorted(samples, key=lambda item: item["combined_score"], reverse=True)[:10]
        max_ref_score = max((item["ref_score"] for item in samples), default=0.0)
        max_text_score = max((item["text_score"] for item in samples), default=0.0)
        max_combined_score = max((item["combined_score"] for item in samples), default=0.0)

        if len(hits) >= self.settings.approve_min_hits:
            status = "approved"
            reason = "fast_cpu_hit"
        elif manual_candidates:
            status = "manual"
            reason = "fast_cpu_weak_match"
        else:
            status = "rejected"
            reason = "fast_cpu_no_match"

        return {
            "status": status,
            "result": {
                "reason": reason,
                "model": self.settings.model_id,
                "engine": "fast_cpu",
                "device": self.settings.device,
                "video_id": video_id,
                "video_path": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
                "sample_fps": self.settings.sample_fps,
                "frames_requested": frames_requested,
                "frames_checked": frames_checked,
                "max_width": self.settings.max_width,
                "crop_mode": self.settings.crop_mode,
                "ref_images_count": len(self.ref_paths),
                "text_prompts": self.settings.text_prompts,
                "thresholds": {
                    "approve_ref": self.settings.approve_ref_threshold,
                    "approve_text": self.settings.approve_text_threshold,
                    "manual_ref": self.settings.manual_ref_threshold,
                    "manual_text": self.settings.manual_text_threshold,
                    "approve_min_hits": self.settings.approve_min_hits,
                },
                "hits_count": len(hits),
                "manual_candidates_count": len(manual_candidates),
                "max_ref_score": max_ref_score,
                "max_text_score": max_text_score,
                "max_combined_score": max_combined_score,
                "top_samples": top_samples,
            },
        }


def _normalize(feats: torch.Tensor) -> torch.Tensor:
    return feats / feats.norm(dim=-1, keepdim=True)


def _feature_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        tensor = getattr(value, attr, None)
        if torch.is_tensor(tensor):
            return tensor
    raise TypeError(f"unexpected CLIP feature output type: {type(value)}")


def _load_reference_images(refs_dirs: list[str], max_refs: int) -> tuple[list[str], list[Image.Image]]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths: list[Path] = []
    for refs_dir in refs_dirs:
        base = Path(refs_dir)
        if not base.exists():
            continue
        paths.extend(path for path in sorted(base.iterdir()) if path.is_file() and path.suffix.lower() in exts)

    if max_refs > 0:
        paths = paths[:max_refs]

    images = [Image.open(path).convert("RGB") for path in paths]
    return [str(path) for path in paths], images


def _sample_frame_indices(
    *,
    fps: float,
    total_frames: int,
    sample_fps: float,
    max_sampled_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return [0]

    duration_sec = total_frames / fps if fps > 0 else 0
    desired = max(1, int(math.ceil(duration_sec * sample_fps))) if duration_sec > 0 else 1
    if max_sampled_frames > 0:
        desired = min(desired, max_sampled_frames)
    desired = min(desired, total_frames)

    if desired <= 1:
        return [max(0, min(total_frames - 1, total_frames // 2))]

    points = np.linspace(0.08, 0.92, desired)
    indices = sorted({max(0, min(total_frames - 1, int(round((total_frames - 1) * point)))) for point in points})
    return indices or [0]


def _resize_rgb(rgb: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return rgb
    height, width = rgb.shape[:2]
    if width <= max_width:
        return rgb
    scale = max_width / width
    next_height = max(1, int(round(height * scale)))
    return cv2.resize(rgb, (max_width, next_height), interpolation=cv2.INTER_AREA)


def _make_crops(rgb: np.ndarray, mode: str) -> list[np.ndarray]:
    if mode == "none":
        return [rgb]

    height, width = rgb.shape[:2]

    def crop(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(1, min(width, x1))
        y1 = max(1, min(height, y1))
        return rgb[y0:y1, x0:x1]

    if mode == "9":
        xs = [0, width // 3, 2 * width // 3, width]
        ys = [0, height // 3, 2 * height // 3, height]
        return [crop(xs[x], ys[y], xs[x + 1], ys[y + 1]) for y in range(3) for x in range(3)]

    crop_width = int(width * 0.62)
    crop_height = int(height * 0.62)
    return [
        crop((width - crop_width) // 2, (height - crop_height) // 2, (width + crop_width) // 2, (height + crop_height) // 2),
        crop(0, 0, crop_width, crop_height),
        crop(width - crop_width, 0, width, crop_height),
        crop(0, height - crop_height, crop_width, height),
        crop(width - crop_width, height - crop_height, width, height),
    ]
