import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
class CvMatchSettings:
    refs_dirs: list[str]
    sample_fps: float
    max_sampled_frames: int
    max_width: int
    crop_mode: str
    orb_features: int
    orb_distance_threshold: int
    approve_min_good_matches: int
    manual_min_good_matches: int
    approve_score_threshold: float
    manual_score_threshold: float
    max_refs: int


@dataclass(frozen=True)
class ReferenceImage:
    path: str
    keypoints_count: int
    descriptors: np.ndarray | None
    hist: np.ndarray


def settings_from_env() -> CvMatchSettings:
    sample_fps = _env_float("VIDEO_CV_SAMPLE_FPS", 0.35)
    if sample_fps <= 0:
        raise ValueError("VIDEO_CV_SAMPLE_FPS must be > 0")

    return CvMatchSettings(
        refs_dirs=_env_list("VIDEO_CV_REFS_DIRS", ["refs/can", "refs/chips", "refs/logo"]),
        sample_fps=sample_fps,
        max_sampled_frames=_env_int("VIDEO_CV_MAX_SAMPLED_FRAMES", 12),
        max_width=_env_int("VIDEO_CV_MAX_WIDTH", 720),
        crop_mode=os.getenv("VIDEO_CV_CROP_MODE", "5").strip() or "5",
        orb_features=_env_int("VIDEO_CV_ORB_FEATURES", 900),
        orb_distance_threshold=_env_int("VIDEO_CV_ORB_DISTANCE_THRESHOLD", 58),
        approve_min_good_matches=_env_int("VIDEO_CV_APPROVE_MIN_GOOD_MATCHES", 18),
        manual_min_good_matches=_env_int("VIDEO_CV_MANUAL_MIN_GOOD_MATCHES", 8),
        approve_score_threshold=_env_float("VIDEO_CV_APPROVE_SCORE_THRESHOLD", 0.55),
        manual_score_threshold=_env_float("VIDEO_CV_MANUAL_SCORE_THRESHOLD", 0.35),
        max_refs=_env_int("VIDEO_CV_MAX_REFS", 36),
    )


class CvMatchVideoAnalyzer:
    def __init__(self, settings: CvMatchSettings):
        self.settings = settings
        self.orb = cv2.ORB_create(nfeatures=settings.orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.refs: list[ReferenceImage] = []

    def load(self) -> None:
        if self.refs:
            return

        paths = _load_reference_paths(self.settings.refs_dirs, self.settings.max_refs)
        if not paths:
            raise FileNotFoundError(f"no reference images found in {self.settings.refs_dirs}")

        refs: list[ReferenceImage] = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = _resize_bgr(image, self.settings.max_width)
            keypoints, descriptors = self._features(image)
            refs.append(
                ReferenceImage(
                    path=str(path),
                    keypoints_count=len(keypoints),
                    descriptors=descriptors,
                    hist=_hsv_hist(image),
                )
            )

        if not refs:
            raise FileNotFoundError(f"reference images could not be read from {self.settings.refs_dirs}")
        self.refs = refs

    def analyze(self, video_path: str, video_id: str = "") -> dict[str, Any]:
        if not os.path.exists(video_path):
            return {"status": "invalid", "result": {"reason": "video_not_found", "video_path": video_path}}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "invalid", "result": {"reason": "video_open_failed", "video_path": video_path}}

        self.load()
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
            ok, frame = cap.read()
            if not ok:
                continue

            frame = _resize_bgr(frame, self.settings.max_width)
            crops = _make_crops(frame, self.settings.crop_mode)
            best = self._best_crop_match(crops)
            timestamp_sec = float(frame_idx / fps) if fps > 0 else 0.0
            sample = {
                "frame_idx": int(frame_idx),
                "timestamp_sec": round(timestamp_sec, 3),
                **best,
            }
            samples.append(sample)

            if (
                best["good_matches"] >= self.settings.approve_min_good_matches
                and best["score"] >= self.settings.approve_score_threshold
            ):
                hits.append(sample)
                break

            if (
                best["good_matches"] >= self.settings.manual_min_good_matches
                or best["score"] >= self.settings.manual_score_threshold
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

    def _features(self, image: np.ndarray) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return tuple(keypoints or ()), descriptors

    def _best_crop_match(self, crops: list[np.ndarray]) -> dict[str, Any]:
        best: dict[str, Any] = {
            "crop": 0,
            "ref_path": "",
            "good_matches": 0,
            "match_score": 0.0,
            "hist_score": 0.0,
            "score": 0.0,
        }

        for crop_idx, crop in enumerate(crops):
            keypoints, descriptors = self._features(crop)
            hist = _hsv_hist(crop)

            for ref in self.refs:
                good_matches = _good_match_count(
                    matcher=self.matcher,
                    left=descriptors,
                    right=ref.descriptors,
                    distance_threshold=self.settings.orb_distance_threshold,
                )
                denom = max(1, min(len(keypoints), ref.keypoints_count, self.settings.approve_min_good_matches))
                match_score = min(1.0, good_matches / denom)
                hist_score = _hist_score(hist, ref.hist)
                score = 0.82 * match_score + 0.18 * hist_score

                if score > best["score"]:
                    best = {
                        "crop": crop_idx,
                        "ref_path": ref.path,
                        "good_matches": int(good_matches),
                        "match_score": float(match_score),
                        "hist_score": float(hist_score),
                        "score": float(score),
                    }

        return best

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
        top_samples = sorted(samples, key=lambda item: item["score"], reverse=True)[:10]
        max_score = max((item["score"] for item in samples), default=0.0)
        max_good_matches = max((item["good_matches"] for item in samples), default=0)

        if hits:
            status = "approved"
            reason = "cv_match_hit"
        elif manual_candidates:
            status = "manual"
            reason = "cv_match_weak_match"
        else:
            status = "rejected"
            reason = "cv_match_no_match"

        return {
            "status": status,
            "result": {
                "reason": reason,
                "engine": "cv_match",
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
                "ref_images_count": len(self.refs),
                "thresholds": {
                    "approve_min_good_matches": self.settings.approve_min_good_matches,
                    "manual_min_good_matches": self.settings.manual_min_good_matches,
                    "approve_score": self.settings.approve_score_threshold,
                    "manual_score": self.settings.manual_score_threshold,
                    "orb_distance": self.settings.orb_distance_threshold,
                },
                "hits_count": len(hits),
                "manual_candidates_count": len(manual_candidates),
                "max_score": max_score,
                "max_good_matches": max_good_matches,
                "top_samples": top_samples,
            },
        }


def _load_reference_paths(refs_dirs: list[str], max_refs: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths: list[Path] = []
    for refs_dir in refs_dirs:
        base = Path(refs_dir)
        if not base.exists():
            continue
        paths.extend(path for path in sorted(base.iterdir()) if path.is_file() and path.suffix.lower() in exts)
    if max_refs > 0:
        return paths[:max_refs]
    return paths


def _good_match_count(
    *,
    matcher: cv2.BFMatcher,
    left: np.ndarray | None,
    right: np.ndarray | None,
    distance_threshold: int,
) -> int:
    if left is None or right is None or len(left) == 0 or len(right) == 0:
        return 0
    matches = matcher.match(left, right)
    return sum(1 for match in matches if match.distance <= distance_threshold)


def _hsv_hist(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def _hist_score(left: np.ndarray, right: np.ndarray) -> float:
    correlation = float(cv2.compareHist(left, right, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, (correlation + 1.0) / 2.0))


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


def _resize_bgr(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / width
    next_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (max_width, next_height), interpolation=cv2.INTER_AREA)


def _make_crops(image: np.ndarray, mode: str) -> list[np.ndarray]:
    if mode == "none":
        return [image]

    height, width = image.shape[:2]

    def crop(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(1, min(width, x1))
        y1 = max(1, min(height, y1))
        return image[y0:y1, x0:x1]

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
