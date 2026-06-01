import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def _default_prompts() -> list[str]:
    return ["lit energy can", "lit energy drink can", "energy drink can"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual SAM3 video debug: draw masks/bboxes on a specific video and save artifacts."
    )
    parser.add_argument("--video-path", required=True, help="Path to local video file")
    parser.add_argument("--output-dir", default="", help="Directory for annotated video, frames and summary")
    parser.add_argument("--model-id", default=os.getenv("VIDEO_SAM3_MODEL_ID", "facebook/sam3"))
    parser.add_argument("--sample-fps", type=float, default=float(os.getenv("VIDEO_SAM3_SAMPLE_FPS", "2.0")))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("VIDEO_SAM3_SCORE_THRESHOLD", "0.5")))
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=float(os.getenv("VIDEO_SAM3_MASK_THRESHOLD", "0.5")),
    )
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=float(os.getenv("VIDEO_SAM3_MIN_MASK_AREA_RATIO", "0.001")),
    )
    parser.add_argument("--prompt", action="append", dest="prompts", default=None)
    parser.add_argument(
        "--save-all-sampled",
        action="store_true",
        help="Save every sampled frame, not only frames with detections",
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Optional limit for sampled frames to process, 0 means no limit",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not render annotated mp4, save only frames + summary",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"video not found: {args.video_path}")
    if args.sample_fps <= 0:
        raise ValueError("sample-fps must be > 0")

    if not args.output_dir:
        stem = Path(args.video_path).stem
        args.output_dir = str(Path("debug_sam3") / stem)

    args.prompts = args.prompts or _default_prompts()
    return args


def load_model(model_id: str, device: str) -> tuple[Sam3Model, Sam3Processor]:
    model = Sam3Model.from_pretrained(model_id).to(device)
    processor = Sam3Processor.from_pretrained(model_id)
    model.eval()
    return model, processor


def to_numpy_mask(mask: Any) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def detect_prompt(
    model: Sam3Model,
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    threshold: float,
    mask_threshold: float,
    min_mask_area_ratio: float,
    device: str,
) -> list[dict[str, Any]]:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    original_sizes = inputs.get("original_sizes")
    target_sizes = original_sizes.tolist() if original_sizes is not None else [[image.height, image.width]]
    post = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )
    if not post:
        return []

    result = post[0]
    masks = result.get("segmentation", [])
    labels = result.get("labels", [])
    scores = result.get("scores", [])

    image_area = float(image.width * image.height)
    hits: list[dict[str, Any]] = []
    for idx, mask in enumerate(masks):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score < threshold:
            continue

        mask_np = to_numpy_mask(mask)
        mask_area = int(mask_np.sum())
        area_ratio = float(mask_area / image_area) if image_area > 0 else 0.0
        if area_ratio < min_mask_area_ratio:
            continue

        bbox = mask_bbox(mask_np)
        if bbox is None:
            continue

        label = labels[idx] if idx < len(labels) else None
        hits.append(
            {
                "prompt": prompt,
                "label": None if label is None else str(label),
                "score": round(score, 4),
                "bbox": bbox,
                "mask_area": mask_area,
                "mask_area_ratio": round(area_ratio, 6),
                "mask": mask_np,
            }
        )

    return hits


def overlay_hits(frame_bgr: np.ndarray, hits: list[dict[str, Any]]) -> np.ndarray:
    overlay = frame_bgr.copy()
    colors = [
        (57, 255, 20),
        (0, 215, 255),
        (255, 128, 0),
        (255, 0, 255),
        (0, 128, 255),
    ]

    for idx, hit in enumerate(hits):
        color = colors[idx % len(colors)]
        mask = hit["mask"]
        bbox = hit["bbox"]
        x1, y1, x2, y2 = bbox

        colored = np.zeros_like(frame_bgr)
        colored[:, :] = color
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.45, colored, 0.55, 0)[mask_bool]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f'{hit["prompt"]} {hit["score"]:.2f}'
        text_y = max(20, y1 - 10)
        cv2.putText(
            overlay,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def sanitized_hits_for_json(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for hit in hits:
        item = dict(hit)
        item.pop("mask", None)
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model_id, device)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(fps / args.sample_fps))) if fps > 0 else 1

    writer = None
    video_out_path = output_dir / "annotated.mp4"
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    frame_idx = 0
    sampled_idx = 0
    hit_frames = 0
    frames_summary: list[dict[str, Any]] = []

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        annotated = bgr
        frame_record: dict[str, Any] | None = None

        if frame_idx % frame_step == 0:
            sampled_idx += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            timestamp_sec = round(frame_idx / fps, 3) if fps > 0 else 0.0

            hits: list[dict[str, Any]] = []
            for prompt in args.prompts:
                hits.extend(
                    detect_prompt(
                        model=model,
                        processor=processor,
                        image=image,
                        prompt=prompt,
                        threshold=args.threshold,
                        mask_threshold=args.mask_threshold,
                        min_mask_area_ratio=args.min_mask_area_ratio,
                        device=device,
                    )
                )

            if hits:
                hit_frames += 1
                annotated = overlay_hits(bgr.copy(), hits)

            if hits or args.save_all_sampled:
                frame_name = f"frame_{frame_idx:06d}_{timestamp_sec:09.3f}.jpg"
                cv2.imwrite(str(frames_dir / frame_name), annotated)

            frame_record = {
                "frame_idx": frame_idx,
                "sampled_idx": sampled_idx,
                "timestamp_sec": timestamp_sec,
                "detections_count": len(hits),
                "detections": sanitized_hits_for_json(hits),
            }
            frames_summary.append(frame_record)

            if args.max_sampled_frames > 0 and sampled_idx >= args.max_sampled_frames:
                if writer is not None:
                    writer.write(annotated)
                break

        if writer is not None:
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    summary = {
        "video_path": args.video_path,
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "device": device,
        "fps": fps,
        "sample_fps": args.sample_fps,
        "frame_step": frame_step,
        "total_frames": total_frames,
        "sampled_frames": sampled_idx,
        "frames_with_hits": hit_frames,
        "prompts": args.prompts,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "annotated_video_path": None if args.no_video else str(video_out_path),
        "frames": frames_summary,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
