import argparse
import json
import os

from sam3_analyzer import Sam3VideoAnalyzer, settings_from_env


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze one video with persistent-compatible SAM3 runtime.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--model-id", default=os.getenv("VIDEO_SAM3_MODEL_ID", "facebook/sam3"))
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--approve-min-hits", type=int, default=None)
    parser.add_argument("--approve-min-unique-seconds", type=int, default=None)
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        default=None,
        help="Repeatable text prompt. If omitted, VIDEO_SAM3_PROMPTS or defaults are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = settings_from_env(
        model_id=args.model_id,
        sample_fps=args.sample_fps,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        min_mask_area_ratio=args.min_mask_area_ratio,
        approve_min_hits=args.approve_min_hits,
        approve_min_unique_seconds=args.approve_min_unique_seconds,
        prompts=args.prompts,
    )
    analyzer = Sam3VideoAnalyzer(settings)
    payload = analyzer.analyze(args.video_path, args.video_id)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
