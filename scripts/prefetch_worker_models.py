import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch worker model files into a Hugging Face cache directory.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--fast-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prefetch-fast-model", default="false")
    return parser.parse_args()


def is_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def prefetch_fast_model(model_id: str, cache_dir: str) -> None:
    from transformers import CLIPModel, CLIPProcessor

    print(f"[prefetch] downloading CLIP model: {model_id}")
    CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
    CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    print(f"[prefetch] CLIP model cached in {cache_dir}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    if is_enabled(args.prefetch_fast_model):
        prefetch_fast_model(args.fast_model_id, args.cache_dir)
    else:
        print("[prefetch] fast model prefetch disabled")


if __name__ == "__main__":
    main()
