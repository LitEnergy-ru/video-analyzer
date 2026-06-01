import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json

# --------------------------
# SETTINGS (CPU-friendly)
# --------------------------
VIDEO_PATH = "6.mp4"

REFS_DIR = "refs/can"           # put 5-10 reference images of the Lit Energy can/logo here
SAVE_DIR = "hits"
META_PATH = os.path.join(SAVE_DIR, "hits_meta.jsonl")

EVERY_N_FRAMES = 2          # for 32s video: 2 => ~15 fps at 30fps (very sensitive). Try 3-5 if slow.
MAX_W = 640                 # resize frames for speed

CROP_MODE = "5"             # "none" / "5" / "9"
TOP_K_PRINT = 12            # print best candidates
TOP_K_SAVE = 10             # save top matches above thresholds

# Dual-threshold (image-to-image AND image-to-text)
THRESH_REF = 0.32           # similarity to refs (raise if false positives)
THRESH_TXT = 0.22           # similarity to text prompt (raise to kill TikTok-like screens)
MIN_STREAK = 2              # require N consecutive hits in time

TEXT_QUERIES = [
    "a can of energy drink",
    "energy drink",
    "lit energy can"
]

# Optional: reduce HF warnings on Windows symlinks (doesn't affect functionality)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def resize_rgb(rgb: np.ndarray, max_w: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if w <= max_w:
        return rgb
    scale = max_w / w
    nh = int(h * scale)
    return cv2.resize(rgb, (max_w, nh), interpolation=cv2.INTER_AREA)


def make_crops(rgb: np.ndarray, mode: str = "none"):
    if mode == "none":
        return [rgb]
    h, w = rgb.shape[:2]

    def crop(x0, y0, x1, y1):
        x0 = max(0, min(w - 1, x0)); x1 = max(1, min(w, x1))
        y0 = max(0, min(h - 1, y0)); y1 = max(1, min(h, y1))
        return rgb[y0:y1, x0:x1]

    if mode == "5":
        cw, ch = int(w * 0.6), int(h * 0.6)
        return [
            crop((w - cw)//2, (h - ch)//2, (w + cw)//2, (h + ch)//2),  # center
            crop(0, 0, cw, ch),                                         # tl
            crop(w - cw, 0, w, ch),                                     # tr
            crop(0, h - ch, cw, h),                                     # bl
            crop(w - cw, h - ch, w, h),                                 # br
        ]
    if mode == "9":
        xs = [0, w//3, 2*w//3, w]
        ys = [0, h//3, 2*h//3, h]
        crops = []
        for j in range(3):
            for i in range(3):
                crops.append(crop(xs[i], ys[j], xs[i+1], ys[j+1]))
        return crops

    return [rgb]


def load_reference_images(refs_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not os.path.isdir(refs_dir):
        raise FileNotFoundError(f"Refs dir not found: {refs_dir}")

    paths = []
    for fn in os.listdir(refs_dir):
        if os.path.splitext(fn.lower())[1] in exts:
            paths.append(os.path.join(refs_dir, fn))
    if not paths:
        raise FileNotFoundError(f"No reference images found in: {refs_dir}")

    imgs = [Image.open(p).convert("RGB") for p in paths]
    return paths, imgs


@torch.inference_mode()
def encode_images(model, processor, pil_images, device):
    """
    Robust image encoder that avoids CLIPModel.forward() requiring text input_ids.
    Works across transformers versions.
    """
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    feats = model.get_image_features(pixel_values=inputs["pixel_values"])

    # Some versions may return an output object instead of a raw tensor
    if not torch.is_tensor(feats):
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        else:
            raise TypeError(f"Unexpected get_image_features output type: {type(feats)}")

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.inference_mode()
def encode_text(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

    feats = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
    )

    # В некоторых версиях transformers возвращается output-объект
    if not torch.is_tensor(feats):
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            feats = feats.text_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        else:
            raise TypeError(f"Unexpected get_text_features output type: {type(feats)}")

    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Load refs and build mean reference vector
    ref_paths, ref_imgs = load_reference_images(REFS_DIR)
    print(f"Loaded {len(ref_imgs)} reference images:")
    for p in ref_paths:
        print(" -", p)

    ref_feats = encode_images(model, processor, ref_imgs, device)
    ref_vec = ref_feats.mean(dim=0, keepdim=True)
    ref_vec = ref_vec / ref_vec.norm(dim=-1, keepdim=True)

    # Build mean text vector (helps reject “style-only” false positives like TikTok screen)
    text_feats = encode_text(model, processor, TEXT_QUERIES, device)
    text_vec = text_feats.mean(dim=0, keepdim=True)
    text_vec = text_vec / text_vec.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total_frames / fps) if total_frames else None
    print(f"Video FPS={fps:.2f}, frames={total_frames}, duration={duration:.2f}s" if duration else f"Video FPS={fps:.2f}")

    # Store per-sample best scores
    samples = []  # (frame_idx, t_sec, best_ref, best_txt, best_crop)

    frame_idx = 0
    checked = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        if frame_idx % EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        t_sec = frame_idx / fps
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = resize_rgb(rgb, MAX_W)

        crops = make_crops(rgb, CROP_MODE)
        pil_crops = [Image.fromarray(c) for c in crops]

        crop_feats = encode_images(model, processor, pil_crops, device)

        sims_ref = (crop_feats @ ref_vec.T).squeeze(1).detach().cpu().numpy()
        sims_txt = (crop_feats @ text_vec.T).squeeze(1).detach().cpu().numpy()

        best_crop = int(np.argmax(sims_ref))
        best_ref = float(sims_ref[best_crop])
        best_txt = float(sims_txt[best_crop])

        samples.append((frame_idx, t_sec, best_ref, best_txt, best_crop))
        checked += 1

        if checked == 1 or checked % 20 == 0:
            print(f"Processed {checked} samples, t={t_sec:.2f}s best_ref={best_ref:.3f} best_txt={best_txt:.3f}")

        frame_idx += 1

    cap.release()

    # Print top candidates by ref score (debug)
    top_by_ref = sorted(samples, key=lambda x: x[2], reverse=True)[:TOP_K_PRINT]
    print("\nTOP candidates (by ref score):")
    for fidx, t, ref_s, txt_s, crop in top_by_ref:
        print(f"  t={t:6.2f}s frame={fidx:5d} crop={crop}  ref={ref_s:.3f}  txt={txt_s:.3f}")

    # Determine presence by requiring a streak of consecutive hits in time
    samples_sorted = sorted(samples, key=lambda x: x[0])  # by frame_idx
    streak = 0
    best_streak = 0
    found = False
    first_hit_frame = None
    hit_windows = []  # store hit samples for saving

    for fidx, t, ref_s, txt_s, crop in samples_sorted:
        ok = (ref_s >= THRESH_REF) and (txt_s >= THRESH_TXT) and (0.14 <= ref_s * txt_s)
        if ok:
            streak += 1
            hit_windows.append((fidx, t, ref_s, txt_s, crop))
            if first_hit_frame is None:
                first_hit_frame = fidx
            best_streak = max(best_streak, streak)
            if streak >= MIN_STREAK:
                found = True
                break
        else:
            streak = 0
            first_hit_frame = None
            hit_windows = []

    print("\nThresholds:")
    print(f"  THRESH_REF={THRESH_REF}  THRESH_TXT={THRESH_TXT}  MIN_STREAK={MIN_STREAK}")
    print("RESULT:", "YES (likely Lit Energy present)" if found else "NO (not confident)")

    # Save frames above thresholds (top matches by combined score)
    if found:
        # pick top save candidates by a combined score
        top_save = sorted(samples, key=lambda x: (x[2] + x[3]), reverse=True)
        top_save = [x for x in top_save if (x[2] >= THRESH_REF and x[3] >= THRESH_TXT)][:TOP_K_SAVE]

        cap = cv2.VideoCapture(VIDEO_PATH)
        saved = 0
        for fidx, t, ref_s, txt_s, crop in top_save:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, bgr = cap.read()
            if not ok:
                continue
            out_path = os.path.join(SAVE_DIR, f"hit_{t:.2f}s_ref_{ref_s:.3f}_txt_{txt_s:.3f}.jpg")
            cv2.imwrite(out_path, bgr)
            with open(META_PATH, "a", encoding="utf-8") as f:
                rec = {
                    "video": VIDEO_PATH,
                    "frame_idx": int(fidx),
                    "t_sec": float(t),
                    "ref": float(ref_s),
                    "txt": float(txt_s),
                    "crop_id": int(crop),
                    "file": out_path,
                    "crop_mode": CROP_MODE,
                    "max_w": MAX_W,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
        cap.release()

        print(f"Saved {saved} frames to ./{SAVE_DIR}/")
        if saved == 0:
            print("Note: no frames saved—try lowering thresholds slightly or add more reference images.")


if __name__ == "__main__":
    main()