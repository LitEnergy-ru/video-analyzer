# validate_hits.py
import os, json
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# -----------------------------
# PATHS
# -----------------------------
HITS_DIR = "hits"
META_PATH = os.path.join(HITS_DIR, "hits_meta.jsonl")

REFS_DIR = "refs/can"

OUT_ROOT = "hits_verified"
OUT_OK = os.path.join(OUT_ROOT, "ok")
OUT_REJECT = os.path.join(OUT_ROOT, "reject")

# -----------------------------
# MODELS
# -----------------------------
CLIP_MODEL = "openai/clip-vit-large-patch14"   # можно попробовать large: "openai/clip-vit-large-patch14"
YOLO_MODEL = "yolov8n.pt"                     # можно "yolov8s.pt" для выше recall (медленнее)

# -----------------------------
# THRESHOLDS (2nd contour)
# -----------------------------
# YOLO bbox path = строгий
THR_REF_STRICT_YOLO = 0.36
MARGIN_YOLO = 0.04

# GRID fallback path = мягче (мелкие объекты)
THR_REF_GRID = 0.26
MARGIN_GRID = 0.00

# общий порог "похоже на банку"
THR_CAN_TXT = 0.18

# -----------------------------
# YOLO SETTINGS
# -----------------------------
YOLO_CONF = 0.20
YOLO_IOU = 0.50
YOLO_MAX_DETS = 12

# COCO ids (обычно): bottle=39, wine glass=40, cup=41
KEEP_COCO = {39, 40, 41}

# -----------------------------
# TEXT GATING (can vs pipe)
# -----------------------------
CAN_TEXTS = [
    "a photo of an aluminum drink can",
    "a beverage can",
    "an energy drink can",
    "a soda can",
]
PIPE_TEXTS = [
    "a chrome metal pipe",
    "a car exhaust pipe",
    "a metal tube",
    "a metal rod",
]

# -----------------------------
# HELPERS
# -----------------------------
def load_ref_images(refs_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if not os.path.isdir(refs_dir):
        raise FileNotFoundError(f"Refs dir not found: {refs_dir}")
    paths = [os.path.join(refs_dir, f) for f in os.listdir(refs_dir)
             if os.path.splitext(f.lower())[1] in exts]
    if not paths:
        raise FileNotFoundError(f"No reference images in: {refs_dir}")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return imgs


@torch.inference_mode()
def encode_images(model, proc, pil_images, device):
    inputs = proc(images=pil_images, return_tensors="pt").to(device)
    feats = model.get_image_features(pixel_values=inputs["pixel_values"])
    if not torch.is_tensor(feats):
        if hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        else:
            raise TypeError(f"Unexpected get_image_features output: {type(feats)}")
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.inference_mode()
def encode_text(model, proc, texts, device):
    inputs = proc(text=texts, return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", None),
    )
    if not torch.is_tensor(feats):
        if hasattr(feats, "text_embeds") and feats.text_embeds is not None:
            feats = feats.text_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            feats = feats.pooler_output
        else:
            raise TypeError(f"Unexpected get_text_features output: {type(feats)}")
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def crop_xyxy(rgb, x1, y1, x2, y2, pad=0.12):
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(pad * bw), int(pad * bh)

    x1 = max(0, x1 - px); y1 = max(0, y1 - py)
    x2 = min(W, x2 + px); y2 = min(H, y2 + py)

    c = rgb[y1:y2, x1:x2]
    if c.size == 0:
        return None
    return c


def make_crops_like_first_pass(rgb, mode: str):
    # crops in the same order as your first contour (CROP_MODE="5"/"9"/"none")
    h, w = rgb.shape[:2]

    def crop(x0, y0, x1, y1):
        x0 = max(0, min(w - 1, int(x0))); x1 = max(1, min(w, int(x1)))
        y0 = max(0, min(h - 1, int(y0))); y1 = max(1, min(h, int(y1)))
        return rgb[y0:y1, x0:x1]

    if mode == "none":
        return [rgb]

    if mode == "5":
        cw, ch = int(w * 0.6), int(h * 0.6)
        return [
            crop((w - cw)//2, (h - ch)//2, (w + cw)//2, (h + ch)//2),  # 0 center
            crop(0, 0, cw, ch),                                         # 1 tl
            crop(w - cw, 0, w, ch),                                     # 2 tr
            crop(0, h - ch, cw, h),                                     # 3 bl
            crop(w - cw, h - ch, w, h),                                 # 4 br
        ]

    if mode == "9":
        xs = [0, w//3, 2*w//3, w]
        ys = [0, h//3, 2*h//3, h]
        out = []
        for j in range(3):
            for i in range(3):
                out.append(crop(xs[i], ys[j], xs[i+1], ys[j+1]))
        return out

    return [rgb]


def local_multiscale_grid(rgb_crop, grids=(4, 5), scales=(1.0, 1.35)):
    """
    Dense sliding windows inside crop.
    scales > 1.0 => smaller windows => "zoom-in" effect.
    """
    h, w = rgb_crop.shape[:2]
    pil = []

    for sc in scales:
        for g in grids:
            cell_w = w / g
            cell_h = h / g

            # overlap steps
            step_x = max(1, int(cell_w * 0.55))
            step_y = max(1, int(cell_h * 0.55))

            win_w = max(32, int(cell_w / sc))
            win_h = max(32, int(cell_h / sc))

            for y0 in range(0, max(1, h - win_h + 1), step_y):
                for x0 in range(0, max(1, w - win_w + 1), step_x):
                    c = rgb_crop[y0:y0+win_h, x0:x0+win_w]
                    if min(c.shape[:2]) >= 32:
                        pil.append(Image.fromarray(c))

    return pil


def upscale_small_pil(pil_images):
    out = []
    for im in pil_images:
        w, h = im.size
        m = min(w, h)
        if m < 140:
            # x3 for very small crops
            im = im.resize((w * 3, h * 3), Image.BICUBIC)
        elif m < 220:
            # x2 for medium-small
            im = im.resize((w * 2, h * 2), Image.BICUBIC)
        out.append(im)
    return out


def main():
    os.makedirs(OUT_OK, exist_ok=True)
    os.makedirs(OUT_REJECT, exist_ok=True)

    if not os.path.exists(META_PATH):
        raise FileNotFoundError(
            f"Meta not found: {META_PATH}\n"
            f"Need hits/hits_meta.jsonl from 1st contour (fields: file,crop_mode,crop_id)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch device:", device)

    # CLIP
    clip = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # reference vector
    ref_imgs = load_ref_images(REFS_DIR)
    ref_vec = encode_images(clip, proc, ref_imgs, device).mean(dim=0, keepdim=True)
    ref_vec = ref_vec / ref_vec.norm(dim=-1, keepdim=True)

    # text vectors
    can_vec = encode_text(clip, proc, CAN_TEXTS, device).mean(dim=0, keepdim=True)
    can_vec = can_vec / can_vec.norm(dim=-1, keepdim=True)

    pipe_vec = encode_text(clip, proc, PIPE_TEXTS, device).mean(dim=0, keepdim=True)
    pipe_vec = pipe_vec / pipe_vec.norm(dim=-1, keepdim=True)

    # YOLO
    yolo = YOLO(YOLO_MODEL)

    ok_cnt = 0
    bad_cnt = 0

    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img_path = rec.get("file")
            if not img_path or not os.path.exists(img_path):
                continue

            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # -------------------------
            # 1) YOLO bbox candidates
            # -------------------------
            mode = "YOLO"
            pil_crops = []

            res = yolo.predict(
                source=rgb,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                max_det=YOLO_MAX_DETS,
                verbose=False,
                device=0 if torch.cuda.is_available() else None,
            )[0]

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls = int(b.cls.item())
                    if cls not in KEEP_COCO:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    c = crop_xyxy(rgb, x1, y1, x2, y2, pad=0.12)
                    if c is None or min(c.shape[:2]) < 32:
                        continue
                    pil_crops.append(Image.fromarray(c))

            # -------------------------
            # 2) GRID fallback (if YOLO gave nothing)
            # -------------------------
            if not pil_crops:
                mode = "GRID"
                crop_mode = rec.get("crop_mode", "5")
                crop_id = int(rec.get("crop_id", 0))

                base_crops = make_crops_like_first_pass(rgb, crop_mode)
                roi = base_crops[crop_id] if 0 <= crop_id < len(base_crops) else base_crops[0]

                pil_crops = local_multiscale_grid(roi, grids=(4, 5), scales=(1.0, 1.35))
                pil_crops = upscale_small_pil(pil_crops)

            if not pil_crops:
                # nothing to validate
                continue

            # -------------------------
            # CLIP scoring
            # -------------------------
            feats = encode_images(clip, proc, pil_crops, device)

            s_ref = (feats @ ref_vec.T).squeeze(1).detach().cpu().numpy()
            s_can = (feats @ can_vec.T).squeeze(1).detach().cpu().numpy()
            s_pipe = (feats @ pipe_vec.T).squeeze(1).detach().cpu().numpy()

            i = int(np.argmax(s_ref))
            best_ref = float(s_ref[i])
            best_can = float(s_can[i])
            best_pipe = float(s_pipe[i])

            thr_ref = THR_REF_GRID if mode == "GRID" else THR_REF_STRICT_YOLO
            thr_margin = MARGIN_GRID if mode == "GRID" else MARGIN_YOLO

            ok = (
                (best_ref >= thr_ref) and
                (best_can >= THR_CAN_TXT) and
                ((best_can - best_pipe) >= thr_margin)
            )

            base = os.path.basename(img_path)
            margin = best_can - best_pipe
            print(f"[{base}] mode={mode} ref={best_ref:.3f} can={best_can:.3f} pipe={best_pipe:.3f} margin={margin:.3f} thr_ref={thr_ref:.2f}")

            if ok:
                out = os.path.join(OUT_OK, f"{mode}_ref{best_ref:.3f}_m{margin:.3f}_{base}")
                cv2.imwrite(out, bgr)
                ok_cnt += 1
            else:
                out = os.path.join(OUT_REJECT, f"{mode}_ref{best_ref:.3f}_m{margin:.3f}_{base}")
                cv2.imwrite(out, bgr)
                bad_cnt += 1

    print("\nDONE")
    print("ok:", ok_cnt)
    print("reject:", bad_cnt)
    print("saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()