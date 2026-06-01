import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    Sam3Processor,
    Sam3Model,
    CLIPProcessor,
    CLIPModel,
)

# =========================
# CONFIG
# =========================
MODEL_ID = "facebook/sam3"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

VIDEO_PATH = "1.mp4"
OUTPUT_DIR = "sam3_lit_frames"
REFERENCE_DIR = "refs/can"   # сюда положи фото банок Lit

# Общий промпт для поиска банок.
# Не указываем бренд, чтобы SAM не путался в названиях,
# а просто находим кандидаты-банки.
TEXT_PROMPT = "energy drink can"

# Сохранять каждый N-й кадр
FRAME_STEP = 15

# SAM: порог уверенности детекции
SAM_SCORE_THRESHOLD = 0.45

# CLIP: порог похожести на референсы Lit
# Обычно нормальный диапазон 0.20 - 0.40, подбирается экспериментально.
CLIP_SIM_THRESHOLD = 0.28

# Минимальный размер бокса, чтобы отсеять мусор
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 40

# Прозрачность маски
MASK_ALPHA = 0.35

# Если True: классифицируем не весь bbox, а объект на белом фоне по маске
USE_MASKED_CROP_FOR_CLIP = True

# Если True: сохраняем и кадры без Lit-объектов
SAVE_EMPTY_FRAMES = True

HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

if HF_TOKEN is None:
    raise RuntimeError("Не найден HF_TOKEN в переменных окружения.")

if not os.path.isdir(REFERENCE_DIR):
    raise RuntimeError(f"Папка с референсами не найдена: {REFERENCE_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("[INFO] Loading SAM3...")
sam_processor = Sam3Processor.from_pretrained(MODEL_ID, token=HF_TOKEN)
sam_model = Sam3Model.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
sam_model.eval()

print("[INFO] Loading CLIP...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_model.eval()

# =========================
# HELPERS
# =========================
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)

def encode_pil_images_with_clip(images: list[Image.Image]) -> torch.Tensor:
    """
    Возвращает нормализованные image embeddings [N, D]
    без использования текстовой ветки CLIP.
    """
    inputs = clip_processor(
        images=images,
        return_tensors="pt",
        padding=True
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)

        # pooled output из vision encoder
        pooled_output = vision_outputs.pooler_output

        # прогоняем через visual projection CLIP
        image_features = clip_model.visual_projection(pooled_output)

    image_features = l2_normalize(image_features, dim=-1)
    return image_features


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Накладывает полупрозрачную маску на изображение.
    image_bgr: HxWx3 uint8
    mask: HxW, 0/1 или bool
    """
    out = image_bgr.copy()

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    color = np.zeros_like(out, dtype=np.uint8)
    color[:, :, 1] = 255  # green

    masked = np.where(mask[..., None] > 0, color, 0)
    out = cv2.addWeighted(out, 1.0, masked, alpha, 0)
    return out


def safe_box(box: np.ndarray, width: int, height: int):
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))
    return x1, y1, x2, y2


def extract_crop(
    frame_bgr: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray | None = None,
    use_masked_crop: bool = True,
) -> np.ndarray | None:
    """
    Вырезает crop по bbox.
    Если use_masked_crop=True и есть mask, оставляет объект по маске
    на белом фоне — часто это помогает CLIP лучше сравнивать банки.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = safe_box(box, w, h)

    if x2 <= x1 or y2 <= y1:
        return None

    if (x2 - x1) < MIN_BOX_WIDTH or (y2 - y1) < MIN_BOX_HEIGHT:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if not use_masked_crop or mask is None:
        return crop

    mask = mask.astype(np.uint8)
    local_mask = mask[y1:y2, x1:x2]
    if local_mask.size == 0:
        return crop

    # Нормализуем маску в 0/1
    local_mask = (local_mask > 0).astype(np.uint8)

    # Белый фон, объект сохраняем
    white_bg = np.full_like(crop, 255, dtype=np.uint8)
    masked_crop = np.where(local_mask[..., None] > 0, crop, white_bg)

    return masked_crop


def draw_predictions(
    frame_bgr: np.ndarray,
    detections: list[dict],
    alpha: float = 0.35,
) -> tuple[np.ndarray, int]:
    """
    detections: список словарей
    [
      {
        "box": np.ndarray shape(4,),
        "sam_score": float,
        "clip_score": float,
        "mask": np.ndarray(H, W),
      }
    ]
    """
    vis = frame_bgr.copy()
    count = 0

    for det in detections:
        box = det["box"]
        sam_score = det["sam_score"]
        clip_score = det["clip_score"]
        mask = det["mask"]

        x1, y1, x2, y2 = box.astype(int)
        mask_bin = (mask > 0).astype(np.uint8)

        vis = overlay_mask(vis, mask_bin, alpha=alpha)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"Lit | clip={clip_score:.3f} | sam={sam_score:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        count += 1

    return vis, count


def run_sam3_on_frame(frame_bgr: np.ndarray):
    """
    Прогоняет один кадр через SAM3.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    inputs = sam_processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    results = sam_processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM_SCORE_THRESHOLD,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]

    return results


# =========================
# CLIP REFERENCE EMBEDDINGS
# =========================
def list_reference_images(reference_dir: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for name in os.listdir(reference_dir):
        path = os.path.join(reference_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in exts:
            files.append(path)
    files.sort()
    return files


def encode_pil_images_with_clip(images: list[Image.Image]) -> torch.Tensor:
    """
    Возвращает нормализованные image embeddings [N, D]
    без использования текстовой ветки CLIP.
    """
    inputs = clip_processor(
        images=images,
        return_tensors="pt",
        padding=True
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=pixel_values)

        # pooled output из vision encoder
        pooled_output = vision_outputs.pooler_output

        # прогоняем через visual projection CLIP
        image_features = clip_model.visual_projection(pooled_output)

    image_features = l2_normalize(image_features, dim=-1)
    return image_features


def build_reference_embedding(reference_dir: str) -> tuple[torch.Tensor, list[str]]:
    ref_paths = list_reference_images(reference_dir)
    if len(ref_paths) == 0:
        raise RuntimeError(f"В {reference_dir} нет картинок-референсов.")

    ref_images = []
    valid_paths = []

    for path in ref_paths:
        try:
            img = Image.open(path).convert("RGB")
            ref_images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"[WARN] Failed to read ref image {path}: {e}")

    if len(ref_images) == 0:
        raise RuntimeError("Не удалось прочитать ни одного референса.")

    ref_embs = encode_pil_images_with_clip(ref_images)  # [N, D]
    mean_emb = ref_embs.mean(dim=0, keepdim=True)
    mean_emb = l2_normalize(mean_emb, dim=-1)           # [1, D]

    return mean_emb, valid_paths


def compute_crop_similarity_to_lit(crop_bgr: np.ndarray, lit_ref_embedding: torch.Tensor) -> float:
    """
    Возвращает cosine similarity crop vs mean Lit embedding.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    crop_emb = encode_pil_images_with_clip([crop_pil])  # [1, D]
    sim = (crop_emb @ lit_ref_embedding.T).squeeze().item()
    return float(sim)


# =========================
# FILTER DETECTIONS WITH CLIP
# =========================
def filter_lit_detections(
    frame_bgr: np.ndarray,
    boxes,
    scores,
    masks,
    lit_ref_embedding: torch.Tensor,
    sam_score_threshold: float = 0.45,
    clip_sim_threshold: float = 0.28,
    use_masked_crop: bool = True,
) -> list[dict]:
    """
    Оставляет только те детекции, которые похожи на Lit по CLIP.
    """
    detections = []

    if boxes is None or len(boxes) == 0:
        return detections

    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()

    for i, sam_score in enumerate(scores_np):
        sam_score = float(sam_score)
        if sam_score < sam_score_threshold:
            continue

        box = boxes_np[i]
        mask = masks_np[i]

        # Нормализуем маску в 0/1
        mask_bin = (mask > 0).astype(np.uint8)

        crop = extract_crop(
            frame_bgr=frame_bgr,
            box=box,
            mask=mask_bin,
            use_masked_crop=use_masked_crop,
        )

        if crop is None or crop.size == 0:
            continue

        try:
            clip_score = compute_crop_similarity_to_lit(crop, lit_ref_embedding)
        except Exception as e:
            print(f"[WARN] CLIP failed on detection {i}: {e}")
            continue

        if clip_score >= clip_sim_threshold:
            detections.append(
                {
                    "box": box,
                    "sam_score": sam_score,
                    "clip_score": clip_score,
                    "mask": mask_bin,
                }
            )

    return detections


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Building Lit reference embedding from: {REFERENCE_DIR}")
    lit_ref_embedding, used_ref_paths = build_reference_embedding(REFERENCE_DIR)

    print("[INFO] Reference images used:")
    for p in used_ref_paths:
        print("   ", p)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    frame_idx = 0
    saved_idx = 0
    total_lit_found = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        try:
            results = run_sam3_on_frame(frame)

            boxes = results.get("boxes")
            scores = results.get("scores")
            masks = results.get("masks")

            lit_detections = filter_lit_detections(
                frame_bgr=frame,
                boxes=boxes,
                scores=scores,
                masks=masks,
                lit_ref_embedding=lit_ref_embedding,
                sam_score_threshold=SAM_SCORE_THRESHOLD,
                clip_sim_threshold=CLIP_SIM_THRESHOLD,
                use_masked_crop=USE_MASKED_CROP_FOR_CLIP,
            )

            vis, found_count = draw_predictions(
                frame_bgr=frame,
                detections=lit_detections,
                alpha=MASK_ALPHA,
            )

            total_lit_found += found_count

            if SAVE_EMPTY_FRAMES or found_count > 0:
                out_name = f"frame_{frame_idx:06d}_lit_{found_count}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, vis)
                print(
                    f"[OK] frame={frame_idx} lit_found={found_count} "
                    f"saved={out_path}"
                )
                saved_idx += 1
            else:
                print(f"[SKIP] frame={frame_idx} lit_found=0")

        except Exception as e:
            print(f"[ERR] frame={frame_idx}: {e}")

        frame_idx += 1

    cap.release()
    print(f"[DONE] Saved frames: {saved_idx}")
    print(f"[DONE] Total Lit detections: {total_lit_found}")


if __name__ == "__main__":
    main()
