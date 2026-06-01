# Video Worker

Worker flow per cycle:
- claim up to `VIDEO_BATCH_SIZE` submitted content tasks from backend DB:
  - `user_tasks.status = 'submitted'`
  - `tasks.type = 'content'`
  - `user_tasks.external_ref` contains the video URL
- mark claimed rows as `progress_json.analysisStatus = "processing"`
- download each video:
  - direct `http/https` video URLs via streaming HTTP
  - Reels/TikTok/Shorts/etc. via `yt-dlp` fallback when `VIDEO_DOWNLOADER=auto`
  - local paths / `file://` for local debugging
- process videos one by one via fast CPU analyzer, in-process SAM3 analyzer, or legacy `VIDEO_ANALYZER_CMD`
- PATCH backend task result:
  - `approved` -> `status = approved`
  - `rejected` -> `status = rejected`
  - `manual` / `invalid` / worker error -> keep `status = submitted`, set `progressJson.analysisStatus = "manual_review"`

The worker does not create or update a standalone `video` table anymore.

## 1) Prepare env

Set at least:

```env
VIDEO_DB_DSN=postgres://postgres:postgres@host.docker.internal:5432/litenergy?sslmode=disable
BACKEND_API_BASE_URL=http://host.docker.internal:8080/api/v1
BACKEND_SERVICE_TOKEN=service-token-with-core-tasks-moderate-scope
VIDEO_BATCH_SIZE=15
VIDEO_DOWNLOADER=auto
VIDEO_ANALYZER_ENGINE=fast_cpu
```

The backend service token must have `core:tasks:moderate` scope. The worker sends `X-User-Id` from `user_tasks.user_id` when calling:

```http
PATCH /api/v1/me/tasks/{task_id}
```

By default `VIDEO_ANALYZER_ENGINE=fast_cpu` loads a lightweight CLIP model once, samples a few frames from each video, compares them with local Lit Energy reference images and returns `approved`, `manual`, or `rejected`. This is the cheap CPU-friendly path for basic moderation.

Set `VIDEO_ANALYZER_ENGINE=sam3` when you need the heavier segmentation-based path. It keeps `facebook/sam3` loaded inside the worker process and reuses it across videos.

Set `VIDEO_ANALYZER_ENGINE=none` to skip automatic analysis and send tasks to manual review.

Legacy subprocess mode is still supported with `VIDEO_ANALYZER_ENGINE=subprocess`. In that mode `VIDEO_ANALYZER_CMD` must print JSON:

```json
{"status":"approved","result":{"score":0.97}}
```

Legacy SAM3 subprocess command:

```env
VIDEO_ANALYZER_ENGINE=subprocess
VIDEO_ANALYZER_CMD=python analyze_video_sam3.py --video-path "{video_path}" --video-id "{video_id}"
VIDEO_SAM3_MODEL_ID=facebook/sam3
VIDEO_SAM3_PROMPTS=lit energy can|lit energy drink can|energy drink can
```

## Fast CPU analyzer settings

The fast analyzer is intentionally coarse. It is designed to approve obvious Lit Energy videos, reject clear misses, and send weak matches to manual review.

Recommended CPU defaults:

```env
VIDEO_ANALYZER_ENGINE=fast_cpu
VIDEO_FAST_MODEL_ID=openai/clip-vit-base-patch32
VIDEO_FAST_REFS_DIRS=refs/can|refs/chips|refs/logo
VIDEO_FAST_TEXT_PROMPTS=lit energy drink can|lit energy chips|lit energy logo|energy drink can
VIDEO_FAST_DEVICE=cpu
VIDEO_FAST_SAMPLE_FPS=0.25
VIDEO_FAST_MAX_SAMPLED_FRAMES=8
VIDEO_FAST_MAX_WIDTH=512
VIDEO_FAST_CROP_MODE=5
VIDEO_FAST_APPROVE_REF_THRESHOLD=0.30
VIDEO_FAST_APPROVE_TEXT_THRESHOLD=0.20
VIDEO_FAST_MANUAL_REF_THRESHOLD=0.25
VIDEO_FAST_MANUAL_TEXT_THRESHOLD=0.16
VIDEO_FAST_APPROVE_MIN_HITS=1
```

Lower thresholds approve more videos but increase false positives. Higher thresholds send more videos to manual review.

## SAM3 performance settings

The optimized in-process path does three important things:

- loads SAM3 once per worker instead of once per video;
- computes the frame vision embedding once and reuses it for all text prompts;
- stops early after enough unique hit seconds when `VIDEO_SAM3_EARLY_APPROVE=true`.

Recommended defaults:

```env
VIDEO_ANALYZER_ENGINE=sam3
VIDEO_SAM3_MODEL_ID=facebook/sam3
VIDEO_SAM3_PROMPTS=lit energy can|lit energy drink can|energy drink can
VIDEO_SAM3_SAMPLE_FPS=2
VIDEO_SAM3_DTYPE=auto
VIDEO_SAM3_IMAGE_SIZE=0
VIDEO_SAM3_EARLY_APPROVE=true
VIDEO_SAM3_MAX_SAMPLED_FRAMES=0
```

`VIDEO_SAM3_DTYPE=auto` uses fp16 on CUDA and fp32 on CPU. `VIDEO_SAM3_IMAGE_SIZE=0` keeps the model default resolution. Setting `768` or `560` is faster and uses less VRAM, but can reduce accuracy.

Throughput still scales with:

```text
video duration * VIDEO_SAM3_SAMPLE_FPS * number of prompts
```

If queue latency matters more than recall, lower `VIDEO_SAM3_SAMPLE_FPS` to `1`.

Downloader settings:

```env
VIDEO_DOWNLOADER=auto
VIDEO_PROXY_URL=http://user:password@proxy-host:3128
VIDEO_YTDLP_FORMAT=best[ext=mp4]/best
VIDEO_YTDLP_TIMEOUT_SEC=300
VIDEO_DOWNLOAD_MAX_SIZE_MB=500
```

`auto` first tries direct HTTP for real video files and falls back to `yt-dlp` for social URLs. For Instagram/TikTok links that require an authenticated session, mount a cookies file and set:

```env
VIDEO_YTDLP_COOKIES_FILE=/run/secrets/video_ytdlp_cookies.txt
```

`VIDEO_PROXY_URL` is optional. When set, the worker passes it to both direct HTTP downloads and `yt-dlp`. Supported forms are:

```env
VIDEO_PROXY_URL=http://user:password@proxy-host:3128
VIDEO_PROXY_URL=socks5h://user:password@proxy-host:1080
```

Download metadata stores only a redacted proxy URL, so credentials are not written to task progress JSON.

`yt-dlp` supports many public video platforms, but sites change often. If extraction starts failing, update the worker image so `yt-dlp` is current.

## GPU runtime

The production Dockerfile uses a CUDA PyTorch image and the compose file requests GPU access. On the host install:

- NVIDIA driver
- Docker
- NVIDIA Container Toolkit

Check GPU visibility:

```bash
docker compose -f docker-compose.worker.yml run --rm video-worker python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

If this prints `False cpu`, SAM3 will run on CPU and will be too slow for production.

Available placeholders:
- `{video_path}`
- `{video_id}`: `user_tasks.id`
- `{task_id}`
- `{user_id}`

## 2) Run as always-on container

```powershell
docker compose -f docker-compose.worker.yml up -d --build
```

CPU VPS without NVIDIA runtime:

```powershell
docker compose -f docker-compose.worker.yml -f docker-compose.worker.cpu.yml up -d --build
```

The CPU override uses `python:3.12-slim` and installs `torch==2.4.1+cpu` from the official PyTorch CPU wheel index. Do not use `pytorch/pytorch:*cpu` tags; the official PyTorch Docker repository does not publish the `2.4.1-cpu` tag.

Stop:

```powershell
docker compose -f docker-compose.worker.yml down
```

Logs:

```powershell
docker compose -f docker-compose.worker.yml logs -f video-worker
```

## 3) Local run

```powershell
pip install -r requirements-worker.txt
python worker.py
```
