from worker import Config, prepare_input_video, DOWNLOADER_YTDLP
import os, json

test_video_url = os.getenv("TEST_VIDEO_URL", "").strip()
if not test_video_url:
    raise SystemExit("TEST_VIDEO_URL is required")

cfg = Config(
    dsn="",
    backend_api_base_url="",
    backend_service_token="",
    backend_timeout_sec=15,
    batch_size=1,
    poll_sec=1,
    claim_stale_after_sec=1800,
    worker_id="download-test",
    analyzer_engine="none",
    analyzer_cmd="",
    analyzer_timeout_sec=1200,
    one_shot=True,
    download_dir="./tmp-download-test",
    download_connect_timeout_sec=10,
    download_read_timeout_sec=120,
    download_max_size_mb=500,
    cleanup_downloaded=False,
    downloader=DOWNLOADER_YTDLP,
    proxy_url=os.getenv("VIDEO_PROXY_URL", ""),
    ytdlp_format="bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/best[vcodec^=avc1][ext=mp4]/best",
    ytdlp_cookies_file=os.getenv("VIDEO_YTDLP_COOKIES_FILE", ""),
    ytdlp_timeout_sec=300,
    ytdlp_extra_args=[],
)

path, downloaded, meta, err = prepare_input_video(
    cfg,
    {"id": "download-test", "url": test_video_url},
)

print(json.dumps({
    "path": path,
    "downloaded": downloaded,
    "meta": meta,
    "err": err,
}, ensure_ascii=False, indent=2))
