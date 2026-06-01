import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import psycopg
import requests
from psycopg.rows import dict_row

CLAIM_BATCH_QUERY = """
WITH picked AS (
    SELECT
        ut.id,
        ut.user_id,
        ut.task_id,
        ut.external_ref,
        GREATEST(t.target, 1) AS target
    FROM user_tasks ut
    JOIN tasks t ON t.task_id = ut.task_id
    WHERE ut.status = 'submitted'
      AND t.type = 'content'
      AND NULLIF(BTRIM(ut.external_ref), '') IS NOT NULL
      AND (
          ut.progress_json IS NULL
          OR COALESCE(ut.progress_json->>'analysisStatus', 'pending') IN ('pending', 'queued')
          OR (
              ut.progress_json->>'analysisStatus' = 'processing'
              AND COALESCE(NULLIF(ut.progress_json->>'pickedAt', '')::timestamptz, NOW())
                    < NOW() - (%s * INTERVAL '1 second')
          )
      )
    ORDER BY ut.completed_at NULLS FIRST, ut.updated_at, ut.id
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
UPDATE user_tasks ut
SET progress_json = (
        CASE
            WHEN jsonb_typeof(ut.progress_json) = 'object' THEN ut.progress_json
            ELSE '{}'::jsonb
        END
        || jsonb_build_object(
            'analysisStatus', 'processing',
            'workerId', %s,
            'pickedAt', NOW(),
            'attemptCount', COALESCE((ut.progress_json->>'attemptCount')::int, 0) + 1
        )
    ),
    updated_at = NOW()
FROM picked
WHERE ut.id = picked.id
RETURNING
    ut.id::text AS id,
    ut.user_id::text AS user_id,
    ut.task_id,
    ut.external_ref AS url,
    picked.target,
    COALESCE((ut.progress_json->>'attemptCount')::int, 0) AS attempt_count;
"""

ALLOWED_ANALYZER_STATUSES = {"approved", "rejected", "manual", "invalid"}
HTTP_SCHEMES = {"http", "https"}
FILE_SCHEME = "file"
DIRECT_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DIRECT_VIDEO_CONTENT_TYPES = {"video/", "application/octet-stream"}
DOWNLOADER_AUTO = "auto"
DOWNLOADER_HTTP = "http"
DOWNLOADER_YTDLP = "yt-dlp"
ANALYZER_NONE = "none"
ANALYZER_FAST_CPU = "fast_cpu"
ANALYZER_SAM3 = "sam3"
ANALYZER_SUBPROCESS = "subprocess"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


def _default_analyzer_engine() -> str:
    value = os.getenv("VIDEO_ANALYZER_ENGINE", "").strip().lower()
    if value:
        return value
    if os.getenv("VIDEO_ANALYZER_CMD", "").strip():
        return ANALYZER_SUBPROCESS
    return ANALYZER_FAST_CPU


def _pick_extension_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    suffix = Path(path).suffix.lower()
    if suffix in DIRECT_VIDEO_EXTENSIONS:
        return suffix
    return ".mp4"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    dsn: str
    backend_api_base_url: str
    backend_service_token: str
    backend_timeout_sec: int
    batch_size: int
    poll_sec: float
    claim_stale_after_sec: int
    worker_id: str
    analyzer_engine: str
    analyzer_cmd: str
    analyzer_timeout_sec: int
    one_shot: bool
    download_dir: str
    download_connect_timeout_sec: int
    download_read_timeout_sec: int
    download_max_size_mb: int
    cleanup_downloaded: bool
    downloader: str
    proxy_url: str
    ytdlp_format: str
    ytdlp_cookies_file: str
    ytdlp_timeout_sec: int
    ytdlp_extra_args: list[str]


@dataclass(frozen=True)
class PreparedJob:
    row: dict[str, Any]
    video_path: str
    downloaded: bool
    fetch_meta: dict[str, Any]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Video worker: claim submitted content tasks from user_tasks, download videos, "
            "analyze sequentially, and PATCH task status through backend."
        )
    )
    parser.add_argument("--dsn", default=os.getenv("VIDEO_DB_DSN", ""))
    parser.add_argument("--backend-api-base-url", default=os.getenv("BACKEND_API_BASE_URL", ""))
    parser.add_argument("--backend-service-token", default=os.getenv("BACKEND_SERVICE_TOKEN", ""))
    parser.add_argument(
        "--backend-timeout-sec",
        type=int,
        default=_env_int("BACKEND_REQUEST_TIMEOUT_SEC", 15),
    )
    parser.add_argument("--batch-size", type=int, default=_env_int("VIDEO_BATCH_SIZE", 15))
    parser.add_argument("--poll-sec", type=float, default=_env_float("VIDEO_POLL_SEC", 2.0))
    parser.add_argument(
        "--claim-stale-after-sec",
        type=int,
        default=_env_int("VIDEO_CLAIM_STALE_AFTER_SEC", 1800),
    )
    parser.add_argument("--worker-id", default=os.getenv("VIDEO_WORKER_ID", _default_worker_id()))
    parser.add_argument(
        "--analyzer-engine",
        choices=[ANALYZER_NONE, ANALYZER_FAST_CPU, ANALYZER_SAM3, ANALYZER_SUBPROCESS],
        default=_default_analyzer_engine(),
        help=(
            "fast_cpu uses lightweight CLIP sampling; sam3 keeps the segmentation model warm in-process; "
            "subprocess runs VIDEO_ANALYZER_CMD; none sends manual review."
        ),
    )
    parser.add_argument(
        "--analyzer-cmd",
        default=os.getenv("VIDEO_ANALYZER_CMD", ""),
        help=(
            "Shell command template. Supports placeholders {video_path}, {video_id}, {task_id}, {user_id}. "
            "Command must print JSON: {\"status\":\"approved|rejected|manual|invalid\",\"result\":{...}}."
        ),
    )
    parser.add_argument(
        "--analyzer-timeout-sec",
        type=int,
        default=_env_int("VIDEO_ANALYZER_TIMEOUT_SEC", 1200),
    )
    parser.add_argument("--one-shot", action="store_true")

    parser.add_argument("--download-dir", default=os.getenv("VIDEO_DOWNLOAD_DIR", "/tmp/video-jobs"))
    parser.add_argument(
        "--download-connect-timeout-sec",
        type=int,
        default=_env_int("VIDEO_DOWNLOAD_CONNECT_TIMEOUT_SEC", 10),
    )
    parser.add_argument(
        "--download-read-timeout-sec",
        type=int,
        default=_env_int("VIDEO_DOWNLOAD_READ_TIMEOUT_SEC", 120),
    )
    parser.add_argument(
        "--download-max-size-mb",
        type=int,
        default=_env_int("VIDEO_DOWNLOAD_MAX_SIZE_MB", 500),
    )
    parser.add_argument("--cleanup-downloaded", default=os.getenv("VIDEO_CLEANUP_DOWNLOADED", "true"), help="true|false")
    parser.add_argument(
        "--downloader",
        choices=[DOWNLOADER_AUTO, DOWNLOADER_HTTP, DOWNLOADER_YTDLP],
        default=os.getenv("VIDEO_DOWNLOADER", DOWNLOADER_AUTO),
        help="auto uses direct HTTP for direct video URLs and falls back to yt-dlp for social URLs",
    )
    parser.add_argument(
        "--proxy-url",
        default=os.getenv("VIDEO_PROXY_URL", ""),
        help="optional proxy URL for video downloads, for example http://user:pass@host:port or socks5://host:port",
    )
    parser.add_argument(
        "--ytdlp-format",
        default=os.getenv("VIDEO_YTDLP_FORMAT", "best[ext=mp4]/best"),
        help="yt-dlp format selector",
    )
    parser.add_argument("--ytdlp-cookies-file", default=os.getenv("VIDEO_YTDLP_COOKIES_FILE", ""))
    parser.add_argument(
        "--ytdlp-timeout-sec",
        type=int,
        default=_env_int("VIDEO_YTDLP_TIMEOUT_SEC", 300),
    )
    parser.add_argument(
        "--ytdlp-extra-args",
        default=os.getenv("VIDEO_YTDLP_EXTRA_ARGS", ""),
        help="additional yt-dlp args separated by spaces; keep empty unless needed",
    )

    args = parser.parse_args()
    if not args.dsn:
        raise ValueError("VIDEO_DB_DSN (or --dsn) is required")
    if not args.backend_api_base_url:
        raise ValueError("BACKEND_API_BASE_URL (or --backend-api-base-url) is required")
    if not args.backend_service_token:
        raise ValueError("BACKEND_SERVICE_TOKEN (or --backend-service-token) is required")
    if args.analyzer_engine == ANALYZER_SUBPROCESS and not args.analyzer_cmd:
        raise ValueError("VIDEO_ANALYZER_CMD is required when VIDEO_ANALYZER_ENGINE=subprocess")

    cleanup_downloaded = str(args.cleanup_downloaded).lower() in {"1", "true", "yes", "on"}

    return Config(
        dsn=args.dsn,
        backend_api_base_url=args.backend_api_base_url.rstrip("/"),
        backend_service_token=args.backend_service_token,
        backend_timeout_sec=args.backend_timeout_sec,
        batch_size=args.batch_size,
        poll_sec=args.poll_sec,
        claim_stale_after_sec=args.claim_stale_after_sec,
        worker_id=args.worker_id,
        analyzer_engine=args.analyzer_engine,
        analyzer_cmd=args.analyzer_cmd,
        analyzer_timeout_sec=args.analyzer_timeout_sec,
        one_shot=args.one_shot,
        download_dir=args.download_dir,
        download_connect_timeout_sec=args.download_connect_timeout_sec,
        download_read_timeout_sec=args.download_read_timeout_sec,
        download_max_size_mb=args.download_max_size_mb,
        cleanup_downloaded=cleanup_downloaded,
        downloader=args.downloader,
        proxy_url=args.proxy_url.strip(),
        ytdlp_format=args.ytdlp_format,
        ytdlp_cookies_file=args.ytdlp_cookies_file,
        ytdlp_timeout_sec=args.ytdlp_timeout_sec,
        ytdlp_extra_args=args.ytdlp_extra_args.split() if args.ytdlp_extra_args else [],
    )


def claim_batch(conn: psycopg.Connection[Any], config: Config) -> list[dict[str, Any]]:
    with conn.transaction():
        return conn.execute(
            CLAIM_BATCH_QUERY,
            (config.claim_stale_after_sec, config.batch_size, config.worker_id),
        ).fetchall()


def patch_backend_task(
    config: Config,
    row: dict[str, Any],
    status: str | None,
    progress: int | None,
    progress_json: dict[str, Any],
) -> None:
    task_id = str(row["task_id"])
    user_id = str(row["user_id"])
    url = f"{config.backend_api_base_url}/me/tasks/{quote(task_id, safe='')}"
    payload: dict[str, Any] = {
        "progressJson": progress_json,
    }
    if status:
        payload["status"] = status
    if progress is not None:
        payload["progress"] = progress

    response = requests.patch(
        url,
        headers={
            "Authorization": f"Bearer {config.backend_service_token}",
            "X-User-Id": user_id,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=config.backend_timeout_sec,
    )
    if response.status_code >= 300:
        raise RuntimeError(
            f"backend PATCH failed status={response.status_code} body={response.text[:1000]}"
        )


def finish_job(
    config: Config,
    row: dict[str, Any],
    analyzer_status: str,
    error_message: str | None,
    result_json: dict[str, Any] | None,
) -> None:
    safe_status = analyzer_status if analyzer_status in ALLOWED_ANALYZER_STATUSES else "manual"
    task_status: str | None = None
    progress: int | None = None
    analysis_status = safe_status

    if safe_status == "approved":
        task_status = "approved"
        progress = int(row.get("target") or 1)
    elif safe_status == "rejected":
        task_status = "rejected"
        progress = 0
    else:
        analysis_status = "manual_review"

    progress_json = {
        "analysisStatus": analysis_status,
        "analyzerStatus": safe_status,
        "workerId": config.worker_id,
        "attemptCount": int(row.get("attempt_count") or 0),
        "analyzedAt": _utc_now_iso(),
        "videoUrl": str(row.get("url") or ""),
        "result": result_json or {},
    }
    if error_message:
        progress_json["errorMessage"] = error_message

    patch_backend_task(config, row, task_status, progress, progress_json)


def _resolve_local_path(url: str) -> str | None:
    if not url:
        return None

    if os.path.exists(url):
        return os.path.abspath(url)

    parsed = urlparse(url)
    if parsed.scheme == FILE_SCHEME:
        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def _is_direct_video_url(url: str) -> bool:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path or "")).suffix.lower()
    return suffix in DIRECT_VIDEO_EXTENSIONS


def _is_video_content_type(value: str) -> bool:
    content_type = value.lower().split(";", 1)[0].strip()
    return any(content_type.startswith(prefix) for prefix in DIRECT_VIDEO_CONTENT_TYPES)


def _redact_proxy_url(proxy_url: str) -> str:
    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.netloc or "@" not in parsed.netloc:
        return proxy_url

    host_part = parsed.netloc.rsplit("@", 1)[1]
    return parsed._replace(netloc=f"***:***@{host_part}").geturl()


def _requests_proxies(proxy_url: str) -> dict[str, str] | None:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}


def _download_meta(source: str, config: Config, **fields: Any) -> dict[str, Any]:
    meta = {"source": source, **fields}
    if config.proxy_url:
        meta["proxy"] = _redact_proxy_url(config.proxy_url)
    return meta


def _find_downloaded_ytdlp_file(download_dir: str, video_id: str) -> str | None:
    base = Path(download_dir)
    if not base.exists():
        return None

    candidates = [
        path
        for path in base.glob(f"{video_id}*")
        if path.is_file() and path.suffix.lower() in DIRECT_VIDEO_EXTENSIONS
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _download_http_video(config: Config, video_id: str, url: str) -> tuple[str | None, dict[str, Any], str | None]:
    os.makedirs(config.download_dir, exist_ok=True)
    ext = _pick_extension_from_url(url)
    target_path = os.path.join(config.download_dir, f"{video_id}{ext}")
    max_bytes = config.download_max_size_mb * 1024 * 1024

    timeout = (config.download_connect_timeout_sec, config.download_read_timeout_sec)
    try:
        with requests.get(url, stream=True, timeout=timeout, proxies=_requests_proxies(config.proxy_url)) as response:
            if response.status_code != 200:
                return (
                    None,
                    _download_meta("http", config, reason="download_http_status", status_code=response.status_code),
                    "download failed",
                )

            content_type = response.headers.get("content-type", "")
            if content_type and not _is_video_content_type(content_type) and not _is_direct_video_url(url):
                return (
                    None,
                    _download_meta(
                        "http",
                        config,
                        reason="download_not_video_content_type",
                        content_type=content_type,
                        url=url,
                    ),
                    "not a direct video url",
                )

            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    content_length_value = int(content_length)
                except ValueError:
                    content_length_value = 0
                if content_length_value > max_bytes:
                    return (
                        None,
                        _download_meta("http", config, reason="download_too_large", content_length=content_length_value),
                        "download too large",
                    )

            downloaded = 0
            with open(target_path, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        out_file.close()
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        return (
                            None,
                            _download_meta("http", config, reason="download_too_large_stream", bytes=downloaded),
                            "download too large",
                        )
                    out_file.write(chunk)

        return target_path, _download_meta("http", config, bytes=downloaded, url=url), None
    except requests.RequestException as exc:
        if os.path.exists(target_path):
            os.remove(target_path)
        return None, _download_meta("http", config, reason="download_http_error"), str(exc)


def _download_ytdlp_video(config: Config, video_id: str, url: str) -> tuple[str | None, dict[str, Any], str | None]:
    os.makedirs(config.download_dir, exist_ok=True)
    output_template = os.path.join(config.download_dir, f"{video_id}.%(ext)s")
    max_filesize = f"{config.download_max_size_mb}M"

    command = [
        "python",
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--force-overwrites",
        "--restrict-filenames",
        "--format",
        config.ytdlp_format,
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        "--max-filesize",
        max_filesize,
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--socket-timeout",
        str(config.download_read_timeout_sec),
        "--output",
        output_template,
    ]
    if config.ytdlp_cookies_file:
        command.extend(["--cookies", config.ytdlp_cookies_file])
    if config.proxy_url:
        command.extend(["--proxy", config.proxy_url])
    command.extend(config.ytdlp_extra_args)
    command.append(url)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.ytdlp_timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, _download_meta("yt-dlp", config, reason="ytdlp_timeout", url=url), "yt-dlp timeout"
    except Exception as exc:  # pylint: disable=broad-except
        return None, _download_meta("yt-dlp", config, reason="ytdlp_exec_error", url=url), str(exc)

    if result.returncode != 0:
        return (
            None,
            _download_meta(
                "yt-dlp",
                config,
                reason="ytdlp_non_zero_exit",
                returncode=result.returncode,
                stderr=(result.stderr or "")[:3000],
                stdout=(result.stdout or "")[:1000],
                url=url,
            ),
            f"yt-dlp exited with code {result.returncode}",
        )

    downloaded_path = _find_downloaded_ytdlp_file(config.download_dir, video_id)
    if not downloaded_path:
        return (
            None,
            _download_meta(
                "yt-dlp",
                config,
                reason="ytdlp_output_missing",
                stdout=(result.stdout or "")[:1000],
                stderr=(result.stderr or "")[:1000],
                url=url,
            ),
            "yt-dlp output missing",
        )

    size_bytes = os.path.getsize(downloaded_path)
    if size_bytes > config.download_max_size_mb * 1024 * 1024:
        try:
            os.remove(downloaded_path)
        except OSError:
            pass
        return (
            None,
            _download_meta("yt-dlp", config, reason="download_too_large_ytdlp", bytes=size_bytes),
            "download too large",
        )

    return downloaded_path, _download_meta("yt-dlp", config, bytes=size_bytes, url=url), None


def prepare_input_video(config: Config, row: dict[str, Any]) -> tuple[str | None, bool, dict[str, Any], str | None]:
    url = str(row.get("url") or "")
    local_path = _resolve_local_path(url)
    if local_path:
        return local_path, False, {"source": "local_path", "url": url}, None

    parsed = urlparse(url)
    if parsed.scheme in HTTP_SCHEMES:
        video_id = str(row["id"])
        if config.downloader == DOWNLOADER_YTDLP:
            path, meta, err = _download_ytdlp_video(config, video_id, url)
            return path, path is not None, meta, err

        path, meta, err = _download_http_video(config, video_id, url)
        if not err and path:
            return path, True, meta, None

        if config.downloader == DOWNLOADER_AUTO:
            ytdlp_path, ytdlp_meta, ytdlp_err = _download_ytdlp_video(config, video_id, url)
            if not ytdlp_err and ytdlp_path:
                ytdlp_meta["directHttpFallback"] = meta
                return ytdlp_path, True, ytdlp_meta, None
            return (
                None,
                False,
                {"directHttp": meta, "ytdlp": ytdlp_meta},
                ytdlp_err or err or "download failed",
            )

        return None, False, meta, err

    return None, False, {"reason": "unsupported_url_scheme", "url": url}, "unsupported url"


def _payload_to_analyzer_result(payload: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    status = str(payload.get("status", "manual")).lower()
    if status not in ALLOWED_ANALYZER_STATUSES:
        return "manual", {"reason": "analyzer_unknown_status", "payload": payload}, None

    result_json = payload.get("result", {})
    if not isinstance(result_json, dict):
        result_json = {"raw_result": result_json}

    return status, result_json, None


class AnalyzerRunner:
    def __init__(self, config: Config):
        self.config = config
        self._fast_cpu = None
        self._sam3 = None

    def run(self, row: dict[str, Any], video_path: str) -> tuple[str, dict[str, Any], str | None]:
        if self.config.analyzer_engine == ANALYZER_NONE:
            return "manual", {"reason": "analyzer_disabled"}, None
        if self.config.analyzer_engine == ANALYZER_FAST_CPU:
            return self._run_fast_cpu(row, video_path)
        if self.config.analyzer_engine == ANALYZER_SAM3:
            return self._run_sam3(row, video_path)
        return run_subprocess_analyzer(self.config, row, video_path)

    def _run_fast_cpu(self, row: dict[str, Any], video_path: str) -> tuple[str, dict[str, Any], str | None]:
        if self._fast_cpu is None:
            from fast_cpu_analyzer import FastCpuVideoAnalyzer, settings_from_env

            self._fast_cpu = FastCpuVideoAnalyzer(settings_from_env())

        try:
            payload = self._fast_cpu.analyze(video_path, str(row["id"]))
        except Exception as exc:  # pylint: disable=broad-except
            return "manual", {"reason": "fast_cpu_analyzer_error"}, str(exc)

        return _payload_to_analyzer_result(payload)

    def _run_sam3(self, row: dict[str, Any], video_path: str) -> tuple[str, dict[str, Any], str | None]:
        if self._sam3 is None:
            from sam3_analyzer import Sam3VideoAnalyzer, settings_from_env

            self._sam3 = Sam3VideoAnalyzer(settings_from_env())

        try:
            payload = self._sam3.analyze(video_path, str(row["id"]))
        except Exception as exc:  # pylint: disable=broad-except
            return "manual", {"reason": "sam3_analyzer_error"}, str(exc)

        return _payload_to_analyzer_result(payload)


def run_subprocess_analyzer(config: Config, row: dict[str, Any], video_path: str) -> tuple[str, dict[str, Any], str | None]:
    if not config.analyzer_cmd:
        return "manual", {"reason": "analyzer_not_configured"}, None

    command = config.analyzer_cmd.format(
        video_path=video_path,
        video_id=str(row["id"]),
        task_id=str(row["task_id"]),
        user_id=str(row["user_id"]),
    )
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=config.analyzer_timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "manual", {"reason": "analyzer_timeout"}, "analyzer timeout"
    except Exception as exc:  # pylint: disable=broad-except
        return "manual", {"reason": "analyzer_exec_error"}, str(exc)

    if result.returncode != 0:
        return (
            "manual",
            {
                "reason": "analyzer_non_zero_exit",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[:3000],
            },
            f"analyzer exited with code {result.returncode}",
        )

    stdout = (result.stdout or "").strip()
    if not stdout:
        return "manual", {"reason": "analyzer_empty_output"}, None

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return "manual", {"reason": "analyzer_invalid_json", "stdout": stdout[:3000]}, None

    return _payload_to_analyzer_result(payload)


def prepare_jobs(config: Config, rows: list[dict[str, Any]]) -> list[PreparedJob]:
    prepared: list[PreparedJob] = []
    for row in rows:
        job_id = str(row["id"])
        video_path, downloaded, fetch_meta, fetch_error = prepare_input_video(config, row)
        if fetch_error or not video_path:
            finish_job(
                config,
                row=row,
                analyzer_status="manual",
                error_message=fetch_error or "failed to prepare input video",
                result_json={"fetch": fetch_meta, "jobId": job_id},
            )
            print(f"[worker] user_task_id={job_id} -> manual_review (prepare failed)")
            continue

        prepared.append(
            PreparedJob(
                row=row,
                video_path=video_path,
                downloaded=downloaded,
                fetch_meta=fetch_meta,
            )
        )
    return prepared


def process_prepared_job(config: Config, analyzer: AnalyzerRunner, job: PreparedJob) -> None:
    job_id = str(job.row["id"])
    try:
        status, analyzer_result, error_message = analyzer.run(job.row, job.video_path)
        result_json = {
            "fetch": job.fetch_meta,
            "analyzer": analyzer_result,
            "videoPath": job.video_path,
        }
        finish_job(
            config,
            row=job.row,
            analyzer_status=status,
            error_message=error_message,
            result_json=result_json,
        )
        print(f"[worker] user_task_id={job_id} -> {status}")
    except Exception as exc:  # pylint: disable=broad-except
        try:
            finish_job(
                config,
                row=job.row,
                analyzer_status="manual",
                error_message=f"unexpected worker error: {exc}",
                result_json={"reason": "worker_exception"},
            )
        except Exception as patch_exc:  # pylint: disable=broad-except
            print(f"[worker] user_task_id={job_id} -> backend patch failed: {patch_exc}")
        print(f"[worker] user_task_id={job_id} -> manual_review (exception)")
    finally:
        if config.cleanup_downloaded and job.downloaded and os.path.exists(job.video_path):
            try:
                os.remove(job.video_path)
            except OSError:
                pass


def run(config: Config) -> None:
    print(
        f"[worker] start worker_id={config.worker_id} batch_size={config.batch_size} "
        f"poll_sec={config.poll_sec} analyzer_engine={config.analyzer_engine} one_shot={config.one_shot}"
    )
    analyzer = AnalyzerRunner(config)
    with psycopg.connect(config.dsn, row_factory=dict_row) as conn:
        while True:
            rows = claim_batch(conn, config)
            if not rows:
                if config.one_shot:
                    print("[worker] no jobs, exit one-shot")
                    return
                time.sleep(config.poll_sec)
                continue

            print(f"[worker] claimed={len(rows)}")

            prepared_jobs = prepare_jobs(config, rows)
            for job in prepared_jobs:
                process_prepared_job(config, analyzer, job)


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
