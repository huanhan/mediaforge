"""Run a small, sequential generation matrix against the local H3 API.

The command intentionally submits one task at a time because the local service
uses a single GPU worker.  The pure parsing, media inspection, and report
helpers are kept importable so they can be tested without a running service.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_PROMPT = (
    "Cinematic rain at night, slow dolly forward. "
    "Audio: rain and distant traffic."
)
DEFAULT_RESOLUTIONS = ((608, 352), (768, 448))
DEFAULT_DURATIONS = (4.0,)
MAX_CANVAS_PIXELS = 768 * 1344
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "unknown"}
KNOWN_STATUSES = {"queued", "running", *TERMINAL_STATUSES}
RESOLUTION_RE = re.compile(r"^(\d+)\s*[xX]\s*(\d+)$")


class MatrixInputError(ValueError):
    """Raised when a matrix value cannot be accepted by the API."""


@dataclass(frozen=True, slots=True)
class MatrixCase:
    width: int
    height: int
    duration_seconds: float

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} {format_duration(self.duration_seconds)}s"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    audio_sample_rate: int
    audio_channels: int
    audio_layout: str
    audio_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MatrixResult:
    index: int
    width: int
    height: int
    duration_seconds: float
    expected_frame_count: int
    task_id: str | None = None
    status: str = "not_started"
    error: str | None = None
    queued_seconds: float | None = None
    running_seconds: float | None = None
    total_seconds: float | None = None
    file_size_bytes: int | None = None
    output_path: str | None = None
    progress: float | None = None
    media: MediaInfo | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media"] = self.media.to_dict() if self.media is not None else None
        return data


def format_duration(value: float) -> str:
    """Use a stable short representation for output names and console text."""

    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _tokens(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    tokens: list[str] = []
    for value in values:
        tokens.extend(part.strip() for part in value.split(","))
    return [token for token in tokens if token]


def _append_unique[T](items: list[T], seen: set[T], item: T) -> None:
    if item not in seen:
        items.append(item)
        seen.add(item)


def parse_resolutions(values: Sequence[str] | str | None = None) -> list[tuple[int, int]]:
    """Parse and validate repeated/comma-separated WxH values.

    Values are de-duplicated while retaining their first-seen order.  This
    prevents an accidental repeated option from running the same GPU job twice.
    """

    tokens = _tokens(values)
    if not tokens:
        return list(DEFAULT_RESOLUTIONS)

    parsed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in tokens:
        match = RESOLUTION_RE.fullmatch(token)
        if match is None:
            raise MatrixInputError(f"invalid resolution {token!r}; expected WxH")
        width, height = (int(match.group(1)), int(match.group(2)))
        if width < 32 or height < 32:
            raise MatrixInputError(f"resolution {token!r} must be at least 32x32")
        if width > 8192 or height > 8192:
            raise MatrixInputError(f"resolution {token!r} exceeds the API limit of 8192")
        if width % 32 or height % 32:
            raise MatrixInputError(f"resolution {token!r} width and height must be multiples of 32")
        if width * height > MAX_CANVAS_PIXELS:
            raise MatrixInputError(
                f"resolution {token!r} exceeds the canvas limit of {MAX_CANVAS_PIXELS} pixels"
            )
        _append_unique(parsed, seen, (width, height))
    return parsed


def parse_durations(values: Sequence[str] | str | None = None) -> list[float]:
    """Parse and validate repeated/comma-separated duration values."""

    tokens = _tokens(values)
    if not tokens:
        return list(DEFAULT_DURATIONS)

    parsed: list[float] = []
    seen: set[float] = set()
    for token in tokens:
        try:
            duration = float(token)
        except ValueError as exc:
            raise MatrixInputError(f"invalid duration {token!r}; expected seconds") from exc
        if not math.isfinite(duration) or not 4.0 <= duration <= 15.0:
            raise MatrixInputError(f"duration {token!r} must be between 4 and 15 seconds")
        _append_unique(parsed, seen, duration)
    return parsed


def frame_count_for_duration(duration_seconds: float) -> int:
    """Return the H3-compatible ``17k+5`` frame count for a duration."""

    frame_count = max(5, round(duration_seconds * 24))
    return frame_count + (5 - frame_count % 17) % 17


def build_matrix(
    resolutions: Sequence[tuple[int, int]], durations: Sequence[float]
) -> list[MatrixCase]:
    """Build a resolution-major Cartesian product in deterministic order."""

    return [
        MatrixCase(width=width, height=height, duration_seconds=duration)
        for (width, height), duration in product(resolutions, durations)
    ]


def error_text(value: Any) -> str:
    """Make API error objects concise and useful in reports."""

    if value is None:
        return ""
    if isinstance(value, Mapping):
        for key in ("message", "detail", "error", "code"):
            if key in value and value[key] not in (None, ""):
                nested = error_text(value[key])
                if nested:
                    return nested
        return json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return "; ".join(error_text(item) for item in value)
    return str(value)


def normalize_status(value: Any) -> str:
    """Normalize unknown/missing task states to the API's terminal state."""

    status = str(value or "unknown").strip().lower()
    return status if status in KNOWN_STATUSES else "unknown"


def inspect_mp4(path: Path) -> MediaInfo:
    """Read actual video/audio properties and decode the complete video stream."""

    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("MP4 has no video stream")
        video_stream = container.streams.video[0]
        width = int(video_stream.codec_context.width or video_stream.width or 0)
        height = int(video_stream.codec_context.height or video_stream.height or 0)
        rate = video_stream.average_rate or video_stream.base_rate
        fps = float(rate) if rate else 0.0
        frame_count = sum(1 for _ in container.decode(video_stream))
        stream_duration = 0.0
        if video_stream.duration is not None and video_stream.time_base is not None:
            stream_duration = float(video_stream.duration * video_stream.time_base)
        duration_seconds = max(stream_duration, frame_count / fps if fps > 0 else 0.0)

        if not container.streams.audio:
            raise ValueError("MP4 has no audio stream")
        audio_stream = container.streams.audio[0]
        audio_sample_rate = int(audio_stream.rate or audio_stream.sample_rate or 0)
        audio_channels = int(audio_stream.channels or 0)
        audio_layout_obj = audio_stream.layout
        audio_layout = str(audio_layout_obj.name if audio_layout_obj is not None else "")

    with av.open(str(path)) as container:
        audio_sample_count = sum(frame.samples for frame in container.decode(audio=0))

    return MediaInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        audio_layout=audio_layout,
        audio_sample_count=audio_sample_count,
    )


def validate_media(
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int | None = None,
    *,
    expected_fps: float = 24.0,
    expected_sample_rate: int = 32000,
    expected_duration_seconds: float | None = None,
) -> MediaInfo:
    """Validate the generated MP4 against the matrix case and H3 media rules."""

    info = inspect_mp4(path)
    problems: list[str] = []
    if info.width != expected_width or info.height != expected_height:
        problems.append(
            f"dimensions are {info.width}x{info.height}, expected {expected_width}x{expected_height}"
        )
    if not math.isclose(info.fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
        problems.append(f"FPS is {info.fps:g}, expected {expected_fps:g}")
    if expected_frame_count is not None and info.frame_count != expected_frame_count:
        problems.append(f"frame count is {info.frame_count}, expected {expected_frame_count} (17k+5)")
    elif info.frame_count <= 0:
        problems.append("video has no decoded frames")
    if info.duration_seconds <= 0:
        problems.append("video duration is zero")
    if info.audio_sample_rate != expected_sample_rate:
        problems.append(
            f"audio sample rate is {info.audio_sample_rate}, expected {expected_sample_rate}"
        )
    if info.audio_channels != 2 or info.audio_layout.lower() != "stereo":
        problems.append(
            f"audio layout is {info.audio_layout or 'unknown'} ({info.audio_channels} channels), expected stereo"
        )
    if info.audio_sample_count <= 0:
        problems.append("audio stream has no decoded samples")
    if expected_duration_seconds is not None and not math.isclose(
        info.duration_seconds,
        expected_duration_seconds,
        rel_tol=0.0,
        abs_tol=max(1.0 / expected_fps, 0.05),
    ):
        problems.append(
            f"duration is {info.duration_seconds:.3f}s, expected {expected_duration_seconds:.3f}s"
        )
    if problems:
        raise ValueError("; ".join(problems))
    return info


def _response_error(response: Any) -> str:
    status_code = getattr(response, "status_code", "HTTP")
    try:
        payload = response.json()
    except Exception:
        payload = getattr(response, "text", "")
    detail = error_text(payload)
    return f"HTTP {status_code}: {detail}" if detail else f"HTTP {status_code}"


def _raise_for_status(response: Any) -> None:
    try:
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(_response_error(response)) from exc


def _cancel_remote_task(client: Any, task_id: str) -> str | None:
    try:
        response = client.delete(f"/v1/tasks/{task_id}")
        _raise_for_status(response)
    except Exception as exc:
        return str(exc)
    return None


def ensure_service_ready(client: Any) -> Mapping[str, Any]:
    response = client.get("/readyz")
    _raise_for_status(response)
    payload = response.json()
    if not isinstance(payload, Mapping) or payload.get("status") != "ready":
        raise RuntimeError(f"service is not ready: {error_text(payload)}")
    return payload


def _progress_text(progress: Any) -> str:
    if progress is None:
        return ""
    try:
        return f" progress={float(progress) * 100:.1f}%"
    except (TypeError, ValueError):
        return f" progress={progress}"


def _output_path(output_dir: Path, index: int, case: MatrixCase, task_id: str, output: Mapping[str, Any]) -> Path:
    original = Path(str(output.get("filename") or "output.mp4")).name
    if not original.lower().endswith(".mp4"):
        original = f"{original}.mp4"
    prefix = f"{index:03d}_{case.width}x{case.height}_{format_duration(case.duration_seconds)}s"
    return output_dir / f"{prefix}_{original}"


def _record_timings(
    result: MatrixResult,
    started: float,
    running_started: float | None,
    ended: float | None = None,
) -> None:
    ended = time.monotonic() if ended is None else ended
    result.total_seconds = max(0.0, ended - started)
    queue_end = running_started if running_started is not None else ended
    result.queued_seconds = max(0.0, queue_end - started)
    result.running_seconds = max(0.0, ended - (running_started or ended))


def _set_failure(
    result: MatrixResult,
    message: str,
    *,
    started: float | None = None,
    running_started: float | None = None,
) -> MatrixResult:
    result.status = "failed"
    result.error = message
    if started is not None:
        _record_timings(result, started, running_started)
    return result


def run_case(
    client: Any,
    case: MatrixCase,
    *,
    index: int,
    total_cases: int,
    prompt: str,
    steps: int,
    seed: int | None,
    output_dir: Path,
    poll_interval: float,
    timeout: float,
) -> MatrixResult:
    """Submit, poll, download, and validate one case using an httpx-like client."""

    result = MatrixResult(
        index=index,
        width=case.width,
        height=case.height,
        duration_seconds=case.duration_seconds,
        expected_frame_count=frame_count_for_duration(case.duration_seconds),
    )
    started = time.monotonic()
    running_started: float | None = None
    submission_error: str | None = None
    print(f"[{index}/{total_cases}] submit {case.label}", flush=True)
    payload: dict[str, Any] = {
        "mode": "t2v",
        "prompt": prompt,
        "duration_seconds": case.duration_seconds,
        "width": case.width,
        "height": case.height,
        "steps": steps,
    }
    if seed is not None:
        payload["seed"] = seed

    try:
        response = client.post("/v1/generations", json=payload)
        _raise_for_status(response)
        submitted = response.json()
        task_id = str(submitted.get("id") or "")
        if not task_id:
            return _set_failure(result, "submission response has no task id", started=started)
        result.task_id = task_id
        submitted_frame_count = submitted.get("frame_count")
        if submitted_frame_count is not None and int(submitted_frame_count) != result.expected_frame_count:
            submission_error = (
                "submission frame_count is "
                f"{submitted_frame_count}, expected {result.expected_frame_count} (17k+5)"
            )
        expected_duration = submitted.get("effective_duration_seconds")
        if expected_duration is not None and float(expected_duration) <= 0:
            submission_error = "submission effective duration is not positive"
    except Exception as exc:
        return _set_failure(result, str(exc), started=started)

    deadline = started + timeout
    last_status: tuple[str, Any] | None = None
    task_payload: Mapping[str, Any] = {}
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                cancel_error = _cancel_remote_task(client, result.task_id)
                message = f"timed out after {timeout:g} seconds"
                if cancel_error:
                    message += f"; remote cancellation failed: {cancel_error}"
                return _set_failure(
                    result,
                    message,
                    started=started,
                    running_started=running_started,
                )
            response = client.get(f"/v1/tasks/{result.task_id}")
            _raise_for_status(response)
            task_payload = response.json()
            status = normalize_status(task_payload.get("status"))
            result.progress = task_payload.get("progress")
            if status == "running" and running_started is None:
                running_started = now
            marker = (status, result.progress)
            if marker != last_status:
                print(
                    f"[{index}/{total_cases}] task={result.task_id} status={status}{_progress_text(result.progress)}",
                    flush=True,
                )
                last_status = marker
            if status in TERMINAL_STATUSES:
                result.status = status
                ended = time.monotonic()
                _record_timings(result, started, running_started, ended)
                break
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0.0, remaining)))
    except KeyboardInterrupt:
        cancel_error = _cancel_remote_task(client, result.task_id)
        if cancel_error:
            print(f"task {result.task_id} cancellation failed: {cancel_error}", file=sys.stderr)
        raise
    except Exception as exc:
        cancel_error = _cancel_remote_task(client, result.task_id)
        message = str(exc)
        if cancel_error:
            message += f"; remote cancellation failed: {cancel_error}"
        ended = time.monotonic()
        _record_timings(result, started, running_started, ended)
        return _set_failure(result, message)

    if result.status != "succeeded":
        result.error = error_text(task_payload.get("error")) or result.status
        return result
    if submission_error:
        return _set_failure(
            result,
            submission_error,
            started=started,
            running_started=running_started,
        )

    outputs = task_payload.get("outputs") or []
    if not isinstance(outputs, list) or not outputs or not isinstance(outputs[0], Mapping):
        return _set_failure(
            result,
            "successful task has no downloadable output",
            started=started,
            running_started=running_started,
        )
    output = outputs[0]
    download_url = str(output.get("download_url") or "")
    if not download_url:
        return _set_failure(
            result,
            "successful task output has no download_url",
            started=started,
            running_started=running_started,
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = _output_path(output_dir, index, case, result.task_id, output)
        response = client.get(download_url)
        _raise_for_status(response)
        content = response.content
        path.write_bytes(content)
        result.output_path = str(path)
        result.file_size_bytes = path.stat().st_size
        if result.file_size_bytes <= 0:
            return _set_failure(
                result,
                "downloaded MP4 is empty",
                started=started,
                running_started=running_started,
            )
        result.media = validate_media(
            path,
            case.width,
            case.height,
            result.expected_frame_count,
            expected_duration_seconds=result.expected_frame_count / 24,
        )
    except Exception as exc:
        return _set_failure(
            result,
            f"media/download: {exc}",
            started=started,
            running_started=running_started,
        )

    print(
        f"[{index}/{total_cases}] succeeded task={result.task_id} "
        f"size={result.file_size_bytes} bytes total={result.total_seconds:.2f}s",
        flush=True,
    )
    return result


REPORT_FIELDS = [
    "index",
    "width",
    "height",
    "duration_seconds",
    "expected_frame_count",
    "task_id",
    "status",
    "error",
    "queued_seconds",
    "running_seconds",
    "total_seconds",
    "file_size_bytes",
    "output_path",
    "progress",
    "media_width",
    "media_height",
    "media_fps",
    "media_frame_count",
    "media_duration_seconds",
    "audio_sample_rate",
    "audio_channels",
    "audio_layout",
    "audio_sample_count",
]


def _report_row(result: MatrixResult) -> dict[str, Any]:
    row = result.to_dict()
    media = row.pop("media") or {}
    for key in (
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_seconds",
    ):
        row[f"media_{key}"] = media.get(key)
    for key in ("audio_sample_rate", "audio_channels", "audio_layout", "audio_sample_count"):
        row[key] = media.get(key)
    return {field: row.get(field) for field in REPORT_FIELDS}


def write_reports(
    results: Sequence[MatrixResult], csv_path: Path, json_path: Path
) -> None:
    """Write the same result records as a flat CSV and a JSON array."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(_report_row(result) for result in results)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([result.to_dict() for result in results], handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        "--resolutions",
        dest="resolutions",
        action="append",
        metavar="WxH[,WxH]",
        help="resolution(s); repeat the option or separate values with commas",
    )
    parser.add_argument(
        "--duration",
        "--durations",
        dest="durations",
        action="append",
        metavar="SECONDS[,SECONDS]",
        help="duration(s) from 4 through 15 seconds; repeat or comma-separate",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("matrix-results"),
        help="directory for MP4 files and default reports",
    )
    parser.add_argument("--base-url", default=os.getenv("H3_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("H3_API_KEY"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed case")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without calling the service")
    parser.add_argument(
        "--csv",
        "--csv-output",
        dest="csv_path",
        type=Path,
        help="CSV report path (default: OUTPUT_DIR/matrix_results.csv)",
    )
    parser.add_argument(
        "--json",
        "--json-output",
        dest="json_path",
        type=Path,
        help="JSON report path (default: OUTPUT_DIR/matrix_results.json)",
    )
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    if not args.prompt or not args.prompt.strip():
        raise MatrixInputError("prompt must not be blank")
    if not 1 <= args.steps <= 100:
        raise MatrixInputError("steps must be between 1 and 100")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - 1:
        raise MatrixInputError("seed must be between 0 and 2**63-1")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        raise MatrixInputError("poll interval must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise MatrixInputError("timeout must be positive")
    if not str(args.base_url).strip():
        raise MatrixInputError("base URL must not be blank")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_options(args)
        resolutions = parse_resolutions(args.resolutions)
        durations = parse_durations(args.durations)
    except MatrixInputError as exc:
        parser.error(str(exc))
        return 2

    matrix = build_matrix(resolutions, durations)
    print(f"matrix: {len(matrix)} case(s)")
    for index, case in enumerate(matrix, start=1):
        print(f"[{index}/{len(matrix)}] {case.label} expected_frames={frame_count_for_duration(case.duration_seconds)}")
    if args.dry_run:
        print("dry-run: no requests sent")
        return 0

    output_dir = args.output_dir
    csv_path = args.csv_path or output_dir / "matrix_results.csv"
    json_path = args.json_path or output_dir / "matrix_results.json"
    results: list[MatrixResult] = []
    interrupted = False
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    try:
        with httpx.Client(
            base_url=str(args.base_url).rstrip("/"),
            headers=headers,
            timeout=max(1.0, min(float(args.timeout), 60.0)),
        ) as client:
            readiness = ensure_service_ready(client)
            devices = readiness.get("devices") or []
            device_name = devices[0].get("name") if devices and isinstance(devices[0], Mapping) else "unknown"
            print(f"service: ready backend={readiness.get('backend')} device={device_name}")
            for index, case in enumerate(matrix, start=1):
                result = run_case(
                    client,
                    case,
                    index=index,
                    total_cases=len(matrix),
                    prompt=args.prompt,
                    steps=args.steps,
                    seed=args.seed,
                    output_dir=output_dir,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                )
                results.append(result)
                if result.status != "succeeded" and args.fail_fast:
                    print(f"fail-fast: stopping after case {index}", file=sys.stderr)
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted", file=sys.stderr)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        print(f"matrix test error: {exc}", file=sys.stderr)
        return 1

    write_reports(results, csv_path, json_path)
    passed = sum(result.status == "succeeded" for result in results)
    failed = len(results) - passed
    print(f"summary: {passed} succeeded, {failed} failed")
    print(f"reports: {csv_path} and {json_path}")
    if interrupted:
        return 130
    return 0 if failed == 0 and len(results) == len(matrix) else 1


if __name__ == "__main__":
    raise SystemExit(main())
