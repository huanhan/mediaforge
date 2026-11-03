from __future__ import annotations

import asyncio
import hmac
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal
from urllib.parse import quote, urlencode

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path as PathParameter,
    Query,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from .speech_models import SpeechRequest, SubmittedSpeechTask, SPEAKERS
from . import __version__
from .backend import BackendError, BackendTask, GenerationBackend
from .config import Settings
from .local_backend import LocalGenerationBackend
from .source_collector import (
    MEDIA_DIR,
    SOURCE_FILE,
    Bridge,
    BridgeError,
    check_task_target,
    public_url,
    register_extension_routes,
)
from .models import (
    CollectionList,
    CollectionMedia,
    CollectionRequest,
    CollectionStatus,
    SubmittedCollection,
)
from .models import DepthRequest, SubmittedDepthTask, SkeletonRequest, SubmittedSkeletonTask
from .models import BeatAnalysisRequest, SubmittedBeatAnalysisTask
from .models import TranscriptionRequest, SubmittedTranscriptionTask, AlignmentRequest
from .models import (
    AUDIO_EXTENSIONS,
    MUSIC_MAX_DURATION_SECONDS,
    MusicRequest,
    SubmittedMusicTask,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ErrorResponse,
    GenerationRequest,
    VideoEditRequest,
    HealthStatus,
    MessageResponse,
    ModelCatalog,
    OutputFile,
    ReadinessStatus,
    SubmittedTask,
    RemovalRequest,
    SubmittedRemovalTask,
    SubmittedUpscaleTask,
    TaskStatus,
    UploadedFile,
    UpscaleRequest,
)


logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "speech", "description": "Local non-commercial Breeze TTS 2 speech synthesis and recorded voice cloning, one request per shot."},
    {"name": "transcription", "description": "Local Qwen3-ASR transcription and forced alignment of confirmed lyrics."},
    {"name": "beat", "description": "Analyze uploaded music for beats, downbeats, onsets and energy; timestamps are in original-audio seconds."},
    {"name": "pose", "description": "Local DWPose skeleton extraction."},
    {"name": "depth", "description": "Local temporal video depth estimation with Video Depth Anything Small."},
    {"name": "music", "description": "Generate songs and instrumental music locally with YuE2, including optional ABC score conditioning."},
    {"name": "health", "description": "Process health and local runtime readiness."},
    {"name": "models", "description": "Installed models and generation/upscale capabilities."},
    {"name": "files", "description": "Upload source media and download generated artifacts."},
    {"name": "generation", "description": "Submit MiniMax H3 video generation tasks."},
    {"name": "upscale", "description": "Submit SeedVR2 image or whole-video upscale tasks."},
    {"name": "removal", "description": "Separate foregrounds with SAM 2 and repair video backgrounds with ProPainter."},
    {"name": "collections", "description": "Collect public web page text and visible comments through the local browser extension."},
    {"name": "tasks", "description": "Query, stream, or cancel tasks from the shared GPU queue."},
]
API_KEY_AUTH = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="API key configured through H3_API_KEY.",
    auto_error=False,
)
BEARER_AUTH = HTTPBearer(
    scheme_name="BearerAuth",
    description="The same H3_API_KEY supplied as a Bearer token.",
    auto_error=False,
)
AUTH_ERROR_RESPONSE = {
    "model": MessageResponse,
    "description": "The supplied API key is missing or invalid.",
}
BACKEND_ERROR_RESPONSE = {
    "model": ErrorResponse,
    "description": "The local inference backend rejected or failed the request.",
}
NOT_FOUND_RESPONSE = {
    "model": MessageResponse,
    "description": "The requested task or file does not exist.",
}
ALLOWED_EXTENSIONS = {
    "image": IMAGE_EXTENSIONS,
    "video": VIDEO_EXTENSIONS,
    "audio": AUDIO_EXTENSIONS,
}
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "unknown"}
BackendFactory = Callable[[Settings], GenerationBackend]


def _backend(request: Request) -> GenerationBackend:
    return request.app.state.backend


def _settings(request: Request) -> Settings:
    return request.app.state.settings


async def require_api_key(
    request: Request,
    x_api_key: str | None = Security(API_KEY_AUTH),
    bearer: HTTPAuthorizationCredentials | None = Security(BEARER_AUTH),
) -> None:
    expected = _settings(request).api_key
    if not expected:
        return
    supplied = x_api_key
    if bearer is not None:
        supplied = bearer.credentials
    if supplied is None or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


def _download_url(request: Request, *, filename: str, subfolder: str, file_type: str) -> str:
    query = urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
    return f"{request.url_for('download_file')}?{query}"


def _task_response(request: Request, task: BackendTask) -> TaskStatus:
    outputs = [
        OutputFile(
            filename=artifact.filename,
            subfolder=artifact.subfolder,
            type=artifact.file_type,
            download_url=_download_url(
                request,
                filename=artifact.filename,
                subfolder=artifact.subfolder,
                file_type=artifact.file_type,
            ),
        )
        for artifact in task.outputs
    ]
    return TaskStatus(
        id=task.id,
        kind=task.kind,
        status=task.status,
        progress=task.progress,
        outputs=outputs,
        error=task.error,
        raw_status=task.raw_status,
    )


async def _get_task(request: Request, task_id: str) -> TaskStatus:
    task = await _backend(request).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    return _task_response(request, task)


def create_app(
    settings: Settings | None = None,
    backend_factory: BackendFactory | None = None,
) -> FastAPI:
    service_settings = settings or Settings.from_env()
    make_backend = backend_factory or LocalGenerationBackend

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = service_settings
        application.state.backend = make_backend(service_settings)
        await application.state.backend.start()
        try:
            yield
        finally:
            await application.state.backend.close()

    application = FastAPI(
        title="MiniMax H3 Local API",
        version=__version__,
        summary="Local video generation and image/video upscaling on one GPU queue.",
        description=(
            "Generate audio-video clips with MiniMax H3 and upscale uploaded images or "
            "whole videos with SeedVR2. Generation and upscale tasks share one GPU queue.\n\n"
            "## Typical workflow\n"
            "1. Upload required source media with `POST /v1/files`.\n"
            "2. Submit `POST /v1/generations`, `POST /v1/edits` or `POST /v1/upscales`.\n"
            "3. Poll `GET /v1/tasks/{task_id}` or consume its SSE event stream.\n"
            "4. Download successful outputs from `outputs[].download_url`.\n\n"
            "Authentication is only enforced when `H3_API_KEY` is configured. Supply it "
            "through `X-API-Key` or `Authorization: Bearer <key>`."
        ),
        openapi_tags=OPENAPI_TAGS,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "persistAuthorization": True,
        },
        lifespan=lifespan,
    )

    @application.exception_handler(BackendError)
    async def backend_error_handler(_: Request, exc: BackendError) -> JSONResponse:
        logger.warning("Local inference backend failed: %s", exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "detail": exc.detail,
                }
            },
        )

    @application.get(
        "/healthz",
        response_model=HealthStatus,
        tags=["health"],
        summary="Check process health",
        description="Returns without loading or inspecting model resources.",
        operation_id="health_check",
    )
    async def healthz() -> HealthStatus:
        return HealthStatus(status="ok", version=__version__)

    @application.get(
        "/readyz",
        response_model=ReadinessStatus,
        tags=["health"],
        summary="Check local runtime readiness",
        description=(
            "Reports installed runtimes, weights, CUDA state, and whether H3/SeedVR2 "
            "resources are currently resident. Use capability=video before video generation "
            "to check only H3 dependencies without waiting for optional speech/music probes. "
            "Returns 503 when required assets are missing."
        ),
        operation_id="readiness_check",
        responses={
            503: {
                "model": ReadinessStatus,
                "description": "The local generation or upscale runtime is not ready.",
            }
        },
    )
    async def readyz(request: Request, capability: Literal["video"] | None = None) -> JSONResponse:
        backend = _backend(request)
        readiness = await backend.ready(capability=capability) if capability else await backend.ready()
        ready = readiness.get("status") == "ready"
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=readiness,
        )

    router = APIRouter(
        prefix="/v1",
        dependencies=[Depends(require_api_key)],
        responses={401: AUTH_ERROR_RESPONSE},
    )

    @router.get(
        "/models",
        response_model=ModelCatalog,
        tags=["models"],
        summary="List models and capabilities",
        description="Shows available generation modes and image/video upscale support.",
        operation_id="list_models",
    )
    async def models(request: Request) -> ModelCatalog:
        modes = ["t2v", "fl2v"]
        ref2v_ready = service_settings.model_paths["ref2va"].is_file() and (
            not service_settings.enable_acceleration_lora
            or service_settings.model_paths["ref2va_acceleration_lora"].is_file()
        )
        if ref2v_ready:
            modes.append("ref2v")
        readiness = await _backend(request).ready()
        upscale_readiness = readiness.get("upscale") or {"status": "unknown"}
        return ModelCatalog.model_validate({
            "id": "MiniMax-H3",
            "backend": "native",
            "models": service_settings.model_names,
            "capabilities": {
                "modes": modes,
                "edit_size": {
                    "resolutions": ["360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"],
                    "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    "default_resolution": None, "default_aspect_ratio": "16:9",
                    "max_pixels": 1032192, "alignment": 32,
                    "explicit_dimensions_override_presets": True,
                    "oversize_presets": "scale_down",
                },
                "generation_size": {
                    "resolutions": ["360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"],
                    "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    "default_width": 1344, "default_height": 768,
                    "default_resolution": None, "default_aspect_ratio": "16:9",
                    "max_pixels": 1032192, "alignment": 32,
                    "explicit_dimensions_override_presets": True,
                    "oversize_presets": "scale_down",
                },
                "duration_seconds": {"min": 4, "max": 15},
                "segmentation": {"available": True, "max_duration_seconds": 600,
                    "segment_seconds": {"min": 4, "max": 15, "default": 10},
                    "overlap_frames_default": 48, "overlap_alignment": "floor to 17k+5",
                    "continuation": "fixed_latent_prefix"},
                "edit": {"available": ref2v_ready, "endpoint": "/v1/edits", "audio_modes": ["generate", "preserve_source"]},
                "fps": 24,
                "audio": "32000 Hz stereo",
                "max_reference_images": 9,
                "max_reference_videos": 3,
                "max_reference_audios": 3,
                "acceleration_lora": service_settings.enable_acceleration_lora,
                "skeleton": {**readiness.get("skeleton", {"status": "not_ready"}),
                             "available": readiness.get("skeleton", {}).get("status") == "ready",
                             "endpoint": "/v1/poses", "outputs": ["skeleton"]},
                "removal": readiness.get("removal", {"status": "not_ready"}),
                "speech": {**readiness.get("speech", {"status": "not_ready"}),
                           "available": readiness.get("speech", {}).get("status") == "ready",
                           "endpoint": "/v1/audio/speech", "speakers": list(SPEAKERS), "default_speaker": "Serena"},
                "transcription": {**readiness.get("transcription", {"status": "not_ready"}),
                                  "available": readiness.get("transcription", {}).get("status") == "ready",
                                  "endpoint": "/v1/audio/transcriptions", "outputs": ["json", "txt"]},
                "alignment": {**readiness.get("alignment", {"status": "not_ready"}),
                              "available": readiness.get("alignment", {}).get("status") == "ready",
                              "endpoint": "/v1/audio/alignments", "outputs": ["json", "txt"]},
                "beat_analysis": {**readiness.get("beat_analysis", {"status": "not_ready"}),
                                  "available": readiness.get("beat_analysis", {}).get("status") == "ready",
                                  "endpoint": "/v1/music/analyses", "format": "json"},
                "depth": {**readiness.get("depth", {"status": "not_ready"}),
                          "available": readiness.get("depth", {}).get("status") == "ready",
                          "endpoint": "/v1/depths"},
                "music": {**readiness.get("music", {"status": "not_ready"}),
                          "available": readiness.get("music", {}).get("status") == "ready",
                          "endpoint": "/v1/music/generations", "format": "wav",
                          "max_duration_seconds": readiness.get("music", {}).get("max_duration_seconds", MUSIC_MAX_DURATION_SECONDS)},
                "image_upscale": {
                    "available": upscale_readiness.get("status") == "ready",
                    "backend": upscale_readiness.get("backend", "seedvr2"),
                    "model": service_settings.upscale_dit_model,
                    "max_width": 8192,
                    "max_height": 8192,
                },
                "video_upscale": {
                    "available": upscale_readiness.get("status") == "ready",
                    "backend": upscale_readiness.get("backend", "seedvr2"),
                    "model": service_settings.upscale_dit_model,
                    "preserves_audio": True,
                    "max_width": 8192,
                    "max_height": 8192,
                },
            },
        })

    @router.post(
        "/files",
        response_model=UploadedFile,
        status_code=status.HTTP_201_CREATED,
        tags=["files"],
        summary="Upload source media",
        description=(
            "Stores an image, video, or audio file locally. Use the returned `file_id` in "
            "generation and upscale requests."
        ),
        operation_id="upload_file",
        response_description="The stored asset and reusable file_id.",
        responses={
            413: {"model": MessageResponse, "description": "The upload exceeds H3_MAX_UPLOAD_BYTES."},
            415: {"model": MessageResponse, "description": "The extension is unsupported for kind."},
        },
    )
    async def upload_file(
        request: Request,
        kind: Literal["image", "video", "audio"] = Form(
            ...,
            description="Declared media kind used to validate the filename extension.",
        ),
        file: UploadFile = File(..., description="Image, video, or audio file to store."),
    ) -> UploadedFile:
        extension = Path(file.filename or "").suffix.lower()
        if extension not in ALLOWED_EXTENSIONS[kind]:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS[kind]))
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"unsupported {kind} extension; allowed: {allowed}",
            )
        if file.size is not None and file.size > service_settings.max_upload_bytes:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="file is too large")
        stored = await _backend(request).upload(file, kind=kind)
        return UploadedFile(
            file_id=stored.file_id,
            filename=stored.filename,
            subfolder=stored.subfolder,
            type="input",
            kind=stored.kind,
        )

    @router.post(
        "/generations",
        response_model=SubmittedTask,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["generation"],
        summary="视频生成 / 重叠分段",
        description=(
            "仅接受 t2v、fl2v、ref2v。可启用 segmentation 自动重叠分段。"
            "视频编辑请使用独立 POST /v1/edits；此接口不再接受 mode=edit 或编辑专用字段。"
        ),
        operation_id="create_generation",
        response_description="Queued generation task metadata.",
        responses={
            429: BACKEND_ERROR_RESPONSE,
            503: BACKEND_ERROR_RESPONSE,
        },
    )
    async def create_generation(
        request: Request,
        body: GenerationRequest = Body(openapi_examples={
            "t2v": {
                "summary": "文生视频（默认）",
                "value": {"mode": "t2v", "prompt": "雨夜街道，镜头缓慢向前。Audio: rain.",
                          "duration_seconds": 4, "width": 608, "height": 352, "seed": 42},
            },
            "portrait_t2v": {
                "summary": "文生视频：480p 竖屏（实际 480×832）",
                "value": {"mode": "t2v", "prompt": "雨夜街道，镜头缓慢向前。", "duration_seconds": 4,
                          "resolution": "480p", "aspect_ratio": "9:16"},
            },
            "segmented_t2v": {
                "summary": "文生长视频：自动重叠分段",
                "value": {"mode": "t2v", "prompt": "镜头持续缓慢穿过雨夜街道。Audio: steady rain.",
                          "duration_seconds": 30, "width": 608, "height": 352, "seed": 42,
                          "segmentation": {"enabled": True, "segment_seconds": 10, "overlap_frames": 48}},
            },
        }),
    ) -> SubmittedTask:
        submitted = await _backend(request).submit(body)
        return SubmittedTask(
            id=submitted.id,
            queue_number=submitted.queue_number,
            effective_duration_seconds=submitted.effective_duration_seconds,
            frame_count=submitted.frame_count,
            segment_count=submitted.segment_count,
            actual_overlap_frames=submitted.actual_overlap_frames,
        )

    @router.post(
        "/edits",
        response_model=SubmittedTask,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["editing"],
        summary="视频编辑 / 角色迁移 / 换背景",
        description=("上传源视频及可选参考图后提交。source_video 必填，不传 mode；"
                     "编辑区间只填写 source_video.start_seconds / duration_seconds。"
                     "与视频生成共用任务查询、取消和下载接口，支持重叠分段。"),
        operation_id="create_edit",
        responses={429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE},
    )
    async def create_edit(
        request: Request,
        body: VideoEditRequest = Body(openapi_examples={
            "edit_preset_size": {
                "summary": "视频编辑：480p 竖屏（实际 480×832）",
                "value": {"prompt": "保持人物和动作，将背景改成纯绿色。",
                          "source_video": {"file_id": "uploads/source.mp4"},
                          "resolution": "480p", "aspect_ratio": "9:16",
                          "segmentation": {"enabled": True}},
            },
            "edit_character": {
                "summary": "视频编辑：角色替换 / 动作迁移",
                "description": "上传源视频和角色图。未填写时长表示处理全部源视频；需要 Ref2V 权重。",
                "value": {
                          "prompt": "用 <Picture 1> 中的角色替换 <Video 1> 的主要人物，继承原动作、节奏和道具，保留原背景与镜头。",
                          "source_video": {"file_id": "uploads/source.mp4"},
                          "reference_images": ["uploads/character.png"],
                          "width": 352, "height": 640, "audio_mode": "preserve_source", "seed": 358,
                          "segmentation": {"enabled": True, "segment_seconds": 10, "overlap_frames": 48}},
            },
            "edit_background": {
                "summary": "视频编辑：上传背景图换背景",
                "description": "reference_images 在此作为背景参考；每张图的作用由提示词指定。",
                "value": {
                          "prompt": "<Picture 1> 仅用作目标背景。从第一帧到最后一帧，将 <Video 1> 的背景替换为图片中的场景，保留背景构图与色彩。保留原角色、动作、道具和镜头。不要渐变或延迟切换。",
                          "source_video": {"file_id": "uploads/source.mp4"},
                          "reference_images": ["uploads/background.png"],
                          "width": 352, "height": 640, "audio_mode": "preserve_source", "seed": 358,
                          "segmentation": {"enabled": True, "segment_seconds": 10, "overlap_frames": 48}},
            },
        }),
    ) -> SubmittedTask:
        submitted = await _backend(request).submit_edit(body)
        return SubmittedTask(
            id=submitted.id, queue_number=submitted.queue_number,
            frame_count=submitted.frame_count,
            effective_duration_seconds=submitted.effective_duration_seconds,
            segment_count=submitted.segment_count,
            actual_overlap_frames=submitted.actual_overlap_frames,
        )

    @router.post(
        "/upscales",
        response_model=SubmittedUpscaleTask,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["upscale"],
        summary="Submit an image or video upscale task",
        description=(
            "Queues SeedVR2 upscaling for an uploaded image or whole video. Video outputs "
            "preserve the source frame rate and embedded audio when present."
        ),
        operation_id="create_upscale",
        response_description="Queued upscale task metadata.",
        responses={
            415: BACKEND_ERROR_RESPONSE,
            429: BACKEND_ERROR_RESPONSE,
            503: BACKEND_ERROR_RESPONSE,
        },
    )
    async def create_upscale(
        request: Request,
        body: UpscaleRequest,
    ) -> SubmittedUpscaleTask:
        submitted = await _backend(request).submit_upscale(body)
        return SubmittedUpscaleTask(
            id=submitted.id,
            queue_number=submitted.queue_number,
            width=submitted.width,
            height=submitted.height,
        )

    @router.post("/audio/speech", response_model=SubmittedSpeechTask, status_code=202,
                 tags=["speech"], summary="Synthesize one shot with a preset or recorded voice",
                 operation_id="create_speech", responses={429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE},
                 description="Local Breeze TTS 2 for research/non-commercial use. Clone with a 3–30 second reference_file_id and its exact reference_text. mode=clone preserves delivery; mode=direction adds instruction and cfg_scale; mode=design uses instruction and cfg_scale without reference audio. Text supports Chinese [笑]/[叹气]/[咳嗽]/[清嗓子] and English (laugh)/(sigh)/(cough)/(clears throat). speed is pitch-preserving postprocessing; pause_ms separates text chunks before speed adjustment. Supports Chinese/English. Poll the shared /v1/tasks queue; returns 24 kHz mono WAV and metadata. Does not perform lip synchronization.")
    async def create_speech(request: Request, body: SpeechRequest) -> SubmittedSpeechTask:
        submit = getattr(_backend(request), "submit_speech", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Speech generation is not configured")
        return await submit(body)

    @router.post("/audio/alignments", response_model=SubmittedTranscriptionTask, status_code=202,
                 tags=["transcription"], summary="Align confirmed lyric lines to uploaded audio",
                 description="Local Qwen3 ForcedAligner. Maximum 180 seconds. Supply lyrics with one subtitle per line; returns cues and word timestamps. Uses the shared task queue.",
                 operation_id="create_alignment", responses={503: BACKEND_ERROR_RESPONSE})
    async def create_alignment(request: Request, body: AlignmentRequest) -> SubmittedTranscriptionTask:
        submit = getattr(_backend(request), "submit_alignment", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Lyric alignment is not configured")
        return await submit(body)

    @router.post("/audio/transcriptions", response_model=SubmittedTranscriptionTask, status_code=202,
                 tags=["transcription"], summary="Transcribe a song or speech with local Qwen3-ASR",
                 operation_id="create_transcription",
                 description="Upload audio via POST /v1/files, then submit its file_id as JSON. Uses the shared GPU task queue. Poll /v1/tasks/{id} and download JSON and TXT outputs. Input chunk boundaries are not sentence/word timestamps. This endpoint does not align provided lyrics.",
                 responses={503: BACKEND_ERROR_RESPONSE})
    async def create_transcription(request: Request, body: TranscriptionRequest) -> SubmittedTranscriptionTask:
        submit = getattr(_backend(request), "submit_transcription", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Transcription is not configured")
        return await submit(body)

    @router.post("/music/analyses", response_model=SubmittedBeatAnalysisTask, status_code=202,
                 tags=["beat"], summary="Analyze music timing for video editing",
                 operation_id="create_beat_analysis",
                 description="Upload audio with POST /v1/files (kind=audio), then submit its file_id. Uses the shared task queue. Poll /v1/tasks/{id}, then download the JSON from outputs[].download_url. No transcription or lyric alignment is performed.",
                 responses={503: BACKEND_ERROR_RESPONSE})
    async def create_beat_analysis(request: Request, body: BeatAnalysisRequest) -> SubmittedBeatAnalysisTask:
        submit = getattr(_backend(request), "submit_beat_analysis", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Beat analysis is not configured")
        return await submit(body)

    @router.post("/depths", response_model=SubmittedDepthTask, status_code=202,
                 tags=["depth"], summary="Convert a video to relative inverse depth",
                 operation_id="create_depth",
                 description="Uses the shared GPU queue. Returns a silent grayscale MP4 (near is bright), globally normalized across the clip, plus optional float32 NPZ. Preserves frame count and nominal FPS; variable-rate inputs become constant-rate. Poll /v1/tasks/{id} for downloads.",
                 responses={415: BACKEND_ERROR_RESPONSE, 429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE})
    async def create_depth(request: Request, body: DepthRequest) -> SubmittedDepthTask:
        submit = getattr(_backend(request), "submit_depth", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Video depth is not configured")
        return await submit(body)

    @router.post("/poses", response_model=SubmittedSkeletonTask, status_code=202,
                 tags=["pose"], summary="Extract a skeleton video",
                 operation_id="create_skeleton",
                 description="Extracts a silent skeleton visualization MP4 with local DWPose. Poll /v1/tasks/{id} for the output.",
                 responses={415: BACKEND_ERROR_RESPONSE, 429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE})
    async def create_skeleton(request: Request, body: SkeletonRequest) -> SubmittedSkeletonTask:
        submit = getattr(_backend(request), "submit_skeleton", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Skeleton extraction is not configured")
        return await submit(body)

    @router.post("/music/generations", response_model=SubmittedMusicTask, status_code=202,
                 tags=["music"], summary="Generate instrumental music or a song with YuE2",
                 operation_id="create_music_generation",
                 responses={429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE},
                 description=f"Returns a task ID for the shared GPU queue. Poll /v1/tasks/{{id}} for progress and download 48 kHz stereo PCM WAV, generation metadata JSON and an ABC score when planned. duration_seconds is a 10–{MUSIC_MAX_DURATION_SECONDS} second upper bound and can truncate a phrase; inspect metadata.truncated. Empty lyrics request instrumental music. cot controls symbolic planning. Running tasks can be canceled when H3_ALLOW_INTERRUPT_RUNNING=true.")
    async def create_music_generation(request: Request, body: MusicRequest) -> SubmittedMusicTask:
        submit = getattr(_backend(request), "submit_music", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Music generation is not configured")
        return await submit(body)

    @router.post(
        "/removals", response_model=SubmittedRemovalTask, status_code=202,
        tags=["removal"], summary="Separate a video foreground or repair its background",
        description=("Select up to 8 objects with normalized boxes and reference frame indices. "
                     "Each object is transparent before its reference frame. Exports separate foregrounds "
                     "and repairs the union mask once. Legacy first-frame box requests remain supported. "
                     "One request uses one task in the shared GPU queue."),
        operation_id="create_removal",
        responses={415: BACKEND_ERROR_RESPONSE, 429: BACKEND_ERROR_RESPONSE, 503: BACKEND_ERROR_RESPONSE},
    )
    async def create_removal(request: Request, body: RemovalRequest) -> SubmittedRemovalTask:
        submit = getattr(_backend(request), "submit_removal", None)
        if submit is None:
            raise HTTPException(status_code=503, detail="Video removal is not configured")
        return await submit(body)

    @router.get(
        "/tasks/{task_id}",
        response_model=TaskStatus,
        tags=["tasks"],
        summary="Get task status",
        description="Queries generation, upscale and removal tasks through the same endpoint.",
        operation_id="get_task",
        responses={404: NOT_FOUND_RESPONSE},
    )
    async def get_task(
        request: Request,
        task_id: str = PathParameter(description="Task id returned by a submit endpoint."),
    ) -> TaskStatus:
        return await _get_task(request, task_id)

    @router.get(
        "/tasks/{task_id}/events",
        response_class=StreamingResponse,
        tags=["tasks"],
        summary="Stream task status events",
        description=(
            "Emits an SSE `status` event whenever the task representation changes, then "
            "closes after a terminal status."
        ),
        operation_id="stream_task_events",
        responses={
            200: {
                "description": "Server-sent task status events.",
                "content": {
                    "text/event-stream": {
                        "schema": {"type": "string"},
                        "example": "event: status\ndata: {\"id\":\"...\",\"status\":\"running\"}\n\n",
                    }
                },
            },
            404: NOT_FOUND_RESPONSE,
        },
    )
    async def task_events(
        request: Request,
        task_id: str = PathParameter(description="Task id returned by a submit endpoint."),
        interval_seconds: float = Query(
            default=1.0,
            ge=0.25,
            le=30.0,
            description="Polling interval used by the SSE stream.",
        ),
    ) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[bytes]:
            previous: str | None = None
            while True:
                if await request.is_disconnected():
                    return
                task = await _get_task(request, task_id)
                payload = task.model_dump_json()
                if payload != previous:
                    yield f"event: status\ndata: {payload}\n\n".encode()
                    previous = payload
                if task.status in TERMINAL_STATUSES:
                    return
                await asyncio.sleep(interval_seconds)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.delete(
        "/tasks/{task_id}",
        response_model=TaskStatus,
        tags=["tasks"],
        summary="Cancel a task",
        description=(
            "Cancels queued tasks immediately. Running cancellation requires "
            "H3_ALLOW_INTERRUPT_RUNNING=true."
        ),
        operation_id="cancel_task",
        responses={
            404: NOT_FOUND_RESPONSE,
            409: BACKEND_ERROR_RESPONSE,
        },
    )
    async def cancel_task(
        request: Request,
        task_id: str = PathParameter(description="Task id returned by a submit endpoint."),
    ) -> TaskStatus:
        task = await _backend(request).cancel(task_id)
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
        return _task_response(request, task)

    @router.get(
        "/files/content",
        name="download_file",
        response_class=FileResponse,
        tags=["files"],
        summary="Download a stored file",
        description=(
            "Downloads an input or output artifact. Successful task responses provide a "
            "ready-to-use absolute download_url."
        ),
        operation_id="download_file",
        responses={
            200: {
                "description": "Binary file content with its detected image/video/audio MIME type.",
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            },
            404: {
                "model": ErrorResponse,
                "description": "The stored file does not exist or is not exposed.",
            },
        },
    )
    async def download_file(
        request: Request,
        filename: str = Query(
            min_length=1,
            max_length=512,
            description="Basename from an upload or task output.",
        ),
        subfolder: str = Query(
            default="",
            max_length=512,
            description="Relative storage subfolder from an upload or task output.",
        ),
        file_type: Literal["input", "output", "temp"] = Query(
            default="output",
            alias="type",
            description="Storage area. Temporary files are not exposed.",
        ),
    ) -> FileResponse:
        path = _backend(request).resolve_file(
            filename=filename,
            subfolder=subfolder,
            file_type=file_type,
        )
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename, safe='')}"},
        )

    collector = Bridge(service_settings.data_root.expanduser().resolve() / "collections")
    register_extension_routes(application, collector)

    def _collection_response(request: Request, task: dict) -> CollectionStatus:
        task_dir = collector.task_dir(task["id"])
        source = task_dir / SOURCE_FILE
        source_url = (
            str(request.url_for("download_collection_source", collection_id=task["id"]))
            if source.is_file() else None
        )
        media = [
            CollectionMedia(
                filename=item["file"],
                bytes=item["bytes"],
                source_url=item["url"],
                download_url=str(request.url_for(
                    "download_collection_media", collection_id=task["id"], filename=item["file"],
                )),
            )
            for item in task["captured"]
        ]
        return CollectionStatus(
            id=task["id"],
            url=task["url"],
            title=task["title"],
            status=task["status"],
            error=task["error"],
            created_at=task["created_at"],
            source_download_url=source_url,
            media=media,
        )

    def _describe_collection(collection_id: str) -> dict:
        try:
            return collector.describe(collection_id)
        except BridgeError as error:
            raise HTTPException(status_code=error.status, detail=str(error))

    @router.post("/collections", response_model=SubmittedCollection, status_code=202,
                 tags=["collections"], summary="Queue a public web page for browser collection",
                 operation_id="create_collection",
                 description="Queues one public http(s) page. The Chrome extension running in the operator's browser opens it in a new tab, reads its visible text and loaded comments, and writes source.md. Poll GET /v1/collections/{id}.",
                 responses={400: {"model": MessageResponse}, 429: {"model": MessageResponse}})
    async def create_collection(request: Request, body: CollectionRequest) -> SubmittedCollection:
        try:
            # Reject private/local targets before queuing; the extension re-checks when it claims the task.
            url = public_url(body.url)
            await asyncio.to_thread(check_task_target, url)
            task = collector.enqueue(url, body.title or "")
        except BridgeError as error:
            raise HTTPException(status_code=error.status, detail=str(error))
        return SubmittedCollection(id=task.id, url=task.url)

    @router.get("/collections", response_model=CollectionList, tags=["collections"],
                summary="List recent collection tasks", operation_id="list_collections")
    async def list_collections(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> CollectionList:
        return CollectionList(collections=[_collection_response(request, row) for row in collector.recent(limit)])

    @router.get("/collections/{collection_id}", response_model=CollectionStatus, tags=["collections"],
                summary="Get a collection task", operation_id="get_collection",
                responses={404: NOT_FOUND_RESPONSE})
    async def get_collection(request: Request, collection_id: str = PathParameter(pattern=r"^[0-9a-f]{16}$")) -> CollectionStatus:
        return _collection_response(request, _describe_collection(collection_id))

    @router.get("/collections/{collection_id}/source", tags=["collections"],
                summary="Download the collected Markdown source", operation_id="download_collection_source",
                response_class=FileResponse,
                responses={404: NOT_FOUND_RESPONSE})
    async def download_collection_source(collection_id: str = PathParameter(pattern=r"^[0-9a-f]{16}$")) -> FileResponse:
        _describe_collection(collection_id)
        path = collector.task_dir(collection_id) / SOURCE_FILE
        if not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="source not available yet")
        return FileResponse(path, media_type="text/markdown; charset=utf-8", filename=f"{collection_id}.md")

    @router.get("/collections/{collection_id}/media/{filename}", tags=["collections"],
                summary="Download a captured page response", operation_id="download_collection_media",
                response_class=FileResponse,
                responses={404: NOT_FOUND_RESPONSE})
    async def download_collection_media(
        collection_id: str = PathParameter(pattern=r"^[0-9a-f]{16}$"),
        filename: str = PathParameter(pattern=r"^[0-9a-f]{16}\.(jpg|png|webp|gif|bin|mp4|webm|mov)$"),
    ) -> FileResponse:
        _describe_collection(collection_id)
        path = collector.task_dir(collection_id) / MEDIA_DIR / filename
        if not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="media not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=filename)

    application.include_router(router)
    # The optional UI shares this application's listener and backend lifecycle.
    from importlib.util import find_spec

    if find_spec("gradio") is not None:
        from .ui import mount_ui

        application = mount_ui(application, service_settings)
    return application


app = create_app()
