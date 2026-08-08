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
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__
from .backend import BackendError, BackendTask, GenerationBackend
from .config import Settings
from .local_backend import LocalGenerationBackend
from .models import GenerationRequest, OutputFile, SubmittedTask, TaskStatus, UploadedFile


logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"},
}
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "unknown"}
BackendFactory = Callable[[Settings], GenerationBackend]


def _backend(request: Request) -> GenerationBackend:
    return request.app.state.backend


def _settings(request: Request) -> Settings:
    return request.app.state.settings


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    expected = _settings(request).api_key
    if not expected:
        return
    supplied = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
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
        description="A standalone local MiniMax H3 CUDA video generation service.",
        lifespan=lifespan,
    )

    @application.exception_handler(BackendError)
    async def backend_error_handler(_: Request, exc: BackendError) -> JSONResponse:
        logger.warning("Local H3 backend failed: %s", exc)
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

    @application.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get("/readyz", tags=["health"])
    async def readyz(request: Request) -> JSONResponse:
        readiness = await _backend(request).ready()
        ready = readiness.get("status") == "ready"
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=readiness,
        )

    router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @router.get("/models", tags=["models"])
    async def models() -> dict[str, Any]:
        modes = ["t2v", "fl2v"]
        if service_settings.model_paths["ref2va"].is_file():
            modes.append("ref2v")
        return {
            "id": "MiniMax-H3",
            "backend": "native",
            "models": service_settings.model_names,
            "capabilities": {
                "modes": modes,
                "duration_seconds": {"min": 4, "max": 15},
                "fps": 24,
                "audio": "32000 Hz stereo",
                "max_reference_images": 9,
                "max_reference_videos": 3,
                "max_reference_audios": 3,
            },
        }

    @router.post(
        "/files",
        response_model=UploadedFile,
        status_code=status.HTTP_201_CREATED,
        tags=["files"],
    )
    async def upload_file(
        request: Request,
        kind: Literal["image", "video", "audio"] = Form(...),
        file: UploadFile = File(...),
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
    )
    async def create_generation(request: Request, body: GenerationRequest) -> SubmittedTask:
        submitted = await _backend(request).submit(body)
        return SubmittedTask(
            id=submitted.id,
            queue_number=submitted.queue_number,
            effective_duration_seconds=submitted.effective_duration_seconds,
            frame_count=submitted.frame_count,
        )

    @router.get("/tasks/{task_id}", response_model=TaskStatus, tags=["tasks"])
    async def get_task(request: Request, task_id: str) -> TaskStatus:
        return await _get_task(request, task_id)

    @router.get("/tasks/{task_id}/events", tags=["tasks"])
    async def task_events(
        request: Request,
        task_id: str,
        interval_seconds: float = Query(default=1.0, ge=0.25, le=30.0),
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

    @router.delete("/tasks/{task_id}", response_model=TaskStatus, tags=["tasks"])
    async def cancel_task(request: Request, task_id: str) -> TaskStatus:
        task = await _backend(request).cancel(task_id)
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
        return _task_response(request, task)

    @router.get("/files/content", name="download_file", tags=["files"])
    async def download_file(
        request: Request,
        filename: str = Query(min_length=1, max_length=512),
        subfolder: str = Query(default="", max_length=512),
        file_type: Literal["input", "output", "temp"] = Query(default="output", alias="type"),
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

    application.include_router(router)
    return application


app = create_app()
