from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from .backend import BackendSubmission, BackendTask, StoredAsset, UpscaleSubmission
from .backend import BackendError
from .config import Settings
from .models import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, GenerationRequest, UpscaleRequest
from .native import NativeH3Runner
from .media import video_duration
from .storage import LocalAssetStore
from .tasks import H3Runner, LocalTaskManager, UpscaleRunner
from .upscale import SeedVR2UpscaleRunner


class LocalGenerationBackend:
    def __init__(
        self,
        settings: Settings,
        runner: H3Runner | None = None,
        upscale_runner: UpscaleRunner | None = None,
    ) -> None:
        self.settings = settings
        self.assets = LocalAssetStore(
            settings.input_root,
            settings.output_root,
            settings.max_upload_bytes,
        )
        self.runner = runner or NativeH3Runner(settings, self.assets.resolve_input)
        self.upscale_runner = upscale_runner or SeedVR2UpscaleRunner(
            settings,
            self.assets.resolve_input,
        )
        self.tasks = LocalTaskManager(
            self.runner,
            self.assets.output_path,
            max_queue_size=settings.max_queue_size,
            allow_interrupt_running=settings.allow_interrupt_running,
            idle_release_seconds=settings.idle_release_seconds,
            upscale_runner=self.upscale_runner,
            upscale_output_factory=self.assets.upscale_output_path,
            upscale_idle_release_seconds=settings.upscale_idle_release_seconds,
        )

    async def start(self) -> None:
        self.assets.initialize()
        await self.tasks.start()

    async def close(self) -> None:
        await self.tasks.close()

    async def ready(self) -> dict:
        readiness = dict(self.runner.readiness())
        readiness["upscale"] = self.upscale_runner.readiness()
        return readiness

    async def upload(
        self,
        upload: UploadFile,
        *,
        kind: Literal["image", "video", "audio"],
    ) -> StoredAsset:
        return await self.assets.store_upload(upload, kind=kind)

    async def submit(self, request: GenerationRequest) -> BackendSubmission:
        readiness = self.runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError(
                "local H3 runtime is not ready",
                status_code=503,
                code="not_ready",
                detail=readiness,
            )
        if request.mode.value in {"ref2v", "edit"} and not self.settings.model_paths["ref2va"].is_file():
            raise BackendError(
                "Ref2V model is not installed; rerun minimaxh3-setup with --with-ref2v",
                status_code=503,
                code="model_missing",
            )
        if (
            request.mode.value in {"ref2v", "edit"}
            and self.settings.enable_acceleration_lora
            and not self.settings.model_paths["ref2va_acceleration_lora"].is_file()
        ):
            raise BackendError(
                "Ref2V acceleration LoRA is not installed; rerun "
                "minimaxh3-setup with --with-ref2v",
                status_code=503,
                code="model_missing",
            )
        if request.source_video is not None:
            source = self.assets.resolve_input(request.source_video.file_id)
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise BackendError("source_video must be an uploaded video", status_code=422, code="invalid_source_video")
            try:
                available = await asyncio.to_thread(video_duration, source)
                remaining = available - request.source_video.start_seconds
                duration = request.source_video.duration_seconds
                if duration is None:
                    duration = request.duration_seconds if "duration_seconds" in request.model_fields_set else remaining
                if duration > remaining + 1 / 24 or remaining < 4:
                    raise ValueError("source_video interval exceeds the available video or is shorter than 4 seconds")
                duration = min(duration, remaining)
                data = request.model_dump()
                data["duration_seconds"] = duration
                data["source_video"]["duration_seconds"] = duration
                request = GenerationRequest.model_validate(data)
            except (ValueError, OSError) as exc:
                raise BackendError(str(exc), status_code=422, code="invalid_source_video") from exc
        return await self.tasks.submit(request)

    async def submit_upscale(self, request: UpscaleRequest) -> UpscaleSubmission:
        readiness = self.upscale_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError(
                "local SeedVR2 runtime is not ready",
                status_code=503,
                code="upscale_not_ready",
                detail=readiness,
            )
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise BackendError(
                "upscaling supports uploaded images and videos only",
                status_code=415,
                code="unsupported_upscale_media",
            )
        return await self.tasks.submit_upscale(request)

    async def get_task(self, task_id: str) -> BackendTask | None:
        return self.tasks.get(task_id)

    async def cancel(self, task_id: str) -> BackendTask | None:
        return self.tasks.cancel(task_id)

    def resolve_file(
        self,
        *,
        filename: str,
        subfolder: str,
        file_type: Literal["input", "output", "temp"],
    ) -> Path:
        return self.assets.resolve_file(
            filename=filename,
            subfolder=subfolder,
            file_type=file_type,
        )
