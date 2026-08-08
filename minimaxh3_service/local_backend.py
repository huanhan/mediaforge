from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from .backend import BackendSubmission, BackendTask, StoredAsset
from .backend import BackendError
from .config import Settings
from .models import GenerationRequest
from .native import NativeH3Runner
from .storage import LocalAssetStore
from .tasks import H3Runner, LocalTaskManager


class LocalGenerationBackend:
    def __init__(self, settings: Settings, runner: H3Runner | None = None) -> None:
        self.settings = settings
        self.assets = LocalAssetStore(
            settings.input_root,
            settings.output_root,
            settings.max_upload_bytes,
        )
        self.runner = runner or NativeH3Runner(settings, self.assets.resolve_input)
        self.tasks = LocalTaskManager(
            self.runner,
            self.assets.output_path,
            max_queue_size=settings.max_queue_size,
            allow_interrupt_running=settings.allow_interrupt_running,
        )

    async def start(self) -> None:
        self.assets.initialize()
        await self.tasks.start()

    async def close(self) -> None:
        await self.tasks.close()

    async def ready(self) -> dict:
        return self.runner.readiness()

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
        if request.mode.value == "ref2v" and not self.settings.model_paths["ref2va"].is_file():
            raise BackendError(
                "Ref2V model is not installed; rerun minimaxh3-setup with --with-ref2v",
                status_code=503,
                code="model_missing",
            )
        return await self.tasks.submit(request)

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
