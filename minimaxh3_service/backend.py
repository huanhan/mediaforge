from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import UploadFile

from .models import GenerationRequest, VideoEditRequest, UpscaleRequest


TaskState = Literal["queued", "running", "succeeded", "failed", "canceled", "unknown"]
TaskKind = Literal["generation", "upscale"]


class BackendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        code: str = "backend_error",
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class StoredAsset:
    file_id: str
    filename: str
    subfolder: str
    kind: Literal["image", "video", "audio"]


@dataclass(frozen=True, slots=True)
class OutputArtifact:
    filename: str
    subfolder: str
    path: Path
    file_type: str = "output"


@dataclass(frozen=True, slots=True)
class BackendSubmission:
    id: str
    queue_number: int
    frame_count: int
    effective_duration_seconds: float
    segment_count: int = 1
    actual_overlap_frames: int = 0


@dataclass(frozen=True, slots=True)
class UpscaleSubmission:
    id: str
    queue_number: int
    width: int
    height: int


@dataclass(slots=True)
class BackendTask:
    id: str
    status: TaskState
    kind: TaskKind | None = None
    progress: float | None = None
    outputs: list[OutputArtifact] = field(default_factory=list)
    error: dict[str, Any] | str | None = None
    raw_status: dict[str, Any] | None = None


class GenerationBackend(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def ready(self) -> dict[str, Any]: ...

    async def upload(
        self,
        upload: UploadFile,
        *,
        kind: Literal["image", "video", "audio"],
    ) -> StoredAsset: ...

    async def submit(self, request: GenerationRequest) -> BackendSubmission: ...

    async def submit_edit(self, request: VideoEditRequest) -> BackendSubmission: ...

    async def submit_upscale(self, request: UpscaleRequest) -> UpscaleSubmission: ...

    async def get_task(self, task_id: str) -> BackendTask | None: ...

    async def cancel(self, task_id: str) -> BackendTask | None: ...

    def resolve_file(
        self,
        *,
        filename: str,
        subfolder: str,
        file_type: Literal["input", "output", "temp"],
    ) -> Path: ...
