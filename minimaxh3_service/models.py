from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator


MAX_CANVAS_PIXELS = 768 * 1344
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"})
FileId = Annotated[str, StringConstraints(min_length=1, max_length=512)]


def validate_file_id(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    if not normalized or ".." in normalized.split("/") or ":" in normalized:
        raise ValueError("file_id must be a relative local asset path")
    return normalized


class GenerationMode(StrEnum):
    T2V = "t2v"
    FL2V = "fl2v"
    REF2V = "ref2v"


class ReferenceVideo(BaseModel):
    file_id: FileId
    audio_file_id: FileId | None = None
    include_source_audio: bool = True

    @field_validator("file_id", "audio_file_id")
    @classmethod
    def file_ids_are_safe(cls, value: str | None) -> str | None:
        return validate_file_id(value) if value is not None else None


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    mode: GenerationMode = GenerationMode.T2V
    duration_seconds: float = Field(default=5.0, ge=4.0, le=15.0)
    width: int = Field(default=1344, ge=32, le=8192)
    height: int = Field(default=768, ge=32, le=8192)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    steps: int = Field(default=20, ge=1, le=100)
    sampler_name: str = Field(default="res_multistep", min_length=1, max_length=128)
    scheduler: Literal["simple", "beta", "normal"] = "simple"
    denoise: float = Field(default=1.0, gt=0.0, le=1.0)
    first_frame: FileId | None = None
    last_frame: FileId | None = None
    reference_images: list[FileId] = Field(default_factory=list, max_length=9)
    reference_videos: list[ReferenceVideo] = Field(default_factory=list, max_length=3)
    reference_audios: list[FileId] = Field(default_factory=list, max_length=3)
    ref_image_size: Literal["match", "max"] = "match"
    filename_prefix: str | None = Field(default=None, max_length=240)

    @field_validator("prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("first_frame", "last_frame")
    @classmethod
    def optional_file_ids_are_safe(cls, value: str | None) -> str | None:
        return validate_file_id(value) if value is not None else None

    @field_validator("reference_images", "reference_audios")
    @classmethod
    def file_id_lists_are_safe(cls, value: list[str]) -> list[str]:
        return [validate_file_id(item) for item in value]

    @field_validator("filename_prefix")
    @classmethod
    def filename_prefix_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/").strip("/")
        if not normalized or ".." in normalized.split("/") or not re.fullmatch(r"[\w./-]+", normalized):
            raise ValueError("filename_prefix contains unsupported characters")
        return normalized

    @model_validator(mode="after")
    def validate_mode_and_canvas(self) -> "GenerationRequest":
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32")
        if self.width * self.height > MAX_CANVAS_PIXELS:
            raise ValueError(f"canvas must not exceed {MAX_CANVAS_PIXELS} pixels")

        has_frames = self.first_frame is not None or self.last_frame is not None
        has_references = bool(self.reference_images or self.reference_videos or self.reference_audios)
        if self.mode == GenerationMode.T2V and (has_frames or has_references):
            raise ValueError("t2v does not accept frames or reference media")
        if self.mode == GenerationMode.FL2V:
            if not has_frames:
                raise ValueError("fl2v requires first_frame and/or last_frame")
            if has_references:
                raise ValueError("fl2v does not accept reference media")
        if self.mode == GenerationMode.REF2V:
            if has_frames:
                raise ValueError("ref2v uses reference media instead of first/last frames")
            if not self.reference_images and not self.reference_videos:
                raise ValueError("ref2v requires at least one reference image or video")
            total = len(self.reference_images) + len(self.reference_videos) + len(self.reference_audios)
            total += sum(1 for item in self.reference_videos if item.audio_file_id)
            if total > 12:
                raise ValueError("ref2v accepts at most 12 media files in total")
            audio_count = len(self.reference_audios) + sum(
                1
                for item in self.reference_videos
                if item.audio_file_id or item.include_source_audio
            )
            if audio_count > 3:
                raise ValueError("ref2v accepts at most 3 audio references in total")
        return self


class UpscaleRequest(BaseModel):
    file_id: FileId
    width: int = Field(ge=64, le=8192)
    height: int = Field(ge=64, le=8192)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    filename_prefix: str | None = Field(default=None, max_length=240)

    @field_validator("file_id")
    @classmethod
    def file_id_is_safe(cls, value: str) -> str:
        return validate_file_id(value)

    @field_validator("filename_prefix")
    @classmethod
    def filename_prefix_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/").strip("/")
        if not normalized or ".." in normalized.split("/") or not re.fullmatch(r"[\w./-]+", normalized):
            raise ValueError("filename_prefix contains unsupported characters")
        return normalized


class SubmittedTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int | float | None = None
    effective_duration_seconds: float
    frame_count: int


class SubmittedUpscaleTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int | float | None = None
    width: int
    height: int


class OutputFile(BaseModel):
    filename: str
    subfolder: str = ""
    type: str = "output"
    download_url: str


class TaskStatus(BaseModel):
    id: str
    kind: Literal["generation", "upscale"] | None = None
    status: Literal["queued", "running", "succeeded", "failed", "canceled", "unknown"]
    progress: float | None = None
    outputs: list[OutputFile] = Field(default_factory=list)
    error: dict[str, Any] | str | None = None
    raw_status: dict[str, Any] | None = None


class UploadedFile(BaseModel):
    file_id: str
    filename: str
    subfolder: str = ""
    type: str = "input"
    kind: Literal["image", "video", "audio"]
