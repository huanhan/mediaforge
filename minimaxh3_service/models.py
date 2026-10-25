from __future__ import annotations

import re
import math
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


MAX_CANVAS_PIXELS = 768 * 1344
MUSIC_MAX_DURATION_SECONDS = 500
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".webm"})
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


class MusicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prompt: str = Field(min_length=1, max_length=10000, description="Music style, instruments, mood and vocals.")
    lyrics: str = Field(default="", max_length=20000, description="Lyrics with [Verse]/[Chorus] sections. Empty lyrics request instrumental music; absence of vocals is not guaranteed.")
    duration_seconds: int = Field(default=30, ge=10, le=MUSIC_MAX_DURATION_SECONDS, description="Maximum output duration, enforced by the semantic token budget and audio cropping; may truncate a musical phrase.")
    seed: int = Field(default=7, ge=0, le=2147483647)
    cot: Literal["full", "melody", "off"] = Field(default="full", description="YuE2 symbolic planning: melody and chords, melody only, or direct generation.")
    abc: str | None = Field(default=None, min_length=1, max_length=50000, description="Optional YuE2-compatible ABC score; requires cot=full or melody.")
    cfg_scale: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, description="Optional text guidance; null uses YuE2 defaults.")

    @model_validator(mode="after")
    def validate_score(self):
        if self.abc is not None and self.cot == "off":
            raise ValueError("abc requires cot=full or cot=melody")
        return self


class SubmittedMusicTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int


class BeatAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file_id: FileId = Field(description="Audio uploaded through POST /v1/files with kind=audio.")
    title: str = Field(default="", max_length=200, description="Optional song title; does not affect detected timing.")

    @field_validator("file_id")
    @classmethod
    def safe_file_id(cls, value: str) -> str:
        return validate_file_id(value)


class SubmittedBeatAnalysisTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int


class TranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file_id: FileId = Field(description="Uploaded audio file_id; no external URLs or local absolute paths.")
    title: str = Field(default="", max_length=200, description="Optional song title stored as metadata only.")
    language: Literal["auto", "zh", "en", "yue", "ja", "ko"] = "zh"
    chunk_seconds: int = Field(default=180, ge=30, le=180,
                               description="Maximum input window. Boundaries are NOT lyric timestamps.")

    @field_validator("file_id")
    @classmethod
    def safe_file_id(cls, value: str) -> str:
        return validate_file_id(value)


class SubmittedTranscriptionTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int


class AlignmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file_id: FileId
    lyrics: str = Field(min_length=1, max_length=10000, description="Confirmed lyrics, one subtitle per line. Long audio is aligned in overlapping windows; no default track duration limit.")
    title: str = Field(default="", max_length=200)
    language: Literal["zh", "en", "yue"] = "zh"

    @field_validator("file_id")
    @classmethod
    def safe_file_id(cls, value: str) -> str:
        return validate_file_id(value)


class SourceVideo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: FileId
    start_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    duration_seconds: float | None = Field(default=None, ge=4, le=600)

    @field_validator("file_id")
    @classmethod
    def safe_file_id(cls, value: str) -> str:
        return validate_file_id(value)


class SegmentationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    segment_seconds: float = Field(default=10, ge=4, le=15)
    overlap_frames: int = Field(default=48, ge=5, le=345,
        description="Requested overlap at 24 FPS; rounded down to the H3 17k+5 grid.")

    @model_validator(mode="after")
    def overlap_fits(self) -> "SegmentationOptions":
        frames = round(self.segment_seconds * 24)
        frames += (5 - frames % 17) % 17
        overlap = self.overlap_frames - (self.overlap_frames - 5) % 17
        if overlap >= frames:
            raise ValueError("overlap must be shorter than a segment")
        return self


class ReferenceVideo(BaseModel):
    file_id: FileId = Field(description="Uploaded reference video file_id.")
    audio_file_id: FileId | None = Field(
        default=None,
        description="Optional uploaded audio file_id used instead of the video's audio.",
    )
    include_source_audio: bool = Field(
        default=True,
        description="Use the reference video's embedded audio when no audio_file_id is supplied.",
    )

    @field_validator("file_id", "audio_file_id")
    @classmethod
    def file_ids_are_safe(cls, value: str | None) -> str | None:
        return validate_file_id(value) if value is not None else None


class VideoRequestOptions(BaseModel):
    """Sampling and output options shared by the two independent HTTP requests."""
    model_config = ConfigDict(extra="forbid")

    resolution: Literal["360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"] | None = Field(
        default=None, description="Optional output short-edge preset. Explicit width/height take priority; if neither dimensions nor resolution is supplied, generation uses 1344x768 and editing uses source video dimensions. High presets are scaled down to the model pixel limit.")
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] = Field(
        default="16:9", description="Width:height ratio used with resolution. Ignored when explicit dimensions are supplied or resolution is omitted.")

    @model_validator(mode="before")
    @classmethod
    def resolve_preset_canvas(cls, value):
        if not isinstance(value, dict) or value.get("resolution") is None:
            return value
        width, height = value.get("width"), value.get("height")
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together or both omitted")
        if width is not None:
            return value
        resolution = value.get("resolution", "720p")
        ratio = value.get("aspect_ratio", "16:9")
        if resolution not in ("360p", "480p", "540p", "720p", "1080p", "1440p", "2160p") or ratio not in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"):
            return value  # Field validation reports unsupported presets.
        short = int(resolution[:-1])
        x, y = map(int, ratio.split(":"))
        width, height = short * x / min(x, y), short * y / min(x, y)
        scale = min(1.0, math.sqrt(MAX_CANVAS_PIXELS / (width * height)), 8192 / width, 8192 / height)
        return {**value, "width": max(32, int(width * scale) // 32 * 32),
                "height": max(32, int(height * scale) // 32 * 32)}

    prompt: str = Field(
        min_length=1,
        max_length=100_000,
        description="Visual and audio generation prompt.",
    )
    width: int = Field(
        default=1344,
        ge=32,
        le=8192,
        description="Output width; must be a multiple of 32. Explicit dimensions override resolution/aspect_ratio. Default 1344 when no preset is supplied.",
    )
    height: int = Field(
        default=768,
        ge=32,
        le=8192,
        description="Output height; must be a multiple of 32. Explicit dimensions override resolution/aspect_ratio. Default 768 when no preset is supplied.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Optional deterministic seed. A random seed is used when omitted.",
    )
    denoise: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Denoising strength.",
    )
    segmentation: SegmentationOptions = Field(default_factory=SegmentationOptions)
    reference_images: list[FileId] = Field(
        default_factory=list,
        max_length=9,
        description="Uploaded reference image file_ids, in prompt numbering order.",
    )
    ref_image_size: Literal["match", "max"] = Field(
        default="match",
        description="Resize reference images to the output canvas or their maximum supported size.",
    )
    filename_prefix: str | None = Field(
        default=None,
        max_length=240,
        description="Optional safe output path prefix relative to the output directory.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("reference_images")
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
    def validate_canvas(self) -> "VideoRequestOptions":
        if self.width is None or self.height is None:
            return self
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32")
        if self.width * self.height > MAX_CANVAS_PIXELS:
            raise ValueError(f"canvas must not exceed {MAX_CANVAS_PIXELS} pixels")
        return self


class GenerationRequest(VideoRequestOptions):
    mode: GenerationMode = Field(default=GenerationMode.T2V,
        description="t2v: text; fl2v: first/last frames; ref2v: reference media. Video editing uses POST /v1/edits.")
    duration_seconds: float = Field(default=5, ge=4, le=600,
        description="Requested duration; above 15 seconds requires segmentation.")
    first_frame: FileId | None = None
    last_frame: FileId | None = None
    reference_videos: list[ReferenceVideo] = Field(default_factory=list, max_length=3)
    reference_audios: list[FileId] = Field(default_factory=list, max_length=3)

    @field_validator("first_frame", "last_frame")
    @classmethod
    def optional_file_ids_are_safe(cls, value: str | None) -> str | None:
        return validate_file_id(value) if value is not None else None

    @field_validator("reference_audios")
    @classmethod
    def audio_ids_are_safe(cls, value: list[str]) -> list[str]:
        return [validate_file_id(item) for item in value]

    # Internal runner compatibility; these are not request fields and cannot be submitted.
    @property
    def source_video(self) -> None:
        return None

    @property
    def audio_mode(self) -> str:
        return "generate"

    @model_validator(mode="after")
    def validate_mode(self) -> "GenerationRequest":
        if self.duration_seconds > 15 and not self.segmentation.enabled:
            raise ValueError("duration_seconds above 15 requires segmentation.enabled=true")

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


class VideoEditRequest(VideoRequestOptions):
    width: int | None = Field(default=None, ge=32, le=8192,
        description="Explicit output width, supplied with height; overrides resolution and aspect_ratio. Without dimensions use the resolution preset, or source dimensions when resolution is omitted.")
    height: int | None = Field(default=None, ge=32, le=8192,
        description="Output height. Supply together with width, or omit both to use the resolution preset or automatic source dimensions.")
    source_video: SourceVideo = Field(description="Required source video and optional interval. Omitted interval duration uses the remainder of the source video.")
    audio_mode: Literal["generate", "preserve_source"] = "preserve_source"

    @model_validator(mode="after")
    def validate_interval(self) -> "VideoEditRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be supplied together or both omitted")
        duration = self.source_video.duration_seconds
        if duration is not None and duration > 15 and not self.segmentation.enabled:
            raise ValueError("source duration above 15 requires segmentation.enabled=true")
        return self

    @property
    def mode(self) -> GenerationMode:
        """Editing runs through Ref2V internally; no mode field exists on the wire."""
        return GenerationMode.REF2V

    @property
    def duration_seconds(self) -> float:
        if self.source_video.duration_seconds is None:
            raise ValueError("source duration must be resolved before planning")
        return self.source_video.duration_seconds

    @property
    def first_frame(self) -> None:
        return None

    @property
    def last_frame(self) -> None:
        return None

    @property
    def reference_videos(self) -> list[ReferenceVideo]:
        return []

    @property
    def reference_audios(self) -> list[str]:
        return []


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "file_id": "minimaxh3/20260822/source.mp4",
                    "width": 1280,
                    "height": 768,
                    "seed": 42,
                    "filename_prefix": "upscale/result",
                }
            ]
        }
    )

    file_id: FileId = Field(
        description="Image or video file_id returned by POST /v1/files."
    )
    width: int = Field(ge=64, le=8192, description="Exact output width in pixels.")
    height: int = Field(ge=64, le=8192, description="Exact output height in pixels.")
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**63 - 1,
        description="Optional deterministic SeedVR2 seed.",
    )
    filename_prefix: str | None = Field(
        default=None,
        max_length=240,
        description="Optional safe output path prefix relative to the output directory.",
    )

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


class DepthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: FileId = Field(description="Video file_id returned by POST /v1/files.")
    input_size: Literal[252, 392, 518] = Field(default=392, description="Inference resolution; larger values need more VRAM.")
    max_resolution: int = Field(default=768, ge=64, le=1280, description="Maximum output long edge; never upscales the source.")
    save_raw: bool = Field(default=False, description="Also return float32 relative inverse depth as NPZ (depths, fps).")

    @field_validator("file_id")
    @classmethod
    def file_id_is_safe(cls, value: str) -> str:
        return validate_file_id(value)


class SubmittedDepthTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int



class SkeletonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: FileId
    output_fps: float | None = Field(default=None, ge=1, le=120)
    start_seconds: float = Field(default=0, ge=0, le=600)
    duration_seconds: float | None = Field(default=None, gt=0, le=60)
    resolution: Literal[256, 384, 512, 576] = 384

    @field_validator("file_id")
    @classmethod
    def file_id_is_safe(cls, value: str) -> str:
        return validate_file_id(value)


class SubmittedSkeletonTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int


class SubmittedTask(BaseModel):
    segment_count: int = 1
    actual_overlap_frames: int = 0
    id: str = Field(description="Task identifier used by all task endpoints.")
    status: Literal["queued"] = "queued"
    queue_number: int | float | None = Field(
        default=None,
        description="One-based position in the shared GPU queue at submission time.",
    )
    effective_duration_seconds: float = Field(description="Frame-aligned output duration.")
    frame_count: int = Field(description="Number of video frames to generate.")


class RemovalObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=40, pattern=r"^[\w-]+$")
    frame_index: int = Field(default=0, ge=0, le=899)
    box: tuple[float, float, float, float]

    @field_validator("box")
    @classmethod
    def valid_box(cls, value):
        x1, y1, x2, y2 = value
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("box must be normalized x1,y1,x2,y2 within 0..1")
        return value


class RemovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    box: tuple[float, float, float, float] | None = None
    objects: list[RemovalObject] | None = Field(default=None, min_length=1, max_length=8)
    output_mode: Literal["foreground", "background", "both"] = "foreground"
    foreground_background: Literal["color", "transparent"] = "color"
    background_color: str = Field(default="#00ff00", pattern=r"^#[0-9a-fA-F]{6}$")
    max_side: int = Field(default=1280, ge=256, le=1920)
    repair_size: int = Field(default=640, ge=256, le=768)
    max_frames: int = Field(default=600, ge=2, le=900)
    dilation: int = Field(default=8, ge=0, le=40)

    @field_validator("file_id")
    @classmethod
    def safe_file(cls, value):
        return validate_file_id(value)

    @field_validator("box")
    @classmethod
    def valid_box(cls, value):
        return RemovalObject.valid_box(value) if value is not None else None

    @model_validator(mode="after")
    def valid_targets(self):
        if (self.box is None) == (self.objects is None):
            raise ValueError("Provide either box or objects, not both")
        if self.objects:
            ids = [obj.id.casefold() for obj in self.objects]
            if len(ids) != len(set(ids)):
                raise ValueError("Object ids must be unique (case insensitive)")
            if any(obj.frame_index >= self.max_frames for obj in self.objects):
                raise ValueError("Object frame_index must be smaller than max_frames")
        return self

    @property
    def targets(self) -> list[RemovalObject]:
        return self.objects or [RemovalObject(id="foreground", box=self.box)]


class SubmittedRemovalTask(BaseModel):
    id: str
    status: Literal["queued"] = "queued"
    queue_number: int


class SubmittedUpscaleTask(BaseModel):
    id: str = Field(description="Task identifier used by all task endpoints.")
    status: Literal["queued"] = "queued"
    queue_number: int | float | None = Field(
        default=None,
        description="One-based position in the shared GPU queue at submission time.",
    )
    width: int = Field(description="Requested output width.")
    height: int = Field(description="Requested output height.")


class OutputFile(BaseModel):
    filename: str = Field(description="Output filename.")
    subfolder: str = Field(default="", description="Output subfolder.")
    type: str = Field(default="output", description="Storage area used by the download API.")
    download_url: str = Field(description="Absolute URL for downloading the output.")


class TaskStatus(BaseModel):
    id: str = Field(description="Task identifier.")
    kind: Literal["generation", "upscale", "removal", "music", "depth", "skeleton", "beat_analysis", "transcription", "alignment", "speech"] | None = Field(
        default=None,
        description="Task type. All types use the same status endpoints.",
    )
    status: Literal["queued", "running", "succeeded", "failed", "canceled", "unknown"] = Field(
        description="Current task state."
    )
    progress: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Best-effort progress from 0 to 1.",
    )
    outputs: list[OutputFile] = Field(
        default_factory=list,
        description="Downloadable artifacts, populated after success.",
    )
    error: dict[str, Any] | str | None = Field(
        default=None,
        description="Failure detail for failed tasks.",
    )
    raw_status: dict[str, Any] | None = Field(
        default=None,
        description="Optional backend-specific status detail.",
    )


class UploadedFile(BaseModel):
    file_id: str = Field(description="Stable relative identifier used in generation/upscale requests.")
    filename: str = Field(description="Stored filename.")
    subfolder: str = Field(default="", description="Stored input subfolder.")
    type: str = Field(default="input", description="Storage area.")
    kind: Literal["image", "video", "audio"] = Field(description="Uploaded media kind.")


class HealthStatus(BaseModel):
    status: Literal["ok"] = Field(description="Process health state.")
    version: str = Field(description="Service package version.")


class ReadinessStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ready", "not_ready"] = Field(description="Overall readiness state.")
    backend: str | None = Field(default=None, description="Generation backend name.")
    models_loaded: bool | None = Field(
        default=None,
        description="Whether generation model resources are currently resident.",
    )
    upscale: dict[str, Any] | None = Field(
        default=None,
        description="SeedVR2 readiness and models_loaded state.",
    )
    depth: dict[str, Any] | None = Field(default=None, description="Video Depth Anything installation readiness and frame limit.")


class ModelCatalog(BaseModel):
    id: str = Field(description="Service model identifier.")
    backend: str = Field(description="Inference backend type.")
    models: dict[str, str] = Field(description="Configured local model filenames.")
    capabilities: dict[str, Any] = Field(description="Generation and upscale capabilities.")


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    detail: Any = Field(default=None, description="Optional structured backend detail.")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class MessageResponse(BaseModel):
    detail: str = Field(description="Human-readable HTTP error message.")
