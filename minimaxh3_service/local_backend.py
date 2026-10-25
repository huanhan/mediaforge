from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from .backend import BackendSubmission, BackendTask, StoredAsset, UpscaleSubmission
from .backend import BackendError
from .config import Settings
from .models import MAX_CANVAS_PIXELS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, GenerationRequest, VideoEditRequest, UpscaleRequest, RemovalRequest
from .removal import VideoRemovalRunner
from .native import NativeH3Runner
from .media import video_dimensions, video_duration
from .storage import LocalAssetStore
from .tasks import H3Runner, LocalTaskManager, UpscaleRunner
from .upscale import SeedVR2UpscaleRunner
from .music import MusicRunner
from .speech import SpeechRunner
from .speech_models import SpeechRequest, SubmittedSpeechTask
from .models import MusicRequest, SubmittedMusicTask
from .models import DepthRequest, SubmittedDepthTask
from .depth import VideoDepthRunner
from .skeleton import SkeletonRunner
from .models import SkeletonRequest, SubmittedSkeletonTask
from .models import AUDIO_EXTENSIONS, BeatAnalysisRequest, SubmittedBeatAnalysisTask
from .beat import BeatAnalysisRunner
from .models import TranscriptionRequest, SubmittedTranscriptionTask
from .transcription import TranscriptionRunner
from .alignment import AlignmentRunner
from .models import AlignmentRequest


def source_canvas(width: int, height: int) -> tuple[int, int]:
    """Fit source dimensions to H3 limits, then round down to the 32-pixel grid."""
    if width <= 0 or height <= 0:
        raise ValueError("source video has invalid dimensions")
    scale = min(1.0, 8192 / width, 8192 / height,
                math.sqrt(MAX_CANVAS_PIXELS / (width * height)))
    return max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)


class LocalGenerationBackend:
    def __init__(
        self,
        settings: Settings,
        runner: H3Runner | None = None,
        upscale_runner: UpscaleRunner | None = None,
        removal_runner=None,
        music_runner=None,
        depth_runner=None,
        skeleton_runner=None,
        beat_runner=None,
        transcription_runner=None,
        alignment_runner=None,
        speech_runner=None,
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
            removal_runner=removal_runner or VideoRemovalRunner(settings, self.assets.resolve_input),
            music_runner=music_runner or MusicRunner(settings),
            music_output_factory=self.assets.music_output_path,
            depth_runner=depth_runner or VideoDepthRunner(settings, self.assets.resolve_input),
            skeleton_runner=skeleton_runner or SkeletonRunner(settings, self.assets.resolve_input),
            beat_runner=beat_runner or BeatAnalysisRunner(settings, self.assets.resolve_input),
            transcription_runner=transcription_runner or TranscriptionRunner(settings, self.assets.resolve_input),
            speech_runner=speech_runner or SpeechRunner(settings, self.assets.resolve_input),
            alignment_runner=alignment_runner or AlignmentRunner(settings, self.assets.resolve_input),
        )

    async def start(self) -> None:
        self.assets.initialize()
        await self.tasks.start()

    async def close(self) -> None:
        await self.tasks.close()

    async def ready(self, *, capability: Literal["video"] | None = None) -> dict:
        readiness = dict(await asyncio.to_thread(self.runner.readiness))
        # Video submission must not wait for unrelated speech/music subprocess
        # probes (which can take longer than the video's readiness timeout).
        if capability == "video":
            return readiness
        readiness["upscale"] = self.upscale_runner.readiness()
        readiness["removal"] = self.tasks.removal_runner.readiness()
        readiness["speech"] = await asyncio.to_thread(self.tasks.speech_runner.readiness)
        readiness["music"] = await asyncio.to_thread(self.tasks.music_runner.readiness)
        readiness["depth"] = self.tasks.depth_runner.readiness()
        readiness["skeleton"] = self.tasks.skeleton_runner.readiness()
        readiness["beat_analysis"] = self.tasks.beat_runner.readiness()
        readiness["transcription"] = self.tasks.transcription_runner.readiness()
        readiness["alignment"] = self.tasks.alignment_runner.readiness()
        return readiness

    async def upload(
        self,
        upload: UploadFile,
        *,
        kind: Literal["image", "video", "audio"],
    ) -> StoredAsset:
        return await self.assets.store_upload(upload, kind=kind)

    async def submit(self, request: GenerationRequest) -> BackendSubmission:
        return await self._submit_video(request)

    async def submit_edit(self, request: VideoEditRequest) -> BackendSubmission:
        return await self._submit_video(request)

    async def _submit_video(self, request: GenerationRequest | VideoEditRequest) -> BackendSubmission:
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
        if (
            request.mode.value == "ref2v"
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
                    duration = remaining
                if duration > remaining + 1 / 24 or remaining < 4:
                    raise ValueError("source_video interval exceeds the available video or is shorter than 4 seconds")
                duration = min(duration, remaining)
                data = request.model_dump()
                data["source_video"]["duration_seconds"] = duration
                if request.width is None:
                    dimensions = await asyncio.to_thread(video_dimensions, source)
                    data["width"], data["height"] = source_canvas(*dimensions)
                request = VideoEditRequest.model_validate(data)
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

    async def submit_removal(self, request: RemovalRequest):
        readiness = self.tasks.removal_runner.readiness()
        if not readiness.get("foreground_available") or (
            request.output_mode != "foreground" and not readiness.get("background_available")
        ):
            raise BackendError("人物分离 / 背景修复模型或依赖未安装，参见 docs/VIDEO_REMOVAL.md",
                               status_code=503, code="removal_not_ready", detail=readiness)
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise BackendError("removal requires an uploaded video", status_code=415, code="invalid_video")
        return await self.tasks.submit_removal(request)

    async def submit_speech(self, request: SpeechRequest) -> SubmittedSpeechTask:
        if request.reference_file_id:
            source = self.assets.resolve_input(request.reference_file_id)
            if source.suffix.lower() not in AUDIO_EXTENSIONS:
                raise BackendError("Voice reference must be uploaded audio", status_code=422, code="invalid_audio")
        readiness = await asyncio.to_thread(self.tasks.speech_runner.readiness)
        if readiness.get("status") != "ready":
            raise BackendError("IndexTTS 2.5 is not installed; run python -m minimaxh3_service.setup_tts",
                               status_code=503, code="speech_not_ready", detail=readiness)
        return await self.tasks.submit_speech(request)

    async def submit_music(self, request: MusicRequest) -> SubmittedMusicTask:
        validate = getattr(self.tasks.music_runner, "validate_request", None)
        if validate is not None:
            validate(request)
        readiness = await asyncio.to_thread(self.tasks.music_runner.readiness)
        if readiness.get("status") != "ready":
            raise BackendError("音乐模型未就绪，请检查模型安装与配置，参见 docs/MUSIC.md",
                               status_code=503, code="music_not_ready", detail=readiness)
        return await self.tasks.submit_music(request)

    async def submit_depth(self, request: DepthRequest) -> SubmittedDepthTask:
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise BackendError("depth requires an uploaded video", status_code=415, code="invalid_video")
        readiness = self.tasks.depth_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError("Video Depth Anything is not installed; see docs/DEPTH.md",
                               status_code=503, code="depth_not_ready", detail=readiness)
        return await self.tasks.submit_depth(request)

    async def submit_skeleton(self, request: SkeletonRequest) -> SubmittedSkeletonTask:
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise BackendError("skeleton extraction requires a video", status_code=415, code="invalid_video")
        readiness = self.tasks.skeleton_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError("Local DWPose is not installed", status_code=503, code="skeleton_not_ready", detail=readiness)
        return await self.tasks.submit_skeleton(request)

    async def submit_beat_analysis(self, request: BeatAnalysisRequest) -> SubmittedBeatAnalysisTask:
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise BackendError("beat analysis requires uploaded audio", status_code=415, code="invalid_audio")
        readiness = self.tasks.beat_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError("Beat analysis is not installed; see docs/BEAT_ANALYSIS.md",
                               status_code=503, code="beat_analysis_not_ready", detail=readiness)
        return await self.tasks.submit_beat_analysis(request)

    async def submit_transcription(self, request: TranscriptionRequest) -> SubmittedTranscriptionTask:
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise BackendError("transcription requires uploaded audio", status_code=415, code="invalid_audio")
        readiness = self.tasks.transcription_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError("Qwen3-ASR is not installed; see docs/TRANSCRIPTION.md",
                               status_code=503, code="transcription_not_ready", detail=readiness)
        return await self.tasks.submit_transcription(request)

    async def submit_alignment(self, request: AlignmentRequest) -> SubmittedTranscriptionTask:
        source = self.assets.resolve_input(request.file_id)
        if source.suffix.lower() not in AUDIO_EXTENSIONS:
            raise BackendError("alignment requires uploaded audio", status_code=415, code="invalid_audio")
        readiness = self.tasks.alignment_runner.readiness()
        if readiness.get("status") != "ready":
            raise BackendError("Install the lyric aligner: python -m minimaxh3_service.setup_asr --aligner",
                               status_code=503, code="alignment_not_ready", detail=readiness)
        return await self.tasks.submit_alignment(request)

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
