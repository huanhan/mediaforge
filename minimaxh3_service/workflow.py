from __future__ import annotations

import secrets
from dataclasses import dataclass

from .models import GenerationMode, GenerationRequest


def frame_count_for_duration(duration_seconds: float) -> int:
    frame_count = max(5, round(duration_seconds * 24))
    return frame_count + (5 - frame_count % 17) % 17


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    start_frame: int
    frame_count: int
    trim_head: int
    output_frames: int


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    request: GenerationRequest
    seed: int
    frame_count: int
    segments: tuple[SegmentWindow, ...] = ()
    overlap_frames: int = 0

    @property
    def effective_duration_seconds(self) -> float:
        return self.frame_count / 24


def build_plan(request: GenerationRequest) -> GenerationPlan:
    duration = request.duration_seconds
    if request.source_video and request.source_video.duration_seconds is not None:
        duration = request.source_video.duration_seconds
    segmented = request.segmentation.enabled
    exact = segmented or request.mode == GenerationMode.EDIT
    # A last-frame anchor must remain at the final decoded frame, never in padding that is trimmed.
    total = round(duration * 24) if exact and not request.last_frame else frame_count_for_duration(duration)
    windows = []
    overlap = 0
    if exact:
        size = frame_count_for_duration(request.segmentation.segment_seconds if segmented else duration)
        if segmented:
            requested = request.segmentation.overlap_frames
            overlap = requested - (requested - 5) % 17
        start = 0
        emitted = 0
        while emitted < total:
            trim = overlap if windows else 0
            needed = total - start
            length = min(size, frame_count_for_duration(max(needed, trim + 1) / 24))
            keep = min(length - trim, total - emitted)
            windows.append(SegmentWindow(start, length, trim, keep))
            emitted += keep
            start += length - overlap
    return GenerationPlan(
        request=request,
        seed=request.seed if request.seed is not None else secrets.randbelow(2**63),
        frame_count=total,
        segments=tuple(windows),
        overlap_frames=overlap if len(windows) > 1 else 0,
    )


class SegmentProgress(float):
    def __new__(cls, value: float, segment: int, count: int):
        result = super().__new__(cls, value)
        result.segment = segment
        result.count = count
        return result
