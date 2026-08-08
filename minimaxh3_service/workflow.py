from __future__ import annotations

import secrets
from dataclasses import dataclass

from .models import GenerationRequest


def frame_count_for_duration(duration_seconds: float) -> int:
    frame_count = max(5, round(duration_seconds * 24))
    return frame_count + (5 - frame_count % 17) % 17


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    request: GenerationRequest
    seed: int
    frame_count: int

    @property
    def effective_duration_seconds(self) -> float:
        return self.frame_count / 24


def build_plan(request: GenerationRequest) -> GenerationPlan:
    return GenerationPlan(
        request=request,
        seed=request.seed if request.seed is not None else secrets.randbelow(2**63),
        frame_count=frame_count_for_duration(request.duration_seconds),
    )
