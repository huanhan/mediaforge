from pathlib import Path

import pytest
from pydantic import ValidationError

from minimaxh3_service.config import Settings
from minimaxh3_service.models import GenerationRequest, ReferenceVideo
from minimaxh3_service.workflow import build_plan, frame_count_for_duration


def test_frame_count_uses_h3_grid() -> None:
    assert frame_count_for_duration(4) == 107
    assert frame_count_for_duration(5) == 124
    assert frame_count_for_duration(15) == 362
    assert all(frame_count_for_duration(seconds) % 17 == 5 for seconds in (4, 5, 8, 10, 15))


def test_build_plan_preserves_request_and_seed() -> None:
    request = GenerationRequest(
        prompt="A cinematic city scene",
        seed=42,
        width=864,
        height=480,
    )
    plan = build_plan(request)

    assert plan.request is request
    assert plan.seed == 42
    assert plan.frame_count == 124
    assert plan.effective_duration_seconds == pytest.approx(124 / 24)


def test_build_plan_generates_seed_when_omitted() -> None:
    plan = build_plan(GenerationRequest(prompt="A slow camera move"))
    assert 0 <= plan.seed < 2**63


def test_model_paths_use_local_model_layout(tmp_path: Path) -> None:
    settings = Settings(model_root=tmp_path / "models")
    assert settings.model_paths["fl2va"] == (
        tmp_path / "models" / "diffusion_models" / settings.fl2va_model
    ).resolve()
    assert settings.model_paths["text_encoder"].parent.name == "text_encoders"
    assert settings.model_paths["video_vae"].parent.name == "vae"


def test_rejects_invalid_mode_inputs_and_canvas() -> None:
    with pytest.raises(ValidationError, match="t2v does not accept"):
        GenerationRequest(prompt="x", first_frame="first.png")
    with pytest.raises(ValidationError, match="requires first_frame"):
        GenerationRequest(prompt="x", mode="fl2v")
    with pytest.raises(ValidationError, match="at least one reference image or video"):
        GenerationRequest(prompt="x", mode="ref2v", reference_audios=["voice.wav"])
    with pytest.raises(ValidationError, match="at most 3 audio references"):
        GenerationRequest(
            prompt="x",
            mode="ref2v",
            reference_videos=[ReferenceVideo(file_id="v1.mp4"), ReferenceVideo(file_id="v2.mp4")],
            reference_audios=["a1.wav", "a2.wav"],
        )
    with pytest.raises(ValidationError, match="multiples of 32"):
        GenerationRequest(prompt="x", width=1000)
    with pytest.raises(ValidationError, match="canvas must not exceed"):
        GenerationRequest(prompt="x", width=2048, height=768)
