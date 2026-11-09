import pytest

from mediaforge_service.models import GenerationRequest, VideoEditRequest


@pytest.mark.parametrize("resolution,expected", [
    ("360p", (640, 352)), ("480p", (832, 480)), ("540p", (960, 512)),
    ("720p", (1280, 704)), ("1080p", (1344, 736)),
    ("1440p", (1344, 736)), ("2160p", (1344, 736)),
])
@pytest.mark.parametrize("model,extra", [(GenerationRequest, {}), (VideoEditRequest, {"source_video": {"file_id": "s.mp4"}})])
def test_video_presets(model, extra, resolution, expected):
    request = model(prompt="x", resolution=resolution, **extra)
    assert (request.width, request.height) == expected
    restored = model.model_validate(request.model_dump())
    assert (restored.width, restored.height) == expected


def test_generation_default_dimensions_are_preserved():
    for extra in ({}, {"aspect_ratio": "9:16"}, {"resolution": None}):
        request = GenerationRequest(prompt="x", **extra)
        assert (request.width, request.height) == (1344, 768)


@pytest.mark.parametrize("ratio", ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"])
def test_generation_and_edit_use_identical_size_limits(ratio):
    generation = GenerationRequest(prompt="x", resolution="2160p", aspect_ratio=ratio)
    editing = VideoEditRequest(prompt="x", source_video={"file_id": "s.mp4"}, resolution="2160p", aspect_ratio=ratio)
    assert (generation.width, generation.height) == (editing.width, editing.height)
    assert generation.width % 32 == generation.height % 32 == 0
    assert generation.width * generation.height <= 1032192
