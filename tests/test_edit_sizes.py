import pytest

from minimaxh3_service.models import GenerationRequest, VideoEditRequest


@pytest.mark.parametrize("resolution,expected", [
    ("360p", (640, 352)), ("480p", (832, 480)), ("540p", (960, 512)),
    ("720p", (1280, 704)), ("1080p", (1344, 736)),
    ("1440p", (1344, 736)), ("2160p", (1344, 736)),
])
def test_edit_presets(resolution, expected):
    request = VideoEditRequest(prompt="x", source_video={"file_id": "s.mp4"}, resolution=resolution)
    assert (request.width, request.height) == expected
    restored = VideoEditRequest.model_validate(request.model_dump())
    assert (restored.width, restored.height) == expected


def test_generation_contract_unchanged():
    request = GenerationRequest(prompt="x")
    assert (request.width, request.height) == (1344, 768)
    for extra in ({"resolution": "480p"}, {"aspect_ratio": "9:16"}):
        with pytest.raises(ValueError):
            GenerationRequest(prompt="x", **extra)
