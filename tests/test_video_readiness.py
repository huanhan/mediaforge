import asyncio
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend


@pytest.mark.parametrize("video_status,expected_code", [("ready", 200), ("not_ready", 503)])
def test_video_readiness_checks_h3_without_optional_probes(tmp_path, video_status, expected_code):
    video = Mock()
    video.readiness.return_value = {"status": video_status, "backend": "native", "reason": "video check"}
    optional = Mock()
    optional.readiness.side_effect = AssertionError("video readiness must not probe optional runtimes")
    backend = LocalGenerationBackend(
        Settings(data_root=tmp_path), runner=video, upscale_runner=optional,
        removal_runner=optional, music_runner=optional, depth_runner=optional,
        skeleton_runner=optional, beat_runner=optional, transcription_runner=optional,
        alignment_runner=optional, speech_runner=optional,
    )
    with TestClient(create_app(Settings(data_root=tmp_path), backend_factory=lambda _: backend)) as client:
        response = client.get("/readyz?capability=video")
        assert response.status_code == expected_code
        assert response.json() == video.readiness.return_value
        assert client.get("/readyz?capability=invalid").status_code == 422
    video.readiness.assert_called_once()
    optional.readiness.assert_not_called()


def test_default_readiness_keeps_full_capability_catalog(tmp_path):
    video = Mock()
    video.readiness.return_value = {"status": "ready", "backend": "native"}
    optional = Mock()
    optional.readiness.return_value = {"status": "ready", "backend": "optional"}
    backend = LocalGenerationBackend(
        Settings(data_root=tmp_path), runner=video, upscale_runner=optional,
        removal_runner=optional, music_runner=optional, depth_runner=optional,
        skeleton_runner=optional, beat_runner=optional, transcription_runner=optional,
        alignment_runner=optional, speech_runner=optional,
    )
    result = asyncio.run(backend.ready())
    for name in ("upscale", "removal", "speech", "music", "depth", "skeleton", "beat_analysis", "transcription", "alignment"):
        assert result[name] == optional.readiness.return_value
