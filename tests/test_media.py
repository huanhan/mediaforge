from pathlib import Path

import pytest

from minimaxh3_service.media import save_mp4


def test_save_mp4_muxes_video_and_stereo_audio(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    torch = pytest.importorskip("torch")
    output = tmp_path / "result.mp4"
    images = torch.zeros((4, 32, 32, 3), dtype=torch.float32)
    audio = {
        "waveform": torch.zeros((1, 2, round(4 / 24 * 32000)), dtype=torch.float32),
        "sample_rate": 32000,
    }

    save_mp4(output, images, audio, fps=24)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].average_rate == 24
        assert container.streams.audio[0].sample_rate == 32000
        assert container.streams.audio[0].layout.name == "stereo"


def test_save_mp4_checks_cancellation_during_encoding(tmp_path: Path) -> None:
    pytest.importorskip("av")
    torch = pytest.importorskip("torch")
    output = tmp_path / "canceled.mp4"
    images = torch.zeros((2, 32, 32, 3), dtype=torch.float32)
    audio = {
        "waveform": torch.zeros((1, 2, round(2 / 24 * 32000)), dtype=torch.float32),
        "sample_rate": 32000,
    }

    def cancel() -> None:
        raise RuntimeError("canceled")

    with pytest.raises(RuntimeError, match="canceled"):
        save_mp4(output, images, audio, check_canceled=cancel)
