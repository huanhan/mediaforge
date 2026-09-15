from fractions import Fraction
from pathlib import Path

import pytest

from minimaxh3_service.media import load_image, load_video, load_video_frames, save_mp4


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


def test_video_without_audio_round_trips_frames_and_fractional_fps(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    torch = pytest.importorskip("torch")
    output = tmp_path / "silent.mp4"
    images = torch.zeros((4, 32, 48, 3), dtype=torch.float32)

    save_mp4(output, images, None, fps=Fraction(12, 1))
    decoded, fps = load_video(output)

    assert tuple(decoded.shape) == (4, 32, 48, 3)
    assert fps == Fraction(12, 1)
    with av.open(str(output)) as container:
        assert len(container.streams.audio) == 0


@pytest.mark.parametrize("sampled", [False, True])
def test_transparent_video_hides_rgb_background_before_resize(tmp_path: Path, sampled: bool) -> None:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    output = tmp_path / "transparent.mov"
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[:, :16] = [0, 0, 255, 0]  # Hidden original blue background.
    pixels[:16, 16:] = [255, 0, 0, 255]  # Opaque subject.
    pixels[16:, 16:] = [255, 0, 0, 128]  # Soft mask edge.
    with av.open(str(output), "w") as container:
        stream = container.add_stream("png", rate=24)
        stream.width = stream.height = 32
        stream.pix_fmt = "rgba"
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgba")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    if sampled:
        frames = load_video_frames(output, width=16, height=16, max_frames=1)
    else:
        frames, _ = load_video(output)
    rgb = frames[0].numpy()
    np.testing.assert_allclose(rgb[2, 2], [1, 1, 1], atol=1 / 255)
    np.testing.assert_allclose(rgb[2, -3], [1, 0, 0], atol=1 / 255)
    np.testing.assert_allclose(rgb[-3, -3], [1, 127 / 255, 127 / 255], atol=1 / 255)
    # No blue fringe from resizing hidden RGB values across the mask boundary.
    assert np.max(np.abs(rgb[..., 1] - rgb[..., 2])) <= 1 / 255


def test_transparent_reference_image_uses_white_background(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "reference.png"
    image = Image.new("RGBA", (3, 1), (0, 0, 255, 0))
    image.putpixel((1, 0), (255, 0, 0, 128))
    image.putpixel((2, 0), (255, 0, 0, 255))
    image.save(path)
    np.testing.assert_allclose(load_image(path)[0, 0].numpy(),
                               [[1, 1, 1], [1, 127 / 255, 127 / 255], [1, 0, 0]], atol=1 / 255)
