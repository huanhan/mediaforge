from fractions import Fraction
from pathlib import Path

import pytest

from minimaxh3_service.video_remove import extract_frames, propainter_command, validate_masks


def make_video(path, count=4):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(30000, 1001))
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "yuv420p"
        for i in range(count):
            array = np.full((64, 96, 3), i * 30, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_extract_preserves_fractional_rate(tmp_path):
    video = tmp_path / "input.mp4"
    make_video(video)
    count, size, fps = extract_frames(video, tmp_path / "frames", 640, 10)
    assert (count, size, fps) == (4, (96, 64), Fraction(30000, 1001))


def test_long_input_rejected_instead_of_truncated(tmp_path):
    video = tmp_path / "input.mp4"
    make_video(video)
    with pytest.raises(ValueError, match="exceeds"):
        extract_frames(video, tmp_path / "frames", 640, 2)


def test_masks_require_exact_count_and_size(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("L", (96, 64)).save(tmp_path / "000000.png")
    with pytest.raises(ValueError, match="Expected 2"):
        validate_masks(tmp_path, 2, (96, 64))
    with pytest.raises(ValueError, match="must have size"):
        validate_masks(tmp_path, 1, (64, 64))
    assert len(validate_masks(tmp_path, 1, (96, 64))) == 1


def test_official_command_preserves_paths_with_spaces(tmp_path):
    repo = tmp_path / "Pro Painter"
    command = propainter_command(Path("python"), repo, tmp_path / "frames",
                                tmp_path / "masks", tmp_path / "out", 4)
    assert command[1] == str(repo / "inference_propainter.py")
    assert "--save_frames" in command
    assert "--mask_dilation" in command
