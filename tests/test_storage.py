from pathlib import Path

import pytest

from mediaforge_service.backend import BackendError
from mediaforge_service.storage import LocalAssetStore


def test_resolve_input_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "input", tmp_path / "output", 1024)
    store.initialize()
    with pytest.raises(BackendError, match="invalid file_id"):
        store.resolve_input("../secret.txt")


def test_output_path_stays_under_output_root(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "input", tmp_path / "output", 1024)
    store.initialize()
    artifact = store.output_path("abcdef1234567890", "jobs/result")
    assert artifact.path.parent == (tmp_path / "output" / "jobs").resolve()
    assert artifact.filename == "result_abcdef123456.mp4"


def test_upscale_output_path_uses_png_and_stays_under_output_root(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "input", tmp_path / "output", 1024)
    store.initialize()
    artifact = store.upscale_output_path("abcdef1234567890", "jobs/upscaled")
    assert artifact.path.parent == (tmp_path / "output" / "jobs").resolve()
    assert artifact.filename == "upscaled_abcdef123456.png"
    assert artifact.file_type == "output"


def test_upscale_output_path_uses_mp4_for_video_input(tmp_path: Path) -> None:
    store = LocalAssetStore(tmp_path / "input", tmp_path / "output", 1024)
    store.initialize()
    source = store.input_root / "mediaforge" / "20260822" / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"mp4")

    artifact = store.upscale_output_path(
        "abcdef1234567890",
        "jobs/upscaled",
        source_file_id="mediaforge/20260822/source.mp4",
    )

    assert artifact.path.parent == (tmp_path / "output" / "jobs").resolve()
    assert artifact.filename == "upscaled_abcdef123456.mp4"
    assert artifact.file_type == "output"
