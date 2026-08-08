from pathlib import Path

import pytest

from minimaxh3_service.backend import BackendError
from minimaxh3_service.storage import LocalAssetStore


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
