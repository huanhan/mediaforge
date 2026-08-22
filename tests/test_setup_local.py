import sys
from pathlib import Path
from types import SimpleNamespace

from minimaxh3_service.setup_local import (
    FL2VA_ACCELERATION_LORA_FILE,
    REF2VA_ACCELERATION_LORA_FILE,
    UPSCALE_MODEL_FILES,
    build_parser,
    download_models,
    download_upscale_models,
)


def test_setup_enables_acceleration_lora_download_by_default() -> None:
    assert build_parser().parse_args([]).with_acceleration_lora is True
    assert (
        build_parser()
        .parse_args(["--without-acceleration-lora"])
        .with_acceleration_lora
        is False
    )


def test_download_models_includes_requested_acceleration_loras(
    tmp_path: Path,
    monkeypatch,
) -> None:
    downloaded: list[str] = []

    def fake_download(*, filename: str, **_kwargs) -> None:
        downloaded.append(filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    download_models(
        tmp_path,
        with_ref2v=True,
        with_acceleration_lora=True,
        token=None,
    )

    assert FL2VA_ACCELERATION_LORA_FILE in downloaded
    assert REF2VA_ACCELERATION_LORA_FILE in downloaded


def test_download_models_omits_acceleration_loras_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    downloaded: list[str] = []

    def fake_download(*, filename: str, **_kwargs) -> None:
        downloaded.append(filename)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    download_models(
        tmp_path,
        with_ref2v=False,
        with_acceleration_lora=False,
        token=None,
    )

    assert FL2VA_ACCELERATION_LORA_FILE not in downloaded
    assert REF2VA_ACCELERATION_LORA_FILE not in downloaded


def test_setup_enables_upscale_install_by_default() -> None:
    assert build_parser().parse_args([]).with_upscale is True
    assert build_parser().parse_args(["--without-upscale"]).with_upscale is False


def test_download_upscale_models_uses_seedvr2_repository(
    tmp_path: Path,
    monkeypatch,
) -> None:
    downloaded: list[tuple[str, str, Path]] = []

    def fake_download(*, repo_id: str, filename: str, local_dir: Path, **_kwargs) -> None:
        downloaded.append((repo_id, filename, local_dir))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_download),
    )

    download_upscale_models(tmp_path, token=None)

    assert {filename for _, filename, _ in downloaded} == set(UPSCALE_MODEL_FILES)
    assert {repo for repo, _, _ in downloaded} == {"numz/SeedVR2_comfyUI"}
    assert {local_dir for _, _, local_dir in downloaded} == {tmp_path}
