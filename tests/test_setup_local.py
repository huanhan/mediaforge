import sys
from pathlib import Path
from types import SimpleNamespace

from minimaxh3_service.setup_local import (
    FL2VA_ACCELERATION_LORA_FILE,
    REF2VA_ACCELERATION_LORA_FILE,
    build_parser,
    download_models,
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
