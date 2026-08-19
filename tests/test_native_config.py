from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from minimaxh3_service.config import Settings
from minimaxh3_service.models import GenerationMode
from minimaxh3_service.native import NativeH3Runner


def runner(settings: Settings) -> NativeH3Runner:
    return NativeH3Runner(settings, lambda value: Path(value))


def test_dynamic_vram_arguments_select_cuda_device_and_headroom() -> None:
    arguments = runner(
        Settings(device="cuda:1", vram_mode="dynamic", reserve_vram_gb=1.5)
    )._core_arguments()

    assert arguments == [
        "--cuda-device",
        "1",
        "--reserve-vram",
        "1.5",
        "--enable-dynamic-vram",
    ]


def test_cpu_device_does_not_add_conflicting_vram_arguments() -> None:
    arguments = runner(Settings(device="cpu", vram_mode="lowvram"))._core_arguments()

    assert "--cpu" in arguments
    assert "--lowvram" not in arguments


def fake_core(*, dynamic_vram_enabled: bool = False) -> SimpleNamespace:
    model_management = MagicMock()
    return SimpleNamespace(
        model_management=model_management,
        memory_management=SimpleNamespace(aimdo_enabled=True),
        aimdo_control=MagicMock(),
        dynamic_vram_enabled=dynamic_vram_enabled,
    )


def test_release_models_unloads_references_and_keeps_runtime() -> None:
    instance = runner(Settings())
    core = fake_core()
    instance._core = core
    instance._model = object()
    instance._model_mode = GenerationMode.FL2V
    instance._clip = object()
    instance._video_vae = object()
    instance._audio_vae = object()

    with patch("minimaxh3_service.native.gc.collect") as collect:
        instance.release_models()

    assert instance._core is core
    assert instance._model is None
    assert instance._model_mode is None
    assert instance._clip is None
    assert instance._video_vae is None
    assert instance._audio_vae is None
    core.model_management.unload_all_models.assert_called_once_with()
    core.model_management.soft_empty_cache.assert_called_once_with()
    collect.assert_called_once_with()
    core.aimdo_control.deinit.assert_not_called()


def test_release_models_before_runtime_load_is_a_noop() -> None:
    instance = runner(Settings())

    instance.release_models()

    assert instance._core is None


def test_close_releases_models_and_closes_dynamic_runtime() -> None:
    instance = runner(Settings())
    core = fake_core(dynamic_vram_enabled=True)
    instance._core = core
    instance._model = object()
    instance._model_mode = GenerationMode.REF2V

    with patch("minimaxh3_service.native.gc.collect"):
        instance.close()

    assert instance._core is core
    assert instance._model is None
    assert instance._model_mode is None
    assert core.memory_management.aimdo_enabled is False
    core.model_management.unload_all_models.assert_called_once_with()
    core.model_management.soft_empty_cache.assert_called_once_with()
    core.aimdo_control.deinit.assert_called_once_with()
