import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minimaxh3_service.config import Settings
from minimaxh3_service.models import GenerationMode, GenerationRequest
from minimaxh3_service.native import NativeH3Runner
from minimaxh3_service.workflow import build_plan


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


def test_acceleration_lora_config_reads_env_and_uses_lora_directory(tmp_path: Path) -> None:
    with patch.dict(
        os.environ,
        {
            "H3_ENABLE_ACCELERATION_LORA": "true",
            "H3_MODEL_DIR": str(tmp_path),
            "H3_FL2VA_ACCELERATION_LORA": "fl2v-turbo.safetensors",
            "H3_REF2VA_ACCELERATION_LORA": "ref2v-turbo.safetensors",
        },
        clear=True,
    ):
        settings = Settings.from_env()

    assert settings.enable_acceleration_lora is True
    assert settings.model_paths["fl2va_acceleration_lora"] == (
        tmp_path / "loras" / "fl2v-turbo.safetensors"
    ).resolve()
    assert settings.model_paths["ref2va_acceleration_lora"] == (
        tmp_path / "loras" / "ref2v-turbo.safetensors"
    ).resolve()


def test_acceleration_lora_is_enabled_when_env_is_unset() -> None:
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings.from_env()

    assert settings.enable_acceleration_lora is True


@pytest.mark.parametrize(
    ("mode", "expected_lora_key"),
    [
        (GenerationMode.FL2V, "fl2va_acceleration_lora"),
        (GenerationMode.REF2V, "ref2va_acceleration_lora"),
    ],
)
def test_load_diffusion_model_applies_mode_specific_acceleration_lora(
    tmp_path: Path,
    mode: GenerationMode,
    expected_lora_key: str,
) -> None:
    settings = Settings(model_root=tmp_path, enable_acceleration_lora=True)
    instance = runner(settings)
    base_model = object()
    accelerated_model = object()
    lora = {"adapter": object()}
    metadata = {"format": "pt"}
    core = SimpleNamespace(sd=MagicMock(), utils=MagicMock())
    core.sd.load_diffusion_model.return_value = base_model
    core.utils.load_torch_file.return_value = (lora, metadata)
    core.sd.load_lora_for_models.return_value = (accelerated_model, None)
    instance._core = core

    assert instance._load_diffusion_model(mode) is accelerated_model
    core.utils.load_torch_file.assert_called_once_with(
        str(settings.model_paths[expected_lora_key]),
        safe_load=True,
        return_metadata=True,
    )
    core.sd.load_lora_for_models.assert_called_once_with(
        base_model,
        None,
        lora,
        1.0,
        0.0,
        lora_metadata=metadata,
    )


@pytest.mark.parametrize(
    ("generation_request", "expected_steps"),
    [
        (GenerationRequest(prompt="text to video"), 8),
        (
            GenerationRequest(
                prompt="reference to video",
                mode="ref2v",
                reference_images=["reference.png"],
            ),
            4,
        ),
    ],
)
def test_acceleration_lora_uses_turbo_sampling_parameters(
    generation_request: GenerationRequest,
    expected_steps: int,
) -> None:
    instance = runner(Settings(enable_acceleration_lora=True))

    assert instance._sampling_options(build_plan(generation_request)) == (
        expected_steps,
        "euler",
        "simple",
    )


def test_disabled_acceleration_lora_preserves_request_sampling_parameters() -> None:
    request = GenerationRequest(
        prompt="base model",
        steps=12,
        sampler_name="res_multistep",
        scheduler="beta",
    )

    assert runner(
        Settings(enable_acceleration_lora=False)
    )._sampling_options(build_plan(request)) == (
        12,
        "res_multistep",
        "beta",
    )


def test_readiness_requires_acceleration_lora_only_when_enabled(tmp_path: Path) -> None:
    with (
        patch("minimaxh3_service.native.importlib.util.find_spec", return_value=None),
        patch("minimaxh3_service.native.validate_mp4_encoders"),
    ):
        enabled = runner(
            Settings(model_root=tmp_path, enable_acceleration_lora=True)
        ).readiness()
        disabled = runner(
            Settings(model_root=tmp_path, enable_acceleration_lora=False)
        ).readiness()

    assert "fl2va_acceleration_lora" in enabled["missing_models"]
    assert "ref2va_acceleration_lora" in enabled["missing_optional_models"]
    assert "fl2va_acceleration_lora" not in disabled["missing_models"]
    assert "ref2va_acceleration_lora" not in disabled["missing_optional_models"]


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


def test_generation_oom_has_actionable_error_and_unloads_models(tmp_path: Path) -> None:
    import threading
    from minimaxh3_service.backend import OutputArtifact
    from minimaxh3_service.memory import GenerationMemoryError
    torch = pytest.importorskip("torch")
    instance = runner(Settings(device="cpu"))
    core = fake_core()
    core.torch = torch
    core.model_management.get_torch_device.return_value = torch.device("cpu")
    instance._core = core
    model = SimpleNamespace(model=SimpleNamespace(diffusion_model=None))
    plan = build_plan(GenerationRequest(prompt="test", width=1248, height=704, duration_seconds=8))
    with (
        patch.object(instance, "_load_common_models"),
        patch.object(instance, "_load_diffusion_model", return_value=model),
        patch.object(instance, "_build_image_conditioning", return_value=([], {})),
        patch.object(instance, "_sample", side_effect=torch.OutOfMemoryError("test allocation")),
        pytest.raises(GenerationMemoryError, match="1248×704") as caught,
    ):
        instance.generate(plan, OutputArtifact("test.mp4", "", tmp_path / "test.mp4"), threading.Event(), lambda p: None)
    assert isinstance(caught.value.__cause__, torch.OutOfMemoryError)
    core.model_management.unload_all_models.assert_called_once()
    core.model_management.soft_empty_cache.assert_called_once()
