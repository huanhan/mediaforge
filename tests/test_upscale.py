import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from minimaxh3_service.config import Settings
from minimaxh3_service.upscale import SeedVR2UpscaleRunner


def make_runner(settings: Settings) -> SeedVR2UpscaleRunner:
    return SeedVR2UpscaleRunner(settings, lambda value: Path(value))


def test_readiness_checks_runtime_models_and_dependencies(tmp_path: Path) -> None:
    runtime = tmp_path / "SeedVR2"
    (runtime / "src" / "core").mkdir(parents=True)
    (runtime / "inference_cli.py").touch()
    (runtime / "src" / "core" / "generation_utils.py").touch()
    model_root = tmp_path / "models"
    model_root.mkdir()
    settings = Settings(
        upscale_runtime_path=runtime,
        upscale_model_root=model_root,
    )
    for path in settings.upscale_model_paths.values():
        path.touch()

    with patch("minimaxh3_service.upscale.importlib.util.find_spec", return_value=object()):
        readiness = make_runner(settings).readiness()

    assert readiness["status"] == "ready"
    assert readiness["models_loaded"] is False
    assert readiness["missing_runtime"] == []
    assert readiness["missing_models"] == []


def test_build_args_uses_configured_memory_options_without_leaking_argv(tmp_path: Path) -> None:
    captured: list[str] = []

    class FakeCli:
        @staticmethod
        def parse_arguments() -> SimpleNamespace:
            captured.extend(sys.argv)
            return SimpleNamespace()

    settings = Settings(
        device="cuda:1",
        upscale_model_root=tmp_path,
        upscale_blocks_to_swap=20,
        upscale_vae_tile_size=384,
        upscale_swap_io=True,
    )
    instance = make_runner(settings)
    original = sys.argv

    instance._build_args(FakeCli())

    assert sys.argv is original
    assert captured[captured.index("--cuda_device") + 1] == "1"
    assert instance._device_id() == "1"
    assert captured[captured.index("--blocks_to_swap") + 1] == "20"
    assert captured[captured.index("--vae_encode_tile_size") + 1] == "384"
    assert "--cache_dit" in captured
    assert "--cache_vae" in captured
    assert "--swap_io_components" in captured


def test_release_models_clears_seedvr2_global_cache_and_runner() -> None:
    instance = make_runner(Settings())
    global_cache = MagicMock()
    global_cache.remove_dit.return_value = True
    global_cache.remove_vae.return_value = True
    runner = SimpleNamespace(dit=object(), vae=object())
    instance._cli = SimpleNamespace(debug=object())
    instance._runner_cache = {
        "runner": runner,
        "ctx": {"cache_context": {"global_cache": global_cache}},
    }
    memory = SimpleNamespace(complete_cleanup=MagicMock())

    with patch("minimaxh3_service.upscale.gc.collect") as collect:
        with patch(
            "minimaxh3_service.upscale.importlib.import_module",
            return_value=memory,
        ):
            instance.release_models()

    global_cache.remove_dit.assert_called_once_with(
        {"node_id": "cli_dit"}, debug=instance._cli.debug
    )
    global_cache.remove_vae.assert_called_once_with(
        {"node_id": "cli_vae"}, debug=instance._cli.debug
    )
    assert runner.dit is None
    assert runner.vae is None
    memory.complete_cleanup.assert_called_once_with(
        runner,
        debug=instance._cli.debug,
        dit_cache=False,
        vae_cache=False,
    )
    assert instance._runner_cache == {}
    collect.assert_called_once_with()


def test_release_failure_keeps_cache_state_for_retry() -> None:
    instance = make_runner(Settings())
    global_cache = MagicMock()
    global_cache.remove_dit.side_effect = RuntimeError("cache busy")
    global_cache.remove_vae.return_value = True
    runner = SimpleNamespace(dit=object(), vae=object())
    instance._cli = SimpleNamespace(debug=object())
    instance._runner_cache = {
        "runner": runner,
        "ctx": {"cache_context": {"global_cache": global_cache}},
    }
    memory = SimpleNamespace(complete_cleanup=MagicMock())

    with patch("minimaxh3_service.upscale.gc.collect"):
        with patch(
            "minimaxh3_service.upscale.importlib.import_module",
            return_value=memory,
        ):
            with pytest.raises(RuntimeError, match="fully release"):
                instance.release_models()

    memory.complete_cleanup.assert_called_once()
    assert instance._runner_cache["runner"] is runner
