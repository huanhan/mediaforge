import sys
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mediaforge_service.config import Settings
from mediaforge_service.backend import OutputArtifact
from mediaforge_service.models import UpscaleRequest
from mediaforge_service.upscale import SeedVR2UpscaleRunner


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

    with patch("mediaforge_service.upscale.importlib.util.find_spec", return_value=object()):
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

    with patch("mediaforge_service.upscale.gc.collect") as collect:
        with patch(
            "mediaforge_service.upscale.importlib.import_module",
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

    with patch("mediaforge_service.upscale.gc.collect"):
        with patch(
            "mediaforge_service.upscale.importlib.import_module",
            return_value=memory,
        ):
            with pytest.raises(RuntimeError, match="fully release"):
                instance.release_models()

    memory.complete_cleanup.assert_called_once()
    assert instance._runner_cache["runner"] is runner


def test_image_upscale_keeps_png_path_and_image_extractor(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    output = OutputArtifact("result.png", "", tmp_path / "result.png")
    frames = MagicMock(name="image-frames")
    result = MagicMock(name="image-result")
    result.__getitem__.return_value.cpu.return_value.numpy.return_value = numpy.zeros(
        (2, 2, 3), dtype=numpy.float32
    )
    cli = SimpleNamespace(
        debug=object(),
        extract_frames_from_image=MagicMock(return_value=(frames, 30.0)),
        _single_gpu_direct_processing=MagicMock(return_value=result),
    )
    runner = make_runner(Settings())
    runner._cli = cli
    runner._args = SimpleNamespace()
    runner.resolve_input = lambda _: source
    progress_values: list[float] = []

    runner.upscale(
        UpscaleRequest(file_id="source.png", width=64, height=64, seed=11),
        output,
        threading.Event(),
        progress_values.append,
    )

    cli.extract_frames_from_image.assert_called_once_with(str(source))
    cli._single_gpu_direct_processing.assert_called_once()
    assert output.path.exists()
    assert output.path.suffix == ".png"
    assert progress_values[-1] == 0.99


def test_video_upscale_uses_video_fps_and_preserves_optional_audio(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    output = OutputArtifact("result.mp4", "", tmp_path / "result.mp4")
    frames = torch.zeros((3, 2, 2, 3), dtype=torch.float32)
    result = torch.ones((3, 2, 2, 3), dtype=torch.float32)
    audio = {"waveform": object(), "sample_rate": 32_000}
    cli = SimpleNamespace(
        debug=object(),
        extract_frames_from_image=MagicMock(),
        _single_gpu_direct_processing=MagicMock(return_value=result),
    )
    runner = make_runner(Settings())
    runner._cli = cli
    runner._args = SimpleNamespace()
    runner.resolve_input = lambda _: source

    with patch(
        "mediaforge_service.upscale.load_video",
        return_value=(frames, Fraction(12, 1)),
    ) as load_video:
        with patch(
            "mediaforge_service.upscale.load_audio",
            return_value=audio,
        ) as load_audio:
            with patch("mediaforge_service.upscale.save_mp4") as save_mp4:
                runner.upscale(
                    UpscaleRequest(file_id="source.mp4", width=64, height=64, seed=11),
                    output,
                    threading.Event(),
                    lambda _: None,
                )

    load_video.assert_called_once_with(source)
    load_audio.assert_called_once_with(source, optional=True)
    cli.extract_frames_from_image.assert_not_called()
    save_mp4.assert_called_once()
    save_args, save_kwargs = save_mp4.call_args
    assert save_args[0] == output.path
    assert save_args[2] is audio
    assert save_kwargs["fps"] == Fraction(12, 1)
    assert save_kwargs["check_canceled"] is not None
