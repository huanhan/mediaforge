from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Callable

from .backend import OutputArtifact
from .config import Settings
from .media import load_audio, load_video, save_mp4
from .models import VIDEO_EXTENSIONS, UpscaleRequest
from .tasks import GenerationCanceled


logger = logging.getLogger(__name__)
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR2_MODULE_NAME = "_minimaxh3_seedvr2_cli"
SEEDVR2_DEPENDENCIES = (
    "cv2",
    "diffusers",
    "gguf",
    "matplotlib",
    "numpy",
    "omegaconf",
    "peft",
    "PIL",
    "rotary_embedding_torch",
    "safetensors",
    "torch",
)


class SeedVR2UpscaleRunner:
    def __init__(self, settings: Settings, resolve_input: Callable[[str], Path]) -> None:
        self.settings = settings
        self.resolve_input = resolve_input
        self._cli: ModuleType | None = None
        self._args = None
        self._runner_cache: dict = {}

    def readiness(self) -> dict:
        runtime = self.settings.upscale_runtime_path.expanduser().resolve()
        missing_runtime = []
        for relative in ("inference_cli.py", "src/core/generation_utils.py"):
            if not (runtime / relative).is_file():
                missing_runtime.append(relative)
        missing_models = [
            name
            for name, path in self.settings.upscale_model_paths.items()
            if not path.is_file()
        ]
        missing_dependencies = [
            name for name in SEEDVR2_DEPENDENCIES if importlib.util.find_spec(name) is None
        ]
        reason = None
        if self.settings.device.lower() == "cpu":
            reason = "SeedVR2 upscaling requires a CUDA device"
        ready = not missing_runtime and not missing_models and not missing_dependencies and reason is None
        return {
            "status": "ready" if ready else "not_ready",
            "backend": "seedvr2",
            "runtime_path": str(runtime),
            "model_root": str(self.settings.upscale_model_root.expanduser().resolve()),
            "model": self.settings.upscale_dit_model,
            "vae": self.settings.upscale_vae_model,
            "missing_runtime": missing_runtime,
            "missing_models": missing_models,
            "missing_dependencies": missing_dependencies,
            "reason": reason,
            "models_loaded": bool(self._runner_cache),
        }

    def _import_cli(self) -> ModuleType:
        if self._cli is not None:
            return self._cli
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                try:
                    reconfigure(encoding="utf-8")
                except (AttributeError, OSError):
                    pass
        runtime = self.settings.upscale_runtime_path.expanduser().resolve()
        entrypoint = runtime / "inference_cli.py"
        if not entrypoint.is_file():
            raise RuntimeError(f"SeedVR2 runtime is missing: {entrypoint}")
        runtime_text = str(runtime)
        if runtime_text not in sys.path:
            sys.path.insert(0, runtime_text)
        spec = importlib.util.spec_from_file_location(SEEDVR2_MODULE_NAME, entrypoint)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import SeedVR2 runtime: {entrypoint}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[SEEDVR2_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(SEEDVR2_MODULE_NAME, None)
            raise
        self._cli = module
        return module

    def _device_arguments(self) -> list[str]:
        if self.settings.device.lower().startswith("cuda:"):
            return ["--cuda_device", self._device_id()]
        return []

    def _device_id(self) -> str:
        device = self.settings.device.lower()
        return device.split(":", 1)[1] if device.startswith("cuda:") else "0"

    def _build_args(self, cli: ModuleType):
        arguments = [
            "inference_cli.py",
            "__minimaxh3_upscale__.png",
            "--model_dir",
            str(self.settings.upscale_model_root.expanduser().resolve()),
            "--dit_model",
            self.settings.upscale_dit_model,
            "--batch_size",
            "1",
            "--seed",
            str(self.settings.upscale_seed),
            "--blocks_to_swap",
            str(self.settings.upscale_blocks_to_swap),
            "--dit_offload_device",
            "cpu",
            "--vae_offload_device",
            "cpu",
            "--tensor_offload_device",
            "cpu",
            "--vae_encode_tiled",
            "--vae_encode_tile_size",
            str(self.settings.upscale_vae_tile_size),
            "--vae_decode_tiled",
            "--vae_decode_tile_size",
            str(self.settings.upscale_vae_tile_size),
            "--color_correction",
            "wavelet",
            "--cache_dit",
            "--cache_vae",
            *self._device_arguments(),
        ]
        if self.settings.upscale_swap_io:
            arguments.append("--swap_io_components")
        original_argv = sys.argv
        try:
            sys.argv = arguments
            return cli.parse_arguments()
        finally:
            sys.argv = original_argv

    def upscale(
        self,
        request: UpscaleRequest,
        output: OutputArtifact,
        cancel: threading.Event,
        progress: Callable[[float], None],
    ) -> None:
        def check_canceled() -> None:
            if cancel.is_set():
                raise GenerationCanceled("upscale canceled")

        check_canceled()
        cli = self._import_cli()
        if self._args is None:
            self._args = self._build_args(cli)
        self._args.resolution = min(request.width, request.height)
        self._args.max_resolution = max(request.width, request.height)
        self._args.seed = request.seed if request.seed is not None else self.settings.upscale_seed

        progress(0.05)
        source = self.resolve_input(request.file_id)
        is_video = source.suffix.lower() in VIDEO_EXTENSIONS
        fps = None
        audio = None
        if is_video:
            frames, fps = load_video(source)
            audio = load_audio(source, optional=True)
        else:
            frames, _ = cli.extract_frames_from_image(str(source))
        check_canceled()
        progress(0.1)
        try:
            result = cli._single_gpu_direct_processing(
                frames,
                self._args,
                self._device_id(),
                self._runner_cache,
            )
            check_canceled()
            progress(0.9)

            if is_video:
                import torch.nn.functional as functional

                if tuple(result.shape[1:3]) != (request.height, request.width):
                    result = functional.interpolate(
                        result[..., :3].movedim(-1, 1),
                        size=(request.height, request.width),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).movedim(1, -1)
                save_mp4(
                    output.path,
                    result,
                    audio,
                    fps=fps or 24,
                    check_canceled=check_canceled,
                )
            else:
                import numpy as np
                from PIL import Image

                frame = (result[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                image = Image.fromarray(frame[..., :3], "RGB")
                if image.size != (request.width, request.height):
                    image = image.resize((request.width, request.height), Image.Resampling.LANCZOS)
                output.path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output.path, format="PNG")
            progress(0.99)
        except Exception:
            self.release_models()
            raise

    def release_models(self) -> None:
        if self._cli is None or not self._runner_cache:
            return
        runner = self._runner_cache.get("runner")
        ctx = self._runner_cache.get("ctx") or {}
        cache_context = ctx.get("cache_context") or {}
        global_cache = cache_context.get("global_cache")
        errors: list[Exception] = []
        try:
            if global_cache is None:
                model_cache = importlib.import_module("src.core.model_cache")
                global_cache = model_cache.get_global_cache()
        except Exception as exc:
            errors.append(exc)
        dit_removed = False
        vae_removed = False
        if global_cache is not None:
            try:
                dit_removed = global_cache.remove_dit(
                    {"node_id": "cli_dit"}, debug=self._cli.debug
                )
            except Exception as exc:
                errors.append(exc)
            try:
                vae_removed = global_cache.remove_vae(
                    {"node_id": "cli_vae"}, debug=self._cli.debug
                )
            except Exception as exc:
                errors.append(exc)
        try:
            if runner is not None:
                if dit_removed:
                    runner.dit = None
                if vae_removed:
                    runner.vae = None
                memory = importlib.import_module("src.optimization.memory_manager")
                memory.complete_cleanup(
                    runner,
                    debug=self._cli.debug,
                    dit_cache=False,
                    vae_cache=False,
                )
        except Exception as exc:
            errors.append(exc)
        if not errors:
            self._runner_cache.clear()
        gc.collect()
        if errors:
            raise RuntimeError("failed to fully release SeedVR2 resources") from errors[0]

    def close(self) -> None:
        self.release_models()
