from __future__ import annotations

import gc
import importlib
import importlib.util
import math
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .backend import OutputArtifact
from .config import Settings
from .memory import GenerationMemoryError, h3_memory_scope
from .media import (
    load_audio,
    load_image,
    load_video_frames,
    save_mp4,
    validate_mp4_encoders,
    video_dimensions,
)
from .models import GenerationMode, ReferenceVideo
from .tasks import GenerationCanceled
from .workflow import GenerationPlan


COMFY_CORE_COMMIT = "dd79c643a95402136a75a28f6187d843bcf457ed"
FPS = 24
AUDIO_LATENT_FPS = 40
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
DISCARD_PENULTIMATE_SIGMA_SAMPLERS = {
    "dpm_2",
    "dpm_2_ancestral",
    "uni_pc",
    "uni_pc_bh2",
}
ACCELERATION_STEPS = {
    GenerationMode.T2V: 8,
    GenerationMode.FL2V: 8,
    GenerationMode.REF2V: 4,
}
ACCELERATION_SAMPLER = "euler"
ACCELERATION_SCHEDULER = "simple"


def _conditioning_set_values(conditioning: list, values: dict[str, Any]) -> list:
    output = []
    for tensor, metadata in conditioning:
        updated = metadata.copy()
        updated.update(values)
        output.append([tensor, updated])
    return output


def _video_latent_t(frame_count: int) -> int:
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def _adapt_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(CANVAS_MULTIPLE, round(nominal_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nominal_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


class NativeH3Runner:
    def __init__(self, settings: Settings, resolve_input: Callable[[str], Path]) -> None:
        self.settings = settings
        self.resolve_input = resolve_input
        self._core: SimpleNamespace | None = None
        self._clip: Any = None
        self._video_vae: Any = None
        self._audio_vae: Any = None
        self._model: Any = None
        self._model_mode: GenerationMode | None = None

    def _core_arguments(self) -> list[str]:
        arguments: list[str] = []
        device = self.settings.device.lower()
        cpu_requested = device == "cpu" or self.settings.vram_mode == "cpu"
        if cpu_requested:
            arguments.append("--cpu")
        elif device.startswith("cuda:"):
            arguments.extend(["--cuda-device", device.split(":", 1)[1]])

        if self.settings.reserve_vram_gb > 0:
            arguments.extend(["--reserve-vram", str(self.settings.reserve_vram_gb)])
        if self.settings.vram_mode == "dynamic" and not cpu_requested:
            arguments.append("--enable-dynamic-vram")
        elif not cpu_requested and self.settings.vram_mode in {"lowvram", "novram", "highvram"}:
            arguments.extend([f"--{self.settings.vram_mode}", "--disable-dynamic-vram"])
        return arguments

    def readiness(self) -> dict[str, Any]:
        runtime = self.settings.runtime_path.expanduser().resolve()
        missing_runtime = []
        for relative in ("comfy/sd.py", "comfy/ldm/minimax/model.py", "comfy/text_encoders/minimax.py"):
            if not (runtime / relative).is_file():
                missing_runtime.append(relative)
        required_model_keys = ["fl2va", "text_encoder", "video_vae", "audio_vae"]
        if self.settings.enable_acceleration_lora:
            required_model_keys.append("fl2va_acceleration_lora")
        missing_models = [
            name for name in required_model_keys if not self.settings.model_paths[name].is_file()
        ]
        optional_model_keys = ["ref2va"]
        if self.settings.enable_acceleration_lora:
            optional_model_keys.append("ref2va_acceleration_lora")
        missing_optional_models = [
            name
            for name in optional_model_keys
            if not self.settings.model_paths[name].is_file()
        ]

        devices: list[dict[str, Any]] = []
        runtime_error: str | None = None
        if importlib.util.find_spec("torch") is None:
            runtime_error = "PyTorch is not installed; run the local setup first"
        else:
            try:
                import torch

                device = self.settings.device.lower()
                if self.settings.vram_mode not in {
                    "dynamic",
                    "lowvram",
                    "novram",
                    "highvram",
                    "cpu",
                }:
                    runtime_error = "H3_VRAM_MODE is invalid"
                if self.settings.reserve_vram_gb < 0:
                    runtime_error = "H3_RESERVE_VRAM_GB must not be negative"
                if torch.cuda.is_available():
                    for index in range(torch.cuda.device_count()):
                        properties = torch.cuda.get_device_properties(index)
                        devices.append(
                            {
                                "name": properties.name,
                                "type": "cuda",
                                "index": index,
                                "total_vram": properties.total_memory,
                                "torch_cuda": torch.version.cuda,
                            }
                        )
                elif device.startswith("cuda"):
                    runtime_error = "torch.cuda.is_available() is false"

                if device.startswith("cuda:"):
                    index = int(device.split(":", 1)[1])
                    if index < 0 or index >= torch.cuda.device_count():
                        runtime_error = f"CUDA device index {index} is not available"
                elif device not in {"cuda", "cpu"}:
                    runtime_error = "H3_DEVICE must be cuda, cuda:N, or cpu"

                version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
                if (
                    self.settings.vram_mode == "dynamic"
                    and device != "cpu"
                    and version < (2, 8)
                ):
                    runtime_error = "DynamicVRAM requires PyTorch 2.8 or later"
            except Exception as exc:
                runtime_error = str(exc)

        try:
            validate_mp4_encoders()
        except Exception as exc:
            runtime_error = f"MP4 encoders are unavailable: {exc}"

        ready = not missing_runtime and not missing_models and runtime_error is None
        models_loaded = any(
            model is not None
            for model in (self._model, self._clip, self._video_vae, self._audio_vae)
        )
        return {
            "status": "ready" if ready else "not_ready",
            "backend": "native",
            "runtime": str(runtime),
            "runtime_commit": COMFY_CORE_COMMIT,
            "model_root": str(self.settings.model_root.expanduser().resolve()),
            "missing_runtime": missing_runtime,
            "missing_models": missing_models,
            "missing_optional_models": missing_optional_models,
            "devices": devices,
            "memory": {
                "eager_chunk_rows": self.settings.eager_chunk_rows,
                "mlp_chunk_rows": self.settings.mlp_chunk_rows,
                "eager_memory_fraction": self.settings.eager_memory_fraction,
                "mode": self.settings.vram_mode,
                "reserve_vram_gb": self.settings.reserve_vram_gb,
                "dynamic_enabled": bool(
                    self._core is not None and self._core.dynamic_vram_enabled
                ),
            },
            "acceleration_lora": {
                "enabled": self.settings.enable_acceleration_lora,
                "fl2va_steps": ACCELERATION_STEPS[GenerationMode.FL2V],
                "ref2va_steps": ACCELERATION_STEPS[GenerationMode.REF2V],
            },
            "reason": runtime_error,
            "loaded": self._core is not None,
            "models_loaded": models_loaded,
        }

    def _import_core(self) -> SimpleNamespace:
        if self._core is not None:
            return self._core
        status = self.readiness()
        if status["status"] != "ready":
            raise RuntimeError(f"local H3 runtime is not ready: {status}")

        runtime = str(self.settings.runtime_path.expanduser().resolve())
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        arguments = self._core_arguments()
        original_argv = sys.argv
        options = importlib.import_module("comfy.options")
        options.enable_args_parsing()
        sys.argv = [original_argv[0], *arguments]
        try:
            torch = importlib.import_module("torch")
            device = self.settings.device.lower()
            if device.startswith("cuda:"):
                torch.cuda.set_device(int(device.split(":", 1)[1]))

            cli_args = importlib.import_module("comfy.cli_args")
            dynamic_requested = self.settings.vram_mode == "dynamic" and device != "cpu"
            aimdo_control = None
            if dynamic_requested:
                aimdo_control = importlib.import_module("comfy_aimdo.control")
                headroom = int(self.settings.reserve_vram_gb * 1024**3)
                if not aimdo_control.init(simple_vram_headroom=headroom, nvml_pressure=True):
                    raise RuntimeError("comfy-aimdo could not initialize DynamicVRAM")

            model_management = importlib.import_module("comfy.model_management")
            memory_management = importlib.import_module("comfy.memory_management")
            model_patcher = importlib.import_module("comfy.model_patcher")
            dynamic_enabled = False
            if dynamic_requested:
                selected = model_management.get_torch_device()
                if selected.type != "cuda" or not aimdo_control.init_devices([selected.index]):
                    aimdo_control.deinit()
                    raise RuntimeError("comfy-aimdo could not initialize the selected CUDA device")
                model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
                memory_management.aimdo_enabled = True
                dynamic_enabled = True

            self._core = SimpleNamespace(
                torch=torch,
                cli_args=cli_args,
                sd=importlib.import_module("comfy.sd"),
                utils=importlib.import_module("comfy.utils"),
                model_management=model_management,
                memory_management=memory_management,
                model_patcher=model_patcher,
                aimdo_control=aimdo_control,
                dynamic_vram_enabled=dynamic_enabled,
                samplers=importlib.import_module("comfy.samplers"),
                sample=importlib.import_module("comfy.sample"),
                nested_tensor=importlib.import_module("comfy.nested_tensor"),
            )
        finally:
            sys.argv = original_argv
            options.enable_args_parsing(False)
        return self._core

    def _load_vae(self, path: Path) -> Any:
        core = self._import_core()
        state_dict, metadata = core.utils.load_torch_file(str(path), return_metadata=True)
        vae = core.sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        vae.patcher.cached_patcher_init = (core.sd.load_vae_patcher, (str(path), metadata, None))
        return vae

    def _load_common_models(self) -> None:
        core = self._import_core()
        paths = self.settings.model_paths
        if self._clip is None:
            self._clip = core.sd.load_clip(
                [str(paths["text_encoder"])],
                clip_type=core.sd.CLIPType.MINIMAX,
            )
        if self._video_vae is None:
            self._video_vae = self._load_vae(paths["video_vae"])
        if self._audio_vae is None:
            self._audio_vae = self._load_vae(paths["audio_vae"])

    def _load_diffusion_model(self, mode: GenerationMode) -> Any:
        core = self._import_core()
        wanted = GenerationMode.REF2V if mode == GenerationMode.REF2V else GenerationMode.FL2V
        if self._model is not None and self._model_mode == wanted:
            return self._model
        if self._model is not None:
            core.model_management.unload_all_models()
            self._model = None
            gc.collect()
            core.model_management.soft_empty_cache()
        path = self.settings.model_paths["ref2va" if wanted == GenerationMode.REF2V else "fl2va"]
        model = core.sd.load_diffusion_model(str(path), model_options={})
        if self.settings.enable_acceleration_lora:
            lora_key = (
                "ref2va_acceleration_lora"
                if wanted == GenerationMode.REF2V
                else "fl2va_acceleration_lora"
            )
            lora, metadata = core.utils.load_torch_file(
                str(self.settings.model_paths[lora_key]),
                safe_load=True,
                return_metadata=True,
            )
            model, _ = core.sd.load_lora_for_models(
                model,
                None,
                lora,
                1.0,
                0.0,
                lora_metadata=metadata,
            )
        self._model = model
        self._model_mode = wanted
        return self._model

    def _sampling_options(self, plan: GenerationPlan) -> tuple[int, str, str]:
        request = plan.request
        if not self.settings.enable_acceleration_lora:
            return request.steps, request.sampler_name, request.scheduler
        return (
            ACCELERATION_STEPS[request.mode],
            ACCELERATION_SAMPLER,
            ACCELERATION_SCHEDULER,
        )

    def _empty_latent(self, plan: GenerationPlan) -> dict[str, Any]:
        core = self._import_core()
        request = plan.request
        duration = plan.frame_count / FPS
        video = core.torch.zeros(
            [1, 24, _video_latent_t(plan.frame_count), request.height // 16, request.width // 16],
            device=core.model_management.intermediate_device(),
        )
        audio = core.torch.zeros(
            [1, 32, 2, round(duration * AUDIO_LATENT_FPS)],
            device=core.model_management.intermediate_device(),
        )
        return {"samples": core.nested_tensor.NestedTensor((video, audio))}

    def _resize(self, image: Any, width: int, height: int, crop: str = "disabled") -> Any:
        core = self._import_core()
        samples = image[..., :3].movedim(-1, 1)
        samples = core.utils.common_upscale(samples, width, height, "lanczos", crop)
        return samples.movedim(1, -1)

    def _build_image_conditioning(self, plan: GenerationPlan) -> tuple[list, dict[str, Any]]:
        request = plan.request
        latent = self._empty_latent(plan)
        images = []
        keyframes = []
        if request.first_frame:
            image = self._resize(load_image(self.resolve_input(request.first_frame)), request.width, request.height)
            images.append(image)
            keyframes.append({"resolved_frame_index": 0, "image": image})
        if request.last_frame:
            image = self._resize(
                load_image(self.resolve_input(request.last_frame)),
                request.width,
                request.height,
                "center",
            )
            images.append(image)
            keyframes.append({"resolved_frame_index": plan.frame_count - 1, "image": image})

        tokens = self._clip.tokenize(request.prompt, images=images)
        conditioning = self._clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for keyframe in keyframes:
                keyframe["latent"] = self._video_vae.encode(keyframe.pop("image"))
            conditioning = _conditioning_set_values(
                conditioning,
                {"minimax_keyframes": keyframes, "minimax_frame_count": plan.frame_count},
            )
        return conditioning, latent

    def _encode_ref_audio(self, audio: dict[str, Any]) -> tuple[Any, int]:
        core = self._import_core()
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        vae_sample_rate = int(getattr(self._audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_sample_rate:
            torchaudio = importlib.import_module("torchaudio")
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sample_rate)
        latent = self._audio_vae.encode(waveform[:1].movedim(1, -1))
        return latent, int(latent.shape[-1])

    def _reference_video(
        self,
        plan: GenerationPlan,
        reference: ReferenceVideo,
    ) -> tuple[Any, dict[str, Any] | None]:
        path = self.resolve_input(reference.file_id)
        source_width, source_height = video_dimensions(path)
        width, height = _adapt_canvas(source_width, source_height)
        if source_width * source_height < width * height:
            width = max(CANVAS_MULTIPLE, round(source_width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            height = max(CANVAS_MULTIPLE, round(source_height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = load_video_frames(
            path,
            width=width,
            height=height,
            max_frames=plan.frame_count,
            fps=FPS,
        )
        frame_count = int(frames.shape[0])
        if frame_count < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames")
        frame_count -= (frame_count - 5) % 17
        frames = frames[:frame_count]
        soundtrack = None
        if reference.audio_file_id:
            soundtrack = load_audio(self.resolve_input(reference.audio_file_id))
        elif reference.include_source_audio:
            soundtrack = load_audio(path, optional=True)
        return frames, soundtrack

    def _build_reference_conditioning(self, plan: GenerationPlan) -> tuple[list, dict[str, Any]]:
        request = plan.request
        latent = self._empty_latent(plan)
        ref_items: list[dict[str, Any]] = []
        ref_blocks: list[dict[str, Any]] = []

        for file_id in request.reference_images:
            image = load_image(self.resolve_input(file_id))
            height, width = int(image.shape[1]), int(image.shape[2])
            if request.ref_image_size == "match":
                scale = min(1.0, math.sqrt((request.width * request.height) / (width * height)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(width, height))
            target_width = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            target_height = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = self._resize(image, target_width, target_height)
            encoded = self._video_vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append(
                {
                    "kind": "image",
                    "latent_h": target_height // 16,
                    "latent_w": target_width // 16,
                    "latent": encoded,
                }
            )

        for reference in request.reference_videos:
            frames, soundtrack = self._reference_video(plan, reference)
            encoded = self._video_vae.encode(frames)
            audio_latent, ref_audio_t = None, 0
            if soundtrack is not None:
                audio_latent, ref_audio_t = self._encode_ref_audio(soundtrack)
                ref_items.append({"type": "audio"})
            sample_indices = list(range(0, int(frames.shape[0]), FPS // 2))
            ref_items.append(
                {
                    "type": "video",
                    "data": frames[sample_indices],
                    "timestamps": [index / 2.0 for index in range(len(sample_indices))],
                }
            )
            ref_blocks.append(
                {
                    "kind": "video_audio" if ref_audio_t else "video",
                    "latent_t": int(encoded.shape[2]),
                    "latent_h": int(encoded.shape[3]),
                    "latent_w": int(encoded.shape[4]),
                    "ref_audio_t": ref_audio_t,
                    "latent": encoded,
                    "audio_latent": audio_latent,
                }
            )

        for file_id in request.reference_audios:
            audio = load_audio(self.resolve_input(file_id))
            if audio is None:
                continue
            audio_latent, ref_audio_t = self._encode_ref_audio(audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append(
                {"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent}
            )

        tokens = self._clip.tokenize(request.prompt, minimax_ref_items=ref_items)
        conditioning = self._clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            conditioning = _conditioning_set_values(conditioning, {"minimax_refs": ref_blocks})
        return conditioning, latent

    def _sample(
        self,
        plan: GenerationPlan,
        model: Any,
        conditioning: list,
        latent: dict[str, Any],
        cancel: threading.Event,
        progress: Callable[[float], None],
    ) -> Any:
        core = self._import_core()
        request = plan.request
        steps, sampler_name, scheduler = self._sampling_options(plan)
        latent_image = core.sample.fix_empty_latent_channels(model, latent["samples"])
        noise = core.sample.prepare_noise(latent_image, plan.seed, None)
        sampler = core.samplers.sampler_object(sampler_name)
        total_steps = steps if request.denoise >= 1.0 else int(steps / request.denoise)
        sigma_steps = total_steps
        discard_penultimate = sampler_name in DISCARD_PENULTIMATE_SIGMA_SAMPLERS
        if discard_penultimate:
            sigma_steps += 1
        sigmas = core.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, sigma_steps
        )
        if discard_penultimate:
            sigmas = core.torch.cat([sigmas[:-2], sigmas[-1:]])
        sigmas = sigmas.cpu()[-(steps + 1) :]
        guider = core.samplers.CFGGuider(model)
        guider.inner_set_conds({"positive": conditioning})

        def callback(step: int, _x0: Any, _x: Any, total: int) -> None:
            if cancel.is_set():
                raise GenerationCanceled("generation canceled")
            progress(0.2 + 0.65 * ((step + 1) / max(total, 1)))

        return guider.sample(
            noise,
            latent_image,
            sampler,
            sigmas,
            callback=callback,
            disable_pbar=True,
            seed=plan.seed,
        ).to(core.model_management.intermediate_device())

    def generate(
        self,
        plan: GenerationPlan,
        output: OutputArtifact,
        cancel: threading.Event,
        progress: Callable[[float], None],
    ) -> None:
        core = self._import_core()
        failed = False
        logging.getLogger(__name__).info(
            "H3 request: mode=%s size=%sx%s frames=%s requested_seconds=%s seed=%s refs=%s/%s/%s",
            plan.request.mode, plan.request.width, plan.request.height, plan.frame_count,
            plan.request.duration_seconds, plan.seed, len(plan.request.reference_images),
            len(plan.request.reference_videos), len(plan.request.reference_audios),
        )

        def check_canceled() -> None:
            if cancel.is_set():
                raise GenerationCanceled("generation canceled")

        try:
            check_canceled()
            with core.torch.inference_mode():
                self._load_common_models()
                model = self._load_diffusion_model(plan.request.mode)
                progress(0.1)
                if plan.request.mode == GenerationMode.REF2V:
                    conditioning, latent = self._build_reference_conditioning(plan)
                else:
                    conditioning, latent = self._build_image_conditioning(plan)
                progress(0.2)
                with h3_memory_scope(
                    core.torch, core.model_management.get_torch_device(), self.settings,
                    model.model.diffusion_model, check_canceled,
                ):
                    samples = self._sample(plan, model, conditioning, latent, cancel, progress)
                    check_canceled()

                    video_latent, audio_latent = samples.unbind()
                    images = self._video_vae.decode(video_latent)
                    check_canceled()
                    if images.ndim == 5:
                        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    progress(0.9)
                    waveform = self._audio_vae.decode(audio_latent).movedim(-1, 1)
                    check_canceled()
                    standard_deviation = core.torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
                    standard_deviation[standard_deviation < 1.0] = 1.0
                    waveform /= standard_deviation
                    audio = {
                        "waveform": waveform,
                        "sample_rate": int(getattr(self._audio_vae, "audio_sample_rate", 32000)),
                    }
                    save_mp4(
                        output.path,
                        images,
                        audio,
                        fps=FPS,
                        check_canceled=check_canceled,
                    )
                    progress(0.99)
        except core.torch.OutOfMemoryError as exc:
            failed = True
            raise GenerationMemoryError(
                f"生成超出当前显存容量（{plan.request.width}×{plan.request.height}，"
                f"{plan.frame_count} 帧）。已停止任务；请缩短视频、降低分辨率、"
                "减少参考素材或关闭其他 GPU 程序后重试。未自动降低输出规格。"
            ) from exc
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                core.model_management.unload_all_models()
            core.model_management.soft_empty_cache()

    def release_models(self) -> None:
        if self._core is None:
            return
        core = self._core
        core.model_management.unload_all_models()
        self._model = None
        self._model_mode = None
        self._clip = None
        self._video_vae = None
        self._audio_vae = None
        gc.collect()
        core.model_management.soft_empty_cache()

    def close(self) -> None:
        if self._core is None:
            return
        core = self._core
        self.release_models()
        if core.dynamic_vram_enabled:
            core.memory_management.aimdo_enabled = False
            core.aimdo_control.deinit()
