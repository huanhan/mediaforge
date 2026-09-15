"""Video separation on the shared GPU worker; optional models stay lazily imported."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .backend import OutputArtifact
from .tasks import GenerationCanceled
from .video_remove import extract_frames, segment_objects, propainter_command, validate_masks


def run_command(command, cancel, directory=None):
    """Keep only a log tail on failure; cancellation reaps the direct child."""
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=log)
        try:
            while process.poll() is None:
                if cancel.wait(.2):
                    raise GenerationCanceled("task canceled")
            if process.returncode:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 4000))
                raise RuntimeError(log.read().decode("utf-8", errors="replace"))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def repair_windows(count, length=80, overlap=16):
    start = 0
    while start < count:
        end = min(start + length, count)
        yield start, end
        if end == count:
            break
        start = end - overlap


class VideoRemovalRunner:
    def __init__(self, settings, resolve_input):
        self.resolve_input = resolve_input
        self.repo = Path(os.getenv("H3_PROPAINTER_PATH", ".runtime/ProPainter")).resolve()
        self.checkpoint = Path(os.getenv("H3_SAM2_CHECKPOINT",
                                        str(settings.model_root / "sam2/sam2.1_hiera_tiny.pt"))).resolve()
        self.config = os.getenv("H3_SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")

    def readiness(self):
        missing = []
        for module in ("sam2", "hydra", "iopath", "av", "cv2", "torch"):
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        if not self.checkpoint.is_file():
            missing.append(str(self.checkpoint))
        if not shutil.which("ffmpeg"):
            missing.append("ffmpeg")
        repair_missing = []
        for name in ("inference_propainter.py", "weights/raft-things.pth",
                     "weights/recurrent_flow_completion.pth", "weights/ProPainter.pth"):
            if not (self.repo / name).is_file():
                repair_missing.append(str(self.repo / name))
        for module in ("imageio", "imageio_ffmpeg", "timm", "scipy"):
            if importlib.util.find_spec(module) is None:
                repair_missing.append(module)
        return {"status": "not_ready" if missing else "ready", "backend": "sam2-propainter",
                "foreground_available": not missing,
                "background_available": not missing and not repair_missing,
                "max_objects": 8, "object_reference_frames": True,
                "missing": missing, "repair_missing": repair_missing}

    def release_models(self):
        gc.collect()
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()

    def close(self):
        self.release_models()

    def process(self, request, output, cancel, progress):
        def check():
            if cancel.is_set():
                raise GenerationCanceled("task canceled")

        published = []
        try:
            with tempfile.TemporaryDirectory(prefix="removal-", dir=output.path.parent) as temp:
                work = Path(temp)
                source = self.resolve_input(request.file_id)
                paths = self._process(request, source, work, cancel, check, progress)
                for path in paths:
                    if path.name == "objects.json":
                        data = json.loads(path.read_text(encoding="utf-8"))
                        for obj in data["objects"]:
                            for key in ("preview", "alpha", "mask"):
                                if key in obj:
                                    obj[key] = f"{output.path.stem}-{obj[key]}"
                        for key in ("union_mask", "background"):
                            if data[key]:
                                data[key] = f"{output.path.stem}-{data[key]}"
                        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                for path in paths:
                    check()
                    target = output.path.with_name(f"{output.path.stem}-{path.name}")
                    shutil.copyfile(path, target)
                    published.append(OutputArtifact(target.name, output.subfolder, target))
                check()
                return published
        except BaseException:
            for artifact in published:
                artifact.path.unlink(missing_ok=True)
            raise
        finally:
            self.release_models()

    def _process(self, request, source, work, cancel, check, progress):
        import av

        frames, masks = work / "source", work / "masks"
        masks.mkdir()
        count, size, fps = extract_frames(source, frames, request.max_side, request.max_frames, check)
        progress(.05)
        objects = request.targets
        object_dirs = [work / f"object-{i}" / "masks" for i in range(len(objects))]
        for folder in object_dirs:
            folder.mkdir(parents=True)
        segment_objects(frames, masks, object_dirs, count, size, self.checkpoint, self.config,
                        objects, check, lambda p: progress(.05 + p * .35))
        validate_masks(masks, count, size)
        for folder in object_dirs:
            validate_masks(folder, count, size)
        self.release_models()
        foreground = request.output_mode in {"foreground", "both"}
        background = request.output_mode in {"background", "both"}
        results = []

        def encode(pattern, name, options, audio=True):
            check()
            target = work / name
            command = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-framerate", str(fps),
                       "-i", str(pattern)]
            if audio:
                command += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                            "-c:a", "libopus" if target.suffix == ".webm" else "aac"]
            command += [*options, "-t", str(count / float(fps)), str(target)]
            run_command(command, cancel)
            with av.open(str(target)) as container:
                if sum(1 for _ in container.decode(video=0)) != count:
                    raise RuntimeError("Output frame count does not match source")
            results.append(target)

        mp4 = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
        object_results = []
        for obj, selected_masks in zip(objects, object_dirs):
            prefix = f"object-{obj.id}-" if request.objects else ""
            files = {"id": obj.id, "frame_index": obj.frame_index, "box": obj.box}
            if foreground:
                self._export_foreground(source, size, selected_masks, check,
                                        request.foreground_background, request.background_color)
                if request.foreground_background == "transparent":
                    encode(selected_masks.parent / "rgba/%06d.png", prefix + "foreground-preview.webm",
                           ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                            "-b:v", "0", "-crf", "24"])
                else:
                    encode(selected_masks.parent / "preview/%06d.jpg", prefix + "foreground-preview.mp4", mp4)
                files["preview"] = results[-1].name
                encode(selected_masks.parent / "rgba/%06d.png", prefix + "foreground-alpha.mov",
                       ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"])
                files["alpha"] = results[-1].name
            if request.objects:
                encode(selected_masks / "%06d.png", prefix + "mask.mp4",
                       ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p"], False)
                files["mask"] = results[-1].name
            object_results.append(files)
        progress(.5)
        if background:
            self._repair_background(request, source, work, frames, masks, count, size,
                                    cancel, check, progress, encode, mp4)
        progress(.95)
        encode(masks / "%06d.png", "mask.mp4", ["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p"], False)
        manifest = work / "objects.json"
        manifest.write_text(json.dumps({"objects": object_results, "frames": count, "fps": str(fps),
            "size": size, "union_mask": "mask.mp4", "background": "background.mp4" if background else None,
            "foreground_background": request.foreground_background, "background_color": request.background_color,
            "before_reference_frame": "transparent", "overlap_policy": "highest_score_visible_pixel"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(manifest)
        return results

    @staticmethod
    def _export_foreground(source, size, masks, check, foreground_background="color", background_color="#00ff00"):
        import av
        from PIL import Image
        rgba_dir, preview = masks.parent / "rgba", masks.parent / "preview"
        rgba_dir.mkdir(); preview.mkdir()
        # Re-decode the source to avoid JPEG compression in exported foregrounds.
        with av.open(str(source)) as container:
            for i, decoded in enumerate(container.decode(video=0)):
                check()
                image = decoded.to_image().convert("RGB").resize(size)
                with Image.open(masks / f"{i:06d}.png") as mask:
                    rgba = image.convert("RGBA"); rgba.putalpha(mask)
                    rgba.save(rgba_dir / f"{i:06d}.png")
                    if foreground_background == "color":
                        canvas = Image.new("RGB", size, background_color)
                        canvas.paste(image, (0, 0), mask)
                        canvas.save(preview / f"{i:06d}.jpg", quality=95)

    def _repair_background(self, request, source, work, frames, masks, count, size,
                           cancel, check, progress, encode, mp4):
        import av
        import cv2
        import numpy as np
        from PIL import Image, ImageFilter
        ratio = min(1, request.repair_size / max(size))
        repair_size = tuple(max(8, int(v * ratio) // 8 * 8) for v in size)
        windows = list(repair_windows(count))
        chunks = []
        for j, (start, end) in enumerate(windows):
            check()
            chunk = work / f"chunk-{start}"
            inputs, selected = chunk / "frames", chunk / "masks"
            inputs.mkdir(parents=True); selected.mkdir()
            for local, index in enumerate(range(start, end)):
                shutil.copyfile(frames / f"{index:06d}.jpg", inputs / f"{local:06d}.jpg")
                shutil.copyfile(masks / f"{index:06d}.png", selected / f"{local:06d}.png")
            command = propainter_command(Path(sys.executable), self.repo, inputs, selected,
                                         chunk / "repair", request.dilation)
            command += ["--width", str(repair_size[0]), "--height", str(repair_size[1])]
            run_command(command, cancel, self.repo)
            repaired = chunk / "repair/frames/frames"
            validate_masks(repaired, end - start, repair_size)
            chunks.append((start, end, repaired))
            progress(.5 + .35 * (j + 1) / len(windows))
        composed = work / "background"
        composed.mkdir()
        with av.open(str(source)) as container:
            for i, decoded in enumerate(container.decode(video=0)):
                check()
                total, weight = None, 0
                for start, end, folder in chunks:
                    if start <= i < end:
                        w = min(i - start + 1, end - i, 16)
                        with Image.open(folder / f"{i-start:04d}.png") as image:
                            array = np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
                        total = array * w if total is None else total + array * w
                        weight += w
                repaired = Image.fromarray(np.uint8(np.clip(total / weight, 0, 255)))
                with Image.open(masks / f"{i:06d}.png") as mask:
                    # Match upstream cross-shaped dilation at its processing resolution.
                    small = np.asarray(mask.resize(repair_size, Image.Resampling.NEAREST))
                    if request.dilation:
                        small = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                                           iterations=request.dilation)
                    alpha = Image.fromarray(small).resize(size, Image.Resampling.NEAREST)
                    alpha = alpha.filter(ImageFilter.GaussianBlur(1))
                original = decoded.to_image().convert("RGB").resize(size)
                original.paste(repaired, (0, 0), alpha)
                original.save(composed / f"{i:06d}.png")
        encode(composed / "%06d.png", "background.mp4", mp4)
