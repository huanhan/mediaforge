"""Standalone SAM 2 + official ProPainter adapter (no H3 process required)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def validate_masks(directory: Path, count: int, size: tuple[int, int]) -> list[Path]:
    from PIL import Image

    paths = sorted(directory.glob("*.png"))
    if len(paths) != count:
        raise ValueError(f"Expected {count} PNG masks, found {len(paths)}")
    for path in paths:
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"Mask {path.name} must have size {size}")
    return paths


def extract_frames(video: Path, directory: Path, max_side: int, max_frames: int, check=lambda: None):
    import av

    directory.mkdir()
    timestamps = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = stream.average_rate or stream.base_rate
        if not fps or fps <= 0:
            raise ValueError("Cannot determine source frame rate")
        ratio = min(1.0, max_side / max(stream.width, stream.height))
        size = tuple(max(8, int(v * ratio) // 8 * 8) for v in (stream.width, stream.height))
        for index, frame in enumerate(container.decode(stream)):
            check()
            if index >= max_frames:
                raise ValueError(f"Video exceeds {max_frames} frames; trim it first or raise --max-frames")
            timestamps.append(float(frame.time) if frame.time is not None else index / float(fps))
            frame.to_image().resize(size).save(directory / f"{index:06d}.jpg", quality=98)
    if len(timestamps) < 2:
        raise ValueError("At least two video frames are required")
    # Do not silently change the timing of variable-frame-rate inputs.
    period = 1 / float(fps)
    if any(abs((b - a) - period) > max(.002, period * .1)
           for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Variable frame rate detected; convert to constant frame rate first")
    return len(timestamps), size, fps


def segment(frames: Path, masks: Path, count: int, size: tuple[int, int],
            checkpoint: Path, config: str, box: list[float],
            check=lambda: None, progress=lambda value: None):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 video removal requires a CUDA GPU")
    predictor = build_sam2_video_predictor(config, str(checkpoint), device="cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        check()
        state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        coords = np.asarray(box, dtype=np.float32) * np.asarray([*size, *size])
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0,
                                         obj_id=1, box=coords)
        seen = set()
        for index, ids, logits in predictor.propagate_in_video(state):
            check()
            if 1 not in ids:
                raise RuntimeError(f"Selected object missing at frame {index}")
            mask = (logits[list(ids).index(1), 0] > 0).cpu().numpy()
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks / f"{index:06d}.png")
            seen.add(index)
            progress(len(seen) / count)
        if seen != set(range(count)):
            raise RuntimeError("SAM 2 did not produce a mask for every frame")
    del state, predictor
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def exclusive_masks(scores, active):
    """Give overlapping visible pixels to the active object with the highest score."""
    import numpy as np
    scores = np.asarray(scores, dtype=np.float32).copy()
    scores[~np.asarray(active, dtype=bool)] = -np.inf
    winner = scores.argmax(axis=0)
    valid = scores.max(axis=0) > 0
    return np.asarray([(winner == i) & valid for i in range(len(scores))])


def segment_objects(frames, union_dir, object_dirs, count, size, checkpoint, config,
                    objects, check=lambda: None, progress=lambda value: None):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    if any(obj.frame_index >= count for obj in objects):
        raise ValueError(f"目标参考帧超出视频范围（共 {count} 帧）。")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2 video removal requires a CUDA GPU")
    predictor = build_sam2_video_predictor(config, str(checkpoint), device="cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        check()
        state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        for index, obj in enumerate(objects):
            check()
            coords = np.asarray(obj.box, dtype=np.float32) * np.asarray([*size, *size])
            predictor.add_new_points_or_box(inference_state=state, frame_idx=obj.frame_index,
                                             obj_id=index + 1, box=coords)
        start = min(obj.frame_index for obj in objects)
        empty = Image.new("L", size)
        for i in range(start):
            check()
            for folder in [union_dir, *object_dirs]:
                empty.save(folder / f"{i:06d}.png")
        seen = set(range(start))
        for i, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start):
            check()
            if set(ids) != set(range(1, len(objects) + 1)):
                raise RuntimeError("SAM 2 returned an incomplete object set")
            scores = logits[:, 0].float().cpu().numpy()
            scores = scores[[list(ids).index(j + 1) for j in range(len(objects))]]
            separated = exclusive_masks(scores, [i >= obj.frame_index for obj in objects])
            for folder, mask in zip(object_dirs, separated):
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{i:06d}.png")
            Image.fromarray(separated.any(axis=0).astype(np.uint8) * 255).save(union_dir / f"{i:06d}.png")
            seen.add(i)
            progress(len(seen) / count)
        if seen != set(range(count)):
            raise RuntimeError("SAM 2 did not produce a mask for every frame")
    del state, predictor
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def propainter_command(python: Path, repo: Path, frames: Path, masks: Path,
                       output: Path, dilation: int) -> list[str]:
    return [str(python), str(repo / "inference_propainter.py"),
            "--video", str(frames), "--mask", str(masks), "--output", str(output),
            "--mask_dilation", str(dilation), "--neighbor_length", "10",
            "--subvideo_length", "40", "--save_frames", "--fp16"]


def run(args):
    from PIL import Image

    source = args.video.resolve()
    repo = args.propainter.resolve()
    python = args.propainter_python.resolve()
    for path in (source, python, repo / "inference_propainter.py"):
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
    if not args.masks and (not args.sam2_checkpoint or not args.sam2_checkpoint.is_file()):
        raise ValueError("Provide --sam2-checkpoint for box selection")
    if args.box and not (0 <= args.box[0] < args.box[2] <= 1 and
                         0 <= args.box[1] < args.box[3] <= 1):
        raise ValueError("Box must be normalized x1 y1 x2 y2 within 0..1")
    if args.max_side < 64 or args.max_frames < 2 or args.dilation < 0:
        raise ValueError("Invalid size, frame limit, or dilation")
    subprocess.run([args.ffmpeg, "-version"], check=True, capture_output=True)
    work = args.output.resolve()
    # A fresh job directory prevents stale masks/results and overwriting user media.
    work.mkdir(parents=True, exist_ok=False)
    frames, masks = work / "source_frames", work / "masks"
    count, size, fps = extract_frames(source, frames, args.max_side, args.max_frames)
    masks.mkdir()
    if args.masks:
        paths = validate_masks(args.masks, count, size)
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                image.convert("L").point(lambda v: 255 if v >= 128 else 0).save(
                    masks / f"{index:06d}.png")
    else:
        segment(frames, masks, count, size, args.sam2_checkpoint.resolve(),
                args.sam2_config, args.box)
    validate_masks(masks, count, size)
    # Also export the selected foreground. SAM 2 produces hard masks, not hair matting.
    foreground = work / "foreground"
    foreground.mkdir()
    for index in range(count):
        with Image.open(frames / f"{index:06d}.jpg") as frame, Image.open(masks / f"{index:06d}.png") as mask:
            rgba = frame.convert("RGBA")
            rgba.putalpha(mask.convert("L"))
            rgba.save(foreground / f"{index:06d}.png")
    metadata = {"source": str(source), "frames": count, "size": size,
                "fps": str(fps), "mask_convention": "white = remove", "status": "masks_ready"}
    manifest = work / "job.json"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    subprocess.run(propainter_command(python, repo, frames, masks, work / "repair", args.dilation),
                   cwd=repo, check=True)
    repaired = work / "repair" / frames.name / "frames"
    validate_masks(repaired, count, size)
    # Encode saved frames ourselves: upstream --save_fps is integer-only.
    target = work / "background.mp4"
    subprocess.run([args.ffmpeg, "-nostdin", "-n", "-v", "error", "-framerate", str(fps),
                    "-start_number", "0", "-i", str(repaired / "%04d.png"),
                    "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-t", str(count / float(fps)), str(target)], check=True)
    import av
    with av.open(str(target)) as container:
        if sum(1 for _ in container.decode(video=0)) != count:
            raise RuntimeError("Output frame count does not match source")
    metadata["status"] = "complete"
    manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New job directory")
    parser.add_argument("--propainter", type=Path, required=True, help="Official ProPainter checkout")
    parser.add_argument("--propainter-python", type=Path, default=Path(sys.executable))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                           help="Normalized box on first frame, e.g. .2 .1 .7 .95")
    selection.add_argument("--masks", type=Path, help="One PNG per frame, sorted by zero-padded filename")
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--dilation", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Video removal failed: {exc}\n")


if __name__ == "__main__":
    main()
