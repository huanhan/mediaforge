"""Private subprocess entry point for bounded local depth inference."""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
import sys


def read_frames(source: Path, max_resolution: int, max_frames: int):
    import av
    import numpy as np

    frames = []
    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError("Input contains no video stream")
        stream = container.streams.video[0]
        fps = Fraction(stream.average_rate or stream.base_rate or 24)
        if fps <= 0:
            raise ValueError("Input has invalid frame rate")
        width, height = stream.codec_context.width, stream.codec_context.height
        if min(width, height) < 2:
            raise ValueError("Input has invalid dimensions")
        scale = min(1.0, max_resolution / max(width, height))
        width, height = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
        for frame in container.decode(stream):
            if len(frames) >= max_frames:
                raise ValueError(f"Video exceeds H3_DEPTH_MAX_FRAMES={max_frames}; split the video or raise the configured limit")
            frames.append(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
    if not frames:
        raise ValueError("Input contains no decodable frames")
    return np.stack(frames), fps


def write_depths(depths, fps, output: Path, save_raw: bool):
    import av
    import numpy as np

    depths = np.asarray(depths, dtype=np.float32)
    if depths.ndim != 3 or not depths.size or not np.isfinite(depths).all():
        raise ValueError("Model returned invalid depth values")
    # One range per clip preserves the upstream model's temporal alignment.
    low, high = float(depths.min()), float(depths.max())
    span = max(high - low, 1e-8)
    partial = output.with_suffix(".partial.mp4")
    raw = output.with_suffix(".npz")
    raw_partial = raw.with_suffix(".partial.npz")
    try:
        with av.open(str(partial), "w") as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = depths.shape[2], depths.shape[1]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for index, depth in enumerate(depths):
                gray = np.clip((depth - low) / span * 255, 0, 255).astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(gray, format="gray")
                frame.pts, frame.time_base = index, 1 / fps
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if save_raw:
            np.savez_compressed(raw_partial, depths=depths, fps=float(fps),
                                fps_numerator=fps.numerator, fps_denominator=fps.denominator,
                                depth_type="relative_inverse_depth", visualization_min=low, visualization_max=high)
            raw_partial.replace(raw)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
        raw_partial.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "runtime", "model"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, default=392)
    parser.add_argument("--max-resolution", type=int, default=768)
    parser.add_argument("--max-frames", type=int, default=720)
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args(argv)
    # The upstream package imports a top-level `utils`; isolate it in this worker.
    sys.path.insert(0, str(args.runtime.resolve()))
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    frames, fps = read_frames(args.source, args.max_resolution, args.max_frames)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA PyTorch build or set H3_DEVICE=cpu")
        torch.cuda.set_device(device.index if device.index is not None else 0)
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    # Upstream uses device for both Tensor.to() and autocast(device_type=...),
    # so select the CUDA index above and pass the bare device type here.
    depths, _ = model.infer_video_depth(frames, float(fps), input_size=args.input_size,
                                        device=device.type, fp32=device.type == "cpu")
    if len(depths) != len(frames):
        raise ValueError("Model returned an incorrect frame count")
    write_depths(depths, fps, args.output, args.save_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
