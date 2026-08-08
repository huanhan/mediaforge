from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Callable


def validate_mp4_encoders() -> None:
    import av

    av.Codec("libx264", "w")
    av.Codec("aac", "w")


def load_image(path: Path) -> Any:
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def video_dimensions(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path.name}")
        stream = container.streams.video[0]
        return int(stream.codec_context.width), int(stream.codec_context.height)


def load_video_frames(
    path: Path,
    *,
    width: int,
    height: int,
    max_frames: int,
    fps: int = 24,
) -> Any:
    import av
    import numpy as np
    import torch

    selected: list[Any] = []
    next_timestamp = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path.name}")
        stream = container.streams.video[0]
        guessed_fps = float(stream.average_rate or stream.base_rate or fps)
        frame_interval = 1.0 / max(guessed_fps, 1.0)
        for decoded_index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else decoded_index * frame_interval
            while timestamp + frame_interval / 2 >= next_timestamp and len(selected) < max_frames:
                resized = frame.reformat(width=width, height=height, format="rgb24")
                selected.append(resized.to_ndarray())
                next_timestamp += 1.0 / fps
            if len(selected) >= max_frames:
                break
    if not selected:
        raise ValueError(f"video contains no decodable frames: {path.name}")
    array = np.stack(selected).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def load_audio(path: Path, *, sample_rate: int = 32000, optional: bool = False) -> dict[str, Any] | None:
    import av
    import numpy as np
    import torch

    chunks: list[Any] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            if optional:
                return None
            raise ValueError(f"file has no audio stream: {path.name}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray())
    if not chunks:
        if optional:
            return None
        raise ValueError(f"file contains no decodable audio: {path.name}")
    waveform = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sample_rate}


def save_mp4(
    path: Path,
    images: Any,
    audio: dict[str, Any],
    *,
    fps: int = 24,
    check_canceled: Callable[[], None] | None = None,
) -> None:
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    waveform = audio["waveform"][0].detach().float().cpu()
    sample_rate = int(audio["sample_rate"])
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    required_samples = round(frame_count / fps * sample_rate)
    if waveform.shape[1] < required_samples:
        padded = waveform.new_zeros((2, required_samples))
        padded[:, : waveform.shape[1]] = waveform
        waveform = padded
    waveform = waveform[:, :required_samples].clamp(-1.0, 1.0).numpy()

    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = "yuv420p"
        video_stream.options = {"crf": "20", "preset": "medium"}

        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "stereo"
        audio_stream.bit_rate = 192_000

        for image in images:
            if check_canceled is not None:
                check_canceled()
            array = (
                image[..., :3]
                .detach()
                .float()
                .clamp(0.0, 1.0)
                .mul(255)
                .byte()
                .cpu()
                .numpy()
            )
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        frame_size = audio_stream.codec_context.frame_size or 1024
        for offset in range(0, required_samples, frame_size):
            if check_canceled is not None:
                check_canceled()
            chunk = np.ascontiguousarray(waveform[:, offset : offset + frame_size])
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
            frame.sample_rate = sample_rate
            frame.pts = offset
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)
