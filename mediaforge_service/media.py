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
        oriented = ImageOps.exif_transpose(image)
        rgba = oriented.convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def video_dimensions(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path.name}")
        stream = container.streams.video[0]
        return int(stream.codec_context.width), int(stream.codec_context.height)


def _video_frame_rgb(frame: Any, *, width: int | None = None, height: int | None = None) -> Any:
    """Flatten alpha before resizing so hidden source colors cannot reach the model."""
    if any(component.is_alpha for component in frame.format.components):
        import av
        import numpy as np
        from PIL import Image

        rgba = Image.fromarray(frame.to_ndarray(format="rgba"))
        rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
        frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
    return frame.reformat(width=width, height=height, format="rgb24").to_ndarray()


def load_video(path: Path) -> tuple[Any, Fraction]:
    import av
    import numpy as np
    import torch

    frames: list[Any] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path.name}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(24, 1)
        fps = Fraction(rate)
        for frame in container.decode(stream):
            frames.append(_video_frame_rgb(frame))
    if not frames:
        raise ValueError(f"video contains no decodable frames: {path.name}")
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array), fps


def load_video_frames(
    path: Path,
    *,
    width: int,
    height: int,
    max_frames: int,
    fps: int = 24,
    start_seconds: float = 0.0,
    check_canceled: Callable[[], None] | None = None,
) -> Any:
    import av
    import numpy as np
    import torch

    selected: list[Any] = []
    next_timestamp = start_seconds
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"video has no video stream: {path.name}")
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        if start_seconds > 0:
            container.seek(int((start_seconds + origin) / stream.time_base), stream=stream, backward=True)
        guessed_fps = float(stream.average_rate or stream.base_rate or fps)
        frame_interval = 1.0 / max(guessed_fps, 1.0)
        for decoded_index, frame in enumerate(container.decode(stream)):
            if check_canceled:
                check_canceled()
            timestamp = float(frame.time) - origin if frame.time is not None else decoded_index * frame_interval
            while timestamp + frame_interval / 2 >= next_timestamp and len(selected) < max_frames:
                selected.append(_video_frame_rgb(frame, width=width, height=height))
                next_timestamp += 1.0 / fps
            if len(selected) >= max_frames:
                break
    if not selected:
        raise ValueError(f"video contains no decodable frames: {path.name}")
    array = np.stack(selected).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def video_duration(path: Path) -> float:
    """Read video duration, falling back to timestamp scanning without storing frames."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("source_video has no video stream")
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        origin = float((stream.start_time or 0) * stream.time_base)
        rate = float(stream.average_rate or stream.base_rate or 24)
        end = 0.0
        for i, frame in enumerate(container.decode(stream)):
            end = (float(frame.time) - origin if frame.time is not None else i / rate) + 1 / rate
        if end <= 0:
            raise ValueError("source_video has no decodable frames")
        return end


def load_audio_window(path: Path, start_seconds: float, duration_seconds: float,
                      *, sample_rate: int = 32000,
                      check_canceled: Callable[[], None] | None = None) -> dict[str, Any] | None:
    """Decode a bounded interval on the video timeline, retaining silence and AV offsets."""
    import av
    import numpy as np
    import torch

    count = round(duration_seconds * sample_rate)
    output = np.zeros((2, count), dtype=np.float32)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        timeline = container.streams.video[0] if container.streams.video else stream
        origin = float((timeline.start_time or 0) * timeline.time_base)
        begin = origin + start_seconds
        container.seek(int(max(0, begin - 0.25) * av.time_base), backward=True)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        fallback_time = begin

        def consume(frame):
            nonlocal fallback_time
            timestamp = float(frame.time) if frame.time is not None else fallback_time
            array = frame.to_ndarray()
            offset = round((timestamp - begin) * sample_rate)
            left, right = max(0, offset), min(count, offset + array.shape[-1])
            if right > left:
                output[:, left:right] = array[:, left - offset:right - offset]
            fallback_time = timestamp + array.shape[-1] / sample_rate

        for frame in container.decode(stream):
            if check_canceled:
                check_canceled()
            if frame.time is not None and float(frame.time) > begin + duration_seconds + 0.1:
                break
            for converted in resampler.resample(frame):
                consume(converted)
        for converted in resampler.resample(None):
            consume(converted)
    return {"waveform": torch.from_numpy(output).unsqueeze(0), "sample_rate": sample_rate}


class SegmentedMP4Writer:
    """Encode successive batches with one codec session and a continuous timestamp clock."""

    def __init__(self, path: Path, width: int, height: int, *, with_audio: bool,
                 fps: int = 24, sample_rate: int = 32000):
        import av
        self.path = path
        self.temporary = path.with_name(path.stem + ".partial.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(self.temporary), "w")
        self.video = self.container.add_stream("libx264", rate=fps)
        self.video.width, self.video.height = width, height
        self.video.pix_fmt = "yuv420p"
        self.video.options = {"crf": "20", "preset": "medium"}
        self.audio = self.container.add_stream("aac", rate=sample_rate) if with_audio else None
        if self.audio:
            self.audio.layout = "stereo"
            self.audio.bit_rate = 192_000
        self.fps, self.sample_rate = fps, sample_rate
        self.frames = self.samples = 0

    def append(self, images: Any, audio: dict[str, Any] | None,
               check_canceled: Callable[[], None]) -> None:
        import av
        import numpy as np
        for image in images:
            check_canceled()
            array = image[..., :3].detach().float().clamp(0, 1).mul(255).byte().cpu().numpy()
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
            for packet in self.video.encode(frame):
                self.container.mux(packet)
            self.frames += 1
        if self.audio is not None:
            end = round(self.frames * self.sample_rate / self.fps)
            required = end - self.samples
            waveform = np.zeros((2, required), dtype=np.float32)
            if audio is not None:
                if audio["sample_rate"] != self.sample_rate:
                    raise ValueError("segment audio sample rate mismatch")
                data = audio["waveform"][0].detach().float().cpu().clamp(-1, 1).numpy()
                if data.shape[0] == 1:
                    data = np.repeat(data, 2, axis=0)
                count = min(required, data.shape[-1])
                waveform[:, :count] = data[:2, :count]
            for offset in range(0, required, 1024):
                check_canceled()
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform[:, offset:offset + 1024]),
                                                  format="fltp", layout="stereo")
                frame.sample_rate = self.sample_rate
                frame.pts, frame.time_base = self.samples + offset, Fraction(1, self.sample_rate)
                for packet in self.audio.encode(frame):
                    self.container.mux(packet)
            self.samples = end

    def finish(self, check_canceled: Callable[[], None]) -> None:
        for stream in (self.video, self.audio):
            if stream is not None:
                for packet in stream.encode():
                    check_canceled()
                    self.container.mux(packet)
        self.container.close()
        check_canceled()
        self.temporary.replace(self.path)

    def close(self) -> None:
        self.container.close()
        self.temporary.unlink(missing_ok=True)


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
    audio: dict[str, Any] | None,
    *,
    fps: int | Fraction = 24,
    check_canceled: Callable[[], None] | None = None,
) -> None:
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    waveform = None
    sample_rate = 0
    required_samples = 0
    if audio is not None:
        waveform = audio["waveform"][0].detach().float().cpu()
        sample_rate = int(audio["sample_rate"])
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        required_samples = round(frame_count / float(fps) * sample_rate)
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

        audio_stream = None
        if waveform is not None:
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

        if audio_stream is not None and waveform is not None:
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
