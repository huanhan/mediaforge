import asyncio
import contextlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from minimaxh3_service.backend import BackendError, OutputArtifact
from minimaxh3_service.config import Settings
from minimaxh3_service.continuation import continue_latent
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.media import (SegmentedMP4Writer, load_audio_window, load_video,
                                     load_video_frames, save_mp4, video_duration)
from minimaxh3_service.models import GenerationRequest
from minimaxh3_service.native import NativeH3Runner, _video_latent_t
from minimaxh3_service.workflow import SegmentProgress, build_plan


@pytest.mark.parametrize("duration", [4, 4.5, 10, 10.01, 15, 60, 67.033008, 600])
@pytest.mark.parametrize("seconds,overlap", [(4, 5), (10, 48), (15, 345)])
def test_windows_cover_exact_timeline(duration, seconds, overlap):
    plan = build_plan(GenerationRequest(prompt="x", duration_seconds=duration,
        segmentation={"enabled": True, "segment_seconds": seconds, "overlap_frames": overlap}))
    emitted = 0
    for index, window in enumerate(plan.segments):
        assert window.frame_count % 17 == 5
        assert window.frame_count > window.trim_head
        assert window.output_frames > 0
        assert window.start_frame + window.trim_head == emitted
        assert window.output_frames <= window.frame_count - window.trim_head
        if index:
            previous = plan.segments[index - 1]
            assert window.start_frame == previous.start_frame + previous.frame_count - plan.overlap_frames
        emitted += window.output_frames
    assert emitted == plan.frame_count == round(duration * 24)


def test_last_frame_anchor_is_never_trimmed():
    plan = build_plan(GenerationRequest(prompt="x", mode="fl2v", last_frame="last.png",
        duration_seconds=31, segmentation={"enabled": True}))
    last = plan.segments[-1]
    assert last.trim_head + last.output_frames == last.frame_count
    assert plan.effective_duration_seconds >= 31


@pytest.mark.parametrize("kwargs", [
    {"duration_seconds": 60},
    {"mode": "edit"},
    {"source_video": {"file_id": "v.mp4"}},
    {"audio_mode": "preserve_source"},
    {"mode": "edit", "source_video": {"file_id": "../v.mp4"}},
    {"mode": "edit", "source_video": {"file_id": "v.mp4"}, "reference_videos": [{"file_id": "v2.mp4"}]},
    {"segmentation": {"enabled": True, "segment_seconds": 4, "overlap_frames": 200}},
    {"segmentation": {"enabled": True, "overlap_frames": 1}},
    {"mode": "edit", "duration_seconds": 4, "source_video": {"file_id": "v.mp4", "duration_seconds": 8}},
])
def test_invalid_edit_and_segment_requests(kwargs):
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="x", **kwargs)


class Nested:
    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors


def tensor_core():
    torch = pytest.importorskip("torch")
    return SimpleNamespace(torch=torch, nested_tensor=SimpleNamespace(NestedTensor=Nested))


def test_latent_continuation_keeps_predecessor_clean_and_new_region_free():
    core = tensor_core()
    torch = core.torch
    old = Nested((torch.rand(1, 24, 32, 2, 2), torch.rand(1, 32, 2, 180)))
    original = [x.clone() for x in old.unbind()]
    target = {"samples": Nested((torch.zeros(1, 24, 32, 2, 2), torch.zeros(1, 32, 2, 180)))}
    result = continue_latent(core, target, old, 39, include_audio=True)
    video, audio = result["samples"].unbind()
    vm, am = result["noise_mask"].unbind()
    assert torch.equal(video[:, :, :12], original[0][:, :, -12:])
    assert torch.equal(audio[..., :65], original[1][..., -65:])
    assert not vm[:, :, :12].any() and vm[:, :, 12:].all()
    assert not am[..., :65].any() and am[..., 65:].all()
    video.zero_()
    audio.zero_()
    assert all(torch.equal(a, b) for a, b in zip(old.unbind(), original))
    result = continue_latent(core, target, old, 39, include_audio=False)
    assert result["noise_mask"].unbind()[1].all()
    assert not result["samples"].unbind()[1].any()


def make_source(path, seconds=9):
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    images = torch.zeros((seconds * 24, 32, 32, 3))
    images[24 * 3:] = 0.75
    t = torch.arange(seconds * 32000) / 32000
    waveform = (0.2 * torch.sin(t * 440 * 2 * torch.pi)).reshape(1, 1, -1).repeat(1, 2, 1)
    save_mp4(path, images, {"waveform": waveform, "sample_rate": 32000})


def test_media_windows_seek_and_stream_without_duplicating_overlap(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "source.mp4"
    make_source(source)
    assert video_duration(source) == pytest.approx(9, abs=1 / 24)
    frames = load_video_frames(source, width=32, height=32, max_frames=24, start_seconds=4)
    assert frames.shape[0] == 24 and frames.mean() > 0.6
    first = load_audio_window(source, 1, 2.125)
    second = load_audio_window(source, 3.125, 1.875)
    whole = load_audio_window(source, 1, 4)
    joined = torch.cat((first["waveform"], second["waveform"]), dim=-1)
    assert torch.max(torch.abs(joined - whole["waveform"])) < 0.005
    output = tmp_path / "joined.mp4"
    writer = SegmentedMP4Writer(output, 32, 32, with_audio=True)
    try:
        writer.append(torch.zeros(51, 32, 32, 3), first, lambda: None)
        writer.append(torch.ones(45, 32, 32, 3) * 0.75, second, lambda: None)
        writer.finish(lambda: None)
    finally:
        writer.close()
    decoded, fps = load_video(output)
    assert len(decoded) == 96 and fps == 24
    assert decoded[:51].mean() < 0.1 and decoded[51:].mean() > 0.6
    audio = load_audio_window(output, 0, 4)
    assert audio["waveform"].shape[-1] == 128000
    assert torch.mean(torch.abs(audio["waveform"] - whole["waveform"])) < 0.01
    assert not writer.temporary.exists()


def test_source_interval_resolved_before_queueing(tmp_path, monkeypatch):
    async def scenario():
        settings = Settings(data_root=tmp_path, model_root=tmp_path / "models", enable_acceleration_lora=False)
        settings.model_paths["ref2va"].parent.mkdir(parents=True)
        settings.model_paths["ref2va"].touch()
        backend = LocalGenerationBackend(settings, runner=SimpleNamespace(readiness=lambda: {"status": "ready"}))
        monkeypatch.setattr(backend.assets, "resolve_input", lambda _: tmp_path / "source.mp4")
        monkeypatch.setattr("minimaxh3_service.local_backend.video_duration", lambda _: 67)
        request = GenerationRequest(prompt="change the actor", mode="edit", source_video={"file_id": "source.mp4", "start_seconds": 7},
                                    segmentation={"enabled": True})
        result = await backend.submit(request)
        plan = backend.tasks.records[result.id].payload
        assert result.frame_count == 60 * 24
        assert result.actual_overlap_frames == 39
        assert plan.request.source_video.duration_seconds == 60
        assert request.source_video.duration_seconds is None
        with pytest.raises(BackendError, match="exceeds"):
            await backend.submit(GenerationRequest(prompt="x", mode="edit", source_video={
                "file_id": "source.mp4", "start_seconds": 65, "duration_seconds": 4}))
        with pytest.raises(BackendError, match="segmentation"):
            await backend.submit(GenerationRequest(prompt="x", mode="edit", source_video={"file_id": "source.mp4"}))
    asyncio.run(scenario())


@pytest.mark.parametrize("edit", [False, True])
def test_native_segment_loop_masks_trims_and_preserves_audio(tmp_path, monkeypatch, edit):
    core = tensor_core()
    torch = core.torch
    core.model_management = SimpleNamespace(get_torch_device=lambda: torch.device("cpu"))
    source = tmp_path / "source.mp4"
    make_source(source)
    runner = NativeH3Runner(Settings(enable_acceleration_lora=False), lambda _: source)
    monkeypatch.setattr(runner, "_import_core", lambda: core)
    monkeypatch.setattr("minimaxh3_service.native.h3_memory_scope", lambda *args: contextlib.nullcontext())
    requests, masks = [], []

    def conditioning(plan, **kwargs):
        requests.append(plan.request)
        return [], {"samples": Nested((torch.zeros(1, 24, _video_latent_t(plan.frame_count), 2, 2),
                                       torch.zeros(1, 32, 2, round(plan.frame_count * 40 / 24))))}

    def sample(plan, model, conditioning, latent, cancel, report):
        masks.append(latent.get("noise_mask"))
        runner._test_frame_count = plan.frame_count
        report(0.85)
        return Nested(tuple(x + len(masks) / 4 for x in latent["samples"].unbind()))

    monkeypatch.setattr(runner, "_build_image_conditioning", conditioning)
    monkeypatch.setattr(runner, "_build_reference_conditioning", conditioning)
    monkeypatch.setattr(runner, "_sample", sample)
    runner._video_vae = SimpleNamespace(decode=lambda _: torch.ones(runner._test_frame_count, 32, 32, 3) * len(masks) / 4)
    runner._audio_vae = SimpleNamespace(audio_sample_rate=32000,
        decode=lambda _: torch.zeros(1, round(runner._test_frame_count * 32000 / 24), 2))
    kwargs = {"mode": "edit", "source_video": {"file_id": "source.mp4", "start_seconds": 1, "duration_seconds": 8},
              "audio_mode": "preserve_source"} if edit else {}
    plan = build_plan(GenerationRequest(prompt="x", duration_seconds=8, width=32, height=32,
        segmentation={"enabled": True, "segment_seconds": 4}, **kwargs))
    output = OutputArtifact("result.mp4", "", tmp_path / "result.mp4")
    progress = []
    runner._generate_windows(plan, SimpleNamespace(model=SimpleNamespace(diffusion_model=None)),
                             output, threading.Event(), progress.append, lambda: None)
    decoded, _ = load_video(output.path)
    assert len(decoded) == 192
    assert masks[0] is None and all(m is not None for m in masks[1:])
    assert all(isinstance(p, SegmentProgress) for p in progress)
    assert progress == sorted(progress)
    assert progress[-1].segment == len(plan.segments)
    if edit:
        assert [r.source_video.start_seconds for r in requests] == [1 + w.start_frame / 24 for w in plan.segments]
        actual = load_audio_window(output.path, 0, 8)["waveform"]
        expected = load_audio_window(source, 1, 8)["waveform"]
        assert torch.mean(torch.abs(actual - expected)) < 0.01


def test_stream_writer_cancellation_removes_partial_output(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    output = tmp_path / "cancel.mp4"
    writer = SegmentedMP4Writer(output, 32, 32, with_audio=False)
    def canceled():
        raise RuntimeError("canceled")
    try:
        with pytest.raises(RuntimeError, match="canceled"):
            writer.append(torch.zeros(4, 32, 32, 3), None, canceled)
    finally:
        writer.close()
    assert not output.exists() and not writer.temporary.exists()


def test_native_sampler_passes_continuation_mask_to_core(tmp_path, monkeypatch):
    core = tensor_core()
    torch = core.torch
    guider = Mock()
    output = Mock()
    guider.sample.return_value.to.return_value = output
    core.model_management = SimpleNamespace(intermediate_device=lambda: "cpu")
    core.sample = SimpleNamespace(fix_empty_latent_channels=lambda model, latent: latent,
                                 prepare_noise=lambda *args: "noise")
    core.samplers = SimpleNamespace(sampler_object=lambda _: "euler",
        calculate_sigmas=lambda *args: torch.tensor([1., 0.5, 0.]), CFGGuider=lambda _: guider)
    runner = NativeH3Runner(Settings(enable_acceleration_lora=False), lambda _: tmp_path)
    monkeypatch.setattr(runner, "_import_core", lambda: core)
    mask = object()
    plan = build_plan(GenerationRequest(prompt="x", steps=2))
    result = runner._sample(plan, Mock(), [], {"samples": "latent", "noise_mask": mask},
                            threading.Event(), lambda _: None)
    assert result is output
    assert guider.sample.call_args.kwargs["denoise_mask"] is mask


def test_http_segment_contract_progress_and_existing_endpoint(tmp_path):
    from starlette.testclient import TestClient
    from minimaxh3_service.api import create_app

    started, release = threading.Event(), threading.Event()

    class Runner:
        def readiness(self):
            return {"status": "ready"}

        def generate(self, plan, output, cancel, report):
            report(SegmentProgress(0.2, 1, len(plan.segments)))
            started.set()
            assert release.wait(5)
            output.path.parent.mkdir(parents=True, exist_ok=True)
            output.path.write_bytes(b"test output")

        def release_models(self):
            pass

        def close(self):
            pass

    settings = Settings(data_root=tmp_path / "data", model_root=tmp_path / "models")
    backend = LocalGenerationBackend(settings, runner=Runner())
    with TestClient(create_app(settings, backend_factory=lambda _: backend)) as client:
        try:
            result = client.post("/v1/generations", json={"mode": "t2v", "prompt": "x",
                "duration_seconds": 60, "segmentation": {"enabled": True}})
            assert result.status_code == 202
            body = result.json()
            assert body["segment_count"] == 7 and body["actual_overlap_frames"] == 39
            assert body["frame_count"] == 1440 and body["effective_duration_seconds"] == 60
            assert started.wait(2)
            task = client.get("/v1/tasks/" + body["id"]).json()
            assert task["raw_status"]["current_segment"] == 1
            assert task["raw_status"]["segment_count"] == 7
            invalid = client.post("/v1/generations", json={"mode": "t2v", "prompt": "x", "duration_seconds": 60})
            assert invalid.status_code == 422
        finally:
            release.set()
