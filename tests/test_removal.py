import asyncio
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest

from minimaxh3_service.backend import OutputArtifact
from minimaxh3_service.models import GenerationRequest, RemovalRequest
from minimaxh3_service.removal import repair_windows, run_command
from minimaxh3_service.tasks import GenerationCanceled, LocalTaskManager
from minimaxh3_service.ui import VideoClient, build_ui
from minimaxh3_service.ui_removal import selection_click
from minimaxh3_service.ui_removal import add_target
from minimaxh3_service.video_remove import exclusive_masks


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 1, 1, 0], [-.1, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_boxes(box):
    with pytest.raises(ValueError):
        RemovalRequest(file_id="video.mp4", box=box)


@pytest.mark.parametrize("extra", [
    {}, {"objects": []},
    {"box": [0, 0, 1, 1], "objects": [{"id": "a", "box": [0, 0, 1, 1]}]},
    {"objects": [{"id": "a", "box": [0, 0, 1, 1]}, {"id": "A", "box": [0, 0, 1, 1]}]},
    {"objects": [{"id": "../bad", "box": [0, 0, 1, 1]}]},
    {"objects": [{"id": "a", "frame_index": 600, "box": [0, 0, 1, 1]}]},
    {"objects": [{"id": str(i), "box": [0, 0, 1, 1]} for i in range(9)]},
])
def test_multi_object_validation(extra):
    with pytest.raises(ValueError):
        RemovalRequest(file_id="a.mp4", **extra)


def test_multi_objects_and_legacy_box_normalize():
    legacy = RemovalRequest(file_id="a.mp4", box=[0, 0, 1, 1])
    assert legacy.targets[0].frame_index == 0
    request = RemovalRequest(file_id="a.mp4", objects=[{"id": "角色", "frame_index": 3, "box": [0, 0, 1, 1]}])
    assert request.targets[0].id == "角色"
    items = add_target([], "dog", 3, [0, 0, 1, 1])
    assert items[0]["frame_index"] == 3
    with pytest.raises(ValueError, match="重复"):
        add_target(items, "DOG", 0, [0, 0, 1, 1])


def test_foreground_background_defaults_and_color_validation():
    request = RemovalRequest(file_id="a.mp4", box=[0, 0, 1, 1])
    assert request.foreground_background == "color" and request.background_color == "#00ff00"
    with pytest.raises(ValueError):
        RemovalRequest(file_id="a.mp4", box=[0, 0, 1, 1], background_color="not-a-color")


@pytest.mark.parametrize("mode,color", [("color", "#00ff00"), ("color", "#1234ab"), ("transparent", "#00ff00")])
def test_foreground_export_uses_color_or_real_alpha(tmp_path, mode, color):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from minimaxh3_service.removal import VideoRemovalRunner
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), 100, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    masks = tmp_path / "masks"; masks.mkdir()
    mask = np.zeros((64, 96), dtype=np.uint8); mask[:, :32] = 255
    Image.fromarray(mask).save(masks / "000000.png")
    VideoRemovalRunner._export_foreground(source, (96, 64), masks, lambda: None, mode, color)
    with Image.open(tmp_path / "rgba/000000.png") as rgba:
        assert rgba.getpixel((80, 30))[3] == 0
        assert rgba.getpixel((10, 30))[3] == 255
    if mode == "color":
        with Image.open(tmp_path / "preview/000000.jpg") as image:
            expected = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            assert max(abs(a-b) for a,b in zip(image.getpixel((80, 30)), expected)) <= 3
    else:
        assert not list((tmp_path / "preview").iterdir())


def test_overlap_assignment_and_late_objects():
    np = pytest.importorskip("numpy")
    scores = np.array([[[2, -1], [3, 1]], [[4, 0], [2, 5]]])
    masks = exclusive_masks(scores, [True, True])
    assert not (masks[0] & masks[1]).any()
    assert masks[0].tolist() == [[False, False], [True, False]]
    assert masks[1].tolist() == [[True, False], [False, True]]
    late = exclusive_masks(scores, [True, False])
    assert late[0].tolist() == [[True, False], [True, True]]
    assert not late[1].any()
    assert not exclusive_masks(scores, [False, False]).any()


def test_one_tracking_state_exports_all_masks_and_blank_prefix(monkeypatch, tmp_path):
    import contextlib
    import types
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from minimaxh3_service.models import RemovalObject
    from minimaxh3_service.video_remove import segment_objects
    calls = []

    class Predictor:
        def init_state(self, **kwargs):
            calls.append("init")
            return {}
        def add_new_points_or_box(self, **kwargs):
            calls.append((kwargs["obj_id"], kwargs["frame_idx"]))
        def propagate_in_video(self, state, start_frame_idx):
            calls.append("propagate")
            assert start_frame_idx == 2
            for i in (2, 3):
                # Returned ID order differs from request order.
                yield i, [2, 1], torch.tensor([[[[4., -1.], [4., -1.]]], [[[2., 3.], [2., 3.]]]])

    monkeypatch.setitem(sys.modules, "sam2.build_sam", types.SimpleNamespace(build_sam2_video_predictor=lambda *a, **k: Predictor()))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    folders = [tmp_path / name for name in ["union", "first", "second"]]
    for folder in folders: folder.mkdir()
    objects = [RemovalObject(id="a", frame_index=2, box=[0, 0, 1, 1]),
               RemovalObject(id="b", frame_index=3, box=[0, 0, 1, 1])]
    segment_objects(tmp_path, folders[0], folders[1:], 4, (2, 2), tmp_path/"model", "config", objects)
    assert calls == ["init", (1, 2), (2, 3), "propagate"]
    def read(folder, i):
        with Image.open(folder / f"{i:06d}.png") as image:
            return np.asarray(image) > 0
    for folder in folders:
        assert not read(folder, 0).any() and not read(folder, 1).any()
    assert read(folders[1], 2).all() and not read(folders[2], 2).any()
    assert read(folders[2], 3).tolist() == [[True, False], [True, False]]
    assert not (read(folders[1], 3) & read(folders[2], 3)).any()
    assert read(folders[0], 3).all()


def test_selection_is_normalized_and_restarts():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (400, 800))
    _, points, box = selection_click(image, [], (300, 600))
    assert box is None
    _, points, box = selection_click(image, points, (100, 200))
    assert box == [.25, .25, .75, .75]
    _, points, box = selection_click(image, points, (20, 20))
    assert len(points) == 1 and box is None


def test_windows_cover_last_frame_without_tiny_redundant_tail():
    for count in [2, 80, 81, 144, 452, 900]:
        windows = list(repair_windows(count))
        assert windows[-1][1] == count
        assert {i for start, end in windows for i in range(start, end)} == set(range(count))
        assert all(end-start <= 80 for start, end in windows)
        if len(windows) > 1:
            assert windows[-1][1] - windows[-1][0] > 16


def test_command_cancellation_reaps_child(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(.3, cancel.set)
    timer.start()
    try:
        with pytest.raises(GenerationCanceled):
            run_command([sys.executable, "-c", "import time; time.sleep(30)"], cancel)
    finally:
        timer.cancel()


def test_removal_shares_queue_releases_generation_and_returns_all_outputs(tmp_path):
    events = []

    class Generation:
        def generate(self, *args): events.append("generate")
        def release_models(self): events.append("release-generation")
        def close(self): pass

    class Removal:
        def process(self, request, output, cancel, progress):
            events.append("remove")
            return [OutputArtifact("preview.mp4", "", tmp_path / "preview.mp4"),
                    OutputArtifact("alpha.mov", "", tmp_path / "alpha.mov")]
        def release_models(self): events.append("release-removal")
        def close(self): pass

    async def scenario():
        manager = LocalTaskManager(Generation(), lambda id, prefix: OutputArtifact(id+".mp4", "", tmp_path/(id+".mp4")),
            max_queue_size=3, allow_interrupt_running=True, removal_runner=Removal())
        await manager.start()
        await manager.submit(GenerationRequest(prompt="test"))
        submitted = await manager.submit_removal(RemovalRequest(file_id="v.mp4", objects=[
            {"id": "a", "box": [0, 0, .5, 1]}, {"id": "b", "box": [.5, 0, 1, 1]}]))
        assert len(manager.records) == 2  # One generation and one multi-object job.
        await manager.queue.join()
        assert events[:3] == ["generate", "release-generation", "remove"]
        task = manager.get(submitted.id)
        assert task.kind == "removal" and task.status == "succeeded"
        assert len(task.outputs) == 2
        await manager.close()
    asyncio.run(scenario())


def test_client_uploads_authenticated_removal_and_ui_refreshes_both_results(tmp_path):
    calls = []
    def handle(request):
        assert request.headers["Authorization"] == "Bearer secret"
        calls.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"capabilities": {"removal": {"background_available": True}}})
        if request.url.path == "/v1/files":
            return httpx.Response(201, json={"file_id": "uploads/v.mp4"})
        body = json.loads(request.content)
        assert body["file_id"] == "uploads/v.mp4" and body["output_mode"] == "both"
        return httpx.Response(202, json={"id": "r1", "queue_number": 1})
    source = tmp_path / "v.mp4"; source.write_bytes(b"video")
    client = VideoClient("http://test", "secret", transport=httpx.MockTransport(handle))
    assert client.submit_removal(source, {"box": [0, 0, 1, 1], "output_mode": "both"})["id"] == "r1"
    assert calls == ["/v1/models", "/v1/files", "/v1/removals"]
    pytest.importorskip("gradio")
    demo = build_ui(client)
    refresh = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_removal")
    client.task = lambda id: {"status": "succeeded", "outputs": []}
    client.download = lambda outputs: ["a-foreground-preview.mp4", "a-foreground-alpha.mov", "a-background.mp4"]
    result = refresh("r1", "", [])
    assert result[1:3] == ("a-foreground-preview.mp4", "a-background.mp4")
    assert len(result[3]) == 3
    submit = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "submit_removal")
    with pytest.raises(Exception, match="首帧"):
        submit("new.mp4", "old.mp4", [0, 0, 1, 1], "both", 640, 640, 600, 8)


def test_ui_submits_multiple_objects_and_preserves_preview_choice():
    pytest.importorskip("gradio")
    client = VideoClient("http://test")
    demo = build_ui(client)
    submit = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "submit_removal")
    captured = []
    def send(video, parameters):
        captured.append(parameters)
        return {"id": "multi", "queue_number": 1}
    client.submit_removal = send
    targets = [{"id": "person", "frame_index": 0, "box": [0, 0, .5, 1]},
               {"id": "dog", "frame_index": 5, "box": [.5, .5, 1, 1]}]
    result = submit("v.mp4", "v.mp4", None, "both", 640, 640, 600, 8, targets, 5, 5, "transparent", "#123456")
    assert result[0] == "multi" and captured[0]["objects"] == targets
    assert captured[0]["foreground_background"] == "transparent"
    assert captured[0]["background_color"] == "#123456"
    assert "box" not in captured[0]
    with pytest.raises(Exception, match="尚未加入"):
        submit("v.mp4", "v.mp4", [0, 0, 1, 1], "both", 640, 640, 600, 8, targets, 5, 5)
    refresh = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "refresh_removal")
    files = ["test-object-person-foreground-preview.mp4", "test-object-dog-foreground-preview.webm"]
    client.task = lambda id: {"status": "succeeded"}
    result = refresh("multi", "multi", files, files[1])
    assert result[1] == files[1]


def test_removal_api_auth_validation_and_task_kind(tmp_path):
    from starlette.testclient import TestClient
    from minimaxh3_service.api import create_app
    from minimaxh3_service.backend import BackendTask
    from minimaxh3_service.config import Settings
    from minimaxh3_service.models import SubmittedRemovalTask

    class Backend:
        async def start(self): pass
        async def close(self): pass
        async def submit_removal(self, request):
            assert request.output_mode == "background"
            return SubmittedRemovalTask(id="r1", queue_number=1)
        async def get_task(self, task_id):
            return BackendTask(id=task_id, kind="removal", status="queued")

    app = create_app(Settings(api_key="secret", data_root=tmp_path), lambda settings: Backend())
    with TestClient(app) as client:
        body = {"file_id": "a.mp4", "box": [0, 0, 1, 1], "output_mode": "background"}
        assert client.post("/v1/removals", json=body).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        response = client.post("/v1/removals", json=body, headers=headers)
        assert response.status_code == 202 and response.json()["id"] == "r1"
        assert client.get("/v1/tasks/r1", headers=headers).json()["kind"] == "removal"
        body["box"] = [0, 0, 2, 1]
        assert client.post("/v1/removals", json=body, headers=headers).status_code == 422
