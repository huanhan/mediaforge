import asyncio
from fractions import Fraction
from pathlib import Path
import threading
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.backend import BackendError, OutputArtifact
from minimaxh3_service.config import Settings
from minimaxh3_service.depth import VideoDepthRunner
from minimaxh3_service.depth_worker import read_frames, write_depths
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.models import DepthRequest
from minimaxh3_service.tasks import GenerationCanceled


class IdleRunner:
    def readiness(self):
        return {"status": "ready"}

    def release_models(self):
        pass

    def close(self):
        pass


class FakeDepth(IdleRunner):
    def process(self, request, output, cancel, progress):
        output.path.write_bytes(b"depth-mp4")
        raw = output.path.with_suffix(".npz")
        raw.write_bytes(b"depth-npz")
        return [output, OutputArtifact(raw.name, output.subfolder, raw)]


def make_backend(tmp_path, depth=None):
    return LocalGenerationBackend(Settings(data_root=tmp_path), runner=IdleRunner(),
                                  upscale_runner=IdleRunner(), removal_runner=IdleRunner(),
                                  music_runner=IdleRunner(), depth_runner=depth or FakeDepth())


def test_depth_api_upload_queue_status_download_and_auth(tmp_path):
    backend = make_backend(tmp_path)
    app = create_app(Settings(api_key="secret"), backend_factory=lambda _: backend)
    with TestClient(app) as client:
        assert client.post("/v1/depths", json={"file_id": "x.mp4"}).status_code == 401
        client.headers["Authorization"] = "Bearer secret"
        upload = client.post("/v1/files", data={"kind": "video"},
                             files={"file": ("source.mp4", b"video", "video/mp4")})
        assert upload.status_code == 201
        response = client.post("/v1/depths", json={"file_id": upload.json()["file_id"], "save_raw": True})
        assert response.status_code == 202
        task_id = response.json()["id"]
        client.portal.call(backend.tasks.queue.join)
        task = client.get(f"/v1/tasks/{task_id}").json()
        assert task["kind"] == "depth"
        assert task["status"] == "succeeded"
        assert len(task["outputs"]) == 2
        assert [client.get(item["download_url"]).content for item in task["outputs"]] == [b"depth-mp4", b"depth-npz"]


@pytest.mark.parametrize("body", [{"file_id": "../x.mp4"}, {"file_id": "x.mp4", "input_size": 999},
                                  {"file_id": "x.mp4", "max_resolution": 9000}, {"file_id": "x.mp4", "metric": True}])
def test_invalid_depth_requests(body):
    with pytest.raises(ValidationError):
        DepthRequest(**body)


def test_backend_rejects_missing_input_wrong_media_and_missing_model(tmp_path):
    async def scenario():
        backend = make_backend(tmp_path)
        backend.assets.initialize()
        with pytest.raises(BackendError) as caught:
            await backend.submit_depth(DepthRequest(file_id="missing.mp4"))
        assert caught.value.status_code == 404
        (backend.assets.input_root / "image.png").touch()
        with pytest.raises(BackendError) as caught:
            await backend.submit_depth(DepthRequest(file_id="image.png"))
        assert caught.value.status_code == 415
        (backend.assets.input_root / "video.mp4").touch()
        with patch.object(backend.tasks.depth_runner, "readiness", return_value={"status": "not_ready"}):
            with pytest.raises(BackendError) as caught:
                await backend.submit_depth(DepthRequest(file_id="video.mp4"))
        assert caught.value.code == "depth_not_ready"
    asyncio.run(scenario())


def test_depth_encoding_global_range_raw_fps_and_decode_limit(tmp_path):
    np = pytest.importorskip("numpy")
    av = pytest.importorskip("av")
    depths = np.stack([np.full((32, 48), v, dtype=np.float32) for v in (0, 5, 10)])
    path = tmp_path / "depth.mp4"
    write_depths(depths, Fraction(30000, 1001), path, True)
    with np.load(path.with_suffix(".npz")) as raw:
        np.testing.assert_array_equal(raw["depths"], depths)
        assert raw["fps_numerator"] == 30000
        assert raw["fps_denominator"] == 1001
    with av.open(str(path)) as container:
        assert container.streams.video[0].average_rate == Fraction(30000, 1001)
        gray = [frame.to_ndarray(format="gray").mean() for frame in container.decode(video=0)]
    assert len(gray) == 3 and gray[0] < gray[1] < gray[2]
    with pytest.raises(ValueError, match="exceeds"):
        read_frames(path, 64, 2)
    frames, fps = read_frames(path, 64, 3)
    assert frames.shape == (3, 32, 48, 3)


def test_cancel_terminates_worker_and_cleans_partial_files(tmp_path):
    cancel = threading.Event()
    class Process:
        returncode = None
        terminated = False

        def poll(self):
            cancel.set()
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

    process = Process()
    output = OutputArtifact("depth.mp4", "", tmp_path / "depth.mp4")
    for suffix in (".npz", ".partial.npz", ".partial.mp4"):
        output.path.with_suffix(suffix).touch()
    runner = VideoDepthRunner(Settings(), lambda _: tmp_path / "source.mp4")
    with patch("minimaxh3_service.depth.subprocess.Popen", return_value=process):
        with pytest.raises(GenerationCanceled):
            runner.process(DepthRequest(file_id="x.mp4"), output, cancel, lambda _: None)
    assert process.terminated
    assert not list(tmp_path.iterdir())


def test_queued_depth_cancellation(tmp_path):
    async def scenario():
        backend = make_backend(tmp_path)
        backend.assets.initialize()
        submitted = await backend.tasks.submit_depth(DepthRequest(file_id="x.mp4"))
        assert backend.tasks.cancel(submitted.id).status == "canceled"
        await backend.tasks.start()
        await backend.tasks.queue.join()
        assert backend.tasks.get(submitted.id).outputs == []
        await backend.close()
    asyncio.run(scenario())
