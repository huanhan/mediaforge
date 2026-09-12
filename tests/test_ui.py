from pathlib import Path

import httpx
import pytest

from minimaxh3_service.ui import VideoClient, build_ui


def test_video_workflow_authenticated_upload_cancel_download(tmp_path):
    import json

    requests = []

    def handle(request):
        assert request.headers["Authorization"] == "Bearer secret"
        requests.append(request)
        if request.url.path == "/v1/files":
            assert b"frame.png" in request.read()
            return httpx.Response(201, json={"file_id": "uploads/frame.png"})
        if request.url.path == "/v1/generations":
            body = json.loads(request.content)
            assert body["first_frame"] == "uploads/frame.png"
            assert body["seed"] == 2**63 - 1
            assert body["steps"] == 12
            return httpx.Response(202, json={"id": "task-1"})
        if request.url.path == "/v1/tasks/task-1":
            return httpx.Response(200, json={"status": "canceled" if request.method == "DELETE" else "running"})
        if request.url.path == "/v1/files/content":
            assert request.url.params["subfolder"] == "videos"
            return httpx.Response(200, content=b"video-content")
        raise AssertionError(request.url)

    client = VideoClient("http://test", "secret", transport=httpx.MockTransport(handle))
    source = tmp_path / "frame.png"
    source.write_bytes(b"image")
    result, body = client.submit(dict(prompt="Test", mode="fl2v", width=608, height=352,
                                     seed=2**63 - 1, steps=12, first_frame="pending.png"),
                                 {"first_frame": str(source)})
    assert result["id"] == "task-1"
    assert client.task("task-1")["status"] == "running"
    assert client.task("task-1", cancel=True)["status"] == "canceled"
    paths = client.download([dict(filename="result.mp4", subfolder="videos",
                                  download_url="http://untrusted/result.mp4")])
    try:
        assert Path(paths[0]).read_bytes() == b"video-content"
        assert all(request.url.host == "test" for request in requests)
    finally:
        Path(paths[0]).unlink()
        Path(paths[0]).parent.rmdir()


@pytest.mark.parametrize("parameters", [
    {"width": 609}, {"width": 1344, "height": 800},
    {"mode": "fl2v"}, {"mode": "ref2v", "reference_images": ["p.png"] * 10},
    {"seed": -1}, {"duration_seconds": 20},
])
def test_invalid_parameters_do_not_upload(parameters):
    def unexpected(request):
        pytest.fail("Invalid request must not contact API")

    client = VideoClient("http://test", transport=httpx.MockTransport(unexpected))
    with pytest.raises(ValueError):
        client.submit({"prompt": "Test", **parameters}, {})


def test_ref2v_uploads_and_audio_choice(tmp_path):
    import json

    kinds = []

    def handle(request):
        if request.url.path == "/v1/files":
            content = request.read()
            kind = next(kind for kind in ("image", "video", "audio")
                        if f"\r\n\r\n{kind}\r\n".encode() in content)
            kinds.append(kind)
            return httpx.Response(201, json={"file_id": f"uploads/{kind}"})
        body = json.loads(request.content)
        assert body["reference_videos"] == [{"file_id": "uploads/video", "include_source_audio": False}]
        assert body["reference_images"] == ["uploads/image"]
        assert body["reference_audios"] == ["uploads/audio"]
        return httpx.Response(202, json={"id": "ref-task"})

    source = tmp_path / "source.png"
    source.write_bytes(b"media")
    client = VideoClient("http://test", transport=httpx.MockTransport(handle))
    client.submit(dict(prompt="Test", mode="ref2v", reference_images=["pending.png"],
                       reference_videos=[{"file_id": "pending.mp4", "include_source_audio": False}],
                       reference_audios=["pending.wav"]),
                  dict(reference_images=[source], reference_videos=[source],
                       reference_audios=[source], include_source_audio=False))
    assert kinds == ["image", "video", "audio"]


def test_api_errors_are_readable():
    client = VideoClient("http://test", transport=httpx.MockTransport(
        lambda request: httpx.Response(401, json={"detail": "invalid API key"})))
    with pytest.raises(ValueError, match="401.*invalid API key"):
        client.catalog()


def test_gradio_page_and_events():
    gr = pytest.importorskip("gradio")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    demo = build_ui(VideoClient("http://test"))
    app = gr.mount_gradio_app(FastAPI(), demo, path="/")
    with TestClient(app) as browser:
        response = browser.get("/")
        assert response.status_code == 200
        assert "MiniMax H3" in response.text
        config = browser.get("/config").json()
        types = {component["type"] for component in config["components"]}
        assert {"video", "timer", "image", "file", "slider"} <= types
        assert len(config["dependencies"]) >= 7


@pytest.mark.parametrize("api_key", [None, "test-secret"])
def test_ui_shares_api_app_and_preserves_auth_and_lifecycle(tmp_path, api_key):
    pytest.importorskip("gradio")
    from fastapi.testclient import TestClient
    from minimaxh3_service.api import create_app
    from minimaxh3_service.config import Settings
    from test_api import FakeLocalBackend

    backend = FakeLocalBackend(tmp_path / "result.mp4")
    app = create_app(Settings(api_key=api_key), backend_factory=lambda _: backend)
    with TestClient(app) as browser:
        assert browser.get("/healthz").status_code == 200
        assert browser.get("/docs").status_code == 200
        assert browser.get("/ui/").status_code == 200
        if api_key:
            assert browser.get("/ui/config").status_code == 401
            assert browser.get("/v1/models").status_code == 401
            assert browser.post("/ui/login", data={"username": "h3", "password": "wrong"}).status_code == 400
            assert browser.post("/ui/login", data={"username": "h3", "password": api_key}).status_code == 200
        assert browser.get("/ui/config").status_code == 200
        headers = {"X-API-Key": api_key} if api_key else {}
        assert browser.get("/v1/models", headers=headers).status_code == 200
        response = browser.post("/v1/generations", headers=headers, json={"prompt": "test"})
        assert response.status_code == 202
        assert "task-123" in backend.tasks
    assert backend.closed


@pytest.mark.parametrize("host, expected", [("0.0.0.0", "127.0.0.1"), ("::", "[::1]"),
                                          ("127.0.0.1", "127.0.0.1")])
def test_mounted_client_uses_service_port(monkeypatch, host, expected):
    pytest.importorskip("gradio")
    from fastapi import FastAPI
    from minimaxh3_service import ui
    from minimaxh3_service.config import Settings

    original = ui.build_ui
    clients = []

    def capture(client):
        clients.append(client)
        return original(client)

    monkeypatch.setattr(ui, "build_ui", capture)
    ui.mount_ui(FastAPI(), Settings(host=host, port=8123))
    assert clients[0].base_url == f"http://{expected}:8123"
