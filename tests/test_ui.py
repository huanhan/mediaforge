from pathlib import Path

import httpx
import pytest

from mediaforge_service.ui import VideoClient, build_ui


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:8000", False), ("http://127.0.0.2:8000", False),
    ("http://localhost:8000", False), ("http://LOCALHOST.:8000", False),
    ("http://[::1]:8000", False), ("https://api.example.com", True),
])
def test_proxy_policy_only_bypasses_loopback(url, expected):
    with VideoClient(url, transport=httpx.MockTransport(lambda r: httpx.Response(200))).connect() as client:
        assert client.trust_env is expected


def test_local_removal_upload_bypasses_broken_environment_proxy(monkeypatch, tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def respond(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self.respond(200, {"capabilities": {"removal": {"foreground_available": True}}})
        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer local-secret"
            data = self.rfile.read(int(self.headers["Content-Length"]))
            received.append((self.path, data))
            self.respond(201 if self.path == "/v1/files" else 202,
                         {"file_id": "uploads/video.mp4"} if self.path == "/v1/files" else {"id": "queued"})

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video-payload" * 10000)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VideoClient(f"http://127.0.0.1:{server.server_port}", "local-secret")
        result = client.submit_removal(video, {"box": [0, 0, 1, 1]})
        assert result["id"] == "queued"
        assert [path for path, data in received] == ["/v1/files", "/v1/removals"]
        assert b"video-payload" in received[0][1]
        assert json.loads(received[1][1])["file_id"] == "uploads/video.mp4"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
            assert not {"steps", "sampler_name", "scheduler"} & body.keys()
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
                                     seed=2**63 - 1, first_frame="pending.png"),
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


@pytest.mark.parametrize("seed,prefix", [
    ("42", None), (None, None), ("", ""), ("   ", "   "), ("0", " demo/video "),
])
def test_submit_callback_accepts_empty_optional_textboxes(seed, prefix):
    pytest.importorskip("gradio")
    import json

    requests = []

    def handle(request):
        assert request.url.path == "/v1/generations"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(202, json={
            "id": "task-empty-fields", "queue_number": 1,
            "effective_duration_seconds": 4.4583, "frame_count": 107,
        })

    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(handle)))
    submit = next(event.fn for event in demo.fns.values() if event.fn.__name__ == "submit")
    result = submit("Test video", "t2v", 4, 608, 352, seed, 1, prefix,
                    None, None, None, None, None, True, "match")
    assert result[0] == "task-empty-fields"
    assert len(requests) == 1
    assert requests[0].get("filename_prefix") == ((prefix or "").strip() or None)
    if seed and seed.strip():
        assert requests[0]["seed"] == int(seed)
    else:
        assert 0 <= requests[0]["seed"] <= 2**63 - 1


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
    from mediaforge_service.api import create_app
    from mediaforge_service.config import Settings
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
    from mediaforge_service import ui
    from mediaforge_service.config import Settings

    original = ui.build_ui
    clients = []

    def capture(client):
        clients.append(client)
        return original(client)

    monkeypatch.setattr(ui, "build_ui", capture)
    ui.mount_ui(FastAPI(), Settings(host=host, port=8123))
    assert clients[0].base_url == f"http://{expected}:8123"


@pytest.mark.parametrize("canvas,resolution,ratio,expected", [
    ((None, None), "480p", "9:16", (480, 832)),
    ((0, 0), "720p", "16:9", (1280, 704)),
    ((None, None), "default", "9:16", (1344, 768)),
    ((608, 352), "2160p", "1:1", (608, 352)),
])
def test_generation_ui_resolves_size_presets(canvas, resolution, ratio, expected):
    import json
    pytest.importorskip("gradio")
    payloads = []
    def handle(request):
        assert request.url.path == "/v1/generations"
        payloads.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "preset", "effective_duration_seconds": 4.4583, "frame_count": 107})
    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(handle)))
    event = next(event for event in demo.fns.values() if event.fn.__name__ == "submit")
    args = ["x", "t2v", 4, *canvas, "1", 1, None, None, None, None, None, None, True, "match",
            False, 10, 48, resolution, ratio]
    for index in (3, 4, -2, -1):
        args[index] = event.inputs[index].preprocess(args[index])
    result = event.fn(*args)
    assert (payloads[0]["width"], payloads[0]["height"]) == expected
    assert f"{expected[0]}×{expected[1]}" in result[2]
