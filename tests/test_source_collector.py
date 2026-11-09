from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mediaforge_service import api as api_module
from mediaforge_service import source_collector
from mediaforge_service.api import create_app
from mediaforge_service.config import Settings

from test_api import FakeLocalBackend

EXTENSION_HEADERS = {"X-Collector-Version": "2.0.0"}
LONG_TEXT = "这是一段用于测试的公开网页正文。" * 10
PUBLIC_URL = "https://example.com/articles/1"


@pytest.fixture(autouse=True)
def no_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep tests offline: pretend every public hostname resolves to a public address.
    monkeypatch.setattr(source_collector, "check_task_target", lambda url: None)
    monkeypatch.setattr(api_module, "check_task_target", lambda url: None)


@pytest.fixture
def make_client(tmp_path: Path):
    """Build a client with the app lifespan running (state.settings is set on startup)."""
    clients: list[TestClient] = []

    def build(root: Path | None = None, *, api_key: str | None = None) -> TestClient:
        base = root or tmp_path
        output_file = base / "out.mp4"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(b"video")
        fake = FakeLocalBackend(output_file)
        app = create_app(
            Settings(api_key=api_key, data_root=base / "data"),
            backend_factory=lambda _: fake,
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


def test_collection_rejects_private_and_credentialed_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client) -> None:
    client = make_client(tmp_path)
    assert client.post("/v1/collections", json={"url": "ftp://example.com/a"}).status_code == 400
    assert client.post("/v1/collections", json={"url": "https://user:pw@example.com/a"}).status_code == 400

    def refuse(url: str) -> None:
        raise source_collector.BridgeError("浏览器采集不能访问本机或内网")

    monkeypatch.setattr(api_module, "check_task_target", refuse)
    response = client.post("/v1/collections", json={"url": "http://127.0.0.1/admin"})
    assert response.status_code == 400
    # A rejected URL must not occupy the queue.
    assert client.get("/v1/collections").json()["collections"] == []


def test_full_collection_flow_writes_source_and_serves_it(tmp_path: Path, make_client) -> None:
    client = make_client(tmp_path)
    submitted = client.post("/v1/collections", json={"url": PUBLIC_URL, "title": "测试文章"})
    assert submitted.status_code == 202
    collection_id = submitted.json()["id"]

    claim = client.get("/api/collector?mode=ping", headers=EXTENSION_HEADERS).json()
    assert claim["ok"] is True and claim["browser"]["connected"] is True
    task = client.get("/api/collector", headers=EXTENSION_HEADERS).json()["task"]
    assert task["id"] == collection_id and task["url"] == PUBLIC_URL

    queued_status = client.get(f"/v1/collections/{collection_id}").json()
    assert queued_status["status"] == "running"
    assert queued_status["source_download_url"] is None

    result = {"id": task["id"], "lease": task["lease"], "status": "completed",
              "title": "真实标题", "url": PUBLIC_URL, "text": LONG_TEXT,
              "comments": [{"author": "甲", "likes": 3, "text": "好"}], "media": []}
    assert client.post("/api/collector", json=result, headers=EXTENSION_HEADERS).status_code == 200

    done = client.get(f"/v1/collections/{collection_id}").json()
    assert done["status"] == "completed"
    assert done["source_download_url"].endswith(f"/v1/collections/{collection_id}/source")
    source = client.get(done["source_download_url"])
    assert source.status_code == 200
    assert "# 来源：真实标题" in source.text
    assert "甲（赞 3）：好" in source.text

    listing = client.get("/v1/collections").json()["collections"]
    assert [row["id"] for row in listing] == [collection_id]


def test_short_or_stale_results_are_refused(tmp_path: Path, make_client) -> None:
    client = make_client(tmp_path)
    collection_id = client.post("/v1/collections", json={"url": PUBLIC_URL}).json()["id"]
    task = client.get("/api/collector", headers=EXTENSION_HEADERS).json()["task"]

    short = {"id": task["id"], "lease": task["lease"], "status": "completed", "text": "太短"}
    response = client.post("/api/collector", json=short, headers=EXTENSION_HEADERS)
    assert response.status_code == 400 and "正文" in response.json()["error"]

    bad_lease = {"id": task["id"], "lease": "0" * 32, "status": "failed", "error": "x"}
    assert client.post("/api/collector", json=bad_lease, headers=EXTENSION_HEADERS).status_code == 400

    failed = {"id": task["id"], "lease": task["lease"], "status": "waiting_login", "error": "请登录"}
    assert client.post("/api/collector", json=failed, headers=EXTENSION_HEADERS).status_code == 200
    row = client.get(f"/v1/collections/{collection_id}").json()
    assert row["status"] == "waiting_login" and row["error"] == "请登录"


def test_redirect_to_other_host_is_refused(tmp_path: Path, make_client) -> None:
    client = make_client(tmp_path)
    client.post("/v1/collections", json={"url": PUBLIC_URL})
    task = client.get("/api/collector", headers=EXTENSION_HEADERS).json()["task"]
    moved = {"id": task["id"], "lease": task["lease"], "status": "completed",
             "url": "https://other.example/x", "text": LONG_TEXT}
    response = client.post("/api/collector", json=moved, headers=EXTENSION_HEADERS)
    assert response.status_code == 400 and "其他站点" in response.json()["error"]


def test_openapi_lists_collection_routes(tmp_path: Path, make_client) -> None:
    schema = make_client(tmp_path).get("/openapi.json").json()
    assert "/v1/collections" in schema["paths"]
    assert "/v1/collections/{collection_id}/source" in schema["paths"]
    # The extension protocol stays out of the public API surface.
    assert "/api/collector" not in schema["paths"]
