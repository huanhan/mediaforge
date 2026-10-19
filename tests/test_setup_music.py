import hashlib
import io

import pytest

from minimaxh3_service import setup_music


def test_range_download_resumes_verifies_and_publishes(tmp_path, monkeypatch):
    data = b"official pinned model bytes"
    target = tmp_path / "model.safetensors"
    target.with_name(target.name + ".chunk-000000").write_bytes(data[:8])
    ranges = []

    def urlopen(request, timeout):
        first, last = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        ranges.append((first, last))
        response = io.BytesIO(data[first:last + 1])
        response.status = 206
        response.headers = {"Content-Range": f"bytes {first}-{last}/{len(data)}"}
        return response

    monkeypatch.setattr(setup_music.urllib.request, "urlopen", urlopen)
    digest = hashlib.sha256(data).hexdigest()
    setup_music.download_weights("https://example/model", target, len(data), digest, chunk_size=8)
    assert target.read_bytes() == data
    assert all(start >= 8 for start, _ in ranges)
    assert list(tmp_path.iterdir()) == [target]
    ranges.clear()
    setup_music.download_weights("https://example/model", target, len(data), digest, chunk_size=8)
    assert not ranges


@pytest.mark.parametrize("failure", ["range", "checksum", "truncated"])
def test_download_never_installs_invalid_model(tmp_path, monkeypatch, failure):
    data = b"model"
    target = tmp_path / "model.safetensors"

    def urlopen(request, timeout):
        response = io.BytesIO(data[:-1] if failure == "truncated" else data)
        response.status = 200 if failure == "range" else 206
        response.headers = {"Content-Range": "bytes 0-4/5"}
        return response

    monkeypatch.setattr(setup_music.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(setup_music.time, "sleep", lambda _: None)
    digest = "0" * 64 if failure == "checksum" else hashlib.sha256(data).hexdigest()
    with pytest.raises((ValueError, RuntimeError)):
        setup_music.download_weights("https://example/model", target, 5, digest, chunk_size=8)
    assert not target.exists()


def test_setup_uses_separate_interpreter_and_pinned_runtime(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_music.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(setup_music.subprocess, "run", lambda args, **kw: calls.append(args))
    assert setup_music.main(["--runtime-dir", str(tmp_path / "env"), "--runtime-only"]) == 0
    python = str(setup_music.runtime_python(tmp_path / "env"))
    assert calls[0] == ["uv", "venv", "--python", "3.12", str(tmp_path / "env")]
    assert calls[1][calls[1].index("--python") + 1] == python
    assert "torch==2.10.0" in calls[1]
    assert calls[2][-1].endswith("@" + setup_music.RUNTIME_REVISION)
