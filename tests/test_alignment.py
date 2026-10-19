import asyncio
import json

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from minimaxh3_service.api import create_app
from minimaxh3_service.backend import BackendError, OutputArtifact
from minimaxh3_service.config import Settings
from minimaxh3_service.local_backend import LocalGenerationBackend
from minimaxh3_service.models import AlignmentRequest


class Runner:
    def readiness(self):
        return {"status": "ready"}

    def release_models(self):
        pass

    def close(self):
        pass

    def process(self, request, output, cancel, progress):
        output.path.write_text(json.dumps({"text": request.lyrics, "timing": "forced_alignment"}), encoding="utf-8")
        txt = output.path.with_suffix(".txt")
        txt.write_text(request.lyrics, encoding="utf-8")
        return [output, OutputArtifact(txt.name, output.subfolder, txt)]


def test_alignment_api_auth_queue_and_original_lyrics(tmp_path):
    idle = Runner()
    backend = LocalGenerationBackend(Settings(data_root=tmp_path), runner=idle, upscale_runner=idle,
              removal_runner=idle, music_runner=idle, depth_runner=idle, skeleton_runner=idle,
              beat_runner=idle, transcription_runner=idle, alignment_runner=idle)
    with TestClient(create_app(Settings(api_key="test"), backend_factory=lambda _: backend)) as client:
        assert client.post("/v1/audio/alignments", json={"file_id": "a.wav", "lyrics": "人改的词"}).status_code == 401
        client.headers["X-API-Key"] = "test"
        file_id = client.post("/v1/files", data={"kind": "audio"}, files={"file": ("song.wav", b"audio", "audio/wav")}).json()["file_id"]
        response = client.post("/v1/audio/alignments", json={"file_id": file_id, "lyrics": "人改的词\n第二行"})
        assert response.status_code == 202
        client.portal.call(backend.tasks.queue.join)
        task = client.get("/v1/tasks/" + response.json()["id"]).json()
        assert task["status"] == "succeeded" and task["kind"] == "alignment"
        assert len(task["outputs"]) == 2
        assert client.get(task["outputs"][0]["download_url"]).json()["text"] == "人改的词\n第二行"
        assert client.get("/v1/models").json()["capabilities"]["alignment"]["available"] is True
        backend.tasks.alignment_runner.readiness = lambda: {"status": "not_ready"}
        assert client.post("/v1/audio/alignments", json={"file_id": file_id, "lyrics": "歌词"}).status_code == 503


def test_alignment_groups_word_times_without_changing_line_text():
    pytest.importorskip("numpy")
    from minimaxh3_service.alignment_worker import make_cues
    words = [{"text": "风", "start_time": 2, "end_time": 2.2},
             {"text": "来", "start_time": 2.2, "end_time": 3},
             {"text": "了", "start_time": 4, "end_time": 5}]
    cues, result = make_cues(["风来，", "了！"], [["风", "来"], ["了"]], words, 10)
    assert cues == [{"id": "line_1", "start": 2, "end": 3, "text": "风来，"},
                    {"id": "line_2", "start": 4, "end": 5, "text": "了！"}]
    assert result[0]["start"] == 2  # Leading music is not trimmed away.
    with pytest.raises(ValueError, match="tokens"):
        make_cues(["别的词"], [["别"]], words, 10)
    with pytest.raises(ValueError, match="timestamps"):
        make_cues(["风"], [["风"]], [{"text": "风", "start_time": 3, "end_time": 2}], 10)
    with pytest.raises(ValueError, match="could not be aligned"):
        make_cues(["风"], [["风"]], [{"text": "风", "start_time": 3, "end_time": 3}], 10)


@pytest.mark.parametrize("body", [{"file_id": "../x.wav", "lyrics": "词"}, {"file_id": "x.wav", "lyrics": " "},
                                  {"file_id": "x.wav", "lyrics": "词", "language": "auto"}])
def test_alignment_rejects_invalid_requests(body):
    with pytest.raises(ValidationError):
        AlignmentRequest(**body)
