from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "matrix_test.py"
SPEC = importlib.util.spec_from_file_location("matrix_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
matrix_test = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix_test
SPEC.loader.exec_module(matrix_test)


def test_parse_values_and_build_cartesian_matrix() -> None:
    resolutions = matrix_test.parse_resolutions(["608x352, 768X448", "608x352"])
    durations = matrix_test.parse_durations(["4,5", "6"])

    assert resolutions == [(608, 352), (768, 448)]
    assert durations == [4.0, 5.0, 6.0]
    assert [case.label for case in matrix_test.build_matrix(resolutions, durations)] == [
        "608x352 4s",
        "608x352 5s",
        "608x352 6s",
        "768x448 4s",
        "768x448 5s",
        "768x448 6s",
    ]
    assert matrix_test.frame_count_for_duration(4) == 107
    assert matrix_test.frame_count_for_duration(15) == 362


@pytest.mark.parametrize(
    ("parser", "value"),
    [
        (matrix_test.parse_resolutions, "607x352"),
        (matrix_test.parse_resolutions, "1344x800"),
        (matrix_test.parse_resolutions, "31x64"),
        (matrix_test.parse_durations, "3"),
        (matrix_test.parse_durations, "16"),
        (matrix_test.parse_durations, "nan"),
    ],
)
def test_parse_rejects_values_outside_service_constraints(parser, value: str) -> None:
    with pytest.raises(matrix_test.MatrixInputError):
        parser(value)


def test_validate_media_accepts_expected_h3_mp4(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    output = tmp_path / "valid.mp4"
    with av.open(str(output), mode="w") as container:
        video = container.add_stream("mpeg4", rate=24)
        video.width = 32
        video.height = 32
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "stereo"
        for index in range(5):
            frame = av.VideoFrame(width=32, height=32, format="yuv420p")
            frame.pts = index
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        audio_frame = av.AudioFrame.from_ndarray(
            np.zeros((2, 1024), dtype=np.float32), format="fltp", layout="stereo"
        )
        audio_frame.sample_rate = 32000
        audio_frame.pts = 0
        for packet in audio.encode(audio_frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)

    info = matrix_test.validate_media(output, 32, 32, expected_frame_count=5)
    assert info.width == 32
    assert info.height == 32
    assert info.fps == 24
    assert info.frame_count == 5
    assert info.duration_seconds > 0
    assert info.audio_sample_rate == 32000
    assert info.audio_channels == 2
    assert info.audio_layout == "stereo"
    assert info.audio_sample_count > 0


def test_write_reports_contains_csv_and_json(tmp_path: Path) -> None:
    result = matrix_test.MatrixResult(
        index=1,
        width=608,
        height=352,
        duration_seconds=4.0,
        expected_frame_count=107,
        task_id="task-1",
        status="failed",
        error="example failure",
        queued_seconds=1.25,
        running_seconds=2.5,
        total_seconds=3.75,
        file_size_bytes=12,
    )
    csv_path = tmp_path / "reports" / "results.csv"
    json_path = tmp_path / "reports" / "results.json"

    matrix_test.write_reports([result], csv_path, json_path)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["task_id"] == "task-1"
    assert rows[0]["queued_seconds"] == "1.25"
    assert rows[0]["status"] == "failed"
    assert rows[0]["audio_sample_rate"] == ""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["error"] == "example failure"
    assert payload[0]["media"] is None


def test_run_case_submits_polls_and_downloads_sequentially(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload=None, content: bytes = b"") -> None:
            self.status_code = 200
            self._payload = payload
            self.content = content

        def json(self):
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.submissions: list[dict] = []
            self.task_requests: list[str] = []
            self.task_states = iter(
                [
                    {"id": "task-1", "status": "queued", "progress": 0.0},
                    {"id": "task-1", "status": "running", "progress": 0.5},
                    {
                        "id": "task-1",
                        "status": "succeeded",
                        "progress": 1.0,
                        "outputs": [{"filename": "generated.mp4", "download_url": "/download"}],
                    },
                ]
            )

        def post(self, path: str, *, json: dict) -> FakeResponse:
            assert path == "/v1/generations"
            self.submissions.append(json)
            return FakeResponse({"id": "task-1", "frame_count": 107})

        def get(self, path: str) -> FakeResponse:
            if path == "/download":
                return FakeResponse(content=b"valid-mp4")
            self.task_requests.append(path)
            return FakeResponse(next(self.task_states))

    fake = FakeClient()
    expected_media = matrix_test.MediaInfo(608, 352, 24.0, 107, 4.45, 32000, 2, "stereo", 142400)
    monkeypatch.setattr(matrix_test, "validate_media", lambda *args, **kwargs: expected_media)
    result = matrix_test.run_case(
        fake,
        matrix_test.MatrixCase(608, 352, 4.0),
        index=1,
        total_cases=1,
        prompt="test prompt",
        seed=42,
        output_dir=tmp_path,
        poll_interval=0.001,
        timeout=2.0,
    )

    assert result.status == "succeeded"
    assert result.task_id == "task-1"
    assert result.file_size_bytes == len(b"valid-mp4")
    assert result.queued_seconds is not None
    assert result.running_seconds is not None
    assert result.total_seconds is not None
    assert fake.submissions == [
        {
            "mode": "t2v",
            "prompt": "test prompt",
            "duration_seconds": 4.0,
            "width": 608,
            "height": 352,
            "seed": 42,
        }
    ]
    assert fake.task_requests == ["/v1/tasks/task-1"] * 3


def test_timeout_attempts_remote_cancellation(tmp_path: Path) -> None:
    class FakeResponse:
        status_code = 200

        def __init__(self, payload) -> None:
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def post(self, path: str, *, json: dict) -> FakeResponse:
            return FakeResponse({"id": "task-timeout", "frame_count": 107})

        def get(self, path: str) -> FakeResponse:
            return FakeResponse({"id": "task-timeout", "status": "queued", "progress": 0.0})

        def delete(self, path: str) -> FakeResponse:
            self.deleted.append(path)
            return FakeResponse({"id": "task-timeout", "status": "canceled"})

    fake = FakeClient()
    result = matrix_test.run_case(
        fake,
        matrix_test.MatrixCase(608, 352, 4.0),
        index=1,
        total_cases=1,
        prompt="test",
        seed=1,
        output_dir=tmp_path,
        poll_interval=0.001,
        timeout=0.005,
    )

    assert result.status == "failed"
    assert "timed out" in (result.error or "")
    assert fake.deleted == ["/v1/tasks/task-timeout"]


def test_service_readiness_preflight() -> None:
    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "ready", "backend": "native"}

    class Client:
        def get(self, path: str) -> Response:
            assert path == "/readyz"
            return Response()

    assert matrix_test.ensure_service_ready(Client())["backend"] == "native"


def test_dry_run_prints_matrix_without_creating_output(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "should-not-exist"
    code = matrix_test.main(
        [
            "--dry-run",
            "--resolution",
            "608x352,768x448",
            "--duration",
            "4,5",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "matrix: 4 case(s)" in captured.out
    assert "608x352 4s" in captured.out
    assert "768x448 5s" in captured.out
    assert "dry-run: no requests sent" in captured.out
    assert not output_dir.exists()
