import json
from pathlib import Path

import httpx
import pytest

from minimaxh3_service.ui import VideoClient, build_ui


def callback(demo, name):
    return next(event.fn for event in demo.fns.values() if event.fn.__name__ == name)


@pytest.mark.parametrize("whole", [True, False])
@pytest.mark.parametrize("canvas", [(352, 640), (None, None), (0, 0), (None, 0), (0, None)])
def test_edit_ui_uploads_source_and_background_and_preserves_duration_omission(tmp_path, whole, canvas):
    pytest.importorskip("gradio")
    source, background = tmp_path / "source.mp4", tmp_path / "background.png"
    source.write_bytes(b"video")
    background.write_bytes(b"image")
    uploaded, submitted = [], []

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"capabilities": {"edit": {"available": True, "endpoint": "/v1/edits"}, "segmentation": {"available": True}}})
        if request.url.path == "/v1/files":
            data = request.read()
            kind = "video" if b"source.mp4" in data else "image"
            assert f"\r\n\r\n{kind}\r\n".encode() in data
            uploaded.append(kind)
            return httpx.Response(201, json={"file_id": "uploads/" + kind})
        assert request.url.path == "/v1/edits"
        payload = json.loads(request.content)
        submitted.append(payload)
        return httpx.Response(202, json={"id": "edit-test", "effective_duration_seconds": 30,
                                       "frame_count": 720, "segment_count": 4, "actual_overlap_frames": 39})

    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(handle)))
    event = next(event for event in demo.fns.values() if event.fn.__name__ == "edit_submit")
    # Exercise the actual Number preprocessing which previously rejected cleared inputs.
    processed_canvas = [component.preprocess(value) for component, value in zip(event.inputs[7:9], canvas)]
    result = callback(demo, "edit_submit")("Replace background", str(source), [str(background)],
        2, whole, 30, "preserve_source", *processed_canvas, "358", 20,
        "res_multistep", "simple", 1, None, "match", True, 10, 48)
    assert uploaded == ["video", "image"]
    body = submitted[0]
    assert (body.get("width"), body.get("height")) == tuple(value or None for value in canvas)
    assert "mode" not in body
    assert "duration_seconds" not in body
    assert body["source_video"]["file_id"] == "uploads/video"
    assert body["source_video"]["start_seconds"] == 2
    assert body["source_video"].get("duration_seconds") == (None if whole else 30)
    assert body["reference_images"] == ["uploads/image"]
    assert body["audio_mode"] == "preserve_source"
    assert body["segmentation"]["enabled"] is True
    assert "共 4 段" in result[2] and "39 帧" in result[2]
    assert result[5] == body


def test_old_server_is_rejected_before_upload(tmp_path):
    requests = []
    def handle(request):
        requests.append(request.url.path)
        return httpx.Response(200, json={"capabilities": {"modes": ["t2v", "fl2v", "ref2v"]}})
    client = VideoClient("http://test", transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError, match="重启新版 API"):
        client.submit_edit({"prompt": "x", "source_video": {"file_id": "pending.mp4"}},
                      {"source_video": str(tmp_path / "not-uploaded.mp4")})
    assert requests == ["/v1/models"]


@pytest.mark.parametrize("canvas", [(0, 640), (352, 0), (16, 640), (350, 640)])
def test_edit_ui_still_rejects_invalid_custom_sizes_before_upload(canvas):
    gr = pytest.importorskip("gradio")
    def unexpected(request):
        pytest.fail("invalid dimensions contacted the API")
    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(unexpected)))
    event = next(event for event in demo.fns.values() if event.fn.__name__ == "edit_submit")
    processed = [component.preprocess(value) for component, value in zip(event.inputs[7:9], canvas)]
    with pytest.raises(gr.Error):
        event.fn("x", "source.mp4", [], 0, True, 10, "preserve_source", *processed,
                 "1", 20, "res_multistep", "simple", 1, None, "match")


def test_edit_requires_source_and_validates_reference_limit_before_network():
    gr = pytest.importorskip("gradio")
    def unexpected(request):
        pytest.fail("invalid request contacted the API")
    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(unexpected)))
    submit = callback(demo, "edit_submit")
    args = ["x", None, [], 0, True, 10, "preserve_source", 352, 640, "1", 20,
            "res_multistep", "simple", 1, None, "match"]
    with pytest.raises(gr.Error):
        submit(*args)
    args[1:3] = ["v.mp4", ["p.png"] * 10]
    with pytest.raises(gr.Error, match="9"):
        submit(*args)


def test_mode_controls_and_segment_progress():
    pytest.importorskip("gradio")
    client = VideoClient("http://test", transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "status": "running", "progress": 0.7, "raw_status": {"current_segment": 2, "segment_count": 3}})))
    demo = build_ui(client)
    tabs = [c for c in demo.config["components"] if c["type"] == "tabitem"]
    assert len(tabs) >= 2
    generation = next(e for e in demo.fns.values() if e.fn.__name__ == "submit")
    editing = next(e for e in demo.fns.values() if e.fn.__name__ == "edit_submit")
    assert set(c._id for c in generation.outputs).isdisjoint(c._id for c in editing.outputs)
    updates = callback(demo, "change_mode")("ref2v")
    assert updates[0].visible is False and updates[1].visible is True
    updates = callback(demo, "change_segmentation")(False, 60)
    assert updates[0].visible is False and updates[1].maximum == 15 and updates[1].value == 15
    message = callback(demo, "refresh")("task", "", [])[0]
    assert "第 2/3 段" in message and "70%" in message


def test_segmented_t2v_callback_keeps_edit_fields_out():
    pytest.importorskip("gradio")
    payloads = []
    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"capabilities": {"segmentation": {"available": True}}})
        payloads.append(json.loads(request.content))
        return httpx.Response(202, json={"id": "long", "effective_duration_seconds": 60, "frame_count": 1440})
    demo = build_ui(VideoClient("http://test", transport=httpx.MockTransport(handle)))
    callback(demo, "submit")("x", "t2v", 60, 608, 352, "1", 20, "res_multistep", "simple", 1,
        None, None, None, None, None, None, True, "match", segmented=True)
    assert payloads[0]["duration_seconds"] == 60 and "source_video" not in payloads[0]
