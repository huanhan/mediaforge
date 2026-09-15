"""Optional Gradio client for the existing HTTP service; no GPU runtime is loaded."""
from __future__ import annotations

import argparse
import json
import ipaddress
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx

from .models import GenerationRequest, VideoEditRequest, RemovalRequest


class VideoClient:
    def __init__(self, base_url: str, api_key: str = "", *, transport=None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.transport = transport

    def connect(self):
        host = httpx.URL(self.base_url).host.rstrip(".").lower()
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost"
        # Mounted Gradio calls its own API: environment proxies must not intercept it.
        return httpx.Client(base_url=self.base_url + "/", headers=self.headers,
                            timeout=httpx.Timeout(120, connect=10), transport=self.transport,
                            trust_env=not local)

    @staticmethod
    def unpack(response):
        if response.is_error:
            raise ValueError(f"服务返回 {response.status_code}：{response.text[:2000]}")
        return response.json()

    def catalog(self):
        with self.connect() as client:
            return self.unpack(client.get("v1/models"))

    def submit(self, parameters: dict, media: dict):
        return self._submit_video(parameters, media, editing=False)

    def submit_edit(self, parameters: dict, media: dict):
        return self._submit_video(parameters, media, editing=True)

    def _submit_video(self, parameters: dict, media: dict, *, editing: bool):
        # Validate mode, canvas, seed and reference counts before uploading anything.
        request_type = VideoEditRequest if editing else GenerationRequest
        request = request_type.model_validate(parameters)
        payload = request.model_dump(mode="json", exclude_none=True)
        with self.connect() as client:
            if editing or request.segmentation.enabled:
                capabilities = self.unpack(client.get("v1/models"))["capabilities"]
                edit_cap = capabilities.get("edit", {})
                if editing and (not edit_cap.get("available") or edit_cap.get("endpoint") != "/v1/edits"):
                    raise ValueError("服务暂不支持视频编辑，请重启新版 API 服务并确认 Ref2V 权重已安装。")
                if request.segmentation.enabled and not capabilities.get("segmentation", {}).get("available"):
                    raise ValueError("服务暂不支持重叠分段，请重启新版 API 服务。")
            def upload(path, kind):
                with Path(path).open("rb") as source:
                    return self.unpack(client.post(
                        "v1/files", data={"kind": kind},
                        files={"file": (Path(path).name, source)},
                    ))["file_id"]

            if request.mode == "fl2v":
                for name in ("first_frame", "last_frame"):
                    if media.get(name):
                        payload[name] = upload(media[name], "image")
            if not editing and request.mode == "ref2v":
                for name, kind in (("reference_images", "image"),
                                   ("reference_videos", "video"),
                                   ("reference_audios", "audio")):
                    ids = [upload(path, kind) for path in media.get(name, [])]
                    payload[name] = ([{"file_id": file_id, "include_source_audio":
                                      media.get("include_source_audio", True)} for file_id in ids]
                                     if kind == "video" else ids)
            if editing:
                if media.get("source_video"):
                    payload["source_video"]["file_id"] = upload(media["source_video"], "video")
                payload["reference_images"] = [upload(path, "image") for path in media.get("reference_images", [])]
            payload = request_type.model_validate(payload).model_dump(mode="json", exclude_none=True)
            return self.unpack(client.post("v1/edits" if editing else "v1/generations", json=payload)), payload

    def submit_music(self, parameters):
        from .models import MusicRequest
        payload = MusicRequest.model_validate(parameters).model_dump(mode="json")
        with self.connect() as client:
            return self.unpack(client.post("v1/music/generations", json=payload))

    def submit_removal(self, video, parameters):
        if not video:
            raise ValueError("请上传源视频。")
        payload = RemovalRequest.model_validate({"file_id": "pending.mp4", **parameters}).model_dump(mode="json")
        with self.connect() as client:
            capability = self.unpack(client.get("v1/models"))["capabilities"].get("removal", {})
            key = "foreground_available" if payload["output_mode"] == "foreground" else "background_available"
            if not capability.get(key):
                raise ValueError("服务尚未就绪：请安装 SAM 2 / ProPainter 依赖和权重，并重启新版服务。")
            with Path(video).open("rb") as source:
                payload["file_id"] = self.unpack(client.post("v1/files", data={"kind": "video"},
                    files={"file": (Path(video).name, source)}))["file_id"]
            return self.unpack(client.post("v1/removals", json=payload))

    def task(self, task_id: str, *, cancel=False):
        with self.connect() as client:
            method = client.delete if cancel else client.get
            return self.unpack(method(f"v1/tasks/{quote(task_id, safe='')}"))

    def download(self, outputs: list[dict]) -> list[str]:
        directory = Path(tempfile.mkdtemp(prefix="h3-ui-"))
        files = []
        with self.connect() as client:
            for index, output in enumerate(outputs):
                # Keep authentication on the configured service; do not follow arbitrary URLs.
                filename = Path(output["filename"].replace("\\", "/")).name
                target = directory / f"{index}-{filename}"
                with client.stream("GET", "v1/files/content", params={
                    "filename": output["filename"], "subfolder": output.get("subfolder", ""),
                    "type": output.get("type", "output"),
                }) as response:
                    if response.is_error:
                        response.read()
                        self.unpack(response)
                    with target.open("wb") as destination:
                        for chunk in response.iter_bytes():
                            destination.write(chunk)
                files.append(str(target))
        return files


def build_ui(client: VideoClient):
    import gradio as gr

    def check_service():
        try:
            catalog = client.catalog()
            capabilities = catalog["capabilities"]
            turbo = capabilities.get("acceleration_lora", False)
            removal = capabilities.get("removal", {})
            return ("连接成功 · 可用模式：" + " / ".join(capabilities["modes"]) +
                    (" · 视频编辑可用" if capabilities.get("edit", {}).get("available") else " · 视频编辑未就绪") +
                    (" · 支持重叠分段" if capabilities.get("segmentation", {}).get("available") else " · 分段不可用") +
                    (" · 人物分离可用" if removal.get("foreground_available") else " · 人物分离未就绪") +
                    (" · 背景修复可用" if removal.get("background_available") else " · 背景修复未就绪") +
                    ("\n\nTurbo 已开启：文生/首尾帧固定 Euler + simple + 8 步，参考/编辑模式固定 4 步。"
                     if turbo else "\n\nTurbo 已关闭：使用服务端基础采样配置。"))
        except (httpx.HTTPError, ValueError) as exc:
            return f"无法连接服务：{exc}"

    def submit(prompt, mode, duration, width, height, seed,
               denoise, prefix, first, last, images, videos, audios, source_audio, ref_size,
               segmented=False,
               segment_seconds=10, overlap_frames=48, resolution="default", aspect_ratio="16:9"):
        try:
            seed = (seed or "").strip()
            prefix = (prefix or "").strip()
            parameters = dict(prompt=prompt, mode=mode, duration_seconds=duration,
                              width=width, height=height, seed=int(seed) if seed else None,
                              denoise=denoise, filename_prefix=prefix or None,
                              ref_image_size=ref_size)
            if (width in (None, 0)) != (height in (None, 0)):
                raise ValueError("请同时填写宽高，或同时留空。")
            if width in (None, 0) and height in (None, 0):
                parameters.pop("width")
                parameters.pop("height")
            parameters.update(resolution=None if resolution == "default" else resolution,
                              aspect_ratio=aspect_ratio)
            parameters["segmentation"] = dict(enabled=segmented, segment_seconds=segment_seconds,
                                               overlap_frames=overlap_frames) if segmented else {"enabled": False}
            media = {}
            if mode == "fl2v":
                media = {"first_frame": first, "last_frame": last}
                parameters.update({key: "pending.png" for key, value in media.items() if value})
            elif mode == "ref2v":
                media = dict(reference_images=images or [], reference_videos=videos or [],
                             reference_audios=audios or [], include_source_audio=source_audio)
                parameters.update(reference_images=["pending.png"] * len(images or []),
                                  reference_videos=[{"file_id": "pending.mp4",
                                                     "include_source_audio": source_audio}] * len(videos or []),
                                  reference_audios=["pending.wav"] * len(audios or []))
            # Show the actual seed so the user can repeat a run even with a random seed.
            if parameters["seed"] is None:
                import secrets
                parameters["seed"] = secrets.randbelow(2**63)
            submitted, payload = client.submit(parameters, media)
            task_id = submitted["id"]
            message = (f"已提交 · {task_id} · 队列位置 {submitted.get('queue_number', '未知')}\n"
                       f"有效时长 {submitted['effective_duration_seconds']:.2f} 秒 · "
                       f"{submitted['frame_count']} 帧 · 种子 {payload['seed']}")
            message += f"\n输出尺寸 {payload['width']}×{payload['height']}"
            if segmented:
                message += (f"\n共 {submitted.get('segment_count', 1)} 段 · "
                            f"实际重叠 {submitted.get('actual_overlap_frames', 0)} 帧")
            return task_id, task_id, message, None, [], payload, gr.Timer(active=True)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(task_id, cached_id, cached_files):
        if not task_id:
            return (gr.skip(),) * 6
        try:
            task = client.task(task_id)
            status = task["status"]
            labels = dict(queued="排队中", running="生成中", succeeded="已完成",
                          failed="生成失败", canceled="已取消", unknown="任务状态未知")
            message = f"{labels.get(status, status)} · {task_id}"
            if task.get("progress") is not None:
                message += f" · {task['progress']:.0%}"
            detail = task.get("raw_status") or {}
            if detail.get("current_segment"):
                message += f" · 第 {detail['current_segment']}/{detail.get('segment_count', '?')} 段"
            if task.get("error"):
                message += "\n" + json.dumps(task["error"], ensure_ascii=False)
            files = cached_files if cached_id == task_id else []
            if status == "succeeded" and not files:
                files = client.download(task.get("outputs", []))
            video = next((path for path in files if Path(path).suffix.lower() in
                          {".mp4", ".webm", ".mov", ".mkv"}), None)
            return message, video, files, task_id, files, gr.Timer(active=status in {"queued", "running"})
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return f"查询失败，可点击刷新重试：{exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False)

    def cancel(task_id):
        if not task_id:
            raise gr.Error("请先提交任务。")
        try:
            task = client.task(task_id, cancel=True)
            return f"取消请求已处理 · {task['status']} · {task_id}"
        except (httpx.HTTPError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    def change_mode(value):
        return gr.Group(visible=value == "fl2v"), gr.Group(visible=value == "ref2v")

    def change_segmentation(enabled, seconds):
        return (gr.Group(visible=enabled),
                gr.Slider(maximum=600 if enabled else 15, value=min(seconds, 600 if enabled else 15)))

    with gr.Blocks(title="MiniMax H3 视频工作台", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# MiniMax H3 视频工作台\n描述画面与声音，生成带音频的视频。")
        service = gr.Markdown("点击检查服务，查看可用模式和 Turbo 状态。")
        check = gr.Button("检查服务", size="sm")
        with gr.Tab("视频生成"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt = gr.Textbox(label="提示词", lines=5, placeholder="雨夜城市，镜头缓慢向前。Audio: rain and distant traffic.")
                    mode = gr.Radio([("文生视频", "t2v"), ("首尾帧生成", "fl2v"),
                                     ("参考素材生成", "ref2v")], value="t2v", label="生成模式")
                    with gr.Group(visible=False) as frames:
                        with gr.Row():
                            first = gr.Image(label="首帧（至少提供一帧）", type="filepath")
                            last = gr.Image(label="尾帧（可选）", type="filepath")
                    with gr.Group(visible=False) as refs:
                        gr.Markdown("需要安装 Ref2V 权重。图片最多 9 张，视频最多 3 个，音频（含视频原声）最多 3 路；合计最多 12 个素材。")
                        images = gr.File(label="参考图片", file_count="multiple", file_types=["image"])
                        videos = gr.File(label="参考视频", file_count="multiple", file_types=["video"])
                        audios = gr.File(label="参考音频", file_count="multiple", file_types=["audio"])
                        source_audio = gr.Checkbox(label="使用参考视频原声", value=True)
                        ref_size = gr.Radio(["match", "max"], value="match", label="参考图片尺寸策略")
                    duration = gr.Slider(4, 15, value=4, step=0.1, label="时长（秒，实际时长按模型帧数对齐）")
                    segmented = gr.Checkbox(value=False, label="启用重叠分段生成（超过 15 秒必选）")
                    with gr.Group(visible=False) as segment_group:
                        segment_seconds = gr.Slider(4, 15, value=10, step=0.1, label="每段时长（秒）")
                        overlap_frames = gr.Slider(5, 345, value=48, step=1, label="重叠帧数（24 FPS）")
                        gr.Markdown("总时长最长 600 秒。默认每段实际 10.125 秒、重叠 39 帧（1.625 秒）；实际重叠须短于段长。自动续接和合并为一个视频。")
                    with gr.Row():
                        resolution = gr.Dropdown([("默认尺寸 1344×768", "default"), "360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"],
                                                 value="360p", label="生成输出清晰度")
                        aspect_ratio = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                                   value="16:9", label="生成画面比例（选择清晰度档位时生效）")
                    with gr.Row():
                        width = gr.Number(value=None, minimum=0, maximum=8192, precision=0, label="生成输出宽度（留空或 0 自动）")
                        height = gr.Number(value=None, minimum=0, maximum=8192, precision=0, label="生成输出高度（留空或 0 自动）")
                    gr.Markdown("尺寸优先级：自定义宽高 → 清晰度和比例 → 默认 1344×768。自定义宽高须同时填写，为 32 的倍数，总像素不超过 1,032,192。高档位会缩小并对齐到模型尺寸限制。")
                    seed = gr.Textbox(label="随机种子", placeholder="留空随机；填入整数可复现")
                    with gr.Accordion("采样与输出参数", open=False):
                        denoise = gr.Slider(0.01, 1, value=1, step=0.01, label="去噪强度")
                        prefix = gr.Textbox(label="输出文件名前缀（可选）", placeholder="例如 gradio/demo")
                    generate = gr.Button("生成视频", variant="primary")
                with gr.Column(scale=2):
                    status = gr.Textbox(label="任务状态", lines=4, interactive=False)
                    task_display = gr.Textbox(label="任务 ID", interactive=False)
                    with gr.Row():
                        update = gr.Button("刷新进度")
                        stop = gr.Button("取消任务")
                    gr.Markdown("生成期间每 3 秒刷新。运行中取消需服务端启用 H3_ALLOW_INTERRUPT_RUNNING。")
                    result = gr.Video(label="生成结果", interactive=False)
                    downloads = gr.File(label="下载结果", file_count="multiple", interactive=False)
                    with gr.Accordion("本次请求参数", open=False):
                        request_json = gr.JSON()
            task_state = gr.State("")
            cached_id = gr.State("")
            cached_files = gr.State([])
            timer = gr.Timer(3, active=False)
            resolution.change(lambda value: gr.Dropdown(interactive=value != "default"), resolution, aspect_ratio, queue=False)
            mode.change(change_mode, mode, [frames, refs], queue=False)
            segmented.change(change_segmentation, [segmented, duration], [segment_group, duration], queue=False)
            check.click(check_service, outputs=service)
            demo.load(check_service, outputs=service)
            generate.click(submit, [prompt, mode, duration, width, height, seed, denoise, prefix,
                                   first, last, images, videos, audios, source_audio, ref_size, segmented, segment_seconds, overlap_frames, resolution, aspect_ratio],
                           [task_state, task_display, status, result, downloads, request_json, timer])
            for event in (timer.tick, update.click):
                event(refresh, [task_state, cached_id, cached_files],
                      [status, result, downloads, cached_id, cached_files, timer],
                      concurrency_id="task-refresh", concurrency_limit=1)
            stop.click(cancel, task_state, status)
        with gr.Tab("视频编辑"):
            from .ui_edit import add_edit_ui
            add_edit_ui(client, refresh, cancel)
        from .ui_music import add_music_ui
        with gr.Tab("音乐生成"):
            add_music_ui(client)
        from .ui_removal import add_removal_ui
        with gr.Tab("人物分离 / 背景修复"):
            add_removal_ui(client)
    return demo


def main():
    parser = argparse.ArgumentParser(description="MiniMax H3 Gradio 视频工作台")
    parser.add_argument("--api-url", default=os.getenv("H3_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui(VideoClient(args.api_url, os.getenv("H3_API_KEY", ""))).queue().launch(
        server_name=args.host, server_port=args.port, share=False,
    )


def mount_ui(app, settings):
    """Mount on the API listener; callbacks call that same listener locally."""
    import gradio as gr

    host = settings.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    client = VideoClient(f"http://{host}:{settings.port}", settings.api_key or "")
    demo = build_ui(client).queue()
    # Do not let the mounted UI bypass the API key via its server-side client.
    auth = ("h3", settings.api_key) if settings.api_key else None
    return gr.mount_gradio_app(
        app, demo, path="/ui", auth=auth, ssr_mode=False,
        auth_message="使用用户名 h3 和服务的 H3_API_KEY 登录。" if auth else None,
        max_file_size=settings.max_upload_bytes,
    )


if __name__ == "__main__":
    main()
