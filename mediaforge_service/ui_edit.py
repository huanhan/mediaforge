"""Independent Gradio video editing form and task state."""
import secrets

import httpx


def add_edit_ui(client, refresh, cancel):
    import gradio as gr

    def edit_submit(prompt, source, images, start, whole, duration, audio,
                    width, height, seed, denoise,
                    prefix, ref_size, segmented=True, segment_seconds=10, overlap_frames=48,
                    resolution="source", aspect_ratio="16:9"):
        try:
            if not source:
                raise ValueError("请上传源视频。")
            interval = dict(file_id="pending.mp4", start_seconds=start)
            if not whole:
                interval["duration_seconds"] = duration
            parameters = dict(prompt=prompt, source_video=interval,
                reference_images=["pending.png"] * len(images or []), audio_mode=audio,
                # Gradio's numeric input can serialize a cleared field as zero.
                width=None if width == 0 else width, height=None if height == 0 else height,
                resolution=None if resolution == "source" else resolution, aspect_ratio=aspect_ratio,
                seed=int(seed.strip()) if seed and seed.strip() else secrets.randbelow(2**63),
                denoise=denoise,
                filename_prefix=(prefix or "").strip() or None, ref_image_size=ref_size,
                segmentation=dict(enabled=segmented, segment_seconds=segment_seconds,
                                  overlap_frames=overlap_frames) if segmented else {"enabled": False})
            submitted, payload = client.submit_edit(parameters,
                dict(source_video=source, reference_images=images or []))
            task_id = submitted["id"]
            message = (f"已提交 · {task_id} · 队列位置 {submitted.get('queue_number', '未知')}\n"
                       f"有效时长 {submitted['effective_duration_seconds']:.2f} 秒 · "
                       f"{submitted['frame_count']} 帧 · 种子 {payload['seed']}")
            if segmented:
                message += (f"\n共 {submitted.get('segment_count', 1)} 段 · "
                            f"实际重叠 {submitted.get('actual_overlap_frames', 0)} 帧")
            if payload.get("width") is not None:
                message += f"\n输出尺寸 {payload['width']}×{payload['height']}"
            else:
                message += "\n输出尺寸：按源视频自动适配"
            return task_id, task_id, message, None, [], payload, gr.Timer(active=True)
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    gr.Markdown("上传源视频进行动作迁移、换背景或风格编辑。参考图片可用于指定角色或背景；编辑结果由模型重新生成。")
    gr.Markdown("透明视频和参考图片会先合成到白色背景再参与编辑；生成的 MP4 不保留透明通道。如需透明结果，请在生成后进行主体分离。")
    with gr.Row():
        with gr.Column(scale=3):
            source = gr.File(label="源视频（必填）", file_types=["video"])
            prompt = gr.Textbox(label="编辑提示词", lines=5,
                placeholder="保持源视频主体和动作，将背景替换为 <Picture 1> 中的场景。")
            images = gr.File(label="角色 / 背景参考图片（可选，最多 9 张）", file_count="multiple", file_types=["image"])
            gr.Markdown("上传背景后，在提示词中用 `<Picture 1>` 指定它；纯色背景可直接写入提示词，例如“背景替换为均匀纯绿色”。")
            start = gr.Number(value=0, minimum=0, label="源视频起点（秒）")
            whole = gr.Checkbox(value=True, label="处理起点之后的全部视频")
            duration = gr.Number(value=10, minimum=4, maximum=600, visible=False, label="截取时长（秒）")
            audio = gr.Radio([("保留源视频声音", "preserve_source"), ("重新生成声音", "generate")],
                             value="preserve_source", label="声音处理")
            segmented = gr.Checkbox(value=True, label="启用重叠分段（超过 15 秒必选）")
            with gr.Group() as segment_group:
                segment_seconds = gr.Slider(4, 15, value=10, step=0.1, label="每段时长（秒）")
                overlap_frames = gr.Slider(5, 345, value=48, step=1, label="重叠帧数（24 FPS）")
                gr.Markdown("默认实际重叠 39 帧（1.625 秒），自动续接并合并为一个视频；最长 600 秒。")
            with gr.Row():
                resolution = gr.Dropdown([("源视频尺寸（默认）", "source"), "360p", "480p", "540p", "720p", "1080p", "1440p", "2160p"],
                                         value="source", label="编辑输出清晰度")
                aspect_ratio = gr.Dropdown(["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                                           value="16:9", label="编辑画面比例（选择清晰度档位时生效）", interactive=False)
            with gr.Row():
                width = gr.Number(value=None, minimum=0, maximum=8192, precision=0, label="编辑输出宽度（留空或 0 自动）")
                height = gr.Number(value=None, minimum=0, maximum=8192, precision=0, label="编辑输出高度（留空或 0 自动）")
            gr.Markdown("尺寸优先级：自定义宽高 → 清晰度和比例 → 源视频尺寸。宽高同时留空或填 0 时使用所选档位；默认沿用源视频尺寸和比例。高档位受总像素上限 1,032,192 限制，会缩小并对齐为 32 的倍数；自定义宽高须同时填写且满足该限制。")
            seed = gr.Textbox(label="编辑随机种子", placeholder="留空随机")
            with gr.Accordion("编辑与输出参数", open=False):
                denoise = gr.Slider(0.01, 1, value=1, step=0.01, label="去噪强度")
                ref_size = gr.Radio(["match", "max"], value="match", label="参考图片尺寸策略")
                prefix = gr.Textbox(label="输出文件名前缀（可选）")
            submit = gr.Button("提交视频编辑", variant="primary")
        with gr.Column(scale=2):
            status = gr.Textbox(label="编辑任务状态", lines=4, interactive=False)
            display = gr.Textbox(label="编辑任务 ID", interactive=False)
            with gr.Row():
                update = gr.Button("刷新编辑进度")
                stop = gr.Button("取消编辑任务")
            result = gr.Video(label="编辑结果", interactive=False)
            downloads = gr.File(label="下载编辑结果", file_count="multiple", interactive=False)
            with gr.Accordion("本次编辑请求参数", open=False):
                request_json = gr.JSON()
    task_state, cached_id, cached_files = gr.State(""), gr.State(""), gr.State([])
    timer = gr.Timer(3, active=False)
    resolution.change(lambda value: gr.Dropdown(interactive=value != "source"), resolution, aspect_ratio, queue=False)
    whole.change(lambda value: gr.Number(visible=not value), whole, duration, queue=False)
    segmented.change(lambda value: gr.Group(visible=value), segmented, segment_group, queue=False)
    submit.click(edit_submit, [prompt, source, images, start, whole, duration, audio,
        width, height, seed, denoise, prefix, ref_size,
        segmented, segment_seconds, overlap_frames, resolution, aspect_ratio],
        [task_state, display, status, result, downloads, request_json, timer])
    for event in (timer.tick, update.click):
        event(refresh, [task_state, cached_id, cached_files],
              [status, result, downloads, cached_id, cached_files, timer],
              concurrency_id="edit-task-refresh", concurrency_limit=1)
    stop.click(cancel, task_state, status)
