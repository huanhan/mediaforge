"""Gradio controls for HTTP video separation tasks; inference stays in the service."""
from pathlib import Path

import httpx


def first_frame(video, frame_index=0):
    import av
    if not video:
        return None
    with av.open(str(video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i != frame_index:
                continue
            image = frame.to_image().convert("RGB")
            image.thumbnail((960, 960))
            return image
    raise ValueError("参考帧超出视频范围，或视频没有可读取的画面。")


def add_target(objects, name, frame_index, box):
    from .models import RemovalObject
    if not box:
        raise ValueError("请先点击目标框的两个角。")
    target = RemovalObject(id=name.strip(), frame_index=frame_index, box=box)
    if len(objects) >= 8:
        raise ValueError("每次最多添加 8 个目标。")
    if target.id.casefold() in {obj["id"].casefold() for obj in objects}:
        raise ValueError("目标名称不能重复。")
    return [*objects, target.model_dump(mode="json")]


def target_rows(objects):
    return [[obj["id"], obj["frame_index"], ", ".join(f"{v:.3f}" for v in obj["box"])] for obj in objects]


def selection_click(image, points, index):
    from PIL import ImageDraw
    if image is None:
        raise ValueError("请先上传视频。")
    x, y = index
    width, height = image.size
    point = (min(1., max(0., x / width)), min(1., max(0., y / height)))
    points = [point] if len(points or []) != 1 else [points[0], point]
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    box = None
    if len(points) == 1:
        draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
    else:
        a, b = points
        box = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
        if box[0] == box[2] or box[1] == box[3]:
            raise ValueError("两个角不能位于同一条水平线或垂直线上，请重新选择。")
        draw.rectangle((box[0]*width, box[1]*height, box[2]*width, box[3]*height), outline="red", width=3)
    return canvas, points, box


def add_removal_ui(client):
    import gradio as gr

    def load_removal_frame(video):
        try:
            image = first_frame(video)
            return image, image, [], None, "请点击目标框的第一个角，再点击对角。", video, 0, 0, [], [], "object-1"
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def select_removal_box(image, points, evt: gr.SelectData):
        try:
            canvas, points, box = selection_click(image, points, evt.index)
            return canvas, points, box, "目标框已选好，可以开始处理。" if box else "请点击目标框的对角。"
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def load_reference(video, index):
        try:
            index = int(index)
            if not 0 <= index <= 899:
                raise ValueError("参考帧须在 0 到 899 之间。")
            image = first_frame(video, index)
            return image, image, [], None, "参考帧已加载，请圈选目标。", video, index
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    def add_removal_target(objects, name, index, loaded_index, box, video, selected_video, image):
        try:
            if video != selected_video or index != loaded_index:
                raise ValueError("请先加载当前视频的参考帧。")
            objects = add_target(objects, name, int(index), box)
            return objects, target_rows(objects), image, [], None, f"已添加 {len(objects)} 个目标。", f"object-{len(objects)+1}"
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def submit_removal(video, selected_video, box, output_mode, max_side, repair_size, max_frames, dilation,
                       objects=None, index=0, loaded_index=0, foreground_background="color", background_color="#00ff00"):
        try:
            if not video or video != selected_video:
                raise ValueError("请等待当前视频首帧加载完毕，再选择目标。")
            if index != loaded_index:
                raise ValueError("请先加载当前参考帧。")
            if objects and box:
                raise ValueError("当前框选尚未加入列表，请点击添加目标或重新选择清除它。")
            if not objects:
                if not box:
                    raise ValueError("请至少添加一个目标。")
                objects = add_target([], "object-1", int(index), box)
            result = client.submit_removal(video, dict(objects=objects, output_mode=output_mode,
                foreground_background=foreground_background, background_color=background_color,
                max_side=int(max_side), repair_size=int(repair_size), max_frames=int(max_frames), dilation=int(dilation)))
            return (result["id"], f"已排队 · {result['id']} · 队列位置 {result['queue_number']}",
                    None, None, [], "", [], gr.Timer(active=True), gr.Dropdown(choices=[], value=None))
        except (ValueError, OSError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh_removal(task_id, cached_id, cached_files, selected=None):
        if not task_id:
            return (gr.skip(),) * 8
        try:
            task = client.task(task_id)
            state = task["status"]
            labels = dict(queued="排队中", running="处理中", succeeded="已完成", failed="处理失败", canceled="已取消")
            message = f"{labels.get(state, state)} · {task_id}"
            if task.get("progress") is not None:
                message += f" · {task['progress']:.0%}"
            if task.get("error"):
                message += f"\n{task['error']}"
            files = cached_files if task_id == cached_id else []
            if state == "succeeded" and not files:
                files = client.download(task.get("outputs", []))
            previews = [p for p in files if p.endswith(("foreground-preview.mp4", "foreground-preview.webm"))]
            foreground = selected if selected in previews else next(iter(previews), None)
            background = next((p for p in files if p.endswith("background.mp4")), None)
            choices = [(Path(p).stem.split("-object-")[-1].removesuffix("-foreground-preview"), p) for p in previews]
            return message, foreground, background, files, task_id, files, gr.Timer(active=state in {"queued", "running"}), gr.Dropdown(choices=choices, value=foreground)
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return f"查询失败，可刷新重试：{exc}", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.Timer(active=False), gr.skip()

    def cancel_removal(task_id):
        if not task_id:
            raise gr.Error("请先提交任务。")
        try:
            result = client.task(task_id, cancel=True)
            return f"取消请求已处理 · {result['status']}"
        except (ValueError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Accordion("人物分离 / 背景修复", open=True):
        gr.Markdown("上传视频，选择参考帧，点击两个对角圈选目标并添加到列表。最多 8 个目标，分别导出前景，合并移除后修复背景。")
        with gr.Row():
            with gr.Column():
                video = gr.File(label="待分离视频", file_types=["video"], type="filepath")
                with gr.Row():
                    frame_index = gr.Number(value=0, minimum=0, maximum=899, precision=0, label="参考帧（从 0 开始）")
                    load_frame = gr.Button("加载参考帧")
                selector = gr.Image(label="参考帧：点击两个角圈住目标", type="pil", interactive=False)
                selection_status = gr.Markdown("请先上传视频。")
                reset = gr.Button("重新选择目标", size="sm")
                target_name = gr.Textbox(value="object-1", label="目标名称（字母、数字、中文、下划线或短横线）")
                add = gr.Button("添加目标")
                targets_table = gr.Dataframe(headers=["目标", "起始帧", "目标框"], datatype=["str", "number", "str"],
                                             value=[], interactive=False, label="已添加目标")
                with gr.Row():
                    undo = gr.Button("移除最后一个目标", size="sm")
                    clear = gr.Button("清空目标列表", size="sm")
                output_mode = gr.Radio([("分离前景", "foreground"), ("只保留背景", "background"),
                                        ("前景和背景", "both")], value="foreground", label="输出内容")
                with gr.Group() as foreground_options:
                    foreground_background = gr.Radio([("指定颜色（默认绿幕）", "color"), ("透明", "transparent")],
                                                       value="color", label="分离前景的背景")
                    background_color = gr.ColorPicker(value="#00ff00", label="背景颜色")
                with gr.Accordion("处理参数", open=False):
                    max_side = gr.Slider(256, 1920, value=1280, step=8, label="输出最长边（不放大源视频）")
                    repair_size = gr.Slider(256, 768, value=640, step=8, label="背景修复精度（最长边）")
                    max_frames = gr.Slider(2, 900, value=600, step=1, label="最多处理帧数（超出时报错，不截断）")
                    dilation = gr.Slider(0, 40, value=8, step=1, label="移除区域扩展（修复尺寸下的像素）")
                gr.Markdown("目标从各自参考帧开始跟踪，之前为透明。相互遮挡时只拆出可见部分；未做精细发丝处理。背景补全可能有纹理变化，阴影或倒影可能需要额外处理。")
                start = gr.Button("开始分离 / 修复", variant="primary")
            with gr.Column():
                status = gr.Textbox(label="分离任务状态", interactive=False, lines=3)
                task_id = gr.Textbox(label="分离任务 ID", interactive=False)
                with gr.Row():
                    refresh = gr.Button("刷新分离进度")
                    cancel = gr.Button("取消分离任务")
                preview_choice = gr.Dropdown(choices=[], label="选择前景预览", interactive=True)
                foreground = gr.Video(label="分离前景（绿幕 / 自定义底色 / 透明）", interactive=False)
                background = gr.Video(label="背景修复结果", interactive=False)
                downloads = gr.File(label="下载分离结果（MOV / WebM / MP4 / 遮罩）", file_count="multiple", interactive=False)
                gr.Markdown("指定颜色输出 MP4；透明模式输出带 Alpha 的 WebM，不烧录棋盘格。两种模式都额外提供透明 MOV 用于剪辑。运行中取消需服务端启用 H3_ALLOW_INTERRUPT_RUNNING。")
        original, points, box = gr.State(None), gr.State([]), gr.State(None)
        selected_video = gr.State(None)
        loaded_index = gr.State(0)
        objects = gr.State([])
        cached_id, cached_files = gr.State(""), gr.State([])
        timer = gr.Timer(3, active=False)
        foreground_background.change(lambda mode: gr.ColorPicker(visible=mode == "color"),
                                     foreground_background, background_color, queue=False)
        output_mode.change(lambda mode: gr.Group(visible=mode != "background"), output_mode,
                           foreground_options, queue=False)
        video.change(load_removal_frame, video, [selector, original, points, box, selection_status, selected_video,
                                               frame_index, loaded_index, objects, targets_table, target_name])
        load_frame.click(load_reference, [video, frame_index], [selector, original, points, box, selection_status, selected_video, loaded_index])
        add.click(add_removal_target, [objects, target_name, frame_index, loaded_index, box, video, selected_video, original],
                  [objects, targets_table, selector, points, box, selection_status, target_name])
        undo.click(lambda items: (items[:-1], target_rows(items[:-1])), objects, [objects, targets_table])
        clear.click(lambda: ([], []), outputs=[objects, targets_table])
        selector.select(select_removal_box, [original, points], [selector, points, box, selection_status])
        reset.click(lambda image: (image, [], None, "请重新点击两个角。"), original,
                    [selector, points, box, selection_status])
        start.click(submit_removal, [video, selected_video, box, output_mode, max_side, repair_size, max_frames, dilation,
                                    objects, frame_index, loaded_index, foreground_background, background_color],
                    [task_id, status, foreground, background, downloads, cached_id, cached_files, timer, preview_choice])
        for event in (timer.tick, refresh.click):
            event(refresh_removal, [task_id, cached_id, cached_files, preview_choice],
                  [status, foreground, background, downloads, cached_id, cached_files, timer, preview_choice],
                  concurrency_id="removal-refresh", concurrency_limit=1)
        cancel.click(cancel_removal, task_id, status)
        preview_choice.change(lambda path: path, preview_choice, foreground)
