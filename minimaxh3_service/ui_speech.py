"""Optional local speech preview tab; all inference goes through the shared API queue."""
from pathlib import Path
import httpx
from .speech_models import SPEAKERS

def add_speech_ui(client):
    import gradio as gr

    def submit(text, speaker, source, instruction, cfg_scale, reference, reference_text):
        try:
            design = source == "design"
            mode = "design" if design else "direction" if instruction.strip() else "clone"
            result = client.submit_speech(dict(text=text, language="Auto",
                mode=mode, instruction=instruction, cfg_scale=cfg_scale,
                **({} if design else {"speaker": speaker}),
                **({"reference_text": reference_text} if not design and reference and reference_text else {})), None if design else reference)
            return result["id"], f"已排队 · {result['id']}", None, {}, gr.Timer(active=True)
        except (ValueError, OSError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(task_id, cached):
        if not task_id:
            return (gr.skip(),) * 4
        try:
            task = client.task(task_id)
            cache = cached if isinstance(cached, dict) and cached.get("id") == task_id else {}
            if task["status"] == "succeeded" and not cache.get("audio"):
                files = client.download([o for o in task.get("outputs", []) if o["filename"].endswith(".wav")])
                if not files:
                    raise ValueError("没有可用的语音文件")
                cache = {"id": task_id, "audio": files[0]}
            label = {"queued":"排队中","running":"语音生成中","succeeded":"已完成","failed":"失败","canceled":"已取消"}.get(task["status"],task["status"])
            if task.get("error"):
                label += "：" + str(task["error"])
            return label, cache.get("audio"), cache, gr.Timer(active=task["status"] in {"queued", "running"})
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return f"查询失败，可刷新重试：{exc}", gr.skip(), gr.skip(), gr.Timer(active=False)

    gr.Markdown("### 本地 Breeze TTS 2 · 非商业使用\n模型与本地生成结果仅限研究及非商业使用，详见 licenses/BREEZE_MODEL_LICENSE.txt。每次生成一个镜头的口播。支持预置音色和自己的录音；生成 WAV，可下载后用于视频。人物口型同步需另行制作。")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="要说的话", lines=4, max_length=1000, value="[笑] 你好，欢迎来到我的频道。")
            source = gr.Dropdown([("参考音色 · 复刻声音", "reference"), ("文字设计 · 无需参考音频", "design")], value="reference", label="声音来源")
            with gr.Column() as references:
                speaker = gr.Dropdown(list(SPEAKERS), value="Serena", label="预置参考音色")
                with gr.Accordion("录制或上传自己的声音", open=False):
                    gr.Markdown("建议 10–15 秒，允许 3–30 秒，单人、清晰、无背景音乐。必须填写与参考录音完全一致的文字。麦克风需要 localhost / HTTPS。")
                    reference = gr.Audio(label="参考录音", sources=["upload", "microphone"], type="filepath")
                    reference_text = gr.Textbox(label="录音实际文字", lines=3)
            instruction = gr.Textbox(label="表演指令 instruction（选填，留空沿用录音）", value="", lines=3)
            cfg_scale = gr.Slider(1, 6, value=4, step=.5, label="CFG scale · 指令引导强度", visible=False)
            source.change(lambda choice: (gr.Column(visible=choice!="design"), gr.Textbox(value="", label="声音描述 instruction（必填）" if choice=="design" else "表演指令 instruction（选填，留空沿用录音）"), gr.Slider(visible=choice=="design")), source, [references,instruction,cfg_scale])
            instruction.change(lambda choice, value: gr.Slider(visible=choice=="design" or bool(value.strip())), [source,instruction], cfg_scale)
            start = gr.Button("生成本镜头配音", variant="primary")
        with gr.Column():
            status = gr.Markdown("等待提交")
            audio = gr.Audio(label="配音试听与下载", type="filepath", interactive=False)
            update = gr.Button("刷新状态")
            stop = gr.Button("取消配音任务")
    task, cached, timer = gr.State(""), gr.State({}), gr.Timer(3, active=False)
    start.click(submit, [text,speaker,source,instruction,cfg_scale,reference,reference_text], [task,status,audio,cached,timer], concurrency_id="speech-task", concurrency_limit=1)
    for event in (timer.tick,update.click):
        event(refresh,[task,cached],[status,audio,cached,timer],concurrency_id="speech-task",concurrency_limit=1)
    def cancel(task_id):
        if not task_id:
            return "没有正在执行的配音任务"
        try:
            return f"任务状态：{client.task(task_id,cancel=True)['status']}"
        except (ValueError,httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc
    stop.click(cancel,task,status)
