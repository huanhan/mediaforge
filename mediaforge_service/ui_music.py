"""Music creation, task status, audio preview and download controls."""
import json
from pathlib import Path

import httpx

from .models import MUSIC_MAX_DURATION_SECONDS, MusicRequest


def add_music_ui(client):
    import gradio as gr

    def check():
        try:
            music = client.catalog()["capabilities"].get("music", {})
            model = music.get("model", "未知模型")
            if music.get("available"):
                return f"音乐模型已就绪 · {model}" + (f"。{music['warning']}" if music.get("warning") else "")
            detail = music.get("reason") or "；".join(music.get("missing", []))
            return f"音乐模型未就绪 · {model}。{detail or '请运行音乐安装脚本。'}"
        except (ValueError, httpx.HTTPError) as exc:
            return f"检查失败：{exc}"

    def submit(prompt, lyrics, duration, seed, cot, abc, cfg_scale):
        try:
            payload = MusicRequest.model_validate(dict(prompt=prompt, lyrics=lyrics,
                duration_seconds=duration, seed=seed, cot=cot,
                abc=(abc or "").strip() or None, cfg_scale=cfg_scale)).model_dump(mode="json")
            result = client.submit_music(payload)
            return (result["id"], f"已排队 · 队列位置 {result['queue_number']} · {result['id']}",
                    None, None, {}, payload, gr.Timer(active=True))
        except (ValueError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    def refresh(task_id, cached):
        if not task_id:
            return (gr.skip(),) * 6
        try:
            task = client.task(task_id)
            state = task["status"]
            labels = dict(queued="排队中", running="音乐生成中", succeeded="已完成", failed="生成失败", canceled="已取消")
            message = labels.get(state, state)
            if task.get("error"):
                message += f"：{task['error']}"
            message += f" · {task_id}"
            # A late refresh or an older browser session can supply another
            # task's cache. A path alone is never evidence that this task was downloaded.
            cache = cached if isinstance(cached, dict) and cached.get("task_id") == task_id else {}
            path = cache.get("path")
            if state == "succeeded" and not path:
                files = client.download([item for item in task.get("outputs", [])
                                         if item["filename"].endswith((".wav", ".json"))])
                path = next((item for item in files if Path(item).suffix == ".wav"), None)
                if not path:
                    raise ValueError("任务完成但没有音频文件")
                metadata_path = next((item for item in files if Path(item).suffix == ".json"), None)
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8")) if metadata_path else {}
                cache = dict(task_id=task_id, path=path, request=metadata.get("request"))
            parameters = cache.get("request")
            return (message, path or None, path or None, cache,
                    parameters if isinstance(parameters, dict) else gr.skip(),
                    gr.Timer(active=state in {"queued", "running"}))
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return (f"查询失败，可刷新重试：{exc}", gr.skip(), gr.skip(),
                    gr.skip(), gr.skip(), gr.Timer(active=False))

    def cancel(task_id):
        if not task_id:
            raise gr.Error("请先提交任务")
        try:
            result = client.task(task_id, cancel=True)
            return f"任务状态：{result['status']}"
        except (ValueError, httpx.HTTPError) as exc:
            raise gr.Error(str(exc)) from exc

    gr.Markdown("YuE2 音乐生成：描述曲风、乐器和情绪，填写歌词生成歌曲。歌词留空请求纯音乐，但不保证完全无人声。输出 48 kHz 立体声 WAV；时长为上限，可能截断乐句。")
    service = gr.Markdown("点击检查音乐服务，确认是否可用。")
    gr.Button("检查音乐服务").click(check, outputs=service)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="曲风与人声描述", lines=3,
                                info="支持中文描述；曲风和男／女声是生成目标，模型不保证遵从。避免同时写入相互矛盾的人声要求。",
                                value="Mandarin pop, acoustic piano, rounded bass and light drums, 88 BPM")
            lyrics = gr.Textbox(label="歌词（留空请求纯音乐）", lines=10, value="[Verse]\n清晨的风轻轻吹过\n街角传来熟悉的歌\n[Chorus]\n让阳光照亮每个梦\n和你走过春夏秋冬")
            duration = gr.Slider(10, MUSIC_MAX_DURATION_SECONDS, value=30, step=1, label="最长时长（秒）")
            seed = gr.Number(value=7, minimum=0, maximum=2147483647, precision=0, label="随机种子")
            with gr.Accordion("乐谱与引导设置", open=False):
                cot = gr.Dropdown(["full", "melody", "off"], value="full", label="乐谱规划（full：旋律和和弦，melody：旋律，off：关闭）")
                abc = gr.Textbox(label="可选 ABC 乐谱", lines=6)
                cfg_scale = gr.Number(value=None, minimum=0, maximum=20, label="引导强度（留空使用模型默认值）")
            start = gr.Button("生成音乐", variant="primary")
        with gr.Column():
            status = gr.Markdown("等待提交")
            audio = gr.Audio(label="音乐试听", type="filepath", interactive=False)
            download = gr.File(label="下载 WAV", interactive=False)
            with gr.Accordion("本次生成参数", open=False):
                parameters = gr.JSON(label="提交参数；完成后读取生成记录")
            with gr.Row():
                update = gr.Button("刷新状态")
                stop = gr.Button("取消任务")
    task = gr.State("")
    cached = gr.State({})
    timer = gr.Timer(3, active=False)
    start.click(submit, [prompt, lyrics, duration, seed, cot, abc, cfg_scale],
                [task, status, audio, download, cached, parameters, timer],
                concurrency_id="music-task", concurrency_limit=1)
    for event in (timer.tick, update.click):
        event(refresh, [task, cached], [status, audio, download, cached, parameters, timer],
              concurrency_id="music-task", concurrency_limit=1)
    stop.click(cancel, task, status)
