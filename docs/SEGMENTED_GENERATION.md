# 视频编辑与重叠分段生成

使用现有 `POST /v1/generations`。上游上传素材后只提交一次任务；服务内部切片、
传递上一段的重叠 latent、采样、裁掉重复画面并输出一个 MP4。无需 ComfyUI 服务或 JSON 工作流。
重启 API 服务后生效。Gradio `/ui/` 已提供视频编辑与分段控件；
上游流程和场景请求见 [对接指南](INTEGRATION.md)，字段定义见 Swagger `/docs`。

## 编辑源视频

先通过 `POST /v1/files` 上传 `kind=video` 的源视频，以及可选的 `kind=image` 人物参考图。
然后提交：

```json
{
  "mode": "edit",
  "prompt": "将主要人物替换为 <Picture 1> 中的男子，保留 <Video 1> 的舞蹈动作、镜头、节奏和背景。",
  "source_video": {
    "file_id": "上传返回的视频file_id",
    "start_seconds": 0,
    "duration_seconds": 60
  },
  "reference_images": ["上传返回的人物图片file_id"],
  "width": 352,
  "height": 640,
  "segmentation": {
    "enabled": true,
    "segment_seconds": 10,
    "overlap_frames": 48
  },
  "audio_mode": "preserve_source",
  "seed": 358
}
```

`edit` 使用现有 Ref2V 权重和对应 Turbo LoRA，需要安装 `--with-ref2v`。
它把源片段与图片作为参考条件重新生成，不保证背景逐像素不变，也不等同于蒙版局部编辑。
这不是 `qianyi.json` 中融合权重及插件的逐项复刻。

源视频规则：

- `start_seconds` 默认为 0。
- `source_video.duration_seconds` 省略时，使用顶层 `duration_seconds`；两处都省略时处理剩余整段视频。
- 两处同时填写时必须相等。源区间必须存在，时长 4–600 秒；超过 15 秒须明确开启分段。
- 编辑只接受一个 `source_video` 和最多 9 张 `reference_images`，不混入首尾帧或其他参考视频/音频。
- 源视频按输出尺寸、24 FPS 解码；末段不足 H3 网格的部分重复末帧作为参考补齐，最终裁掉多余生成帧。
- `audio_mode=preserve_source` 按相同源区间保留音轨，解码为 32 kHz 立体声后重新编码 AAC；不是音频字节复制。
  源视频没有音轨时，输出无音轨。
- `audio_mode=generate` 为默认值，由模型生成声音，跨段复制音频 latent 尾部并裁掉重复音频。
  生成音频的接缝和音量仍可能变化；舞蹈和音乐视频推荐 `preserve_source`。

## 其他模式的长视频

`t2v`、`fl2v`、`ref2v` 都可增加 `segmentation`。例如：

```json
{
  "mode": "t2v",
  "prompt": "镜头缓慢穿过雨夜街道，保持连续运动。Audio: steady rain.",
  "duration_seconds": 30,
  "width": 608,
  "height": 352,
  "segmentation": {"enabled": true, "segment_seconds": 10, "overlap_frames": 48},
  "seed": 42
}
```

`fl2v` 的首帧仅用于第一段，尾帧仅用于最后一段。有尾帧时，总时长向上对齐 H3 帧数，
确保尾帧不会被最终裁剪丢弃。`ref2v` 每段重复使用同一批参考素材；按源时间线移动切片请用 `edit`。

## 对齐与进度

每段请求时长为 4–15 秒，实际段长向上对齐为 `17k+5` 帧。
重叠请求至少 5 帧，向下对齐到相同网格，且必须短于段长：
默认 48 帧实际为 **39 帧（1.625 秒）**；10 秒段长实际为 **243 帧（10.125 秒）**。
默认窗口因此从第 0、204、408…帧开始。最后一段单独补齐，输出裁到总目标帧数。
除了上述尾帧情况，分段和编辑输出时长四舍五入到 24 FPS 的最近一帧。

提交响应增加：

```json
{
  "id": "任务ID",
  "status": "queued",
  "queue_number": 1,
  "frame_count": 1440,
  "effective_duration_seconds": 60,
  "segment_count": 7,
  "actual_overlap_frames": 39
}
```

继续使用 `GET /v1/tasks/{id}` 或 `/events` 获取总进度；`raw_status.current_segment`
表示当前段（从 1 开始），`raw_status.segment_count` 表示总段数。
取消沿用 `DELETE /v1/tasks/{id}` 和服务现有的运行中取消配置。失败或取消会清理临时 MP4；
当前不提供失败续跑或中间段下载。

## 连续性实现

首版采用 **固定 latent 前缀掩码**：复制上一段视频末尾 latent 到下一段开头；
采样器按当前噪声水平重加噪该区域，并在去噪输出中保留前缀。
生成音频使用相同机制，保留源音轨模式仅延续视频 latent。
这是与当前固定推理内核兼容的方案，未移植 TimelineDirector 的动态 Drift-Control AV 补丁。
跨段身份、运动和接缝质量需要按模型与素材验证，不承诺无缝。

各段在同一个 GPU 队列任务中串行执行，复用模型；只保留下一段需要的尾部 latent，
画面逐段编码，不把整条长视频堆在显存或内存中。分段时长和分辨率仍决定单段显存峰值。

`GET /v1/models` 的 `capabilities.segmentation` 返回能力与时长限制。
未启用分段的原有生成请求仍保持 4–15 秒和原有帧数对齐行为。
