# H3 视频编辑与分段生成对接指南

本文面向上游开发者，说明调用流程和场景组合。完整字段、默认值和错误响应以运行中服务的
Swagger `/docs`、OpenAPI `/openapi.json` 为准；仓库静态版本为 [openapi.json](openapi.json)。

## 1. 接入前确认

默认地址 `http://127.0.0.1:8000`，Gradio 页面 `/ui/`。
远程客户端将主机和端口替换为部署地址。安装新代码后需要重启 API 进程。

1. `GET /readyz`：要求 HTTP 200 且 `status=ready`。`models_loaded=false` 表示尚未加载或已释放模型，是正常状态。
2. `GET /v1/models`：视频编辑需要 `capabilities.modes` 包含 `edit`；分段需要 `capabilities.segmentation.available=true`。
3. 编辑依赖 Ref2V 权重及启用 Turbo 时对应的 LoRA；超分不可用不影响视频编辑。

配置 `H3_API_KEY` 后，所有 `/v1/*` 请求（包括上传、查询、SSE 和下载）均需携带以下任一种请求头：

```http
X-API-Key: YOUR_API_KEY
```

或 `Authorization: Bearer YOUR_API_KEY`。`/healthz` 与 `/readyz` 不需要认证。
Gradio 登录用户名为 `h3`，密码为该 API Key。

## 2. 一次任务的完整流程

| 步骤 | 请求 | 上游处理 |
|---|---|---|
| 上传素材 | `POST /v1/files` | multipart，`kind=video` 或 `image`，二进制字段名 `file`；保存返回的 `file_id` |
| 提交生成 | `POST /v1/generations` | JSON，HTTP 202 仅表示入队；立即保存任务 `id` |
| 查询 | `GET /v1/tasks/{id}` | 建议每 3 秒查询，显示总进度与当前段 |
| 下载 | `GET /v1/files/content` | 成功后使用 `outputs` 的文件信息下载；认证与其他请求相同 |
| 取消 | `DELETE /v1/tasks/{id}` | 由用户触发；运行中取消需服务配置允许 |

不要向生成接口传本地绝对路径、任意远程素材 URL、Base64 或 ComfyUI 工作流。
先上传，再使用服务返回的 `file_id`。分段由服务完成，上游不需要手动切片、逐段提交或传递 latent。

## 3. 场景请求

### 上传背景图换背景

先上传源视频和目标背景图，将两个返回值分别填入下面的占位符：

```json
{
  "mode": "edit",
  "prompt": "<Video 1> 是源视频。<Picture 1> 仅作为目标背景，不作为人物参考。从第一帧到最后一帧，用图片中的场景替换整个原背景，保留其树木、建筑、道具的相对位置和色彩，不要渐变或延迟切换。保留原角色外貌、舞蹈动作、手中道具、画面位置、镜头和节奏。",
  "source_video": {"file_id": "SOURCE_VIDEO_FILE_ID"},
  "reference_images": ["BACKGROUND_IMAGE_FILE_ID"],
  "width": 352,
  "height": 640,
  "segmentation": {"enabled": true, "segment_seconds": 10, "overlap_frames": 48},
  "audio_mode": "preserve_source",
  "seed": 358
}
```

角色替换使用相同结构，把参考图换成人物图，并将提示词改成
“用 `<Picture 1>` 中的角色替换 `<Video 1>` 的主要人物，继承原动作，保留背景与道具”。
多张图片按 `reference_images` 顺序对应 `<Picture 1>`、`<Picture 2>` 等，提示词应分别说明用途。

### 只用文字换背景或指定纯色

省略 `reference_images`，通过 `prompt` 描述目标背景即可。例如：

> 从第一帧到最后一帧，将背景替换为均匀纯绿色 #00FF00，无纹理、渐变或背景物体。
> 保留原角色与手中道具，不要转场。

纯色并非专用 API 参数，颜色由模型解释，不保证输出每个像素严格等于指定 RGB。
模型编辑也不会锁住主体像素；需要精确纯色或原样背景合成时，应使用独立抠像合成流程。

### 指定源视频区间

```json
{
  "file_id": "SOURCE_VIDEO_FILE_ID",
  "start_seconds": 5,
  "duration_seconds": 20
}
```

把它作为 `source_video` 的值，表示处理第 5–25 秒。
区间时长省略时，优先使用顶层 `duration_seconds`；两处都省略则处理从起点到结尾的全部视频。
**不要让 SDK 自动补入默认 5 秒。** 两处时长同时填写必须相等。
可用区间为 4–600 秒，超过 15 秒需 `segmentation.enabled=true`。

### 文生长视频

```json
{
  "mode": "t2v",
  "prompt": "镜头持续缓慢穿过雨夜街道。Audio: steady rain.",
  "duration_seconds": 30,
  "width": 608,
  "height": 352,
  "segmentation": {"enabled": true, "segment_seconds": 10, "overlap_frames": 48},
  "seed": 42
}
```

首尾帧 `fl2v` 和参考生成 `ref2v` 同样可开启分段。首帧仅用于第一段，尾帧仅用于最后一段。
`ref2v` 每段重复使用同一批参考素材；按源视频时间线推进请使用 `edit`。

## 4. 上游必须处理的语义

- 编辑只接受一个 `source_video` 和最多 9 张参考图，不混用首尾帧、额外参考视频或音频。
- 输出固定 24 FPS；宽高为 32 的倍数，总像素不超过 1,032,192。352×640 适合竖屏测试。
- 默认 10 秒段长实际为 243 帧（10.125 秒），请求重叠 48 帧实际为 39 帧（1.625 秒）。
  以响应中的 `actual_overlap_frames` 为准，不能简单把每段时长相加。
- `frame_count` 和 `effective_duration_seconds` 描述最终输出，`segment_count` 是总段数。
  60 秒的默认分段请求返回 1440 帧、7 段、39 帧重叠。
- `audio_mode=preserve_source` 保留对应源区间音轨并重编码 AAC；源视频无音轨时输出无音轨。
  `generate` 由模型生成声音。保留源音轨仅适用于编辑模式。
- Turbo 默认开启：参考/编辑固定 Euler + simple + 4 步；文生/首尾帧固定 8 步。
  请求中的步数、采样器和调度器此时会被覆盖。
- 生成与超分共享单 GPU 队列。任务状态保存在服务内存中，重启后不能靠旧任务 ID 继续查询。
  没有幂等键、断点续跑或中间段下载；POST 超时不能直接无限重试，否则可能重复生成。

## 5. 进度、事件与异常

`status` 为 `queued`、`running`、`succeeded`、`failed`、`canceled` 或 `unknown`。
`progress` 可为空，否则是 0–1。进度不是准确的剩余时间估计。
`raw_status.current_segment` 从 1 开始，和 `raw_status.segment_count` 一起显示“第 2/3 段”；
兼容字段缺失，不要把 `raw_status` 当必填数据。

可用 `GET /v1/tasks/{id}/events` 替代轮询。它发送具名事件 `status`，`data` 为同一任务 JSON，
终态后连接关闭。浏览器原生 `EventSource` 不能设置自定义认证头；有 API Key 时使用 fetch 流、
支持请求头的 SSE 客户端或服务端代理。服务未默认配置跨域 CORS，浏览器上游推荐同源代理。

| 情况 | 处理建议 |
|---|---|
| HTTP 401 | 检查 API Key 请求头；不要把凭据放进查询字符串 |
| HTTP 422 | 根据 `detail` 或 `error.message` 展示参数错误，常见为超长但未分段、源区间无效、尺寸不合规 |
| HTTP 503 | 检查就绪状态、Ref2V 权重与 Turbo LoRA |
| HTTP 429 | 队列已满，提示用户稍后重试 |
| HTTP 404 | 任务或素材不存在；服务重启后旧任务 ID 可能失效 |
| HTTP 409（取消） | 运行中取消未启用；停止轮询不等于取消任务 |
| `status=failed` | 展示 `error`，其类型可能是字符串或对象；显存不足应调整分辨率/段长后重新提交 |

API 成功仅证明任务完成；换人、换背景的视觉质量需要预览确认。

## 6. Gradio 操作

1. 打开 `/ui/`，点击“检查服务”，确认列出 `edit` 并支持重叠分段。
2. 选择“视频编辑 / 换角色 / 换背景”，上传源视频和可选参考图。
3. 输入编辑提示词。整段处理保持勾选“处理从起点到视频结束的全部内容”；区间处理取消勾选并填写时长。
4. 选择音轨处理方式。切换到编辑模式会默认启用重叠分段；短视频可手动关闭。
5. 设置输出宽高，点击“生成视频”。任务状态显示实际段数、重叠帧数和当前段，完成后可预览、下载。

参考实测与限制见 [视频编辑测试记录](VIDEO_EDIT_TESTS.md)，帧数与连续性机制见 [分段说明](SEGMENTED_GENERATION.md)。
