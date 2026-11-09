# MiniMax H3 本地生成与超分服务

本项目是一个独立的本地 AI 媒体生成 API，运行在 CUDA GPU 上：

- 视频生成（MiniMax H3，文生视频、首尾帧、参考素材、长视频分段）与视频编辑
- 图片/视频超分（SeedVR2）
- 歌词识别与强制对齐（Qwen3-ASR / Qwen3-ForcedAligner）
- 音乐节奏分析（Beat This!）、视频深度图（Video Depth Anything）、骨骼提取（DWPose）
- 音乐生成（YuE2）、人物分离与背景修复（SAM 2 + ProPainter）
- 本地配音（Breeze TTS 2）
- 基于 Chrome 扩展的公开网页素材采集

服务直接加载模型权重，不启动 ComfyUI Web 服务，也不连接 8188 端口。H3 推理复用固定版本的 ComfyUI 底层 `comfy` 推理内核，Web UI、图执行器和 ComfyUI HTTP API 均不参与运行。

API 和 Gradio 界面共用一个端口（默认 `8000`）：

- 视频界面：`http://127.0.0.1:8000/ui/`
- API 文档：`http://127.0.0.1:8000/docs`
- 就绪检查：`http://127.0.0.1:8000/readyz`

## 目录

- [功能与接口一览](#功能与接口一览)
- [硬件与磁盘空间](#硬件与磁盘空间)
- [快速开始](#快速开始)
- [模型目录](#模型目录)
- [启动与检查](#启动与检查)
- [视频生成](#视频生成)
- [视频编辑](#视频编辑)
- [清晰度与画面比例](#清晰度与画面比例)
- [图片与视频超分](#图片与视频超分)
- [歌词识别与对齐](#歌词识别与对齐)
- [节奏分析](#节奏分析)
- [视频深度](#视频深度)
- [骨骼提取](#骨骼提取)
- [音乐生成](#音乐生成)
- [人物分离与背景修复](#人物分离与背景修复)
- [本地配音](#本地配音)
- [素材采集](#素材采集)
- [OpenAPI 与对接](#openapi-与对接)
- [Claude Code Skill 使用教程](#claude-code-skill-使用教程)
- [配置](#配置)
- [Docker](#docker)
- [测试](#测试)
- [构建发行包](#构建发行包)
- [故障排查](#故障排查)
- [许可](#许可)

## 功能与接口一览

| 功能 | 接口 | 说明 | 额外依赖 |
|---|---|---|---|
| 视频生成 | `POST /v1/generations` | `t2v` / `fl2v` / `ref2v`，可分段生成最长 600 秒 | 基础 H3 权重；`ref2v` 需 Ref2V 权重 |
| 视频编辑 | `POST /v1/edits` | 按源视频时间线做参考条件编辑，可保留原音轨 | Ref2V 权重 |
| 超分 | `POST /v1/upscales` | 图片或视频超分到指定宽高 | SeedVR2 |
| 歌词识别 | `POST /v1/audio/transcriptions` | Qwen3-ASR-1.7B，输出 TXT 与 JSON | `asr` 扩展 |
| 歌词对齐 | `POST /v1/audio/alignments` | 人工确认的歌词逐行对齐到音频 | ForcedAligner 权重 |
| 节奏分析 | `POST /v1/music/analyses` | 拍点、小节第一拍、能量、起音 | `beat` 扩展 |
| 视频深度 | `POST /v1/depths` | 灰度深度 MP4，可选 float32 NPZ | `depth` 扩展 |
| 骨骼提取 | `POST /v1/poses` | DWPose 骨骼可视化 MP4 | `pose` 扩展 |
| 音乐生成 | `POST /v1/music/generations` | YuE2，48 kHz 立体声 WAV | `setup_music` 独立环境 |
| 人物移除 | `POST /v1/removals` | 多目标分离前景、修复背景 | SAM 2、ProPainter |
| 本地配音 | `POST /v1/audio/speech` | Breeze TTS 2，24 kHz 单声道 WAV | `setup_tts` 独立环境 |
| 素材采集 | `POST /v1/collections` | Chrome 扩展读取公开网页，保存 Markdown | 扩展 |

视频和超分共用一个 GPU 队列，不会并行占用同一张卡。连续同类任务复用模型；切换任务类型时先释放上一类模型。音乐、歌词、深度、骨骼、节奏分析等任务也加入同一队列，各自在独立进程中执行。

## 硬件与磁盘空间

- 系统：Windows 10/11 x64 或 Linux，NVIDIA 驱动，PyTorch 2.8+（DynamicVRAM 依赖 2.8 或更高版本）。
- 测试环境：RTX 4070 Ti 12 GB，建议至少 64 GB 系统内存并留足页面文件。
- 视频基础权重（FL2VA + 文本编码器 + VAE）约 43 GB；默认启用的 Turbo LoRA 另需约 2 GB。
- Ref2V 主模型另需约 21 GB。
- SeedVR2 超分权重体积较小，仅在需要超分时下载。
- 速度取决于分辨率、时长、内存和磁盘。默认只允许一个 GPU 任务同时执行。

12 GB 显存必须使用默认的 DynamicVRAM 和 CPU offload（`H3_VRAM_MODE=dynamic`）。

## 快速开始

### 方式一：Windows 免安装包（推荐给普通用户）

1. 完整解压发行 ZIP 到可写目录，不要在压缩包里直接运行。
2. 按 [模型目录](#模型目录) 下载并放置模型。
3. 双击 `check.cmd` 检查 CUDA、运行时和缺少的模型。
4. 双击 `start.cmd` 启动，保持控制台窗口开启。
5. 打开 `http://127.0.0.1:8000/ui/`。

无需安装 Python、Git、uv 或 CUDA Toolkit，只需要已有兼容 CUDA 12.6 的 NVIDIA 驱动。包内不含模型权重、输入或输出。包内的 `使用说明-Windows.txt` 是同一套步骤的简版。

### 方式二：Docker 镜像

前提：Docker Desktop（WSL2 Linux 容器模式）或 Linux 上的 Docker Engine，并已配置 GPU 支持（Linux 需 NVIDIA Container Toolkit）。镜像为 `linux/amd64`。

导入交付包（TAR）并启动，在包含 TAR 和 `compose.yaml` 的目录中执行：

```bash
docker load -i mediaforge-<版本>-linux-amd64.docker.tar
docker compose up -d --no-build --pull never
docker compose logs -f
docker compose down
```

请保留 `--no-build --pull never`，使用导入的本地镜像。Compose 同目录需有 `models/` 和 `data/`（首次运行会创建 `data/`）。模型只读挂载到容器的 `/models`，输入输出保存在宿主 `./data`。`docker compose logs -f` 中按 Ctrl+C 只退出日志跟踪，容器仍在后台运行。

不使用 Compose 时：

```bash
docker run --rm --gpus all --shm-size=8g -p 127.0.0.1:8000:8000 \
  --mount type=bind,source=/absolute/path/models,target=/models,readonly \
  --mount type=bind,source=/absolute/path/data,target=/data \
  mediaforge:<版本>
```

### 方式三：源码安装（开发与自定义）

```powershell
uv sync --extra local --extra dev
uv run mediaforge-setup --accept-license
uv run mediaforge
```

`mediaforge-setup` 会：

1. 只检出固定提交的 ComfyUI 推理内核到 `.runtime/ComfyUI`，不安装 UI 或启动 ComfyUI。
2. 检出固定提交的 SeedVR2 推理后端到 `.runtime/SeedVR2`。
3. 下载 H3 基础权重、Turbo LoRA，以及 SeedVR2 FP8 DiT 和 FP16 VAE 到 `models/`。

`--accept-license` 表示你已阅读并接受 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)。

常用选项：

```powershell
uv run mediaforge-setup --accept-license --with-ref2v            # 额外下载 Ref2V（视频编辑、参考素材生成）
uv run mediaforge-setup --accept-license --without-upscale       # 不装超分后端和权重
uv run mediaforge-setup --accept-license --without-acceleration-lora  # 不下载 Turbo LoRA
```

Gradio 界面需要 `ui` 扩展：

```powershell
uv sync --extra local --extra dev --extra ui
uv run --extra ui mediaforge
```

## 模型目录

模型不随仓库或发行包分发，不会在启动时自动下载。下载后保留以下子目录结构，文件名保持不变，不要解压成 `models/models/`。

H3 下载地址：<https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main>（国内可尝试 <https://hf-mirror.com/Comfy-Org/MiniMax-H3/tree/main>）

SeedVR2 下载地址：<https://huggingface.co/numz/SeedVR2_comfyUI/tree/main>（国内可尝试 <https://hf-mirror.com/numz/SeedVR2_comfyUI/tree/main>）

```text
models/
  diffusion_models/
    minimax_h3_fl2va_pruned_int8_convrot.safetensors        文生视频 / 首尾帧（必需）
    minimax_h3_ref2va_pruned_int8_convrot.safetensors       参考素材 / 视频编辑（可选）
  text_encoders/
    qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors            （必需）
  vae/
    minimax_h3_video_vae_fp16.safetensors                   （必需）
    minimax_h3_audio_vae_fp32.safetensors                   （必需）
  loras/
    minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors         Turbo（默认启用）
    minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors        Ref2V Turbo（可选）
  SEEDVR2/
    seedvr2_ema_3b_fp8_e4m3fn.safetensors                   超分（可选）
    ema_vae_fp16.safetensors                                超分（可选）
```

- 仅文生视频、首尾帧：下载上面标注为必需的 5 个文件，合计约 45 GB。
- 参考素材生成和视频编辑：另加 Ref2V 的 2 个文件。
- 图片/视频超分：另加 `SEEDVR2/` 下的 2 个文件。
- 如果关闭 Turbo（`H3_ENABLE_ACCELERATION_LORA=false`），不需要下载 LoRA 文件。

路径配置：

- Windows 免安装包：`models/` 与 `start.cmd` 同级。也可在 `.env` 中设置 `H3_MODEL_DIR=D:/AI/models` 和 `H3_UPSCALE_MODEL_DIR=D:/AI/models/SEEDVR2`。
- Docker：默认挂载 Compose 同目录的 `models/`，自定义宿主目录设置 `H3_MODELS_PATH`。
- 源码：`models/` 位于项目根目录，可用 `H3_MODEL_DIR` 修改。

其他可选功能的模型路径：

| 功能 | 默认路径 | 环境变量 |
|---|---|---|
| 歌词识别 | `models/asr/Qwen3-ASR-1.7B/` | `H3_ASR_MODEL_PATH` |
| 歌词对齐 | `models/asr/Qwen3-ForcedAligner-0.6B/` | `H3_ALIGNER_MODEL_PATH` |
| 节奏分析 | `models/beat/final0.ckpt` | `H3_BEAT_MODEL_PATH` |
| 视频深度 | `models/depth/video_depth_anything_vits.pth` | `H3_DEPTH_MODEL_PATH` |
| 音乐生成 | `models/music/yue2/`（含 `YuE2-3B` 和 `YuE2-Vae`） | `H3_MUSIC_MODEL_PATH` |
| 本地配音 | `models/tts/Breeze-TTS-2/` | `H3_TTS_MODEL_PATH`（父级目录） |
| 人物移除 | `models/sam2/sam2.1_hiera_tiny.pt` | `H3_SAM2_CHECKPOINT` |

输入文件保存在 `data/input/`，输出保存在 `data/output/`，首次运行时创建，不属于发行包内容。

## 启动与检查

```powershell
uv run mediaforge
Invoke-RestMethod http://127.0.0.1:8000/readyz | ConvertTo-Json -Depth 6
```

- `/healthz`：进程存活。
- `/readyz`：模型和依赖是否就绪。缺模型时返回 503，响应中会说明缺少什么。视频客户端可用 `/readyz?capability=video` 只检查 H3 视频能力，不被可选功能的冷启动探测拖慢。
- `GET /v1/models`：各能力的可用状态（`capabilities.*`），例如 `edit`、`segmentation`、`music`、`speech`、`depth`、`removal`。

服务在第一次对应任务时才加载大模型。视频和超分任务完成后，默认各自空闲 5 秒释放模型，下次任务再重新加载。

**不要使用多个 Uvicorn worker**，否则每个进程都会各自加载模型。同一块 GPU 只运行一个服务实例。

## 视频生成

### Gradio 界面

打开 `http://127.0.0.1:8000/ui/`，点击“检查服务”，选择“视频生成”。界面支持文生视频、首尾帧与参考素材生成，可调宽高、时长、随机种子、输出前缀，支持进度刷新、取消、预览和下载。第一次建议 608×352、4 秒。

- 界面顶部显示服务可用模式与 Turbo 状态。Turbo 开启时会覆盖步数、采样器和调度器；需要自定义这些参数时设置 `H3_ENABLE_ACCELERATION_LORA=false` 并**重启 API 服务**。
- 设置 `H3_API_KEY` 后，界面登录用户名为 `h3`，密码为该 Key。
- 结果副本保存在系统临时目录的 `h3-ui-*` 文件夹，API 原始输出仍在服务的数据目录。

仅需远程连接时，可以使用独立客户端：`uv run --extra ui mediaforge-ui --api-url http://server:8000 --port 7860`，它读取 `H3_BASE_URL` 和 `H3_API_KEY`。

### HTTP 调用

```powershell
$body = @{
  mode = "t2v"
  prompt = "雨夜城市，镜头缓慢向前。Audio: rain and distant traffic."
  duration_seconds = 4
  width = 608
  height = 352
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/generations `
  -ContentType application/json -Body $body
```

返回 HTTP `202`：

```json
{
  "id": "local-task-id",
  "status": "queued",
  "queue_number": 1,
  "effective_duration_seconds": 4.4583333333,
  "frame_count": 107
}
```

- H3 帧数必须满足 `17k+5`；非分段请求的时长为 4–15 秒。
- 模式：`t2v`（文生视频，带 32 kHz 立体声音频）、`fl2v`（首帧、尾帧或首尾帧）、`ref2v`（参考图片、视频、音频，需 Ref2V 权重）。
- 首尾帧：把上传返回的 `file_id` 放入 `first_frame` / `last_frame`，设置 `mode=fl2v`。
- 参考素材：使用 `reference_images`、`reference_videos`、`reference_audios`，分别以 `kind=image|video|audio` 上传。
- 采样参数由服务端管理（见下表），请求不接受 `steps`、`sampler_name`、`scheduler`，传入会返回 HTTP 422。

| 模式 | Turbo 开启（默认） | Turbo 关闭 |
|---|---|---|
| 文生视频 / 首尾帧 | `euler + simple + 8 steps` | `res_multistep + simple + 20 steps` |
| 参考素材 / 视频编辑 | `euler + simple + 4 steps` | `res_multistep + simple + 20 steps` |

`denoise` 仍然有效。

查询和取消：

```text
GET    /v1/tasks/{id}           轮询状态
GET    /v1/tasks/{id}/events    SSE 状态流
DELETE /v1/tasks/{id}           取消（运行中取消需 H3_ALLOW_INTERRUPT_RUNNING=true）
GET    /v1/files/content        下载结果
```

任务成功后，`outputs[].download_url` 指向生成的 MP4。完整示例见 [examples/client.py](examples/client.py)。浏览器原生 `EventSource` 不能设置认证头，有 API Key 时请用 fetch 流或服务端代理订阅 SSE。

### 长视频分段

在生成请求中设置 `segmentation`，可生成最长 600 秒视频：

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

- 超过 15 秒必须开启分段。每段时长 4–15 秒，实际段长向上对齐为 `17k+5` 帧。
- 重叠帧至少 5，向下对齐到同一网格，且必须短于段长。默认 48 帧实际为 39 帧（1.625 秒）。
- 首尾帧 `fl2v` 中，首帧仅用于第一段，尾帧仅用于最后一段。`ref2v` 每段重复使用同一批参考素材。
- 连续性方式为固定 latent 前缀掩码：复制上一段末尾的 latent 作为下一段开头。跨段身份、运动和接缝质量需要按素材验证，不承诺无缝。
- 各段在同一个 GPU 队列任务中串行执行，只保留下一段需要的尾部 latent，逐段编码 MP4。
- 提交响应中会增加 `segment_count` 和 `actual_overlap_frames`。进度中的 `raw_status.current_segment` 从 1 开始，与 `raw_status.segment_count` 一起显示“第 N/M 段”。
- 失败或取消会清理临时 MP4。当前不支持失败续跑或中间段下载。

## 视频编辑

`POST /v1/edits` 按源视频时间线，用参考图重新生成画面。它不是蒙版局部编辑，也不保证背景逐像素不变。

先上传源视频（`kind=video`）和可选的参考图（`kind=image`），再提交：

```json
{
  "prompt": "<Video 1> 是源视频。<Picture 1> 仅作为目标背景。从第一帧到最后一帧，用图片中的场景替换整个原背景，保留原角色外貌、舞蹈动作、手中道具、画面位置和镜头。",
  "source_video": {"file_id": "SOURCE_VIDEO_FILE_ID", "start_seconds": 5, "duration_seconds": 20},
  "reference_images": ["BACKGROUND_IMAGE_FILE_ID"],
  "width": 352,
  "height": 640,
  "segmentation": {"enabled": true, "segment_seconds": 10, "overlap_frames": 48},
  "audio_mode": "preserve_source",
  "seed": 358
}
```

规则：

- 只接受一个 `source_video` 和最多 9 张 `reference_images`，不混入首尾帧或其他参考视频/音频。
- `start_seconds` 默认为 0；`source_video.duration_seconds` 省略时处理到结尾。编辑接口不接受顶层 `duration_seconds` 和 `mode`。源区间时长 4–600 秒。
- 超过 15 秒须开启分段。
- `audio_mode=preserve_source` 按相同源区间保留音轨（解码后重新编码为 AAC），源视频无音轨时输出无音轨。`audio_mode=generate`（默认）由模型生成声音，舞蹈和音乐视频推荐 `preserve_source`。
- 末段不足网格的部分重复末帧补齐，最终裁掉多余帧。
- 角色替换、换背景、纯色背景都用提示词描述即可。纯色不是专用参数，模型不保证输出像素严格等于指定 RGB；需要精确纯色或原样合成时，应使用抠像流程（见[人物分离与背景修复](#人物分离与背景修复)）。
- 视频编辑需要 Ref2V 权重及对应 Turbo LoRA（安装时加 `--with-ref2v`）。

Gradio 的“视频编辑”Tab 与“视频生成”Tab 独立。编辑 Tab 默认启用分段，默认沿用源视频尺寸，可以取消“处理起点之后的全部视频”以填写区间时长。

## 清晰度与画面比例

`POST /v1/generations`（t2v / fl2v / ref2v）和 `POST /v1/edits` 支持 `resolution` 与 `aspect_ratio`。尺寸优先级：

1. 显式 `width` + `height`（必须同时填写，且为 32 的倍数）
2. `resolution` + `aspect_ratio`
3. 接口默认值：生成为 1344×768，编辑沿用源视频尺寸

| 字段 | 可选值 |
|---|---|
| `resolution` | `360p`、`480p`、`540p`、`720p`、`1080p`、`1440p`、`2160p` |
| `aspect_ratio` | `16:9`（默认）、`9:16`、`1:1`、`4:3`、`3:4`、`21:9` |

清晰度指目标短边像素。总像素上限为 1,032,192，超出时等比例缩小，再向下对齐为 32 的倍数。

| 清晰度 | 16:9 | 9:16 |
|---|---|---|
| 360p | 640×352 | 352×640 |
| 480p | 832×480 | 480×832 |
| 540p | 960×512 | 512×960 |
| 720p | 1280×704 | 704×1280 |
| 1080p / 1440p / 2160p | 1344×736 | 736×1344 |

高档位受当前模型上限约束，不是原生生成 1080p 或 4K。需要更大成品尺寸时，编辑后可调用 `/v1/upscales`。可以从 `/v1/models` 的 `capabilities.generation_size` 和 `capabilities.edit_size` 获取支持的选项。

## 图片与视频超分

先上传（`kind=image` 或 `kind=video`），再提交目标宽高：

```powershell
$body = @{
  file_id = "mediaforge/20260822/source-abcd1234.png"
  width = 2336
  height = 3520
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/upscales `
  -ContentType application/json -Body $body
```

图片结果为 PNG，视频结果为 MP4，视频保留原帧率和可用音频。超分与视频共用 GPU 队列。超分使用 SeedVR2 3B FP8，默认 12 GB 显存参数（BlockSwap 16 块、VAE 分块 512）。

## 歌词识别与对齐

### 识别（Qwen3-ASR-1.7B）

```powershell
uv sync --extra local --extra asr --extra beat --extra dev
uv run --no-sync mediaforge-setup-asr
uv run --no-sync mediaforge
```

- 需要 Python 3.13 和 PATH 中的 FFmpeg。模型约 4.1 GB，官方原生 Transformers 格式，不依赖旧版 `qwen-asr` 包。
- 使用流程：`POST /v1/files`（`kind=audio`）→ `POST /v1/audio/transcriptions`，JSON 为 `{"file_id":"...","title":"可选歌名","language":"zh","chunk_seconds":180}` → 轮询任务 → 下载 `outputs` 中的 JSON 和 TXT。
- `language` 支持 `auto`、`zh`（默认）、`en`、`yue`、`ja`、`ko`；`chunk_seconds` 为 30–180。不接受 URL、服务器路径或 `lyrics` 参数。
- 同时提供 Node.js 脚本（无需 npm 依赖，要求 Node.js 20.3+）：

  ```powershell
  $env:H3_BASE_URL = "http://127.0.0.1:8000"
  # 如启用了服务认证，设置 $env:H3_API_KEY
  node scripts/transcribe_audio.mjs "path/to/song.m4a" "outputs/transcriptions/song" ""
  ```

- 结果中的 `chunks[].start/end` 只是输入窗口范围，**不能用于字幕同步**（`timing="chunk_boundaries_only"`）。需要时间戳请使用对齐接口。
- 没有参考歌词时不能宣称识别准确率。伴奏混合、长拖音和合唱会影响结果；输出 `warnings` 中有提示。每段最多 2048 token，达到上限时会标记，可减小 `chunk_seconds` 后重试。

环境变量：`H3_ASR_MODEL_PATH`、`H3_ASR_DEVICE`（默认 `cuda`）、`H3_ASR_CPU_THREADS`（4）、`H3_ASR_MAX_SECONDS`（0 即不限制）、`H3_ASR_TIMEOUT_SECONDS`（1800）。

### 对齐（Qwen3-ForcedAligner-0.6B）

人工确认歌词后再对齐，获得字幕时间轴和逐字时间戳：

```powershell
.venv/Scripts/python.exe -m mediaforge_service.setup_asr --aligner
```

`POST /v1/audio/alignments` 的请求为 `{"file_id":"...","lyrics":"第一行\n第二行","language":"zh"}`，支持 `zh`、`en`、`yue`，任务类型为 `alignment`。

- 输出 `cues[{id,start,end,text}]`（逐行字幕）和 `words[{text,start,end}]`（逐字），时间相对原音乐起点，保留前奏。
- 超过 180 秒的音频会先用 ASR 定位确认歌词所在段落，再用重叠窗口对齐。ASR 结果只用于定位，不覆盖人工歌词。
- 重复段落需要按演唱顺序在歌词中写出，不会自动补词。
- 匹配不足、识别截断或时间重叠时明确报错，不会按字数平均分配时间。

## 节奏分析

用于视频卡点、按节拍切镜和转场。只输出时间数据，不做转写或歌词对齐。

```powershell
uv sync --extra local --extra beat --extra dev
uv run --no-sync mediaforge-setup-beat
```

模型约 81 MB，代码与模型为上游 MIT 许可。FFmpeg 必须在 PATH 中。默认 CPU 推理，API 任务只读取本地权重。

流程：`POST /v1/files`（`kind=audio`）→ `POST /v1/music/analyses`，JSON `{"file_id":"...","title":"可选歌名"}` → 轮询 → 下载 `kind=beat_analysis` 的 JSON。

结果约定：

- 所有时间单位为秒，原点为解码后音轨起点（`time_origin=decoded_audio_start`），保留前奏和静音。
- `beats_seconds`：逐拍时间。`downbeats_seconds`：小节第一拍，不代表最大音量或每次鼓点。
- `tempo_bpm`：局部速度中位数，仅为摘要，无法检测时为 `null`。`tempo_intervals` 为相邻拍点之间的 BPM。
- `energy`：约每 50 ms 一个 `{time,rms,peak}`，线性振幅，未归一化。
- `onsets`：频谱变化起音候选，相对强度 0–1，不是拍点。
- `warnings`：结果仍需试听校准。无节奏音乐、错拍、半速/倍速不能用平均 BPM 强行补点。

剪辑时，从选中的绝对拍点时间计算 `round(time * fps)`，用相邻帧边界相减得到片段长度，不要逐段舍入再累加。字幕时间应跟随实际人声，不吸附到拍点。

环境变量：`H3_BEAT_MODEL_PATH`、`H3_BEAT_DEVICE`（`cpu` 或 `cuda[:index]`）、`H3_BEAT_CPU_THREADS`（4）、`H3_BEAT_MAX_SECONDS`（0 即不限制）、`H3_BEAT_TIMEOUT_SECONDS`（600）。

## 视频深度

`POST /v1/depths` 使用官方 Video Depth Anything Small（vits）进行时序深度估计。

```powershell
uv sync --extra local --extra depth --extra ui --extra dev
uv run --no-sync python -m mediaforge_service.setup_depth
```

已有环境可用 `uv pip install --python .venv/Scripts/python.exe "easydict>=1.13,<2"` 补充依赖。安装脚本固定官方源码和模型版本，不覆盖已有权重。只支持 **Small 相对深度权重**，不能替换为 Base、Large 或 metric 权重。

请求示例：

```json
{"file_id": "mediaforge/20260916/source-xxxx.mp4", "input_size": 392, "max_resolution": 768, "save_raw": true}
```

- 输出无声灰度 MP4，近处更亮，远处更暗。整段共用归一化范围。
- `save_raw=true` 额外返回 NPZ：`depths` 为 `[T,H,W]` float32 **相对逆深度**（越大越近，不是米数），包含 `fps`、精确帧率和可视化范围。
- `input_size` 可选 `252`、`392`（默认）、`518`；`max_resolution` 默认 768，最大 1280，不放大源视频。
- 默认最多 720 帧（24 FPS 下约 30 秒），超限任务失败，不会静默截断。
- 进度为阶段性估计，推理期间可能停在 5%。

环境变量：`H3_DEPTH_RUNTIME_PATH`（`.runtime/Video-Depth-Anything`）、`H3_DEPTH_MODEL_PATH`、`H3_DEPTH_MAX_FRAMES`（720）。官方项目：<https://github.com/DepthAnything/Video-Depth-Anything>

## 骨骼提取

`POST /v1/poses` 使用 DWPose 从源视频提取人体关键点，输出静音骨骼可视化 MP4。

```powershell
uv sync --extra local --extra pose
uv run python -m mediaforge_service.setup_pose
```

```json
{"file_id": "uploads/source.mp4", "output_fps": 12, "start_seconds": 0, "duration_seconds": 10, "resolution": 384}
```

- `duration_seconds` 省略或为 `0` 时处理到原视频结尾。
- 默认在 GPU 上推理（CUDAExecutionProvider），`H3_DEVICE=cpu` 切换为 CPU。
- 源视频应为清晰、单人的人体。动物或卡通角色不被支持，会报告“无可用人体姿态”。

## 音乐生成

音乐使用 [YuE 当前主线的 YuE2](https://github.com/multimodal-art-projection/YuE)（`m-a-p/YuE2-3B` + `YuE2-Vae`）。支持曲风与歌词生成歌曲、空歌词生成纯音乐、乐谱规划、ABC 乐谱输入。输出为 48 kHz 双声道 16-bit WAV，同时提供生成记录 JSON；有乐谱时另提供 `.abc`。

### 安装

YuE2 使用独立的 Python 3.12 环境（固定 PyTorch 2.10.0 / Transformers 4.57.6），不修改 H3 的环境。需先安装 Git、uv 和 NVIDIA 驱动。

```powershell
python -m mediaforge_service.setup_music
python -m mediaforge_service
```

- 安装器通过 uv 创建 `.runtime/yue2-env`，安装 CUDA 12.8 PyTorch 和固定提交的 YuE2，并把模型下载到 `models/music/yue2/`（约 7.8 GB，固定 SHA-256 校验，可续传）。
- 生成阶段强制离线，不会临时下载模型。
- `--runtime-only` 只装环境；`--runtime-dir` 和 `--model-dir` 可指定位置，随后配置 `H3_MUSIC_PYTHON` 和 `H3_MUSIC_MODEL_PATH`。
- 官方推荐 24 GB 显存的 BF16 GPU。默认预算 `H3_MUSIC_MEMORY_BUDGET_GIB=12`，此预算包含上游预留的 2 GiB，并非额外可用的 12 GiB。24 GB 卡可设为 24。

### 请求

```json
{
  "prompt": "Mandarin, warm piano pop, expressive female voice, acoustic piano, rounded bass, light drums, 88 BPM",
  "lyrics": "[Verse]\n清晨的风轻轻吹过\n街角传来熟悉的歌\n[Chorus]\n让阳光照亮每个梦",
  "duration_seconds": 30,
  "seed": 42,
  "cot": "full"
}
```

| 参数 | 说明 |
|---|---|
| `prompt` | 必填。曲风、语言、乐器、情绪、人声与速度描述，支持中文 |
| `lyrics` | 默认空。可用 `[Verse]`、`[Chorus]` 等段落标签 |
| `duration_seconds` | 10–500，默认 30，是输出时长上限 |
| `seed` | 0–2147483647，默认 7 |
| `cot` | `full`（默认，旋律和和弦）、`melody`（仅旋律）、`off`（直接生成） |
| `abc` | 可选 YuE2 格式 ABC 乐谱，与 `cot=off` 不能同时使用 |
| `cfg_scale` | 可选 0–20，省略使用模型默认值 |

提交返回 HTTP `202`，之后轮询 `GET /v1/tasks/{id}`（或订阅 `/events`）。成功后 `outputs` 依次提供 WAV、JSON、可选 ABC 的 `download_url`。队列满时返回 429。

### 需要知道的限制

- **人声与曲风不是硬约束**。`prompt` 中的性别、音色、曲风都是文本生成目标，当前接口没有独立的歌手、音色或人声性别控制，也没有参考歌手音频入口。上游也有相关反馈，不能代替试听。
- **时长不保证自然结束**。时长通过语义预算和音频裁切限制，短请求可能只唱出部分歌词。JSON 中的 `truncated.abc`、`truncated.semantic`、`truncated.audio_cropped` 标记是否截断，请检查这些字段。
- **长歌**：把 `duration_seconds` 设为 180、300 或最多 500，并提供匹配长度的分段歌词。
- 任务记录在进程内，服务重启后历史任务 ID 不可查询。
- 在 Windows 上，原生 GQA 可能退回 math 注意力。本适配器通过展开 K/V 头或分块来控制峰值，策略记录在输出 JSON 的 `config.memory_policy` 中。

### 许可

YuE2 代码为 Apache-2.0。当前 YuE2 权重采用 **CC BY-NC 4.0 加创作者额外许可**，个人创作者与企业商业使用的条款不同，企业商用需按官方条款联系授权。详见 [MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/MODEL_LICENSE)。

版本：YuE2 代码 `1647252d68b70cbead046ffd3f7caf036dde47ee`（`yue2-infer 0.1.6`），YuE2-3B `8f24312f187763b9854adeec87e437e2b03bbff1`，YuE2-Vae `152733a19ad43aa67e367f9b5503ef8075bb5126`。

环境变量：`H3_MUSIC_PYTHON`、`H3_MUSIC_MODEL_PATH`、`H3_MUSIC_DEVICE`（`cuda`）、`H3_MUSIC_OFFLOAD_AR`（`true`）、`H3_MUSIC_TIMEOUT_SECONDS`（3600）。

## 人物分离与背景修复

基于 SAM 2 分割和 ProPainter 修复。可以框选最多 8 个目标，分别输出前景和遮罩，并合并移除后输出干净背景。HTTP 入口为 `POST /v1/removals`，Gradio 位于“人物分离 / 背景修复”区域。

### 安装

需要本项目 `local` 依赖、CUDA 和 PATH 中的 FFmpeg。另需两个上游项目：

```powershell
git clone https://github.com/sczhou/ProPainter.git .runtime/ProPainter
git clone https://github.com/facebookresearch/sam2.git .runtime/sam2
git -C .runtime/ProPainter checkout e870e79321c31b733e2031af5aa2fb1fe3ac7eec
git -C .runtime/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
$env:SAM2_BUILD_CUDA = "0"
$env:PIP_NO_CACHE_DIR = "1"
python -m pip install --no-build-isolation --no-deps .runtime/sam2
python -m pip install --no-build-isolation hydra-core iopath imageio imageio-ffmpeg addict timm scikit-image yapf future
```

- SAM 2 权重：下载 `sam2.1_hiera_tiny.pt` 到 `models/sam2/`，地址 <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt>。
- ProPainter 权重：官方程序首次运行时下载到其 `weights/` 目录，也可从 <https://github.com/sczhou/ProPainter/releases/tag/v0.1.0> 手动下载 `raft-things.pth`、`recurrent_flow_completion.pth`、`ProPainter.pth`。
- 环境变量：`H3_PROPAINTER_PATH`、`H3_SAM2_CHECKPOINT`、`H3_SAM2_CONFIG`。

**ProPainter 使用 S-Lab License 1.0，默认仅限非商业用途**，商业使用需联系上游授权。

### HTTP 请求

```json
{
  "file_id": "mediaforge/20260915/example.mp4",
  "objects": [
    {"id": "person", "frame_index": 0, "box": [0.1, 0.1, 0.4, 0.9]},
    {"id": "dog", "frame_index": 30, "box": [0.5, 0.5, 0.8, 0.9]}
  ],
  "output_mode": "both",
  "foreground_background": "color",
  "background_color": "#00ff00",
  "max_side": 1280,
  "repair_size": 640,
  "max_frames": 600,
  "dilation": 8
}
```

- `box` 为归一化的 `[x1,y1,x2,y2]`。`objects` 为 1–8 个目标，ID 不区分大小写且不能重复，最长 40 字符。旧的单个 `box` 写法仍可用，但不能与 `objects` 同时传。
- `output_mode`：`foreground`、`background` 或 `both`。
- `foreground_background`：`color`（默认，绿幕或指定 `background_color`）或 `transparent`（带 Alpha 的 VP9 WebM）。指定颜色时额外输出 ProRes 4444 MOV。这两个参数只影响分离前景，不改变修复后的背景。
- 默认最长边 1280，不放大源视频，宽高取 8 的倍数。最多 600 帧（可调至 900），超出上限报错，不会静默截断。
- 每个目标从各自的参考帧开始跟踪，之前的帧输出透明，不会倒推。相互遮挡时只输出可见部分。
- 所有目标的遮罩取并集后，只修复一次背景。阴影和倒影可能仍需额外处理。
- 响应包含任务 ID，沿用 `/v1/tasks/{id}` 查询、取消和下载接口。`/v1/models` 的 `capabilities.removal` 显示依赖是否齐全。

输出清单 `objects.json` 记录目标 ID、参考帧、目标框与各文件名的对应关系，各目标文件带 `object-目标ID-` 前缀，`mask.mp4` 为合并遮罩。

### 独立命令行（单目标）

```powershell
.venv/Scripts/python.exe -m mediaforge_service.video_remove `
  --video input.mp4 --output outputs/remove-person-01 `
  --propainter .runtime/ProPainter `
  --sam2-checkpoint models/sam2/sam2.1_hiera_tiny.pt `
  --box 0.2 0.1 0.7 0.95
```

`--box` 为首帧上归一化的 `左 上 右 下`。输出目录必须不存在，包含 `background.mp4`、`masks/`、`foreground/`（透明人物 PNG，不是精细发丝抠像）、`source_frames/`、`repair/` 和 `job.json`。命令行独立运行，不经过服务队列，一次只处理一个对象，不支持中途补点。默认最长边 640、最多 120 帧，可用 `--max-side`、`--max-frames`、`--dilation` 调整。

使用已修正的遮罩：用 `--masks path/to/masks` 替代 `--box` 和 SAM 2 参数。每帧一张 PNG，零填充文件名排序，灰度值大于等于 128 的像素视为待移除区域。

背景从未露出时，修复结果是推测性的。建议先用短片检查遮罩漂移、阴影残留和闪烁。

## 本地配音

Breeze TTS 2 支持预设音色、参考音频克隆、表演指令和音色设计。推理代码为 Apache-2.0，**模型仅限研究和非商业用途，本地生成结果也受限制**，见 [许可](#许可)。

### 安装

请先阅读 [完整模型许可](licenses/BREEZE_MODEL_LICENSE.txt)。用户须自行确认用途并接受许可后执行：

```powershell
.venv/Scripts/python.exe -m mediaforge_service.setup_tts --accept-non-commercial-license
```

安装器固定源码、模型版本和 SHA-256（`mediaforge_service/breeze_tts_models.json`），用 uv 创建独立 CPython 3.11 环境：

- `.runtime/breeze-tts-env`：Torch / Torchaudio 2.9.1 CUDA 12.8。
- `.runtime/breeze-tts-src`：固定版本的推理源码。
- `models/tts/Breeze-TTS-2`：约 7.2 GiB 模型、许可、NOTICE 和接受记录。

需要 FFmpeg 在 PATH 中。不安装 FlashAttention，不启用 Torch compile。模型不打入发行包，推理时不会下载。`--models-only` / `--runtime-only` 可分别准备权重或环境，仍需显式许可参数。安装后需重启 API。

### 接口

```json
{
  "text": "[叹气] 没想到过了这么久，你还记得我的声音。",
  "model": "Breeze-TTS-2",
  "speaker": "Serena",
  "language": "Auto",
  "mode": "clone"
}
```

- `mode=clone`：复刻参考录音的声线。使用 `reference_file_id` 上传参考录音（`kind=audio`，3–30 秒，建议 10–15 秒干净单人语音），并提供与录音完全一致的 `reference_text`。不能循环拼接音频，也不能填写与录音不同的文字。
- `mode=direction`：需要 `instruction` 描述情绪、节奏或表达方式，`cfg_scale` 为 1–6（默认 4）。它不是精确的情绪百分比。
- `mode=design`：只需 `instruction` 描述声音，不需要参考录音，不能与自定义参考录音同时使用。
- 台词中可直接写 `[笑]`、`[叹气]`、`[咳嗽]`、`[清嗓子]`，或英文 `(laugh)`、`(sigh)`、`(cough)`、`(clears throat)`。这些是官方示例，并非完整清单。
- 每个请求最多 1000 字，自动分段。命中音频 token 上限时报错，不会发布被截断的台词。
- 底层保留的可选参数：`speed`（0.5–2，FFmpeg 保音高变速，不是模型原生语速）、`pause_ms`、`volume_db`、`temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`。

预设音色为 9 个历史 ID（默认 Serena），是固定的合成参考 WAV，不是 Breeze 内置说话人，来源和文字见 `voice_presets/manifest.json`。

环境变量：`H3_TTS_MODEL_PATH`（`models/tts` 的父级目录）、`H3_TTS_DEVICE`（`cuda`）。旧的 `H3_TTS_PYTHON` 若已设置，请改为新的独立环境。旧的 IndexTTS 安装器和八维情绪字段已移除，旧客户端需更新，传入旧字段会返回 422。

## 素材采集

通过本机 Chrome 扩展读取公开网页的可见正文、可见评论和已加载的图片/视频响应，保存为 Markdown。扩展位于 `extensions/source-collector`，安装见其中的 README。

前提：本地调用，无需 `H3_API_KEY`。在扩展弹窗中点击“开启”，扩展会连接 `http://127.0.0.1:8000` 领取任务。

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/collections `
  -ContentType application/json -Body '{"url": "https://example.com/articles/1"}'
```

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/collections` | 提交网址，返回 `202` 与任务 `id` |
| GET | `/v1/collections` | 最近的任务列表（`limit` 默认 20） |
| GET | `/v1/collections/{id}` | 状态、错误、Markdown 与媒体下载地址 |
| GET | `/v1/collections/{id}/source` | 下载 `source.md` |
| GET | `/v1/collections/{id}/media/{filename}` | 下载已捕获的响应 |

状态：`queued`（等待扩展领取）、`running`、`completed`、`waiting_login`（需要在该浏览器中登录或验证，然后重新提交同一网址）、`failed`。

限制：只接受 http/https 公开网址（标准端口），拒绝 localhost、内网地址和含账号信息的链接。扩展不读取 cookie、本地存储或隐藏表单，不处理验证码。单个页面最多 12 个媒体响应，单个图片 12 MB、视频 32 MB。任务队列在内存中，服务重启后需重新提交。结果保存在 `data/collections/<id>/`。

## OpenAPI 与对接

服务启动后可访问：

- Swagger UI：`http://127.0.0.1:8000/docs`
- ReDoc：`http://127.0.0.1:8000/redoc`
- OpenAPI JSON：`http://127.0.0.1:8000/openapi.json`

仓库中的静态文档 [openapi.json](openapi.json) 可导入 Swagger Editor、Postman 等 OpenAPI 3.1 工具。修改接口后请重新生成并检查：

```powershell
uv run mediaforge-openapi
uv run mediaforge-openapi --check
```

对接时的通用约定：

- 接入前先检查 `GET /readyz`（HTTP 200 且 `status=ready`）和 `GET /v1/models` 的 `capabilities`。`models_loaded=false` 表示模型尚未加载或已释放，是正常状态。
- 配置 `H3_API_KEY` 后，所有 `/v1/*` 请求需带 `X-API-Key: <Key>` 或 `Authorization: Bearer <Key>`。`/healthz` 与 `/readyz` 不需要认证。
- 不要向接口传本地绝对路径、任意远程 URL、Base64 或 ComfyUI 工作流。先上传（`POST /v1/files`），再使用返回的 `file_id`。
- POST 超时不要直接无限重试，否则可能重复提交。服务没有幂等键。
- 任务状态保存在内存中，服务重启后旧任务 ID 失效。
- 任务状态 `status` 为 `queued`、`running`、`succeeded`、`failed`、`canceled` 或 `unknown`；`progress` 为 0–1 或空，不是剩余时间预测。
- 常见 HTTP 状态：401 认证失败；422 参数错误（旧字段、尺寸不合规、未分段却超长等）；429 队列已满；503 能力未就绪；409 运行中取消未启用。

## 配置

环境变量可写入 `.env`（从 `.env.example` 复制）。系统环境变量优先于 `.env`。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `H3_HOST` | `0.0.0.0` | 监听地址。仅本机访问时可设为 `127.0.0.1`；局域网访问时务必配置 `H3_API_KEY` |
| `H3_PORT` | `8000` | 服务端口 |
| `H3_API_KEY` | 空 | 设置后启用认证；界面用户名为 `h3` |
| `H3_COMFY_CORE_PATH` | `.runtime/ComfyUI` | 固定版本的 H3 推理内核源码 |
| `H3_UPSCALE_RUNTIME_PATH` | `.runtime/SeedVR2` | 固定版本的 SeedVR2 推理后端源码 |
| `H3_MODEL_DIR` | `models` | 模型根目录 |
| `H3_UPSCALE_MODEL_DIR` | `models/SEEDVR2` | SeedVR2 模型目录 |
| `H3_DATA_DIR` | `data` | 输入和输出目录 |
| `H3_DEVICE` | `cuda` | 推理设备（`cuda`、`cuda:0`、`cpu`） |
| `H3_VRAM_MODE` | `dynamic` | `dynamic`、`lowvram`、`novram`、`highvram` 或 `cpu` |
| `H3_RESERVE_VRAM_GB` | `1.0` | 为桌面和其他程序预留的显存 |
| `H3_EAGER_CHUNK_ROWS` | `2048` | H3 CUDA eager INT8/RMS-RoPE 分块行数；0 关闭这套显存优化 |
| `H3_MLP_CHUNK_ROWS` | `0` | MLP 串联分块的每块 token 数；显存不足时可设为 `2048` |
| `H3_EAGER_MEMORY_FRACTION` | `0.7` | 采样和解码阶段的 PyTorch 显存预算比例；1 为不限制 |
| `H3_ENABLE_ACCELERATION_LORA` | `true` | 启用官方 Turbo LoRA 及对应的 8/4 步采样 |
| `H3_FL2VA_ACCELERATION_LORA` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | T2V/FL2V LoRA 文件名 |
| `H3_REF2VA_ACCELERATION_LORA` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | Ref2V LoRA 文件名 |
| `H3_MAX_QUEUE_SIZE` | `8` | 最大排队任务数 |
| `H3_IDLE_RELEASE_SECONDS` | `5.0` | 视频任务完成后，队列空闲多久释放 H3 模型 |
| `H3_UPSCALE_IDLE_RELEASE_SECONDS` | `5.0` | 超分完成后，队列空闲多久释放 SeedVR2 模型 |
| `H3_UPSCALE_DIT_MODEL` | `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | SeedVR2 DiT 文件名 |
| `H3_UPSCALE_VAE_MODEL` | `ema_vae_fp16.safetensors` | SeedVR2 VAE 文件名 |
| `H3_UPSCALE_BLOCKS_TO_SWAP` | `16` | 超分 BlockSwap 块数（12 GB 默认） |
| `H3_UPSCALE_VAE_TILE_SIZE` | `512` | 超分 VAE 分块尺寸 |
| `H3_UPSCALE_SWAP_IO` | `true` | 是否把 SeedVR2 DiT I/O 组件换出到 CPU |
| `H3_ALLOW_INTERRUPT_RUNNING` | `false` | 允许取消运行中的任务（适用于音乐、歌词、深度等） |

修改环境变量后需重启服务。

`H3_MLP_CHUNK_ROWS` 说明：默认 `0` 为计算速度优先；显存不足时设为 `2048` 可降低峰值，实测耗时会增加（同一 8 秒 720P 任务，默认约 10 分 39 秒，`2048` 约 16 分 14 秒）。可尝试 `4096` 或 `8192`，但中间显存占用会上升，需按设备实测。`/readyz` 的 `memory.mlp_chunk_rows` 可确认当前生效值。

## 测试

```powershell
uv run pytest
```

普通测试不下载或加载模型。真实端到端测试需要服务就绪后运行：

```powershell
uv run python examples/client.py
```

成功标准是任务状态为 `succeeded`，下载的 MP4 含 24 FPS 视频和 32 kHz 立体声音轨。

分辨率和时长矩阵测试：

```powershell
uv run python examples/matrix_test.py --dry-run     # 只预览矩阵，不调用服务
uv run python examples/matrix_test.py `
  --resolution 608x352,768x448 --duration 4,6 --seed 42 --output-dir matrix-results
```

分辨率必须为 32 的倍数，时长 4–15 秒。可用 `--fail-fast` 在首个失败时停止，或用 `--timeout` 设置单任务超时。矩阵结果写入 `matrix_results.csv` 和 `matrix_results.json`。

## 构建发行包

以下步骤面向构建机，不适用于下载发行包的用户。每次使用新的输出目录（例如 `dist/release-003`），脚本不会覆盖已有产物。

### 前提

- Windows 构建：Windows x64、Python 3.13、uv、Git 和 x64 Visual C++ 运行库（脚本列出的 DLL，见 `scripts/build_portable.py` 的 `VC_RUNTIME_FILES`）。
- Docker 构建：Python 3.13（用于导出脚本）、运行中的 Docker（Linux 容器模式）。
- 两者都需要联网，建议预留 50 GB 以上空间。构建不包含模型权重。

### Windows 免安装 ZIP

```powershell
python scripts/build_windows.py                                     # 产物在 dist/windows-日期-时间/
python scripts/build_portable.py --output dist/release-003 --runtime-source .runtime  # 复用已有的推理代码
```

`--runtime-source` 要求 `.runtime/ComfyUI` 和 `.runtime/SeedVR2` 的 HEAD 与项目固定提交一致，只导出已提交的固定版本代码。省略该参数则自动获取固定版本。提示缺少 DLL 时，用 `--vc-runtime <目录>` 指定包含 x64 DLL 的目录。

脚本会下载便携 Python 3.13.7、复制运行库、按 `uv.lock` 和哈希安装依赖、复制应用，然后生成文件清单、压缩 ZIP 和 SHA-256 校验文件。不下载模型。

产物：`mediaforge-<版本>-windows-x64.zip` 及 `.zip.sha256`。展开目录也保存在输出目录中，供本机检查。

验证（需要 Windows NVIDIA GPU）：

```powershell
python scripts/verify_portable.py dist/release-003/mediaforge-<版本>-windows-x64.zip --destination dist/verify-003
python scripts/verify_portable.py dist/release-003/mediaforge-<版本>-windows-x64.zip --destination dist/verify-003-models --models <项目根目录>\models
```

验证脚本会校验文件、检查依赖与 CUDA，并临时启动服务做 HTTP 验证，但不执行完整视频生成。交付前还需手动打开 `/ui/`，生成一个短视频并检查预览和下载。

### Docker 镜像

```powershell
python scripts/build_docker.py --output dist/release-003
python scripts/build_docker.py --output dist/release-004 --skip-build   # 仅导出已构建的镜像
```

产物：镜像 TAR、SHA-256 校验文件、`.image.json`、`compose.yaml`、`README.md`、`THIRD_PARTY_NOTICES.md` 和 `使用说明-Docker.txt`。`--skip-build` 只导出本地已有镜像，不包含未重新构建的源码修改，发布新代码时请重新构建。

只构建本地镜像：

```powershell
docker build --platform linux/amd64 --progress plain -t mediaforge:<版本> .
docker image inspect mediaforge:<版本> --format '{{.Os}}/{{.Architecture}}'   # 应为 linux/amd64
```

Docker 构建上下文采用允许清单（`.dockerignore`），只有应用源码、依赖锁、`LICENSE` 和运行时导出脚本会被发送。发行镜像不读取项目 `models/`、`data/`、`outputs/` 或 `.env`。

### 发布新版本

1. 同步修改 `pyproject.toml` 的 `version`、`mediaforge_service/__init__.py` 的 `__version__`，以及 `compose.yaml` 的镜像标签。
2. 只有依赖或版本变化时才执行 `uv lock`。
3. 重新执行构建脚本，不要用 `--skip-build` 代替重新构建。
4. 检查校验文件，试启动，验证 `/ui/`、`/docs` 和一次真实生成。
5. 保留这一版源码、锁文件和产物，便于追溯。

快速回归（不能替代最终包的实际验证）：

```powershell
uv run --extra local --extra ui --extra dev python -m pytest tests/test_packaging.py tests/test_ui.py tests/test_api.py
```

## Claude Code Skill 使用教程

仓库的 `skills/` 目录提供 12 个 Claude Code Skill。其中 11 个 `h3-*` Skill 让 Claude 直接通过本地 API 调用上面的功能，每个都写明了完整流程（检查就绪 → 上传素材 → 提交 → 轮询 → 下载）、参数规则和常见错误处理；`newsnow` 不依赖本地服务，用于查询各平台热榜。

### 1. 安装

Claude Code 只会自动读取 `~/.claude/skills/` 或项目内 `.claude/skills/` 下的 Skill。把 `skills/` 中的目录复制到其中一处即可。以下以项目级安装为例（在项目根目录执行）：

```powershell
New-Item -ItemType Directory -Force .claude/skills | Out-Null
Copy-Item -Recurse -Force skills/h3-* .claude/skills/
```

想在所有项目中使用，则复制到 `~/.claude/skills/`。复制后重新启动 Claude Code，输入 `/` 即可看到 `h3-*` 列表。

### 2. 前提

- 本地服务已启动：`uv run mediaforge`（或 Windows 包的 `start.cmd`），并确认 `http://127.0.0.1:8000/readyz` 可访问。
- Skill 使用 `curl` 调用，Windows 用户请使用 Git Bash 或 WSL 中的 `curl`，或让 Claude 使用 `curl.exe`。
- 若设置了 `H3_API_KEY`，在启动 Claude Code 的环境中同时设置同一个 `H3_API_KEY`，Skill 会自动加上 `X-API-Key` 请求头。
- Skill 只负责调用，不会替你下载模型。缺模型时 Skill 会读取 `/readyz` 的 `missing_models` 并告诉你缺什么。

### 3. Skill 一览

| Skill | 用途 | 需要的功能 |
|---|---|---|
| `h3-generate-video` | 文生视频、首尾帧过渡、参考图/视频/音频生成视频，支持超过 15 秒的分段长视频 | 基础 H3 权重；`ref2v` 需 Ref2V 权重 |
| `h3-edit-video` | 换背景、换角色、动作迁移、按提示词改写已有视频，可指定时间区间 | Ref2V 权重 |
| `h3-upscale` | 图片或视频超分到指定宽高 | SeedVR2 |
| `h3-lyrics` | 歌词转写、人工校对后逐行对齐，输出字幕与逐字时间戳 | ASR 与 ForcedAligner |
| `h3-beat-analysis` | 分析音乐拍点、小节第一拍、能量，用于卡点剪辑 | `beat` 扩展 |
| `h3-depth` | 视频转相对逆深度图（灰度 MP4，可选 NPZ） | `depth` 扩展 |
| `h3-pose` | DWPose 提取骨架视频 | `pose` 扩展 |
| `h3-music-generate` | YuE2 生成纯音乐或带歌词的歌曲 | YuE2 环境与权重 |
| `h3-remove-object` | 抠出或移除视频中的人物/物体（最多 8 个），支持绿幕和透明输出 | SAM 2、ProPainter |
| `h3-speech` | Breeze TTS 2 配音，支持预设音色、参考音频克隆、表演指令 | Breeze 环境与权重（仅非商业用途） |
| `h3-web-collect` | 通过 Chrome 扩展采集公开网页正文、评论为 Markdown | 扩展已开启 |
| `newsnow` | 查询知乎、微博、B 站等平台热榜（不依赖本地服务，需要 Node.js 18+） | 外网访问 NewsNow 接口 |

### 4. 怎么用

直接用自然语言描述需求，Claude 会根据 Skill 的 `description` 自动选择。例如：

- 「用 H3 生成一段 4 秒的雨夜街道视频，竖屏 480p」→ `h3-generate-video`
- 「把 `clip.mp4` 的背景换成 `bg.png`，保留人物动作」→ `h3-edit-video`
- 「把这张图放大到 2336×3520」→ `h3-upscale`
- 「给 `song.m4a` 转写歌词，我校对后再对齐」→ `h3-lyrics`
- 「分析 `bgm.wav` 的节拍，我要按小节切镜」→ `h3-beat-analysis`
- 「把 `dance.mp4` 转成深度图」→ `h3-depth`
- 「提取 `dance.mp4` 的骨架视频」→ `h3-pose`
- 「生成一首 30 秒的中文钢琴流行曲，歌词如下……」→ `h3-music-generate`
- 「抠出视频里的狗，输出透明 WebM」→ `h3-remove-object`
- 「用 `ref.wav` 的声音念这段旁白」→ `h3-speech`
- 「把这个网页的正文保存成 Markdown」→ `h3-web-collect`
- 「看看知乎和微博今天的热搜」→ `newsnow`

也可以直接用斜杠命令调用，例如 `/h3-generate-video`，再补充参数。

Skill 通常会先询问缺少的信息，例如输入文件路径、时长、输出目录。对于耗时任务（视频、音乐、超分），Skill 会轮询状态并在完成后给出下载结果；长任务中可以随时说「取消任务」。

### 5. 使用须知

- **视频生成**：超过 15 秒必须开启分段；`fl2v` 至少提供首帧或尾帧；`ref2v` 的参考素材要先上传。
- **采样参数**：Turbo 默认开启，会覆盖步数、采样器和调度器，即使在提示词中要求也不会生效。
- **素材**：所有素材先通过 `POST /v1/files` 上传，使用返回的 `file_id`，不能直接传本地路径或外部 URL。
- **许可**：Breeze 语音、YuE2 音乐的模型仅限非商业用途（见[许可](#许可)）。`h3-speech` 会提示许可限制；`h3-music-generate` 目前不提示，商用前请自行确认 YuE2 的授权条款。
- **不包含的 Skill**：`skills/` 只包含与本项目相关的 Skill，不含 `image-gen`、`motion-showreel` 等与本项目无关的 Skill。

## 故障排查

| 现象 | 处理 |
|---|---|
| 页面打不开 | 确认控制台或容器仍在运行；查看实际端口（Docker 使用宿主映射端口，容器日志显示的是内部 8000） |
| `/readyz` 返回 503 | 查看返回中的缺失项；双击 `check.cmd`，核对目录、文件名和下载是否完整 |
| 提示缺模型 | 按 [模型目录](#模型目录) 检查，文件名不要改，不要多套一层 `models/` |
| 参数不生效 | Turbo 会覆盖步数、采样器和调度器；需要自定义时设置 `H3_ENABLE_ACCELERATION_LORA=false` 并重启 |
| 显存不足或很慢 | 降低宽高和时长；Docker 或源码可设 `H3_MLP_CHUNK_ROWS=2048`；关闭占用显存的其他程序 |
| `/ui/` 返回 404 | 确认使用的是新包，且安装了 `ui` 扩展（源码模式需 `--extra ui`） |
| 端口占用 | Windows 改 `H3_PORT`；Docker 改 `H3_PUBLISHED_PORT` |
| Docker 无法连接 | 启动 Docker Desktop/Engine，确认为 Linux 容器模式 |
| Docker 找不到 GPU | 检查 NVIDIA 驱动、Docker GPU 支持（Linux 需 NVIDIA Container Toolkit） |
| Docker 修改源码后没生效 | 重新构建镜像，重新创建容器；不要仅导出旧镜像 |
| 提示 Release already exists / Artifact exists | 换一个新的输出目录 |
| 提示 `.runtime` 提交不匹配 | 去掉 `--runtime-source`，让脚本获取固定版本 |
| 提示 Missing x64 redistributable DLLs | 按报错补齐 DLL，或用 `--vc-runtime` 指定目录 |
| 锁文件过期 | 确认依赖修改后执行 `uv lock` |
| 歌词识别、节奏分析等缺少模型 | 运行对应的 `setup_*` 命令，并重启 API 服务 |
| 音乐或配音任务报 `torch_dtype is deprecated` | 这是上游加载接口的提示，真正的失败原因在日志最后一行异常 |

## 许可

本项目的服务端代码、脚本和文档采用 [Apache License 2.0](LICENSE) 发布。

**使用本项目不会授予任何模型权重或生成结果的许可。** 各模型受其自身许可约束，商用前请逐一确认：

| 组件 | 许可 | 说明 |
|---|---|---|
| MiniMax H3 权重 | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | 安装时需接受 |
| Breeze TTS 2 | [BreezeBlue Research and Non-Commercial License 1.1](licenses/BREEZE_MODEL_LICENSE.txt) | 仅限研究和非商业用途，本地生成结果也受限 |
| YuE2 音乐模型 | CC BY-NC 4.0 + 创作者额外条款 | 个人与企业商用条款不同，企业商用需联系授权 |
| ProPainter | S-Lab License 1.0 | 默认非商业用途 |
| ComfyUI 推理内核 | GPL-3.0 | 运行时随发行包分发，见 [第三方组件说明](packaging/THIRD_PARTY_NOTICES.md) |
| Qwen3-ASR | Apache-2.0 | 官方 Transformers 格式 |

- 模型权重不随仓库分发，需用户按各自许可自行获取。
- 其他第三方组件的许可与版本见 [THIRD_PARTY_NOTICES.md](packaging/THIRD_PARTY_NOTICES.md)。Breeze TTS 2 的完整许可文本位于 [licenses/](licenses/)。
