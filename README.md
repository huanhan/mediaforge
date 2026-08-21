# MiniMax H3 本地生成服务

这是一个独立的本地 MiniMax H3 CUDA 视频生成 API。服务直接加载 H3、Qwen3-VL、视频 VAE 和音频 VAE 权重，不安装或启动 ComfyUI Web 服务，也不连接 8188 端口。

推理实现复用固定版本的 ComfyUI 底层 `comfy` 推理内核，以保留 H3 原生节点已经验证的模型结构、联合音视频采样、量化权重和 DynamicVRAM/CPU offload；Web UI、图执行器、ComfyUI HTTP API 和队列均不参与运行。

支持：

- `t2v`：文本生成带 32 kHz 立体声音频的视频
- `fl2v`：首帧、尾帧或首尾帧生成视频
- `ref2v`：参考图片、视频和音频生成视频，需要额外 Ref2V 权重
- 本地素材上传、单 GPU 任务队列、SSE 状态、取消和 MP4 下载

## 硬件和空间

基础 T2V/FL2V 权重约 43 GB，默认启用的 Turbo acceleration LoRA 另需约 2 GB：

| 文件 | 大小 |
|---|---:|
| H3 FL2VA pruned int8 | 21 GB |
| Qwen3-VL-32B NVFP4 AWQ | 16 GB |
| H3 video VAE fp16 | 5.2 GB |
| H3 audio VAE fp32 | 605 MB |

Ref2V 模型另外约 21 GB。RTX 4070 Ti 12GB 必须使用 DynamicVRAM 和 CPU offload，建议至少 64 GB 系统内存并留足页面文件。速度取决于分辨率、时长、系统内存和磁盘速度；默认只允许一个 GPU 任务同时执行。

## 安装

需要 Windows/Linux、Python 3.13、Git、NVIDIA 驱动和 CUDA 可用的 PyTorch 2.8+。DynamicVRAM 依赖 PyTorch 2.8 或更高版本。

```powershell
uv sync --extra local --extra dev
uv run minimaxh3-setup --accept-license
```

第二条命令会：

1. 只检出固定提交的 ComfyUI 推理内核到 `.runtime/ComfyUI`，不安装 UI 或启动 ComfyUI。
2. 从 `Comfy-Org/MiniMax-H3` 下载基础四个权重和 FL2V/T2V Turbo LoRA 到 `models/`。

模型受 [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) 约束；`--accept-license` 表示已阅读并接受该协议。

需要 Ref2V 时执行：

```powershell
uv run minimaxh3-setup --accept-license --with-ref2v
```

Acceleration LoRA 默认下载并启用：T2V/FL2V 使用官方 8-step Turbo LoRA，Ref2V
安装时同时下载 4-step Turbo LoRA。开关启用时，T2V/FL2V 固定使用官方推荐的
`euler + simple + 8 steps`，Ref2V 使用 `euler + simple + 4 steps`；请求中的
`steps`、`sampler_name` 和 `scheduler` 会由 Turbo 配置覆盖。

如需关闭并跳过 LoRA 下载：

```powershell
uv run minimaxh3-setup --accept-license --without-acceleration-lora
$env:H3_ENABLE_ACCELERATION_LORA = "false"
```

关闭时完全保留请求的原有采样参数。

## 启动

```powershell
uv run minimaxh3
```

服务地址为 `http://127.0.0.1:8000`，API 文档为 `http://127.0.0.1:8000/docs`。启动后先检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/readyz | ConvertTo-Json -Depth 6
```

就绪响应应包含 `"backend": "native"`、`"status": "ready"` 和 CUDA 设备信息。服务在第一次生成时才加载大模型；完成任务后默认空闲 5 秒释放模型，下次生成时重新加载。`"models_loaded"` 可用于查看模型权重是否当前驻留。

## 文本生成视频

```powershell
$body = @{
  mode = "t2v"
  prompt = "雨夜城市，镜头缓慢向前。Audio: rain and distant traffic."
  duration_seconds = 4
  width = 608
  height = 352
  seed = 42
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/v1/generations `
  -ContentType application/json `
  -Body $body
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

H3 帧数必须满足 `17k+5`。查询和下载接口：

```text
GET    /v1/tasks/{id}
GET    /v1/tasks/{id}/events
DELETE /v1/tasks/{id}
GET    /v1/files/content
GET    /readyz
```

任务成功后，`outputs[].download_url` 指向本地生成的 MP4。完整调用示例见 [examples/client.py](examples/client.py)。

## 首尾帧和参考素材

先上传素材：

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/files -F "kind=image" -F "file=@first.png"
```

把返回的 `file_id` 放入 `first_frame`/`last_frame`，并设置 `mode=fl2v`。Ref2V 使用 `reference_images`、`reference_videos` 和 `reference_audios`；图片、视频、音频分别以 `kind=image|video|audio` 上传。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `H3_COMFY_CORE_PATH` | `.runtime/ComfyUI` | 固定版本的底层推理内核源码 |
| `H3_MODEL_DIR` | `models` | 本地模型根目录 |
| `H3_DATA_DIR` | `data` | 本地输入和输出目录 |
| `H3_DEVICE` | `cuda` | 推理设备 |
| `H3_VRAM_MODE` | `dynamic` | `dynamic`、`lowvram`、`novram`、`highvram` 或 `cpu` |
| `H3_RESERVE_VRAM_GB` | `1.0` | 为桌面和其他程序预留的显存 |
| `H3_ENABLE_ACCELERATION_LORA` | `true` | 启用官方 Turbo acceleration LoRA 和对应 8/4-step 采样参数 |
| `H3_FL2VA_ACCELERATION_LORA` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | T2V/FL2V acceleration LoRA 文件名 |
| `H3_REF2VA_ACCELERATION_LORA` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | Ref2V acceleration LoRA 文件名 |
| `H3_MAX_QUEUE_SIZE` | `8` | 最大排队任务数 |
| `H3_IDLE_RELEASE_SECONDS` | `5.0` | 完成任务后无新任务多久释放模型 |
| `H3_API_KEY` | 空 | 可选 API Key |

完整配置见 [.env.example](.env.example)。不要使用多个 Uvicorn worker，否则每个进程都会各自加载模型。

## Docker

镜像包含本地推理依赖和固定推理内核，但模型通过卷挂载：

```bash
docker build -t minimaxh3-local .
docker run --rm --gpus all -p 8000:8000 -v /path/to/models:/models -v /path/to/data:/data minimaxh3-local
```

## 测试

```powershell
uv run pytest
```

普通测试不会下载或加载模型。真实端到端测试需要服务就绪后运行：

```powershell
uv run python examples/client.py
```

成功标准是任务状态为 `succeeded`，下载的 `result.mp4` 含 24 FPS 视频和 32 kHz 立体声音轨。

本项目在 RTX 4070 Ti 12GB、PyTorch 2.8.0+cu126 上完成过真实 T2V 冒烟测试：608x352、107 帧、20 步约 6 分 32 秒，输出 24 FPS 视频和 32 kHz 立体声。cu126 会使用量化 eager fallback，速度慢于 cu130 优化内核；具体耗时随提示词、分辨率、时长、磁盘和系统内存变化。

同机启用默认 Turbo LoRA 后，608x352、107 帧、8 步的冷启动 HTTP 实测从提交到首次观察到成功约 3 分 21 秒；轮询粒度使该数字是完成耗时上界。服务返回的 MP4 为 976,035 字节。

### 分辨率和时长矩阵测试

先预览默认矩阵，不调用服务：

```powershell
uv run python examples/matrix_test.py --dry-run
```

运行自定义矩阵（尺寸和时长取笛卡尔积，任务按顺序执行）：

```powershell
uv run python examples/matrix_test.py `
  --resolution 608x352,768x448 `
  --duration 4,6 `
  --seed 42 `
  --output-dir matrix-results
```

工具会先检查 `/readyz`，然后为每个任务下载并校验 MP4，最终写入 `matrix_results.csv` 和 `matrix_results.json`。分辨率必须是 32 的倍数，时长范围为 4-15 秒。可用 `--fail-fast` 在首个失败时停止，或用 `--timeout` 设置单任务超时。
