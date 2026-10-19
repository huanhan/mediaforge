# 本地 Qwen3-ASR 歌词识别

`POST /v1/audio/transcriptions` 使用 **Qwen3-ASR-1.7B** 从上传音频识别歌词/语音，返回原始识别文本 TXT 和 JSON。Node.js 只调用 HTTP，模型和 Python 环境均在 minimaxh3 项目内。

识别接口只输出文本；人工修改歌词后可调用下方的独立强制对齐接口。画面卡点使用 [节奏分析 API](BEAT_ANALYSIS.md)，字幕时间跟随人声。歌名 `title` 仅保存为元数据，不用于猜测或改写歌词。

## 安装与运行

需要 Python 3.13、PATH 中的 FFmpeg。默认 NVIDIA CUDA 推理，也可配置 CPU（更慢）。模型权重约 4.1 GB，推理显存还取决于音频分段长度。使用官方原生 Transformers 格式，无需安装会固定旧版 Transformers 的 `qwen-asr` 包。

```powershell
uv sync --extra local --extra asr --extra beat --extra dev
uv run --no-sync minimaxh3-setup-asr
uv run --no-sync minimaxh3
```

保留部署中原有的其他 `--extra`（如 `ui`、`pose`）。已有匹配依赖的环境可直接执行 `.venv/Scripts/python.exe -m minimaxh3_service.setup_asr`。代码变更后重启 API 服务。

- 官方模型：<https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf>，Apache-2.0 许可。
- 固定版本：`bcd2b5b7f32b480ab5790554cfa8347f246a14f3`。
- 下载路径：`models/asr/Qwen3-ASR-1.7B/`，与 H3 权重分开存放；模型不进 Git。
- 首次安装需要联网；请求处理使用本地权重，不上传音频到外部模型服务、不触发自动下载。
- 与视频、节奏分析共用任务队列。切换任务类型前释放上一类模型，每次 ASR 使用独立进程，完成、取消、超时后释放显存。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| H3_ASR_MODEL_PATH | models/asr/Qwen3-ASR-1.7B | 模型目录 |
| H3_ASR_DEVICE | cuda | cpu 或 cuda[:index] |
| H3_ASR_CPU_THREADS | 4 | CPU 线程数 |
| H3_ASR_MAX_SECONDS | 900 | 音频时长上限，超限报错 |
| H3_ASR_TIMEOUT_SECONDS | 1800 | 单任务解码与推理总超时 |

`GET /readyz` 的 `transcription` 及 `GET /v1/models` 的同名 capability 报告模型文件/依赖是否齐全。该检查不加载权重或探测可用显存；CUDA/显存故障通过任务 `error` 返回。不需要安装视频权重才能执行识别。

## HTTP 调用

1. `POST /v1/files` 上传，multipart 字段 `kind=audio`、`file`，支持现有 M4A、MP3、WAV 等扩展。
2. 提交 JSON 到 `POST /v1/audio/transcriptions`：

```json
{"file_id":"上传接口返回的 file_id","title":"可选歌名","language":"zh","chunk_seconds":180}
```

返回 HTTP 202：`{"id":"任务编号","status":"queued","queue_number":1}`。

`language` 支持 `auto`、`zh`（默认）、`en`、`yue`、`ja`、`ko`；`chunk_seconds` 为 30–180 秒（默认 180）。不接受 URL、服务器绝对路径或 `lyrics` 参数。需要识别其他语言时可用 `auto`。

3. 轮询 `GET /v1/tasks/{id}` 或监听 `/events`。成功时 `kind=transcription`，`outputs` 分别包含 JSON 和 TXT，读取各自的 `download_url`。
4. `DELETE /v1/tasks/{id}` 取消；运行中取消需要 `H3_ALLOW_INTERRUPT_RUNNING=true`。沿用 `X-API-Key` 或 Bearer 认证。

Node.js 20.3+ 的完整调用脚本（无需 npm 安装依赖）：

```powershell
$env:H3_BASE_URL = "http://127.0.0.1:8000"
# 如启用了服务认证，设置 $env:H3_API_KEY。
node scripts/transcribe_audio.mjs "C:/workspace/webws/videocreater/Suno_b96cc9c5.m4a" "outputs/transcriptions/Suno_b96cc9c5" ""
```

脚本负责上传、查询、下载至 `qwen3-asr.json` / `qwen3-asr.txt`；失败返回非零退出码，终止或超时会尝试取消服务端任务。指定同一输出目录会覆盖其中这两个文件。

## 结果含义

- `text`：模型识别的完整文本，保留模型给出的标点；不根据歌名补写歌词。
- `duration_seconds`：16 kHz 解码样本时长，包含前奏/静音，可能与压缩容器时长略有差异。
- `chunks`：每段输入的 `start`、`end`、`text`、`language`、`token_limit_reached`。
- `timing="chunk_boundaries_only"`：**start/end 只是输入窗口范围，不能用于字幕同步。**
- `model`、`model_revision`、`source_file_id`、`title`：来源和版本信息。
- `elapsed_seconds`：识别子进程耗时，含模型加载，不含排队/FFmpeg 解码。
- `warnings`：自动识别可能出错、分段边界可能影响歌词、输出上限等提示。

超过分段时长的音频按连续、不重叠窗口处理，保留完整采样时间范围。每段最多生成 2048 token，达到上限会标记结果可能不完整，可减小 `chunk_seconds` 重试。伴奏与人声混合、长拖音、合唱仍可能影响识别；没有参考歌词时不能宣称准确率。

测试无需下载模型：`python -m pytest tests/test_transcription.py`。使用真实权重的完整 HTTP 链路可通过上述 Node.js 脚本验证。

## 本机验证记录（2026-10-19）

RTX 4070 Ti 12 GB、PyTorch 2.8.0+cu126、Transformers 5.14.1、CUDA BF16，以默认 180 秒窗口测试 `Suno_b96cc9c5.m4a`：解码时长 147.9735 秒，单窗口识别子进程耗时 **32.191 秒**（含模型加载），未触发 token 上限。Node.js 已验证上传、认证、排队、查询和两个文件下载；任务完成后子进程退出。

原始结果保存在 `outputs/transcriptions/Suno_b96cc9c5/qwen3-asr.json` 与 `qwen3-asr.txt`（本机产物，不随代码提交）。相比之前的 SenseVoice 样本，本次输出恢复了 GPT/Claude 名称及多句漏词；没有标准歌词，不计算识别准确率。该耗时是一次实测，不是性能保证。

本次后端回归：340 项测试通过。已有运行中的服务需要重启，才会加载新增路由。

## 人工确认歌词后的时间对齐

安装独立的 Qwen3-ForcedAligner-0.6B 权重，再重启 API：

```powershell
.venv/Scripts/python.exe -m minimaxh3_service.setup_asr --aligner
```

模型位于 `models/asr/Qwen3-ForcedAligner-0.6B`，可通过 `H3_ALIGNER_MODEL_PATH` 修改。固定上游版本 `c07281df297b9905d24a508279258cccf987a064`，使用相同 Transformers 环境和 GPU 队列；识别模型与对齐模型分进程运行。

`POST /v1/audio/alignments`，请求：

```json
{"file_id":"上传的音乐 ID","lyrics":"第一行歌词\n第二行歌词","language":"zh"}
```

支持 1–180 秒音频，语言 `zh/en/yue`。每个非空行生成一条字幕。返回 202，沿用任务查询与下载接口，任务 `kind=alignment`。JSON 包含 `timing=forced_alignment`、原样的 `text`、逐行 `cues[{id,start,end,text}]`、逐词 `words[{text,start,end}]`，所有时间相对原音乐起点，保留前奏。重复段落须在歌词中按实际演唱顺序提供，不自动补词。

对齐仍可能受长拖音、合唱和错误歌词影响；无效、越界或整行零时长结果会报错，不把平均分配的时间伪装成模型对齐。相同测试歌曲对齐实测 11.603 秒，16 行，首行从 22.88 秒开始。

加入对齐接口后的后端全量回归：345 项通过。另通过真实 HTTP 请求验证了同首歌曲的对齐提交、轮询和下载，16 行、首句 22.88 秒，原歌词保持不变；该次推理耗时 32.366 秒。
