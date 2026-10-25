# 本地文本转语音与录音克隆

采用 Qwen3-TTS-12Hz-1.7B：CustomVoice 提供九个预置音色，Base 根据用户录音及准确文字克隆声音。默认 Serena 中文温柔女声。选择考虑中文自然度、可用的声音克隆接口、Apache 2.0 模型许可和 12GB 显存部署；不声称在所有语种和场景中绝对最佳。

## 安装和运行

```powershell
.venv\Scripts\python.exe -m minimaxh3_service.setup_tts
.venv\Scripts\python.exe main.py
```

安装器使用 uv 创建 `.runtime/qwen3-tts-env` Python 3.12 环境，安装 qwen-tts 0.1.1 / PyTorch 2.8 CUDA 12.6，避免与 H3 / ASR 的 Transformers 依赖冲突。两个 checkpoint 下载至 `models/tts`，固定官方 revision，校验 safetensors 的 SHA-256。含 tokenizer 约 9GB，另需运行环境空间。运行期间只读本地权重，禁止自动联网补模型。需要 PATH 中的 FFmpeg。

可设置 `H3_TTS_PYTHON`、`H3_TTS_MODEL_PATH`、`H3_TTS_DEVICE`（默认 cuda）、`H3_TTS_TIMEOUT_SECONDS`（默认 1800）。Windows 使用 PyTorch SDPA，不依赖 FlashAttention 的 Windows 编译安装。模型随独立子进程退出释放，与视频、音乐共用串行 GPU 队列。`/v1/models` 中 `capabilities.speech` 提供就绪状态。

## API

每次提交是一份镜头配音，最多 1000 字；长内容在创作阶段拆镜。

```json
POST /v1/audio/speech
{
  "text": "你好，欢迎来到我的频道。今天一起了解一个有趣的知识。",
  "speaker": "Serena",
  "language": "Chinese",
  "instruct": "自然清晰，像与观众交谈一样讲解。",
  "speed": 1.0,
  "seed": 42
}
```

返回 HTTP 202 和 `id`，使用 `GET /v1/tasks/{id}` 查询，通过 `outputs[].download_url` 下载 24kHz 单声道 PCM16 WAV 和实际时长 JSON。沿用 API Key 鉴权、队列上限和取消策略；运行中取消需 `H3_ALLOW_INTERRUPT_RUNNING=true`。

声音克隆：先用 `POST /v1/files` 上传 `kind=audio` 文件（支持浏览器 WebM 录音），再提交：

```json
{
  "text": "这是用录制音色生成的新句子。",
  "reference_file_id": "上传返回的 file_id",
  "reference_text": "参考录音中实际说出的完整文字。",
  "language": "Chinese",
  "speed": 1.0
}
```

参考音频须 3–30 秒，建议 10–20 秒单人、清晰、无背景音乐。克隆使用音频与准确文字的上下文模式，不使用质量较低的只取声纹模式。Base 不支持 instruct，录制音色保持参考样本语气；需要不同情绪时录制对应语气。语速使用 FFmpeg atempo，保持音高。

## 视频工作室

设置 → 文本转语音 · 本地音色，可录音、上传、试听并选择默认音色。浏览器录音需要 localhost / HTTPS 和麦克风权限；上传录音不需要麦克风。录音保存在视频工作室数据目录的 `voices/`，合成时传给已配置的本地 H3 服务。

画布新增“文本转语音”工具节点；视频生成配置上提供“添加本镜头配音”。每镜一个节点，保存时固定默认音色，随后可独立修改。场景预设实际传入预置音色的语气指令：知识讲解、产品介绍、新闻播报、故事旁白、自然口播。自录音色使用参考样本语气。

配音节点连接视频生成配置的“本地配音”输入，每个配置一个；点击视频生成时自动合成或复用配音，并写入视频结果，默认替换原声，可改为混音。结果自带声音，预览、下载和合并均可直接使用；不连接视频结果节点，不在导出时再次配音。超出画面长度时生成失败并提示调整文案、语速或镜头长度，不截断口播。文案、声音、语速变更只影响下次视频生成，已有结果保留。配乐放在整片合并节点。此功能不提供人物口型同步，也不把生成语音当成精确字幕时间轴。

官方模型与 API 资料：https://github.com/QwenLM/Qwen3-TTS
