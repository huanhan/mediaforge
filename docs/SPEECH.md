# IndexTTS 2.5 本地配音与音色克隆

采用官方 [IndexTTS](https://github.com/index-tts/index-tts) 的 `infer_v2_5.IndexTTS2` 和 [IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)。具体源码、模型 revision、文件大小和权重 SHA-256 固定在 `minimaxh3_service/index_tts_models.json`。

文本转语音仅保留 IndexTTS 2.5 的安装、推理和接口实现，无旧引擎回退。情绪描述统一使用 `emotion_text`，不再接受旧 `instruct` 参数。IndexTTS 自带的 `qwen0.6bemo4-merge` 是文字情绪分析组件，必须保留；ASR、歌词对齐和 H3 视频文本编码器也不属于旧 TTS，仍由各自模块使用。

## 安装

```powershell
.venv\Scripts\python.exe -m minimaxh3_service.setup_tts
.venv\Scripts\python.exe main.py
```

安装器使用 uv 创建 `.runtime/index-tts-env` Python 3.11 环境，安装 Torch/Torchaudio 2.8 CUDA 12.8 与固定源码到 `.runtime/index-tts-src`，SentencePiece 固定 0.2.1（0.2.2 Windows wheel 在本机导入崩溃）。模型及辅助权重约 7.9 GiB，位于 `models/tts/IndexTTS-2.5`。需 PATH 中 FFmpeg。模型下载校验大小及权重 SHA-256；推理禁用自动联网补模型，不需要 NLTK 语料。

支持 `--models-only`、`--runtime-only`；可设置 `H3_TTS_PYTHON`、`H3_TTS_MODEL_PATH`、`H3_TTS_DEVICE`、`H3_TTS_TIMEOUT_SECONDS`。旧环境变量若指向 Qwen 解释器，请更新。默认在 CUDA 用 BF16，关闭需编译的 CUDA kernel、DeepSpeed 和 torch.compile。模型随独立 worker 退出释放显存，与 H3 视频共用串行 GPU 队列。

`GET /v1/models` 的 `capabilities.speech` 返回就绪状态、后端 `index_tts`、语言与情绪能力。更新代码后需重启 API 才会使用新后端。

## 接口

每请求对应一个镜头，最多 1000 字。

```json
{
  "model": "IndexTTS-2.5",
  "text": "你好，欢迎来到我的频道。",
  "speaker": "Serena",
  "language": "Chinese",
  "speed": 1.1,
  "emotion_mode": "text",
  "emotion_text": "自然、平静地讲述。",
  "emotion_strength": 0.65,
  "pause_ms": 200,
  "volume_db": 0,
  "seed": 42
}
```

POST `/v1/audio/speech` 返回 202 + id，GET `/v1/tasks/{id}` 查询并下载 WAV 与参数/时长 JSON。输出 24kHz 单声道 PCM16，沿用鉴权、取消与队列策略。

语速 0.5–2× 映射到模型 `duration_factor=1/speed`，不再使用 FFmpeg atempo。实际长度由模型生成，倍率不是精确时长保证。语言为 Chinese、English、Japanese、Spanish、Arabic、Auto。自动识别限制在上述五种语言，短文本建议明确选择。

情绪模式：
- `text`：`emotion_text` 使用本地 0.6B 情绪分析器映射到八维情绪。
- `vector`：`emotion_vector` 为八个 0–1 数值，顺序是 happy、angry、sad、afraid、disgusted、melancholic、surprised、calm。`emotion_strength` 为总强度。
- `reference`：可传 `emotion_reference_file_id`，借用独立音频的情绪；省略时沿用说话人样本，模型强制使用样本完整情绪。

其他参数：`pause_ms` 0–2000（内部文本分段之间），`volume_db` -12–12，`temperature` .1–1.5，`top_p` .1–1，`top_k` 1–100，`repetition_penalty` 1–20，`max_segment_tokens` 40–200，`text_normalization` 布尔值。默认值见 OpenAPI；文本规范化的具体支持由上游各语言前端决定。

音色克隆先 POST `/v1/files` 上传 `kind=audio`，然后传 `reference_file_id`。录音 3–30 秒，建议 10–15 秒；上游最多使用参考前 15 秒。`reference_text` 可选（仅作兼容备注）。克隆音色也支持全部情绪参数。

九个历史预设 ID 保留，默认 Serena；`voice_presets/` 包含由旧引擎一次性生成的固定合成参考录音，来源记录在 manifest 和 README。后续推理只使用 IndexTTS 克隆这些样本，无需 Qwen 模型；音色表现可能与旧引擎不同。参考录音的语言不代表输出模型支持该语言。

## 视频工作室

设置中支持默认音色和所有控制参数、多语言／自填文本试听、录制／拖拽上传音色。自录音频默认轻度 FFmpeg 预处理（高通、温和降噪、首尾静音整理、有限增益与峰值限制），提供原始／处理后对比，保存规范化原始 WAV 与处理后 WAV。此流程位于视频工作室服务器，H3 API 收到的是已选定的参考音频；H3 的独立 Gradio 录音入口不执行该增强流程。

每个镜头一个配音节点，连接视频生成配置，点击视频生成时合成或复用当前配音并写入视频结果。更改参数使旧音频失效；超长台词提示调整，不截断。节点可以应用已保存默认参数，也可独立控制。错误旧连线不阻止打开画布。详见视频工作室 `docs/speech.md`。

源码和模型按上游提供的 LICENSE / MODEL_LICENSE 使用，许可文本随下载源保存；不要将旧 Qwen 的 Apache-2.0 许可误套到 IndexTTS 模型上。

## 本机验证（2026-10-25）

RTX 4070 Ti 12GB，本地离线推理，同一句中文、同一音色与种子：1× 为 5.700 秒，1.35× 为 4.226 秒。完整 SpeechRunner 使用经过前端同一预处理流程的参考样本，分别验证八维情绪和独立情绪参考，输出 24kHz 单声道 WAV，时长为 4.064 / 4.029 秒，含启动加载耗时约 50.1 / 71.6 秒。

实际音频和参数记录保存在本机 `outputs/index-tts-smoke/`，汇总为 `validation.json`。参考样本是预置合成声音，不是用户真实麦克风测试；本次为功能验证，不是主观音质评分或识别准确率评测。前端相关 29 项测试、H3 语音/接口/配置/打包相关 60 项测试通过；前端类型检查、画布构建和生产构建通过。
