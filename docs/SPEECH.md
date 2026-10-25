# Breeze TTS 2 本地配音

本项目独立管理 [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) 推理、模型与依赖。视频工作室或其他客户端仅通过 HTTP 上传录音、提交任务、轮询与下载，不读取本项目目录、不启动本服务、不安装模型。

## 替换影响

- 旧 IndexTTS 安装器、权重清单和推理实现已移除，不再回退。
- 保留 `/v1/audio/speech` 和 `/v1/tasks` 任务协议、共享 GPU 队列、24 kHz 单声道 WAV 输出；模型改为 `Breeze-TTS-2`。
- 原有八维情绪、独立情绪参考、模型原生语速等字段已移除。新调用使用 `mode`、`instruction`、`cfg_scale`，旧字段返回 422。
- 自录音色必须同时提交参考录音和准确完整的文字稿；预设录音由本服务提供对应文字。
- 本地权重支持中文和英语。旧客户端需更新后使用。
- 普通 eager 模式单请求运行，任务结束释放显存；不启用高显存 fast-all，也不在 API 进程加载模型。

## 安装

先阅读 [完整模型许可](licenses/BREEZE_MODEL_LICENSE.txt)。仅允许研究和非商业用途，本地生成结果也受限制。用户须自行确认其用途并接受许可后执行：

```powershell
.venv/Scripts/python.exe -m minimaxh3_service.setup_tts --accept-non-commercial-license
```

安装器固定源码、模型 revision 和 SHA-256（`minimaxh3_service/breeze_tts_models.json`），使用 uv 创建独立 CPython 3.11 环境（不继承系统 Conda）：

- `.runtime/breeze-tts-env`：Torch/Torchaudio 2.9.1 CUDA 12.8 及上游依赖。
- `.runtime/breeze-tts-src`：固定版本的 Apache-2.0 推理源码。
- `models/tts/Breeze-TTS-2`：约 7.2 GiB 模型文件、原许可、NOTICE 和本机接受记录。

默认支持在本机 Windows 中以 eager 路径运行；上游主要文档面向 Linux，Linux 环境的 Python 位于 `bin/python`。不安装 FlashAttention、不启用 Torch compile/CUDA graphs。需 FFmpeg 在 PATH。Breeze 使用上游 Qwen3 音频 tokenizer；它是 Breeze 必需的编解码依赖，并非旧 Qwen TTS 推理后端。

`--models-only` / `--runtime-only` 可以分别准备权重与环境，仍需显式许可参数。模型不打入便携包，不在推理时下载。若曾设置 `H3_TTS_PYTHON`，请改为新的独立环境；`H3_TTS_MODEL_PATH` 为 `models/tts` 的父级配置根，`H3_TTS_DEVICE` 默认 cuda。API 需由使用者手动重启以加载代码，安装器不会启动服务。

`GET /v1/models` 的 `capabilities.speech` 提供后端 `breeze_tts`、就绪状态、语言、模式、语速处理方式和非商业许可链接。就绪检查不加载权重；真实生成仍可能受空闲显存影响。官方建议普通推理至少 12GB 显存，视频模型需先释放，由共享队列串行管理。

## 接口

```json
{
  "text": "[叹气] 没想到过了这么久，你还记得我的声音。",
  "model": "Breeze-TTS-2",
  "speaker": "Serena",
  "language": "Auto",
  "mode": "clone"
}
```

上传音频：POST `/v1/files`，`kind=audio`。复刻时用 `reference_file_id` 和 `reference_text` 替代预设音色；录音 3–30 秒，建议 10–15 秒干净单人语音，文字须与完整录音一致。不能循环拼接音频或填写与录音不同的文案。

- `mode=clone`：不传入任何额外表演指令，CFG 固定为 1，优先保留参考声线与语气。
- `mode=direction`：需要 `instruction`，可描述情绪、节奏、表达方式；`cfg_scale` 为指令引导系数 1–6，默认 4。它不等于情绪百分比，不保证精确控制任意表演。
- `mode=design`：需要 `instruction` 描述声音，不需要参考音频/文字稿，不使用默认预设；CFG 默认 4。与 clone/direction 的录音复刻分开，不能同时传自定义参考录音。
- 中文 `[笑]`、`[叹气]`、`[咳嗽]`、`[清嗓子]`，英文 `(laugh)`、`(sigh)`、`(cough)`、`(clears throat)` 直接写在台词中。这是官方举例，并非完整支持清单。Gradio 已移除整句效果示例，仅保留台词、参考音色或 instruction、CFG；高级设置已移除，其他参数使用服务默认值。

以下为兼容底层 HTTP 调用而保留的可选参数；工作室和 Gradio 不展示、不传入这些参数：

- `speed` 0.5–2：合成后通过 FFmpeg atempo 保音高变速；不是模型原生时长控制。
- `pause_ms` 0–2000：换行/长文本分段之间追加的静音，随后与整段一起变速，不是每个标点的停顿。
- `volume_db` -12–12；temperature .1–1.5（默认 .9），top_p .1–1（默认 1），top_k 1–100（默认 50），repetition_penalty 1–2（默认 1.1），seed。

每个请求最多 1000 字，自动按句子/长度分段，命中音频 token 上限会报错而不是发布截断台词。输出 JSON 记录模型、参数、非商业许可与变速方式。任务可取消、失败清理未完成输出。字幕和人物口型同步另行制作。

## 预设与许可

九个历史预设 ID 保留，默认 Serena。它们是固定的合成参考 WAV，不是 Breeze 内置说话人。来源和文字保存在 `voice_presets/manifest.json`，不代表支持参考语言的输出或方言。

推理代码 Apache-2.0 与模型的非商业许可严格分开。见 `docs/licenses/`、`THIRD_PARTY_NOTICES.md` 和下载目录的 LICENSE/NOTICE。项目代码开放不授予模型或本地输出商用权。模型权重分发须附完整原许可、NOTICE、保留归属，并仅供研究或非商业用途；衍生模型仍受同一许可限制。云服务订阅不授权本地商用。

## 本机验证（2026-10-26）

Windows、RTX 4070 Ti 12GB、独立 CPython 3.11、Torch 2.9.1 CUDA 12.8，BF16 eager。
通过一次性 `SpeechRunner` 调用验证，未启动或重启 H3 API 服务。每次均重新加载模型，耗时含模型加载与 FFmpeg 后处理：

| 场景 | 输出时长 | 耗时 |
| --- | ---: | ---: |
| Serena 中文纯复刻 | 5.03 秒 | 85.7 秒 |
| 中文表演指令，CFG 4，1.15× 语速 | 2.30 秒 | 60.4 秒 |
| 本机已保存录音，准确文字稿，纯复刻 | 2.71 秒 | 59.2 秒 |
| Ryan 英语纯复刻 | 2.23 秒 | 42.0 秒 |

四份音频均为非静音的完整 24 kHz 单声道 PCM WAV。独立本地 SenseVoice 转写与目标台词一致（忽略英文大小写与标点）；这验证内容完整性，不等于主观音质、情感表现或说话人相似度评测。12GB 保守路径可以运行，但当前冷启动试听不是即时播放。测试输出保存在本机 `outputs/breeze-smoke-20261026/`，不随代码分发。

H3 配音、安装许可、OpenAPI、Gradio 相关测试共 26 项通过。客户端另验证默认参数、旧画布迁移、逐镜配音合成、录音管理，以及文字稿修改后缓存失效。

补充验证：instruction-only `design` 模式在本机成功生成带 `[笑]` 的中文示例，2.06 秒音频，冷启动约 42.4 秒；独立本地转写为“太好了，我们终于做到了。”，与目标台词一致（忽略标点）。输出位于 `outputs/breeze-design-smoke-20261026/`，未启动 H3 服务。扩展后的 H3 相关测试共 30 项通过。
