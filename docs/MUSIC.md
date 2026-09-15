# 本地音乐生成

默认使用 **ACE-Step v1 3.5B**，在本机直接加载权重，通过已有 Comfy 推理内核运行，不启动 ComfyUI 服务，不需要 MiniMax 云 API 或独立 Linux 音乐服务器。实际模型不是 MiniMax Music 3。

支持曲风描述生成纯音乐、歌词生成歌曲、10–120 秒、随机种子、任务状态、试听和 44.1 kHz 立体声 PCM WAV 下载。英文曲风提示词通常更稳定。模型采样结果受种子影响，不保证精确遵循所有乐器和歌词要求。

## 安装与启动

沿用本项目的 Python 3.13 / CUDA PyTorch / Comfy 内核运行环境，然后下载音乐权重：

```powershell
uv run python -m minimaxh3_service.setup_music
uv run --extra ui minimaxh3
```

下载器从 ModelScope 的 Comfy-Org/ACE-Step_ComfyUI_repackaged 获取固定版本权重，并验证大小和 SHA-256，支持中断续传。模型约 7.7 GB，保存至 models/music/ace_step_v1_3.5b.safetensors。下载完成后生成过程无需联网。

可用 H3_MUSIC_MODEL_PATH 指定这个权重文件的位置。H3_MUSIC_BACKEND 默认为 native。音乐与视频、超分共用 GPU 队列，切换任务前释放前一种模型，空闲时释放音乐模型。首次生成包含权重加载时间。

访问 /ui/ 的“音乐生成”，输入曲风描述，歌词留空生成纯音乐。默认提供“轻快、尤克里里、钢琴、拍手、118 BPM”示例。点击“检查音乐服务”可以查看实际模型及就绪状态。

排队任务可以取消；设置 H3_ALLOW_INTERRUPT_RUNNING=true 后，本地音乐任务可在采样步之间取消。模型加载和解码期间需等到下一个取消检查点。

## API

POST /v1/music/generations：

```json
{
  "prompt": "upbeat, cheerful acoustic pop, ukulele, bright piano, handclaps, light drums, 118 BPM, instrumental",
  "lyrics": "",
  "duration_seconds": 30,
  "seed": 42
}
```

返回 HTTP 202，包含 id、status、queue_number。通过 GET /v1/tasks/{id} 或现有 SSE 接口查询；完成后的 outputs 提供 WAV 下载地址。音频帧网格可能使实际时长比指定值短不到 0.1 秒。

生成歌曲时填写 lyrics，段落标签各占一行，例如：

```text
[Verse]
Morning sunshine at my door
Every step brings something more
[Chorus]
Let the music lift us high
Like a kite across the sky
```

## 可选：外部 MiniMax Music 3

前一版实现的 Music 3 HTTP 适配器仍可使用，需显式选择：

```powershell
$env:H3_MUSIC_BACKEND = "remote"
$env:H3_MUSIC_BASE_URL = "http://your-music-host:8001"
# 网关需要认证时配置 H3_MUSIC_API_KEY
```

按 [MiniMax 官方自部署指南](https://platform.minimax.io/docs/guides/local-deploy-music-3) 部署 SGLang-Omni。可选 H3_MUSIC_MODEL 指定模型 ID，H3_MUSIC_TIMEOUT_SECONDS 默认为 1800 秒。远程模式需填写歌词；API 支持最长 300 秒，输出 32 kHz WAV。上游没有运行中取消和卸载接口，也不提供采样进度。本项目无法释放远程模型显存。

## 来源

- [ACE-Step 项目及许可证](https://github.com/ace-step/ACE-Step)
- [Comfy 官方音频模型工作流](https://comfyanonymous.github.io/ComfyUI_examples/audio/)
- [重打包权重](https://huggingface.co/Comfy-Org/ACE-Step_ComfyUI_repackaged)

ACE-Step 项目采用 Apache-2.0；部署的模型应同时遵循其模型卡与随附许可。模型下载器不会接受网站协议或运行下载的代码。

## 本机实测

RTX 4070 Ti 12GB / Windows 已通过真实 API 生成验证：30 秒请求得到 29.907 秒、44.1 kHz 立体声 WAV，首次任务约 33.63 秒，包含模型加载。完成后模型自动释放。结果见 outputs/music-local-test/REPORT.md。这是单次实测，不代表所有时长和负载的性能。

带歌词测试使用四句英文歌词生成约 30 秒歌曲。独立的本地 Whisper base.en 在未输入参考歌词的情况下识别出全部四句，归一化后均匹配，其中第二句重复一次。该结果验证了这组英文歌词的生效情况，不代表中文歌词或所有曲风都达到同样准确率。音频与识别记录保存在本地 outputs/music-lyrics-test/，不随源码提交。
