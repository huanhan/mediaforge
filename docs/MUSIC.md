# YuE2 音乐生成与 API

音乐使用 [YuE 仓库当前主线的 YuE2](https://github.com/multimodal-art-projection/YuE)，模型为 **m-a-p/YuE2-3B + YuE2-Vae**。旧 MiniMax Music 3 HTTP 适配器和 ACE-Step 推理入口已移除；视频仍使用 MiniMax H3。旧音乐权重文件不会自动删除。

支持曲风与歌词生成歌曲、空歌词请求纯音乐、乐谱规划、ABC 乐谱输入、种子、进度、取消及下载。输出为 **48 kHz、双声道、16-bit PCM WAV**，同时提供生成记录 JSON；有乐谱时另提供 `.abc`。空歌词和 `instrumental` 提示不能保证完全无人声。

## 安装

YuE2 使用独立 Python 3.12 环境，固定 PyTorch 2.10.0 / Transformers 4.57.6，不修改 H3 的 Python 3.13 / PyTorch 2.8 环境。需先安装 Git、uv 和 NVIDIA 驱动。在项目目录使用服务环境的 Python：

```powershell
python -m minimaxh3_service.setup_music
python -m minimaxh3_service
```

安装器通过 uv 创建 `.runtime/yue2-env`，安装 CUDA 12.8 PyTorch 和固定提交的 YuE2，然后下载模型至 `models/music/yue2/`。权重共约 7.8 GB，此外需要独立 PyTorch 环境空间。下载可续传，并以固定 SHA-256 校验权重。生成阶段强制离线，不会临时下载模型。

```powershell
# 只装运行环境
python -m minimaxh3_service.setup_music --runtime-only
# 自定义位置；随后配置 H3_MUSIC_PYTHON 和 H3_MUSIC_MODEL_PATH
python -m minimaxh3_service.setup_music --runtime-dir D:/runtimes/yue2 --model-dir D:/models/yue2
# 单独重试下载，必须使用 YuE2 环境的 Python
.runtime/yue2-env/Scripts/python.exe minimaxh3_service/setup_music.py --models-only
```

Linux 的独立解释器路径为 `.runtime/yue2-env/bin/python`。官方推荐 Linux、支持 BF16 的 NVIDIA GPU、24 GB 显存。此适配器默认启用 AR 卸载、12 GiB 预算和 eager 推理；Windows 使用 PyTorch 内置的节省显存注意力内核，不需要额外安装 Flash Attention。不可用时按 128 个查询分块计算，MLP 按 256 帧分块；保留全部上下文、原有 32 步声学采样及歌曲时长。CPU 可用于兼容性调试，生成速度会很慢。

预算包含上游预留的 2 GiB：设置 12 时，PyTorch 分配上限约为 10 GiB，并不是可以额外使用 12 GiB。它不能释放其他应用占用的显存，也不是完整的逐层卸载。长歌词、长 ABC 乐谱、双分支 CFG 或其他 GPU 程序仍会影响可用余量。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `H3_MUSIC_PYTHON` | Windows: `.runtime/yue2-env/Scripts/python.exe`；Linux: `.runtime/yue2-env/bin/python` | 独立解释器 |
| `H3_MUSIC_MODEL_PATH` | `models/music/yue2` | 包含 `YuE2-3B` 和 `YuE2-Vae` 的目录 |
| `H3_MUSIC_DEVICE` | `cuda` | `cuda`、`cuda:0` 等或 `cpu` |
| `H3_MUSIC_MEMORY_BUDGET_GIB` | `12` | 必须大于 2；24 GB 卡可设为 24 |
| `H3_MUSIC_OFFLOAD_AR` | `true` | 声学合成阶段卸载 AR 权重 |
| `H3_MUSIC_TIMEOUT_SECONDS` | `3600` | 单任务超时，含加载与解码 |
| `H3_ALLOW_INTERRUPT_RUNNING` | `false` | 设为 `true` 可终止运行中的任务 |
| `H3_API_KEY` | 空 | 服务鉴权，支持 `X-API-Key` 和 Bearer |

原 `H3_MUSIC_BACKEND`、`H3_MUSIC_BASE_URL`、`H3_MUSIC_MODEL`、`H3_MUSIC_API_KEY` 已停止使用。`H3_MUSIC_MODEL_PATH` 现在指向 YuE2 模型目录，不再是单个 ACE-Step 文件。环境变量需由启动环境设置；`.env.example` 只是配置示例。

## API

Swagger：`http://127.0.0.1:8000/docs`。可用性见 `GET /v1/models` → `capabilities.music`。缺少模型或依赖时提交返回 HTTP 503，`error.detail` 包含就绪信息；视频模型未安装不妨碍已就绪的音乐功能。

`POST /v1/music/generations`：

```json
{
  "prompt": "Mandarin, warm piano pop, expressive female voice, acoustic piano, rounded bass, light drums, 88 BPM",
  "lyrics": "[Verse]\n清晨的风轻轻吹过\n街角传来熟悉的歌\n[Chorus]\n让阳光照亮每个梦\n和你走过春夏秋冬",
  "duration_seconds": 30,
  "seed": 42,
  "cot": "full"
}
```

返回 HTTP 202：

```json
{"id": "任务ID", "status": "queued", "queue_number": 1}
```

| 参数 | 说明 |
|---|---|
| `prompt` | 必填，曲风、语言、乐器、情绪、人声与速度描述 |
| `lyrics` | 默认空；歌曲可用 `[Verse]`、`[Chorus]` 等段落标签 |
| `duration_seconds` | 10–500，默认 30，输出时长上限 |
| `seed` | 0–2147483647，默认 7 |
| `cot` | `full`（默认，旋律和和弦）、`melody`（仅旋律）、`off`（直接生成） |
| `abc` | 可选 YuE2 格式 ABC 乐谱；与 `cot=off` 不能同时使用 |
| `cfg_scale` | 可选 0–20，省略或 null 使用模型默认引导强度 |

时长通过每秒 25 个语义帧的生成预算及最终音频裁切限制，**不保证整首歌或乐句自然结束**。短请求可能只唱出部分歌词。JSON 中的 `truncated.abc`、`truncated.semantic`、`truncated.audio_cropped` 分别标记乐谱、语义采样和音频是否截断。达到预算仍可返回有效音频；请检查这些标记。

生成长音乐时，把页面“最长时长”或 API 的 `duration_seconds` 设置为 `180`（3 分钟）、`300`（5 分钟）或最多 `500`（8 分 20 秒），并提供匹配长度的分段歌词。时长是上限，模型可能提前结束。建议先使用默认 `cot=full`、默认引导强度。无需修改显存预算来延长歌曲。

如果日志出现 `torch_dtype is deprecated`，这是上游加载接口的提示；实际失败原因应查看最后一行异常。Windows 上原生 GQA 可能退回 math 注意力，产生随序列长度平方增长的临时矩阵。本适配器通过展开 K/V 头启用 efficient SDPA，或分块后备计算来控制该峰值。策略记录在输出 JSON 的 `config.memory_policy` 中；浮点运算顺序变化可能使相同种子的结果与旧版本不同。工作进程每次任务重新启动，更新这部分代码后新音乐任务立即生效。

用 `GET /v1/tasks/{id}` 轮询，或 `GET /v1/tasks/{id}/events` 订阅 SSE。成功后 `outputs` 按 WAV、JSON、可选 ABC 的顺序提供 `download_url`。取消使用 `DELETE /v1/tasks/{id}`；队列满返回 429。任务记录在进程内，重启后历史任务 ID 不再可查询。

Python 示例（依赖 `httpx`）：

```python
import time
from pathlib import Path
import httpx

with httpx.Client(base_url="http://127.0.0.1:8000", timeout=90,
                  headers={"X-API-Key": "替换为密钥；未启用鉴权可删除此项"}) as client:
    response = client.post("/v1/music/generations", json={
        "prompt": "Mandarin, warm piano pop, female voice, 88 BPM",
        "lyrics": "[Verse]\n清晨的风轻轻吹过\n街角传来熟悉的歌",
        "duration_seconds": 30, "seed": 42, "cot": "full",
    })
    response.raise_for_status()
    task_id = response.json()["id"]
    while True:
        response = client.get(f"/v1/tasks/{task_id}")
        response.raise_for_status()
        task = response.json()
        if task["status"] in {"succeeded", "failed", "canceled", "unknown"}:
            break
        time.sleep(3)
    if task["status"] != "succeeded":
        raise RuntimeError(task.get("error") or task["status"])
    for artifact in task["outputs"]:
        response = client.get(artifact["download_url"])
        response.raise_for_status()
        Path(artifact["filename"]).write_bytes(response.content)
```

音乐与视频等共用单 GPU 队列，切换前释放前一个模型。每个音乐任务在独立子进程中执行，完成、失败、取消或超时后结束进程并释放 CUDA 资源。进度为阶段与 token 数估计，不是剩余时间预测。每次任务都包含模型加载和校验时间。

## 版本与许可

- YuE2 代码：`1647252d68b70cbead046ffd3f7caf036dde47ee`（`yue2-infer 0.1.6`）。
- YuE2-3B 权重：`8f24312f187763b9854adeec87e437e2b03bbff1`。
- YuE2-Vae 权重：`152733a19ad43aa67e367f9b5503ef8075bb5126`。
- [官方生成文档](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/docs/generation.md)。
- 代码为 Apache-2.0；当前 YuE2 权重采用 [CC BY-NC 4.0 加创作者额外许可](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/MODEL_LICENSE)。个人创作者与企业商业使用的条款不同，企业商用需按官方条款联系授权。

## 本机验证（2026-10-19）

Windows / RTX 4070 Ti 12 GB，通过真实 HTTP API 提交中文歌词、30 秒上限、`cot=full`、`seed=42`，57.08 秒完成任务（含模型加载与校验）。下载得到 29.9987 秒、48 kHz、双声道 PCM WAV，以及生成记录和 ABC 乐谱。音频非静音，RMS 约 0.0649；任务结束后 GPU 总显存回落至约 2.1 GB。样例在 `outputs/yue2-smoke-20261019/`，不随源码提交。

该样例的 `truncated.semantic=true`，表示达到时长预算；这不是完整歌曲或歌词准确率测试，也不代表 12 GB 卡能稳定生成所有 300 秒请求。全量 361 项回归测试通过，另外补充了 Windows 就绪输出兼容和进程树终止测试。

### 长音乐显存修复验证（同日）

在相同的 Windows / RTX 4070 Ti 12 GB 上，通过真实 HTTP API 提交分段中文歌词、180 秒上限、`cot=full`、`seed=20261019`：

- 实际输出 **179.9987 秒**，48 kHz、双声道、16-bit PCM；WAV、JSON、ABC 均可下载，SSE 正常返回成功状态。
- 端到端约 **511.87 秒（8 分 32 秒）**；每 3 秒采样的整卡显存峰值为 **10570 MiB（10.32 GiB，含其他应用）**，结束后回落到 2066 MiB。
- 音频 RMS 为 0.1473，峰值 0.9426，无满幅截波；`truncated.semantic=true` 表示用完 180 秒预算。这是可运行性验证，未评估歌词准确率或音乐质量。
- 原故障发生在 NAR prefill；此任务完整通过 prefill、32 步声学合成和分块 VAE 解码。

真实 CUDA 注意力对比使用 4096 个查询 / 键、16 个查询头、8 个 K/V 头、BF16：因果计算额外峰值由 2536 MiB 降至 48 MiB，非因果计算由 2464 MiB 降至 48 MiB；相对 L2 误差分别约 0.21% 和 0.25%。这些是单次注意力的临时张量占用，不是整模型显存。

测试音频和报告保存在 `outputs/yue2-memory-fix-20261019/163847/`，CUDA 对比报告为同目录上一级的 `attention_check.json`，不随源码提交。全项目 **371 项测试通过**，包含分块掩码、GQA 数值、MLP 分块、异常恢复及 API 回归。300 秒仍是接口上限，本轮长曲实测长度为 180 秒。
