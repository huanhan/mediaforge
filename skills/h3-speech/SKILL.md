---
name: h3-speech
description: 用 MiniMax H3 本地服务（http://127.0.0.1:8000）的 Breeze TTS 2 合成语音（24kHz 单声道 WAV），支持预设音色、参考音频克隆、语气指令和音色设计。完整流程为上传参考音频（如需）、提交文本与参数、轮询任务、下载结果。当用户要做配音、口播、旁白、语音合成或声音克隆时使用。仅限研究与非商业用途，不做口型同步。
---

# H3 语音合成（Breeze TTS 2）

完整流程：**上传参考音频（克隆时）→ 提交文本与参数 → 轮询任务 → 下载 WAV**。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

**许可提示**：Breeze TTS 2 仅限研究与非商业用途。用于商业项目前需要确认授权。

## 第 0 步：检查服务就绪

```bash
curl -s "http://127.0.0.1:8000/readyz"
```

确认服务为 `ready`，且语音相关依赖已安装（缺失时会体现在 `missing_*` 字段）。

## 第 1 步：上传参考音频（仅克隆需要）

```bash
curl -s -X POST http://127.0.0.1:8000/v1/files -F "kind=audio" -F "file=@/path/ref.wav"
```

参考音频时长需 3–30 秒。记下 `file_id`。

## 第 2 步：提交合成

接口：`POST /v1/audio/speech`，JSON。

三种模式：

| mode | 用途 | 必要字段 |
|---|---|---|
| `clone`（默认） | 克隆参考音频的音色与说话方式 | `reference_file_id` + 与音频逐字一致的 `reference_text` |
| `direction` | 预设音色，用 `instruction` 控制语气 | `speaker`、`instruction`、`cfg_scale` |
| `design` | 无参考音频，用文字描述设计音色 | `instruction`、`cfg_scale` |

其他字段：

- `text`（必填）：支持中文 `[笑]` `[叹气]` `[咳嗽]` `[清嗓子]`，英文 `(laugh)` `(sigh)` `(cough)` `(clears throat)`。
- `speaker`：`Serena`（默认）、`Vivian`、`Uncle_Fu`、`Dylan`、`Eric`、`Ryan`、`Aiden`、`Ono_Anna`、`Sohee`。
- `language`：`Chinese`（默认）、`English`、`Auto`。
- `speed`：0.5–2.0，音调保持不变的后处理变速。
- `pause_ms`：0–2000，文本分段间的停顿。
- `volume_db`：-12 到 12。
- `cfg_scale`：1–6。
- `temperature`、`top_p`、`top_k`、`repetition_penalty`、`seed`：采样参数，一般保持默认。

示例（预设音色 + 语气）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"欢迎来到今天的节目。","speaker":"Serena","language":"Chinese","mode":"direction","instruction":"语气温暖自然","cfg_scale":4}'
```

示例（音色克隆）：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text":"这是一段克隆音色的测试。","mode":"clone","reference_file_id":"uploads/ref.wav","reference_text":"参考音频里的原话"}'
```

返回 202，包含 `id`、`queue_number`。

## 第 3 步：轮询任务

```bash
TASK_ID=<上一步返回的 id>
while true; do
  s=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID")
  st=$(echo "$s" | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status=$st"
  case "$st" in succeeded|failed|canceled) break;; esac
  sleep 5
done
```

## 第 4 步：取结果

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/tasks/$TASK_ID" | python -c "import sys,json;print(json.load(sys.stdin)['outputs'][0]['download_url'])")
curl -s -o speech.wav "$URL"
```

## 注意

- 输出只有语音，不做口型同步。
- `speed` 和 `pause_ms` 会改变时长，需要与画面对齐时重新核对。
- `reference_text` 必须和参考音频逐字一致，否则克隆效果会变差。
