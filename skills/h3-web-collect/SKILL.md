---
name: h3-web-collect
description: 通过 MiniMax H3 本地服务（http://127.0.0.1:8000）让操作者浏览器中的 Chrome 扩展打开公开网页，读取可见正文和已加载的评论，生成 Markdown 源文件，并可下载页面的完整响应资源。完整流程为提交网址、轮询采集状态、处理需要登录或验证的情况、下载结果。当用户要采集、抓取、保存某个网页的正文或评论，或要把网页内容存成 Markdown 时使用。
---

# H3 网页采集（浏览器扩展）

完整流程：**提交公开网址 → 轮询采集状态 → （需要时）用户在浏览器中登录或验证后重新提交 → 下载 Markdown 与页面资源**。

这个能力和其他 H3 业务不同：采集不在 GPU 上运行，而是由**操作者本机浏览器里的 Chrome 扩展**打开页面完成。因此服务端必须有扩展在线，任务才会被处理。

基础地址：`http://127.0.0.1:8000`。若配置了 `H3_API_KEY`，每个请求都加 `-H "X-API-Key: $H3_API_KEY"`。

## 第 0 步：确认采集可用

```bash
curl -s "http://127.0.0.1:8000/v1/collections?limit=5"
```

返回 `{"collections": [...]}` 说明服务可访问。若返回 401，检查 API Key；若需要确认扩展是否在线，可以提交一个任务后观察是否一直停在 `queued`。

## 第 1 步：提交网址

接口：`POST /v1/collections`，JSON。

| 字段 | 必填 | 说明 |
|---|---|---|
| `url` | 是 | 公开的 http(s) 网址，使用标准端口（80/443）。**localhost 和内网地址会被拒绝** |
| `title` | 否 | 任务标签，显示在状态里，默认为网址本身，最长 300 字符 |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/article/123","title":"示例文章"}'
```

返回 202，包含 `id`（16 位十六进制）和 `status: "queued"`。

注意：
- 只允许采集**公开**页面。内网、本机地址、非标准端口会返回 400。
- 返回 429 表示提交过于频繁，稍后再试，不要连续重复提交同一个网址。

## 第 2 步：轮询采集状态

```bash
COLLECTION_ID=<上一步返回的 id>
while true; do
  s=$(curl -s "http://127.0.0.1:8000/v1/collections/$COLLECTION_ID")
  st=$(echo "$s" | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status=$st"
  case "$st" in completed|failed|waiting_login) break;; esac
  sleep 10
done
echo "$s"
```

状态含义：

| status | 含义 | 下一步 |
|---|---|---|
| `queued` | 等待浏览器处理 | 继续轮询 |
| `running` | 扩展正在读取页面 | 继续轮询 |
| `completed` | 已写出 `source.md` | 进入第 3 步取结果 |
| `waiting_login` | 页面需要登录或通过验证 | 进入第 2a 步 |
| `failed` | 采集失败 | 读取 `error` 告知用户 |

## 第 2a 步：处理 waiting_login

页面需要用户登录或完成人机验证。**这一步必须由用户本人在自己的浏览器里完成**，不要尝试绕过登录或验证。

1. 把 `error` 字段的说明告诉用户，请他们在操作者的浏览器里打开该网址，完成登录或验证。
2. 用户确认完成后，**用同一个 URL 再次提交** `POST /v1/collections`。接口文档要求这样做，返回的 `id` 应以响应为准（是否复用原 ID 文档未说明）。
3. 回到第 2 步轮询新的任务。

## 第 3 步：取结果

任务 `completed` 后：

- `source_download_url`：Markdown 源文件，包含页面正文和可见评论。

```bash
URL=$(curl -s "http://127.0.0.1:8000/v1/collections/$COLLECTION_ID" | python -c "import sys,json;print(json.load(sys.stdin)['source_download_url'])")
curl -s -o source.md "$URL"
```

- `media`：页面加载过程中捕获的响应资源（图片、视频等），仅在扩展开启资源捕获时才有。每项含 `filename`、`bytes`、`source_url`、`download_url`。

```bash
curl -s "http://127.0.0.1:8000/v1/collections/$COLLECTION_ID" | python -c "
import sys,json
for m in json.load(sys.stdin).get('media',[]): print(m['filename'], m['bytes'], m['download_url'])"
```

也可以直接用接口下载 Markdown：`GET /v1/collections/{id}/source`；资源用 `GET /v1/collections/{id}/media/{filename}`。

## 查看最近任务

```bash
curl -s "http://127.0.0.1:8000/v1/collections?limit=20"
```

`limit` 为 1–100，默认 20。用于找回丢失的任务 ID，或检查之前采集的状态。

## 注意

- 这个能力会在操作者的真实浏览器中打开页面。只采集用户有权访问和保存的公开内容，遵守目标网站的服务条款与版权要求。
- 采集结果包含页面可见的评论。若评论含有个人信息，交付前应提醒用户注意处理。
- 页面需要登录时，只能由用户本人登录。不要索取、保存或代为输入账号密码。
- 扩展不在线或浏览器未运行时，任务会长时间停在 `queued`。此时请用户检查 Chrome 和扩展是否开启。
- 采集任务使用独立的 `/v1/collections` 接口查询。接口文档没有说明它是否占用 GPU 队列，也没有说明 `/v1/tasks` 能否查到它，所以不要假设，统一用 `/v1/collections` 查。
