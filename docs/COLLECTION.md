# 素材采集 API

通过本机 Chrome 扩展读取公开网页的正文和已加载评论，结果保存为 Markdown，并可选保存页面已加载的完整图片/视频响应。扩展位于 `extensions/source-collector`，安装说明见其中的 README。

## 前提

- 本地调用，无需设置 `H3_API_KEY`。
- 在扩展弹窗中点击“开启”，扩展会连接 `http://127.0.0.1:8000` 并领取任务。页面资源捕获默认开启。
- 采集结果保存在 `H3_DATA_DIR/collections/<id>/`（默认 `data/collections/`）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/collections` | 提交一个公开网页，返回 `202` 与任务 `id` |
| GET | `/v1/collections` | 最近的采集任务列表，`limit` 默认 20 |
| GET | `/v1/collections/{id}` | 任务状态、错误原因、Markdown 与媒体下载地址 |
| GET | `/v1/collections/{id}/source` | 下载 `source.md`（完成后可用） |
| GET | `/v1/collections/{id}/media/{filename}` | 下载已捕获的完整响应 |

### 状态

| status | 含义 |
|---|---|
| `queued` | 等待扩展领取（约 30 秒一次） |
| `running` | 扩展正在打开网页 |
| `completed` | `source.md` 已写入 |
| `waiting_login` | 页面需要登录或验证。请在该浏览器中打开的网页完成操作，然后再次提交同一网址 |
| `failed` | 查看 `error` 字段 |

### 示例

```bash
curl -X POST http://127.0.0.1:8000/v1/collections   -H "Content-Type: application/json"   -d '{"url": "https://example.com/articles/1", "title": "示例文章"}'

curl http://127.0.0.1:8000/v1/collections/<id>

curl -o source.md http://127.0.0.1:8000/v1/collections/<id>/source
```

## 限制与安全

- 只接受 http/https 公开网址，标准端口（80/443）。拒绝 localhost、内网与链接中的账号信息。提交时就会检查，不会入队。
- 扩展只读取页面可见正文、可见评论和已加载响应。不读取 cookie、本地存储、密码或隐藏表单，不处理验证码。
- 扩展协议 `/api/collector` 不在 OpenAPI 中，供扩展领取任务和回传结果使用。
- 任务队列在内存中，服务重启后未完成的任务需要重新提交。已完成的结果文件保留在磁盘上。
- 单个页面最多 12 个媒体响应，单个文件图片 12 MB、视频 32 MB。
