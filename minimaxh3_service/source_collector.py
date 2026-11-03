"""素材采集：浏览器扩展与 H3 服务之间的任务队列，以及采集结果落盘。

外部通过 /v1/collections 提交网址、查询状态并下载结果；浏览器扩展通过 /api/collector 领取任务并回传结果。
采集只读取可见正文、可见评论和已加载的响应，不读取 cookie、本地存储、密码或隐藏表单。

任务状态保存在内存中，服务重启后未完成的任务需要重新提交。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

LEASE_SECONDS = 300
STALE_SECONDS = 90
MAX_RESULT_BYTES = 350_000
MAX_MEDIA_TOTAL = 64 * 1024 * 1024
MAX_MEDIA_PER_TASK = 12
MAX_RESOURCE = {"image": 12 * 1024 * 1024, "video": 32 * 1024 * 1024}
MAX_TASKS = 500
API_PATH = "/api/collector"
SOURCE_FILE = "source.md"
MEDIA_DIR = "media"


class BridgeError(Exception):
    """带 HTTP 状态码的错误，消息会显示给用户。"""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---------- 校验 ----------

def public_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2000:
        raise BridgeError("链接无效")
    parts = urllib.parse.urlsplit(value.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BridgeError("请填写 HTTP(S) 公开链接")
    if parts.username or parts.password:
        raise BridgeError("链接不能包含账号信息")
    return value.strip()


def public_address(ip: str) -> bool:
    """只允许全球单播地址，与扩展端 publicPage() 的规则保持一致。"""
    try:
        addr = ip_address(ip)
    except ValueError:
        return False
    if addr.version == 4:
        a, b = (int(x) for x in ip.split(".")[:2])
        return not (a in (0, 10, 127) or a >= 224 or (a == 169 and b == 254)
                    or (a == 172 and 16 <= b <= 31) or (a == 192 and b in (0, 168))
                    or (a == 100 and 64 <= b <= 127) or (a == 198 and b in (18, 19)))
    return bool(re.match(r"^[23][0-9a-f]{3}:", ip, re.I)) and not re.match(r"^2001:(db8|0):", ip, re.I)


def check_task_target(url: str) -> None:
    """拒绝非标准端口，以及解析到非公开地址的主机。"""
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        raise BridgeError("仅支持公开网页的标准端口")
    if port not in (None, 80, 443):
        raise BridgeError("仅支持公开网页的标准端口")
    host = (parts.hostname or "").strip("[]")
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise BridgeError("无法解析采集链接的地址")
    addresses = {info[4][0] for info in infos}
    if not addresses or any(not public_address(a) for a in addresses):
        raise BridgeError("浏览器采集不能访问本机或内网")


def host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def digest_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------- 任务状态 ----------

@dataclass
class Task:
    id: str
    url: str
    title: str
    status: str = "queued"          # queued | running | completed | waiting_login | failed
    lease: str = ""
    lease_until: float = 0.0
    error: str = ""
    captured: list[dict] = field(default_factory=list)  # {url, bytes, file}
    created_at: float = field(default_factory=time.time)


class Bridge:
    """保存任务队列与扩展连接状态；HTTP 线程共用同一把锁。"""

    def __init__(self, root: Path):
        self.lock = threading.RLock()
        self.root = root
        self.last_seen = 0.0
        self.extension_version = ""
        self.capture_enabled = False
        self.tasks: dict[str, Task] = {}

    def task_dir(self, task_id: str) -> Path:
        return self.root / task_id

    def report(self, headers) -> None:
        self.last_seen = time.time()
        version = headers.get("X-Collector-Version", "")
        if version and re.match(r"^[\d.]{1,30}$", version):
            self.extension_version = version
        capture = headers.get("X-Collector-Capture", "")
        if capture in ("true", "false"):
            self.capture_enabled = capture == "true"

    def status(self) -> dict:
        connected = self.last_seen > 0 and time.time() - self.last_seen < STALE_SECONDS
        return {"connected": connected, "captureEnabled": self.capture_enabled,
                "extensionVersion": self.extension_version, "lastSeenAt": self.last_seen}

    def enqueue(self, url: str, title: str = "") -> Task:
        url = public_url(url)
        task = Task(id=secrets.token_hex(8), url=url, title=(title or "").strip()[:300] or url)
        with self.lock:
            if len(self.tasks) >= MAX_TASKS:
                self._forget_finished()
            if len(self.tasks) >= MAX_TASKS:
                raise BridgeError("采集队列已满，请稍后再试", 429)
            self.tasks[task.id] = task
        return task

    def _forget_finished(self) -> None:
        """超过上限时丢弃最早的已结束任务（结果文件保留在磁盘上）。"""
        finished = sorted((t for t in self.tasks.values() if t.status not in ("queued", "running")),
                          key=lambda t: t.created_at)
        for old in finished[: max(1, len(self.tasks) - MAX_TASKS + 1)]:
            del self.tasks[old.id]

    def describe(self, task_id: str) -> dict:
        with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise BridgeError("采集任务不存在", 404)
            return {"id": task.id, "url": task.url, "title": task.title, "status": task.status,
                    "error": task.error, "created_at": task.created_at,
                    "captured": [dict(item) for item in task.captured]}

    def recent(self, limit: int) -> list[dict]:
        with self.lock:
            tasks = sorted(self.tasks.values(), key=lambda t: t.created_at, reverse=True)[:limit]
            ids = [t.id for t in tasks]
        return [self.describe(task_id) for task_id in ids if task_id in self.tasks]

    def claim(self) -> Task | None:
        """交出最早排队的任务，或租约已过期的运行中任务。"""
        with self.lock:
            now = time.time()
            candidates = [t for t in self.tasks.values()
                          if t.status == "queued" or (t.status == "running" and t.lease_until < now)]
            if not candidates:
                return None
            task = min(candidates, key=lambda t: t.created_at)
            task.status = "running"
            task.lease = secrets.token_hex(16)
            task.lease_until = now + LEASE_SECONDS
            task.captured = []
            return task

    def find_leased(self, task_id: str, lease: str) -> Task:
        task = self.tasks.get(task_id)
        if not task or task.status != "running" or task.lease != lease or task.lease_until < time.time():
            raise BridgeError("采集租约已失效", 400)
        return task


# ---------- 扩展回传 ----------

def save_media(bridge: Bridge, meta: dict, body: bytes, mime: str) -> dict:
    if not isinstance(meta, dict):
        raise BridgeError("资源描述无效")
    kind = meta.get("kind")
    if kind not in MAX_RESOURCE:
        raise BridgeError("资源类型无效")
    url = public_url(meta.get("url", ""))
    page_url = public_url(meta.get("pageUrl", ""))
    status = meta.get("responseStatus")
    if status not in (200, 206):
        raise BridgeError("资源状态无效")
    content_range = str(meta.get("contentRange", ""))[:100]
    context = str(meta.get("context", ""))[:2000]
    with bridge.lock:
        task = bridge.find_leased(meta.get("id", ""), meta.get("lease", ""))
        if host_of(page_url) != host_of(task.url):
            raise BridgeError("资源不属于当前采集页面")
        if len(task.captured) >= MAX_MEDIA_PER_TASK or sum(r["bytes"] for r in task.captured) > MAX_MEDIA_TOTAL:
            raise BridgeError("当前页面已达到资源捕获上限")
        if any(r["url"] == url for r in task.captured):
            return {}
    if len(body) > MAX_RESOURCE[kind]:
        raise BridgeError("资源超过捕获大小限制")
    if not body:
        raise BridgeError("没有浏览器响应内容")
    if status == 206:
        m = re.match(r"^bytes 0-(\d+)/(\d+)$", content_range)
        if not m or int(m.group(1)) + 1 != int(m.group(2)) or int(m.group(2)) != len(body):
            raise BridgeError("只捕获到视频片段，不能作为完整素材")
    with bridge.lock:
        task = bridge.find_leased(meta["id"], meta["lease"])
        if sum(r["bytes"] for r in task.captured) + len(body) > MAX_MEDIA_TOTAL:
            raise BridgeError("当前页面已达到资源捕获上限")
    digest = digest_of(body)
    directory = bridge.task_dir(task.id) / MEDIA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest[:16]}{_extension_for(mime, kind)}"
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
    _write_json(directory / f"{digest[:16]}.json", {
        "kind": kind, "sourceUrl": url, "pageUrl": page_url, "context": context,
        "sha256": digest, "bytes": len(body), "mime": mime, "file": path.name,
        "capturedAt": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with bridge.lock:
        task = bridge.find_leased(meta["id"], meta["lease"])
        task.captured.append({"url": url, "bytes": len(body), "file": path.name})
    return {"saved": path.name}


def _extension_for(mime: str, kind: str) -> str:
    table = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif",
             "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"}
    return table.get(mime.split(";")[0].strip().lower(), ".bin" if kind == "image" else ".mp4")


def save_result(bridge: Bridge, result: dict) -> dict:
    if not isinstance(result, dict):
        raise BridgeError("返回内容无效")
    status = result.get("status")
    if status not in ("completed", "waiting_login", "failed"):
        raise BridgeError("返回状态无效")
    task_id, lease = result.get("id", ""), result.get("lease", "")
    with bridge.lock:
        bridge.last_seen = time.time()
        task = bridge.tasks.get(task_id)
        if not task or task.lease != lease:
            raise BridgeError("采集任务已更新，请重新连接")
        if task.status == "completed":
            return {"ok": True}
        if task.status != "running":
            raise BridgeError("任务不在采集中")
        if result.get("url") and host_of(public_url(result["url"])) != host_of(task.url):
            raise BridgeError("网页跳转到了其他站点，请补充实际链接后重试")
        text = (result.get("text") or "").strip()
        if status == "completed" and len(text) < 80:
            raise BridgeError("网页没有足够正文，不能标记为采集完成")
        comments = result.get("comments") or []
        media = result.get("media") or []
        if len(comments) > 60 or len(media) > MAX_MEDIA_PER_TASK:
            raise BridgeError("返回的评论或媒体数量超出限制")
        task.status = status
        task.error = str(result.get("error") or "")[:1000]
        task.lease_until = 0.0
    if status == "completed":
        _write_source(bridge, task, result, text, comments, media)
    return {"ok": True}


def _write_source(bridge: Bridge, task: Task, result: dict, text: str, comments: list, media: list) -> None:
    directory = bridge.task_dir(task.id)
    directory.mkdir(parents=True, exist_ok=True)
    title = str(result.get("title") or task.title)[:300]
    url = str(result.get("url") or task.url)
    lines = [f"# 来源：{title}", f"- 网址：{url}", f"- 采集网址：{task.url}",
             f"- 采集时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 来源类型：（待判断：官方 / 媒体 / 个人 / 社交平台）",
             "- 采集方式：素材采集扩展（读取可见正文，未读取 cookie 或隐藏内容）",
             "", "## 正文（原文，未经改写）", "", text, ""]
    if comments:
        lines += ["## 可见评论（仅已加载内容，保留归属）", ""]
        for c in comments:
            author = str(c.get("author") or "网友")[:200]
            likes = c.get("likes")
            suffix = f"（赞 {int(likes)}）" if isinstance(likes, (int, float)) else ""
            lines.append(f"- {author}{suffix}：{str(c.get('text', ''))[:4000]}")
        lines.append("")
    if media:
        lines += ["## 媒体线索", ""]
        for m in media:
            note = f"（未捕获：{m['captureError']}）" if m.get("captureError") else ""
            lines.append(f"- [{m.get('kind')}] {m.get('url')} {m.get('context') or ''} {note}".rstrip())
        if task.captured:
            lines += ["", "已保存到 media/ 的完整响应："]
            lines += [f"- {r['file']}（原网址 {r['url']}）" for r in task.captured]
        lines.append("")
    (directory / SOURCE_FILE).write_text("\n".join(lines), encoding="utf-8")


# ---------- 扩展协议（/api/collector，不进入 OpenAPI） ----------

def register_extension_routes(application, bridge: Bridge) -> None:
    """注册浏览器扩展使用的协议路由：轮询任务、回传结果、上传资源。"""

    async def _read(request: Request, limit: int) -> bytes:
        body = await request.body()
        if len(body) > limit:
            raise BridgeError("浏览器返回内容过大")
        return body

    @application.api_route(API_PATH, methods=["GET", "POST"], include_in_schema=False)
    async def collector_protocol(request: Request):
        try:
            mode = request.query_params.get("mode", "")
            if request.method == "POST" and mode == "media":
                raw = request.headers.get("x-collector-media", "")
                if len(raw) > 18000:
                    raise BridgeError("资源描述过长")
                try:
                    meta = json.loads(urllib.parse.unquote(raw))
                except (ValueError, TypeError):
                    raise BridgeError("资源描述无效")
                bridge.report(request.headers)
                body = await _read(request, MAX_RESOURCE["video"])
                result = await asyncio.to_thread(
                    save_media, bridge, meta, body, request.headers.get("content-type", ""))
                return JSONResponse({"ok": True, **result})
            if request.method == "GET" and mode == "ping":
                bridge.report(request.headers)
                return JSONResponse({"ok": True, "browser": bridge.status()})
            if request.method == "GET":
                bridge.report(request.headers)
                task = bridge.claim()
                if task is None:
                    return JSONResponse({"task": None})
                try:
                    await asyncio.to_thread(check_task_target, task.url)
                except BridgeError as error:
                    with bridge.lock:
                        task.status = "failed"
                        task.error = str(error)
                    return JSONResponse({"task": None})
                return JSONResponse({"task": {"id": task.id, "lease": task.lease,
                                              "url": task.url, "title": task.title}})
            raw = await _read(request, MAX_RESULT_BYTES)
            try:
                result = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise BridgeError("返回内容无效")
            bridge.report(request.headers)
            return JSONResponse(await asyncio.to_thread(save_result, bridge, result))
        except BridgeError as error:
            return JSONResponse({"error": str(error)}, status_code=error.status)
        except Exception as error:  # noqa: BLE001 - 不把内部细节返回给扩展
            return JSONResponse({"error": "采集服务内部错误：" + type(error).__name__}, status_code=500)
