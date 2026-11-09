#!/usr/bin/env node
/**
 * 查询 NewsNow 热点新闻的命令行工具（Node.js 18+，无第三方依赖）。
 *
 * 接口: GET {base}/api/s?id=<源ID>[&latest]
 * 源码参考: https://github.com/newsnext/newsnow  (server/api/s/index.ts, shared/pre-sources.ts)
 *
 * 用法示例:
 *   node newsnow.js -l                       # 列出常用源 ID
 *   node newsnow.js weibo                    # 微博热搜（默认 10 条）
 *   node newsnow.js zhihu -n 20              # 知乎前 20 条
 *   node newsnow.js wallstreetcn-quick       # 子源用 "主ID-子ID"
 *   node newsnow.js weibo --json             # 原始 JSON 输出
 *   node newsnow.js weibo --base https://你的部署地址
 */

import { parseArgs } from "node:util"

const DEFAULT_BASE = "https://newsnow.busiyi.world"
const USER_AGENT = "Mozilla/5.0 (newsnow-cli)"
const TIMEOUT_MS = 20_000
const MAX_LIMIT = 30

// 常用源 ID（来自 shared/pre-sources.ts，可能随上游变化；未列出的 ID 也可直接传入）
const KNOWN_SOURCES = {
  "v2ex": "V2EX",
  "v2ex-share": "V2EX 最新分享",
  "zhihu": "知乎",
  "weibo": "微博实时热搜",
  "zaobao": "联合早报",
  "coolapk": "酷安今日最热",
  "mktnews": "MKTNews",
  "mktnews-flash": "MKTNews 快讯",
  "wallstreetcn-quick": "华尔街见闻 快讯",
  "wallstreetcn-news": "华尔街见闻 最新",
  "wallstreetcn-hot": "华尔街见闻 最热",
  "36kr-quick": "36氪 快讯",
  "36kr-renqi": "36氪 人气榜",
  "douyin": "抖音",
  "hupu": "虎扑",
}

function usage() {
  return `用法: node newsnow.js [源ID] [选项]

选项:
  -n, --limit <数字>   显示条数（默认 10，最多 30）
  -l, --list           列出常用源 ID
      --latest         强制获取最新数据（需服务端允许登录态）
      --json           输出原始 JSON
      --base <地址>    服务地址（默认 ${DEFAULT_BASE}）
  -h, --help           显示帮助`
}

async function fetchSource(base, sourceId, latest) {
  const params = new URLSearchParams({ id: sourceId })
  if (latest) params.set("latest", "")
  const url = `${base.replace(/\/+$/, "")}/api/s?${params}`

  let res
  try {
    res = await fetch(url, {
      headers: { "User-Agent": USER_AGENT, "Accept": "application/json" },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  } catch (e) {
    const reason = e.cause?.message || e.message
    throw new Error(`网络错误: ${reason}`)
  }

  const text = await res.text()
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`)
  }
  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`返回内容不是合法 JSON: ${text.slice(0, 200)}`)
  }
}

function formatTime(ts) {
  if (typeof ts === "number") {
    // 毫秒时间戳（秒级时间戳会小于 1e11）
    const d = new Date(ts > 1e11 ? ts : ts * 1000)
    const pad = n => String(n).padStart(2, "0")
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  }
  return ts ? String(ts) : "-"
}

function printItems(data, limit) {
  const items = data.items ?? []
  const status = data.status ?? ""
  console.log(`源: ${data.id}  状态: ${status}  更新时间: ${formatTime(data.updatedTime)}`)
  if (status === "cache") console.log("（上游抓取失败，返回的是缓存数据）")
  const shown = Math.min(limit, items.length)
  console.log(`共 ${items.length} 条，显示前 ${shown} 条\n`)

  items.slice(0, limit).forEach((item, i) => {
    const extra = item.extra ?? {}
    const hint = extra.info || extra.hover || ""
    let line = `${String(i + 1).padStart(2, " ")}. ${item.title ?? ""}`
    if (hint) line += `  [${hint}]`
    console.log(line)
    const url = item.mobileUrl || item.url
    if (url) console.log(`    ${url}`)
  })
}

async function main() {
  let values, positionals
  try {
    ;({ values, positionals } = parseArgs({
      allowPositionals: true,
      options: {
        limit: { type: "string", short: "n", default: "10" },
        list: { type: "boolean", short: "l", default: false },
        latest: { type: "boolean", default: false },
        json: { type: "boolean", default: false },
        base: { type: "string", default: DEFAULT_BASE },
        help: { type: "boolean", short: "h", default: false },
      },
    }))
  } catch (e) {
    console.error(`参数错误: ${e.message}\n`)
    console.error(usage())
    return 1
  }

  if (values.help) {
    console.log(usage())
    return 0
  }

  const sourceId = positionals[0]
  if (values.list || !sourceId) {
    console.log("常用源 ID：")
    for (const [id, desc] of Object.entries(KNOWN_SOURCES)) {
      console.log(`  ${id.padEnd(22)} ${desc}`)
    }
    if (!sourceId) {
      console.log("\n" + usage())
    }
    return 0
  }

  const limit = Math.max(1, Math.min(Number.parseInt(values.limit, 10) || 10, MAX_LIMIT))

  let data
  try {
    data = await fetchSource(values.base, sourceId, values.latest)
  } catch (e) {
    console.error(`请求失败: ${e.message}`)
    return 1
  }

  if (values.json) {
    console.log(JSON.stringify(data, null, 2))
  } else {
    printItems(data, limit)
  }
  return 0
}

process.exitCode = await main()
