---
name: mf-newsnow
description: 查询 NewsNow 聚合的热门新闻榜单（知乎、微博、抖音、B站、百度、华尔街见闻等）。当用户想看某个平台今天的热点、热榜、热搜、快讯，或问“知乎/微博现在什么热门”时使用。不支持小红书。与 mediaforge 本地服务无关。
---

# NewsNow 热榜查询

通过 `scripts/newsnow.js` 调用 NewsNow 公开接口，获取各平台热榜。需要 Node.js 18+，无第三方依赖。

## 使用方式

脚本位于本仓库 `skills/mf-newsnow/scripts/newsnow.js`，在项目根目录运行：

```bash
node skills/mf-newsnow/scripts/newsnow.js -l                  # 列出常用源 ID
node skills/mf-newsnow/scripts/newsnow.js zhihu -n 10         # 知乎前 10 条
node skills/mf-newsnow/scripts/newsnow.js weibo -n 20         # 微博热搜前 20 条
node skills/mf-newsnow/scripts/newsnow.js weibo --json        # 原始 JSON（含完整链接等字段）
node skills/mf-newsnow/scripts/newsnow.js wallstreetcn-quick  # 子源用 "主ID-子ID"
```

选项：`-n/--limit`（默认 10，最多 30）、`-l/--list`、`--json`、`--latest`、`--base <地址>`（自建服务时使用）。

## 工作流程

1. 确认用户想要的平台。若只说“热点”而未指定，可先列出几个常用源让用户选，或默认查知乎或微博。
2. 运行脚本，取需要的条数。
3. 把结果整理成编号列表返回给用户，保留标题和热度，需要时附上链接。

## 注意事项

- 输出中 `状态: cache` 表示上游抓取失败，返回的是缓存数据，不一定是最新内容。需要如实告诉用户。
- 源 ID 随上游变化。遇到 `Invalid source id` 时，运行 `-l` 查看常用 ID，或到 NewsNow 源码 `shared/pre-sources.ts` 中确认。
- 小红书目前不在 NewsNow 支持列表内，无法查询，应直接告知用户并建议替代平台。
- 默认服务地址 `https://newsnow.busiyi.world` 为第三方部署，可能不稳定；失败时可提示用户通过 `--base` 指定其他地址。
