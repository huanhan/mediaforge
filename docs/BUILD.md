# 一键构建

在项目根目录选择需要的一条命令运行。两个入口独立，每次自动创建新的输出目录，都包含 Gradio 界面。

## Windows 免安装包

```powershell
python scripts/build_windows.py
```

产物：`dist/windows-日期-时间/`，包含 ZIP 和校验文件。解压、放好模型后双击 `start.cmd`。

需要 Windows x64、Python 3.13、uv、Git 和所需的 x64 运行库；不需要 Docker。

## Docker 镜像

```powershell
python scripts/build_docker.py
```

产物：`dist/docker-日期-时间/`，包含镜像 TAR、校验文件、Compose 配置和使用说明，同时保留本地构建的镜像。

需要 Python 3.13 和运行中的 Docker（Linux 容器模式）；不需要 Windows 打包环境。

构建需要联网。两种包均不包含模型权重，模型需另外放置。

服务启动后访问：

- 视频界面：http://127.0.0.1:8000/ui/
- API 文档：http://127.0.0.1:8000/docs

更多选项与问题排查见 [详细说明](BUILD_DETAILS.md)。
