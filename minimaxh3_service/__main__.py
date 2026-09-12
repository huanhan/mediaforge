from importlib.util import find_spec

import uvicorn

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    host = settings.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    base_url = f"http://{host}:{settings.port}"
    if find_spec("gradio") is not None:
        print(f"Video UI: {base_url}/ui/", flush=True)
    else:
        print("Video UI: unavailable (install the ui extra to enable Gradio)", flush=True)
    print(f"API docs: {base_url}/docs", flush=True)
    uvicorn.run(
        "minimaxh3_service.api:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
