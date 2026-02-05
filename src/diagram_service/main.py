"""Entrypoint v4: same as original (pipeline, tiling), model = Qwen3-VL-2B."""

import uvicorn

from diagram_service.api import app, mount_ui

mount_ui(app, path="/ui")

if __name__ == "__main__":
    uvicorn.run(
        "diagram_service.main:app",
        host="0.0.0.0",
        port=int(__import__("os").environ.get("PORT", "8000")),
        reload=False,
    )
