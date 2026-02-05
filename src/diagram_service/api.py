"""FastAPI application v4: same as original (pipeline, tiling), model = Qwen3-VL-2B."""

import io
import logging
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from diagram_service.pipeline import get_model_adapter, run_pipeline
from diagram_service.prompts import STRUCTURE_PROMPTS

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diagram Recognition Service v4",
    description="Same pipeline as original (tiling, two-pass). Model: Qwen3-VL-2B.",
    version="4.0.0",
)

_model_adapter = None


def get_adapter():
    global _model_adapter
    if _model_adapter is None:
        _model_adapter = get_model_adapter()
        _model_adapter.load()
    return _model_adapter


REQUEST_COUNT = None
REQUEST_DURATION = None
INFERENCE_DURATION = None


def _ensure_metrics():
    global REQUEST_COUNT, REQUEST_DURATION, INFERENCE_DURATION
    if REQUEST_COUNT is not None:
        return
    try:
        from prometheus_client import Counter, Histogram
        REQUEST_COUNT = Counter(
            "diagram_requests_total",
            "Total diagram recognition requests",
            ["diagram_type", "status"],
        )
        REQUEST_DURATION = Histogram(
            "diagram_request_duration_seconds",
            "Request duration in seconds",
            ["diagram_type"],
        )
        INFERENCE_DURATION = Histogram(
            "model_inference_duration_seconds",
            "Model inference duration in seconds",
        )
    except ImportError:
        pass


@app.get("/health")
def health():
    try:
        adapter = get_adapter()
        if adapter.is_loaded():
            return {"status": "ok", "model_loaded": True, "version": "4.0.0"}
        raise HTTPException(status_code=503, detail="model not loaded")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/describe")
def describe(
    file: UploadFile = File(...),
    diagram_type: Optional[str] = None,
):
    _ensure_metrics()
    if diagram_type is None:
        diagram_type = "OTHER"
    if diagram_type not in STRUCTURE_PROMPTS:
        raise HTTPException(
            status_code=400,
            detail=f"diagram_type must be one of {list(STRUCTURE_PROMPTS.keys())}",
        )

    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.warning("Invalid image upload: %s", e)
        if REQUEST_COUNT:
            REQUEST_COUNT.labels(diagram_type=diagram_type, status="error").inc()
        raise HTTPException(status_code=400, detail="Invalid image file")

    start = time.perf_counter()
    try:
        adapter = get_adapter()
        inf_start = time.perf_counter()
        result = run_pipeline(img, diagram_type=diagram_type, model_adapter=adapter)
        if INFERENCE_DURATION:
            INFERENCE_DURATION.observe(time.perf_counter() - inf_start)
        response = result["response"]
        if REQUEST_COUNT:
            REQUEST_COUNT.labels(diagram_type=diagram_type, status="success").inc()
        if REQUEST_DURATION:
            REQUEST_DURATION.labels(diagram_type=diagram_type).observe(time.perf_counter() - start)
        return response
    except Exception as e:
        logger.exception("Pipeline failed")
        if REQUEST_COUNT:
            REQUEST_COUNT.labels(diagram_type=diagram_type, status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    _ensure_metrics()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def mount_ui(app_instance: FastAPI, path: str = "/ui") -> None:
    try:
        import gradio as gr
        from diagram_service.ui.app import create_ui
        demo = create_ui()
        gr.mount_gradio_app(app_instance, demo, path=path)
    except ImportError:
        pass
