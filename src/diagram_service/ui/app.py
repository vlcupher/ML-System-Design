"""Gradio UI v4: upload image, choose diagram type, show table (№, Шаг, Роль) and JSON. Same as original."""

import json
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from diagram_service.pipeline import get_model_adapter, run_pipeline
from diagram_service.prompts import STRUCTURE_PROMPTS


def response_to_table(response: Dict[str, Any]) -> List[List[Any]]:
    """
    Build table rows for Gradio Dataframe. Always 3 columns: №, Шаг, Роль.
    For OTHER, Роль is left empty.
    """
    steps = response.get("steps") or []
    diagram_type = (response.get("diagram_type") or "OTHER").upper()
    rows: List[List[Any]] = []

    if diagram_type == "OTHER":
        for s in steps:
            step_num = s.get("step", len(rows) + 1)
            desc = s.get("description", "") or "—"
            rows.append([step_num, desc, ""])
        return rows

    for s in steps:
        step_num = len(rows) + 1
        action = (s.get("action") or "").strip() or "—"
        role = (s.get("role") or "").strip() or "—"
        rows.append([step_num, action, role])
    return rows


def recognize(
    image: Optional[Image.Image],
    diagram_type: str,
) -> Tuple[List[List[Any]], str, str]:
    """
    Run pipeline and return (table_data, json_str, error_message).
    If error, table and json are empty, error_message is set.
    """
    if image is None:
        return [], "", "Загрузите изображение диаграммы."

    if diagram_type not in STRUCTURE_PROMPTS:
        return [], "", f"Тип диаграммы должен быть один из: {list(STRUCTURE_PROMPTS.keys())}"

    try:
        adapter = get_model_adapter()
        if not adapter.is_loaded():
            adapter.load()
        img = image.convert("RGB") if image.mode != "RGB" else image
        result = run_pipeline(img, diagram_type=diagram_type, model_adapter=adapter)
        response = result["response"]
        rows = response_to_table(response)
        json_str = json.dumps(response, ensure_ascii=False, indent=2)
        return rows, json_str, ""
    except Exception as e:
        return [], "", str(e)


def create_ui() -> gr.Blocks:
    """Create Gradio interface: image upload, diagram type, table output, JSON (v4: Qwen3-VL-2B)."""
    with gr.Blocks(title="Распознавание диаграмм v4", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Распознавание диаграмм (v4)\n"
            "Тот же пайплайн, что и в первом проекте (тайлинг, two-pass). Модель: **Qwen3-VL-2B**. Загрузите изображение и выберите тип диаграммы."
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Изображение диаграммы")
                diagram_type = gr.Dropdown(
                    choices=list(STRUCTURE_PROMPTS.keys()),
                    value="OTHER",
                    label="Тип диаграммы",
                )
                run_btn = gr.Button("Распознать")
            with gr.Column():
                table_out = gr.Dataframe(
                    headers=["№", "Шаг", "Роль"],
                    label="Шаги алгоритма",
                    datatype=["number", "str", "str"],
                )
                json_out = gr.JSON(label="Ответ API (JSON)")
                error_out = gr.Textbox(label="Ошибка", interactive=False, visible=True)

        def run(image: Optional[Image.Image], dtype: str):
            rows, json_str, err = recognize(image, dtype)
            if err:
                return [], None, err
            try:
                json_obj = json.loads(json_str) if json_str else None
            except Exception:
                json_obj = None
            return rows, json_obj, ""

        run_btn.click(
            fn=run,
            inputs=[image_input, diagram_type],
            outputs=[table_out, json_out, error_out],
        )

    return demo


def mount_gradio_app():
    """Return Gradio app for mounting on FastAPI."""
    return create_ui()
