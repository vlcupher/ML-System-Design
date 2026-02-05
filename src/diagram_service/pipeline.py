"""Pipeline v4: preprocess -> tiling -> two-pass -> API response. Model: Qwen3-VL-2B."""

import os
from typing import Any, Dict, List

from PIL import Image

from diagram_service.models.base import ModelAdapter
from diagram_service.prompts import (
    EXTRACT_PROMPTS,
    STRUCTURE_FROM_EXTRACT_PROMPTS,
    STRUCTURE_PROMPTS,
    build_api_response,
    json_repair_prompt,
    merge_steps_simple,
    parse_json_strict,
)
from diagram_service.preprocess import crop_empty_edges
from diagram_service.tiling import Tile, allowed_sides, make_tiles


def _normalize_steps(obj: Dict[str, Any], diagram_type: str) -> None:
    raw = obj.get("steps")
    if not raw or not isinstance(raw, list):
        obj["steps"] = []
        return
    out: List[Any] = []
    if (diagram_type or "").upper() == "OTHER":
        for s in raw:
            if isinstance(s, str) and s.strip():
                out.append(s.strip())
            elif isinstance(s, dict) and (s.get("text") or s.get("action")):
                out.append((s.get("text") or s.get("action") or "").strip())
    else:
        for s in raw:
            if isinstance(s, dict):
                action = (s.get("action") or s.get("text") or "").strip() or "???"
                role = (s.get("role") or "").strip() or "???"
                out.append({"action": action, "role": role})
            elif isinstance(s, str) and s.strip():
                out.append({"action": s.strip(), "role": "—"})
    obj["steps"] = out


def extract_tile_structure(
    tile: Tile,
    diagram_type: str,
    model_adapter: ModelAdapter,
) -> Dict[str, Any]:
    header = (
        f"TILE_META: index={tile.index}, axis={tile.axis}, bbox={list(tile.bbox)}, "
        f"allowed_boundary_sides={allowed_sides(tile.axis)}\n"
    )
    prompt = header + STRUCTURE_PROMPTS[diagram_type]
    raw = model_adapter.run_vlm(tile.image, prompt, max_new_tokens=850)
    obj = None
    try:
        obj = parse_json_strict(raw)
    except Exception:
        try:
            repaired = model_adapter.run_text_only(
                json_repair_prompt(raw), max_new_tokens=800
            )
            obj = parse_json_strict(repaired)
        except Exception:
            obj = {"steps": []}
    if obj is None or not isinstance(obj, dict):
        obj = {"steps": obj if isinstance(obj, list) else []}
    obj.setdefault("tile", {})
    obj["tile"]["index"] = tile.index
    obj["tile"]["axis"] = tile.axis
    obj["tile"]["bbox"] = list(tile.bbox)
    _normalize_steps(obj, diagram_type)
    return obj


def _merge_extracted_to_text(extract_results: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for i, data in enumerate(extract_results):
        if not isinstance(data, dict):
            parts.append(f"[Tile {i + 1}]: (non-dict output)")
            continue
        snippets = data.get("text_snippets") or data.get("text") or []
        elements = data.get("elements") or []
        if isinstance(snippets, list):
            parts.append(f"[Tile {i + 1}] text: " + " | ".join(str(s) for s in snippets))
        else:
            parts.append(f"[Tile {i + 1}] text: {snippets}")
        if elements:
            parts.append(f"[Tile {i + 1}] elements: " + ", ".join(str(e) for e in elements))
    return "\n\n".join(parts)


def extract_tile_raw(
    tile: Tile,
    diagram_type: str,
    model_adapter: ModelAdapter,
) -> Dict[str, Any]:
    header = (
        f"TILE_META: index={tile.index}, axis={tile.axis}, bbox={list(tile.bbox)}\n"
    )
    prompt = header + EXTRACT_PROMPTS[diagram_type]
    raw = model_adapter.run_vlm(tile.image, prompt, max_new_tokens=850)
    try:
        return parse_json_strict(raw)
    except Exception:
        try:
            repaired = model_adapter.run_text_only(
                json_repair_prompt(raw), max_new_tokens=800
            )
            return parse_json_strict(repaired)
        except Exception:
            return {"text_snippets": [raw], "elements": []}


def run_pipeline_two_pass(
    img: Image.Image,
    diagram_type: str,
    model_adapter: ModelAdapter,
    threshold: int = 800,
    overlap: float = 0.15,
    crop_padding: int = 4,
) -> Dict[str, Any]:
    if diagram_type not in EXTRACT_PROMPTS or diagram_type not in STRUCTURE_FROM_EXTRACT_PROMPTS:
        raise ValueError(
            f"diagram_type must be one of {list(STRUCTURE_PROMPTS.keys())}, got {diagram_type!r}"
        )
    img = crop_empty_edges(img, padding=crop_padding)
    tiles = make_tiles(img, threshold=threshold, overlap=overlap)
    extract_results: List[Dict[str, Any]] = []
    for t in tiles:
        extract_results.append(extract_tile_raw(t, diagram_type, model_adapter))
    merged_text = _merge_extracted_to_text(extract_results)
    structure_prompt = (
        STRUCTURE_FROM_EXTRACT_PROMPTS[diagram_type]
        + "\n\n--- Extracted content ---\n\n"
        + merged_text
    )
    raw_structured = model_adapter.run_text_only(
        structure_prompt, max_new_tokens=1024
    )
    obj = None
    try:
        obj = parse_json_strict(raw_structured)
    except Exception:
        try:
            repaired = model_adapter.run_text_only(
                json_repair_prompt(raw_structured), max_new_tokens=800
            )
            obj = parse_json_strict(repaired)
        except Exception:
            obj = {"steps": []}
    if obj is None or not isinstance(obj, dict):
        obj = {"steps": []}
    _normalize_steps(obj, diagram_type)
    unified = {"diagram_type": diagram_type, "steps": obj.get("steps") or []}
    response = build_api_response(unified, tiles_count=len(tiles))
    return {
        "response": response,
        "tiles": [{"index": t.index, "axis": t.axis, "bbox": t.bbox} for t in tiles],
        "tile_structures": extract_results,
        "unified": unified,
    }


def _use_two_pass() -> bool:
    if os.environ.get("TWO_PASS", "").strip() == "0":
        return False
    return True


def run_pipeline(
    img: Image.Image,
    diagram_type: str,
    model_adapter: ModelAdapter,
    threshold: int = 800,
    overlap: float = 0.15,
    crop_padding: int = 4,
) -> Dict[str, Any]:
    if diagram_type not in STRUCTURE_PROMPTS:
        raise ValueError(
            f"diagram_type must be one of {list(STRUCTURE_PROMPTS.keys())}, got {diagram_type!r}"
        )
    if _use_two_pass():
        return run_pipeline_two_pass(
            img, diagram_type, model_adapter,
            threshold=threshold, overlap=overlap, crop_padding=crop_padding,
        )
    img = crop_empty_edges(img, padding=crop_padding)
    tiles = make_tiles(img, threshold=threshold, overlap=overlap)
    tile_structs: List[Dict[str, Any]] = []
    for t in tiles:
        tile_structs.append(extract_tile_structure(t, diagram_type, model_adapter))
    unified = merge_steps_simple(tile_structs, diagram_type)
    response = build_api_response(unified, tiles_count=len(tiles))
    return {
        "response": response,
        "tiles": [{"index": t.index, "axis": t.axis, "bbox": t.bbox} for t in tiles],
        "tile_structures": tile_structs,
        "unified": unified,
    }


def get_model_adapter() -> ModelAdapter:
    """v4: Qwen3-VL-2B only."""
    from diagram_service.models.qwen3vl import Qwen3VLAdapter
    return Qwen3VLAdapter()
