"""Prompts for diagram parsing and response building."""

import json
import re
from typing import Any, Dict, List


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    return s


def parse_json_strict(s: str) -> Any:
    """Parse JSON from model output, stripping markdown code fences."""
    return json.loads(_strip_code_fences(s))


def json_repair_prompt(bad: str) -> str:
    """Prompt to ask model to fix invalid JSON into strict valid JSON."""
    return (
        "Fix the following into STRICT VALID JSON ONLY. "
        "Do not add any extra text, no markdown.\n\n" + bad
    )


STRUCTURE_PROMPTS = {
    "BPMN": r"""You are a BPMN diagram parser. You see a TILE (crop) of a process diagram.
Extract process STEPS with WHO performs each step (lane/pool name or system).

Return STRICT VALID JSON ONLY (no markdown, no comments). Do NOT translate text.

Schema:
{
  "steps": [
    {"action": "<наименование действия>", "role": "<роль/дорожка/система или несколько через запятую>"}
  ],
  "boundary": {
    "incoming": [{"side": "LEFT|RIGHT|TOP|BOTTOM", "to_text": <string or null>}],
    "outgoing": [{"side": "LEFT|RIGHT|TOP|BOTTOM", "from_text": <string or null>}]
  }
}

Rules:
- "action" = short name of the task/action (as on diagram).
- "role" = lane name, pool name, or system (e.g. "Jira", "Разработчик"). If several actors — list comma-separated.
- Order steps top-to-bottom or left-to-right as on diagram.
- boundary: only if this tile is not the whole image; side: LEFT/RIGHT for horizontal split, TOP/BOTTOM for vertical.
- Include only visible elements in this tile.
""",
    "UML": r"""You are a UML diagram parser. You see a TILE of a diagram.
Return STRICT VALID JSON. Do NOT translate.

Schema:
{
  "steps": [{"action": "<действие/элемент>", "role": "<актор/роль или null>"}],
  "boundary": {"incoming": [], "outgoing": []}
}
Order by reading flow. Use "role" for actor/lifeline where applicable.
""",
    "C4": r"""You are a C4 diagram parser. You see a TILE. Return STRICT VALID JSON. Do NOT translate.
Schema: {"steps": [{"action": "<element name>", "role": "<container/system or null>"}], "boundary": {"incoming": [], "outgoing": []}}
""",
    "OTHER": r"""You are a flowchart/block diagram parser. You see a TILE of a diagram.
Extract steps of the algorithm in ORDER (as they follow on the diagram: top-to-bottom or left-to-right).

Return STRICT VALID JSON ONLY (no markdown). Do NOT translate text.

Schema:
{
  "steps": ["<текст первого шага/блока>", "<второй шаг>", ...]
}
Rules:
- Each string = text inside one block/step. Keep original language.
- Order = sequence of execution. If unclear, use visual order (top→bottom or left→right).
- boundary only if tile is part of bigger image:
  "boundary": {"incoming": [{"side": "LEFT|RIGHT|TOP|BOTTOM", "to_text": null}], "outgoing": [{"side": "...", "from_text": null}]}
""",
}

# --- Two-pass: Pass 1 = extract all text + elements; Pass 2 = build structured description ---

EXTRACT_PROMPTS = {
    "BPMN": r"""You see a TILE (crop) of a process/BPMN diagram.
Extract ALL visible text and list every element in reading order (top-to-bottom, left-to-right).
Do NOT interpret or structure as steps yet. Just capture raw content.

Return STRICT VALID JSON ONLY (no markdown). Do NOT translate.

Schema:
{"text_snippets": ["<exact text 1>", "<exact text 2>", ...], "elements": ["<short element description>", ...]}
""",
    "UML": r"""You see a TILE of a UML diagram.
Extract ALL visible text and list every element (boxes, lines, labels) in reading order.
Do NOT interpret yet. Just capture raw content.

Return STRICT VALID JSON ONLY. Do NOT translate.
Schema: {"text_snippets": ["...", ...], "elements": ["...", ...]}
""",
    "C4": r"""You see a TILE of a C4 diagram.
Extract ALL visible text and list every element (containers, systems, labels) in order.
Do NOT interpret yet. Just capture raw content.

Return STRICT VALID JSON ONLY. Do NOT translate.
Schema: {"text_snippets": ["...", ...], "elements": ["...", ...]}
""",
    "OTHER": r"""You see a TILE of a flowchart/block diagram.
Extract ALL visible text from blocks and labels, and list every element in order (top-to-bottom or left-to-right).
Do NOT interpret the algorithm yet. Just capture raw content.

Return STRICT VALID JSON ONLY. Do NOT translate.
Schema: {"text_snippets": ["<text from block 1>", ...], "elements": ["<element type or label>", ...]}
""",
}

STRUCTURE_FROM_EXTRACT_PROMPTS = {
    "BPMN": r"""Given the following extracted text and elements from a BPMN diagram (possibly from multiple tiles), produce a single structured description.

Return STRICT VALID JSON ONLY (no markdown). Do NOT translate.

Schema:
{"steps": [{"action": "<task/action name>", "role": "<lane/pool/system>"}, ...]}

Order steps as in the process flow. Use the "role" for who performs each step.
""",
    "UML": r"""Given the following extracted text and elements from a UML diagram, produce a single structured description.

Return STRICT VALID JSON ONLY. Do NOT translate.
Schema: {"steps": [{"action": "<element/action>", "role": "<actor/role or null>"}, ...]}
Order by reading flow.
""",
    "C4": r"""Given the following extracted text and elements from a C4 diagram, produce a single structured description.

Return STRICT VALID JSON ONLY. Do NOT translate.
Schema: {"steps": [{"action": "<element name>", "role": "<container/system or null>"}, ...]}
""",
    "OTHER": r"""Given the following extracted text and elements from a flowchart/block diagram, produce a single structured description of the algorithm steps in order.

Return STRICT VALID JSON ONLY. Do NOT translate.

Schema: {"steps": ["<step 1 description>", "<step 2 description>", ...]}

Each string = one step of the algorithm in execution order. Keep original language.
""",
}


def merge_steps_simple(tile_objs: List[Dict[str, Any]], diagram_type: str) -> Dict[str, Any]:
    all_steps: List[Any] = []
    for obj in tile_objs:
        steps = obj.get("steps")
        if not steps:
            continue
        if isinstance(steps, list):
            for s in steps:
                if s is None or (isinstance(s, dict) and not s):
                    continue
                all_steps.append(s)
    return {"diagram_type": diagram_type, "steps": all_steps}


def build_api_response(unified: Dict[str, Any], tiles_count: int = 0) -> Dict[str, Any]:
    """
    Form response for frontend: JSON ready for serialization.
    OTHER: steps = [{"step": 1, "description": "..."}, ...]
    BPMN/UML/C4: steps = [{"action": "...", "role": "..."}, ...]
    """
    steps = unified.get("steps") or []
    diagram_type = (unified.get("diagram_type") or "OTHER").upper()
    out: Dict[str, Any] = {
        "diagram_type": diagram_type,
        "steps": [],
        "tiles_count": tiles_count,
    }
    if diagram_type == "OTHER":
        for i, s in enumerate(steps, 1):
            text = (s if isinstance(s, str) else str(s)).strip() or "???"
            out["steps"].append({"step": i, "description": text})
    else:
        for s in steps:
            if isinstance(s, dict):
                action = (s.get("action") or s.get("text") or "???").strip()
                r = s.get("role")
                role = ", ".join(r) if isinstance(r, list) else (r or "???")
                role = str(role).strip()
            else:
                action = str(s).strip() or "???"
                role = "—"
            out["steps"].append({"action": action, "role": role})
    return out
