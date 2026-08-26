from __future__ import annotations

import json
from typing import Any


def answer_text(content: Any, payload: dict[str, Any] | None = None) -> str:
    """Return user-facing answer text from current or legacy Agent outputs.

    Older cached ``ask_papers`` tools returned the complete payload as JSON in
    ``ToolMessage.content``.  New tools keep that payload in ``artifact`` and
    return only the answer as content.  The UI accepts both so a hot-reloaded
    session and historical messages never expose transport JSON as chat text.
    """
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict) and isinstance(decoded.get("answer"), str):
                return decoded["answer"]
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("answer"), str)
            and stripped.startswith("{")
        ):
            return payload["answer"]
        return content
    if isinstance(content, dict):
        if isinstance(content.get("answer"), str):
            return content["answer"]
        return json.dumps(content, ensure_ascii=False, indent=2)
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        return payload["answer"]
    return str(content) if content is not None else ""
