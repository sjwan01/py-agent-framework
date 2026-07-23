"""MessageSerializer — Pydantic AI messages ↔ JSONB dict.

Decouples SessionManager adapters from Pydantic AI internal schema.
Uses dataclasses.asdict for serialization and kind-discriminated
reconstruction for deserialization.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import ujson


class MessageSerializer:
    """Serialize and deserialize Pydantic AI messages for storage."""

    def serialize(self, msg) -> str:
        """Message → JSON string."""
        d = asdict(msg)
        _convert_datetimes(d)
        return ujson.dumps(d)

    def deserialize(self, data: str) -> object:
        """JSON string → Message. Uses 'kind' field to discriminate."""
        from pydantic_ai.messages import ModelRequest, ModelResponse
        d = ujson.loads(data) if isinstance(data, str) else data
        kind = d.get("kind", "")
        if kind == "request":
            return _dict_to_request(d)
        if kind == "response":
            return _dict_to_response(d)
        raise ValueError(f"Unknown message kind: {kind}")


def _dict_to_request(d: dict):
    from pydantic_ai.messages import ModelRequest, UserPromptPart, ToolReturnPart
    parts = []
    for p in d.get("parts", []):
        pk = p.get("part_kind", "")
        if pk == "user-prompt":
            parts.append(UserPromptPart(content=p.get("content", ""), timestamp=p.get("timestamp")))
        elif pk == "tool-return":
            parts.append(ToolReturnPart(
                tool_name=p.get("tool_name", ""),
                content=p.get("content", ""),
                tool_call_id=p.get("tool_call_id", ""),
                timestamp=p.get("timestamp"),
            ))
    return ModelRequest(
        parts=parts,
        kind="request",
        run_id=d.get("run_id"),
        conversation_id=d.get("conversation_id"),
        instructions=d.get("instructions"),
        timestamp=d.get("timestamp"),
    )


def _dict_to_response(d: dict):
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    parts = []
    for p in d.get("parts", []):
        pk = p.get("part_kind", "")
        if pk == "text":
            parts.append(TextPart(content=p.get("content", "")))
        elif pk == "tool-call":
            parts.append(ToolCallPart(
                tool_name=p.get("tool_name", ""),
                args=p.get("args", {}),
                tool_call_id=p.get("tool_call_id", ""),
            ))
    return ModelResponse(
        parts=parts,
        kind="response",
        run_id=d.get("run_id"),
        conversation_id=d.get("conversation_id"),
        model_name=d.get("model_name"),
        timestamp=d.get("timestamp"),
    )


def _convert_datetimes(d: dict) -> None:
    """Recursively convert datetime values to ISO strings in-place."""
    from datetime import datetime
    for key, value in d.items():
        if isinstance(value, datetime):
            d[key] = value.isoformat()
        elif isinstance(value, dict):
            _convert_datetimes(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _convert_datetimes(item)
