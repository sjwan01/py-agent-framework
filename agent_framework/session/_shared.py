"""Session shared utilities — roles, serialization, no public API."""
from __future__ import annotations

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

# DB 按“每行一条消息”存储，不是整个消息列表，所以用单消息 TypeAdapter。
_MessageAdapter: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


def _infer_role(msg) -> str:
    """根据消息类型和第一个 part 推断角色，写入 DB 的 role 列。

    这只是标记用途，不影响反序列化后的消息对象本身。
    """
    # ModelRequest：如果第一个 part 有 tool_name 属性，说明是 tool 返回消息。
    if isinstance(msg, ModelRequest):
        parts = msg.parts
        if parts and hasattr(parts[0], "tool_name"):
            return "tool"
        return "user"
    # ModelResponse：assistant 的回复。
    if isinstance(msg, ModelResponse):
        return "assistant"
    return "unknown"
