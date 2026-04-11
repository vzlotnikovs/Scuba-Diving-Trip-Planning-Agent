"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path
from typing import Any
# RAG_System_Class raises at import time if this is unset.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key-for-pytest")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

# Ensure project root is on path so "Agent" and "constants" can be imported
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def make_tool_runtime(
    state: dict[str, Any],
    tool_call_id: str = "test-tool-call-id",
) -> Any:
    """Build a LangChain ``ToolRuntime`` for invoking workflow tools in isolation."""
    from langchain.tools import ToolRuntime

    return ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _x: None,
        tool_call_id=tool_call_id,
        store=None,
    )


def tool_message_texts(command: Any) -> list[str]:
    """Extract ``ToolMessage`` string contents from a ``Command`` update."""
    from langchain_core.messages import ToolMessage

    update = getattr(command, "update", None) or {}
    messages = update.get("messages") or []
    out: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            out.append(m.content)
    return out