"""Unit tests for ``Agent.workflow`` reducers, tools, and middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from Agent import workflow as wf
from Agent.workflow import (
    certified_reducer,
    dict_merge_reducer,
    disqualify_user,
    enforce_tool_sequence,
    save_trip_summary,
    scalar_reducer,
    search_tavily,
    validate_safety_with_rag,
)
from constants import MAX_TRIP_DAYS, MIN_TRIP_DAYS

from tests.conftest import make_tool_runtime, tool_message_texts


def _override_request(**kwargs: Any) -> MagicMock:
    """Builds a mock model request carrying a narrowed ``tools`` list.

    Args:
        **kwargs: Must include ``tools`` assigned onto the returned mock.

    Returns:
        ``MagicMock`` suitable as the return value of ``req.override``.
    """
    inner = MagicMock()
    inner.tools = kwargs["tools"]
    return inner


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (None, None, None),
        (True, None, True),
        (None, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_certified_reducer(
    a: bool | None, b: bool | None, expected: bool | None
) -> None:
    """Verifies boolean merge semantics for the ``certified`` channel reducer.

    Args:
        a: Prior certified flag from state.
        b: Incoming certified flag from an update.
        expected: Expected merged result.
    """
    assert certified_reducer(a, b) is expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (None, {"x": 1}, {"x": 1}),
        ({"x": 1}, None, {"x": 1}),
        ({"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_dict_merge_reducer(
    a: dict[str, Any] | None,
    b: dict[str, Any] | None,
    expected: dict[str, Any] | None,
) -> None:
    """Merges optional dict slices without mutating unrelated keys.

    Args:
        a: Left-hand summary fragment.
        b: Right-hand summary fragment.
        expected: Expected merged mapping.
    """
    assert dict_merge_reducer(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, None, 1),
        (1, 2, 2),
        (None, "x", "x"),
    ],
)
def test_scalar_reducer(a: Any, b: Any, expected: Any) -> None:
    """Prefers the newest non-``None`` scalar when merging annotated state fields.

    Args:
        a: Existing scalar value.
        b: Incoming scalar value.
        expected: Result of ``scalar_reducer``.
    """
    assert scalar_reducer(a, b) == expected


def test_save_trip_summary_updates_destination() -> None:
    """Persists a destination into both top-level state and nested ``trip_summary``."""
    rt = make_tool_runtime({})
    cmd = save_trip_summary.invoke({"runtime": rt, "destination": "Bali"})
    assert cmd.update["destination"] == "Bali"
    assert cmd.update["trip_summary"]["destination"] == "Bali"


def test_save_trip_summary_invalid_duration_returns_tool_error() -> None:
    """Returns a tool error message when trip duration is outside allowed bounds."""
    rt = make_tool_runtime({})
    cmd = save_trip_summary.invoke({"runtime": rt, "trip_duration": MAX_TRIP_DAYS + 1})
    texts = tool_message_texts(cmd)
    assert texts
    assert "Invalid trip duration" in texts[0]
    assert str(MIN_TRIP_DAYS) in texts[0]
    assert str(MAX_TRIP_DAYS) in texts[0]


def test_save_trip_summary_not_certified_sets_certified_false() -> None:
    """Marks ``certified`` false when the user reports no valid certification."""
    rt = make_tool_runtime({})
    cmd = save_trip_summary.invoke({"runtime": rt, "certification_type": "N/A"})
    assert cmd.update["certified"] is False
    assert cmd.update["certification_type"] == "Not certified"


def test_save_trip_summary_unrecognized_certification_returns_tool_error() -> None:
    """Rejects unknown certification strings without persisting them to state."""
    rt = make_tool_runtime(
        {
            "trip_summary": {
                "destination": "Bali",
                "certification_type": "Open Water",
            }
        }
    )
    cmd = save_trip_summary.invoke(
        {"runtime": rt, "certification_type": "asdfnotacertlevel"}
    )
    texts = tool_message_texts(cmd)
    assert texts
    assert "not recognized" in texts[0].lower()
    assert "certification_type" not in cmd.update


def test_save_trip_summary_canonicalizes_certification() -> None:
    """Maps common shorthand to a canonical certification label."""
    rt = make_tool_runtime({})
    cmd = save_trip_summary.invoke({"runtime": rt, "certification_type": "  aow  "})
    assert cmd.update["certified"] is True
    assert cmd.update["certification_type"] == "Advanced Open Water"


def test_save_trip_summary_all_fields_complete_message() -> None:
    """Emits the completion hint once all required trip summary keys are populated."""
    rt = make_tool_runtime({"trip_summary": {}})
    cmd = save_trip_summary.invoke(
        {
            "runtime": rt,
            "destination": "Maldives",
            "trip_month": "June",
            "trip_duration": 7,
            "certification_type": "AOW",
            "nitrox": True,
        }
    )
    texts = tool_message_texts(cmd)
    assert any("All 5 required fields are collected" in t for t in texts)
    assert cmd.update.get("trip_duration") == 7
    assert cmd.update.get("certification_type") == "Advanced Open Water"


def test_disqualify_user_sets_certified_false() -> None:
    """Forces ``certified`` false and returns a disqualification tool message."""
    rt = make_tool_runtime({})
    cmd = disqualify_user.invoke({"runtime": rt})
    assert cmd.update["certified"] is False
    texts = tool_message_texts(cmd)
    assert any("disqualified" in t.lower() for t in texts)


@patch.object(wf, "_invoke_tavily", return_value="search results")
def test_search_tavily_formats_query_and_sanitizes(mock_invoke: MagicMock) -> None:
    """Builds a Tavily query string from trip state and forwards it to the invoker.

    Args:
        mock_invoke: Patched low-level Tavily call used to capture the query text.
    """
    state = {
        "destination": "Bali",
        "trip_month": "May",
        "certification_type": "OW",
        "trip_duration": 5,
        "nitrox": False,
    }
    rt = make_tool_runtime(state)
    out = search_tavily.invoke({"runtime": rt})
    assert out == "search results"
    mock_invoke.assert_called_once()
    q = mock_invoke.call_args[0][0]
    assert "Bali" in q and "May" in q and "Regular Air" in q


@patch.object(wf, "_invoke_tavily", side_effect=RuntimeError("network down"))
def test_search_tavily_handles_invoke_error(_mock: MagicMock) -> None:
    """Returns a user-facing fallback string when the web search layer raises.

    Args:
        _mock: Patched Tavily invocation that raises ``RuntimeError``.
    """
    rt = make_tool_runtime(
        {
            "destination": "X",
            "trip_month": "Jan",
            "certification_type": "OW",
            "trip_duration": 3,
            "nitrox": True,
        }
    )
    out = search_tavily.invoke({"runtime": rt})
    assert "Web search failed" in out


@patch.object(wf, "RAGSystem")
def test_validate_safety_with_rag_rag_failure_returns_fallback(
    mock_rag: MagicMock,
) -> None:
    """Uses a static fallback message when the RAG stack cannot be initialized.

    Args:
        mock_rag: Patched ``RAGSystem`` accessor that raises during ``get_instance``.
    """
    mock_rag.get_instance.side_effect = RuntimeError("no index")
    out = validate_safety_with_rag.invoke(
        {"itinerary_draft": "Day 1: shallow dive", "nitrox": False}
    )
    assert "Safety validation unavailable" in out


@patch.object(wf, "RAGSystem")
@patch.object(wf, "safety_check_llm")
def test_validate_safety_with_rag_success(
    mock_llm: MagicMock, mock_rag: MagicMock
) -> None:
    """Runs retrieval plus the safety LLM and returns the model's string content.

    Args:
        mock_llm: Patched safety-check language model.
        mock_rag: Patched ``RAGSystem`` providing deterministic context text.
    """
    mock_rag.get_instance.return_value.retrieve_context.return_value = "ctx"
    mock_llm.invoke.return_value = MagicMock(content="Final itinerary text")

    out = validate_safety_with_rag.invoke(
        {"itinerary_draft": "Day 1: dive", "nitrox": True}
    )
    assert out == "Final itinerary text"
    mock_llm.invoke.assert_called_once()


def _tool_mock(tool_name: str) -> MagicMock:
    """Creates a lightweight stand-in tool object exposing ``name``.

    Args:
        tool_name: Identifier expected by ``enforce_tool_sequence`` filtering.

    Returns:
        ``MagicMock`` with ``name`` set to ``tool_name``.
    """
    m = MagicMock()
    m.name = tool_name
    return m


def test_enforce_tool_sequence_disqualified_only_disqualify_tool() -> None:
    """Leaves only ``disqualify_user`` visible when the diver is not certified."""
    captured: dict[str, Any] = {}

    def handler(req: Any) -> Any:
        """Records ordered tool names then returns a dummy model response.

        Args:
            req: Model request whose ``tools`` list was filtered by middleware.

        Returns:
            Placeholder ``MagicMock`` response.
        """
        captured["tools"] = [getattr(t, "name", None) for t in req.tools]
        return MagicMock()

    req = MagicMock()
    req.state = {"certified": False, "trip_summary": {}}
    req.tools = [
        _tool_mock(n)
        for n in (
            "save_trip_summary",
            "disqualify_user",
            "search_tavily",
            "validate_safety_with_rag",
        )
    ]
    req.override = MagicMock(side_effect=_override_request)

    enforce_tool_sequence.wrap_model_call(req, handler)
    assert captured["tools"] == ["disqualify_user"]


def test_enforce_tool_sequence_incomplete_summary_limits_tools() -> None:
    """Exposes summary and disqualify tools until the trip summary is complete."""
    captured: dict[str, Any] = {}

    def handler(req: Any) -> Any:
        """Records sorted tool names exposed after middleware rewriting.

        Args:
            req: Model request whose ``tools`` list was filtered by middleware.

        Returns:
            Placeholder ``MagicMock`` response.
        """
        captured["tools"] = sorted(getattr(t, "name", None) for t in req.tools)
        return MagicMock()

    req = MagicMock()
    req.state = {"certified": True, "trip_summary": {"destination": "Bali"}}
    req.tools = [
        _tool_mock(n)
        for n in (
            "save_trip_summary",
            "disqualify_user",
            "search_tavily",
            "validate_safety_with_rag",
        )
    ]
    req.override = MagicMock(side_effect=_override_request)

    enforce_tool_sequence.wrap_model_call(req, handler)
    assert captured["tools"] == sorted(["disqualify_user", "save_trip_summary"])


def test_enforce_tool_sequence_complete_summary_exposes_all_tools() -> None:
    """Allows search and RAG tools once every required trip summary field is set."""
    captured: dict[str, Any] = {}

    def handler(req: Any) -> Any:
        """Records sorted tool names when the summary gate is satisfied.

        Args:
            req: Model request whose ``tools`` list was filtered by middleware.

        Returns:
            Placeholder ``MagicMock`` response.
        """
        captured["tools"] = sorted(getattr(t, "name", None) for t in req.tools)
        return MagicMock()

    summary = {
        "destination": "Bali",
        "trip_month": "May",
        "trip_duration": 7,
        "certification_type": "OW",
        "nitrox": False,
    }
    req = MagicMock()
    req.state = {"certified": True, "trip_summary": summary}
    req.tools = [
        _tool_mock(n)
        for n in (
            "save_trip_summary",
            "disqualify_user",
            "search_tavily",
            "validate_safety_with_rag",
        )
    ]
    req.override = MagicMock(side_effect=_override_request)

    enforce_tool_sequence.wrap_model_call(req, handler)
    assert captured["tools"] == sorted(
        [
            "disqualify_user",
            "save_trip_summary",
            "search_tavily",
            "validate_safety_with_rag",
        ]
    )
