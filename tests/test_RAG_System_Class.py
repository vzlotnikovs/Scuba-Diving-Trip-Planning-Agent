"""Tests for ``Agent.RAG_System_Class`` with heavy dependencies mocked."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from Agent.RAG_System_Class import RAGSystem


@pytest.fixture(autouse=True)
def reset_rag_singleton() -> Generator[None, None, None]:
    """Ensure ``get_instance`` tests do not leak state across cases."""
    RAGSystem._instance = None
    yield
    RAGSystem._instance = None


def test_rag_system_init_sets_config_from_constants() -> None:
    """Constructs ``RAGSystem`` with sources and vector store unset before indexing."""
    rag = RAGSystem()
    assert rag.sources is None
    assert rag.vector_store is None
    assert isinstance(rag.K_CONSTANT, int)


@patch.object(RAGSystem, "get_or_create_vector_store")
@patch.object(RAGSystem, "load_source_content")
def test_get_instance_initializes_once(
    mock_load: MagicMock, mock_vs: MagicMock
) -> None:
    """Loads sources and builds the vector store only on the first ``get_instance`` call.

    Args:
        mock_load: Patched ``load_source_content``.
        mock_vs: Patched ``get_or_create_vector_store``.
    """
    mock_load.return_value = [MagicMock(page_content="x")]
    mock_vs.return_value = MagicMock()

    a = RAGSystem.get_instance()
    b = RAGSystem.get_instance()
    assert a is b
    mock_load.assert_called_once()
    mock_vs.assert_called_once()


def test_retrieve_context_formats_results() -> None:
    """Joins document lines and prefixes each hit with its ``source`` metadata."""
    doc = MagicMock()
    doc.metadata = {"source": "guide.pdf"}
    doc.page_content = "line one\nline two"

    rag = RAGSystem()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.return_value = [doc]
    rag.K_CONSTANT = 4

    text = rag.retrieve_context("nitrox limits")
    assert "Source: guide.pdf" in text
    assert "line one line two" in text
    rag.vector_store.similarity_search.assert_called_once_with("nitrox limits", k=4)


def test_retrieve_context_missing_vector_store_raises() -> None:
    """Raises ``RuntimeError`` when ``retrieve_context`` runs without a store."""
    rag = RAGSystem()
    rag.vector_store = None
    with pytest.raises(RuntimeError, match="vector_store must be initialized"):
        rag.retrieve_context("q")


def test_retrieve_context_search_failure_wraps_runtime_error() -> None:
    """Wraps underlying search errors in a ``RuntimeError`` with a stable message."""
    rag = RAGSystem()
    rag.vector_store = MagicMock()
    rag.vector_store.similarity_search.side_effect = ValueError("boom")
    rag.K_CONSTANT = 2
    with pytest.raises(RuntimeError, match="similarity search failed"):
        rag.retrieve_context("q")
