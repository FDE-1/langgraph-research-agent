from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.tools.base import ToolException

from langgraph_research_agent.tools.search_memory import search_memory, search_memory_func


@pytest.fixture
def mock_chroma() -> Iterator[tuple[MagicMock, MagicMock]]:
    with patch(
        "langgraph_research_agent.tools.search_memory.chromadb.PersistentClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        yield mock_client, mock_collection


def test_search_memory_success(mock_chroma: tuple[MagicMock, MagicMock]) -> None:
    mock_client, mock_collection = mock_chroma

    mock_collection.query.return_value = {
        "documents": [["Mémoire 1", "Mémoire 2"]],
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]],
    }

    result = search_memory_func(query="IA", nb_results=2)

    assert result == ["Mémoire 1", "Mémoire 2"]

    mock_client.get_collection.assert_called_once()
    mock_collection.query.assert_called_once_with(query_texts=["IA"], n_results=2)


def test_search_memory_value_error(mock_chroma: tuple[MagicMock, MagicMock]) -> None:
    _, mock_collection = mock_chroma

    mock_collection.query.side_effect = ValueError("n_results ne peut pas être négatif")

    with pytest.raises(ToolException, match="Erreur de paramètre"):
        search_memory_func(query="IA", nb_results=-1)


def test_search_memory_chroma_error(mock_chroma: tuple[MagicMock, MagicMock]) -> None:
    _, mock_collection = mock_chroma

    mock_collection.query.side_effect = chromadb.errors.InternalError("Base corrompue")

    with pytest.raises(ToolException, match="Erreur interne Chroma"):
        search_memory_func(query="IA")


def test_search_memory_tool(mock_chroma: tuple[MagicMock, MagicMock]) -> None:
    mock_client, _ = mock_chroma
    mock_client.get_collection.side_effect = Exception("Erreur de connexion au disque")

    result = search_memory.invoke({"query": "Test"})

    assert "Erreur d'embedding ou inattendue" in result
