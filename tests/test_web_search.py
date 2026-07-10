import pytest
from unittest.mock import MagicMock
from langchain_core.tools.base import ToolException
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError, TimeoutError

from langgraph_research_agent.tools.web_search import web_search_func


@pytest.fixture
def mock_client():
    return MagicMock()


def test_web_search_success(mock_client):
    expected_response = {"results": [{"title": "Titre", "content": "Contenu mocké"}]}
    mock_client.search.return_value = expected_response

    result = web_search_func(client=mock_client, query="Météo Paris", max_result=3)

    assert result == expected_response
    mock_client.search.assert_called_once_with("Météo Paris", max_results=3)


def test_web_search_exceptions(mock_client):
    mock_client.search.side_effect = InvalidAPIKeyError("Mauvaise clé")
    with pytest.raises(ToolException, match="Invalid API Key"):
        web_search_func(client=mock_client, query="Test")

    mock_client.search.side_effect = UsageLimitExceededError("Plus de crédits")
    with pytest.raises(ToolException, match="Out of credits"):
        web_search_func(client=mock_client, query="Test")

    mock_client.search.side_effect = TimeoutError("Délai dépassé")
    with pytest.raises(ToolException, match="Search took too long"):
        web_search_func(client=mock_client, query="Test")

    mock_client.search.side_effect = ValueError("Erreur bizarre")
    with pytest.raises(ToolException, match="Erreur bizarre"):
        web_search_func(client=mock_client, query="Test")
