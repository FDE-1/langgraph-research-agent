import pytest
from unittest.mock import patch, MagicMock
from langchain_core.tools.base import ToolException
import wikipediaapi

from langgraph_research_agent.tools.wikipedia import wikipedia_func, wikipedia

@pytest.fixture
def mock_wikipedia_class():
    with patch("langgraph_research_agent.tools.wikipedia.wikipediaapi.Wikipedia") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance

def test_wikipedia_success(mock_wikipedia_class):
    mock_page = MagicMock()
    mock_page.exists.return_value = True
    mock_page.text = "Voici le contenu de la page."
    
    mock_wikipedia_class.page.return_value = mock_page
    
    result = wikipedia_func("Python (programming language)")
    
    assert result["success"] is True
    assert result["content"] == "Voici le contenu de la page."
    mock_wikipedia_class.page.assert_called_once_with("Python (programming language)")

def test_wikipedia_not_found(mock_wikipedia_class):
    mock_page = MagicMock()
    mock_page.exists.return_value = False
    mock_page.text = ""
    
    mock_wikipedia_class.page.return_value = mock_page
    
    result = wikipedia_func("UneRechercheQuiNexistePas123")
    
    assert result["success"] is False
    assert result["content"] == ""

def test_wikipedia_api_error(mock_wikipedia_class):
    mock_wikipedia_class.page.side_effect = wikipediaapi.WikipediaException("API Down")
    
    with pytest.raises(ToolException, match="A wikipedia API error occured"):
        wikipedia_func("Test")

def test_wikipedia_tool(mock_wikipedia_class):
    mock_wikipedia_class.page.side_effect = Exception("Erreur inattendue")
    
    result = wikipedia.invoke({"subject": "Test"})
    
    assert "An unexpected error occurred" in result