from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.tools.base import ToolException

from langgraph_research_agent.tools.save_file import save_file, save_file_func


@pytest.fixture
def mock_workspace(tmp_path: Path) -> Iterator[Path]:
    with patch("langgraph_research_agent.tools.save_file.settings.workspace", new=tmp_path):
        yield tmp_path


def test_save_file_success(mock_workspace: Path) -> None:
    result = save_file_func("test.txt", "Bonjour le monde")

    assert result == "Content successfully written"

    saved_file = mock_workspace / "test.txt"
    assert saved_file.exists()
    assert saved_file.read_text() == "Bonjour le monde"


def test_save_file_outside_workspace(mock_workspace: Path) -> None:
    result = save_file_func("../hack.txt", "Piratage")

    assert result == "Not accessible"
    assert not (mock_workspace / "../hack.txt").resolve().exists()


def test_save_file_exceptions(mock_workspace: Path) -> None:
    with pytest.raises(ToolException, match="The file was not found"):
        save_file_func("dossier_inconnu/test.txt", "Texte")

    dossier = mock_workspace / "sous_dossier"
    dossier.mkdir()
    with pytest.raises(ToolException, match="Not a valid directory"):
        save_file_func("sous_dossier", "Texte")

    with patch("builtins.open", side_effect=PermissionError):
        with pytest.raises(ToolException, match="Permission error"):
            save_file_func("test_perm.txt", "Texte")


def test_save_file_tool(mock_workspace: Path) -> None:
    result = save_file.invoke({"path": "dossier_inconnu/test.txt", "content": "Texte"})

    assert result == "The file was not found"
