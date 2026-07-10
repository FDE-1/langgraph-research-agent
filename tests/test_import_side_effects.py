"""Importing the package must not open a database, a network client, or read a key.

Run in a subprocess with a scrubbed environment and a throwaway CWD: an in-process
import would be a no-op, since pytest has already imported the module.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "TAVILY_API_KEY"}}
    env["PYTHONPATH"] = str(SRC)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_import_agent_without_api_key(tmp_path: Path) -> None:
    """Import must succeed with no OPENAI_API_KEY in the environment."""
    result = _run("import langgraph_research_agent.agent", tmp_path)

    assert result.returncode == 0, result.stderr


def test_import_agent_creates_no_files(tmp_path: Path) -> None:
    """Import must not create checkpoint.sqlite (or anything else) in the CWD."""
    result = _run("import langgraph_research_agent.agent", tmp_path)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_import_tools_creates_no_files(tmp_path: Path) -> None:
    """The tool modules resolve settings at import, but must not open the Chroma store."""
    result = _run("import langgraph_research_agent.tools.search_memory", tmp_path)

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
