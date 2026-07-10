import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from pytest import MonkeyPatch

from langgraph_research_agent.agent import Agent, prompt
from langgraph_research_agent.utils.state import AgentState


class MakeAgent(Protocol):
    """Callable[..., Agent] would smuggle in an implicit Any, which strict mypy rejects."""

    def __call__(
        self,
        system: str = ...,
        funcs: list[BaseTool] | None = ...,
        max_turn: int = ...,
    ) -> Agent: ...


fake_text = SimpleNamespace(type="response.output_text.delta", delta="Bonjour")
fake_done = SimpleNamespace(type="response.completed")
fake_item_done = SimpleNamespace(
    type="response.output_item.done", item=SimpleNamespace(type="message")
)


@tool
def dummy_weather(query: str) -> str:
    """Dummy weather"""
    return f"Météo pour {query}: Ensoleillé"


@pytest.fixture
def mock_chroma() -> MagicMock:
    """Stands in for the Chroma collection; `.add` is what the embed node calls."""
    return MagicMock()


@pytest.fixture
def checkpointer(tmp_path: Path) -> Iterator[SqliteSaver]:
    """A real SqliteSaver, but on a per-test database instead of the repo root."""
    conn = sqlite3.connect(tmp_path / "checkpoint.sqlite", check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


@pytest.fixture
def make_agent(mock_chroma: MagicMock, checkpointer: SqliteSaver) -> MakeAgent:
    """Builds Agents with every external resource injected — no disk, no network, no key."""

    def _make(
        system: str = prompt, funcs: list[BaseTool] | None = None, max_turn: int = 5
    ) -> Agent:
        return Agent(
            system=system,
            funcs=funcs,
            max_turn=max_turn,
            client=MagicMock(),
            memory_collection=mock_chroma,
            checkpointer=checkpointer,
        )

    return _make


def test_agent_initialization(make_agent: MakeAgent) -> None:
    agent = make_agent(system="Tu es un assistant", funcs=[dummy_weather])

    assert agent.graph
    assert agent.system == "Tu es un assistant"
    assert "dummy_weather" in agent.tools_list
    assert agent.tools[0]["name"] == "dummy_weather"
    assert "function" not in agent.tools[0]


def test_agent_run(monkeypatch: MonkeyPatch, make_agent: MakeAgent, mock_chroma: MagicMock) -> None:
    agent = make_agent(funcs=[dummy_weather])

    def fake_execute(state: AgentState) -> list[SimpleNamespace]:
        return [fake_text, fake_item_done, fake_done]

    monkeypatch.setattr(agent, "execute", fake_execute)

    user: dict[str, object] = {"role": "user", "content": "Salut"}
    system: dict[str, object] = {"role": "system", "content": "nothing"}
    messages: list[dict[str, object]] = [system, user]
    initial_state: AgentState = {"messages": messages, "turn": 0, "pending_calls": []}
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = agent.graph.invoke(initial_state, config)

    assert final_state["messages"][-1]["content"] == "Bonjour"

    mock_chroma.add.assert_called_once()
    kwargs = mock_chroma.add.call_args.kwargs
    assert kwargs["documents"] == ["Bonjour"]
    assert kwargs["metadatas"][0]["user_query"] == "Salut"


fake_tool_call = SimpleNamespace(
    type="response.output_item.done",
    item=SimpleNamespace(
        type="function_call",
        name="dummy_weather",
        call_id="call_123",
        arguments='{"query": "Paris"}',
    ),
)


def test_agent_run2(
    monkeypatch: MonkeyPatch, make_agent: MakeAgent, mock_chroma: MagicMock
) -> None:
    agent = make_agent(funcs=[dummy_weather])
    stream_tour1 = [fake_tool_call, fake_done]
    stream_tour2 = [fake_text, fake_item_done, fake_done]
    fake = MagicMock(side_effect=[stream_tour1, stream_tour2])
    monkeypatch.setattr(agent, "execute", fake)

    user: dict[str, object] = {"role": "user", "content": "Salut"}
    system: dict[str, object] = {"role": "system", "content": "nothing"}
    messages: list[dict[str, object]] = [system, user]
    initial_state: AgentState = {"messages": messages, "turn": 0, "pending_calls": []}
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = agent.graph.invoke(initial_state, config)

    assert final_state["messages"][-1]["content"] == "Bonjour"
    assert any(
        isinstance(m, dict) and m.get("type") == "function_call_output"
        for m in final_state["messages"]
    )

    mock_chroma.add.assert_called_once()
    assert mock_chroma.add.call_args.kwargs["documents"] == ["Bonjour"]


def test_agent_run3(
    monkeypatch: MonkeyPatch, make_agent: MakeAgent, mock_chroma: MagicMock
) -> None:
    agent = make_agent(funcs=[dummy_weather])
    stream_tour1 = [fake_text, fake_item_done, fake_done]
    stream_tour2 = [fake_text, fake_item_done, fake_done]
    fake = MagicMock(side_effect=[stream_tour1, stream_tour2])
    monkeypatch.setattr(agent, "execute", fake)

    agent.run("Salut", 1)
    agent.run("Ca va?", 1)

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}
    state = agent.graph.get_state(config)
    messages = state.values["messages"]
    contents = [m.get("content") for m in messages]

    assert "Salut" in contents
    assert "Ca va?" in contents

    assert mock_chroma.add.call_count == 2
