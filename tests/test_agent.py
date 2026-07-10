import uuid
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from pytest import MonkeyPatch
from langchain_core.runnables import RunnableConfig

from langgraph_research_agent.agent import Agent
from langgraph_research_agent.utils.state import AgentState
from langchain_core.tools import tool

fake_text = SimpleNamespace(type="response.output_text.delta", delta="Bonjour")
fake_done = SimpleNamespace(type="response.completed")
fake_item_done = SimpleNamespace(
    type="response.output_item.done", item=SimpleNamespace(type="message")
)


@tool
def dummy_weather(query: str) -> str:
    """Dummy weather"""
    return f"Météo pour {query}: Ensoleillé"


@pytest.fixture(autouse=True)
def mock_chroma(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_add = MagicMock()
    monkeypatch.setattr("langgraph_research_agent.agent.memory_collection.add", mock_add)
    return mock_add


def test_agent_initialization() -> None:
    agent = Agent(system="Tu es un assistant", funcs=[dummy_weather])

    assert agent.graph
    assert agent.system == "Tu es un assistant"
    assert "dummy_weather" in agent.tools_list
    assert agent.tools[0]["name"] == "dummy_weather"
    assert "function" not in agent.tools[0]


def test_agent_run(monkeypatch: MonkeyPatch, mock_chroma: MagicMock) -> None:
    agent = Agent(funcs=[dummy_weather])

    def fake_execute(State: AgentState) -> list[SimpleNamespace]:
        return [fake_text, fake_item_done, fake_done]

    monkeypatch.setattr(agent, "execute", fake_execute)

    user = {"role": "user", "content": "Salut"}
    system = {"role": "system", "content": "nothing"}
    messages = [system, user]
    initial_state: AgentState = {"messages": messages, "turn": 0}
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = agent.graph.invoke(initial_state, config)

    assert final_state["messages"][-1]["content"] == "Bonjour"

    mock_chroma.assert_called_once()
    kwargs = mock_chroma.call_args.kwargs
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


def test_agent_run2(monkeypatch: MonkeyPatch, mock_chroma: MagicMock) -> None:
    agent = Agent(funcs=[dummy_weather])
    stream_tour1 = [fake_tool_call, fake_done]
    stream_tour2 = [fake_text, fake_item_done, fake_done]
    fake = MagicMock(side_effect=[stream_tour1, stream_tour2])
    monkeypatch.setattr(agent, "execute", fake)

    user = {"role": "user", "content": "Salut"}
    system = {"role": "system", "content": "nothing"}
    messages = [system, user]
    initial_state: AgentState = {"messages": messages, "turn": 0}
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = agent.graph.invoke(initial_state, config)

    assert final_state["messages"][-1]["content"] == "Bonjour"
    assert any(
        isinstance(m, dict) and m.get("type") == "function_call_output"
        for m in final_state["messages"]
    )

    mock_chroma.assert_called_once()
    assert mock_chroma.call_args.kwargs["documents"] == ["Bonjour"]


def test_agent_run3(monkeypatch: MonkeyPatch, mock_chroma: MagicMock) -> None:
    agent = Agent(funcs=[dummy_weather])
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

    assert mock_chroma.call_count == 2
