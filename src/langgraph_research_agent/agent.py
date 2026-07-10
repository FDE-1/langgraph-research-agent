import json
import os
import sqlite3
import uuid
from typing import cast

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from openai import OpenAI, Stream
from openai.types.responses import (
    FunctionToolParam,
    ResponseInputParam,
    ResponseStreamEvent,
)
from ratelimit import limits

from .utils.logger import logger
from .utils.setting import Settings, get_settings
from .utils.state import AgentState

prompt = """ Tu es un assistant qui répond aux questions.\n
Tu as accès à des outils, utilise-les quand c'est pertinent.\n
N'invente jamais une info que tu peux vérifier avec un outil.\n
Si une recherche échoue, reformule plutôt que d'abandonner"""


def build_memory_collection(settings: Settings) -> Collection:
    """Open the Chroma store and return the memory collection."""
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY", model_name="text-embedding-3-small"
    )
    chroma_client = chromadb.PersistentClient(path=settings.client_path)
    return chroma_client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=openai_ef,  # type: ignore[arg-type]
    )


def build_checkpointer(settings: Settings) -> SqliteSaver:
    """Open the checkpoint database at the configured path."""
    conn = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    return SqliteSaver(conn)


def build_openai_client(settings: Settings) -> OpenAI:
    """Build the OpenAI client."""
    return OpenAI(api_key=settings.openai_api_key)


class Agent:
    def __init__(
        self,
        system: str = prompt,
        funcs: list[BaseTool] | None = None,
        max_turn: int = 5,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        memory_collection: Collection | None = None,
        checkpointer: SqliteSaver | None = None,
    ) -> None:
        """Initialize the agent."""
        self.system = system
        self.max_turn = max_turn
        self.settings = settings or get_settings()
        self.client = client or build_openai_client(self.settings)
        self.memory_collection = (
            memory_collection
            if memory_collection is not None
            else build_memory_collection(self.settings)
        )
        self.checkpointer = checkpointer or build_checkpointer(self.settings)
        funcs = funcs or []
        self.tools: list[FunctionToolParam] = []
        self.tools_list: dict[str, BaseTool] = {f.name: f for f in funcs}

        for t in funcs:
            openai_schema = convert_to_openai_function(t)

            if "parameters" in openai_schema:
                openai_schema["parameters"]["additionalProperties"] = False

            tool_param: FunctionToolParam = {
                "type": "function",
                "name": openai_schema["name"],
                "description": openai_schema["description"],
                "parameters": openai_schema.get(
                    "parameters",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                ),
                "strict": True,
            }
            self.tools.append(tool_param)
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason)
        graph.add_node("action", self._action)
        graph.add_node("observe", self._observe)
        graph.add_node("embed", self._embed)
        graph.add_node("max_turn", self._max_turn)
        graph.add_conditional_edges(
            "reason", self.is_type_function_call_, {True: "action", False: "embed"}
        )
        graph.add_edge("action", "observe")
        graph.add_conditional_edges(
            "observe", self.is_max_turn, {True: "max_turn", False: "reason"}
        )
        graph.add_edge("max_turn", "embed")
        graph.add_edge("embed", END)

        graph.set_entry_point("reason")
        self.graph = graph.compile(checkpointer=self.checkpointer)

    def is_type_function_call_(self, state: AgentState) -> bool:
        """Check if there is a need to call a function"""
        return state["pending_calls"] != []

    def is_max_turn(self, state: AgentState) -> bool:
        """Check if current turn is max turn"""
        return state["turn"] >= self.max_turn

    def _reason(self, state: AgentState) -> dict[str, object]:
        """Node of reason"""
        turn = state.get("turn", 0)
        logger.info(f"turn {turn}: reasoning")
        logger.debug(f"state.turn={turn} state.messages={state['messages']}")
        stream = self.execute(state)
        text = ""
        pending_calls = []
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
            elif event.type == "response.output_item.done":
                if event.item.type == "function_call":
                    logger.info(f"chose tool: {event.item.name}")
                    logger.debug(
                        f"tool_call name={event.item.name} "
                        f"call_id={event.item.call_id} arguments={event.item.arguments}"
                    )
                    pending_calls.append(
                        {
                            "type": "function_call",
                            "call_id": event.item.call_id,
                            "name": event.item.name,
                            "arguments": event.item.arguments,
                        }
                    )
        if text:
            logger.info("chose to answer")
            logger.debug(f"answer={text!r}")
            return {"messages": [{"role": "assistant", "content": text}], "pending_calls": []}
        else:
            logger.debug(f"pending_calls={pending_calls}")
            return {"messages": pending_calls, "pending_calls": pending_calls}

    def _action(self, state: AgentState) -> dict[str, object]:
        """Node of action that call all pending_calls to function"""
        result = []
        logger.debug(f"pending_calls={state['pending_calls']}")
        for pending_call in state["pending_calls"]:
            event = pending_call
            name = cast(str, event["name"])
            logger.info(f"calling tool: {name}")
            args = json.loads(cast(str, event["arguments"]))
            logger.debug(f"invoking {name} with args={args}")
            output = self.tools_list[name].invoke(args)
            logger.debug(f"{name} returned {output!r}")
            result.append(
                {
                    "type": "function_call_output",
                    "call_id": event["call_id"],
                    "output": json.dumps(output, default=str),
                }
            )
        return {"messages": result, "pending_calls": []}

    def _observe(self, state: AgentState) -> dict[str, object]:
        """Node that observe the result"""
        logger.info("turn complete")
        logger.debug(f"turn {state['turn']} -> {state['turn'] + 1} (max_turn={self.max_turn})")
        return {"turn": state["turn"] + 1}

    def _embed(self, state: AgentState) -> dict[str, object]:
        """Node that embeded the response"""
        logger.info("saving answer to memory")
        if self.is_max_turn(state):
            logger.info("turn budget exhausted, skipping memory save")
            return {}
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.get("role") == "assistant" and last_message.get("content"):
            assistant_text = cast(str, last_message["content"])

            user_text = "Unknown question"
            for msg in reversed(messages[:-1]):
                if msg.get("role") == "user":
                    user_text = cast(str, msg["content"])
                    break

            doc_id = str(uuid.uuid4())
            logger.debug(f"doc_id={doc_id} user_query={user_text!r} document={assistant_text!r}")

            try:
                self.memory_collection.add(
                    documents=[assistant_text],
                    metadatas=[{"user_query": user_text, "role": "assistant"}],
                    ids=[doc_id],
                )
                logger.info(f"memory saved (id={doc_id})")

            except Exception as e:
                logger.error(f"memory save failed: {e}")

        return {}

    def _max_turn(self, state: AgentState) -> dict[str, object]:
        logger.debug(f"budget exhausted at turn={state['turn']}")
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        f"Max number of turn {state['turn']} was reached "
                        "try to make the question shorter"
                    ),
                }
            ]
        }

    @limits(calls=10, period=60)  # type: ignore[untyped-decorator]
    def execute(self, state: AgentState) -> Stream[ResponseStreamEvent]:
        """Excute the prompt with the message given in the state"""
        completion = self.client.responses.create(
            model="gpt-4o",
            temperature=0,
            input=cast(ResponseInputParam, state["messages"]),
            tools=self.tools,
            stream=True,
        )
        return completion

    def run(self, message: str, thread_id: int | str = 0) -> str:
        """Run the agent on the given prompt"""
        config: RunnableConfig = {"configurable": {"thread_id": str(thread_id)}}
        existing = self.checkpointer.get(config)
        user: dict[str, object] = {"role": "user", "content": message}
        messages: list[dict[str, object]]
        if existing is None:
            system: dict[str, object] = {"role": "system", "content": self.system}
            messages = [system, user]
        else:
            messages = [user]
        initial_state: AgentState = {"messages": messages, "turn": 0, "pending_calls": []}
        logger.debug(f"thread_id={thread_id} resumed={existing is not None}")
        logger.debug(f"initial_state={initial_state}")
        final_state = self.graph.invoke(initial_state, config)
        logger.debug(f"final turn={final_state['turn']} messages={final_state['messages']}")
        return cast(str, final_state["messages"][-1]["content"])

    def draw(self) -> str:
        """Return the graph as a mermaid diagram."""
        return self.graph.get_graph().draw_mermaid()
