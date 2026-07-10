import json
from typing import cast
from openai import OpenAI, Stream
from openai.types.responses import (
    FunctionToolParam,
    ResponseInputParam,
    ResponseStreamEvent,
)
from ratelimit import limits
from langgraph.graph import StateGraph, END
from .utils.state import AgentState
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.runnables import RunnableConfig
from .utils.logger import logger
import uuid
from chromadb.utils import embedding_functions
from langchain_core.tools import BaseTool
import chromadb
from langchain_core.utils.function_calling import convert_to_openai_function
from .utils.setting import client_path, collection_name, openapi_key

chroma_client = chromadb.PersistentClient(path=str(client_path))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key_env_var="OPENAI_API_KEY", model_name="text-embedding-3-small"
)

memory_collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef,  # type: ignore[arg-type]  # chroma's own EF, stub generic mismatch
)
conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)

client = OpenAI(api_key=openapi_key)

prompt = """ Tu es un assistant qui répond aux questions.\n
Tu as accès à des outils, utilise-les quand c'est pertinent.\n
N'invente jamais une info que tu peux vérifier avec un outil.\n
Si une recherche échoue, reformule plutôt que d'abandonner"""


class Agent:
    def __init__(
        self, system: str = prompt, funcs: list[BaseTool] | None = None, max_turn: int = 5
    ) -> None:
        """Initialize the agent"""
        self.system = system
        self.max_turn = max_turn
        funcs = funcs or []
        self.tools: list[FunctionToolParam] = []
        self.tools_list: dict[str, BaseTool] = {f.name: f for f in funcs}
        self.checkpointer = SqliteSaver(conn)

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
        result = self.graph.get_graph().draw_mermaid()
        print(result)

    def is_type_function_call_(self, state: AgentState) -> bool:
        """Check if there is a need to call a function"""
        return state["pending_calls"] != []

    def is_max_turn(self, state: AgentState) -> bool:
        """Check if current turn is max turn"""
        return state["turn"] >= self.max_turn

    def _reason(self, state: AgentState) -> dict[str, object]:
        """Node of reason"""
        turn = state.get("turn", 0)
        logger.info(f"--- [Turn {turn}] Agent is Reasoning ---")
        stream = self.execute(state)
        text = ""
        pending_calls = []
        for event in stream:
            if event.type == "response.output_text.delta":
                text += event.delta
            elif event.type == "response.output_item.done":
                if event.item.type == "function_call":
                    logger.info(f"[Choice] Agent choose to use tools : {event.item.name}")
                    pending_calls.append(
                        {
                            "type": "function_call",
                            "call_id": event.item.call_id,
                            "name": event.item.name,
                            "arguments": event.item.arguments,
                        }
                    )
        if text:
            logger.info("[Choice] Agent choose to respond")
            return {"messages": [{"role": "assistant", "content": text}], "pending_calls": []}
        else:
            return {"messages": pending_calls, "pending_calls": pending_calls}

    def _action(self, state: AgentState) -> dict[str, object]:
        """Node of action that call all pending_calls to function"""
        result = []
        for pending_call in state["pending_calls"]:
            event = pending_call
            name = cast(str, event["name"])
            logger.info(f"---  Action : going to {name} ---")
            args = json.loads(cast(str, event["arguments"]))
            output = self.tools_list[name].invoke(args)
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
        logger.info("--- Observation : End of current turn ---")
        return {"turn": state["turn"] + 1}

    def _embed(self, state: AgentState) -> dict[str, object]:
        """Node that embeded the response"""
        logger.info("--- Embeding : Saving in memory ---")
        if self.is_max_turn(state):
            logger.info("--- Embeding : Max turn reach no save in memory ---")
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

            try:
                memory_collection.add(
                    documents=[assistant_text],
                    metadatas=[{"user_query": user_text, "role": "assistant"}],
                    ids=[doc_id],
                )
                logger.info(f"Memory saved to Chroma (ID: {doc_id})")

            except Exception as e:
                logger.error(f"Error when embeding and saving : {e}")

        return {}

    def _max_turn(self, state: AgentState) -> dict[str, object]:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Max number of turn {state['turn']} was reached try to make the question shorter",
                }
            ]
        }

    @limits(calls=10, period=60)  # type: ignore[untyped-decorator]  # ratelimit decorator is untyped
    def execute(self, state: AgentState) -> Stream[ResponseStreamEvent]:
        """Excute the prompt with the message given in the state"""
        completion = client.responses.create(
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
        final_state = self.graph.invoke(initial_state, config)
        return cast(str, final_state["messages"][-1]["content"])
