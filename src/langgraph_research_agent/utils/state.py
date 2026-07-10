from typing import Annotated, TypedDict
import operator


class AgentState(TypedDict):
    """State that is shared between nodes"""

    messages: Annotated[list[dict[str, object]], operator.add]
    turn: int
    pending_calls: list[dict[str, object]]
