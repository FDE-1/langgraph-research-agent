
from langchain_core.tools import StructuredTool, ToolException, InjectedToolArg

from ..utils.logger import logger
from tavily import TavilyClient
from tavily.errors import (
    InvalidAPIKeyError, 
    UsageLimitExceededError, 
    TimeoutError
)
from typing import Annotated, cast

def web_search_func(client: Annotated[TavilyClient, InjectedToolArg], query : str, max_result: int = 5) -> dict[str, object]:
    """Search the web via Tavily and return a dict.
    Use when you have to search on a general subject.
    Do not use for specific search and calculator."""
    try:
        logger.info(f"Tool usage: web_search with following arguments {query}")
        result = client.search(query, max_results=max_result)
        logger.info(f"Tool return: {result}")
        return cast(dict[str, object], result)
    except InvalidAPIKeyError:
        raise ToolException("Invalid API Key")
    except UsageLimitExceededError:
        raise ToolException("Out of credits")
    except TimeoutError:
        raise ToolException("Search took too long")
    except Exception as e:
        raise ToolException(str(e))
    

web_search = StructuredTool.from_function(
    func=web_search_func,
    name="web_search",
    handle_tool_error=True
)