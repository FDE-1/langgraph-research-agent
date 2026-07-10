from typing import Annotated, cast

from langchain_core.tools import InjectedToolArg, StructuredTool, ToolException
from tavily import TavilyClient
from tavily.errors import InvalidAPIKeyError, TimeoutError, UsageLimitExceededError

from ..utils.logger import logger


def web_search_func(
    client: Annotated[TavilyClient, InjectedToolArg], query: str, max_result: int = 5
) -> dict[str, object]:
    """Search the web via Tavily and return a dict.
    Use when you have to search on a general subject.
    Do not use for specific search and calculator."""
    try:
        logger.info(f"query: {query}")
        result = client.search(query, max_results=max_result)
        logger.debug(f"result: {result}")
        return cast(dict[str, object], result)
    except InvalidAPIKeyError as e:
        raise ToolException("Invalid API Key") from e
    except UsageLimitExceededError as e:
        raise ToolException("Out of credits") from e
    except TimeoutError as e:
        raise ToolException("Search took too long") from e
    except Exception as e:
        raise ToolException(str(e)) from e


web_search = StructuredTool.from_function(
    func=web_search_func, name="web_search", handle_tool_error=True
)
