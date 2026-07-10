import wikipediaapi
from langchain_core.tools import StructuredTool, ToolException

from ..utils.logger import logger
from typing import TypedDict


class WikipediaResponse(TypedDict):
    success: bool
    content: str


def wikipedia_func(subject: str) -> WikipediaResponse:
    """Search wikipedia via it's api and return a WikipediaResponse that has success and content.
    Use when you have to search on the wikipedia on a specific subject.
    Do not use for general search and calculator."""
    try:
        logger.info(f"Tool usage: wikipedia with following arguments {subject}")
        wiki_wiki = wikipediaapi.Wikipedia(
            user_agent="MyProjectName (merlin@example.com)",
            language="en",
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )
        result = wiki_wiki.page(subject)
        logger.info(f"Tool return: {result}")
        return {"success": result.exists(), "content": result.text}
    except wikipediaapi.WikipediaException as e:
        raise ToolException(f"A wikipedia API error occured: {e}")
    except Exception as e:
        raise ToolException(f"An unexpected error occurred: {e}")


wikipedia = StructuredTool.from_function(
    func=wikipedia_func, name="wikipedia", handle_tool_error=True
)
