from langchain_core.tools import StructuredTool, ToolException

from ..utils.logger import logger
from ..utils.setting import get_settings

settings = get_settings()


def save_file_func(path: str, content: str) -> str:
    """Save the given content in a file in the workspace dir.
    Use when you want to save a file for persistance.
    Do not use for search or calculate."""
    try:
        logger.info(f"path: {path} ({len(content)} chars)")
        file_to_get = (settings.workspace / path).resolve()
        logger.debug(f"workspace={settings.workspace} resolved={file_to_get}")
        if file_to_get.is_relative_to(settings.workspace):
            if file_to_get.is_dir():
                raise IsADirectoryError()
            with open(file_to_get, "w") as f:
                f.write(content)
                return "Content successfully written"
        return "Not accessible"
    except FileNotFoundError as e:
        raise ToolException("The file was not found") from e
    except PermissionError as e:
        raise ToolException("Permission error") from e
    except IsADirectoryError as e:
        raise ToolException("Not a valid directory") from e
    except Exception as e:
        raise ToolException(f"An unexpected error occurred: {e}") from e


save_file = StructuredTool.from_function(
    func=save_file_func, name="save_file", handle_tool_error=True
)
