

from langchain_core.tools import StructuredTool, ToolException
from ..utils.logger import logger
from ..utils.setting import workspace

def save_file_func(path: str, content: str) -> str:
    """Save the given content in a file in the workspace dir.
    Use when you want to save a file for persistance.
    Do not use for search or calculate."""    
    try:
        logger.info(f"Tool usage: save_file with following arguments path={path} and content={content}")
        file_to_get = (workspace / path).resolve()
        if file_to_get.is_relative_to(workspace):
            if file_to_get.is_dir():
                raise IsADirectoryError()
            with open(file_to_get, "w") as f:
                f.write(content)
                return "Content successfully written"
        return "Not accessible"
    except FileNotFoundError:
        raise ToolException("The file was not found")
    except PermissionError:
        raise ToolException("Permission error")
    except IsADirectoryError:
        raise ToolException("Not a valid directory")
    except Exception as e:
        raise ToolException(f"An unexpected error occurred: {e}")

save_file = StructuredTool.from_function(
    func=save_file_func,
    name="save_file",
    handle_tool_error=True
)