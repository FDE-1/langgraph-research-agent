import simpleeval
from langchain_core.tools import StructuredTool, ToolException

from ..utils.logger import logger


def calculator_func(a: str) -> int | float | bool | str:
    """Use the given expression and calculate the expression.
    Use if you have an expression you need to calculate. Do not use
    it for search or research."""
    try:
        logger.info(f"Tool usage: calculator with following arguments {a}")
        result: int = simpleeval.simple_eval(a)
        logger.info(f"Tool return: {result}")
        return result
    except ZeroDivisionError as e:
        raise ToolException("Math Error: You cannot divide by zero.") from e
    except TypeError as e:
        raise ToolException(
            "Type Error: Invalid operation between different types (e.g., adding text to a number)."
        ) from e
    except ValueError as e:
        raise ToolException(
            "Value Error: Invalid value passed to a function or math operation."
        ) from e
    except SyntaxError as e:
        raise ToolException(
            "Syntax Error: The expression is malformed and cannot be parsed."
        ) from e
    except simpleeval.InvalidExpression as e:
        raise ToolException(f"Evaluation Error: {e}") from e
    except Exception as e:
        raise ToolException(f"An unexpected error occurred: {e}") from e


calculator = StructuredTool.from_function(
    func=calculator_func, name="calculator", handle_tool_error=True
)
