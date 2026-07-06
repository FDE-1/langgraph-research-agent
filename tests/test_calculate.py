
import pytest

from langgraph_research_agent.tools.calculator import calculator, calculator_func
from langchain_core.tools.base import ToolException

@pytest.mark.parametrize("expression, expected", [
    ("2 + 2", 4),
    ("10 / 2", 5.0),
    ("5 > 3", True),
    ("'a' + 'b'", "ab")
])
def test_calculator_func_success(expression, expected):
    assert calculator_func(expression) == expected

def test_calculator_func_errors():
    with pytest.raises(ToolException, match="Math Error: You cannot divide by zero."):
        calculator_func("1 / 0")
        
    with pytest.raises(ToolException, match="Syntax Error"):
        calculator_func("2 + * 2")

    with pytest.raises(ToolException, match="Type Error"):
        calculator_func("'texte' + 5")

    with pytest.raises(ToolException, match="Evaluation Error"):
        calculator_func("variable_inconnue + 2")

def test_calculator_tool():
    assert calculator.invoke({"a": "3 * 3"}) == 9
    
    result = calculator.invoke({"a": "1 / 0"})
    assert result == "Math Error: You cannot divide by zero."