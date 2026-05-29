import re
import numexpr
from langchain.tools import tool

SAFE_EXPR = re.compile(r"^[0-9\.\+\-\*\/\(\)\,\s\%\^a-zA-Z_]+$")

MAX_LENGTH = 200

def validate_expression(
    expression: str,
) -> None:
    if len(expression) > MAX_LENGTH:
        raise ValueError(
            "Expression too long."
        )

    if not SAFE_EXPR.match(expression):
        raise ValueError(
            "Invalid characters detected."
        )

@tool
def calculator(
    expression: str,
) -> str:
    """
    Perform arithmetic calculations.
    Examples:
        2 + 2
        sqrt(144)
        120 * 0.15
        (50 + 20) / 2
    """

    try:
        validate_expression(expression)

        result = numexpr.evaluate(
            expression,
            global_dict={},
            local_dict={},
        )

        return f"Result: {float(result)}"

    except Exception as e:
        return f"Calculation error: {str(e)}"