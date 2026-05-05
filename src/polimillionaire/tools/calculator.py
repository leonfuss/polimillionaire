"""sympy-backed calculator for ReAct-style strategies.

The model emits an `expression` string; we sympify and evalf it. Errors are
returned as `ERROR: ...` strings so the model sees them in the next turn and
can self-correct (rather than the strategy aborting silently).

We deliberately use `sympy.sympify` rather than `eval`: sympify's parser
rejects arbitrary Python (no imports, no attribute access, no calls outside
sympy's allow-list) so a malformed or hostile expression fails closed.
"""

from __future__ import annotations

import sympy


def calc(expression: str) -> str:
    """Evaluate a math expression. Returns a numeric string or `ERROR: <msg>`.

    Examples:
        calc("2 + 2")          -> "4"
        calc("sqrt(2)")        -> "1.41421356237310"
        calc("1/3")            -> "0.333333333333333"
        calc("factorial(10)")  -> "3628800"
        calc("not a number")  -> "ERROR: ..."
    """
    try:
        expr = sympy.sympify(expression, evaluate=True)
        return str(expr.evalf())
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, AttributeError) as e:
        return f"ERROR: {type(e).__name__}: {e}"
