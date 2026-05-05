"""sympy-backed calculator for ReAct-style strategies.

The model emits an `expression` string; we sympify and evalf it. Errors are
returned as `ERROR: ...` strings so the model sees them in the next turn and
can self-correct (rather than the strategy aborting silently).

For symbolic results (Rational, sqrt, pi, solve(...)) the symbolic form is
shown alongside the decimal -- e.g. `Rational(27, 99)` returns
`"3/11 = 0.272727272727273"` -- so the model can match against fraction
options without doing the simplification in its head. Pure numeric results
(Integer, Float) return the decimal alone.

We deliberately use `sympy.sympify` rather than `eval`: sympify's parser
rejects arbitrary Python (no imports, no attribute access, no calls outside
sympy's allow-list) so a malformed or hostile expression fails closed.
"""

from __future__ import annotations

import sympy


def calc(expression: str) -> str:
    """Evaluate a math expression. Returns a numeric string or `ERROR: <msg>`.

    Examples:
        calc("2 + 2")                 -> "4.00000000000000"
        calc("Rational(27, 99)")      -> "3/11 = 0.272727272727273"
        calc("sqrt(2)")               -> "sqrt(2) = 1.41421356237310"
        calc("solve(x**2 - 5*x + 6, x)") -> "[2, 3]"
        calc("1/3")                   -> "1/3 = 0.333333333333333"
        calc("not a number")          -> "ERROR: ..."
    """
    try:
        expr = sympy.sympify(expression, evaluate=True)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError, AttributeError) as e:
        return f"ERROR: {type(e).__name__}: {e}"

    # solve(), Sum(), etc. return Python lists/tuples — render as-is.
    # We allow this narrowly (not via hasattr) because sympify happily evaluates
    # arbitrary Python (e.g. `__import__("os").system(...)` returns int 0);
    # admitting only sympy objects + list/tuple keeps that exfiltration shut.
    if isinstance(expr, list | tuple):
        return str(expr)
    if not isinstance(expr, sympy.Basic):
        return f"ERROR: unexpected non-sympy result: {type(expr).__name__}"

    try:
        numeric = expr.evalf()
    except (TypeError, ValueError, AttributeError) as e:
        return f"ERROR: {type(e).__name__}: {e}"

    # Pure numeric: only show decimal — symbolic form is just the same number.
    if getattr(expr, "is_Integer", False) or getattr(expr, "is_Float", False):
        return str(numeric)

    # Symbolic: show both forms so the model can match against fraction /
    # surd / pi options without doing the simplification mentally.
    return f"{expr} = {numeric}"
