"""sympy-backed calculator for ReAct-style strategies.

The model emits an `expression` string; we sympify and evalf it. Errors are
returned as `ERROR: ...` strings so the model sees them in the next turn and
can self-correct (rather than the strategy aborting silently).

For symbolic results (Rational, sqrt, pi, solve(...)) the symbolic form is
shown alongside the decimal -- e.g. `Rational(27, 99)` returns
`"3/11 = 0.272727272727273"` -- so the model can match against fraction
options without doing the simplification in its head. Pure numeric results
(Integer, Float) return the decimal alone.

Output is capped at `MAX_OUTPUT_CHARS` so that pathologically large results
(e.g. a cubic system whose `solve(...)` returns 16 solutions including
massive complex symbolic forms) don't swamp the next LLM prompt and crash
the calc-react loop. The first N chars typically still contain the real /
rational solutions, which are what the model needs.

We deliberately use `sympy.sympify` rather than `eval`: sympify's parser
rejects arbitrary Python (no imports, no attribute access, no calls outside
sympy's allow-list) so a malformed or hostile expression fails closed.

A small `STATS_LOCALS` dict adds varargs helpers for stats idioms the LLM
reaches for naturally (`mean`, `median`, `stdev`, `variance`, `range_of`).
Sympy has no built-in `Mean` / `Median` / `Range`-as-statistical-range, so
without these helpers the model's natural Python-style phrasing failed
with confusing errors in live play.
"""

from __future__ import annotations

from typing import Any

import sympy

MAX_OUTPUT_CHARS = 600


def _to_numeric_list(args: tuple[Any, ...]) -> list[Any]:
    """Accept either varargs (`mean(1, 2, 3)`) or a single iterable
    (`mean([1, 2, 3])`). Sympy parses bare `[...]` as a python-style list
    in expressions, so both call shapes hit this helper."""
    if len(args) == 1 and hasattr(args[0], "__iter__") and not isinstance(args[0], sympy.Basic):
        return list(args[0])
    return list(args)


def _stat_mean(*args: Any) -> sympy.Expr:
    vals = _to_numeric_list(args)
    if not vals:
        raise ValueError("mean() requires at least one value")
    return sympy.Rational(
        sum(sympy.Integer(int(v)) if isinstance(v, int) else v for v in vals), 1
    ) / len(vals)


def _stat_median(*args: Any) -> sympy.Expr:
    vals = sorted(_to_numeric_list(args), key=lambda x: float(x))
    if not vals:
        raise ValueError("median() requires at least one value")
    n = len(vals)
    if n % 2 == 1:
        return sympy.sympify(vals[n // 2])
    return (sympy.sympify(vals[n // 2 - 1]) + sympy.sympify(vals[n // 2])) / 2


def _stat_variance(*args: Any) -> sympy.Expr:
    """Sample variance (divide by n-1), matching scipy / R / Excel defaults."""
    vals = _to_numeric_list(args)
    if len(vals) < 2:
        raise ValueError("variance() requires at least two values")
    m = _stat_mean(*vals)
    return sum((sympy.sympify(v) - m) ** 2 for v in vals) / (len(vals) - 1)


def _stat_stdev(*args: Any) -> sympy.Expr:
    return sympy.sqrt(_stat_variance(*args))


def _stat_range(*args: Any) -> sympy.Expr:
    """Statistical range = max - min. Named `range_of` because `range` is
    reserved by sympy.Range (an integer iterator), which is what bit us in
    the first live run."""
    vals = _to_numeric_list(args)
    if not vals:
        raise ValueError("range_of() requires at least one value")
    return sympy.sympify(max(vals, key=lambda x: float(x))) - sympy.sympify(
        min(vals, key=lambda x: float(x))
    )


def _normal_cdf(mu: Any, sigma: Any, x: Any) -> sympy.Expr:
    """P(X <= x) for X ~ Normal(mu, sigma). Curries sympy.stats.cdf so the
    model can write `normal_cdf(mu, sigma, x)` rather than the two-step
    `cdf(Normal('X', mu, sigma))(x)` form, which it never gets right in
    one shot."""
    from sympy.stats import Normal, cdf

    X = Normal("_X", sympy.sympify(mu), sympy.sympify(sigma))
    return cdf(X)(sympy.sympify(x))


STATS_LOCALS: dict[str, Any] = {
    "mean": _stat_mean,
    "median": _stat_median,
    "stdev": _stat_stdev,
    "variance": _stat_variance,
    "range_of": _stat_range,
    "normal_cdf": _normal_cdf,
}


def _cap(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    return s[:MAX_OUTPUT_CHARS] + f"... [truncated; original length {len(s)} chars]"


def calc(expression: str) -> str:
    """Evaluate a math expression. Returns a numeric string or `ERROR: <msg>`.

    Examples:
        calc("2 + 2")                 -> "4.00000000000000"
        calc("Rational(27, 99)")      -> "3/11 = 0.272727272727273"
        calc("sqrt(2)")               -> "sqrt(2) = 1.41421356237310"
        calc("solve(x**2 - 5*x + 6, x)") -> "[2, 3]"
        calc("1/3")                   -> "1/3 = 0.333333333333333"
        calc("mean(10, 30, 50)")      -> "30.0000000000000"
        calc("median(10, 30, 45, 50, 90)") -> "45.0000000000000"
        calc("range_of(10, 30, 90)")  -> "80.0000000000000"
        calc("not a number")          -> "ERROR: ..."
    """
    try:
        expr = sympy.sympify(expression, locals=STATS_LOCALS, evaluate=True)
    except Exception as e:  # noqa: BLE001 -- sandbox boundary, must never raise
        # sympy raises across many submodule-specific exception types
        # (SympifyError, OptionError, PolynomialError, ...) which don't
        # share a common base. Catching narrowly leaks novel ones up
        # through the calc-react loop and crashes the run.
        return f"ERROR: {type(e).__name__}: {e}"

    # solve(), Sum(), factorint() etc. return Python lists/tuples/dicts —
    # render as-is. We allow this narrowly (not via hasattr) because sympify
    # happily evaluates arbitrary Python (e.g. `__import__("os").system(...)`
    # returns int 0); admitting only sympy objects + list/tuple/dict keeps
    # that exfiltration shut.
    if isinstance(expr, list | tuple | dict):
        return _cap(str(expr))
    if not isinstance(expr, sympy.Basic):
        return f"ERROR: unexpected non-sympy result: {type(expr).__name__}"

    try:
        numeric = expr.evalf()
    except AttributeError:
        # Boolean compounds (And/Or/Not) and Sets returned by
        # solve(inequality, var) — e.g. `(-oo < y) & (y < 8)` — don't have
        # evalf. Show the symbolic form so the model still gets the bound.
        return _cap(str(expr))
    except Exception as e:  # noqa: BLE001 -- sandbox boundary, must never raise
        return f"ERROR: {type(e).__name__}: {e}"

    # Pure numeric: only show decimal — symbolic form is just the same number.
    if getattr(expr, "is_Integer", False) or getattr(expr, "is_Float", False):
        return _cap(str(numeric))

    # Symbolic: show both forms so the model can match against fraction /
    # surd / pi options without doing the simplification mentally.
    return _cap(f"{expr} = {numeric}")
