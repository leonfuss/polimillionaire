"""Unit tests for the sympy-backed calculator tool."""

from __future__ import annotations

from polimillionaire.tools import calc


def test_calc_integer_arithmetic() -> None:
    # evalf returns a Float, so "4" comes out as "4.00000000000000".
    assert calc("2 + 2").startswith("4")


def test_calc_irrational_returns_symbolic_and_decimal() -> None:
    # Symbolic form lets the model match against `sqrt(2)`-shaped options
    # without doing the conversion mentally.
    out = calc("sqrt(2)")
    assert "sqrt(2)" in out
    assert "1.41421356" in out


def test_calc_rational_returns_symbolic_and_decimal() -> None:
    out = calc("1/3")
    assert "1/3" in out
    assert "0.333333" in out


def test_calc_simplifies_rational_to_lowest_terms() -> None:
    # Rational(27, 99) must come back as 3/11 so the model can match it
    # against an option spelled `3/11`.
    out = calc("Rational(27, 99)")
    assert "3/11" in out
    assert "0.27272" in out


def test_calc_solves_quadratic_via_solve() -> None:
    out = calc("solve(x**2 - 5*x + 6, x)")
    assert "[2, 3]" in out


def test_calc_truncates_pathologically_large_results() -> None:
    """Regression: a cubic system whose `solve(...)` returned 16 solutions
    with massive complex symbolic forms produced a tens-of-thousands-of-chars
    string, which swamped the next LLM prompt and crashed the calc-react
    loop. Output must be capped so the loop survives."""
    # Three-variable cubic system with many solutions -- mirrors the real
    # G2L5 trigger from the live run.
    out = calc("solve([a*(a+2*b) - 104/3, b*(b+2*c) - 7/9, c*(c+2*a) + 7], (a, b, c))")
    assert "truncated" in out
    # The leading rational solutions are still visible, which is what the
    # model needs to compute |a + b + c|.
    assert "(-4, -7/3, 1)" in out or "(4, 7/3, -1)" in out


def test_calc_factorial() -> None:
    assert calc("factorial(10)").startswith("3628800")


def test_calc_log_base() -> None:
    assert calc("log(100, 10)").startswith("2")


def test_calc_invalid_expression_returns_error_string() -> None:
    out = calc("not a number $$$")
    assert out.startswith("ERROR:")


def test_calc_unknown_sympy_error_subclass_returns_error_string() -> None:
    """Regression: the model emitted `factor(x**4 + 4, domain='Z5')` during
    a replay; sympy raised `polyerrors.OptionError`, which the previous
    narrow except list didn't catch, killing the whole replay run.

    The calculator is a sandbox -- *any* sympy failure must come back as
    a string the model can read on the next turn, regardless of which
    sympy submodule's exception hierarchy it came from."""
    out = calc("factor(x**4 + 4, domain='Z5')")
    assert out.startswith("ERROR:")


def test_calc_rejects_arbitrary_python() -> None:
    # sympify must not let the model exec import statements / attribute access.
    out = calc("__import__('os').system('echo pwned')")
    assert out.startswith("ERROR:")


def test_calc_handles_empty_string() -> None:
    out = calc("")
    assert out.startswith("ERROR:")
