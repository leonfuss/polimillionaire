"""Unit tests for the sympy-backed calculator tool."""

from __future__ import annotations

from polimillionaire.tools import calc


def test_calc_integer_arithmetic() -> None:
    # evalf returns a Float, so "4" comes out as "4.00000000000000".
    assert calc("2 + 2").startswith("4")


def test_calc_irrational_returns_decimal_expansion() -> None:
    out = calc("sqrt(2)")
    assert out.startswith("1.41421356")


def test_calc_rational_evalfs_to_decimal() -> None:
    assert calc("1/3").startswith("0.333333")


def test_calc_factorial() -> None:
    assert calc("factorial(10)").startswith("3628800")


def test_calc_log_base() -> None:
    assert calc("log(100, 10)").startswith("2")


def test_calc_invalid_expression_returns_error_string() -> None:
    out = calc("not a number $$$")
    assert out.startswith("ERROR:")


def test_calc_rejects_arbitrary_python() -> None:
    # sympify must not let the model exec import statements / attribute access.
    out = calc("__import__('os').system('echo pwned')")
    assert out.startswith("ERROR:")


def test_calc_handles_empty_string() -> None:
    out = calc("")
    assert out.startswith("ERROR:")
