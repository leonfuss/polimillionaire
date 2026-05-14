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


def test_calc_stats_mean_varargs() -> None:
    """mean() must accept Python-style varargs since the live LLM emits
    expressions like `mean(10, 30, 50)` rather than the (unsupported)
    `mean(X)` reference to a question-text symbol."""
    assert calc("mean(10, 30, 50)").startswith("30")
    # the regression-case mean comparison from comp 3 game 2 question 1
    assert calc("mean(10, 30, 45, 50, 55, 70, 90)").startswith("50")


def test_calc_stats_median() -> None:
    # odd count: middle element
    assert calc("median(10, 30, 45, 50, 90)").startswith("45")
    # even count: avg of middle two
    assert "30" in calc("median(10, 20, 40, 50)")


def test_calc_stats_range_of() -> None:
    """range_of, not range -- sympy.Range is an integer iterator, which
    is exactly what bit us when the LLM emitted `Range(X)` in a live run."""
    assert calc("range_of(10, 30, 90)").startswith("80")


def test_calc_stats_stdev_returns_symbolic_form() -> None:
    """Standard deviation uses sample variance (n-1 denominator). The
    canonical wikipedia example {2,4,4,4,5,5,7,9} has variance 32/7."""
    out = calc("variance(2, 4, 4, 4, 5, 5, 7, 9)")
    assert "32/7" in out
    # stdev = sqrt(variance) — symbolic form so the model can match
    # against irrational option labels.
    assert "sqrt" in calc("stdev(2, 4, 4, 4, 5, 5, 7, 9)")


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


def test_pow_guard_detects_flat_huge_exponent() -> None:
    """`Pow(2, 100001)` (flat) should be rejected by the guard, exercised
    directly on the expression tree so the test never depends on sympify's
    parse-time evaluation behaviour."""
    import sympy

    from polimillionaire.tools.calculator import MAX_POW_EXPONENT, _exceeds_pow_limit

    safe = sympy.Pow(sympy.Integer(2), sympy.Integer(100), evaluate=False)
    huge = sympy.Pow(sympy.Integer(2), sympy.Integer(MAX_POW_EXPONENT + 1), evaluate=False)
    assert _exceeds_pow_limit(safe) is False
    assert _exceeds_pow_limit(huge) is True


def test_pow_guard_detects_nested_exponent() -> None:
    """The reported hang: `10**(10**10)`. Outer Pow's exponent is itself a
    Pow node with concrete Integer base+exp. The guard estimates the inner
    without materialising and rejects."""
    import sympy

    from polimillionaire.tools.calculator import _exceeds_pow_limit

    inner = sympy.Pow(sympy.Integer(10), sympy.Integer(10), evaluate=False)  # = 10b symbolic
    outer = sympy.Pow(sympy.Integer(10), inner, evaluate=False)
    assert _exceeds_pow_limit(outer) is True


def test_calc_rejects_huge_inner_exponent_without_hanging() -> None:
    """End-to-end via calc(): the user-emitted "10**(10**10)" must come
    back as an ERROR string with the modular-arithmetic hint, NOT hang."""
    out = calc("10**(10**10)")
    assert out.startswith("ERROR:")
    assert "exponent" in out
    assert "modular" in out.lower() or "mod" in out.lower()


def test_calc_rejects_direct_huge_exponent() -> None:
    """Flat `2**100000`: 30k-digit result, materializable in ~ms but the
    guard refuses because the LLM can't usefully display it. Forces
    modular reasoning instead."""
    out = calc("2**100000")
    assert out.startswith("ERROR:")
    assert "exponent" in out


def test_calc_accepts_modest_exponent() -> None:
    """Polynomials and reasonable powers must not be caught by the guard."""
    out = calc("2**100")
    assert not out.startswith("ERROR:")
    # 2**100 = 1267650600228229401496703205376 -- evalf comes back ~1.26e30.
    assert out.startswith("1.26765")


def test_calc_with_timeout_passes_through_fast_calls() -> None:
    """The subprocess wrapper should return correct results on cheap calls.
    First call pays ~0.5-1s for the worker to import sympy; that's
    acceptable in a calc-react loop where each step already costs 1-3s."""
    from polimillionaire.tools import calc_with_timeout

    out = calc_with_timeout("2 + 2", timeout=30.0)
    assert out.startswith("4")


# ---- count_integers_satisfying --------------------------------------------


def test_count_integers_solves_factorial_trailing_zeros_99() -> None:
    """Live failure case: 'For how many positive integers k does k! end in
    exactly 99 trailing zeros?' Legendre: floor(k/5) + floor(k/25) + ... = 99
    holds for k in [400, 404] (k=400 picks up the new 5^3 factor, k=405
    bumps to 100). Answer: Five."""
    out = calc(
        "count_integers_satisfying("
        "floor(k/5) + floor(k/25) + floor(k/125) + floor(k/625) - 99, "
        "k, 300, 500)"
    )
    # evalf gives "5.000..." -- the count itself is an Integer.
    assert out.startswith("5")
    assert not out.startswith("ERROR:")


def test_count_integers_handles_eq_predicate() -> None:
    """Relational predicates: Eq(k**2, 25) is True only at k=5 (and k=-5,
    but we scan [1, 10])."""
    out = calc("count_integers_satisfying(Eq(k**2, 25), k, 1, 10)")
    assert out.startswith("1")


def test_count_integers_handles_inequality_predicate() -> None:
    """Gt predicate: how many k in [1, 10] have k**2 > 50? k=8,9,10."""
    out = calc("count_integers_satisfying(k**2 - 50 > 0, k, 1, 10)")
    assert out.startswith("3")


def test_count_integers_empty_range_returns_zero() -> None:
    """hi < lo is treated as the empty range, not an error."""
    out = calc("count_integers_satisfying(k - 5, k, 10, 1)")
    assert out.startswith("0")


def test_count_integers_rejects_oversized_range() -> None:
    """MAX_ENUM_RANGE bounds the loop so a runaway scan can't burn the
    whole budget. The error surfaces back to the LLM with the actual
    limit so it can narrow the window."""
    out = calc("count_integers_satisfying(k - 1, k, 0, 10**6)")
    assert out.startswith("ERROR:")
    assert "range" in out.lower()


def test_count_integers_rejects_huge_pow_in_expression() -> None:
    """The Pow guard applies inside the helper too -- substituting
    Pow(k, 10**10) at any k would re-trigger the bignum hang."""
    # Construct via str so we don't materialise the inner power at parse.
    out = calc("count_integers_satisfying(k**(10**10) - 1, k, 1, 5)")
    assert out.startswith("ERROR:")


# ---- targeted error hints -------------------------------------------------


def test_calc_hint_on_python_comprehension() -> None:
    """Live failure case: LLM emits Python set comprehension. The raw
    SympifyError ('cannot assign to function call') is useless to the
    model; we append a hint pointing at the sympy-friendly alternatives."""
    out = calc("len({Rational(n, d) for n in [2,3,4,6] for d in [2,3,4,6]})")
    assert out.startswith("ERROR:")
    assert "HINT" in out
    assert "FiniteSet" in out or "count_integers_satisfying" in out


def test_calc_hint_on_solve_with_floor() -> None:
    """Live failure case: solve(floor(k/5) + ... = 99) raises
    NotImplementedError('multiple generators'). Hint should steer the
    model to count_integers_satisfying."""
    out = calc("solve(floor(k/5) + floor(k/25) + floor(k/125) - 99, k)")
    assert out.startswith("ERROR:")
    assert "HINT" in out
    assert "count_integers_satisfying" in out


def test_calc_no_hint_on_unrelated_error() -> None:
    """An ordinary syntax-only mistake (e.g. unmatched paren) gets the
    bare ERROR without a misleading hint."""
    out = calc("2 + (")
    assert out.startswith("ERROR:")
    assert "HINT" not in out
