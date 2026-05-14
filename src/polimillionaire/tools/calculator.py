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

import atexit
import concurrent.futures
import contextlib
import multiprocessing
import threading
from typing import Any

import sympy

MAX_OUTPUT_CHARS = 600

# Reject `Pow(_, n)` when |n| exceeds this many. 10**10000 has 10001 digits
# and computes in a few ms; anything bigger pegs CPython's bignum path for
# minutes-to-hours while holding the GIL, which leaks past the wrapper's
# thread-pool timeout (Python can't actually kill a CPU-bound thread). The
# trigger in live play was the "10^(10^10) days from now" question: after
# sympify, the outer Pow has exp=10_000_000_000, then evalf() hangs forever.
# Such questions are modular-arithmetic problems in disguise; the hint in
# the error response gives the next ReAct step a recovery path.
MAX_POW_EXPONENT = 10_000


def _bounded_int_estimate(node: Any) -> int | None:
    """Conservative magnitude estimate for a sympy node, used to peek inside
    a Pow's exponent without forcing CPython to materialise it.

    Returns:
        int: a value we're confident the node *exactly* equals.
        None: either symbolic (no concrete answer) or "would exceed the
              guard's limit if we tried to materialise" -- callers should
              treat None coming from a numeric subtree as "too big".

    Handles Integer leaves and one-level-deep `Pow(Integer, Integer)`. The
    one-level depth is enough to catch the reported "10**(10**10)" shape;
    deeper nests get caught by the same logic applied recursively at the
    next outer Pow.
    """
    if isinstance(node, sympy.Integer):
        return abs(int(node))
    if isinstance(node, sympy.Pow):
        b, e = node.args
        if isinstance(b, sympy.Integer) and isinstance(e, sympy.Integer):
            b_val = abs(int(b))
            e_val = abs(int(e))
            if b_val in (0, 1):
                return b_val
            # log10(b**e) = e * log10(b); if that exceeds the guard's
            # magnitude bound, it'll be rejected anyway -- don't bother
            # materialising.
            import math

            if e_val * math.log10(max(b_val, 2)) > math.log10(MAX_POW_EXPONENT):
                return None
            return b_val**e_val
    return None


def _exceeds_pow_limit(expr: Any) -> bool:
    """True iff the expression tree contains a Pow whose integer exponent
    would exceed MAX_POW_EXPONENT.

    Catches both flat (`10**100000`) and nested (`10**(10**10)`) shapes.
    For the nested case the exponent itself is a Pow node; we use
    `_bounded_int_estimate` to evaluate it without triggering CPython's
    long_pow on a multi-billion-digit number.
    """
    if not isinstance(expr, sympy.Basic):
        return False
    if isinstance(expr, sympy.Pow):
        _, exp = expr.args
        if isinstance(exp, sympy.Integer) and abs(exp) > MAX_POW_EXPONENT:
            return True
        # Nested pow inside the exponent: `10**(10**10)`. Estimate the
        # inner without materialising; if the estimator gave up (None) and
        # the exp is a concrete Pow, treat as exceeding the limit.
        if isinstance(exp, sympy.Pow):
            est = _bounded_int_estimate(exp)
            if est is None or est > MAX_POW_EXPONENT:
                return True
    return any(_exceeds_pow_limit(a) for a in expr.args)


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


# Hard cap on the loop in count_integers_satisfying so a model emitting
# `(_, k, 0, 10**9)` doesn't burn the whole budget enumerating a billion
# integers. Calc-react questions that need numerical search converge in
# the low thousands; 100k is generous headroom.
MAX_ENUM_RANGE = 100_000


def _count_integers_satisfying(expr: Any, var: Any, lo: Any, hi: Any) -> sympy.Expr:
    """Count integer values of ``var`` in ``[lo, hi]`` (inclusive) for
    which ``expr`` is satisfied.

    - Relational / Boolean ``expr`` (e.g. ``Eq(k**2, 25)``,
      ``Gt(floor(k/5), 3)``): "satisfied" = substitutes to True.
    - Anything else: "satisfied" = substitutes to 0.

    Bounded to ``MAX_ENUM_RANGE`` integers per call.

    This is the workhorse for number-theory enumerations that sympy's
    `solve()` can't handle symbolically: the floor()/ceil() family
    ("how many k satisfy floor(k/5) + floor(k/25) + ... = 99?",
    "find all primes/perfect squares/divisors in [a, b]", and similar).
    """
    expr = sympy.sympify(expr)
    var = sympy.sympify(var)
    if isinstance(expr, sympy.Basic) and _exceeds_pow_limit(expr):
        raise ValueError(
            f"count_integers_satisfying: expression contains a power exceeding "
            f"MAX_POW_EXPONENT={MAX_POW_EXPONENT}; substituting would still "
            "overflow the bignum guard."
        )
    try:
        lo_i = int(lo)
        hi_i = int(hi)
    except (TypeError, ValueError) as e:
        raise ValueError(f"count_integers_satisfying: lo/hi must be integers ({e})") from e
    if hi_i < lo_i:
        return sympy.Integer(0)
    span = hi_i - lo_i + 1
    if span > MAX_ENUM_RANGE:
        raise ValueError(
            f"count_integers_satisfying: range size {span} exceeds "
            f"MAX_ENUM_RANGE={MAX_ENUM_RANGE}; narrow the search window"
        )
    count = 0
    for k_val in range(lo_i, hi_i + 1):
        sub = expr.subs(var, sympy.Integer(k_val))
        # Boolean shortcut: relational/Eq under integer subs usually
        # collapses to BooleanTrue/False directly.
        if sub is sympy.S.true:
            count += 1
            continue
        if sub is sympy.S.false:
            continue
        # Unevaluated Eq/Relational: try a simplify pass before giving up.
        if isinstance(sub, sympy.core.relational.Relational):
            simplified = sympy.simplify(sub)
            if simplified is sympy.S.true:
                count += 1
            continue
        # Numeric path: count when the expression is exactly zero.
        try:
            simplified = sympy.simplify(sub)
        except Exception:  # noqa: BLE001 -- per-k failure must not abort the scan
            continue
        if simplified == 0:
            count += 1
    return sympy.Integer(count)


STATS_LOCALS: dict[str, Any] = {
    "mean": _stat_mean,
    "median": _stat_median,
    "stdev": _stat_stdev,
    "variance": _stat_variance,
    "range_of": _stat_range,
    "normal_cdf": _normal_cdf,
    "count_integers_satisfying": _count_integers_satisfying,
}


def _hint_for_failure(expression: str, exc: BaseException) -> str:
    """Return a helpful suffix to append to calc()'s ERROR message for
    known failure patterns. Empty string when no specific hint applies.

    Why: the raw sympy / Python errors ("cannot assign to function call",
    "multiple generators") give the LLM no actionable recovery path. A
    short, targeted hint lets the next ReAct step retry productively
    instead of repeating the same call.
    """
    expr_lower = expression.lower()
    has_for_in = " for " in expr_lower and any(
        token in expr_lower for token in (" in [", " in (", " in {")
    )
    # Python-style comprehension or len() on a comprehension: sympify can't
    # parse comprehensions, period.
    if has_for_in or "len(" in expr_lower:
        return (
            " HINT: the calculator is sympy, not Python — no comprehensions, "
            "no `len()`, no `for ... in ...`. To count distinct values, "
            "enumerate explicitly: `FiniteSet(Rational(2,2), Rational(2,3), ...)` "
            "and read its size, or use `count_integers_satisfying(expr, var, lo, hi)` "
            "for integer-range searches."
        )
    # solve() with floor()/ceil(): sympy raises NotImplementedError with
    # "multiple generators". Steer the model to numerical enumeration.
    if (
        isinstance(exc, NotImplementedError)
        and "solve" in expr_lower
        and ("floor(" in expr_lower or "ceil(" in expr_lower)
    ):
        return (
            " HINT: solve() can't handle equations with floor() or ceil() "
            "symbolically. Use `count_integers_satisfying(expression, var, lo, hi)` "
            "to enumerate integer solutions in a bounded range."
        )
    return ""


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
        # evaluate=False so sympy's Pow.__new__ doesn't eagerly call
        # Python's int.__pow__ on a multi-billion-digit exponent during
        # parsing -- that path holds the GIL forever and the wrapper's
        # thread-pool timeout can't recover. We run the bignum guard
        # against the lazy tree, then doit() to evaluate the safe cases.
        expr = sympy.sympify(expression, locals=STATS_LOCALS, evaluate=False)
    except Exception as e:  # noqa: BLE001 -- sandbox boundary, must never raise
        # sympy raises across many submodule-specific exception types
        # (SympifyError, OptionError, PolynomialError, ...) which don't
        # share a common base. Catching narrowly leaks novel ones up
        # through the calc-react loop and crashes the run.
        return f"ERROR: {type(e).__name__}: {e}{_hint_for_failure(expression, e)}"

    # Bignum-exponent guard: catch the "10**(10**10)" class of expressions
    # before doit/evalf hang in CPython's long_pow. The pre-check is cheap
    # (just a tree walk) and prevents the runaway-thread bug entirely for
    # the common case. The subprocess timeout in calc_with_timeout is the
    # backstop for unknown hang shapes (large factorial, deep expansion).
    if _exceeds_pow_limit(expr):
        return (
            f"ERROR: expression contains a power with exponent > {MAX_POW_EXPONENT}; "
            "refusing to evaluate (would overflow CPython bignum). For very large "
            "powers, use modular arithmetic (`pow(base, exp, modulus)` or "
            "`Mod(base, modulus)**exp`)."
        )

    # solve(), Sum(), factorint() etc. return Python lists/tuples/dicts —
    # render as-is. We allow this narrowly (not via hasattr) because sympify
    # happily evaluates arbitrary Python (e.g. `__import__("os").system(...)`
    # returns int 0); admitting only sympy objects + list/tuple/dict keeps
    # that exfiltration shut.
    if isinstance(expr, list | tuple | dict):
        return _cap(str(expr))
    if not isinstance(expr, sympy.Basic):
        return f"ERROR: unexpected non-sympy result: {type(expr).__name__}"

    # The lazy tree is now safe to evaluate. doit() applies the same
    # simplifications evaluate=True would have done at parse time --
    # e.g. Add(2, 2) -> Integer(4). Rational(27, 99) is already reduced
    # inside Rational.__new__ regardless of evaluate flag, so no special
    # case needed there.
    try:
        expr = expr.doit()
    except Exception as e:  # noqa: BLE001 -- sandbox boundary
        return f"ERROR: {type(e).__name__}: {e}{_hint_for_failure(expression, e)}"

    try:
        numeric = expr.evalf()
    except AttributeError:
        # Boolean compounds (And/Or/Not) and Sets returned by
        # solve(inequality, var) — e.g. `(-oo < y) & (y < 8)` — don't have
        # evalf. Show the symbolic form so the model still gets the bound.
        return _cap(str(expr))
    except Exception as e:  # noqa: BLE001 -- sandbox boundary, must never raise
        return f"ERROR: {type(e).__name__}: {e}{_hint_for_failure(expression, e)}"

    # Pure numeric: only show decimal — symbolic form is just the same number.
    if getattr(expr, "is_Integer", False) or getattr(expr, "is_Float", False):
        return _cap(str(numeric))

    # Symbolic: show both forms so the model can match against fraction /
    # surd / pi options without doing the simplification mentally.
    return _cap(f"{expr} = {numeric}")


# -- subprocess-based hard timeout ------------------------------------------
#
# Threads can't be killed in CPython, so a runaway calc() inside a thread
# (e.g. CPython bignum exponentiation) wedges the whole interpreter --
# `future.result(timeout=N)` returns to the caller but the worker keeps the
# GIL and any subsequent question grinds against it. A subprocess can be
# terminated cleanly. We run calc() in a single long-lived worker process
# (spawn context for macOS+OpenBLAS safety) and kill+respawn on timeout.

DEFAULT_TOOL_TIMEOUT_S = 5.0

_pool: concurrent.futures.ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def _get_or_create_pool() -> concurrent.futures.ProcessPoolExecutor:
    """Lazily build a 1-worker pool. The worker imports sympy on first use
    (~0.5-1s on Colab), reused across all subsequent calls until a timeout
    or atexit shuts it down."""
    global _pool
    with _pool_lock:
        if _pool is None:
            # spawn (not fork) so macOS + OpenBLAS / Accelerate don't crash;
            # also gives us a clean Python on each respawn after a timeout.
            ctx = multiprocessing.get_context("spawn")
            _pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _pool


def _reset_pool() -> None:
    """Terminate the worker (killing any in-flight calc) and drop the pool.
    A subsequent call to _get_or_create_pool spawns a fresh worker."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            # _processes is a private dict[int, Process]; terminate each.
            # On Python 3.11+ this is stable enough to rely on for cleanup.
            for proc in list(getattr(_pool, "_processes", {}).values()):
                with contextlib.suppress(Exception):
                    proc.terminate()
            with contextlib.suppress(Exception):
                _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


atexit.register(_reset_pool)


def calc_with_timeout(expression: str, timeout: float = DEFAULT_TOOL_TIMEOUT_S) -> str:
    """Run `calc(expression)` in a worker process with a hard wall-clock
    cap. Returns the same string shape as `calc`. On timeout the worker is
    terminated (no orphan thread, no zombie computation); the next call
    spawns a fresh worker.

    The MAX_POW_EXPONENT pre-check inside calc() catches the common bignum
    hang in-process without paying the subprocess round-trip. This wrapper
    is the backstop for unknown hang shapes (large factorial, deep symbolic
    expansion, etc.) and for the case where the LLM smuggles a bad expr
    past the pre-check via an indirect form.
    """
    try:
        pool = _get_or_create_pool()
        return pool.submit(calc, expression).result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        _reset_pool()
        return (
            f"ERROR: calc exceeded {timeout:.1f}s; expression too expensive to "
            "evaluate. Try simplifying, or use modular arithmetic for very large numbers."
        )
    except concurrent.futures.process.BrokenProcessPool:
        # Worker died mid-call (OOM, segfault). Reset and surface as an error
        # rather than crashing the strategy.
        _reset_pool()
        return "ERROR: calculator worker crashed; expression likely caused an internal sympy fault."
    except Exception as e:  # noqa: BLE001 -- tool boundary, must never raise
        return f"ERROR: {type(e).__name__}: {e}"
