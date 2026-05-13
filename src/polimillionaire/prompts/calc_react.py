"""Calc-react prompt variants.

v2 (current default) adds four hand-crafted few-shot exemplars showing the
correct problem-decomposition pattern for the failure types we've seen so far:

  1. Inclusion-exclusion counting (G1L7 near-miss).
  2. LCM x GCD = a x b (G1L8 near-miss).
  3. Repeating decimal -> fraction (G1L10 loss; needs symbolic Rational).
  4. Quadratic -> interval between roots (cannonball loss; needs sympy.solve).
"""

from __future__ import annotations

import json

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import PromptVariant, render_question_block

_V2_SYSTEM = SYSTEM = (
    "You are an expert trivia player with access to a calculator tool.\n"
    "\n"
    "On every turn, output JSON matching exactly one of these two shapes:\n"
    '  - {"action": "calculate", "expression": "<sympy expression>"}\n'
    '  - {"action": "answer", "rationale": "...", "confidence": <0..1>, "answer_id": <int>}\n'
    "\n"
    "Calculator syntax is sympy. Useful primitives:\n"
    "  - Arithmetic: +, -, *, /, ** (power). ALWAYS write * between variables.\n"
    "  - Symbolic: Rational(a, b), sqrt, pi, E, factorial, log, exp.\n"
    "  - Solving: solve(expr, var)  -- one variable, never a sum of variables.\n"
    "  - Rounding: floor, ceil, abs.\n"
    "  - Statistics: mean(v1, v2, ...), median(...), stdev(...), variance(...), range_of(...).\n"
    "    Pass numbers as varargs: mean(10, 30, 50) -> 30. NEVER write Mean(X) or Range(X).\n"
    "  - When options are fractions, prefer Rational(a,b) so the symbolic form\n"
    "    comes back (e.g. `3/11`) rather than a decimal.\n"
    "\n"
    "Common pitfalls to avoid:\n"
    "  - The calculator is STATELESS. It does not know any names defined in\n"
    "    the question text. If the question says 'X = {10, 30, 45}', you must\n"
    "    write the literal values: `mean(10, 30, 45)`, NEVER `mean(X)`.\n"
    "    Same for Y, set A, list L, function f(x), etc. -- inline the values.\n"
    "  - DO NOT juxtapose variables for product: write `a*b*c`, never `abc`\n"
    "    (sympy parses `abc` as a single symbol named 'abc').\n"
    "  - DO NOT call `solve(eq, a+b+c)` -- pick one variable to solve for.\n"
    "  - Keep expressions short (under 200 chars). If a setup is getting long,\n"
    "    you're probably overcomplicating; reason it out and answer directly.\n"
    "  - For combinatorics or symbolic factoring problems where the answer\n"
    "    follows from a clever observation rather than a numeric computation,\n"
    "    SKIP the calculator and answer with a brief rationale.\n"
    "\n"
    "Use the calculator whenever concrete arithmetic, exponents, factorials,\n"
    "square roots, logs, fractions, or solving equations is involved. Do not\n"
    "compute those mentally -- the calculator is more reliable. For non-numeric\n"
    "questions, answer directly.\n"
    "\n"
    "When you answer, write the rationale first (no more than three sentences) "
    "and only commit to an answer_id consistent with both the rationale and "
    "the calculator results."
)


# Each exemplar is the full multi-turn trace of solving one question:
# user -> assistant action(s) -> user (calc result) -> assistant answer.
# Rendered into the message list before the real question.
_EXEMPLARS: list[list[Message]] = [
    # 1) Inclusion-exclusion counting.
    [
        {
            "role": "user",
            "content": (
                "Q: How many integers from 1 to 100 (inclusive) are divisible by 3 or 5?\n"
                "\n"
                "Options:\n"
                "[1] 47\n"
                "[2] 53\n"
                "[3] 60\n"
                "[4] 67"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {"action": "calculate", "expression": "100//3 + 100//5 - 100//15"}
            ),
        },
        {
            "role": "user",
            "content": "Calculator: `100//3 + 100//5 - 100//15` = 47.0000000000000",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "answer",
                    "rationale": (
                        "By inclusion-exclusion: |A or B| = |A| + |B| - |A and B|. "
                        "Multiples of 3: 33, multiples of 5: 20, multiples of 15: 6. "
                        "Total = 33 + 20 - 6 = 47."
                    ),
                    "confidence": 1.0,
                    "answer_id": 1,
                }
            ),
        },
    ],
    # 2) LCM x GCD = a x b.
    [
        {
            "role": "user",
            "content": (
                "Q: The least common multiple of two integers is 60, and the greatest "
                "common divisor is 6. If one integer is 30, what is the other?\n"
                "\n"
                "Options:\n"
                "[1] 12\n"
                "[2] 18\n"
                "[3] 24\n"
                "[4] 36"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({"action": "calculate", "expression": "60 * 6 / 30"}),
        },
        {"role": "user", "content": "Calculator: `60 * 6 / 30` = 12.0000000000000"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "answer",
                    "rationale": (
                        "Use the identity LCM(a, b) * GCD(a, b) = a * b. Solving for the "
                        "unknown b: b = LCM * GCD / a = 60 * 6 / 30 = 12."
                    ),
                    "confidence": 1.0,
                    "answer_id": 1,
                }
            ),
        },
    ],
    # 3) Repeating decimal -> fraction. Uses Rational so the simplified
    #    symbolic form comes back from calc.
    [
        {
            "role": "user",
            "content": (
                "Q: Express 0.272727... (the repeating decimal 0.27 with 27 repeating) "
                "as a common fraction in lowest terms.\n"
                "\n"
                "Options:\n"
                "[1] 27/100\n"
                "[2] 27/99\n"
                "[3] 3/11\n"
                "[4] 27/90"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps({"action": "calculate", "expression": "Rational(27, 99)"}),
        },
        {
            "role": "user",
            "content": "Calculator: `Rational(27, 99)` = 3/11 = 0.272727272727273",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "answer",
                    "rationale": (
                        "Let x = 0.272727... Then 100x = 27.272727..., so 100x - x = 27, "
                        "giving x = 27/99. Sympy simplifies this to 3/11, which matches "
                        "option [3]."
                    ),
                    "confidence": 1.0,
                    "answer_id": 3,
                }
            ),
        },
    ],
    # 5) Statistics: mean comparison using the stats helpers. The vars
    #    are inlined (no symbolic X, Y) because the calculator has no
    #    notion of named datasets.
    [
        {
            "role": "user",
            "content": (
                "Q: Consider the two sets X = {10, 30, 45, 50, 55, 70, 90} "
                "and Y = {10, 30, 35, 50, 65, 70, 90}. Which of the following "
                "statements is FALSE?\n"
                "\n"
                "Options:\n"
                "[1] X and Y have identical ranges.\n"
                "[2] X and Y have identical medians.\n"
                "[3] X and Y have identical means.\n"
                "[4] None of the above are false."
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "calculate",
                    "expression": "mean(10, 30, 45, 50, 55, 70, 90) - mean(10, 30, 35, 50, 65, 70, 90)",
                }
            ),
        },
        {
            "role": "user",
            "content": (
                "Calculator: `mean(10, 30, 45, 50, 55, 70, 90) - "
                "mean(10, 30, 35, 50, 65, 70, 90)` = 10/7 = 1.42857142857143"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "answer",
                    "rationale": (
                        "Ranges: 90-10=80 for both -> identical. Medians: both "
                        "sets have 50 in position 4 -> identical. Means differ by "
                        "10/7 != 0, so option [3] is the false statement."
                    ),
                    "confidence": 1.0,
                    "answer_id": 3,
                }
            ),
        },
    ],
    # 4) Quadratic -> interval between roots, via sympy.solve.
    [
        {
            "role": "user",
            "content": (
                "Q: A ball thrown upward has height h(t) = -t**2 + 6*t meters at time t "
                "seconds. For how many seconds is the ball above 5 meters?\n"
                "\n"
                "Options:\n"
                "[1] 4\n"
                "[2] 3\n"
                "[3] 2\n"
                "[4] 1"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {"action": "calculate", "expression": "solve(-t**2 + 6*t - 5, t)"}
            ),
        },
        {"role": "user", "content": "Calculator: `solve(-t**2 + 6*t - 5, t)` = [1, 5]"},
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "answer",
                    "rationale": (
                        "The ball is above 5 meters between the two roots of "
                        "-t**2 + 6*t - 5 = 0. Sympy gives roots t=1 and t=5, so the "
                        "duration is 5 - 1 = 4 seconds."
                    ),
                    "confidence": 1.0,
                    "answer_id": 1,
                }
            ),
        },
    ],
]


# Public so RAG-augmented variants can splice these in without re-listing
# the whole exemplar set. They teach the action JSON format -- the
# RAG-retrieved reference solutions teach the math pattern.
EXEMPLAR_MESSAGES: list[Message] = [m for ex in _EXEMPLARS for m in ex]


def _render_v2(question: Question) -> list[Message]:
    return [
        {"role": "system", "content": _V2_SYSTEM},
        *EXEMPLAR_MESSAGES,
        {"role": "user", "content": render_question_block(question)},
    ]


PROMPTS: dict[str, PromptVariant] = {
    "v2": PromptVariant(version="v2", render=_render_v2),
}

LATEST = "v2"

# legacy module-level aliases so existing callers that do `prompt.render(...)` keep working
PROMPT_VERSION = LATEST
render = PROMPTS[LATEST].render
