"""Prompt for the calculator-equipped ReAct strategy.

v1 was a system message + the question. The model invoked calc reasonably
often but tripped on harder math: it would emit a calculate action with the
*wrong setup* (e.g. `48 * 24 / 240` instead of `240 * 24 / 48` for an
LCM/GCD problem), or skip calc entirely on a quadratic / repeating-decimal
question, then guess.

v2 adds four hand-crafted few-shot exemplars showing the correct
problem-decomposition pattern for the failure types we've seen so far:

  1. Inclusion-exclusion counting (G1L7 near-miss).
  2. LCM x GCD = a x b (G1L8 near-miss).
  3. Repeating decimal -> fraction (G1L10 loss; needs symbolic Rational).
  4. Quadratic -> interval between roots (cannonball loss; needs sympy.solve).

Bump `PROMPT_VERSION` whenever the wording or exemplars change; the version
string is written into the predictions log so we can attribute accuracy
shifts to prompt changes vs model changes.
"""

from __future__ import annotations

import json

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import render_question_block

PROMPT_VERSION = "v2"

SYSTEM = (
    "You are an expert trivia player with access to a calculator tool.\n"
    "\n"
    "On every turn, output JSON matching exactly one of these two shapes:\n"
    '  - {"action": "calculate", "expression": "<sympy expression>"}\n'
    '  - {"action": "answer", "rationale": "...", "confidence": <0..1>, "answer_id": <int>}\n'
    "\n"
    "Calculator syntax is sympy. Useful primitives: Rational(a, b), sqrt, pi, "
    "factorial, log, exp, solve(expr, var), floor, ceil, summation/Sum. "
    "When the answer options are fractions, prefer Rational(...) so the "
    "result comes back symbolic (e.g. `3/11`) rather than as a decimal.\n"
    "\n"
    "Use the calculator whenever arithmetic, exponents, factorials, square "
    "roots, logs, fractions, or solving equations is involved. Do not compute "
    "those mentally -- the calculator is more reliable. For non-numeric "
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


def _flatten_exemplars() -> list[Message]:
    flat: list[Message] = []
    for ex in _EXEMPLARS:
        flat.extend(ex)
    return flat


def render(question: Question) -> list[Message]:
    """Build the initial message list for a calc-react turn.

    Layout: system message + flattened exemplars + the real question.
    """
    return [
        {"role": "system", "content": SYSTEM},
        *_flatten_exemplars(),
        {"role": "user", "content": render_question_block(question)},
    ]
