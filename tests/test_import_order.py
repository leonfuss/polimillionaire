"""Regression tests: importing `prompts.*` before `strategies.*` must work.

The original bug: `prompts/calc_react.py` imported `render_question_block`
from `strategies/_common`, so loading `polimillionaire.prompts.calc_react`
as the first import triggered `strategies/__init__.py`, which loaded
`strategies.calc_react`, which re-imported `prompts.calc_react` while it was
still mid-init -- AttributeError on `PROMPT_VERSION`.

We use a subprocess so each test gets a clean module table; importing in the
*same* test process leaves modules cached and hides the bug.
"""

from __future__ import annotations

import subprocess
import sys


def _run(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )


def test_prompts_calc_react_can_be_imported_first() -> None:
    result = _run(
        "from polimillionaire.prompts import calc_react; "
        "assert calc_react.PROMPT_VERSION, 'PROMPT_VERSION missing'"
    )
    assert result.returncode == 0, result.stderr


def test_prompts_zero_shot_can_be_imported_first() -> None:
    result = _run(
        "from polimillionaire.prompts import zero_shot; "
        "assert zero_shot.PROMPT_VERSION, 'PROMPT_VERSION missing'"
    )
    assert result.returncode == 0, result.stderr


def test_strategies_module_still_re_exports_strategies() -> None:
    """The fix must not break `from polimillionaire.strategies import X`."""
    result = _run(
        "from polimillionaire.strategies import "
        "ZeroShotStrategy, CalcReactStrategy, AnswerDecision, Context"
    )
    assert result.returncode == 0, result.stderr
