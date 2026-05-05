"""Prompt-formatting helpers shared across prompt modules.

Lives under `prompts/` (not `strategies/`) on purpose: prompts depend on
formatting helpers, not on strategies. Keeping the helper here means
`prompts/*` never has to import from `strategies/*`, which would create a
circular import the moment a script imports `polimillionaire.prompts.*`
before `polimillionaire.strategies.*`.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Question


def render_question_block(question: Question) -> str:
    """Format the question + numbered options for the user turn.

    The bracketed integer is the actual server-side option id, which is what
    the model commits to in `answer_id` -- no letter-to-id mapping layer.
    """
    lines = [f"Q: {question.text}", "", "Options:"]
    for opt in question.options:
        lines.append(f"[{opt.id}] {opt.text}")
    return "\n".join(lines)
