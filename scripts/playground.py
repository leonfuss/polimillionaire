"""Local playground for poking at strategies with full visibility.

Right now: walks calc-react step-by-step on a few math questions, printing
every action, every calculator call, every committed answer. The strategy
classes hide the loop; here we open it up so we can see *exactly* what the
model is emitting and where it goes wrong.

Run on a Mac with the LLM group installed:

    uv sync --group llm
    uv run python scripts/playground.py

Edit `QUESTIONS` to add cases. Edit `MODEL_NAME` / `MAX_STEPS` to sweep.
"""

from __future__ import annotations

import json

from polimillionaire import load_llm
from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.prompts import calc_react as prompt
from polimillionaire.strategies._common import make_action_schema, make_schema
from polimillionaire.tools import calc

MODEL_NAME = "qwen3-8b"
MAX_STEPS = 3


# Hand-written math questions, correct answer noted in the comment.
QUESTIONS: list[Question] = [
    Question(
        id=1,
        text="What is 12 * 17?",
        options=[
            Option(id=1, text="184"),
            Option(id=2, text="204"),  # correct
            Option(id=3, text="214"),
            Option(id=4, text="224"),
        ],
        level=1,
    ),
    Question(
        id=2,
        text="What is the square root of 144?",
        options=[
            Option(id=1, text="10"),
            Option(id=2, text="11"),
            Option(id=3, text="12"),  # correct
            Option(id=4, text="14"),
        ],
        level=1,
    ),
    Question(
        id=3,
        text="What is 25% of 80?",
        options=[
            Option(id=1, text="15"),
            Option(id=2, text="20"),  # correct
            Option(id=3, text="25"),
            Option(id=4, text="30"),
        ],
        level=1,
    ),
    Question(
        id=4,
        text="What is 2 to the power of 10?",
        options=[
            Option(id=1, text="512"),
            Option(id=2, text="1000"),
            Option(id=3, text="1024"),  # correct
            Option(id=4, text="2048"),
        ],
        level=2,
    ),
    Question(
        id=5,
        text="How many seconds are in one day?",
        options=[
            Option(id=1, text="3600"),
            Option(id=2, text="36000"),
            Option(id=3, text="86400"),  # correct
            Option(id=4, text="864000"),
        ],
        level=2,
    ),
]


def run_question(llm, question: Question, *, max_steps: int = MAX_STEPS) -> None:
    """Walk one question through the calc-react loop with verbose printing."""
    messages = list(prompt.render(question))
    schema = make_action_schema(question)

    print()
    print("=" * 70)
    print(f"Q{question.id}: {question.text}")
    for opt in question.options:
        print(f"  [{opt.id}] {opt.text}")
    print("=" * 70)

    for step in range(max_steps):
        print(f"\n--- step {step + 1}/{max_steps} ---")
        out = llm.complete_json(messages, schema)
        print(f"model emitted: {out}")
        if out["action"] == "answer":
            print(f"COMMIT: option_id={out['answer_id']} confidence={out['confidence']}")
            print(f"rationale: {out.get('rationale')}")
            return
        result = calc(out["expression"])
        print(f"calc({out['expression']!r}) -> {result}")
        messages.append({"role": "assistant", "content": json.dumps(out)})
        messages.append(
            {"role": "user", "content": f"Calculator: `{out['expression']}` = {result}"}
        )

    print("\n--- step cap hit, forcing answer ---")
    messages.append(
        {"role": "user", "content": "Step limit reached. Answer now using the answer schema."}
    )
    out = llm.complete_json(messages, make_schema(question))
    print(f"forced: option_id={out['answer_id']} confidence={out['confidence']}")
    print(f"rationale: {out.get('rationale')}")


def main() -> None:
    print(f"loading {MODEL_NAME}...")
    llm = load_llm(MODEL_NAME)
    print(f"loaded. running {len(QUESTIONS)} question(s).")

    for q in QUESTIONS:
        run_question(llm, q)


if __name__ == "__main__":
    main()
