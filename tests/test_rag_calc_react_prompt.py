"""Unit tests for the RAG calc-react prompt's reference formatter.

`_format_reference` branches on `metadata.source`: a MATH problem passage
renders as a Problem/Solution pair; a math-wiki chunk renders as an
encyclopedia excerpt. Mixed retrieval results must render both kinds
side-by-side in one prompt.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.prompts.rag_calc_react import _render_rag_v1
from polimillionaire.retrieval.retriever import Passage


def _q() -> Question:
    return Question(
        id=1,
        text="What is the order of the symmetric group S_5?",
        options=[
            Option(id=1, text="60"),
            Option(id=2, text="120"),
            Option(id=3, text="240"),
            Option(id=4, text="720"),
        ],
        level=10,
    )


def _math_problem_passage() -> Passage:
    return Passage(
        id="train/42",
        text="Compute the order of S_4.",
        metadata={
            "source": "math_problems",
            "subject": "Algebra",
            "level": "Level 5",
            "solution": "S_4 has 4! = 24 elements.",
        },
        score=0.71,
    )


def _math_wiki_passage() -> Passage:
    return Passage(
        id="Symmetric group#chunk0",
        text="# Symmetric group\n\nThe symmetric group S_n on a finite set "
        "of n symbols is the group whose elements are all the permutations "
        "of the n symbols. The order of S_n is n!.",
        metadata={
            "source": "math_wiki",
            "title": "Symmetric group",
            "url": "https://en.wikipedia.org/wiki/Symmetric_group",
        },
        score=0.83,
    )


def test_problem_passage_renders_as_problem_solution_pair() -> None:
    messages = _render_rag_v1(_q(), [_math_problem_passage()])
    system = messages[0]["content"]
    assert "Reference 1 (Algebra, similarity=0.71)" in system
    assert "Problem: Compute the order of S_4." in system
    assert "Solution: S_4 has 4! = 24 elements." in system


def test_wiki_passage_renders_as_encyclopedia_excerpt() -> None:
    messages = _render_rag_v1(_q(), [_math_wiki_passage()])
    system = messages[0]["content"]
    assert "Reference 1 (Wikipedia: Symmetric group, similarity=0.83)" in system
    # No Problem:/Solution: framing for wiki passages -- the chunk text
    # is shown directly.
    assert "Problem: # Symmetric group" not in system
    assert "Solution:" not in system
    assert "The symmetric group S_n on a finite set" in system


def test_mixed_passages_render_in_order_with_distinct_framing() -> None:
    messages = _render_rag_v1(
        _q(),
        [_math_wiki_passage(), _math_problem_passage()],
    )
    system = messages[0]["content"]
    wiki_idx = system.find("Reference 1 (Wikipedia: Symmetric group")
    problem_idx = system.find("Reference 2 (Algebra")
    assert wiki_idx >= 0 and problem_idx >= 0
    assert wiki_idx < problem_idx
    # The block intro now mentions both kinds.
    assert "Wikipedia excerpts" in system
    assert "step-by-step solutions" in system


def test_passages_without_source_field_still_render_as_problems() -> None:
    """Backwards-compat: pre-augmentation indexes don't carry `source`.
    Those passages should keep rendering with the Problem/Solution shape
    rather than getting dropped or mis-framed.
    """
    legacy = Passage(
        id="train/1",
        text="Solve x^2 = 4.",
        metadata={"subject": "Algebra", "solution": "x = +/- 2."},
        score=0.5,
    )
    messages = _render_rag_v1(_q(), [legacy])
    system = messages[0]["content"]
    assert "Problem: Solve x^2 = 4." in system
    assert "Solution: x = +/- 2." in system


def test_long_wiki_chunk_is_capped() -> None:
    long_text = "x" * 5000
    p = Passage(
        id="Long#chunk0",
        text=long_text,
        metadata={"source": "math_wiki", "title": "Long"},
        score=0.5,
    )
    messages = _render_rag_v1(_q(), [p])
    system = messages[0]["content"]
    assert "[...]" in system
    # The cap is 1200 chars; the rendered block must be shorter than the
    # raw text by a meaningful margin.
    assert len(system) < len(long_text) + 1000
