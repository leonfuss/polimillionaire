"""Flat per-question replay records — the shape the notebook builds tables from."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class ReplayResult:
    strategy_name: str
    model_name: str
    prompt_version: str
    question_id: int
    competition_id: int
    level: int
    predicted_option_id: int
    correct_option_id: int
    correct: bool
    confidence: float | None
    latency_ms: int
    # True when the ground truth was hand-reasoned (not server-confirmed at play time);
    # lets eval weight or separate those rows when the corpus mixes the two.
    generated_answer: bool


def to_polars(records: list[ReplayResult]) -> pl.DataFrame:
    """Build a polars DataFrame from a list of ReplayResult records."""
    import polars as pl

    return pl.DataFrame([asdict(r) for r in records])
