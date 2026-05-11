"""Per-question competition-aware dispatcher.

Replay iterates over a mixed-competition log; this strategy delegates
each question to the right sub-strategy based on `Context.competition_id`,
and falls back to a default for unrouted competitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polimillionaire.strategies.base import AnswerDecision, Context, Strategy

if TYPE_CHECKING:
    from polimillionaire._vendor.millionaire_client.models import Question


class RoutedStrategy:
    """Dispatch each question to a competition-specific sub-strategy.

    The strategy_name / model_name / prompt_version reported on each
    decision is the *sub-strategy's* -- so replay's per-question records
    correctly attribute results to whichever inner strategy answered.
    """

    strategy_name = "routed"

    def __init__(
        self,
        routes: dict[int, Strategy],
        default: Strategy,
    ) -> None:
        self._routes = routes
        self._default = default

    @property
    def routes(self) -> dict[int, Strategy]:
        return self._routes

    @property
    def model_name(self) -> str:
        # routed strategies don't have a single model; callers can inspect .routes
        return "routed"

    @property
    def prompt_version(self) -> str:
        return "routed"

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:
        strategy = self._routes.get(ctx.competition_id, self._default)
        return strategy(question, ctx)
