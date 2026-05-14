"""DB-backed retrieval wrapper around any inner strategy.

When enabled, this wrapper checks `data/questions.sqlite` for a prior
*server-confirmed* answer to the current question. If found, it pauses
7-18s (so we don't look like a bot pegged at the 30s timer) and returns
that option. If not found, it runs the inner strategy under a 30s
budget, blocks any options we've previously been told are wrong, and
falls back to a random remaining option if the inner times out or
insists on a known dead-end.

Drift detection: if we ever see a (question_id, question_text)
collision -- same id, materially different text -- we assume the
server's question pool was rebuilt and the old id ↔ answer mapping no
longer holds. A loud banner is printed and the index_valid flag in
`meta` is flipped to "0", which makes all subsequent lookups bypass the
DB for the rest of this run (and any future run that picks up the same
DB file). The flag is sticky on purpose; flipping it back is a manual
operation.
"""

from __future__ import annotations

import concurrent.futures as _futures
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from polimillionaire.recording import QuestionLog
from polimillionaire.strategies.base import AnswerDecision, Context

if TYPE_CHECKING:
    from polimillionaire._vendor.millionaire_client.models import Question
    from polimillionaire.strategies.base import Strategy

_META_INDEX_VALID = "index_valid"
_DRIFT_BANNER = (
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "!!! QUESTION INDEX DRIFT DETECTED                            !!!\n"
    "!!! question_id={qid} has a different text than logged.       \n"
    "!!! Assuming the server rebuilt its question pool.            \n"
    "!!! Marking index_valid=0; db_retrieval will skip DB lookups  \n"
    "!!! for the rest of this run.                                 \n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
)


class DbRetrievalStrategy:
    """Wraps an inner strategy with a SQLite lookup + LLM-failure guards.

    Construction is cheap: the SQLite file is opened per call inside
    `QuestionLog`, so no long-lived handles are held here.
    """

    strategy_name = "db_retrieval"

    def __init__(
        self,
        inner: Strategy,
        db_path: str,
        *,
        sleep_min: float = 7.0,
        sleep_max: float = 18.0,
        llm_timeout_s: float = 30.0,
        rng: random.Random | None = None,
        sleeper: Callable[[float], None] | None = None,
        verbose: bool = False,
    ) -> None:
        if sleep_max < sleep_min:
            raise ValueError("sleep_max must be >= sleep_min")
        self._inner = inner
        self._log = QuestionLog(db_path)
        self._sleep_min = sleep_min
        self._sleep_max = sleep_max
        self._llm_timeout_s = llm_timeout_s
        self._rng = rng or random.Random()
        # Injectable so tests don't actually wait 7-18s on the happy path.
        self._sleep = sleeper if sleeper is not None else time.sleep
        self._verbose = verbose

    @property
    def model_name(self) -> str:
        return getattr(self._inner, "model_name", "")

    @property
    def prompt_version(self) -> str:
        return getattr(self._inner, "prompt_version", "")

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:
        start = time.perf_counter()

        # The DB lookup short-circuits the inner LLM call when valid.
        # Drift detection runs before the lookup so a poisoned index gets
        # invalidated before we trust any of its rows.
        if self._index_valid_or_invalidate(question):
            known = self._log.lookup_known_correct(question.id, question.text)
            if known is not None:
                option_id = self._resolve_cached_option(question, known)
                if option_id is not None:
                    return self._db_hit_decision(option_id, start)

        failed = (
            self._log.lookup_failed_options(question.id, question.text)
            if self._index_valid()
            else set()
        )
        if self._verbose and failed:
            print(f"   [db_retrieval] blocked options for q{question.id}: {sorted(failed)}")

        return self._run_inner_with_guards(question, ctx, start, failed)

    # ---- helpers ----------------------------------------------------------

    def _index_valid(self) -> bool:
        # "0" only when we've explicitly marked drift; any other value
        # (including missing) is treated as valid. Lazy default avoids a
        # write on first construction.
        return self._log.get_meta(_META_INDEX_VALID) != "0"

    def _index_valid_or_invalidate(self, question: Question) -> bool:
        if not self._index_valid():
            return False
        prior = self._log.find_text_mismatch(question.id, question.text)
        if prior is None:
            return True
        # Loud, unmistakable banner -- and flip the flag so we don't
        # repeat this on every question for the rest of the session.
        print(_DRIFT_BANNER.format(qid=question.id))
        print(f"   prior text: {prior!r}")
        print(f"   current   : {question.text!r}")
        self._log.set_meta(_META_INDEX_VALID, "0")
        return False

    def _resolve_cached_option(self, question: Question, cached: tuple[int, str]) -> int | None:
        """Match the cached (id, text) against the current options.

        Three outcomes:
        - same id still carries the same text -> use it
        - text moved to a different id (server reshuffled options) -> use
          the new id, log a verbose note
        - text is gone entirely (server edited the question or the
          options) -> fall through to the LLM
        """
        cached_id, cached_text = cached
        same_slot = next((o for o in question.options if o.id == cached_id), None)
        if same_slot is not None and same_slot.text == cached_text:
            return cached_id

        # Try a text remap. Reshuffles are common enough to be worth
        # handling silently-but-loggable; the answer hasn't changed, only
        # the option_id pointing at it has.
        remapped = next((o for o in question.options if o.text == cached_text), None)
        if remapped is not None:
            if self._verbose:
                print(
                    f"   [db_retrieval] options reshuffled for q{question.id}: cached "
                    f"correct text moved from id {cached_id} -> id {remapped.id}; using new id"
                )
            return remapped.id

        if self._verbose:
            print(
                f"   [db_retrieval] cached correct text not present in current options "
                f"for q{question.id} (cached: {cached_text!r}, options: "
                f"{[o.text for o in question.options]}); falling through to LLM"
            )
        return None

    def _db_hit_decision(self, option_id: int, start: float) -> AnswerDecision:
        delay = self._rng.uniform(self._sleep_min, self._sleep_max)
        if self._verbose:
            print(
                f"   [db_retrieval] DB hit -> option {option_id}; "
                f"sleeping {delay:.1f}s before submitting"
            )
        self._sleep(delay)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return AnswerDecision(
            option_id=option_id,
            confidence=1.0,
            rationale=(
                f"DB lookup: server-confirmed correct option from a prior session. "
                f"Submitted after {delay:.1f}s pacing delay."
            ),
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
            latency_ms=latency_ms,
        )

    def _run_inner_with_guards(
        self,
        question: Question,
        ctx: Context,
        start: float,
        failed: set[int],
    ) -> AnswerDecision:
        """Run inner under timeout; log its choice; rewrite if it's blocked."""
        all_ids = [o.id for o in question.options]
        remaining = [i for i in all_ids if i not in failed]
        if not remaining:
            # Pathological: every option has been tried and confirmed wrong.
            # Pick anything -- we can't do better, and the server expects an
            # answer. Use the first option for determinism.
            pick = all_ids[0]
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AnswerDecision(
                option_id=pick,
                confidence=0.0,
                rationale="All options previously confirmed wrong; defaulting to option 0.",
                model_name=self.model_name,
                strategy_name=self.strategy_name,
                prompt_version=self.prompt_version,
                latency_ms=latency_ms,
            )

        decision: AnswerDecision | None = None
        timed_out = False
        # ThreadPoolExecutor gives us future.result(timeout=...) without
        # propagating into the worker. We can't actually kill the thread
        # in Python, but daemon=False is fine here: shutdown(wait=False)
        # leaves the orphan thread to finish on its own (likely just an
        # HTTP request still pending) without blocking the next question.
        with _futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._inner, question, ctx)
            try:
                decision = future.result(timeout=self._llm_timeout_s)
            except _futures.TimeoutError:
                timed_out = True
                if self._verbose:
                    print(
                        f"   [db_retrieval] inner timed out after {self._llm_timeout_s:.0f}s; "
                        "picking randomly from remaining options"
                    )
                # don't wait for the orphan; it'll wind down on its own
                ex.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                if self._verbose:
                    print(
                        f"   [db_retrieval] inner raised "
                        f"({type(exc).__name__}: {exc}); picking randomly"
                    )

        if timed_out or decision is None:
            return self._fallback_decision(
                remaining, start, reason="inner strategy did not return in time"
            )

        # Inner returned -- preserve its rationale/confidence in the log
        # but rewrite the option if it landed on a known-failed pick.
        if decision.option_id in failed:
            if self._verbose:
                print(
                    f"   [db_retrieval] inner picked option {decision.option_id} "
                    "(previously confirmed wrong); rewriting to a remaining option"
                )
            new_pick = self._rng.choice(remaining)
            return AnswerDecision(
                option_id=new_pick,
                confidence=0.0,
                rationale=(
                    f"Inner strategy picked option {decision.option_id} which was previously "
                    f"confirmed wrong; rewrote to random remaining option {new_pick}. "
                    f"Original rationale: {decision.rationale or '(none)'}"
                ),
                model_name=decision.model_name or self.model_name,
                strategy_name=self.strategy_name,
                prompt_version=decision.prompt_version or self.prompt_version,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
        return decision

    def _fallback_decision(
        self, remaining: list[int], start: float, *, reason: str
    ) -> AnswerDecision:
        pick = self._rng.choice(remaining)
        return AnswerDecision(
            option_id=pick,
            confidence=0.0,
            rationale=f"{reason}; picked option {pick} from remaining untested options.",
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
