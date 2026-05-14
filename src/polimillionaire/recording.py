"""SQLite-backed log of every question + prediction the bot encounters.

The questions are server-side and closed: each one we see, logged
correctly, is a permanent training/eval datapoint nobody else can
recreate. Treat the DB as the canonical artefact of the project.

Concurrency: WAL mode lets multiple Colab runtimes append safely to the
same file (e.g. on a shared Drive mount).

Schema rationale:
- One row per *prediction event*, not per question. Question dedup is a
  query-time concern (`SELECT DISTINCT question_id, ...`) — running the
  same question through three strategies should yield three rows.
- `correct_option_id_if_known` is nullable: we only learn the truth when
  the server tells us via AnswerResult. Capture it whenever we do.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                   TEXT    NOT NULL,
    account_username            TEXT    NOT NULL,
    session_id                  INTEGER NOT NULL,
    competition_id              INTEGER NOT NULL,
    level                       INTEGER NOT NULL,
    question_id                 INTEGER NOT NULL,
    question_text               TEXT    NOT NULL,
    options_json                TEXT    NOT NULL,
    predicted_option_id         INTEGER NOT NULL,
    correct_option_id_if_known  INTEGER,
    strategy_name               TEXT    NOT NULL,
    model_name                  TEXT    NOT NULL,
    prompt_version              TEXT    NOT NULL,
    confidence                  REAL,
    rationale                   TEXT,
    latency_ms                  INTEGER NOT NULL,
    -- 1 if `correct_option_id_if_known` was filled in by us reasoning
    -- about the question rather than confirmed by the server's "correct"
    -- response. Lets eval/replay weight or filter generated ground truth.
    generated_answer            INTEGER NOT NULL DEFAULT 0,
    -- 'text' or 'speech'. The same question_id can appear under both modes;
    -- speech-mode rows store the Whisper transcript as question_text, which
    -- will *not* byte-match the text-mode server-provided text. Mode-tagging
    -- lets db_retrieval keep the lookup buckets disjoint by default while
    -- still allowing an explicit cross-mode fallback when desired.
    mode                        TEXT    NOT NULL DEFAULT 'text'
);

CREATE INDEX IF NOT EXISTS idx_predictions_question_id ON predictions(question_id);
CREATE INDEX IF NOT EXISTS idx_predictions_competition ON predictions(competition_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp   ON predictions(timestamp);
-- idx_predictions_mode_qid is created in _migrate(), after the ALTER TABLE
-- step adds the mode column to pre-existing DBs.

-- Key/value side-table for index-wide flags. The only consumer today is
-- the `index_valid` sentinel: db_retrieval flips it to "0" the first time
-- it sees a (question_id, question_text) collision, on the assumption
-- that the server rebuilt its question pool and the old id ↔ answer
-- mapping no longer holds.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _migrate(con: sqlite3.Connection) -> None:
    """Idempotent migrations for DBs created before a column was added."""
    cur = con.execute("PRAGMA table_info(predictions)")
    cols = {row[1] for row in cur.fetchall()}
    if "generated_answer" not in cols:
        con.execute(
            "ALTER TABLE predictions ADD COLUMN generated_answer INTEGER NOT NULL DEFAULT 0"
        )
    if "mode" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN mode TEXT NOT NULL DEFAULT 'text'")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_mode_qid ON predictions(mode, question_id)"
    )


@dataclass(frozen=True)
class PredictionRecord:
    account_username: str
    session_id: int
    competition_id: int
    level: int
    question_id: int
    question_text: str
    options: list[dict]  # [{"id": 0, "text": "..."}, ...]
    predicted_option_id: int
    correct_option_id_if_known: int | None
    strategy_name: str
    model_name: str
    prompt_version: str
    confidence: float | None
    rationale: str | None
    latency_ms: int
    # True iff `correct_option_id_if_known` was filled in by reasoning rather
    # than server confirmation. Live play always sets this False.
    generated_answer: bool = False
    # "text" or "speech". `speech_play_loop` sets this to "speech"; everything
    # else defaults to "text". See db_retrieval for how this gates lookups.
    mode: str = "text"


class QuestionLog:
    """Append-only log of question/prediction events, backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            _migrate(con)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            yield con
        finally:
            con.close()

    def record(self, rec: PredictionRecord) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO predictions
                  (timestamp, account_username, session_id, competition_id, level,
                   question_id, question_text, options_json, predicted_option_id,
                   correct_option_id_if_known, strategy_name, model_name,
                   prompt_version, confidence, rationale, latency_ms,
                   generated_answer, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    rec.account_username,
                    rec.session_id,
                    rec.competition_id,
                    rec.level,
                    rec.question_id,
                    rec.question_text,
                    json.dumps(rec.options),
                    rec.predicted_option_id,
                    rec.correct_option_id_if_known,
                    rec.strategy_name,
                    rec.model_name,
                    rec.prompt_version,
                    rec.confidence,
                    rec.rationale,
                    rec.latency_ms,
                    int(rec.generated_answer),
                    rec.mode,
                ),
            )

    def has_question(self, question_id: int) -> bool:
        with self._connect() as con:
            cur = con.execute(
                "SELECT 1 FROM predictions WHERE question_id = ? LIMIT 1",
                (question_id,),
            )
            return cur.fetchone() is not None

    # ----- meta key/value (used by db_retrieval for the index_valid flag) -----

    def get_meta(self, key: str) -> str | None:
        with self._connect() as con:
            cur = con.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = cur.fetchone()
            return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ----- read helpers for db_retrieval strategy -----

    def lookup_text_for_id(self, question_id: int, mode: str = "text") -> str | None:
        """Return any logged text for this (mode, id), or None.

        Multiple rows can share an id (we record per prediction event, not
        per question) but all rows for a given mode should agree on the
        text -- text drift inside a mode is the signal we want for
        `find_text_mismatch`. Speech-mode rows store the Whisper transcript,
        text-mode rows store the server-provided text; they live in disjoint
        buckets and we never compare across them here.
        """
        with self._connect() as con:
            cur = con.execute(
                "SELECT question_text FROM predictions WHERE question_id = ? AND mode = ? LIMIT 1",
                (question_id, mode),
            )
            row = cur.fetchone()
            return None if row is None else row[0]

    def find_text_mismatch(
        self, question_id: int, question_text: str, mode: str = "text"
    ) -> str | None:
        """If any logged row for (mode, question_id) has a different
        `question_text`, return that prior text. Otherwise None.

        Trivial whitespace-only differences are normalised away -- the server
        sometimes round-trips through stores that collapse spaces. Mode-scoped
        so an ASR transcript in 'speech' doesn't trip the drift banner against
        a server-provided text in 'text'.
        """

        def _norm(s: str) -> str:
            return " ".join(s.split()).strip()

        target = _norm(question_text)
        with self._connect() as con:
            cur = con.execute(
                "SELECT DISTINCT question_text FROM predictions WHERE question_id = ? AND mode = ?",
                (question_id, mode),
            )
            for (prior,) in cur.fetchall():
                if _norm(prior) != target:
                    return prior
        return None

    def lookup_known_correct(
        self, question_id: int, question_text: str, mode: str = "text"
    ) -> tuple[int, str] | None:
        """Return (option_id, option_text) for a server-confirmed correct
        answer, or None.

        The text is the option's text *at the moment the server confirmed
        it* (pulled from `options_json` of that row). Callers verify it
        against the current question's option set so reshuffled option ids
        or admin-edited answer keys don't trick us into submitting a stale
        pick.

        Filters:
        - `generated_answer=0` (server truth only, not our own reasoning)
        - Rows where the option_id also appears as a confirmed-wrong pick
          for this question are excluded. That contradiction means the
          server's answer key changed after we cached it; defer to the
          inner strategy until a new confirmation overrides the old one.
        - Most-recent row wins on ties, so a later confirmation overrides
          an earlier one (helps after a server-side answer-key edit
          flips back).
        """
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT correct_option_id_if_known, options_json
                FROM predictions
                WHERE question_id = ?
                  AND question_text = ?
                  AND mode = ?
                  AND correct_option_id_if_known IS NOT NULL
                  AND generated_answer = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (question_id, question_text, mode),
            )
            row = cur.fetchone()
            if row is None:
                return None
            correct_id = int(row[0])

        # Self-contradiction check: if we have also submitted this option
        # and been told it's wrong, the server-side truth is in flux.
        if correct_id in self.lookup_failed_options(question_id, question_text, mode):
            return None

        try:
            options = json.loads(row[1])
            text = next(o["text"] for o in options if int(o["id"]) == correct_id)
        except (json.JSONDecodeError, StopIteration, KeyError, ValueError, TypeError):
            return None
        return correct_id, text

    def lookup_known_correct_text(self, question_id: int, mode: str) -> str | None:
        """Return the option *text* for a server-confirmed correct answer in
        `mode`, ignoring our current question_text. Used by cross-mode lookup
        (e.g. speech-mode sessions reading text-mode rows): the cached option
        text gets fuzzy-matched against the current question's option texts.

        Returns None if no confirmed-correct row exists for (mode, qid), or
        if the cached row's options_json can't be parsed. Most-recent wins.
        """
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT correct_option_id_if_known, options_json
                FROM predictions
                WHERE question_id = ?
                  AND mode = ?
                  AND correct_option_id_if_known IS NOT NULL
                  AND generated_answer = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (question_id, mode),
            )
            row = cur.fetchone()
        if row is None:
            return None
        try:
            options = json.loads(row[1])
            correct_id = int(row[0])
            return next(o["text"] for o in options if int(o["id"]) == correct_id)
        except (json.JSONDecodeError, StopIteration, KeyError, ValueError, TypeError):
            return None

    def lookup_failed_options(
        self, question_id: int, question_text: str, mode: str = "text"
    ) -> set[int]:
        """Option ids we previously picked for this (mode, question) and got
        back a confirmed wrong outcome for.

        Schema note: the server only reports `correct: bool` -- not which
        option was the right one. So in our rows, `correct_option_id_if_known`
        is set only when we picked correctly (then it equals predicted). A
        NULL value is the unambiguous signal that we submitted and were
        told wrong. `generated_answer=0` excludes our own offline reasoning."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT DISTINCT predicted_option_id
                FROM predictions
                WHERE question_id = ?
                  AND question_text = ?
                  AND mode = ?
                  AND correct_option_id_if_known IS NULL
                  AND generated_answer = 0
                """,
                (question_id, question_text, mode),
            )
            return {int(r[0]) for r in cur.fetchall()}
