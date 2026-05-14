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
    generated_answer            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_predictions_question_id ON predictions(question_id);
CREATE INDEX IF NOT EXISTS idx_predictions_competition ON predictions(competition_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp   ON predictions(timestamp);

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
                   generated_answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def lookup_text_for_id(self, question_id: int) -> str | None:
        """Return any logged text for this id, or None if we've never seen it.

        Multiple rows can share an id (we record per prediction event, not
        per question) but all should agree on the text -- text drift is the
        signal we want for `find_text_mismatch`.
        """
        with self._connect() as con:
            cur = con.execute(
                "SELECT question_text FROM predictions WHERE question_id = ? LIMIT 1",
                (question_id,),
            )
            row = cur.fetchone()
            return None if row is None else row[0]

    def find_text_mismatch(self, question_id: int, question_text: str) -> str | None:
        """If any logged row for `question_id` has a different `question_text`,
        return that prior text. Otherwise None.

        Trivial whitespace-only differences are normalised away -- the server
        sometimes round-trips through stores that collapse spaces.
        """

        def _norm(s: str) -> str:
            return " ".join(s.split()).strip()

        target = _norm(question_text)
        with self._connect() as con:
            cur = con.execute(
                "SELECT DISTINCT question_text FROM predictions WHERE question_id = ?",
                (question_id,),
            )
            for (prior,) in cur.fetchall():
                if _norm(prior) != target:
                    return prior
        return None

    def lookup_known_correct(self, question_id: int, question_text: str) -> int | None:
        """Return a server-confirmed correct option_id for this question, or None.

        Filters out `generated_answer=1` rows (our own reasoning, not server
        truth) and requires the logged text to match exactly so we don't
        cross-contaminate after a drift event.
        """
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT correct_option_id_if_known
                FROM predictions
                WHERE question_id = ?
                  AND question_text = ?
                  AND correct_option_id_if_known IS NOT NULL
                  AND generated_answer = 0
                LIMIT 1
                """,
                (question_id, question_text),
            )
            row = cur.fetchone()
            return None if row is None else int(row[0])

    def lookup_failed_options(self, question_id: int, question_text: str) -> set[int]:
        """Option ids we previously picked for this question and got back a
        confirmed wrong outcome for. Used to block the LLM from re-picking
        a known dead-end."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT DISTINCT predicted_option_id
                FROM predictions
                WHERE question_id = ?
                  AND question_text = ?
                  AND correct_option_id_if_known IS NOT NULL
                  AND correct_option_id_if_known != predicted_option_id
                  AND generated_answer = 0
                """,
                (question_id, question_text),
            )
            return {int(r[0]) for r in cur.fetchall()}
