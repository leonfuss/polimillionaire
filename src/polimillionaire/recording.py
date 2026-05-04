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
    latency_ms                  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_question_id ON predictions(question_id);
CREATE INDEX IF NOT EXISTS idx_predictions_competition ON predictions(competition_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp   ON predictions(timestamp);
"""


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


class QuestionLog:
    """Append-only log of question/prediction events, backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

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
                   prompt_version, confidence, rationale, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )

    def has_question(self, question_id: int) -> bool:
        with self._connect() as con:
            cur = con.execute(
                "SELECT 1 FROM predictions WHERE question_id = ? LIMIT 1",
                (question_id,),
            )
            return cur.fetchone() is not None
