"""Settings loaded from environment variables.

Locally: reads `.env` via python-dotenv.
On Colab: also pulls values from `google.colab.userdata` (Colab Secrets) when
present; those take precedence over `.env` so cloud runs don't accidentally
fall back to a checked-out file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REQUIRED_KEYS = (
    "POLIMILLIONAIRE_API_URL",
    "POLIMILLIONAIRE_USER",
    "POLIMILLIONAIRE_PASSWORD",
)


@dataclass(frozen=True)
class Settings:
    api_url: str
    username: str
    password: str
    db_path: Path


def _try_colab_userdata() -> dict[str, str]:
    """Pull settings from Colab Secrets if running in Colab."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return {}

    out: dict[str, str] = {}
    for key in (*REQUIRED_KEYS, "POLIMILLIONAIRE_DB_PATH"):
        try:
            value = userdata.get(key)
        except Exception:
            continue
        if value:
            out[key] = value
    return out


def load_settings() -> Settings:
    """Load settings from `.env` (local) or Colab Secrets (cloud).

    Colab Secrets win when both are set.

    Raises:
        RuntimeError: if any required setting is missing.
    """
    load_dotenv()
    for k, v in _try_colab_userdata().items():
        os.environ[k] = v

    missing = [k for k in REQUIRED_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required setting(s): {', '.join(missing)}. "
            "Set them in `.env` (local) or Colab Secrets (cloud). "
            "See `.env.example` for the template."
        )

    db_path = Path(os.environ.get("POLIMILLIONAIRE_DB_PATH", "data/questions.sqlite"))
    return Settings(
        api_url=os.environ["POLIMILLIONAIRE_API_URL"],
        username=os.environ["POLIMILLIONAIRE_USER"],
        password=os.environ["POLIMILLIONAIRE_PASSWORD"],
        db_path=db_path,
    )
