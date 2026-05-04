"""Thin facade over the vendored millionaire_client.

Centralises construction so scratch notebooks don't import from the
vendor path directly. Use `make_client()` and forget about credentials.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client import MillionaireClient
from polimillionaire.config import Settings, load_settings


def make_client(settings: Settings | None = None) -> MillionaireClient:
    """Build an authenticated MillionaireClient using env-loaded settings."""
    settings = settings or load_settings()
    client = MillionaireClient(settings.api_url)
    client.login(settings.username, settings.password)
    return client
