"""Prompt registries per strategy.

Each module exposes a `PROMPTS: dict[str, PromptVariant]` keyed by version
(e.g. "v1", "v2") plus a `LATEST: str` alias selecting the default. The
strategy constructor takes `prompt_version="..."` to pick a variant. Adding
a new variant means adding a new entry to the module's PROMPTS dict and
bumping LATEST if you want it to become the default.
"""

from polimillionaire.prompts._common import PromptVariant, render_question_block

__all__ = ["PromptVariant", "render_question_block"]
