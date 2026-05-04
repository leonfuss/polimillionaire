"""Smoke tests for the LLM loader.

Skipped on machines without llama-cpp-python installed (CI, most laptops).
We never download a real GGUF in tests — that's an integration concern.
"""

from __future__ import annotations

import json

import pytest

from polimillionaire.llm import MODELS, ModelSpec, load_llm


def test_default_model_in_registry() -> None:
    assert "qwen3-8b" in MODELS
    spec = MODELS["qwen3-8b"]
    assert isinstance(spec, ModelSpec)
    assert spec.repo_id.endswith("Qwen3-8B-GGUF")
    assert "Q4_K_M" in spec.filename
    assert "/no_think" in spec.user_suffix


def test_registry_covers_all_shortlisted_aliases() -> None:
    expected = {"qwen3-8b", "qwen3-14b", "gemma3-12b", "granite-8b", "hermes3-8b", "phi4-14b"}
    assert expected <= set(MODELS)


def test_load_llm_rejects_unknown_alias() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        load_llm("does-not-exist")


def test_grammar_round_trips_a_trivial_schema() -> None:
    """Sanity check that llama_cpp.LlamaGrammar can derive a grammar
    from the kind of MCQ schema complete_json() will be passed."""
    pytest.importorskip("llama_cpp")
    from llama_cpp import LlamaGrammar

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
        "required": ["answer"],
    }
    grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
    assert grammar is not None
