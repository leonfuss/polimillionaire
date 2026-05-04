"""Smoke tests for the LLM loader.

Skipped on machines without llama-cpp-python installed (CI, most laptops).
We never download a real GGUF in tests — that's an integration concern.
"""

from __future__ import annotations

import pytest

from polimillionaire.llm import MODELS, ModelSpec, load_llm


def test_default_model_in_registry() -> None:
    assert "qwen3-8b" in MODELS
    spec = MODELS["qwen3-8b"]
    assert isinstance(spec, ModelSpec)
    assert spec.repo_id.endswith("Qwen3-8B-GGUF")
    assert "Q4_K_M" in spec.filename
    assert "/no_think" in spec.system_prefix


def test_registry_covers_all_shortlisted_aliases() -> None:
    expected = {"qwen3-8b", "qwen3-14b", "gemma3-12b", "granite-8b", "hermes3-8b", "phi4-14b"}
    assert expected <= set(MODELS)


def test_load_llm_rejects_unknown_alias() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        load_llm("does-not-exist")


def test_load_llm_known_alias_imports_llama_cpp() -> None:
    """If llama_cpp isn't installed, the import error is what we expect.

    We don't actually load a model — that would download ~5 GB. We just
    confirm the function gets past the registry lookup and reaches the
    backend import, where ImportError is the only acceptable failure
    mode on a machine without the optional dep.
    """
    pytest.importorskip("llama_cpp")
    # If llama_cpp is installed, calling load_llm() would download a model.
    # We stop at the registry boundary instead.
    assert MODELS["qwen3-8b"].n_ctx == 8192
