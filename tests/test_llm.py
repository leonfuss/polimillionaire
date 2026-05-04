"""Smoke tests for the LLM loader.

Skipped on machines without llama-cpp-python installed (CI, most laptops).
We never download a real GGUF in tests — that's an integration concern.
"""

from __future__ import annotations

import json

import pytest

from polimillionaire import llm as llm_module
from polimillionaire.llm import LLM, MODELS, ModelSpec, load_llm, unload


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


class _FakeInner:
    """Stand-in for llama_cpp.Llama; tracks whether close() ran."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch Llama.from_pretrained so load_llm doesn't hit HF or build a model.

    Returns a list that records the alias passed for each load, so tests can
    distinguish cached returns from fresh loads.
    """
    calls: list[str] = []

    class FakeLlama:
        @classmethod
        def from_pretrained(cls, repo_id: str, filename: str, **_: object) -> _FakeInner:
            calls.append(repo_id)
            return _FakeInner()

    fake_module = type("M", (), {"Llama": FakeLlama})
    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", fake_module)
    monkeypatch.setattr(llm_module, "_active", None, raising=False)
    return calls


def test_load_llm_caches_same_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_loader(monkeypatch)

    a = load_llm("qwen3-8b")
    b = load_llm("qwen3-8b")

    assert a is b
    assert len(calls) == 1
    unload()


def test_load_llm_evicts_previous_when_alias_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_loader(monkeypatch)

    first = load_llm("qwen3-8b")
    inner_first = first._inner  # capture before eviction
    second = load_llm("hermes3-8b")

    assert first is not second
    assert isinstance(inner_first, _FakeInner)
    assert inner_first.closed is True
    assert first.is_loaded is False
    assert second.is_loaded is True
    assert len(calls) == 2
    unload()


def test_unload_clears_active_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch)
    handle = load_llm("qwen3-8b")
    assert handle.is_loaded is True

    unload()
    assert handle.is_loaded is False

    # Idempotent.
    unload()
    handle.unload()


def test_force_reload_rebuilds_even_for_same_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_loader(monkeypatch)

    a = load_llm("qwen3-8b")
    b = load_llm("qwen3-8b", force_reload=True)

    assert a is not b
    assert len(calls) == 2
    unload()


def test_llm_unload_is_idempotent_on_already_unloaded() -> None:
    inner = _FakeInner()
    handle = LLM(inner=inner, spec=MODELS["qwen3-8b"], name="qwen3-8b")  # type: ignore[arg-type]
    handle.unload()
    handle.unload()  # second call must not raise
    assert inner.closed is True


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
