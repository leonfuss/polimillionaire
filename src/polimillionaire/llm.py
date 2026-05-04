"""Local LLM loader for the PoliMillionaire bot.

Wraps `llama-cpp-python` so strategies can call a single `LLM` interface
regardless of which GGUF model is loaded. GGUF models auto-download from
Hugging Face on first use and live in the standard HF cache
(`~/.cache/huggingface/hub` locally, `/root/.cache/huggingface/hub` on Colab) —
no Drive caching, since llama.cpp's mmap'd random reads are slow over Drive's
FUSE layer and the HF->/content download is ~1-2 minutes on Colab anyway.

Tool calling is intentionally not exposed yet. When we wire up RAG retrieval
later, add a `complete_with_tools()` method here.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_cpp import Llama


@dataclass(frozen=True)
class ModelSpec:
    """Where a GGUF lives on the Hub and how to load it."""

    repo_id: str
    filename: str  # may contain glob wildcards (e.g. "*Q4_K_M*.gguf")
    n_ctx: int = 8192
    # Appended to the final user turn. Currently used for Qwen3's `/no_think`
    # switch — the model card places this token at the end of the user message,
    # not in the system prompt.
    user_suffix: str = ""
    # Extra kwargs passed straight to llama_cpp.Llama.from_pretrained.
    extra: dict[str, Any] = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # Default. Top BFCL v3 score in 8B class, broad world knowledge.
    # `/no_think` disables Qwen3's thinking mode, which is documented to
    # interfere with tool/structured-output reliability.
    "qwen3-8b": ModelSpec(
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="*Q4_K_M*.gguf",
        user_suffix=" /no_think",
    ),
    "qwen3-14b": ModelSpec(
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="*Q4_K_M*.gguf",
        user_suffix=" /no_think",
    ),
    "gemma3-12b": ModelSpec(
        repo_id="bartowski/google_gemma-3-12b-it-GGUF",
        filename="*Q4_K_M*.gguf",
    ),
    "granite-8b": ModelSpec(
        repo_id="bartowski/ibm-granite_granite-4.1-8b-GGUF",
        filename="*Q4_K_M*.gguf",
    ),
    "hermes3-8b": ModelSpec(
        repo_id="NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        filename="*Q4_K_M*.gguf",
    ),
    "phi4-14b": ModelSpec(
        repo_id="bartowski/phi-4-GGUF",
        filename="*Q4_K_M*.gguf",
    ),
}


Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class LLM:
    """Thin wrapper around `llama_cpp.Llama` exposing two completion methods.

    - `complete(messages)` -- free-form chat completion, returns the assistant string.
    - `complete_json(messages, schema)` -- GBNF-grammar-constrained completion,
      returns a dict guaranteed to match the JSON schema.
    """

    def __init__(self, inner: Llama, spec: ModelSpec, name: str) -> None:
        self._inner = inner
        self.spec = spec
        self.name = name

    @property
    def is_loaded(self) -> bool:
        return self._inner is not None

    def unload(self) -> None:
        """Free the underlying VRAM/RAM allocation.

        Idempotent. After this call, `complete` and `complete_json` will fail —
        the wrapper still exists so any external references stay valid, but
        the C-level model is gone. `load_llm` calls this automatically before
        loading a different model.
        """
        if self._inner is None:
            return
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()
        self._inner = None  # type: ignore[assignment]
        gc.collect()

    def _prepare(self, messages: list[Message]) -> list[Message]:
        if not self.spec.user_suffix or not messages:
            return messages
        # Append the suffix to the last user turn (Qwen3's `/no_think` switch
        # is recognised there, not in the system prompt).
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                patched = dict(messages[i])
                patched["content"] = patched["content"] + self.spec.user_suffix
                return [*messages[:i], patched, *messages[i + 1 :]]
        return messages

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        out = self._inner.create_chat_completion(
            messages=self._prepare(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return out["choices"][0]["message"]["content"]

    def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
        out = self._inner.create_chat_completion(
            messages=self._prepare(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=grammar,
            **kwargs,
        )
        raw = out["choices"][0]["message"]["content"]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            # Almost always max_tokens hit mid-object — surface the raw output
            # so the caller can bump max_tokens or shrink the schema.
            raise ValueError(f"grammar-constrained output did not parse: {raw!r}") from e


# Module-level handle to the most recently loaded model. Caching by alias
# avoids redundant reloads in notebooks; eviction on alias change releases
# VRAM that IPython would otherwise pin via _, Out[N], or stale references.
_active: LLM | None = None


def load_llm(name: str = "qwen3-8b", *, force_reload: bool = False, **overrides: Any) -> LLM:
    """Load a model from the registry, evicting any previously loaded model.

    Notebook callers can re-run this cell freely: a second call with the same
    alias and no overrides returns the cached `LLM`; a different alias unloads
    the prior one before loading. Pass `force_reload=True` to rebuild even
    when the alias is unchanged.

    `overrides` are forwarded to `llama_cpp.Llama.from_pretrained`.
    Defaults: `n_gpu_layers=-1` (offload everything to GPU when available),
    `verbose=False`.
    """
    global _active

    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")

    if (
        _active is not None
        and _active.is_loaded
        and _active.name == name
        and not overrides
        and not force_reload
    ):
        return _active

    if _active is not None:
        _active.unload()
        _active = None

    spec = MODELS[name]
    from llama_cpp import Llama

    kwargs: dict[str, Any] = {
        "n_ctx": spec.n_ctx,
        "n_gpu_layers": -1,
        "verbose": False,
        **spec.extra,
        **overrides,
    }
    inner = Llama.from_pretrained(
        repo_id=spec.repo_id,
        filename=spec.filename,
        **kwargs,
    )
    _active = LLM(inner=inner, spec=spec, name=name)
    return _active


def unload() -> None:
    """Force-release whatever model `load_llm` last returned."""
    global _active
    if _active is not None:
        _active.unload()
        _active = None
        gc.collect()


__all__ = ["LLM", "MODELS", "Message", "ModelSpec", "load_llm", "unload"]
