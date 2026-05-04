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
    # Prepended to every system prompt. Empty unless the model needs it
    # (Qwen3's `/no_think` switch is the canonical case).
    system_prefix: str = ""
    # Extra kwargs passed straight to llama_cpp.Llama.from_pretrained.
    extra: dict[str, Any] = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    # Default. Top BFCL v3 score in 8B class, broad world knowledge.
    # `/no_think` disables Qwen3's thinking mode, which is documented to
    # interfere with tool/structured-output reliability.
    "qwen3-8b": ModelSpec(
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="*Q4_K_M*.gguf",
        system_prefix="/no_think\n",
    ),
    "qwen3-14b": ModelSpec(
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="*Q4_K_M*.gguf",
        system_prefix="/no_think\n",
    ),
    "gemma3-12b": ModelSpec(
        repo_id="bartowski/google_gemma-3-12b-it-GGUF",
        filename="*Q4_K_M*.gguf",
    ),
    "granite-8b": ModelSpec(
        repo_id="ibm-granite/granite-4.1-8b-GGUF",
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

    def _prepare(self, messages: list[Message]) -> list[Message]:
        if not self.spec.system_prefix:
            return messages
        if messages and messages[0]["role"] == "system":
            head = messages[0]
            return [
                {"role": "system", "content": self.spec.system_prefix + head["content"]},
                *messages[1:],
            ]
        return [{"role": "system", "content": self.spec.system_prefix.rstrip()}, *messages]

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
        return json.loads(out["choices"][0]["message"]["content"])


def load_llm(name: str = "qwen3-8b", **overrides: Any) -> LLM:
    """Load a model from the registry. `overrides` are passed to llama_cpp.Llama.

    Defaults: `n_gpu_layers=-1` (offload everything to GPU when one is available;
    falls back to CPU), `verbose=False`.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
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
    return LLM(inner=inner, spec=spec, name=name)


__all__ = ["LLM", "MODELS", "Message", "ModelSpec", "load_llm"]
