"""LLM loading.

`load_llm(name)` returns a callable `prompt -> completion`. Wrapping HF
`transformers.pipeline` is the obvious baseline; vLLM, llama.cpp, or any
other backend can be swapped in here without touching strategy code.
"""

from __future__ import annotations

from collections.abc import Callable

LLM = Callable[[str], str]


def load_llm(name: str, **kwargs) -> LLM:
    """Load an LLM by HuggingFace model name.

    Implement this when you start wiring up real strategies. Suggested
    minimal version using transformers:

        from transformers import pipeline
        pipe = pipeline("text-generation", model=name, **kwargs)
        return lambda prompt: pipe(prompt, max_new_tokens=64)[0]["generated_text"]
    """
    raise NotImplementedError("Implement load_llm when a strategy needs it.")
