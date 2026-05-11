"""Strategy factory + registry.

Construct any strategy by name from a notebook cell or script with one call:

    strategy = make_strategy("auto", llm)
    strategy = make_strategy("wiki_rag", llm, competition_id=0)
    strategy = make_strategy("zero_shot", llm)

To add a new strategy, write `strategies/foo.py` and register a builder here:

    @register("foo")
    def _build_foo(llm, *, competition_id=None, project_root=None, **kw):
        return FooStrategy(llm, **kw)
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polimillionaire.llm import LLM
    from polimillionaire.strategies.base import Strategy

StrategyBuilder = Callable[..., "Strategy"]


def _accepts(cls: type, **kw: Any) -> dict[str, Any]:
    """Return only the kwargs that the class's __init__ actually accepts.

    `make_strategy("auto", llm, max_steps=3, k=3)` fans out to wiki_rag +
    rag_calc_react. Each strategy takes a different kwarg set; without filtering,
    forwarding `max_steps` to WikiRagStrategy raises TypeError. This helper
    introspects each constructor so the factory doesn't need to track which
    kwarg belongs to which strategy by hand.
    """
    sig = inspect.signature(cls)
    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    return {k: v for k, v in kw.items() if k in accepted}


_REGISTRY: dict[str, StrategyBuilder] = {}

# scripts/ -> repo root -> src/polimillionaire/strategies/factory.py is 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Shared caches keyed by model name / index dir so a long-running session
# (continuous_play, or a notebook sweep) doesn't repeatedly load the same
# embedder, FAISS index, BM25 sidecar, or reranker from disk.
_embedder_cache: dict[str, Any] = {}
_math_retriever_cache: dict[str, Any] = {}
_wiki_components_cache: dict[str, Any] = {}
_reranker_cache: dict[str, Any] = {}


def register(name: str) -> Callable[[StrategyBuilder], StrategyBuilder]:
    """Decorator that registers a strategy builder under `name`."""

    def decorator(builder: StrategyBuilder) -> StrategyBuilder:
        if name in _REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def make_strategy(
    name: str,
    llm: LLM,
    *,
    competition_id: int | None = None,
    project_root: Path | None = None,
    **kwargs: Any,
) -> Strategy:
    """Construct a strategy by name.

    `competition_id` is consumed by RAG-style strategies that load a
    per-competition retrieval index. `project_root` overrides where the
    factory looks for data/index/ (defaults to the repo root).

    Extra `kwargs` are forwarded to the strategy's constructor (e.g.
    `verbose=True`, `max_steps=5`).
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return _REGISTRY[name](llm, competition_id=competition_id, project_root=project_root, **kwargs)


def _resolve_project_root(override: Path | None) -> Path:
    return Path(override) if override is not None else _PROJECT_ROOT


def _get_embedder(model_name: str) -> Any:
    from polimillionaire.retrieval.embedder import Embedder

    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = Embedder(model_name)
    return _embedder_cache[model_name]


def _get_reranker(model_name: str | None = None) -> Any:
    from polimillionaire.retrieval.reranker import DEFAULT_RERANKER, Reranker

    key = model_name or DEFAULT_RERANKER
    if key not in _reranker_cache:
        _reranker_cache[key] = Reranker(key) if model_name else Reranker()
    return _reranker_cache[key]


def _math_retriever(project_root: Path) -> Any:
    from polimillionaire.retrieval.retriever import Retriever

    index_dir = project_root / "data" / "index" / "math"
    cache_key = str(index_dir)
    if cache_key in _math_retriever_cache:
        return _math_retriever_cache[cache_key]
    if not index_dir.exists():
        return None
    try:
        manifest = json.loads((index_dir / "manifest.json").read_text())
        embedder = _get_embedder(manifest["model_name"])
        retriever = Retriever(index_dir, embedder=embedder)
        _math_retriever_cache[cache_key] = retriever
        return retriever
    except Exception as exc:  # noqa: BLE001
        print(f"!! math retriever unavailable ({type(exc).__name__}: {exc})")
        return None


def _wiki_components(project_root: Path, competition_id: int) -> Any:
    """Return (Retriever, BM25Index, Reranker) for this competition's wiki index, or None."""
    slug = {0: "wiki_entertainment", 1: "wiki_history", 2: "wiki_science"}.get(competition_id)
    if slug is None:
        return None
    index_dir = project_root / "data" / "index" / slug
    cache_key = str(index_dir)
    if cache_key in _wiki_components_cache:
        return _wiki_components_cache[cache_key]
    if not index_dir.exists():
        return None
    try:
        from polimillionaire.retrieval.bm25 import BM25Index
        from polimillionaire.retrieval.retriever import Retriever

        manifest = json.loads((index_dir / "manifest.json").read_text())
        embedder = _get_embedder(manifest["model_name"])
        retriever = Retriever(index_dir, embedder=embedder)
        bm25 = BM25Index.load(index_dir)
        reranker = _get_reranker()
        components = (retriever, bm25, reranker)
        _wiki_components_cache[cache_key] = components
        return components
    except Exception as exc:  # noqa: BLE001
        print(
            f"!! wiki components unavailable for competition {competition_id} "
            f"({type(exc).__name__}: {exc})"
        )
        return None


@register("zero_shot")
def _build_zero_shot(
    llm: LLM,
    *,
    competition_id: int | None = None,  # noqa: ARG001
    project_root: Path | None = None,  # noqa: ARG001
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.zero_shot import ZeroShotStrategy

    return ZeroShotStrategy(llm, **_accepts(ZeroShotStrategy, **kw))


@register("calc_react")
def _build_calc_react(
    llm: LLM,
    *,
    competition_id: int | None = None,  # noqa: ARG001
    project_root: Path | None = None,  # noqa: ARG001
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.calc_react import CalcReactStrategy

    return CalcReactStrategy(llm, **_accepts(CalcReactStrategy, **kw))


@register("rag_calc_react")
def _build_rag_calc_react(
    llm: LLM,
    *,
    competition_id: int | None = None,  # noqa: ARG001
    project_root: Path | None = None,
    strict: bool = False,
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.calc_react import CalcReactStrategy
    from polimillionaire.strategies.rag_calc_react import RagCalcReactStrategy

    retriever = _math_retriever(_resolve_project_root(project_root))
    if retriever is None:
        if strict:
            raise FileNotFoundError(
                "rag_calc_react requested but no MATH index at "
                f"{_resolve_project_root(project_root) / 'data' / 'index' / 'math'}"
            )
        return CalcReactStrategy(llm, **_accepts(CalcReactStrategy, **kw))
    return RagCalcReactStrategy(llm, retriever, **_accepts(RagCalcReactStrategy, **kw))


@register("wiki_rag")
def _build_wiki_rag(
    llm: LLM,
    *,
    competition_id: int | None = None,
    project_root: Path | None = None,
    strict: bool = False,
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.wiki_rag import WikiRagStrategy
    from polimillionaire.strategies.zero_shot import ZeroShotStrategy

    if competition_id is None:
        raise ValueError("wiki_rag requires competition_id (0, 1, or 2)")
    components = _wiki_components(_resolve_project_root(project_root), competition_id)
    if components is None:
        if strict:
            raise FileNotFoundError(
                f"wiki_rag requested for competition {competition_id} but no "
                f"index found under {_resolve_project_root(project_root) / 'data' / 'index'}"
            )
        # _wiki_components already printed why; degrade silently to bare LLM.
        return ZeroShotStrategy(llm, **_accepts(ZeroShotStrategy, **kw))
    retriever, bm25, reranker = components
    return WikiRagStrategy(llm, retriever, bm25, reranker, **_accepts(WikiRagStrategy, **kw))


@register("auto")
def _build_auto(
    llm: LLM,
    *,
    competition_id: int | None = None,
    project_root: Path | None = None,
    **kw: Any,
) -> Strategy:
    """Per-competition routing: wiki_rag for 0/1/2, rag_calc_react for 3, zero_shot as default.

    When `competition_id` is provided, only that route is fully built; all
    others use ZeroShotStrategy as a placeholder. This avoids loading three
    wiki indexes for a live game that only plays one competition. When
    `competition_id` is None (replay path), all four routes are built.
    """
    from polimillionaire.strategies.routed import RoutedStrategy
    from polimillionaire.strategies.zero_shot import ZeroShotStrategy

    root = _resolve_project_root(project_root)
    placeholder = ZeroShotStrategy(llm)

    if competition_id is not None:
        # single-competition live play: only build the one route we need
        routes: dict[int, Strategy] = {}
        if competition_id in (0, 1, 2):
            routes[competition_id] = _build_wiki_rag(
                llm, competition_id=competition_id, project_root=root, **kw
            )
        elif competition_id == 3:
            routes[3] = _build_rag_calc_react(llm, project_root=root, **kw)
        # the RoutedStrategy default handles any unmatched competition_id at call time
        return RoutedStrategy(routes=routes, default=placeholder)

    # replay path: build all four routes
    routes = {}
    for cid in (0, 1, 2):
        routes[cid] = _build_wiki_rag(llm, competition_id=cid, project_root=root, **kw)
    routes[3] = _build_rag_calc_react(llm, project_root=root, **kw)
    return RoutedStrategy(routes=routes, default=placeholder)
