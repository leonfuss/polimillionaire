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
import os
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

# Competitions with no pre-built static index: we hit Wikipedia per question
# instead. Cheap path -- no embedder, no BM25, no reranker model -- just the
# MediaWiki API + LLM. Add new live-only cids here.
_LIVE_ONLY_CIDS: frozenset[int] = frozenset({4, 5})

# Shared caches keyed by model name / index dir so a long-running session
# (continuous_play, or a notebook sweep) doesn't repeatedly load the same
# embedder, FAISS index, BM25 sidecar, or reranker from disk.
_embedder_cache: dict[str, Any] = {}
_math_retriever_cache: dict[str, Any] = {}
_wiki_components_cache: dict[str, Any] = {}
_reranker_cache: dict[str, Any] = {}
# LiveWikiRetriever holds a requests.Session + an in-process query cache;
# one shared instance per session keeps the session warm and lets the
# cache hit across competitions for any incidental overlap.
_live_wiki_cache: dict[str, Any] = {}


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
    db_retrieval: bool = False,
    db_path: str | None = None,
    mode: str = "text",
    use_text_mode_retrieval: bool = False,
    **kwargs: Any,
) -> Strategy:
    """Construct a strategy by name.

    `competition_id` is consumed by RAG-style strategies that load a
    per-competition retrieval index. `project_root` overrides where the
    factory looks for data/index/ (defaults to the repo root).

    `db_retrieval=True` wraps the built strategy with `DbRetrievalStrategy`:
    look up the question in `data/questions.sqlite` first, return the
    server-confirmed answer (after a 7-18s pacing delay) when found, and
    otherwise fall through to the underlying strategy under a ~25s budget.

    `mode` is "text" or "speech"; it gates which DB rows the wrapper reads.
    `use_text_mode_retrieval=True` (speech only) adds a cross-mode fallback:
    on a same-mode miss, check the text-mode row for this question_id and
    fuzzy-match the cached correct option text against the (transcribed)
    current options. Useful as a cold-start helper before speech-mode rows
    exist for a question.

    Extra `kwargs` are forwarded to the strategy's constructor (e.g.
    `verbose=True`, `max_steps=5`).
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    inner = _REGISTRY[name](llm, competition_id=competition_id, project_root=project_root, **kwargs)
    if not db_retrieval:
        return inner
    from polimillionaire.strategies.db_retrieval import DbRetrievalStrategy

    return DbRetrievalStrategy(
        inner,
        _resolve_db_path(db_path, project_root),
        mode=mode,
        use_text_mode_retrieval=use_text_mode_retrieval,
        verbose=kwargs.get("verbose", False),
    )


def _resolve_db_path(db_path: str | None, project_root: Path | None) -> str:
    """Mirror play._resolve_db_path so a wrapper built standalone (no play
    loop) lands at the same questions.sqlite as live play does."""
    raw = db_path or os.environ.get("POLIMILLIONAIRE_DB_PATH") or "data/questions.sqlite"
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    root = _resolve_project_root(project_root)
    return str(root / p)


def _resolve_project_root(override: Path | None) -> Path:
    return Path(override) if override is not None else _PROJECT_ROOT


def preload(
    competition_ids: list[int] | None = None,
    *,
    project_root: Path | None = None,
) -> None:
    """Eagerly load embedders, reranker, and retrievers for `competition_ids`.

    Without preload, the first question of each competition pays the HF
    download + model-load tax (~3-10s for bge weights, plus FAISS mmap
    warmup), which on a 30s timer turns the first question into a timeout.
    Call this once after `make_strategy(...)` and before the game loop.

    `competition_ids=None` warms everything (0..3 and the live-only cids).
    Live-only cids (4, 5) have nothing to warm -- no embedder, no reranker --
    so they're skipped here. On a single-comp run, pass `[competition_id]`
    to avoid loading wiki indexes you won't use.
    """
    cids = (
        competition_ids if competition_ids is not None else [0, 1, 2, 3, *sorted(_LIVE_ONLY_CIDS)]
    )
    root = _resolve_project_root(project_root)

    for cid in cids:
        if cid in (0, 1, 2):
            components = _wiki_components(root, cid)
            if components is None:
                continue
            retriever, _bm25, reranker = components
            retriever.embedder.preload()
            reranker.preload()
        elif cid == 3:
            retriever = _math_retriever(root)
            if retriever is not None:
                retriever.embedder.preload()
        # cids in _LIVE_ONLY_CIDS: nothing to preload (no local models)


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


def _get_live_wiki(*, verbose: bool = False) -> Any:
    """Singleton LiveWikiRetriever shared across competitions for this session."""
    from polimillionaire.retrieval.live_wiki import LiveWikiRetriever

    key = "default"
    if key not in _live_wiki_cache:
        _live_wiki_cache[key] = LiveWikiRetriever(verbose=verbose)
    elif verbose:
        # Honour a later verbose=True without rebuilding the cache.
        _live_wiki_cache[key]._verbose = True
    return _live_wiki_cache[key]


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


# Math-tir route defaults:
# - max_tokens=768: the action schema's `oneOf` branch needs room beyond
#   the global 256 default; lets the model write rationale + calc args
#   without truncating mid-JSON.
# - max_steps=3: math benefits from retry headroom (try symbolic solve →
#   notice junk output → fall back to plug-and-verify → answer). Validated
#   in live play.
_MATH_TIR_MAX_TOKENS = 768
_MATH_TIR_MAX_STEPS = 3


def _apply_math_tir_defaults(kw: dict[str, Any]) -> dict[str, Any]:
    if kw.get("prompt_version") != "math-tir":
        return kw
    overrides: dict[str, Any] = {}
    if "max_tokens" not in kw:
        overrides["max_tokens"] = _MATH_TIR_MAX_TOKENS
    if "max_steps" not in kw:
        overrides["max_steps"] = _MATH_TIR_MAX_STEPS
    return {**kw, **overrides} if overrides else kw


@register("calc_react")
def _build_calc_react(
    llm: LLM,
    *,
    competition_id: int | None = None,  # noqa: ARG001
    project_root: Path | None = None,  # noqa: ARG001
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.calc_react import CalcReactStrategy

    kw = _apply_math_tir_defaults(kw)
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

    kw = _apply_math_tir_defaults(kw)
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
    live_lookup: bool = False,
    **kw: Any,
) -> Strategy:
    from polimillionaire.strategies.wiki_rag import WikiRagStrategy
    from polimillionaire.strategies.zero_shot import ZeroShotStrategy

    if competition_id is None:
        raise ValueError("wiki_rag requires competition_id (0, 1, 2, 4, or 5)")

    if competition_id in _LIVE_ONLY_CIDS:
        live = _get_live_wiki(verbose=kw.get("verbose", False))
        # Strip live_lookup -- it's already implicit here -- and force the
        # static toggles off so WikiRagStrategy doesn't ask for a retriever
        # we don't have.
        live_kw = {**kw, "use_dense": False, "use_sparse": False, "use_reranker": False}
        return WikiRagStrategy(
            llm,
            retriever=None,
            bm25=None,
            reranker=None,
            live=live,
            **_accepts(WikiRagStrategy, **live_kw),
        )

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
    live = _get_live_wiki(verbose=kw.get("verbose", False)) if live_lookup else None
    return WikiRagStrategy(
        llm, retriever, bm25, reranker, live=live, **_accepts(WikiRagStrategy, **kw)
    )


# Per-competition wiki_rag tuning. Entertainment's 794k-passage corpus is ~4x
# the size of history and ~2x science -- the relevant doc is more likely to be
# outside the default nprobe=32 / top-50 candidate window, so widen for it.
# `live_lookup` enables per-question Wikipedia API fusion. On by default for
# all three static wiki categories -- the rerank pool dedups by title so live
# never displaces a strong static hit, and the worst case is a few hundred ms
# of extra latency on a single API call.
# Cids in `_LIVE_ONLY_CIDS` (4, 5) skip the static index entirely; the factory
# wires them through WikiRagStrategy in live-only mode, so `live_lookup` here
# is irrelevant for them.
# Override on a per-call basis by passing kwargs to make_strategy("auto", ...).
_AUTO_WIKI_DEFAULTS: dict[int, dict[str, Any]] = {
    0: {
        "nprobe": 128,
        "dense_k": 100,
        "sparse_k": 100,
        "fused_k": 50,
        "top_k": 8,
        "live_lookup": True,
    },
    1: {"live_lookup": True},  # history -- defaults work but live catches edge cases
    2: {"live_lookup": True},  # science -- live picks up post-crawl discoveries
    # Live-only: widen the live pool a bit; with no static fallback we want
    # more candidates feeding the LLM.
    4: {"live_k": 6, "top_k": 6},  # philosophy & psychology
    5: {"live_k": 6, "top_k": 6},  # news
}


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

    Per-competition defaults from `_AUTO_WIKI_DEFAULTS` are merged with the
    caller's kwargs (caller wins on conflict) so the entertainment route gets
    wider retrieval automatically without polluting other competitions.
    """
    from polimillionaire.strategies.routed import RoutedStrategy
    from polimillionaire.strategies.zero_shot import ZeroShotStrategy

    root = _resolve_project_root(project_root)
    placeholder = ZeroShotStrategy(llm)

    def _wiki_kw(cid: int) -> dict[str, Any]:
        return {**_AUTO_WIKI_DEFAULTS.get(cid, {}), **kw}

    wiki_cids = (0, 1, 2, *sorted(_LIVE_ONLY_CIDS))

    if competition_id is not None:
        # single-competition live play: only build the one route we need
        routes: dict[int, Strategy] = {}
        if competition_id in wiki_cids:
            routes[competition_id] = _build_wiki_rag(
                llm, competition_id=competition_id, project_root=root, **_wiki_kw(competition_id)
            )
        elif competition_id == 3:
            routes[3] = _build_rag_calc_react(llm, project_root=root, **kw)
        # the RoutedStrategy default handles any unmatched competition_id at call time
        return RoutedStrategy(routes=routes, default=placeholder)

    # replay path: build every known route
    routes = {}
    for cid in wiki_cids:
        routes[cid] = _build_wiki_rag(llm, competition_id=cid, project_root=root, **_wiki_kw(cid))
    routes[3] = _build_rag_calc_react(llm, project_root=root, **kw)
    return RoutedStrategy(routes=routes, default=placeholder)
