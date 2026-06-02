"""Generate notebooks/final.ipynb for the PoliMillionaire NLP deliverable.

Single source of truth for the assembled deliverable notebook. Edit this
file, re-run with ``python scripts/build_final_nb.py`` from the repo root,
and commit the regenerated ``notebooks/final.ipynb`` alongside.

The notebook itself contains no executable logic worth diffing by hand;
all heavy lifting lives in ``src/polimillionaire/``. The build target is
Kaggle (with a Colab fallback inside the bootstrap cell).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _as_source(parts: tuple[str, ...]) -> list[str]:
    """Render cell source as a list of newline-terminated strings.

    Matches the canonical form Jupyter's nbformat writer (and the project's
    pre-commit hooks) prefer: every line ends in ``\\n`` except the last.
    """
    if not parts:
        return []
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(*parts: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _as_source(parts),
    }


def code(*parts: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _as_source(parts),
    }


# ---------------------------------------------------------------------------
# Cell content. Order matters; cells are emitted as-is.
# ---------------------------------------------------------------------------
CELLS: list[dict] = []

CELLS.append(
    md(
        "# Who Wants to be a PoliMillionaire? Final Deliverable",
        "",
        "**Course:** Natural Language Processing, Politecnico di Milano, AY 2025/26  ",
        "**Due:** 2 June 2026, 23:00 (WeBeep)",
        "",
        "| Name | Email | GitHub | PoliMillionaire user |",
        "|---|---|---|---|",
        "| Leon Fuß | leon.fuss@icloud.com | leonfuss | leonfuss |",
        "| Antoine Gaborieau | _ | AntoineGaborieau | _ |",
        "| Aleksa Pulai | _ | aleksapulai | _ |",
        "| Luca Ilardi | luca7203@icloud.com | LucBruc | LucBruc |",
        "",
        "**Video walkthrough:** <link to be added before submission>",
        "",
        "**Repo:** <https://github.com/leonfuss/polimillionaire>",
        "",
        "---",
        "",
        "> **This notebook is built for Kaggle.** The bootstrap cell clones the repo,",
        "> attaches the pre-built FAISS+BM25 indices dataset, pulls the latest version",
        "> of the shared question log from a Kaggle Dataset, and pins retrieval to the",
        "> second T4. A Colab fallback is included but is the secondary path.",
        "",
        "> **`RUN_LIVE` defaults to `False`.** The notebook renders end-to-end without",
        "> playing a live game or running any LLM. Flip the flag in the bootstrap cell",
        "> to play actual games against the PoliMillionaire server.",
        "",
        "**Coding assistants disclosure:** GitHub Copilot and Claude (Anthropic) were",
        "used throughout development for code review, prompt-engineering iteration,",
        "and notebook assembly. The architecture decisions, experimental design,",
        "and result interpretation are ours; no part of the assignment was delegated to a",
        "language model in full.",
    )
)

CELLS.append(
    md(
        "## TL;DR",
        "",
        "- A modular strategy interface, in which every method returns the same",
        "  `AnswerDecision`, permits zero-shot, hybrid RAG, and a ReAct",
        "  calculator agent to be swapped under a single per-competition router.",
        "- **Per-competition winners** measured on N=200 to 400 logged questions:",
        "  Entertainment ≈ 90% (phi4-14b zero-shot), History ≈ 92% (granite-8b",
        "  wiki_rag), Science **94.5%** (phi4-14b zero-shot), Math **91.5%**",
        "  (phi4-14b calc_react). All within the 30-second server budget.",
        "- The two **live-only** competitions, Philosophy (cid 4) and News",
        "  (cid 5), have no static index. The same `wiki_rag` strategy is",
        "  reused with `LiveWikiRetriever` / `LiveTavilyRetriever` as the sole",
        "  source. Their numbers are recomputed live from the SQLite log in the",
        "  Results section below.",
        "- Few-shot scaling on Entertainment is flat from",
        "  K=1 up to K=50, then collapses to **28%** at K=150. The prompt",
        "  exceeds the model's 8192-token KV cache and attention degrades to",
        "  near-random. K=3 is the practical sweet spot.",
    )
)

CELLS.append(
    md(
        "## Problem",
        "",
        "Closed multiple-choice trivia, four options per question, six",
        "competitions on the server:",
        "",
        "| ID | Competition |",
        "|---|---|",
        "| 0 | Entertainment |",
        "| 1 | Ancient History & Politics |",
        "| 2 | Science & Nature |",
        "| 3 | Maths |",
        "| 4 | Philosophy & Psychology *(live-only)* |",
        "| 5 | News *(live-only)* |",
        "",
        "Up to 15 questions per game, **30-second timer per question**, one wrong",
        "answer ends the run. Levels 1 to 15 with difficulty empirically uniform",
        "across rounds. The server also serves an audio variant where each",
        "question + options arrives as 24 kHz mono WAV blobs.",
    )
)

CELLS.append(
    md(
        "## Architecture",
        "",
        "```text",
        "src/polimillionaire/",
        "├── play.py                manual + auto + speech play loops",
        "├── client.py              authenticated game API client",
        "├── config.py              credentials (Kaggle Secrets / Colab Secrets / .env)",
        "├── llm.py                 9 model aliases, llama-cpp-python backend (Q4_K_M GGUFs)",
        "├── recording.py           SQLite log: every prediction is a row",
        "├── kaggle_db.py           pull/push the SQLite log to a private Kaggle Dataset",
        "├── strategies/",
        "│   ├── base.py            AnswerDecision contract",
        "│   ├── zero_shot.py       single LLM call, grammar-constrained JSON",
        "│   ├── wiki_rag.py        hybrid BM25 + dense + reranker + live MediaWiki/Tavily",
        "│   ├── calc_react.py      ReAct loop with a sympy calculator tool",
        "│   ├── rag_calc_react.py  calc_react + retrieved math exemplars",
        "│   ├── routed.py          competition_id → sub-strategy",
        "│   └── factory.py         make_strategy(name, llm, **kwargs) entry point",
        "├── prompts/               versioned templates (zero_shot, wiki_rag, calc_react, math-tir)",
        "├── retrieval/",
        "│   ├── bm25.py            sparse",
        "│   ├── embedder.py        dense (BAAI/bge-base-en-v1.5)",
        "│   ├── fusion.py          Reciprocal Rank Fusion (k=60)",
        "│   ├── reranker.py        cross-encoder (BAAI/bge-reranker-base)",
        "│   ├── wiki_*.py          static Wikipedia index build pipeline",
        "│   ├── live_wiki.py       MediaWiki API",
        "│   ├── live_tavily.py     Tavily News API (used for cid 5)",
        "│   └── retriever.py       FAISS dense lookup",
        "├── tools/calculator.py    sympy-backed, hardened against pow-bombs",
        "├── asr/whisper.py         whisper-large-v3-turbo, fp16, 24→16 kHz resample",
        "└── eval/replay.py         replay strategies over the SQLite log (offline benchmarking)",
        "```",
        "",
        "**Two-GPU layout (Kaggle T4 ×2):**",
        "",
        "- `cuda:0`: answering LLM (llama-cpp-python loads here by default).",
        "- `cuda:1`: explicitly pinned. bge embedders, cross-encoder reranker,",
        "  Whisper. Retrieval and ASR are isolated from the LLM's device so that",
        "  reranking completes in under 100 ms rather than contending for memory.",
    )
)

CELLS.append(
    md(
        "### The strategy contract",
        "",
        "Each answering method is a callable returning the same shape. This",
        "uniformity permits the recording layer to log them identically, the",
        "replay harness to benchmark them interchangeably, and the router to",
        "swap them by competition.",
        "",
        "```python",
        "# src/polimillionaire/strategies/base.py",
        "@dataclass(frozen=True)",
        "class AnswerDecision:",
        "    option_id: int",
        "    confidence: float | None = None",
        "    rationale: str | None = None",
        '    model_name: str = ""',
        '    strategy_name: str = ""',
        '    prompt_version: str = ""',
        "    latency_ms: int = 0",
        "",
        "@dataclass(frozen=True)",
        "class Context:",
        "    competition_id: int",
        "    level: int",
        "",
        "class Strategy(Protocol):",
        "    def __call__(self, question: Question, ctx: Context) -> AnswerDecision: ...",
        "```",
    )
)

CELLS.append(
    md(
        "## Bootstrap",
        "",
        "Detects Kaggle / Colab / local. On Kaggle: clones the repo, installs",
        "what's missing, attaches the pre-built indices dataset by symlink,",
        "and pulls the latest shared question log via `polimillionaire.kaggle_db.pull_db`.",
        "On Colab or locally: best-effort fallback for grading convenience.",
    )
)

CELLS.append(
    code(
        "from __future__ import annotations",
        "",
        "import contextlib",
        "import importlib.util",
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
        "# ---- master switch ---------------------------------------------------------",
        "# False -> render only. True -> bootstrap + load the LLM + play real games.",
        "RUN_LIVE = False",
        "",
        "# ---- environment detection -------------------------------------------------",
        "ON_KAGGLE = Path('/kaggle/working').exists()",
        "ON_COLAB  = importlib.util.find_spec('google.colab') is not None",
        "ON_LOCAL  = not (ON_KAGGLE or ON_COLAB)",
        "print('env:', 'kaggle' if ON_KAGGLE else 'colab' if ON_COLAB else 'local')",
        "",
        "REPO_DIR = Path('/kaggle/working/polimillionaire') if ON_KAGGLE else Path.cwd()",
        "if ON_LOCAL and not (REPO_DIR / 'src' / 'polimillionaire').exists():",
        "    # When the notebook is opened from the repo, cwd is notebooks/. Step up.",
        "    REPO_DIR = REPO_DIR.parent",
        "",
        "# ---- Kaggle bootstrap ------------------------------------------------------",
        "if ON_KAGGLE and RUN_LIVE:",
        "    from huggingface_hub import login",
        "    from kaggle_secrets import UserSecretsClient",
        "",
        "    secrets = UserSecretsClient()",
        "    gh_token = secrets.get_secret('GH_TOKEN')",
        "    os.environ['POLIMILLIONAIRE_API_URL']  = secrets.get_secret('POLIMILLIONAIRE_API_URL')",
        "    os.environ['POLIMILLIONAIRE_USER']     = secrets.get_secret('POLIMILLIONAIRE_USER')",
        "    os.environ['POLIMILLIONAIRE_PASSWORD'] = secrets.get_secret('POLIMILLIONAIRE_PASSWORD')",
        "    # Tavily is optional; cid 5 (news) falls back to MediaWiki if absent.",
        "    with contextlib.suppress(Exception):",
        "        os.environ['TAVILY_API_KEY'] = secrets.get_secret('TAVILY_KEY')",
        "    login(token=secrets.get_secret('HF_TOKEN'), add_to_git_credential=False)",
        "",
        "    if REPO_DIR.exists():",
        "        os.chdir(REPO_DIR)",
        "        os.system('git checkout -q main && git pull --ff-only origin main')",
        "    else:",
        "        os.system(f'git clone -q https://{gh_token}@github.com/leonfuss/polimillionaire.git {REPO_DIR}')",
        "        os.chdir(REPO_DIR)",
        "",
        "    NEEDED = {'sentence_transformers':'sentence-transformers','faiss':'faiss-cpu',",
        "              'bm25s':'bm25s','polars':'polars','dotenv':'python-dotenv','sympy':'sympy'}",
        "    missing = [pkg for mod, pkg in NEEDED.items() if importlib.util.find_spec(mod) is None]",
        "    if missing:",
        "        os.system('pip install -q ' + ' '.join(missing))",
        "    if importlib.util.find_spec('llama_cpp') is None:",
        "        os.system('pip install -q llama-cpp-python --extra-index-url '",
        "                  'https://abetlen.github.io/llama-cpp-python/whl/cu122')",
        "    os.system(f'pip install -q -e {REPO_DIR} --no-deps')",
        "",
        "    # FAISS/BM25 indices live in a separate Kaggle Dataset, attached via UI.",
        "    INDEX_CANDIDATES = [",
        "        Path('/kaggle/input/polimillionaire-indices'),",
        "        Path('/kaggle/input/datasets/leonfuss/polimillionaire-indices'),",
        "    ]",
        "    INDEX_SRC = next((p for p in INDEX_CANDIDATES if p.exists()), None)",
        "    INDEX_DST = REPO_DIR / 'data' / 'index'",
        "    INDEX_DST.parent.mkdir(parents=True, exist_ok=True)",
        "    if INDEX_SRC and not INDEX_DST.exists():",
        "        INDEX_DST.symlink_to(INDEX_SRC)",
        "",
        "if str(REPO_DIR / 'src') not in sys.path:",
        "    sys.path.insert(0, str(REPO_DIR / 'src'))",
        "",
        "# ---- shared question log: pull freshest from Kaggle ------------------------",
        "DB_PATH = REPO_DIR / 'data' / 'questions.sqlite'",
        "try:",
        "    from polimillionaire.kaggle_db import pull_db",
        "    target = '/kaggle/working/questions.sqlite' if ON_KAGGLE else str(DB_PATH)",
        "    DB_PATH = pull_db(target_path=target)",
        "    os.environ['POLIMILLIONAIRE_DB_PATH'] = str(DB_PATH)",
        "    print('pulled latest DB ->', DB_PATH)",
        "except Exception as e:",
        "    # Falls back to whatever is already on disk. Fine for the read-only render.",
        "    print(f'kaggle pull skipped ({type(e).__name__}: {e}); using local DB at {DB_PATH}')",
        "    os.environ['POLIMILLIONAIRE_DB_PATH'] = str(DB_PATH)",
        "",
        "RESULTS_DIR = REPO_DIR / 'data' / 'results'",
        "print('DB:     ', DB_PATH, '| exists:', DB_PATH.exists())",
        "print('results:', RESULTS_DIR, '| exists:', RESULTS_DIR.exists())",
        "",
        "# ---- retrieval on cuda:1 (only when we actually plan to play) --------------",
        "if RUN_LIVE and ON_KAGGLE:",
        "    from polimillionaire.retrieval.embedder import Embedder",
        "    from polimillionaire.retrieval.reranker import DEFAULT_RERANKER, Reranker",
        "    from polimillionaire.strategies import factory",
        "",
        "    RETRIEVAL_DEVICE = 'cuda:1'",
        "    for model in ('BAAI/bge-base-en-v1.5', 'BAAI/bge-small-en-v1.5'):",
        "        factory._embedder_cache[model] = Embedder(model, device=RETRIEVAL_DEVICE)",
        "    factory._reranker_cache[DEFAULT_RERANKER] = Reranker(DEFAULT_RERANKER, device=RETRIEVAL_DEVICE)",
        "    print('retrieval pinned to', RETRIEVAL_DEVICE)",
    )
)

# ---------------------------------------------------------------------------
# Strategies. Narrative sections with code excerpts.
# ---------------------------------------------------------------------------
CELLS.append(
    md(
        "## Strategy 1: Zero-shot",
        "",
        "The question and its options are passed to a single LLM, and the chosen option is parsed under a fixed schema.",
        "The JSON schema lists `rationale` before",
        "`answer_id` so that the model produces its reasoning before commitment. This yields a single-call chain of thought without a",
        "separate prompting stage. Grammar-constrained JSON via llama.cpp's",
        "GBNF guarantees that the output parses on every call and removes the need for regex retries.",
        "",
        "```python",
        "# src/polimillionaire/strategies/_common.py  (excerpt)",
        "def make_schema(question: Question, *, include_rationale: bool = True):",
        "    option_ids = [opt.id for opt in question.options]",
        "    return {",
        '        "type": "object",',
        '        "properties": {',
        '            "rationale":  {"type": "string", "maxLength": 500},',
        '            "confidence": {"type": "number", "minimum": 0, "maximum": 1},',
        '            "answer_id":  {"type": "integer", "enum": option_ids},',
        "        },",
        '        "required": ["rationale", "confidence", "answer_id"],',
        '        "additionalProperties": False,',
        "    }",
        "```",
        "",
        "Property order is load-bearing. GBNF emits fields in declaration order,",
        "so listing `rationale` first provides a reasoning window before",
        "commitment. A 500-character `maxLength` prevents the long-tail failure",
        "mode in which the model rambled past the 30 s timer, observed in early",
        "sessions.",
    )
)

CELLS.append(
    md(
        "## Strategy 2: Hybrid Wikipedia RAG",
        "",
        "For the Wikipedia-style competitions (Entertainment, History, Science),",
        "a hybrid retrieval pipeline is used.",
        "",
        "**Pipeline.** For each question, two parallel searches are issued over a",
        "pre-chunked Wikipedia index, BM25 (sparse, lexical) and dense (bge-base",
        "embeddings via FAISS). The two rankings are merged with Reciprocal",
        "Rank Fusion:",
        "",
        "```python",
        "# src/polimillionaire/retrieval/fusion.py",
        "def reciprocal_rank_fusion(rankings, *, k=60, top_n=None):",
        '    """RRF score = sum(1 / (k + rank))."""',
        "    rrf_scores = {}",
        "    for ranking in rankings:",
        "        for rank, passage in enumerate(ranking, start=1):",
        "            rrf_scores[passage.id] = rrf_scores.get(passage.id, 0.0) + 1.0/(k + rank)",
        "    fused = sorted(rrf_scores, key=rrf_scores.get, reverse=True)",
        "    return [... top_n ...]",
        "```",
        "",
        "A cross-encoder reranker (`BAAI/bge-reranker-base`) re-scores the fused",
        "pool by reading the question and each passage jointly, which is far more sensitive than",
        "either embedding similarity alone. Top-K is capped at 5 for science and",
        "history, 8 for entertainment; passages scoring below the rerank-score",
        "floor of `0.15` are discarded. Below this threshold, retrieved context is more likely to",
        "mislead than to assist, and the model falls back on its parametric knowledge.",
        "",
        "**Live augmentation.** A live MediaWiki retriever runs in parallel, and",
        "its results are fused into the same rerank pool, de-duplicated by article",
        "title against the static hits. For the News competition (cid 5) the",
        "live source is Tavily's news API instead. Live retrieval is not a",
        "fallback. It executes on every question and recovers the long tail, such as recent",
        "films, niche scientists, and breaking news, that the frozen index does not cover.",
        "",
        "```python",
        "# src/polimillionaire/strategies/wiki_rag.py  (excerpt)",
        "rankings = []",
        "if self._use_dense:  rankings.append(self._retriever.search(query, k=self._dense_k))",
        "if self._use_sparse: rankings.append(self._bm25.search(query, k=self._sparse_k))",
        "fused = reciprocal_rank_fusion(rankings, top_n=self._fused_k)",
        "",
        "if self._live is not None:",
        "    live = self._live.search(question.text, k=self._live_k,",
        "                              option_texts=[o.text for o in question.options])",
        "    # dedup by article title, then merge into the rerank pool",
        "    pool = fused + [p for p in live if p.metadata.get('title','').lower()",
        "                                       not in static_titles]",
        "",
        "top_passages = self._reranker.rerank(query, pool, top_k=self._top_k)",
        "top_passages = [p for p in top_passages if p.score >= self._min_rerank_score]",
        "```",
    )
)

CELLS.append(
    md(
        "## Strategy 3: Calculator agent for Math (`calc_react` and `rag_calc_react`)",
        "",
        "Mathematics questions do not yield to Wikipedia RAG, because an encyclopaedia contains nothing to retrieve",
        ". The LLM is therefore coupled to a SymPy calculator. Two",
        "layered variants live in the package and share the same ReAct loop.",
        "",
        "### Layer 1: `calc_react` (calculator only)",
        "",
        "The output schema is a `oneOf`. The model emits either a `calculate`",
        "action with a SymPy expression, or commits to an `answer`. Tool calls",
        "and answers share a single structured-output channel, so the loop is",
        "one `complete_json` per step.",
        "",
        "```python",
        "# src/polimillionaire/strategies/_common.py  (action schema)",
        "make_action_schema = {",
        '    "oneOf": [',
        '        {"properties": {"action":     {"const": "calculate"},',
        '                        "expression": {"type": "string", "maxLength": 200}}},',
        '        {"properties": {"action":     {"const": "answer"},',
        '                        "rationale":  {"type": "string", "maxLength": 500},',
        '                        "confidence": {"type": "number"},',
        '                        "answer_id":  {"type": "integer", "enum": option_ids}}},',
        "    ]",
        "}",
        "```",
        "",
        "The `math-tir` prompt variant is used, tuned for tool-integrated",
        "reasoning, with `max_steps=3`. Up to three `calculate` calls are permitted before",
        "the schema is swapped for an answer-only variant and a commit is forced.",
        "",
        "The calculator is hardened. `sympy.sympify` (rather than `eval`) ensures that",
        "malformed input fails closed. A `MAX_POW_EXPONENT=10_000` guard rejects",
        "expressions such as `10**(10**10)` that would otherwise hang CPython's bignum path",
        "past the 30 s timer. Output is capped at 600 characters.",
        "Symbolic outputs include both the symbolic form and its decimal expansion. For",
        'example, `Rational(27, 99)` returns `"3/11 = 0.272727272727273"`, which permits',
        "matching against fraction options without simplification in",
        "the model's head.",
        "",
        "### Layer 2: `rag_calc_react` (calculator + retrieved reference problems)",
        "",
        "`rag_calc_react` is the strategy the live router ships for",
        "the mathematics competition. Before the ReAct loop begins, it retrieves the",
        "top-K most similar problems from a separate MATH index, built from a",
        "curated problem set rather than Wikipedia, and prepends them to the system",
        "message as worked natural-language reference solutions.",
        "",
        "Two retrieval signals layered:",
        "",
        "1. The hand-crafted ReAct exemplars in the prompt teach the *action",
        "   format*: when to call `calculate`, when to commit to `answer`,",
        "   how to chain partial results.",
        '2. The k=3 retrieved problems teach the *math pattern*: "questions',
        '   like this are usually solved by such-and-such substitution".',
        "",
        "```python",
        "# src/polimillionaire/strategies/rag_calc_react.py  (call-site excerpt)",
        "def __call__(self, question, ctx):",
        "    start = time.perf_counter()  # latency_ms includes retrieval",
        "    try:",
        "        references = self._retriever.search(question.text, k=self._k)",
        "    except Exception:",
        "        # Degrade silently to plain calc_react if the MATH index is",
        "        # missing or FAISS errors out. The prompt renders with an",
        "        # empty reference block and the game keeps playing.",
        "        references = []",
        "    messages = self._variant.render(question, references)",
        "    return run_react_loop(self._llm, messages, question,",
        "                          max_steps=self._max_steps, ...)",
        "```",
        "",
        "Live configuration in the router: `make_strategy('rag_calc_react', llm,",
        "prompt_version='math-tir', k=3, max_steps=3, min_rerank_score=0.15)`.",
        "Retrieval time is counted against the question's latency budget by",
        "design, since the 30 s timer measures end-to-end wall-clock latency",
        "rather than LLM-only.",
        "",
        "**Why both variants exist.** `calc_react` is the clean ablation",
        "baseline that isolates the contribution of the calculator itself. The gallery",
        "below was produced with the plain `calc_react` variant for this reason.",
        "`rag_calc_react` is the shipped configuration; in informal live play it adds",
        "a further one to three points on harder mathematics questions where pattern-matching",
        "against a similar worked example matters more than raw arithmetic.",
        "",
        "**Failure modes.** Even with the tool and the retrieved",
        "exemplars, residual errors cluster in conceptual statistics, probability,",
        "and abstract algebra. These are questions with nothing to compute, where the",
        "calculator cannot assist and no similar worked example exists in the",
        "MATH index. This is the gap between `calc_react`/`rag_calc_react` and a",
        "true reasoning specialist.",
    )
)

CELLS.append(
    md(
        "## Strategy 4: Router",
        "",
        "Each competition is assigned the strategy best suited to its question style.",
        "The router inspects `Context.competition_id` and dispatches to the",
        "appropriate sub-strategy. The per-question record retains the",
        "sub-strategy name, so offline replay attributes results correctly.",
        "",
        "```python",
        "# src/polimillionaire/strategies/routed.py",
        "def __call__(self, question, ctx):",
        "    strategy = self._routes.get(ctx.competition_id, self._default)",
        "    return strategy(question, ctx)",
        "```",
        "",
        "| Competition | Route |",
        "|---|---|",
        "| 0 Entertainment | `wiki_rag` with live MediaWiki |",
        "| 1 History       | `wiki_rag` with live MediaWiki |",
        "| 2 Science       | `wiki_rag` with live MediaWiki |",
        "| 3 Math          | `rag_calc_react` (math-tir prompt, max_steps=3) |",
        "| 4 Philosophy    | `wiki_rag`, live-only (no static index) |",
        "| 5 News          | `wiki_rag`, live Tavily news (no static index) |",
    )
)

CELLS.append(
    md(
        "## Speech mode",
        "",
        "The server also exposes an audio endpoint, in which each question and its four options",
        "arrive as 24 kHz mono 16-bit WAV blobs. Transcription is performed with",
        "`openai/whisper-large-v3-turbo` (fp16, greedy decoding, English pinned),",
        "polyphase-resampled to 16 kHz, on `cuda:1`, so that no memory is shared with",
        "the answering LLM.",
        "",
        "```python",
        "# src/polimillionaire/asr/whisper.py  (excerpt)",
        "TARGET_SAMPLE_RATE = 16_000",
        "DEFAULT_DEVICE     = 'cuda:1'",
        "",
        "def _resample_to_16k(audio, sample_rate):",
        "    g = gcd(sample_rate, TARGET_SAMPLE_RATE)",
        "    return resample_poly(audio, up=TARGET_SAMPLE_RATE // g,",
        "                                down=sample_rate // g).astype(np.float32)",
        "```",
        "",
        '**Transcript hygiene.** Spoken option prefixes such as "Option one" or "Option B"',
        "appear in the Whisper output and must be stripped before the LLM sees them.",
        "Once transcribed and cleaned, the audio question enters the same answering",
        "pipeline as the text-mode equivalent, with the speech `mode` flag recorded",
        "alongside the prediction so the two modes can be analysed separately offline.",
    )
)

CELLS.append(
    md(
        "## The shared question log as canonical artefact",
        "",
        "Every prediction is logged as a row in `data/questions.sqlite`,",
        "opened in WAL mode for concurrent writers and mirrored to a private Kaggle Dataset",
        "between sessions. The log is the project's most valuable artefact.",
        "Server-side questions are closed: each question seen, paired with the answer",
        "the server confirmed, becomes a permanent training datapoint that no party",
        "outside the team can recreate.",
        "",
        "**Schema** (one row per prediction event, not per question):",
        "",
        "```sql",
        "CREATE TABLE predictions (",
        "    id, timestamp, account_username, session_id,",
        "    competition_id, level,",
        "    question_id, question_text, options_json,",
        "    predicted_option_id, correct_option_id_if_known,",
        "    strategy_name, model_name, prompt_version,",
        "    confidence, rationale, latency_ms,",
        "    generated_answer,  /* 1 if we self-labelled, 0 if server-confirmed */",
        "    mode               /* 'text' or 'speech' */",
        ")",
        "```",
        "",
        "Running the same question through three strategies produces three rows.",
        "This is what enables offline replay: any new strategy, model, or ",
        "prompt can be rerun over hundreds of past questions in a few minutes, rather than",
        "waiting for live games. All figures below are drawn from this log.",
    )
)

CELLS.append(
    md(
        "## Live leaderboard from the SQLite log",
        "",
        "The leaderboard is aggregated directly from the freshly-pulled `predictions` table, using SQL and Polars with no LLM",
        "calls. It is the cheapest, freshest view of how",
        "each (strategy, model) combination has performed across the log.",
    )
)

CELLS.append(
    code(
        "import sqlite3",
        "",
        "import polars as pl",
        "",
        "COMPETITION_NAMES = {",
        "    0: 'Entertainment', 1: 'History', 2: 'Science', 3: 'Math',",
        "    4: 'Philosophy', 5: 'News',",
        "}",
        "",
        "# Truth comes from MAX(correct_option_id_if_known) per question_id:",
        "# during live play correct_option_id_if_known is only filled in when the",
        "# server confirms we got the answer right, so any single row with a",
        "# non-null value is reliable ground truth that we can join back onto",
        "# every prediction for that same question (right or wrong) to compute",
        "# honest per-strategy accuracy.",
        "with sqlite3.connect(DB_PATH) as con:",
        "    cols = {r[1] for r in con.execute('PRAGMA table_info(predictions)').fetchall()}",
        "    has_mode = 'mode' in cols",
        "    mode_select = ', mode' if has_mode else ''",
        "    truth = pl.read_database(",
        "        'SELECT question_id, competition_id,'",
        "        ' MAX(correct_option_id_if_known) AS truth'",
        "        ' FROM predictions'",
        "        ' WHERE correct_option_id_if_known IS NOT NULL'",
        "        ' GROUP BY question_id, competition_id',",
        "        con,",
        "    )",
        "    preds = pl.read_database(",
        "        'SELECT question_id, competition_id, model_name, strategy_name,'",
        "        ' prompt_version, predicted_option_id, latency_ms' + mode_select +",
        "        ' FROM predictions',",
        "        con,",
        "    )",
        "",
        "if not has_mode:",
        "    preds = preds.with_columns(pl.lit('text').alias('mode'))",
        "",
        "df = (",
        "    preds.join(truth, on=['question_id', 'competition_id'], how='inner')",
        "         .with_columns(",
        "             (pl.col('predicted_option_id') == pl.col('truth')).alias('correct'),",
        "             pl.col('competition_id').replace_strict(COMPETITION_NAMES, default='?').alias('competition'),",
        "         )",
        ")",
        "",
        "leaderboard = (",
        "    df.filter(pl.col('mode') == 'text')",
        "      .group_by(['competition', 'strategy_name', 'model_name'])",
        "      .agg(",
        "          pl.col('correct').count().alias('n'),",
        "          (pl.col('correct').mean() * 100).round(1).alias('acc_%'),",
        "          pl.col('latency_ms').median().cast(int).alias('lat_ms_p50'),",
        "      )",
        "      .filter(pl.col('n') >= 20)",
        "      .sort(['competition', 'acc_%'], descending=[False, True])",
        ")",
        "",
        "with pl.Config(tbl_rows=80, tbl_cols=10, fmt_str_lengths=40):",
        "    print(leaderboard)",
    )
)

# ---------------------------------------------------------------------------
# Results gallery
# ---------------------------------------------------------------------------
CELLS.append(
    md(
        "## Results gallery: Science & Math (re-rendered from logged predictions)",
        "",
        "Re-rendered in a uniform style from `data/results/{science,math}/",
        "ablation_exports/01_df_ablation_raw_predictions.parquet`. Each parquet is",
        "the exact set of predictions a teammate's ablation run produced on Kaggle",
        "with the LLM in the loop, at N=200 questions per (model, strategy) combo.",
        "Three panels per competition:",
        "",
        "- **A. Global accuracy** per combo.",
        "- **B. Accuracy by question level**, flat across difficulty,",
        "  consistent with empirical level uniformity.",
        "- **C. Latency** (log-scale ms). Every combo stays well under the",
        "  30 s server timer.",
    )
)

CELLS.append(
    code(
        "from pathlib import Path",
        "",
        "import matplotlib.pyplot as plt",
        "import pandas as pd",
        "import seaborn as sns",
        "",
        "sns.set_theme(style='whitegrid')",
        "",
        "def render_competition_gallery(parquet_path: Path, title: str):",
        "    df = pd.read_parquet(parquet_path)",
        "    df['combo'] = df['model_name'] + '\\n' + df['strategy_name']",
        "    df['correct_int'] = df['correct'].astype(int)",
        "",
        "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
        "",
        "    # A. global accuracy",
        "    acc = (df.groupby(['model_name','strategy_name'])['correct_int']",
        "             .mean().reset_index()",
        "             .rename(columns={'correct_int':'win_rate'}))",
        "    sns.barplot(",
        "        data=acc, x='model_name', y='win_rate', hue='strategy_name',",
        "        ax=axes[0], palette='deep',",
        "    )",
        "    axes[0].set_title(f'A. Global accuracy ({title})')",
        "    axes[0].set_ylim(0, 1.0)",
        "    axes[0].set_ylabel('Win rate')",
        "    axes[0].set_xlabel('')",
        "    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{int(v*100)}%'))",
        "",
        "    # B. accuracy by level",
        "    lvl = (df.groupby(['level','model_name','strategy_name'])['correct_int']",
        "             .mean().reset_index())",
        "    sns.lineplot(",
        "        data=lvl, x='level', y='correct_int',",
        "        hue='strategy_name', style='model_name',",
        "        markers=True, dashes=False, ax=axes[1],",
        "    )",
        "    axes[1].set_title('B. Accuracy per level')",
        "    axes[1].set_xlabel('Question level (difficulty)')",
        "    axes[1].set_ylabel('Win rate')",
        "    axes[1].set_ylim(0, 1.05)",
        "    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{int(v*100)}%'))",
        "",
        "    # C. latency log-scale",
        "    sns.boxplot(",
        "        data=df, x='strategy_name', y='latency_ms', hue='model_name',",
        "        ax=axes[2], palette='deep',",
        "    )",
        "    axes[2].set_yscale('log')",
        "    axes[2].set_title('C. Latency per combo (log)')",
        "    axes[2].set_ylabel('Latency (ms)')",
        "    axes[2].set_xlabel('')",
        "",
        "    fig.suptitle(title, fontsize=14, y=1.02)",
        "    plt.tight_layout()",
        "    plt.show()",
        "",
        "render_competition_gallery(RESULTS_DIR / 'science' / 'ablation_exports' / '01_df_ablation_raw_predictions.parquet',",
        "                            title='Science (cid 2, N=200)')",
        "render_competition_gallery(RESULTS_DIR / 'math'    / 'ablation_exports' / '01_df_ablation_raw_predictions.parquet',",
        "                            title='Math (cid 3, N=200)')",
    )
)

CELLS.append(
    md(
        "**Reading the panels.**",
        "",
        "- *Science*: `phi4-14b` zero-shot tops the board at 94.5%; the other",
        "  two models trail by 2.5 points. Zero-shot proved the strongest science strategy",
        "  measured. `wiki_rag` under-performed: chunk granularity",
        "  was poorly matched to the highly specific scientific wording, so the",
        "  parametric baseline was retained.",
        "- *Mathematics*: `phi4-14b + calc_react` wins at 91.5%, beating its own",
        "  few-shot baseline by 9.5 points and `granite-8b`'s `calc_react` by 7.5 points.",
        "  Tool (SymPy), prompt (math-tir), and model",
        "  combine multiplicatively. `qwen3-14b` zero-shot reaches",
        "  88.5%, but its incorrect answers take roughly 6 s longer than its",
        "  correct ones (panel C); the model rambles when uncertain, a",
        "  diagnostic signal.",
    )
)

CELLS.append(
    md(
        "## Entertainment & History: cached figures",
        "",
        "Same three-panel layout, embedded from the teammate notebooks that",
        "produced them on Kaggle. We do not have raw parquet exports for these",
        "two competitions, only the figures themselves.",
    )
)

CELLS.append(
    code(
        "import os",
        "",
        "from IPython.display import Image, display",
        "",
        "for tag, caption in [('comp0', 'Entertainment (cid 0, N=400)'),",
        "                     ('comp1', 'History (cid 1, N=400)')]:",
        "    fig_dir = RESULTS_DIR / tag",
        "    if not fig_dir.exists():",
        "        print(f'{tag}: figure dir missing ({fig_dir})')",
        "        continue",
        "    # The 3-panel headline figure is always fig_00.",
        "    headline = sorted(fig_dir.glob('fig_00_*.png'))",
        "    print(f'--- {caption} ---')",
        "    for f in headline:",
        "        display(Image(str(f)))",
    )
)

CELLS.append(
    md(
        "**Entertainment.** All three models sit in the 85% to 90% band. `phi4-14b` and",
        "`gemma3-12b` zero-shot both clear 90%, while `qwen3-14b + wiki_rag` falls to",
        "about 85%. RAG actively harms accuracy on average for this competition. Entertainment questions",
        'rely on common-word distractors ("Which actor played X in 1987?"),',
        "where retrieval surfaces noisy candidates that the reranker cannot",
        "disambiguate; the parametric prior prevails.",
        "",
        "**History.** A near-tie: `granite-8b + wiki_rag` at approximately 92%, `phi4-14b`",
        "zero-shot at 91%, and `gemma3-12b + wiki_rag` at 91%. Wikipedia chunks carry",
        "tight technical vocabulary for history, and Wikipedia is itself the source",
        "material, so retrieval augmentation pays off, albeit by only one point over a competitive",
        "zero-shot baseline. The cost is visible in panel C: the `wiki_rag` latency",
        "distribution is wider and skewed higher than that of zero-shot.",
    )
)

CELLS.append(
    md(
        "## Philosophy & News: the live-only competitions (cid 4, 5)",
        "",
        "Neither competition was subjected to a formal ablation harness, for a structural reason: no static FAISS+BM25 index was pre-built for",
        "either. The rationale follows.",
        "",
        "",
        "- **Philosophy & Psychology (cid 4)** has a long tail of niche thinkers",
        "  and minor schools where Wikipedia coverage is uneven. Crawling a",
        "  quality-controlled static index would have required curating seed pages",
        "  by hand; deferring to the live MediaWiki API on each question was more economical.",
        "- **News (cid 5)** is adversarial to a static index by design. Many of",
        "  the questions reference events occurring after the crawl date. Tavily News",
        "  (a live news search API) is the only viable source.",
        "",
        "The auto-router therefore treats both as live-only. The same `wiki_rag`",
        "strategy is instantiated, but with `use_dense=False`, `use_sparse=False`,",
        "and a `live` retriever as the sole source: `LiveWikiRetriever` for cid 4,",
        "`LiveTavilyRetriever` for cid 5. The cross-encoder reranker and the",
        "0.15 rerank-score floor still apply, so if live retrieval returns",
        "off-topic articles we fall back to the model's parametric knowledge",
        "rather than poison the prompt.",
        "",
        "Per-question latency is higher than for the static-index comps (~1 to",
        "3 s added for the live API round-trip), but still inside the 30-second",
        "budget.",
        "",
        "The numbers below are recomputed live from the SQLite log and reflect",
        "whatever games have been played on cid 4 and cid 5 by the",
        "time the notebook is opened. If the kernel has just pulled the freshest",
        "version of the dataset, the table below is current.",
    )
)

CELLS.append(
    code(
        "import sqlite3",
        "",
        "import polars as pl",
        "",
        "LIVE_ONLY_CIDS = [4, 5]",
        "",
        "# Same truth-join pattern as the leaderboard: per-question_id we take",
        "# MAX(correct_option_id_if_known) as the authoritative answer and join",
        "# it onto every prediction for that question (right or wrong).",
        "cid_list = ','.join(str(c) for c in LIVE_ONLY_CIDS)",
        "with sqlite3.connect(DB_PATH) as con:",
        "    cols = {r[1] for r in con.execute('PRAGMA table_info(predictions)').fetchall()}",
        "    has_mode = 'mode' in cols",
        "    mode_select = ', mode' if has_mode else ''",
        "    truth = pl.read_database(",
        "        f'SELECT question_id, competition_id,'",
        "        ' MAX(correct_option_id_if_known) AS truth'",
        "        f' FROM predictions WHERE competition_id IN ({cid_list})'",
        "        '   AND correct_option_id_if_known IS NOT NULL'",
        "        ' GROUP BY question_id, competition_id',",
        "        con,",
        "    )",
        "    preds = pl.read_database(",
        "        'SELECT question_id, competition_id, model_name, strategy_name,'",
        "        ' prompt_version, predicted_option_id, latency_ms' + mode_select +",
        "        f' FROM predictions WHERE competition_id IN ({cid_list})',",
        "        con,",
        "    )",
        "",
        "if not has_mode:",
        "    preds = preds.with_columns(pl.lit('text').alias('mode'))",
        "",
        "live_df = preds.join(truth, on=['question_id', 'competition_id'], how='inner')",
        "live_df = live_df.filter(pl.col('mode') == 'text') if has_mode else live_df",
        "",
        "if live_df.is_empty():",
        "    print('No cid=4 or cid=5 rows with derivable truth in this DB snapshot yet.')",
        "    print('Run the live-play cell below on Kaggle with COMPETITIONS=[4,5] to populate.')",
        "else:",
        "    live_df = live_df.with_columns(",
        "        (pl.col('predicted_option_id') == pl.col('truth')).alias('correct'),",
        "        pl.col('competition_id').replace_strict({4: 'Philosophy', 5: 'News'}).alias('competition'),",
        "    )",
        "    summary = (",
        "        live_df.group_by(['competition', 'model_name', 'strategy_name'])",
        "               .agg(",
        "                   pl.col('correct').count().alias('n'),",
        "                   (pl.col('correct').mean() * 100).round(1).alias('acc_%'),",
        "                   pl.col('latency_ms').median().cast(int).alias('lat_ms_p50'),",
        "                   pl.col('latency_ms').quantile(0.9).cast(int).alias('lat_ms_p90'),",
        "               )",
        "               .filter(pl.col('n') >= 5)",
        "               .sort(['competition', 'acc_%'], descending=[False, True])",
        "    )",
        "    with pl.Config(tbl_rows=40, tbl_cols=10, fmt_str_lengths=40):",
        "        print(summary)",
        "",
        "    import matplotlib.pyplot as plt",
        "    import seaborn as sns",
        "",
        "    sns.set_theme(style='whitegrid')",
        "    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)",
        "    for ax, cid, name in zip(axes, LIVE_ONLY_CIDS, ['Philosophy', 'News'], strict=True):",
        "        sub = live_df.filter(pl.col('competition_id') == cid)",
        "        if sub.is_empty():",
        "            ax.text(0.5, 0.5, f'no {name} data', ha='center', va='center',",
        "                    transform=ax.transAxes)",
        "            ax.set_title(f'{name} (cid {cid})')",
        "            ax.set_axis_off()",
        "            continue",
        "        agg = (sub.group_by(['model_name', 'strategy_name'])",
        "                  .agg(pl.col('correct').mean().alias('acc'),",
        "                       pl.col('correct').count().alias('n'))",
        "                  .filter(pl.col('n') >= 5)",
        "                  .sort('acc', descending=True)",
        "                  .to_pandas())",
        "        if agg.empty:",
        "            ax.text(0.5, 0.5, f'no {name} combo with n>=5', ha='center', va='center',",
        "                    transform=ax.transAxes)",
        "            ax.set_title(f'{name} (cid {cid})')",
        "            ax.set_axis_off()",
        "            continue",
        "        agg['combo'] = agg['model_name'] + '\\n' + agg['strategy_name']",
        "        sns.barplot(data=agg, x='combo', y='acc', ax=ax, palette='deep',",
        "                    hue='combo', legend=False)",
        "        ax.set_ylim(0, 1.0)",
        "        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{int(v*100)}%'))",
        "        ax.set_title(f'{name} (cid {cid}, total n={int(agg[\"n\"].sum())})')",
        "        ax.set_xlabel('')",
        "        ax.set_ylabel('Win rate' if ax is axes[0] else '')",
        "        for tick in ax.get_xticklabels():",
        "            tick.set_fontsize(8)",
        "    plt.tight_layout()",
        "    plt.show()",
    )
)

CELLS.append(
    md(
        "## Few-shot context scaling, and where it breaks",
        "",
        "With everything held constant (model `phi4-14b`, Entertainment, N=50 per K),",
        "the number of in-prompt examples was swept across K ∈ {0, 1, 2, 3, 5, 7, 10,",
        "15, 30, 50, 75, 100, 150}. Two questions are asked.",
        "",
        "1. **The sweet spot** (top figure, K=0 to K=50).",
        "2. **The saturation regime** (bottom figure, K=75 to K=150).",
    )
)

CELLS.append(
    code(
        "from IPython.display import Image, display",
        "",
        "scaling_dir = RESULTS_DIR / 'scaling'",
        "if scaling_dir.exists():",
        "    for fname in ['fig_06_cell42_out0.png', 'fig_07_cell46_out0.png']:",
        "        fp = scaling_dir / fname",
        "        if fp.exists():",
        "            display(Image(str(fp)))",
        "        else:",
        "            print(f'missing: {fp}')",
        "else:",
        "    print('scaling figures missing')",
    )
)

CELLS.append(
    md(
        "**Reading the curves.**",
        "",
        "- Accuracy is flat from K=1 to K=50 (88% to 90%). The bump at",
        "  K=2 to K=3 marks the practical sweet spot: identical accuracy to K=50 at lower",
        "  latency (7.1 s median at K=3 against 11.4 s at K=50).",
        "- Latency rises monotonically with K, since each additional in-context example must be",
        "  attended to, until K=100, where the prompt approaches the model's",
        "  KV-cache limit of 8 192 tokens.",
        "- At K=150 the prompt exceeds the cache and accuracy degrades to",
        "  28%, barely above the 25% random-guess baseline. Latency",
        "  collapses in parallel, since the model emits truncated output rapidly. This is",
        "  the canonical long-context cliff.",
        "",
        "Operational consequence: `dynamic_few_shot_k3` is the shipped few-shot",
        "variant; larger K is never used.",
    )
)

CELLS.append(
    md(
        "## Live play",
        "",
        "The remaining cells run a real game against the PoliMillionaire server.",
        "They execute only when `RUN_LIVE` is set to `True` in the bootstrap",
        "cell. Each cell mirrors the loop in `notebooks/polimillionaire.ipynb`,",
        "the Kaggle notebook used for day-to-day play.",
        "",
        "### Text-mode play",
    )
)

CELLS.append(
    code(
        "if RUN_LIVE:",
        "    from polimillionaire import load_llm, make_client, preload",
        "    from polimillionaire.play import auto_play_loop",
        "    from polimillionaire.strategies.factory import make_strategy",
        "",
        "    # Load the answering LLM. Defaults to phi4-14b (the gallery winner).",
        "    # Switch to another alias from polimillionaire.MODELS for ablation.",
        "    llm = load_llm('phi4-14b')",
        "",
        "    COMPETITIONS = [0, 1, 2, 3]   # or e.g. [3] for math-only",
        "    MAX_GAMES    = 12",
        "    MAX_STEPS    = 3              # rag_calc_react / calc_react only",
        "    RAG_K        = 3              # retrieved math exemplars",
        "",
        "    def build_strategy(cid: int):",
        "        if cid == 3:",
        "            return make_strategy(",
        "                'rag_calc_react', llm,",
        "                prompt_version='math-tir', verbose=True,",
        "                max_steps=MAX_STEPS, k=RAG_K,",
        "                min_rerank_score=0.15,",
        "            )",
        "        return make_strategy(",
        "            'auto', llm, competition_id=cid, verbose=True,",
        "            max_steps=MAX_STEPS, min_rerank_score=0.15,",
        "        )",
        "",
        "    preload(COMPETITIONS)",
        "    results = {}",
        "    for cid in COMPETITIONS:",
        "        print(f'\\n========== competition {cid} ==========')",
        "        results[cid] = auto_play_loop(",
        "            make_client(), cid, build_strategy(cid), max_games=MAX_GAMES,",
        "        )",
        "",
        "    print('\\n=== summary ===')",
        "    for cid, s in results.items():",
        "        total = s['correct'] + s['wrong']",
        "        pct = 100 * s['correct'] / total if total else 0",
        '        print(f\'  cid={cid}  {s["correct"]}/{total} ({pct:.0f}%)  timeouts={s["timeouts"]}\')',
        "else:",
        "    print('RUN_LIVE is False, skipping text-mode play')",
    )
)

CELLS.append(
    md(
        "### Speech-mode play",
        "",
        "Same routing, with `mode='speech'` so that predictions are tagged",
        "and stored separately in the SQLite log for offline analysis.",
    )
)

CELLS.append(
    code(
        "if RUN_LIVE:",
        "    from polimillionaire.asr import WhisperTranscriber",
        "    from polimillionaire.play import speech_auto_play_loop",
        "",
        "    transcriber = WhisperTranscriber()",
        "    transcriber.preload()  # ~1 s HF download upfront so it is not paid mid-question",
        "",
        "    def build_speech_strategy(cid: int):",
        "        common = dict(",
        "            verbose=True, max_steps=MAX_STEPS, min_rerank_score=0.15,",
        "            mode='speech',",
        "        )",
        "        if cid == 3:",
        "            return make_strategy('rag_calc_react', llm,",
        "                                  prompt_version='math-tir', k=RAG_K, **common)",
        "        return make_strategy('auto', llm, competition_id=cid, **common)",
        "",
        "    SPEECH_COMPETITIONS = [1]   # cheap to try a single comp first",
        "    SPEECH_MAX_GAMES = 8",
        "    sp_results = {}",
        "    for cid in SPEECH_COMPETITIONS:",
        "        sp_results[cid] = speech_auto_play_loop(",
        "            make_client(), cid, build_speech_strategy(cid),",
        "            transcriber, max_games=SPEECH_MAX_GAMES,",
        "        )",
        "        s = sp_results[cid]",
        "        total = s['correct'] + s['wrong']",
        "        pct = 100 * s['correct'] / total if total else 0",
        '        print(f\'speech cid={cid}  {s["correct"]}/{total} ({pct:.0f}%)  timeouts={s["timeouts"]}\')',
        "else:",
        "    print('RUN_LIVE is False, skipping speech-mode play')",
    )
)

CELLS.append(
    md(
        "### Persist the question log",
        "",
        "Push the freshly-extended SQLite log back to the Kaggle Dataset as a",
        "new version. The next kernel that runs `pull_db` will pick it up.",
    )
)

CELLS.append(
    code(
        "if RUN_LIVE and ON_KAGGLE:",
        "    from polimillionaire.kaggle_db import push_db",
        "    stats = push_db('/kaggle/working/questions.sqlite',",
        "                     version_notes='final.ipynb live play')",
        "    print('pushed.', stats)",
        "else:",
        "    print('RUN_LIVE is False or not on Kaggle, skipping push')",
    )
)

# ---------------------------------------------------------------------------
# Conclusions + Appendix
# ---------------------------------------------------------------------------
CELLS.append(
    md(
        "## Conclusions",
        "",
        "1. **One contract, many strategies.** Every strategy, from zero-shot to a",
        "   ReAct agent, returns the same `AnswerDecision`. This uniform shape made the",
        "   SQLite log the single canonical artefact and turned offline replay",
        '   from "rebuild the harness for every new idea" into "replace one',
        '   line". Most of the iteration speed in the final weeks came from',
        "   replaying strategies over the log rather than waiting for live games.",
        "2. **Retrieval helps where Wikipedia is the source material** (History,",
        "   approximately one point over a competitive zero-shot baseline) and harms accuracy where it",
        "   is not (Entertainment, where noisy distractors retrieve the wrong articles).",
        "   The cross-encoder and rerank-score floor guard against the worst",
        "   off-topic retrievals, but cannot make irrelevant context useful.",
        "3. **Tools beat scale on mathematics.** `phi4-14b + calc_react` (91.5%) beats",
        "   the same model's few-shot baseline by 9.5 points and `qwen3-14b`",
        "   zero-shot by three points, despite identical parameter counts.",
        "   The residual errors are conceptual (statistics, abstract algebra):",
        "   the calculator cannot help where there is nothing to compute.",
        "4. **Context length is a sharp cliff, not a soft trade-off.** Few-shot",
        "   accuracy is flat from K=1 to K=50, then collapses to near-random at",
        "   K=150, when the prompt exceeds the 8 192-token KV cache. The shipped configuration is K=3.",
        "5. **The 30 s timer was never the binding constraint.** Even the",
        "   slowest combination, `phi4-14b + dynamic_few_shot_k3` at p90 ≈ 21 s, clears",
        "   the deadline by nine seconds. The latency budget was spent on",
        "   retrieval and reasoning rather than optimised for its own sake.",
        "6. **Live-only coverage is principled, not forgotten.** Philosophy",
        "   and News (cid 4 and 5) were never given a static FAISS+BM25 index, because",
        "   the offline-index assumption fails for them: niche-topic coverage",
        "   would have required hand-curated seeds, and news questions reference",
        "   events occurring after the crawl date. The same `wiki_rag` strategy is reused,",
        "   wired to live retrievers (MediaWiki and Tavily News); the",
        "   cross-encoder and the rerank floor handle off-topic hits in the same manner.",
    )
)

CELLS.append(
    md(
        "## Appendix: where to look in the repo",
        "",
        "| Topic | File |",
        "|---|---|",
        "| Strategy contract | `src/polimillionaire/strategies/base.py` |",
        "| Zero-shot | `src/polimillionaire/strategies/zero_shot.py` |",
        "| Hybrid RAG | `src/polimillionaire/strategies/wiki_rag.py` |",
        "| RRF fusion | `src/polimillionaire/retrieval/fusion.py` |",
        "| Calculator agent | `src/polimillionaire/strategies/calc_react.py` + `tools/calculator.py` |",
        "| Router | `src/polimillionaire/strategies/routed.py` + `strategies/factory.py` |",
        "| Whisper | `src/polimillionaire/asr/whisper.py` |",
        "| Schema + replay | `src/polimillionaire/recording.py` + `eval/replay.py` |",
        "| Kaggle round-trip | `src/polimillionaire/kaggle_db.py` |",
        "| Runtime notebook (Kaggle) | `notebooks/polimillionaire.ipynb` |",
        "| Result exports (per-comp) | `data/results/{math,science,comp0,comp1,scaling}/` |",
        "",
        "**Sample sizes used in the gallery:** Entertainment 400, History 400,",
        "Science 200, Math 200, Few-shot scaling 50 per K value.",
    )
)

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
# Sequential cell IDs match what nbformat 4.5+ writers and nbstripout produce.
for idx, cell in enumerate(CELLS):
    cell["id"] = str(idx)

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/final.ipynb")
# Trailing newline keeps end-of-file-fixer happy.
out.write_text(json.dumps(nb, indent=1) + "\n")

# Hand off to ruff so the cell code matches the project's formatting rules
# (notably single-quote vs. double-quote). Skip silently if ruff is missing.
if shutil.which("ruff"):
    subprocess.run(["ruff", "format", "--quiet", str(out)], check=False)
elif shutil.which("uv"):
    subprocess.run(["uv", "run", "ruff", "format", "--quiet", str(out)], check=False)

print(f"wrote {out} ({len(CELLS)} cells, {out.stat().st_size // 1024} KB)")
