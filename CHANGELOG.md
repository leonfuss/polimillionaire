# Changelog

What we tried, what we learned, why we changed it. Newest first.
Pairs with git history but reads like notes — the *why*, not the diff.

## 2026-05-13 — Math tool: stats helpers, expression cap, anti-patterns

Three coupled changes to fix the math competition (0/3 games on the first
14B-class live run, all timeouts).

**Stats helpers in the calc tool.** Live LLMs reach for `Mean(X)`,
`Median(X)`, `Range(X)` — sympy has none of those (well, it has `Range`,
but as an integer iterator, which is exactly what bit us). Added
`STATS_LOCALS = {mean, median, stdev, variance, range_of}` via
`sympify(..., locals=...)`. Helpers accept varargs (`mean(10, 30, 50)`)
or a single iterable. Renamed the statistical range to `range_of` so
sympy's `Range` symbol isn't shadowed.

**Expression length capped at 200 chars** in `make_action_schema`. The
first 14B run produced an 80-char `solve(abc + aec + abf + aef + dbc +
dec + dbf + def - 1001, a + b + c + d + e + f)` that's wrong on every
axis: `abc` parses as one symbol, the solve-var is a sum, and the
problem doesn't need a calc at all. Capping forces concision; longer
setups are nearly always overcomplications.

**Prompt anti-patterns + stats exemplar.** New "Common pitfalls"
section in `_V2_SYSTEM` calls out: (1) calculator is stateless — never
reference question-text names like `X` or `f(x)`; inline literal values;
(2) write `a*b*c`, not `abc`; (3) `solve(eq, x)` not `solve(eq, a+b+c)`;
(4) for clever-observation problems, skip calc entirely. Added a 5th
exemplar showing `mean(...)` usage on the same regression case that
failed live (`X = {10, 30, 45, 50, 55, 70, 90}` vs `Y = {...}`).

## 2026-05-13 — Conditional RAG: drop passages below rerank floor

Live-play observation: questions where the cross-encoder's top score was
below ~0.15 (Shawshank oak tree, Judi Dench / York Mystery Plays) had
retrieved passages so off-topic that they actively pulled the LLM toward
the wrong answer. Added `min_rerank_score=0.15` default on
`WikiRagStrategy`. After rerank, passages below the floor are dropped;
if none survive, the prompt renders with no context block and the model
uses parametric knowledge — same path as a retrieval failure.

Gated on `use_reranker=True` so callers running raw RRF (much smaller
score scale) aren't affected. Fake rerankers in tests updated to mirror
production behavior (overwrite `Passage.score` with their logit).

## 2026-05-13 — Latency budget cuts + preload helper

First live 14B-class run showed timeouts dominating: 1 entertainment
timeout where the model *had* the correct answer, 2 science timeouts at
30–34s, all 3 math games timed out before answering. Three changes:

**Rationale length capped.** Added `maxLength: 500` (chars, ≈ ~125
tokens) to the `rationale` field in both `make_schema` and the answer
branch of `make_action_schema`. Default `max_tokens` on the LLM
wrapper dropped from 512 → 256 as a belt-and-suspenders cap. Together
these bound rationale generation to under 5s on T4 Q4_K_M.

**Math `max_steps` 3 → 1.** Live runs showed the model retrying the
same broken sympy expression three times because the prompt didn't
give it enough info to revise. One attempt then forced commit; the
LLM's mathematical intuition often gets the right answer even when the
tool errors. Tests pinned to `max_steps={2,3}` where they exercise
multi-step paths.

**`polimillionaire.preload(competition_ids=...)`** eagerly downloads /
loads embedder, reranker, and FAISS index for the listed competitions
so the first question doesn't pay the cold-load tax. On the first run
we saw 30+s first-question latency from bge-base + bge-reranker-base
HF downloads + FAISS mmap warmup. Public `Embedder.preload()` /
`Reranker.preload()` wrappers around the previously-private
`_ensure_loaded()`.

## 2026-05-13 — Per-competition retrieval tuning + nprobe knob

Entertainment's 794k-passage corpus is ~4x the size of history and ~2x
science. With the default `nprobe=32`, ~40k passages are invisible to
dense retrieval vs ~10k on history — the relevant doc is more likely
to fall outside the candidate window. Added `_AUTO_WIKI_DEFAULTS` in
the strategy factory: entertainment (comp 0) auto-gets `nprobe=128,
dense_k=100, sparse_k=100, fused_k=50, top_k=8` while history /
science keep the defaults (already 83 / 91% in replay).

`Retriever.set_nprobe(n)` exposes the IVF tuning knob at runtime;
no-op for flat indexes. Recall at `nprobe=128` is ~99% vs ~95% at 32,
costing ~3x search latency (still <200ms on 800k vectors).

## 2026-05-13 — Smaller reranker + mmap'd IVF,PQ FAISS path

Resident retrieval memory on the entertainment competition was ~3.5 GB
(flat `IndexFlatIP` over 794k fp32 vectors + bge-reranker-v2-m3 fp16),
co-resident with a ~5 GB LLM on a 13 GB MacBook → OOM. On Kaggle T4
the LLM and reranker fought for cuda:0 and crashed `ggml_cuda_pool_vmm5alloc`
mid-game.

**Reranker default flipped** to `BAAI/bge-reranker-base` (278M params,
~556 MB fp16) from `BAAI/bge-reranker-v2-m3` (568M, ~1.1 GB fp16).
Quality difference is small on English; the floor introduced today
(min_rerank_score=0.15) compensates further by gating low-quality
retrievals out entirely.

**FAISS index mmap'd via `IO_FLAG_MMAP`** when present. New script
`scripts/compress_indexes.py` walks `data/index/` and builds
`IVF{nlist},PQ{m}` indexes from existing `embeddings.npy` files using
`METRIC_INNER_PRODUCT` (matching the existing cosine-on-normalised
setup). Skips corpora under 40k vectors (math). `embeddings.npy` is
now optional when `faiss.index` is present, so Kaggle dataset uploads
can ship without the 4 GB of fp32 array.

Resident dense-index memory drops ~100x: entertainment 2.3 GB → 25 MB,
science 950 MB → 12 MB, history 600 MB → 8 MB. Recall at `nprobe=32`
is ~95%; reranker compensates over the top-50 fused list.

## 2026-05-05 — `generated_answer` column on the predictions table
Added a boolean `generated_answer` column. Live play always sets it
False. We then filled in `correct_option_id_if_known` for the 10
questions the model got wrong (no server validation = no ground truth)
by hand-reasoning the answers, and flagged each one `generated_answer = 1`.
DB now has 54/54 rows with a known answer (44 server-validated + 10
generated). The flag lets future replay/eval code distinguish or weight
generated truth so we don't optimise toward our own reasoning errors.
Schema migration runs idempotently in `QuestionLog.__init__` for
pre-existing DBs.

## 2026-05-05 — Calc output capped at 600 chars
A live-run question (`|a+b+c|` cubic system, G2L5) had the model invoke
the right thing — `solve([...], (a,b,c))` — but sympy returned 16
solutions including massive complex symbolic forms (tens of thousands of
chars). The result swamped the next LLM prompt, `complete_json` couldn't
produce valid JSON, and the resilience layer fell through to "default
to option 0" — losing $500 on a question whose first two solutions
(`(-4, -7/3, 1)`, `(4, 7/3, -1)`) were the answer. Cap output at
`MAX_OUTPUT_CHARS = 600`; leading real solutions stay visible.

## 2026-05-05 — calc-react v2 prompt + symbolic calc output
Added four hand-crafted few-shot exemplars (inclusion-exclusion counting,
`LCM × GCD = a × b`, repeating-decimal via `Rational(...)`, quadratic
interval via `solve(...)`). Calc now returns `"3/11 = 0.272..."`
symbolic+decimal for non-pure-numeric results so the model can match
fraction-shaped options without mental simplification.

**Why.** v1 calc-react had the model invoking calc (~60% of math
questions) but sometimes with wrong setups (LCM/GCD with the operands
swapped) which it then overrode in the rationale. Few-shot in real chat
format lets the model mirror the exemplar's pattern.

**Outcome.** v1's math-setup failures (cannonball quadratic,
`0.1\overline{7}`) are gone. The triangle-area question came back as
`36*sqrt(3) = 62.35` and the model picked 62 with confidence 1.0 — the
symbolic+decimal output doing exactly what it's for. New ceiling: pure
math-knowledge gaps (trace of `A²` can be negative; `4x−2` is irreducible
over `Q` because degree-1 over a field). Calc can't help — needs
retrieval. RAG is the next move.

## 2026-05-05 — DB path anchored to project root
Was: `data/questions.sqlite` relative to cwd → running `live_game.py`
from `scripts/` wrote to `scripts/data/...`, splitting the corpus
across launch directories. Now anchors to `<repo>/data/...` via
`__file__`. Existing 32 rows migrated.

## 2026-05-05 — calc-react resilience
`complete_json` raising `ValueError` in the loop (max_tokens overflow on
long sympy expressions — the original trigger was the model emitting
`I**1 + I**2 + … + I**259` term-by-term) now falls through to the
forced-answer schema instead of bubbling and killing the game. If even
the forced answer fails to parse, default to option 0 at confidence 0
so the loop continues. Added a `verbose=True` flag that prints every
`calc("…") → …` line so we can audit calc usage during a live run.

## 2026-05-05 — Project moved out of iCloud (env, not code)
`~/Documents/02_university/.../polimillionaire` → `~/dev/polimillionaire`.
macOS iCloud was periodically setting the `UF_HIDDEN` flag on
`.venv/.../*.pth` files; Python 3.13's `site.py` silently skips hidden
`.pth`, so the editable install vanished from `sys.path` and
`import polimillionaire` failed with `ModuleNotFoundError`. Also created
`* 2.py` ghost duplicates and was syncing `.git/`. Moving the whole
project out of `~/Documents/` ends the issue permanently.

## 2026-05-05 — Circular import: prompts → strategies
`prompts/calc_react.py` imported `render_question_block` from
`strategies/_common.py`. Tests always entered through `strategies.*`
first, so the cycle never fired in pytest. The local `playground.py`
was the first entry point that imported `prompts.*` first → triggered
mid-init `AttributeError` on `PROMPT_VERSION`. Moved the helper to
`prompts/_common.py` so the dep direction is `strategies → prompts`,
never the reverse. Subprocess-based regression tests pin the import
order.

## 2026-05-04 — calc-react strategy (v1)
ReAct loop: model emits `{action: "calculate", expression: ...}` or
`{action: "answer", ...}` via a GBNF `oneOf` schema with a `const`
discriminator. Sympy-backed calculator (`sympify().evalf()`), `max_steps=3`
cap with forced-answer fallback.

**Why this over native tool calling.** llama-cpp-python supports OpenAI-
style `tools=[...]` via per-model chat handlers, but that couples us to
per-model chat-template formats and complicates model swaps. The
schema-action approach reuses our existing `complete_json` interface
and works with any model that produces sampled text.

**Findings.** First live run: 9/11 correct. Calc invoked on ~60% of math
questions; about half had wrong setups the model then overrode in the
rationale. Triggered the v2 prompt work above.

## 2026-05-04 — Zero-shot baseline
Single `complete_json` per question; per-question JSON schema with
rationale-first field ordering and `answer_id` enum constrained to the
question's actual option ids. Field order forces in-band CoT (rationale
emitted before the answer commits — same prefill, no second call).

**Findings.** ~80% on a sample game. Clean failure mode: multi-step
arithmetic on big numbers (e.g. simplifying `100,000/2,118,760` →
pattern-matched to `1000/52969` instead of correct `2500/52969`).
Triggered calc-react.

## 2026-05-04 — LLM stack: llama-cpp-python + GGUF + GBNF
Need to run on free Colab T4 (~12 GB VRAM) and on Mac laptops. GGUF
Q4_K_M fits 8–14B-parameter models in budget; the same `llama-cpp-python`
package handles Metal on Mac and CUDA on Colab. GBNF
(`LlamaGrammar.from_json_schema`) compiles a JSON schema down to a
sampling grammar, so structurally valid JSON comes out every call — no
parse-recovery layer at the strategy level.

**Why not HF transformers.** ~3× memory at the same quality band; the
14B-class doesn't fit on T4 with KV cache without aggressive offloading.

## 2026-05-04 — Model registry
Default: Qwen3-8B Q4_K_M (~5 GB). Top BFCL v3 score in the 8B class
(70.8); broad world knowledge; permissive Apache-2.0 licence. The
`/no_think` switch is appended to the last user turn (per the model
card — *not* the system prompt) to disable Qwen3's thinking mode, which
interferes with structured-output reliability.

Also registered, swappable via `load_llm("name")`: Qwen3-14B,
Gemma3-12B, Granite-4.1-8B, Hermes-3-Llama-8B, Phi-4-14B.

**Llama 4 deliberately excluded.** Its licence forbids EU academic and
commercial use, and we're at PoliMi.
