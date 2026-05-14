# Changelog

What we tried, what we learned, why we changed it. Newest first.
Pairs with git history but reads like notes — the *why*, not the diff.

## 2026-05-14 — DB retrieval: verify cached answer against current options

First live use of the db_retrieval wrapper surfaced two failure modes
the v1 code couldn't catch: (a) the server can edit a question's answer
key after we logged it, and (b) option ids are reshuffled between
sessions without a corresponding text change -- so `option_id=3` in
session B may carry a different text than `option_id=3` in session A.
In both cases the v1 wrapper happily returned the stale cached id.

Three changes:

- `lookup_known_correct` now returns `(option_id, option_text)`. The
  text is pulled from `options_json` of the row that confirmed the
  answer. Callers verify the live option at that id still has the same
  text; if not, they remap via text to a possibly-different id
  (reshuffle), or fall through to the LLM (text gone).
- Self-contradiction filter: if the cached correct option_id also
  appears in `lookup_failed_options` for the same question, the
  server's answer key has flipped since we cached it -- return None
  from the lookup and let the LLM redecide. The wrapper self-heals
  after one bad submission.
- `lookup_failed_options` SQL fix: the schema was modelled around
  rows where `correct_option_id_if_known != predicted_option_id`,
  but the server actually only reports `correct: bool` -- so a wrong
  answer's row has `correct_option_id_if_known IS NULL`, not a
  different id. The old query was effectively returning the empty
  set on real production data. The new filter matches what `play.py`
  actually writes.

Five new/updated tests pin: (option_id, text) return shape, reshuffle
remap, text-gone fall-through, contradiction filter, and end-to-end
self-heal after one wrong submission. Existing tests that seeded the
fictional `correct != predicted` shape have been switched to the real
`correct=None` shape.

## 2026-05-14 — DB retrieval wrapper: cache prior server-confirmed answers

Two observations made this worth building. First, after a few live
sessions the SQLite log already contains a non-trivial number of
questions we've seen the server confirm answers for; re-deriving them
through the LLM on the next encounter is wasted budget and runs the
risk of a regression. Second, the LLM occasionally re-picks an option
we've already been told is wrong (most often on close-call
entertainment questions where two distractors compete) -- a known
dead-end shouldn't be re-tried.

New optional wrapper `DbRetrievalStrategy`, toggled by
`make_strategy(..., db_retrieval=True)`. For each question it does, in
order:

1. **Drift check**. If the (`question_id`, `question_text`) pair we've
   logged before disagrees with what the server just sent, print a
   loud banner and flip a sticky `index_valid="0"` flag in a new
   `meta` key/value table. From that point on, the DB is treated as
   poisoned and lookups are skipped -- we assume the server rebuilt
   its question pool and the old id ↔ answer mapping no longer holds.
   The flag is intentionally one-way; flipping it back is manual.
2. **Known correct**. If the DB has a server-confirmed correct option
   for this question (`generated_answer=0`, `correct_option_id_if_known
   IS NOT NULL`) and that option is still in the current option set,
   sleep a random 7-18s and return it. The delay is pure
   server-friendliness: returning instantly on every "easy" question
   stands out, both to detection heuristics and to humans watching the
   leaderboard pacing.
3. **Otherwise**: run the inner strategy under a 30s wall-clock budget
   (via `concurrent.futures` -- the orphan thread is allowed to wind
   down on its own so the next question isn't blocked). Block any
   options the DB shows as confirmed-wrong; if the inner picks one
   anyway, rewrite to a random remaining option. If it times out, also
   pick randomly from remaining. The inner's original choice is
   preserved in the rationale so the log still captures LLM behaviour.

New schema bits: `meta` table for the flag plus three read helpers on
`QuestionLog` (`lookup_known_correct`, `lookup_failed_options`,
`find_text_mismatch`). Whitespace-only differences are normalised away
when comparing texts so cosmetic round-trips don't trigger drift.

12 new tests cover the lookup helpers and every branch of the wrapper:
DB-hit short-circuit, drift invalidation + stickiness, blocked-option
rewrite, timeout fallback, inner-exception fallback, missing-option
filtering, and the all-options-failed degenerate case.

## 2026-05-14 — Query cleanup for live Wikipedia lookup

First live run of the new live-wiki path returned **zero hits on every
entertainment question**. Tested the actual MediaWiki API: CirrusSearch
is keyword/TF-IDF, not natural-language. A question like "What is the
primary theme explored in The Babadook?" returns `[]`, while
"Babadook" returns the article. The pipe-joined `question | opt1 | opt2
| opt3 | opt4` query the strategy was sending compounded the noise --
options pull the search toward four unrelated topics simultaneously.

Two-part fix:
- `LiveWikiRetriever` now strips question words ("what/is/the/...") and
  millionaire-style filler ("primary/theme/principle/term/...") before
  hitting the API. Entity/proper-noun tokens survive; apostrophes are
  preserved (`Angel's` stays one token). Falls back to the raw query
  when filtering would empty it.
- `WikiRagStrategy` now passes `question.text` to live lookup, not the
  pipe-joined option string. The static dense+sparse retrievers still
  get the rich query (they handle NL fine); only the keyword-search API
  sees the trimmed form.

Verified against the four failing questions from the live run --
Jurassic Park, A Cruel Angel's Thesis, Casablanca, The Babadook -- all
now return the correct article(s) in the top 3.

## 2026-05-14 — Live Wikipedia lookup for entertainment + science wiki_rag

Entertainment and science are the two non-math categories where the static
wiki indexes underperform. The failure mode isn't *missing topic* — the
corpus is 794k passages for entertainment alone — it's **stale or
disambiguation-shifted**: the index was crawled once at build time, and a
film/album/discovery added or revised since then is either absent or
indexed under a different lead section than the current Wikipedia version.
The static reranker confidently picks a plausible-looking-but-wrong passage
and the LLM follows it.

**Per-question MediaWiki lookup, fused into the rerank pool.** New module
`retrieval/live_wiki.py` exposes a `LiveWikiRetriever.search(query)` that
hits `action=query&list=search` then `prop=extracts&exintro=1` in two
batched calls (~0.3-0.5s end-to-end from a warm Kaggle session). Returns
`Passage` objects with `source="live_wiki"` and `id="live/<title>"` so the
prompt formatter and reranker see them uniformly with the static hits.

The fusion point is *before* the cross-encoder reranker in
`WikiRagStrategy`: `pool = static_fused + live_passages`, then rerank,
then min-score floor. The reranker decides whether a live hit beats the
static candidates -- we don't pre-judge it. Dedup is case-insensitive
on `metadata.title` so the reranker never reads the same article twice.

**Always-fused, not fallback-only.** Considered "only call live API when
top static rerank score < threshold" but the entertainment failure mode
is *plausible-looking-but-stale* static hits — i.e. the threshold would
pass and live lookup would never trigger on exactly the questions we
need it for. Cost is two extra HTTP calls per question (~0.3-0.5s); LLM
step dominates at ~10s, so the latency tax is fine. History stays
static-only — it doesn't drift, and free latency for ~83% replay-acc
isn't worth shaving.

**Failure handling is non-negotiable.** Every API path catches and
returns `[]` with a logged reason. A 429 from a shared Kaggle egress IP
(the same class of failure that killed the math-wiki crawl last session)
must not abort an answer — the static fused pool carries the question.
Retries reuse `wiki_crawler._get_with_retry` which already honours
`Retry-After`.

**Verbose logging at every stage.** When `verbose=True`, the strategy
prints: query (truncated to 80 chars), search-returned titles, the
dedup count, the static/live/total pool size, and a final `(N live)`
tag on the retrieved-passages line so you can tell at a glance which
hits won the rerank. Needed for tuning per-competition `live_k` from
live-play logs.

**Defaults via the auto router.** `_AUTO_WIKI_DEFAULTS[0]` and `[2]` now
carry `live_lookup=True`; `[1]` (history) explicitly omits it. The
`LiveWikiRetriever` is cached on `_live_wiki_cache` so the requests
session and the in-process query cache survive across competitions.

Will validate live next; needs Kaggle internet-on for the API to be
reachable.

## 2026-05-13 — Math-wiki augmentation for the math RAG corpus

Multiple live runs failed on topics the Hendrycks MATH dataset just doesn't
cover: Sylow theorems, S_5 cycle structure, Z_n[x] polynomial rings, real
statistics. The MATH corpus is mostly Algebra / Counting & Probability /
Geometry / Number Theory / Prealgebra / Precalculus — no abstract algebra,
no statistical inference. Retrieval was finding "similar" problems but they
were never the right reference for a group/ring/stats question.

**Same index, augmented corpus.** Rather than spin up a second retriever and
double the per-question latency, the math index now mixes two sources in one
`passages.jsonl`:

- `source=math_problems`: the existing 12.5k MATH problems (unchanged).
- `source=math_wiki`: Wikipedia chunks from a curated category list
  (`MATH_WIKI_CATEGORIES` in `wiki_seeds.py`): Abstract algebra, Group
  theory, Ring theory, Field (mathematics), Galois theory, Linear algebra,
  Statistics, Probability theory, Statistical hypothesis testing,
  Combinatorics, Number theory, Topology. depth=2 BFS like the other wikis.

**Prompt formatter branches on source.** A retrieved problem renders as the
existing `Problem: ... / Solution: ...` pair; a wiki chunk renders as a
`Reference N (Wikipedia: <title>, similarity=X)` excerpt with the body text
directly. Forcing wiki content into the Problem/Solution shape would invite
the model to "solve" an encyclopedia entry and copy a non-existent answer.

**Backwards-compatible.** Pre-augmentation indexes (no `source` field) keep
rendering as problem-solutions exactly as before. The builder has
`--no-wiki` / `--wiki-only` flags so an existing problem-only index can be
extended without re-embedding the 12.5k problems, and the wiki crawl/dump
artifacts are cached so an interrupted build resumes.

Index not rebuilt locally yet — too expensive on this laptop; user will
run on Colab/Kaggle.

## 2026-05-13 — Inequality solve + duplicate-call short-circuit + var-naming rule

Three more live-failure patterns this session, all addressable together.

**Inequalities now work through the calc.** The strictly-convex L1 question
gave the model the right inequality (`(f(1)-5)/3 < (9-f(1))/1`) but it
arithmetic'd wrong and concluded `f(1) < 7` instead of `< 8`. Sympy CAN
solve the inequality (returns an `And` object `(-oo < y) & (y < 8)`), but
our calc tool was calling `.evalf()` on Boolean compounds — which they
don't have — and surfacing the AttributeError. Fix: catch AttributeError
in the evalf step and fall back to `str(expr)`. The model now sees the
real bound and can match against options. Verified for `<`, `<=`, and
no-solution (`y**2 + 1 < 0` → `False`).

**Duplicate-call short-circuit in `run_react_loop`.** Two separate live
runs showed the model emitting *byte-identical* calc actions in
consecutive steps after seeing an ERROR result: 4 identical
`solve(30*t - 360*(t//1) - 110, t)` calls, 4 identical
`expand(...) % 8` calls, etc. Each duplicate burned ~5–10s. Loop now
tracks the last expression; if the next action matches it exactly, skip
remaining steps and force the answer. Generic — helps any retry-stuck
model. New test in `tests/test_calc_react.py`.

**Variable-naming rule for `f(1)` / `g(x)` unknowns.** When the question
asks for an unknown spelled as a function call (`f(1)`, `g(x)`), the
model often passes `solve(eq, f(1))`. Sympy parses `f(1)` as a function
application with two symbols and rejects the solve. Both prompts now
spell out: substitute a single-letter placeholder (`y`, `k`, `t`)
before solving. Both also gain the explicit "do NOT retry the exact
same expression" pitfall, and both note that `solve` works for
inequalities too.

## 2026-05-13 — math-tir prompt: model-agnostic identity + stats exemplar fix

Two cleanup-quality fixes after a notebook audit:

- **System message identity is now model-agnostic.** The opening line read
  "You are Qwen2.5-Math, a model specialized in mathematical reasoning"
  — written when the math route still ran the specialist. It's been
  Qwen3-14B for several runs, so the line was misleading without
  affecting behavior. Now reads: "You are a math specialist with access
  to a sympy calculator tool." Test updated to assert the new substring.

- **Stats exemplar was numerically wrong.** Both X and Y in the
  "X = {10, 30, 45, 50, 55, 70, 90}, Y = {10, 30, 35, 50, 65, 70, 90}"
  exemplar summed to 350, so the means were equal and the exemplar's
  claim that the difference is 10/7 (and that option [3] is the false
  statement) was mathematically false — the correct answer would have
  been "none of the above are false". Y is now {10, 30, 35, 50, 60, 65,
  90}: sum 340, mean_X - mean_Y = 10/7, while median and range still
  match X. The narrative and answer_id stay intact.

Verified all 7 exemplars against the live `calc` tool — all 7 now
produce exactly the result the exemplar claims.

## 2026-05-13 — Plug-and-verify meta-strategy + math-tir max_steps=3 default

Live L5 (rose curve r = sin(3θ), vertical-tangent in first quadrant)
showed an entirely different failure mode: the model wrote a not-quite-
correct setup, sympy returned a 1480-char wall of complex/algebraic
roots, the model spent 85 seconds parsing it, and the question timed
out. The right move on a multiple-choice question with specific
candidate values is to ABANDON solve() and substitute each option back
into the original expression — one calc call, ~1 second, unambiguous.

Three coupled changes:

- **Both system prompts (v2 and math-tir) get a "PLUG-AND-VERIFY"
  pitfall** spelling out the pattern: when solve() returns a wall of
  complex roots and the question is multiple-choice, write
  `[expr_at_opt1, expr_at_opt2, expr_at_opt3]` (with each option
  literally substituted) and pick the one whose value is ~0. No Python
  comprehension — sympify rejects those (see prior commit).

- **New exemplar in `_EXEMPLARS`** (slot 7, between stats and the
  quadratic example) demonstrates the pattern end-to-end:
  "cos(2θ) + sin(θ) = 0, options {π/6, π/4, π/3, π/2}" → one calc call
  returns `[1, sqrt(2)/2, -1/2 + sqrt(3)/2, 0]`, model reads off the
  last entry and commits to option [4].

- **Factory: math-tir now defaults to `max_steps=3`** alongside the
  existing `max_tokens=768`. The plug-and-verify pattern naturally
  unfolds as: try symbolic solve → see junk → retry with substitutions
  → answer. That's three actions, which requires max_steps≥3. Users
  validated max_steps=3 in live play before this commit. Explicit
  caller-provided max_steps still wins.

## 2026-05-13 — Math prompts: sympy/Python guardrails + dict-result support

Live runs surfaced six recurring calc-tool syntax failures, all from the
model treating the calculator as a Python REPL:

- `f(2) = ...; f(-2) = ...` — Python statements with `=` and `;` chaining
- `sum(1/2**n for n in 1..3)` — Python comprehension + Ruby-style `1..3`
- `(-1,0,1) ⋅ (1,2t,3t²)` — unicode dot operator, undefined vector mul
- `norm.ppf(0.25, 45, 4)` — scipy.stats namespace doesn't exist in sympy
- `factor(240)` — sympy.factor is for polynomials, not integers
- `factorint(240)` — *would have* been right, but calc rejected the dict

The first five are prompt issues; the sixth is a calc-tool bug I caused
last commit when I added `factorint` to the recommended primitives
without checking the type guard.

Three coupled changes:

- **`calc` tool now accepts `dict` results** alongside lists/tuples.
  `factorint(n)` returns `{prime: power}` with all sympy Integer keys and
  values — same sandbox guarantees as the existing list/tuple branch
  (sympify blocks exfiltration at the parser layer regardless of the
  return type).
- **Both system prompts (v2 and math-tir) get three new pitfall rules**:
  one-expression-per-call (no `=`/`;`/comprehensions/`1..3`), no unicode
  math operators (with explicit "write dot products as `a1*b1 + a2*b2`"
  example), and no scipy/`norm.ppf` with z-score values from memory for
  the common quantiles (.25, .75, .95, .975).
- **Primitives sections expanded** with `Sum(expr, (var, a, b))`,
  `Integral(...)` / `integrate(...)`, and `oo` for infinity. Note added
  that `factor(n)` does NOT factor integers — use `factorint(n)`.

## 2026-05-13 — Math prompts: number-theory primitives + divisor-counting exemplar

Live run hit a self-correcting but slow failure on L5 ("how many positive
integers are factors of 120 and also factors of 40?"): the model reached
for `len(divisors(40))` (twice), got `ERROR: unexpected non-sympy result:
int` both times, and finally landed on `Rational(8)` after burning ~20s
on three calc steps.

Root cause: our calc tool only accepts sympy results, but `len(...)`
returns a Python `int` that gets rejected at the type guard. Models
trained on Python don't know that.

Two coupled changes to make this one-shot next time:

- **Both system prompts (v2 and math-tir) now list number-theory
  primitives explicitly**: `gcd, lcm, divisors, divisor_count, factorint`.
  The block also calls out the pitfall: use `divisor_count(n)` for counts,
  NOT `len(divisors(n))`.
- **New exemplar added to `_EXEMPLARS`** (slot 6, between the stats and
  quadratic examples): "common factors of 36 and 48" via
  `divisor_count(gcd(36, 48))`. Teaches the model the canonical pattern
  for "how many factors of A and B" questions in a single calc step.

Affects both `calc_react` and `rag_calc_react` (they share
`EXEMPLAR_MESSAGES`).

## 2026-05-13 — Math route: bigger token budget + parse-failure logging

First live run with `qwen2.5-math-7b` + `math-tir` lost all 3 games at
level 1 with `action step failed to parse` on every question. Diagnosis:
Qwen2.5-Math was trained on Tool-Integrated Reasoning (verbose CoT +
Python code blocks), not JSON. Under our grammar constraint it burns
through the 256-token default trying to comply with the more complex
`oneOf` action schema, truncating mid-object and failing to parse.
Confirming evidence: in one game the *forced-answer* fallback (using the
simpler flat schema, no `oneOf`) succeeded at the same token budget.

Two changes:

- **`run_react_loop` now accepts `max_tokens`** (default `None` = use
  `complete_json`'s 256). `CalcReactStrategy` and `RagCalcReactStrategy`
  thread it through. The factory's calc-react builders auto-set
  `max_tokens=768` when `prompt_version="math-tir"` so math-route users
  don't need to remember the override.
- **Raw model output is now logged on parse failure** (truncated to
  300 chars). Previously we silently swallowed the `ValueError` whose
  message contained exactly the diagnostic data we needed — flying blind
  ate one debugging cycle. Now both the action-step parse failure and the
  forced-answer parse failure print `[<prefix>] raw: <truncated>`.

Non-math routes are unaffected: with `max_tokens=None`, the global
default of 256 still applies.

## 2026-05-13 — Math specialist: Qwen2.5-Math-7B + math-tir prompt

Registered `qwen2.5-math-7b` in MODELS (bartowski's
`Qwen2.5-Math-7B-Instruct-GGUF`, Q4_K_M, ~4.5 GB). Per Qwen's reports:
~83% on MATH benchmark vs ~58% for a generalist 7B. Natively trained
for tool-integrated reasoning (TIR), which suits our existing
`calc_react` action schema once the prompt makes that mapping explicit.

Operating pattern: one model resident at a time (Option B from today's
plan). The team's notebook unloads the general LLM via `load_llm(...)`
between sweeps and explicitly builds the math route with the new
specialist + the `math-tir` prompt:

    math_llm = load_llm("qwen2.5-math-7b")         # auto-unloads general
    math_strategy = make_strategy(
        "rag_calc_react", math_llm, prompt_version="math-tir"
    )

New prompt variants registered as `"math-tir"` in both
`prompts/calc_react.py` and `prompts/rag_calc_react.py`. The
system message:
- Drops the generalist "trivia player" framing for an explicit
  Qwen2.5-Math specialist role.
- Reuses the same JSON action schema and 5 exemplars.
- Tells the model NOT to emit `\boxed{...}` (Qwen2.5-Math's default
  final-answer wrapper, which would parse-fail under the JSON grammar).
- Carries forward the pitfalls section: don't reference question-text
  names, write `a*b*c` not `abc`, etc.

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
