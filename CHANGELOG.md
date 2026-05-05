# Changelog

What we tried, what we learned, why we changed it. Newest first.
Pairs with git history but reads like notes — the *why*, not the diff.

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
