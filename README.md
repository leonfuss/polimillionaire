# PoliMillionaire

A chatbot for the **Who Wants to be a PoliMillionaire?** quiz — group assignment for
NLP at PoliMi, AY 2025/26. Due **2 June 2026, 23:00** via WeBeep.

The deliverable is `notebooks/final.ipynb` plus a 5-minute video. This repo is the
working environment we share to build it.

## Team

| Name | Email | GitHub | PoliMillionaire username |
|---|---|---|---|
| _fill in_ | _ | _ | _ |
| _fill in_ | _ | _ | _ |
| _fill in_ | _ | _ | _ |
| _fill in_ | _ | _ | _ |
| _fill in_ | _ | _ | _ |

## Setup

You need **git** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
(the fast Python package manager — it brings its own Python, you don't need a system
one). Then:

```bash
git clone <repo-url> polimillionaire
cd polimillionaire
uv sync                       # creates .venv, installs deps
uv run pre-commit install     # important — keeps the repo clean
cp .env.example .env          # then open .env and fill in your credentials
uv run pytest                 # sanity check
```

> First time with uv? It replaces `pip` + `venv`. You don't activate the venv —
> `uv run <cmd>` runs inside it automatically.

## What's in here

```
polimillionaire/
├── pyproject.toml             light, stable deps
├── requirements-colab.txt     heavy ML deps for Colab runtimes
├── .pre-commit-config.yaml    hook config
├── src/polimillionaire/
│   ├── client.py              make_client() — authenticated MillionaireClient
│   ├── config.py              Settings; loads .env or Colab Secrets
│   ├── recording.py           QuestionLog (SQLite) — our canonical artefact
│   ├── llm.py                 load_llm(name) — wraps HF transformers (TODO)
│   ├── strategies/            the swappable unit: zero-shot, CoT, RAG, …
│   ├── prompts/               versioned prompt templates
│   ├── eval/replay.py         offline replay over the SQLite log
│   └── _vendor/               lecturer's millionaire_client, do not modify
├── tests/
├── notebooks/
│   ├── final.ipynb            the deliverable — assembled in the last week
│   └── scratch/               personal exploration (gitignored, lives in Drive)
└── scripts/sync_vendor.sh     refresh the vendored package if upstream ships a fix
```

## Day-to-day workflow

Feature branches + pull requests. **Nobody commits directly to `main`.**

```bash
git switch main && git pull
git switch -c feat/<short-name>
# … edit, test …
uv run pytest
git commit -m "feat: short description"
git push -u origin feat/<short-name>
gh pr create
```

A teammate reviews, CI passes, you merge (squash is fine).

- Conventional commit subjects: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`. ≤ 72 chars.
- Run `uv run pre-commit run --all-files` before opening a big PR.

### Pre-commit hooks

The hooks strip notebook outputs, format Python with `ruff`, and catch trailing
whitespace, broken YAML/TOML, and accidentally-committed large files. **Don't bypass
them with `--no-verify`** — the same hooks re-run on every PR via GitHub Actions, and
`main` is branch-protected to require them green.

If a commit fails, read the message: most hooks auto-fix and you just `git add` the
fixes and re-commit.

## Notebooks (we keep them out of git)

`.ipynb` files merge horribly, and Colab's "Save to GitHub" bypasses our local hooks.
So:

- **Personal exploration** → `notebooks/scratch/<your-name>.ipynb`. The whole `scratch/`
  folder is gitignored; save to your own Google Drive (Colab does this by default).
- **`notebooks/final.ipynb`** is the deliverable. It starts as a placeholder with the
  team table and the coding-assistant statement. We assemble it from already-merged
  code in the final week — one person owns assembly, everyone pair-reviews before
  submission.
- **All real code lives in `src/polimillionaire/`**. The notebook is markdown narrative
  + thin `from polimillionaire import …` calls.

## Running on Colab

Top cell of any Colab notebook:

```python
!git clone https://github.com/<your-handle>/polimillionaire.git
%cd polimillionaire
!pip install -q -r requirements-colab.txt && pip install -e . --no-deps
```

The `--no-deps` is load-bearing: it stops pip from re-resolving torch/transformers and
burning 5+ minutes per fresh runtime. Colab's preinstalled torch wins.

For credentials, use **Colab Secrets** (key icon in the sidebar) and add
`POLIMILLIONAIRE_API_URL`, `POLIMILLIONAIRE_USER`, `POLIMILLIONAIRE_PASSWORD`.
`load_settings()` picks them up automatically.

## The question log

The questions are server-side and closed: every one of them you see, logged correctly,
is a permanent training/eval datapoint nobody else can recreate. We keep them in one
**SQLite** file — schema in `src/polimillionaire/recording.py`.

Live runs should write to a shared Drive path, set via `POLIMILLIONAIRE_DB_PATH` in
`.env` or Colab Secrets. SQLite WAL mode lets multiple Colab runtimes append safely.
Polars reads it directly via `pl.read_database`.

## Five accounts

Use them as **manual parallelism**: Anna runs strategy A on her account, Marco runs
strategy B on his, the leaderboards tell us which works. Don't automate logging in as
all five and flooding the server — the brief warns explicitly about rate limiting and
we don't want to be the group that triggers it the week before the deadline.

## Coding-assistant statement (assignment requirement)

The brief requires a statement at the start of the deliverable on whether and how
coding assistants were used. Placeholder lives in `notebooks/final.ipynb` — **edit it
before submission**, and make sure everyone on the team agrees on the wording.
