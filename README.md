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

Because the repo is **private**, cloning from Colab needs a GitHub token. One-time
setup per person:

1. Create a fine-grained PAT at <https://github.com/settings/personal-access-tokens/new>
   (each teammate makes their *own* — don't share tokens, they're per-person)
   - **Resource owner**: your own GitHub account (not `leonfuss`)
   - **Repository access**: only `leonfuss/polimillionaire` (it shows up in the
     list because you're a collaborator)
   - **Repository permissions** → **Contents**: Read-only
   - **Expiration**: 90 days is plenty
2. In any Colab notebook, open the **Secrets** tab (key icon in the sidebar) and add:
   - `GH_TOKEN` = the PAT you just created
   - `POLIMILLIONAIRE_API_URL` = `http://131.175.15.22:51111/`
   - `POLIMILLIONAIRE_USER` = your PoliMillionaire username
   - `POLIMILLIONAIRE_PASSWORD` = your PoliMillionaire password

Then the top cell of any Colab notebook is:

```python
import os
import sys

from google.colab import userdata

gh_token = userdata.get('GH_TOKEN')
repo_dir = "/content/polimillionaire"

if os.path.isdir(repo_dir):
    os.chdir(repo_dir)
    !git checkout -q main && git pull --ff-only origin main
else:
    !git clone https://{gh_token}@github.com/leonfuss/polimillionaire.git $repo_dir
    os.chdir(repo_dir)

!pip install -q -r requirements-colab.txt && pip install -q -e . --no-deps

if "/content/polimillionaire/src" not in sys.path:
    sys.path.insert(0, "/content/polimillionaire/src")
```

Re-running this cell on the same runtime is safe: it pulls latest `main` instead
of failing on the second clone.

Two things in here are load-bearing:

- The `--no-deps` on the editable install stops pip from re-resolving
  torch/transformers and burning 5+ minutes per fresh runtime. Colab's
  preinstalled torch wins.
- The `sys.path` line works around a hatchling/Colab quirk: editable installs
  register a PEP 660 import hook via a `.pth` file, but `.pth` files are only
  processed at Python startup — by the time the bootstrap cell runs, the kernel
  is already up. Pointing `sys.path` at `src/` directly is the reliable
  workaround (and a kernel restart would do the same thing).

`load_settings()` picks the PoliMillionaire credentials up from Colab Secrets
automatically — you don't need to handle them explicitly.

## The shared question log

The questions are server-side and closed: every one of them you see, logged correctly,
is a permanent training/eval datapoint nobody else can recreate. We keep them in one
**SQLite** file — schema in `src/polimillionaire/recording.py`.

The DB lives in **Leon's Google Drive** at `MyDrive/PoliMillionaire/questions.sqlite`
and is shared with the other four teammates. To make the path work in Colab, each
teammate has to add a shortcut:

1. Open the shared folder link Leon sends you.
2. Right-click the `PoliMillionaire` folder → **Organize → Add shortcut to Drive**.
3. Place the shortcut in **My Drive**.

After that, the path `/content/drive/MyDrive/PoliMillionaire/questions.sqlite`
resolves correctly when Drive is mounted in Colab. The bootstrap notebook does the
mount + path setup for you.

SQLite WAL mode handles 5-runtime concurrent appends. Polars reads it directly via
`pl.read_database` — see `src/polimillionaire/eval/replay.py` for the pattern.

### Corpus bootstrap (one teammate per competition)

Until baselines are wired up, the DB grows by us playing the game manually:

| Teammate | Competition |
|---|---|
| Leon | 2 — Science and Nature |
| _2_ | 1 — Ancient History and Politics |
| _3_ | 0 — Entertainment |
| _4_ | 3 — Maths |
| _5_ | roving / fills gaps |

In a Colab notebook, after the bootstrap and Drive-mount cells:

```python
from polimillionaire import make_client
from polimillionaire.play import manual_play_loop

client = make_client()
manual_play_loop(client, competition_id=0, max_games=3)  # set competition_id to yours
```

Type the option id at each prompt; the helper logs every question to the shared DB.
Aim for ≥ 50 logged questions on your competition before we start running offline
strategy iteration.

## Five accounts

Use them as **manual parallelism**: Anna runs strategy A on her account, Marco runs
strategy B on his, the leaderboards tell us which works. Don't automate logging in as
all five and flooding the server — the brief warns explicitly about rate limiting and
we don't want to be the group that triggers it the week before the deadline.

## Coding-assistant statement (assignment requirement)

The brief requires a statement at the start of the deliverable on whether and how
coding assistants were used. Placeholder lives in `notebooks/final.ipynb` — **edit it
before submission**, and make sure everyone on the team agrees on the wording.
