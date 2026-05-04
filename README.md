# PoliMillionaire

Group assignment for Natural Language Processing at Politecnico di Milano, AY 2025/26:
build a chatbot that plays **Who Wants to be a PoliMillionaire?**. Due **2 June 2026,
23:00** via WeBeep.

The deliverable is a Colab notebook (`notebooks/final.ipynb`) and a 5-minute
screen-capture video. This repo is the shared working environment we use to build them.

## Team

| Name | Email | GitHub | PoliMillionaire user |
|---|---|---|---|
| Leon Fuß | leon.fuss@icloud.com | leonfuss | leonfuss |
| Antoine Gaborieau | _ | AntoineGaborieau | _ |
| Aleksa Pulai | _ | aleksapulai | _ |
| Luco _ | _ | LucBruc | _ |
| _ | _ | _ | _ |

## Local setup

For working on your laptop (not Colab). You need **git** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**.
uv brings its own Python, so you don't need a system Python install.

```bash
git clone git@github.com:leonfuss/polimillionaire.git
cd polimillionaire
uv sync                       # creates .venv, installs deps
uv run pre-commit install     # installs the git hooks (required, see below)
cp .env.example .env          # then open .env and fill in your credentials
uv run pytest                 # sanity check
```

`uv run <cmd>` runs `<cmd>` inside the project's `.venv`. You never activate the venv
manually.

## What's in here

Three places code lives, with a one-way arrow between them:
**scratch notebook (in your own Drive) → shared package (`src/`) → `final.ipynb`** (the
deliverable, assembled at the end). The repo holds the second and third; the first
never enters git.

```
polimillionaire/
├── pyproject.toml             light, stable deps for local dev
├── requirements-colab.txt     heavy ML deps for Colab runtimes
├── .pre-commit-config.yaml    git hook config
├── src/polimillionaire/
│   ├── client.py              make_client() — authenticated game API client
│   ├── config.py              loads credentials from .env or Colab Secrets
│   ├── recording.py           SQLite question log (the canonical artefact)
│   ├── play.py                manual_play_loop() — human-in-the-loop game runner
│   ├── llm.py                 load_llm(name) — wraps HF transformers
│   ├── strategies/            zero-shot, CoT, RAG, ensemble, agent — each its own file
│   ├── prompts/               versioned prompt templates
│   ├── eval/replay.py         offline replay of strategies over the SQLite log
│   └── _vendor/               lecturer's millionaire_client (do not modify)
├── tests/
├── notebooks/
│   ├── final.ipynb            the deliverable
│   └── scratch/               personal exploration (gitignored, in your Drive)
└── scripts/sync_vendor.sh     refresh the vendored package if upstream ships a fix
```

## Day-to-day workflow

Feature branches + pull requests. **Nobody commits directly to `main`.** Every change
to `src/`, tests, prompts, CI config, or `pyproject.toml` goes through a PR. Edits to
your own scratch notebook in Drive never need one (it's gitignored).

```bash
git switch main && git pull
git switch -c feat/<short-name>
# … edit, test …
uv run pytest
git commit -m "feat: short description"
git push -u origin feat/<short-name>
gh pr create
```

Then: one teammate reviews, CI passes, you merge (squash is fine). Both are required
by branch protection.

- Commit subjects use [conventional commit](https://www.conventionalcommits.org/) types: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`. Keep them ≤ 72 chars.
- Branch names: `feat/<thing>`, `fix/<thing>`, `docs/<thing>`.

### Pre-commit hooks

The hooks run on every `git commit` and:

- strip outputs from any committed `.ipynb`
- format and lint Python with `ruff`
- catch trailing whitespace, broken YAML/TOML, and accidental large-file commits

Most hooks auto-fix — if a commit fails, just `git add` the fixed files and re-commit.

**Don't bypass them with `--no-verify`.** The same hooks re-run on every PR via GitHub
Actions, and `main` is branch-protected to require them green.

## Notebooks

`.ipynb` files merge poorly, and Colab's "Save to GitHub" bypasses our local hooks.
So we only commit the deliverable; everything else stays in Drive.

- **Personal exploration** → a notebook in your own Drive (Colab saves there by
  default). The repo's `notebooks/scratch/` is gitignored as a placeholder; you don't
  need to put anything there.
- **`notebooks/final.ipynb`** is the deliverable. We assemble it from already-merged
  code in the final week — one person owns assembly, everyone pair-reviews before
  submission.
- **Real logic lives in `src/polimillionaire/`**. Notebooks are markdown narrative +
  thin `from polimillionaire import …` calls — never logic.

## Running on Colab

The repo is private, so cloning from Colab needs a GitHub token. One-time setup per
person:

1. **Generate a fine-grained PAT** at <https://github.com/settings/personal-access-tokens/new>:
   - Resource owner: your own GitHub account (not `leonfuss`)
   - Repository access: only `leonfuss/polimillionaire` (it shows up because you're a collaborator)
   - Repository permissions → Contents: Read-only
   - Expiration: 90 days

   Don't share tokens — they're per-person.

2. **Add Colab Secrets** (key icon in the sidebar):
   - `GH_TOKEN` — the PAT
   - `POLIMILLIONAIRE_API_URL` — `http://131.175.15.22:51111/`
   - `POLIMILLIONAIRE_USER` — your PoliMillionaire username
   - `POLIMILLIONAIRE_PASSWORD` — your PoliMillionaire password

The top cell of any Colab notebook bootstraps the package:

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

Re-running it on the same runtime is safe — the cell detects the existing clone and
pulls latest `main` instead. The `--no-deps` flag and the `sys.path` line are both
intentional; don't remove them.

`load_settings()` reads the PoliMillionaire credentials from Colab Secrets
automatically.

## The shared question log

Every question you see is logged to a shared SQLite DB — questions are server-side and
closed, so each one is a permanent training/eval datapoint nobody outside the team can
recreate. Schema lives in `src/polimillionaire/recording.py`.

The file is in **Leon's Google Drive** at `MyDrive/PoliMillionaire/questions.sqlite`.
For the Colab path to resolve, each teammate has to add a shortcut to their own Drive:

1. Open the shared folder link Leon sends you.
2. Right-click the `PoliMillionaire` folder → **Organize → Add shortcut to Drive**.
3. Place the shortcut in **My Drive**.

The bootstrap cell mounts Drive and points `POLIMILLIONAIRE_DB_PATH` at this file
automatically. SQLite WAL mode handles concurrent appends from all five Colab
runtimes.

### Corpus bootstrap

Until LLM strategies are wired up, the DB grows by us playing manually. Every logged
question becomes a labelled datapoint we can replay strategies against later — fast
offline performance testing without burning the live API or the 30-second timer.

Competition IDs:

| ID | Name |
|---|---|
| 0 | Entertainment |
| 1 | Ancient History and Politics |
| 2 | Science and Nature |
| 3 | Maths |

In a Colab notebook, after the bootstrap and Drive-mount cells:

```python
from polimillionaire import make_client
from polimillionaire.play import manual_play_loop

client = make_client()
manual_play_loop(client, competition_id=2, max_games=3)
```

Type the option id at each prompt; the helper logs every question to the shared DB.
