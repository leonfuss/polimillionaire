"""Sanity checks for wiki_seeds.SEEDS."""

from __future__ import annotations

from polimillionaire.retrieval.wiki_seeds import SEEDS


def test_all_three_seeds_exist() -> None:
    assert set(SEEDS.keys()) == {0, 1, 2}


def test_seed_names_match_convention() -> None:
    assert SEEDS[0].name == "wiki_entertainment"
    assert SEEDS[1].name == "wiki_history"
    assert SEEDS[2].name == "wiki_science"


def test_each_seed_has_at_least_five_categories() -> None:
    for cid, seed in SEEDS.items():
        assert len(seed.categories) >= 5, f"seed {cid} has fewer than 5 categories"


def test_no_duplicate_categories_within_seed() -> None:
    for cid, seed in SEEDS.items():
        cats = seed.categories
        assert len(cats) == len(set(cats)), f"seed {cid} has duplicate categories"


def test_competition_ids_match_dict_keys() -> None:
    for key, seed in SEEDS.items():
        assert seed.competition_id == key
