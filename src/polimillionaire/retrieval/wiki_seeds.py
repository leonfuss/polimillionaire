"""Static seed category lists for the three non-math Wikipedia RAG indexes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionSeed:
    competition_id: int
    name: str  # short slug, e.g. "wiki_entertainment"
    categories: list[str]  # Wikipedia category names without the "Category:" prefix


SEEDS: dict[int, CompetitionSeed] = {
    0: CompetitionSeed(
        0,
        "wiki_entertainment",
        [
            "Films",
            "Television series",
            "Music genres",
            "Musical compositions",
            "Musicals",
            "Video games",
            "Internet memes",
            "Opera",
            "Popular music",
            "Albums",
            "Animated films",
            "Comedy films",
            "Action films",
            "Rock music",
            "Hip hop music",
        ],
    ),
    1: CompetitionSeed(
        1,
        "wiki_history",
        [
            "Ancient Greece",
            "Ancient Rome",
            "Ancient Egypt",
            "Classical antiquity",
            "Byzantine Empire",
            "Byzantine military",
            "Political philosophy",
            "Politicians",
            "Ancient Greek philosophers",
            "Ancient Roman history",
            "Egyptian pharaohs",
            "Roman Republic",
            "Roman emperors",
            "Ancient history",
            "History of the ancient Mediterranean",
        ],
    ),
    2: CompetitionSeed(
        2,
        "wiki_science",
        [
            "Physics",
            "Chemistry",
            "Biology",
            "Earth sciences",
            "Astronomy",
            "Natural sciences",
            "Mechanics",
            "Optics",
            "Thermodynamics",
            "Ecology",
            "Geology",
            "Botany",
            "Zoology",
            "Atmospheric sciences",
            "Cosmology",
        ],
    ),
}
