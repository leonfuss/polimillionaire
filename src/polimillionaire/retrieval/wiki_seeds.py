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


# Math Wikipedia categories used to AUGMENT the math problem corpus
# (Hendrycks MATH dataset) with encyclopedic coverage of topics the dataset
# barely touches: abstract algebra (groups, rings, fields, Galois theory),
# real statistics (variance, hypothesis testing), linear algebra, topology.
# These live alongside the math problems in data/index/math, retrieved by
# the same Retriever -- the prompt formatter branches on metadata `source`
# to render wiki chunks differently from problem-solution pairs.
#
# Not a CompetitionSeed because it isn't a separate competition route;
# it's a sub-corpus of the math route (competition 3).
MATH_WIKI_CATEGORIES: list[str] = [
    "Abstract algebra",
    "Group theory",
    "Ring theory",
    "Field theory",
    "Galois theory",
    "Linear algebra",
    "Statistics",
    "Probability theory",
    "Statistical hypothesis testing",
    "Combinatorics",
    "Number theory",
    "Topology",
]
