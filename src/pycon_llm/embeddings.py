"""A hand-drawn 2D embedding space.

Real embeddings have hundreds or thousands of dimensions and are learned from
data. These coordinates were placed by hand so the picture is legible on a
projector and the analogies land every time.

The *shape* of the claim is honest -- related words sit close together, and
directions in the space carry meaning -- even though the numbers are invented.
"""

from __future__ import annotations

import numpy as np

# Laid out so that each theme occupies its own region of the plane, and the
# gender axis (roughly +x) is consistent across the royalty/people pairs.
VECTORS: dict[str, tuple[float, float]] = {
    # --- royalty & people: gender runs left->right, status runs bottom->top ---
    "king": (-1.00, 2.60),
    "queen": (0.20, 2.60),
    "prince": (-1.00, 1.90),
    "princess": (0.20, 1.90),
    "man": (-1.00, 1.20),
    "woman": (0.20, 1.20),
    "boy": (-1.00, 0.60),
    "girl": (0.20, 0.60),
    # --- animals ---
    "cat": (-3.30, -0.60),
    "dog": (-3.00, -1.10),
    "kitten": (-3.60, -0.10),
    "puppy": (-3.30, -1.60),
    "horse": (-2.40, -0.80),
    "mouse": (-3.90, -1.00),
    "lion": (-2.70, -0.20),
    # --- food ---
    "bread": (2.60, -1.50),
    "cheese": (3.10, -1.10),
    "olive": (3.40, -1.80),
    "tomato": (2.80, -2.10),
    "coffee": (3.60, -0.50),
    "wine": (3.10, -0.20),
    "souvlaki": (2.50, -2.60),
    # --- countries & cities ---
    "greece": (1.40, 2.90),
    "athens": (2.00, 2.40),
    "france": (0.90, 3.40),
    "paris": (1.50, 3.90),
    "italy": (2.30, 3.30),
    "rome": (2.90, 3.80),
    "japan": (0.40, 4.10),
    "tokyo": (1.00, 4.60),
    # --- programming ---
    "python": (-2.80, 2.20),
    "java": (-3.30, 1.70),
    "function": (-2.40, 1.30),
    "variable": (-2.90, 0.90),
    "loop": (-2.30, 0.60),
    "compiler": (-3.70, 1.20),
    "debug": (-2.00, 1.80),
    # --- weather ---
    "rain": (0.60, -2.80),
    "snow": (0.10, -3.30),
    "sun": (1.30, -2.60),
    "cloud": (0.30, -2.30),
    "storm": (1.00, -3.40),
}

THEMES: dict[str, tuple[str, ...]] = {
    "people": ("king", "queen", "prince", "princess", "man", "woman", "boy", "girl"),
    "animals": ("cat", "dog", "kitten", "puppy", "horse", "mouse", "lion"),
    "food": ("bread", "cheese", "olive", "tomato", "coffee", "wine", "souvlaki"),
    "places": ("greece", "athens", "france", "paris", "italy", "rome", "japan", "tokyo"),
    "code": ("python", "java", "function", "variable", "loop", "compiler", "debug"),
    "weather": ("rain", "snow", "sun", "cloud", "storm"),
}

WORDS: tuple[str, ...] = tuple(VECTORS)

# Analogies that work in this space, offered as ready-made demo buttons.
ANALOGIES: tuple[tuple[str, str, str], ...] = (
    ("king", "man", "woman"),
    ("queen", "woman", "man"),
    ("athens", "greece", "france"),
    ("paris", "france", "italy"),
    ("kitten", "cat", "dog"),
)


def theme_of(word: str) -> str:
    for theme, members in THEMES.items():
        if word in members:
            return theme
    return "other"


def vector(word: str) -> np.ndarray:
    try:
        return np.array(VECTORS[word], dtype=float)
    except KeyError:
        raise KeyError(f"{word!r} is not in this toy vocabulary") from None


def matrix() -> np.ndarray:
    """All vectors, in ``WORDS`` order."""
    return np.array([VECTORS[w] for w in WORDS], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)


def nearest(word: str, k: int = 5, exclude: tuple[str, ...] = ()) -> list[tuple[str, float]]:
    """The ``k`` closest words by Euclidean distance.

    Distance rather than cosine here: in a 2D picture people read "close on
    screen" as "similar", and the demo should match what they see.
    """
    origin = vector(word)
    skip = {word, *exclude}
    scored = [
        (other, float(np.linalg.norm(vector(other) - origin)))
        for other in WORDS
        if other not in skip
    ]
    scored.sort(key=lambda item: (item[1], item[0]))
    return scored[:k]


def analogy(a: str, b: str, c: str, k: int = 3) -> list[tuple[str, float]]:
    """``a - b + c``: "king is to man as ??? is to woman"."""
    target = vector(a) - vector(b) + vector(c)
    skip = {a, b, c}
    scored = [
        (word, float(np.linalg.norm(vector(word) - target)))
        for word in WORDS
        if word not in skip
    ]
    scored.sort(key=lambda item: (item[1], item[0]))
    return scored[:k]


def analogy_target(a: str, b: str, c: str) -> np.ndarray:
    """The raw point ``a - b + c``, so the demo can draw it."""
    return vector(a) - vector(b) + vector(c)
