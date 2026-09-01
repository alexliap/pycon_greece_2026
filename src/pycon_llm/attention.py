"""Simulated attention patterns.

There is no trained model here. Instead, one hand-written rule reproduces a
head behaviour people have actually found inside real transformers: a word
attends back to whatever earlier word it relates to, whether that relation is
an explicit reference (a pronoun back to its noun), literal repetition, or
nearby meaning in the toy embedding space. Everything else stays quiet, so
whatever lights up is a real relationship, not noise.

The matrix is causal (you can only look backwards, never forwards) and every
row sums to 1, exactly as in the real thing. That constraint is the point of
the slide: it is *why* a language model can only ever predict the next token.
"""

from __future__ import annotations

import numpy as np

# Sentences chosen because the interesting link is visible to the naked eye.
# ``links`` maps a word index to the earlier index it should refer to.
SENTENCES: dict[str, dict[str, object]] = {
    #                0   1      2   3   4   5  6   7        8       9  10  11  12
    "The trophy did not fit in the suitcase because it was too big": {
        "links": {9: 1},  # "it" -> "trophy"
        "note": "Swap **big** for **small** and *it* should point at the suitcase instead. The classic Winograd pair.",
    },
    "The trophy did not fit in the suitcase because it was too small": {
        "links": {9: 7},  # "it" -> "suitcase"
        "note": "Same sentence, one word changed, and the link flips to the suitcase. No grammar rule gets you there, only meaning.",
    },
    #        0     1    2    3       4   5        6        7    8    9
    "Maria gave her ticket to Giorgos because she had two": {
        "links": {2: 0, 7: 0},  # "her" and "she" -> "Maria"
        "note": "Both *her* and *she* reach back to Maria, several words earlier.",
    },
    #      0   1   2   3  4   5   6   7   8
    "The cat sat on the mat and it purred": {
        "links": {7: 1},  # "it" -> "cat"
        "note": "A gentle one: *it* is the cat, not the mat.",
    },
    #        0     1     2    3     4   5   6     7    8   9
    "Maria drank her coffee and ate her bread in Athens": {
        "links": {2: 0, 6: 0},  # both "her" -> "Maria"
        "note": "Two relationships light up at once: *her* reaches back to *Maria* by reference, and *bread* reaches back to *coffee* by meaning alone.",
    },
}

DEFAULT_SENTENCE = "The trophy did not fit in the suitcase because it was too big"


def tokenize(sentence: str) -> list[str]:
    """Word-level split. Real models use subwords; words keep the heatmap readable."""
    return sentence.split()


def _causal_normalise(scores: np.ndarray) -> np.ndarray:
    """Mask out the future, then make every row a probability distribution."""
    n = scores.shape[0]
    causal = np.tril(np.ones((n, n), dtype=bool))
    # Smooth first, mask second, so masked positions end up exactly 0.0 rather
    # than a small epsilon -- "cannot see the future" should mean *cannot*.
    scores = np.where(causal, np.clip(scores, 0.0, None) + 1e-6, 0.0)
    return scores / scores.sum(axis=1, keepdims=True)


def _relation_scores(tokens: list[str], links: dict[int, int]) -> np.ndarray:
    """Score how directly each word relates to an earlier one.

    Three sources of relation, all real head behaviours found in trained
    models: an explicit reference link (a pronoun back to its noun), literal
    repetition, and nearby meaning in the toy embedding space. The self and
    baseline terms are kept deliberately low, so a word with no real
    relationship to anything earlier just stays quiet instead of defaulting
    to a distracting diagonal.
    """
    from . import embeddings

    n = len(tokens)
    scores = np.eye(n) * 0.12 + np.full((n, n), 0.01)
    words = [tok.lower().strip(".,!?;:") for tok in tokens]
    known = {
        i: embeddings.vector(word)
        for i, word in enumerate(words)
        if word in embeddings.VECTORS
    }
    # Distances in the toy space run up to ~9, so divide before exponentiating;
    # otherwise even closely-related words score below the baseline.
    scale = 1.5
    for i in range(n):
        for j in range(i):
            if words[i] == words[j]:
                scores[i, j] += 0.9
            elif i in known and j in known:
                distance = float(np.linalg.norm(known[i] - known[j]))
                scores[i, j] += 1.5 * float(np.exp(-distance / scale))
    for source, target in links.items():
        if source < n and target < n:
            scores[source, target] += 2.5
    return scores


def attention(sentence: str) -> np.ndarray:
    """A causal, row-normalised attention matrix for ``sentence``."""
    tokens = tokenize(sentence)
    n = len(tokens)
    if n == 0:
        return np.zeros((0, 0))

    links = SENTENCES.get(sentence, {}).get("links", {})
    return _causal_normalise(_relation_scores(tokens, links))  # type: ignore[arg-type]


def strongest_links(sentence: str) -> list[tuple[str, str, float]]:
    """``(from_word, to_word, weight)`` for each row's top attention, skipping self-loops."""
    tokens = tokenize(sentence)
    weights = attention(sentence)
    out = []
    for i in range(1, len(tokens)):
        row = weights[i].copy()
        row[i] = 0.0
        j = int(np.argmax(row))
        out.append((tokens[i], tokens[j], float(row[j])))
    return out
