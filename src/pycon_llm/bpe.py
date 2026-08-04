"""A real, tiny byte-pair-encoding trainer.

This is not a simulation -- it is the actual BPE algorithm, small enough to read
in one sitting:

    1. split the corpus into words, each word a sequence of characters
    2. count every adjacent pair of symbols
    3. merge the most frequent pair into a single new symbol
    4. go to 2

Everything is pure Python and deterministic, so dragging a slider on stage
always produces the same merges.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

# Marks the start of a word, sentencepiece style. Without it "in" at the start
# of a word and "in" inside a word look identical, which hides the fact that
# real tokenizers treat the leading space as part of the token.
WORD_START = "▁"

_WORD_RE = re.compile(r"\S+")

MAX_MERGES = 150


@dataclass(frozen=True)
class Merge:
    """One step of the algorithm: ``pair`` occurred ``count`` times, so it became ``token``."""

    rank: int
    pair: tuple[str, str]
    count: int
    token: str


@dataclass(frozen=True)
class BPEModel:
    """The learned merges, in the order they were learned."""

    merges: tuple[Merge, ...]
    base_alphabet: tuple[str, ...]

    @property
    def vocab(self) -> tuple[str, ...]:
        """Base characters plus every merged token."""
        return self.base_alphabet + tuple(m.token for m in self.merges)

    def take(self, n: int) -> "BPEModel":
        """The same model as it looked after only ``n`` merges."""
        return BPEModel(merges=self.merges[:n], base_alphabet=self.base_alphabet)

    def encode_word(self, word: str) -> list[str]:
        """Apply the merges, in rank order, to a single word."""
        symbols = _split_word(word)
        for merge in self.merges:
            symbols = _apply_merge(symbols, merge.pair)
        return symbols

    def encode(self, text: str) -> list[str]:
        """Tokenize arbitrary text with the learned merges."""
        return [tok for word in _WORD_RE.findall(text) for tok in self.encode_word(word)]


def _split_word(word: str) -> list[str]:
    return [WORD_START] + list(word)


def _apply_merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every non-overlapping occurrence of ``pair`` with its concatenation."""
    if len(symbols) < 2:
        return symbols
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def _count_pairs(words: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    """Count adjacent symbol pairs, weighted by how often each word appears."""
    pairs: Counter[tuple[str, str]] = Counter()
    for symbols, freq in words.items():
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq
    return pairs


def train(corpus: str, num_merges: int = MAX_MERGES) -> BPEModel:
    """Learn up to ``num_merges`` merges from ``corpus``.

    Stops early if no pair occurs more than once -- there is nothing left worth
    merging at that point.
    """
    words: dict[tuple[str, ...], int] = {}
    for word, freq in Counter(_WORD_RE.findall(corpus)).items():
        words[tuple(_split_word(word))] = freq

    base_alphabet = tuple(sorted({sym for symbols in words for sym in symbols}))

    merges: list[Merge] = []
    for rank in range(num_merges):
        pairs = _count_pairs(words)
        if not pairs:
            break
        # max() over (count, pair) keeps ties deterministic instead of relying
        # on dict ordering.
        (best_pair, best_count) = max(pairs.items(), key=lambda item: (item[1], item[0]))
        if best_count < 2:
            break
        merges.append(
            Merge(
                rank=rank,
                pair=best_pair,
                count=best_count,
                token=best_pair[0] + best_pair[1],
            )
        )
        words = {
            tuple(_apply_merge(list(symbols), best_pair)): freq
            for symbols, freq in words.items()
        }

    return BPEModel(merges=tuple(merges), base_alphabet=base_alphabet)


@lru_cache(maxsize=32)
def train_history(corpus: str, max_merges: int = MAX_MERGES) -> BPEModel:
    """Cached ``train``.

    The merge list is a prefix-stable history: the model after N merges is
    exactly ``train_history(corpus).take(N)``. That is what lets the "number of
    merges" slider be instant -- we train once and slice.
    """
    return train(corpus, max_merges)


DEFAULT_CORPUS = """
the low lower lowest newer newest wider widest
the cat sat on the mat the cat ate the rat
lower newer wider slower faster stronger
tokenizer tokenize tokenizing tokenized token tokens
learning learner learned learns unlearn relearn
the quick brown fox jumps over the lazy dog
python pythonic pythonista pythons
""".strip()
