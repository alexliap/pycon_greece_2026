"""Real tokenizers, with a fallback so the deck never depends on the network.

``tiktoken`` downloads its BPE ranks on first use. We commit those ranks under
``assets/tiktoken_cache`` and point tiktoken at them, so a conference wifi
outage is not a problem. If the cache is missing *and* the download fails, we
degrade to a local BPE trained by :mod:`pycon_llm.bpe` rather than raising in
the middle of a talk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import bpe
from .settings import TIKTOKEN_CACHE_DIR

# Must be set before tiktoken is imported anywhere.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

ENCODINGS: dict[str, str] = {
    "gpt2": "GPT-2 (2019) — 50k tokens",
    "cl100k_base": "GPT-3.5 / GPT-4 (2022) — 100k tokens",
    "o200k_base": "GPT-4o (2024) — 200k tokens",
}

DEFAULT_ENCODING = "o200k_base"


@dataclass
class Encoder:
    """Uniform interface over tiktoken and the local fallback."""

    name: str
    label: str
    is_fallback: bool = False
    _enc: object = field(default=None, repr=False)
    _bpe: bpe.BPEModel | None = field(default=None, repr=False)

    def encode(self, text: str) -> list[int]:
        if self._enc is not None:
            return self._enc.encode(text)
        return list(range(len(self._pieces(text))))

    def decode_pieces(self, text: str) -> list[str]:
        """The token boundaries as human-readable strings -- what we colour on screen."""
        if self._enc is not None:
            return [
                self._enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace")
                for tid in self._enc.encode(text)
            ]
        return self._pieces(text)

    def tokens(self, text: str) -> list[tuple[str, int]]:
        """``(piece, token_id)`` pairs, ready for rendering."""
        return list(zip(self.decode_pieces(text), self.encode(text)))

    def _pieces(self, text: str) -> list[str]:
        assert self._bpe is not None
        return [p.replace(bpe.WORD_START, " ") for p in self._bpe.encode(text)]

    @property
    def vocab_size(self) -> int:
        if self._enc is not None:
            return self._enc.n_vocab
        assert self._bpe is not None
        return len(self._bpe.vocab)


_CACHE: dict[str, Encoder] = {}

# Enough text that the fallback produces plausible subwords rather than letters.
_FALLBACK_CORPUS = (bpe.DEFAULT_CORPUS + " ") * 3


def get_encoder(name: str = DEFAULT_ENCODING) -> Encoder:
    """Return a tiktoken-backed encoder, or a local BPE one if that is impossible."""
    if name in _CACHE:
        return _CACHE[name]

    label = ENCODINGS.get(name, name)
    try:
        import tiktoken

        encoder = Encoder(name=name, label=label, _enc=tiktoken.get_encoding(name))
    except Exception:
        encoder = Encoder(
            name=name,
            label=f"{label} (offline fallback)",
            is_fallback=True,
            _bpe=bpe.train_history(_FALLBACK_CORPUS),
        )

    _CACHE[name] = encoder
    return encoder


def warm_cache() -> list[str]:
    """Download every encoding into the committed cache. Run this before travelling."""
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ENCODINGS:
        encoder = get_encoder(name)
        status = "fallback" if encoder.is_fallback else "ok"
        results.append(f"{name}: {status} ({encoder.vocab_size} tokens)")
    return results


def token_stats(text: str, encoder: Encoder) -> dict[str, float]:
    """Character and token counts -- the numbers behind the 'Greek is expensive' slide."""
    tokens = encoder.encode(text)
    chars = len(text)
    return {
        "chars": chars,
        "tokens": len(tokens),
        "chars_per_token": chars / len(tokens) if tokens else 0.0,
    }


# Prepared examples for the playground. The Greek/English pair is the point of
# the slide: identical meaning, wildly different token cost.
EXAMPLES: dict[str, str] = {
    "English vs Greek": "Good morning Athens, welcome to PyCon!\nΚαλημέρα Αθήνα, καλώς ήρθατε στο PyCon!",
    "Counting r's": "strawberry raspberry blueberry",
    "Numbers": "1234567 + 89 = 1234656\nThe year 2026, price 19.99",
    "Whitespace & code": "def hello():\n    return 'world'",
    "Emoji": "PyCon Greece 2026 🇬🇷 🐍 🎉",
}
