"""Real tokenizers, backed by tiktoken.

``tiktoken`` downloads its BPE ranks the first time it loads an encoding and
caches them on disk. We point that cache at ``assets/tiktoken_cache`` and warm
it with :func:`warm_cache` before travelling, so the deck does not touch the
network on stage.
"""

import os
from dataclasses import dataclass

import tiktoken

from .settings import TIKTOKEN_CACHE_DIR

# tiktoken reads this when it loads an encoding, so setting it here is enough.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))

ENCODINGS: dict[str, str] = {
    "gpt2": "GPT-2 (2019), 50k tokens",
    "cl100k_base": "GPT-3.5 / GPT-4 (2022), 100k tokens",
    "o200k_base": "GPT-4o (2024), 200k tokens",
}

DEFAULT_ENCODING = "o200k_base"


@dataclass(frozen=True)
class Encoder:
    """A tiktoken encoding, shaped for what the slides need to draw."""

    name: str
    label: str
    encoding: tiktoken.Encoding

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode_pieces(self, text: str) -> list[str]:
        """The token boundaries as human-readable strings, which is what we colour."""
        return [
            self.encoding.decode_single_token_bytes(token_id).decode(
                "utf-8", errors="replace"
            )
            for token_id in self.encode(text)
        ]

    def tokens(self, text: str) -> list[tuple[str, int]]:
        """``(piece, token_id)`` pairs, ready for rendering."""
        return list(zip(self.decode_pieces(text), self.encode(text)))

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab


_CACHE: dict[str, Encoder] = {}


def get_encoder(name: str = DEFAULT_ENCODING) -> Encoder:
    """Load an encoding, memoised so slider drags do not reload it."""
    if name not in _CACHE:
        _CACHE[name] = Encoder(
            name=name,
            label=ENCODINGS.get(name, name),
            encoding=tiktoken.get_encoding(name),
        )
    return _CACHE[name]


def warm_cache() -> list[str]:
    """Download every encoding into the local cache. Run this before travelling."""
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return [
        f"{name}: ok ({get_encoder(name).vocab_size} tokens)" for name in ENCODINGS
    ]


def token_stats(text: str, encoder: Encoder) -> dict[str, float]:
    """The numbers behind the 'Greek is expensive' slide.

    ``bytes`` is the UTF-8 length, which is the honest measure of "how much text
    is this": Greek costs two bytes per character where English costs one, and
    the token count then multiplies that gap again.
    """
    tokens = encoder.encode(text)
    chars = len(text)
    words = len(text.split())
    return {
        "bytes": len(text.encode("utf-8")),
        "chars": chars,
        "words": words,
        "tokens": len(tokens),
        "chars_per_token": chars / len(tokens) if tokens else 0.0,
        "tokens_per_word": len(tokens) / words if words else 0.0,
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
