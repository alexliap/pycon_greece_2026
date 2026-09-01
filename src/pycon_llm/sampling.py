"""Next-token prediction and the knobs that shape it.

The maths here -- softmax, temperature, top-k, top-p -- is exactly what a real
LLM does. The only thing that is fake is where the scores come from: instead of
a 175-billion-parameter network we use a bigram/trigram count model over a few
sentences. That is enough to make the sliders tell the truth about what
temperature and top-p actually do.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

END = "<end>"


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn arbitrary scores into probabilities that sum to 1.

    ``temperature`` divides the logits first: below 1 sharpens the distribution
    (more confident, more repetitive), above 1 flattens it (more surprising,
    more likely to go off the rails).
    """
    logits = np.asarray(logits, dtype=float)
    if temperature <= 0:
        # The limit of temperature -> 0 is "always take the argmax".
        probs = np.zeros_like(logits)
        probs[int(np.argmax(logits))] = 1.0
        return probs
    scaled = logits / temperature
    scaled = scaled - scaled.max()  # for numerical stability, changes nothing
    exp = np.exp(scaled)
    return exp / exp.sum()


def top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    """Keep only the ``k`` most likely tokens, then renormalise."""
    if k <= 0 or k >= probs.size:
        return probs
    keep = np.argsort(probs)[-k:]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    total = filtered.sum()
    return filtered / total if total else probs


def top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches ``p``."""
    if p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    # searchsorted finds where we first cross p; +1 keeps that token too, so the
    # kept mass is always >= p and we never return an empty set.
    cutoff = int(np.searchsorted(cumulative, p)) + 1
    keep = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep] = probs[keep]
    total = filtered.sum()
    return filtered / total if total else probs


def entropy(probs: np.ndarray) -> float:
    """Bits of uncertainty -- 0 means "completely sure"."""
    nonzero = probs[probs > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def perplexity(probs: list[float]) -> float:
    """How surprised the model was, per token. Lower is better; 1.0 is perfect."""
    if not probs:
        return float("nan")
    logs = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, None))
    return float(np.exp(-logs.mean()))


TRAINING_TEXT = """
the cat sat on the mat and the cat purred softly
the cat sat by the window and watched the birds
the dog sat on the floor and the dog barked loudly
the dog ran through the garden and chased the cat
a model predicts the next token again and again
a model learns the shape of the language it reads
the model reads the text and predicts the next word
python makes the model easy to read and easy to change
we write python and the machine reads the numbers
athens is warm and the coffee is strong
the conference starts and the room fills quickly
the speaker opens the laptop and the slides appear
"""


@dataclass
class ToyLM:
    """A trigram-with-bigram-backoff count model. Deliberately small and readable."""

    counts: dict[tuple[str, ...], Counter[str]]
    vocab: tuple[str, ...]
    unigram: Counter[str]

    @classmethod
    def train(cls, text: str = TRAINING_TEXT) -> "ToyLM":
        counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        vocab: set[str] = set()
        unigram: Counter[str] = Counter()
        for line in text.strip().splitlines():
            words = re.findall(r"[\w']+", line.lower())
            if not words:
                continue
            vocab.update(words)
            unigram.update(words)
            padded = words + [END]
            for i, word in enumerate(padded):
                if i >= 1:
                    counts[(padded[i - 1],)][word] += 1
                if i >= 2:
                    counts[(padded[i - 2], padded[i - 1])][word] += 1
        vocab.add(END)
        return cls(counts=dict(counts), vocab=tuple(sorted(vocab)), unigram=unigram)

    def context_used(self, context: list[str]) -> tuple[str, ...]:
        """The longest context this model actually has counts for."""
        words = [w.lower() for w in context]
        if len(words) >= 2 and tuple(words[-2:]) in self.counts:
            return tuple(words[-2:])
        if words and (words[-1],) in self.counts:
            return (words[-1],)
        return ()

    def next_token_logits(self, context: list[str]) -> np.ndarray:
        """Log-counts over the whole vocabulary, in ``self.vocab`` order.

        Log-counts stand in for the logits a real network would produce -- the
        important part is that they are unnormalised scores, so temperature and
        the filters below behave exactly as they do on a real model.
        """
        key = self.context_used(context)
        counts = self.counts.get(key, Counter())
        if not counts:  # unknown context: fall back to overall word frequency
            counts = Counter()
            for tallies in self.counts.values():
                counts.update(tallies)
        # Smoothing keeps every token reachable so top-p has a real tail to cut.
        # A flat epsilon would make every unseen word equally likely, which
        # draws as a uniform sliver of bars. Scaling the floor by how common a
        # word is overall (unigram backoff) keeps the tail varied instead of
        # flat, the same way a real LM's smoothing does. Kept small: the total
        # unigram share added tops out around 0.06, so it shapes the tail
        # without swamping the actual observed counts.
        total = sum(self.unigram.values()) or 1
        floor = 0.01 + 0.25 * np.array(
            [self.unigram.get(w, 0) / total for w in self.vocab]
        )
        return np.array(
            [np.log(counts.get(w, 0) + f) for w, f in zip(self.vocab, floor)]
        )

    def distribution(
        self,
        context: list[str],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> np.ndarray:
        """The full sampling pipeline: logits -> softmax -> top-k -> top-p."""
        probs = softmax(self.next_token_logits(context), temperature)
        probs = top_k_filter(probs, top_k)
        return top_p_filter(probs, top_p)

    def top_tokens(
        self,
        context: list[str],
        n: int = 12,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[tuple[str, float]]:
        """The ``n`` most likely next tokens, ready to plot."""
        probs = self.distribution(context, temperature, top_k, top_p)
        order = np.argsort(probs)[::-1][:n]
        return [(self.vocab[i], float(probs[i])) for i in order]

    def generate(
        self,
        prompt: str,
        steps: int = 14,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        """Run the loop: predict, pick one, append, repeat.

        ``seed`` is explicit so a demo does the same thing in rehearsal and on
        stage -- and so "same seed, same output" is demonstrable.
        """
        rng = np.random.default_rng(seed)
        words = re.findall(r"[\w']+", prompt.lower())
        produced: list[str] = []
        for _ in range(steps):
            probs = self.distribution(words + produced, temperature, top_k, top_p)
            choice = self.vocab[int(rng.choice(len(self.vocab), p=probs))]
            if choice == END:
                break
            produced.append(choice)
        return " ".join(produced)

    def score(self, text: str) -> list[tuple[str, float]]:
        """``(token, probability the model gave it)`` -- the input to perplexity."""
        words = re.findall(r"[\w']+", text.lower())
        out = []
        for i, word in enumerate(words):
            if word not in self.vocab:
                continue
            probs = softmax(self.next_token_logits(words[:i]))
            out.append((word, float(probs[self.vocab.index(word)])))
        return out


DEFAULT_PROMPTS: tuple[str, ...] = (
    "the cat",
    "the dog",
    "a model",
    "python makes",
    "athens is",
)
