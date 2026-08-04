# LLMs: from 0 to 0.1

Interactive Marimo slides for Alexandros Liapatis' PyCon Greece 2026 talk.

## Run the deck

```bash
uv run marimo edit main.py
```

Use Marimo's presentation/slide layout when presenting. The deck is designed to
run without API keys or a hosted model:

- `tiktoken` is used when its encoding data is already cached.
- If encoding data cannot be loaded offline, the deck automatically uses a
  deterministic local tokenizer-shaped fallback so the talk still works.
- The next-token demo is a deliberately tiny NumPy simulator, not a real LLM.

## Talk flow

The cells follow the 30-minute outline: tokenization and BPE, Transformer
intuition, next-token selection, the three training beats, and evaluation via
exact match, multiple choice, and likelihood/perplexity.

Pure demo logic lives in `src/pycon_llm/demos.py`, making the examples easy to
test or reuse in a workshop.

## Checks

```bash
uv run marimo check main.py
uv run python -m unittest discover -s tests -v
```
